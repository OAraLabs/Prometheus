"""AnatomyScanner — detects infrastructure state (hardware, model, resources).

Runs at daemon startup and periodically to keep ANATOMY.md current.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)


@dataclass
class GPUProcess:
    """One compute process holding VRAM on a GPU.

    Populated from ``nvidia-smi --query-compute-apps`` on whichever box
    the GPU lives on. ``memory_mb`` is per-process VRAM, not total card
    usage — so a card with three workers shows three entries summing to
    less than total used (driver overhead lives outside the sum).
    """
    pid: int
    name: str
    memory_mb: int


@dataclass
class AnatomyState:
    """Snapshot of the current infrastructure."""

    # Hardware
    hostname: str = ""
    platform: str = ""
    cpu: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU — ``gpu_*`` always reflects the *inference* GPU (where the agent's
    # LLM actually runs). When inference is local, that's the local card.
    # When inference is remote, we probe via SSH so the report doesn't
    # lie about a local card being the inference card.
    #
    # ``gpu_is_remote`` flags which case applies — ``gpu_inference_host``
    # carries the IP/hostname of the inference box (matches the hostname
    # parsed from the inference URL). Both are None when no GPU was
    # detected at all.
    #
    # ``local_gpu_*`` is populated *only when* inference is remote AND
    # this box has its own GPU. Lets us show both honestly — e.g. for
    # this deployment, gemma runs on the 4090 (inference) while the
    # local 3090 Ti hosts Ollama/ComfyUI. Without these fields the agent
    # confidently confuses the two.
    gpu_name: str | None = None
    gpu_vram_total_mb: int | None = None
    gpu_vram_used_mb: int | None = None
    gpu_vram_free_mb: int | None = None
    gpu_is_remote: bool = False
    gpu_inference_host: str | None = None
    gpu_probe_method: str = "none"  # "local" | "ssh" | "none" — how we got the data
    gpu_probe_error: str | None = None  # populated when probe failed; tells user why

    local_gpu_name: str | None = None
    local_gpu_vram_total_mb: int | None = None
    local_gpu_vram_used_mb: int | None = None
    local_gpu_vram_free_mb: int | None = None

    # VRAM-holding processes per GPU. ``gpu_processes`` matches the
    # inference GPU (remote or local), ``local_gpu_processes`` matches
    # the secondary local card when inference is remote. Empty list
    # means probed-but-nothing-there; we don't disambiguate "no probe"
    # vs "no processes" because that's never the user's question.
    gpu_processes: list[GPUProcess] = field(default_factory=list)
    local_gpu_processes: list[GPUProcess] = field(default_factory=list)

    # Active model
    model_name: str | None = None
    model_file: str | None = None
    model_quantization: str | None = None

    # Inference engine
    inference_engine: str = "unknown"
    inference_url: str = ""
    inference_features: list[str] = field(default_factory=list)

    # Vision / audio
    vision_enabled: bool = False
    whisper_model: str | None = None

    # Connectivity
    tailscale_ip: str | None = None
    tailscale_peers: list[dict] = field(default_factory=list)  # [{name, ip, online}]

    # Storage
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    prometheus_data_size_mb: float = 0.0

    # Timestamp
    scanned_at: str = ""


class AnatomyScanner:
    """Scan and record the current infrastructure state."""

    def __init__(
        self,
        llama_cpp_url: str = "http://localhost:8080",
        ollama_url: str = "http://localhost:11434",
        inference_engine: str = "llama_cpp",
        ssh_user: str | None = None,
        ssh_key: str | None = None,
    ) -> None:
        self._llama_url = llama_cpp_url.rstrip("/")
        self._ollama_url = ollama_url.rstrip("/")
        self._engine = inference_engine
        self._ssh_user = ssh_user
        self._ssh_key = str(Path(ssh_key).expanduser()) if ssh_key else None

    async def scan(self) -> AnatomyState:
        """Full infrastructure scan."""
        state = AnatomyState(
            scanned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._detect_platform(state)
        self._detect_ram(state)
        self._detect_disk(state)

        # Parallel async detections
        gpu_task = asyncio.create_task(self._detect_gpu(state))
        model_task = asyncio.create_task(self._detect_model(state))
        ts_task = asyncio.create_task(self._detect_tailscale(state))
        await asyncio.gather(gpu_task, model_task, ts_task, return_exceptions=True)

        self._detect_whisper(state)
        state.inference_engine = self._engine
        state.inference_url = (
            self._llama_url if self._engine == "llama_cpp" else self._ollama_url
        )
        return state

    async def quick_scan(self) -> AnatomyState:
        """Lightweight scan — model + VRAM only."""
        state = AnatomyState(
            scanned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._detect_platform(state)
        await self._detect_gpu(state)
        await self._detect_model(state)
        state.inference_engine = self._engine
        state.inference_url = (
            self._llama_url if self._engine == "llama_cpp" else self._ollama_url
        )
        return state

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_platform(self, state: AnatomyState) -> None:
        state.hostname = platform.node()
        state.platform = platform.system()
        state.cpu = self._read_cpu_model()

    @staticmethod
    def _read_cpu_model() -> str:
        # Linux: /proc/cpuinfo
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        # macOS
        if platform.system() == "Darwin":
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass
        return platform.processor() or "unknown"

    def _detect_ram(self, state: AnatomyState) -> None:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            data = meminfo.read_text()
            for line in data.splitlines():
                if line.startswith("MemTotal:"):
                    state.ram_total_gb = int(line.split()[1]) / 1_048_576
                elif line.startswith("MemAvailable:"):
                    state.ram_available_gb = int(line.split()[1]) / 1_048_576
            return
        # macOS fallback
        if platform.system() == "Darwin":
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    state.ram_total_gb = int(result.stdout.strip()) / (1024**3)
            except Exception:
                pass

    def _detect_disk(self, state: AnatomyState) -> None:
        try:
            usage = shutil.disk_usage(str(Path.home()))
            state.disk_total_gb = round(usage.total / (1024**3), 1)
            state.disk_free_gb = round(usage.free / (1024**3), 1)
        except OSError:
            pass

        config_dir = get_config_dir()
        if config_dir.exists():
            total = sum(
                f.stat().st_size for f in config_dir.rglob("*") if f.is_file()
            )
            state.prometheus_data_size_mb = round(total / (1024**2), 1)

    async def _detect_gpu(self, state: AnatomyState) -> None:
        """Probe the **inference** GPU, not "whichever GPU happens to be local".

        Old behavior: tried local nvidia-smi first; only fell back to SSH if
        local was missing. On a box with both a local card *and* remote
        inference, the local card always won — and the display layer then
        slapped "(remote)" on it. The agent read the lie, ran local
        ``nvidia-smi`` to "verify", saw the same numbers, and confidently
        confirmed the wrong story.

        New behavior:
        - Decide where inference lives by parsing ``inference_url``.
        - If remote: probe the remote box via SSH first. That's the GPU the
          model actually runs on. Record the local GPU as a *secondary*
          field so we can still surface it (ComfyUI / local Ollama may use
          it) without conflating the two.
        - If local (or no remote configured): probe local. No second probe.
        - Always record ``gpu_probe_method`` and ``gpu_probe_error`` so the
          display + the agent can both tell honestly what was reachable.
        """
        from urllib.parse import urlparse

        url = self._llama_url if self._engine == "llama_cpp" else self._ollama_url
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        local_host_aliases = {
            "", "localhost", "127.0.0.1", "::1",
            platform.node().lower(),
        }
        inference_is_remote = host not in local_host_aliases
        state.gpu_inference_host = host or None

        if inference_is_remote:
            # Probe the inference GPU first — that's the source of truth.
            state.gpu_is_remote = True
            remote_ok, remote_err = await self._probe_remote_gpu(state, host)
            if not remote_ok:
                state.gpu_probe_error = remote_err
                # Don't fall back to local-as-inference — that's the bug
                # we're fixing. The probe failure is itself the truth to
                # report; the user/agent needs to know the remote box is
                # unreachable, not get a substitute reading from this box.
                log.warning(
                    "Inference GPU probe failed for host %s: %s",
                    host, remote_err,
                )
            # Also record the local GPU (if any) as a separate field, so
            # callers can show "this box has a card too" without conflating.
            await self._probe_local_gpu_as_secondary(state)
        else:
            # Inference is local — the local GPU IS the inference GPU.
            local_ok, local_err = await self._probe_local_gpu_as_primary(state)
            if not local_ok:
                state.gpu_probe_error = local_err

    async def _probe_remote_gpu(
        self, state: AnatomyState, host: str,
    ) -> tuple[bool, str | None]:
        """SSH the remote host and read nvidia-smi. Fills primary GPU fields.

        Returns ``(success, error_message_or_None)``. Error messages are
        meant for users — they get surfaced in /anatomy output and explain
        what's wrong: "SSH credentials not configured", "connection refused",
        "no nvidia-smi on remote", etc.
        """
        if not self._ssh_user:
            return False, (
                "SSH credentials not configured "
                "(set anatomy.ssh_user + anatomy.ssh_key in prometheus.yaml)"
            )

        ssh_target = f"{self._ssh_user}@{host}"
        ssh_args = [
            "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]
        if self._ssh_key:
            ssh_args.extend(["-i", self._ssh_key])
        ssh_args.extend([
            ssh_target,
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            return False, f"SSH to {host} timed out after 15s"
        except FileNotFoundError:
            return False, "ssh binary not on PATH on this box"
        except Exception as exc:
            return False, f"SSH error: {exc}"

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()[:300] or f"rc={proc.returncode}"
            return False, f"remote nvidia-smi failed: {err}"

        if self._parse_nvidia_smi_primary(state, stdout.decode()):
            state.gpu_probe_method = "ssh"
            # Also pull the process list from the same remote box. Best-
            # effort: a failure here doesn't invalidate the card-level
            # stats we already have.
            state.gpu_processes = await self._probe_remote_gpu_processes(host)
            return True, None
        return False, "remote nvidia-smi returned no data"

    async def _probe_local_gpu_as_primary(
        self, state: AnatomyState,
    ) -> tuple[bool, str | None]:
        """Run local nvidia-smi, fill the primary GPU fields."""
        ok, err = await self._run_local_nvidia_smi()
        if ok is None:
            return False, err
        if not self._parse_nvidia_smi_primary(state, ok):
            return False, "local nvidia-smi returned unparseable output"
        state.gpu_probe_method = "local"
        state.gpu_processes = await self._probe_local_gpu_processes()
        return True, None

    async def _probe_local_gpu_as_secondary(self, state: AnatomyState) -> None:
        """Run local nvidia-smi, fill the *secondary* local_gpu_* fields.

        Used when inference is remote but this box also has a card —
        we want to report both. Silent on failure (the local card is a
        nice-to-have here, not the source of truth).
        """
        ok, _ = await self._run_local_nvidia_smi()
        if ok is None:
            return
        line = ok.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                state.local_gpu_name = parts[0]
                state.local_gpu_vram_total_mb = int(float(parts[1]))
                state.local_gpu_vram_used_mb = int(float(parts[2]))
                state.local_gpu_vram_free_mb = int(float(parts[3]))
            except ValueError:
                pass
        # Process list — the "what's actually holding VRAM on this card?"
        # answer. Best-effort; failure is silent.
        state.local_gpu_processes = await self._probe_local_gpu_processes()

    async def _probe_local_gpu_processes(self) -> list[GPUProcess]:
        """Read VRAM-holding processes from local nvidia-smi."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                return []
        except Exception:
            return []
        return self._parse_compute_apps(stdout.decode())

    async def _probe_remote_gpu_processes(self, host: str) -> list[GPUProcess]:
        """Read VRAM-holding processes from the remote inference host via SSH.

        Reuses the same ssh_user/ssh_key that ``_probe_remote_gpu`` used —
        if SSH worked once it'll work again. Best-effort: silent failure.
        """
        if not self._ssh_user:
            return []
        ssh_target = f"{self._ssh_user}@{host}"
        ssh_args = [
            "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]
        if self._ssh_key:
            ssh_args.extend(["-i", self._ssh_key])
        ssh_args.extend([
            ssh_target,
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode != 0:
                return []
        except Exception:
            return []
        return self._parse_compute_apps(stdout.decode())

    @staticmethod
    def _parse_compute_apps(output: str) -> list[GPUProcess]:
        """Parse the CSV ``pid,process_name,used_memory`` lines into GPUProcess objects.

        Skips header-like or malformed lines silently — nvidia-smi
        occasionally emits a "[N/A]" placeholder for the memory field on
        compute exclusive mode; treat as 0 rather than crashing.
        """
        out: list[GPUProcess] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            try:
                mem = int(float(parts[2]))
            except ValueError:
                mem = 0
            # nvidia-smi gives the full executable path — keep the
            # basename so display lines stay short.
            name = parts[1].rsplit("/", 1)[-1] or parts[1]
            out.append(GPUProcess(pid=pid, name=name, memory_mb=mem))
        return out

    async def _run_local_nvidia_smi(self) -> tuple[str | None, str | None]:
        """Run local nvidia-smi. Returns (stdout, None) on success or (None, error)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except asyncio.TimeoutError:
            return None, "local nvidia-smi timed out"
        except FileNotFoundError:
            return None, "nvidia-smi not installed on this box"
        except Exception as exc:
            return None, f"nvidia-smi error: {exc}"
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()[:200]
            return None, f"local nvidia-smi failed: {err or f'rc={proc.returncode}'}"
        return stdout.decode(), None

    @staticmethod
    def _parse_nvidia_smi_primary(state: AnatomyState, output: str) -> bool:
        """Parse nvidia-smi CSV into the primary GPU fields. Returns True on success."""
        line = output.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                state.gpu_name = parts[0]
                state.gpu_vram_total_mb = int(float(parts[1]))
                state.gpu_vram_used_mb = int(float(parts[2]))
                state.gpu_vram_free_mb = int(float(parts[3]))
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def _parse_nvidia_smi(state: AnatomyState, output: str) -> bool:
        """Backwards-compatible alias — kept for any external callers."""
        return AnatomyScanner._parse_nvidia_smi_primary(state, output)

    async def _detect_model(self, state: AnatomyState) -> None:
        if self._engine == "llama_cpp":
            await self._detect_model_llama_cpp(state)
        else:
            await self._detect_model_ollama(state)

    async def _detect_model_llama_cpp(self, state: AnatomyState) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._llama_url}/v1/models")
                resp.raise_for_status()
                body = resp.json()
                models = body.get("data", [])
                if models:
                    model_id = models[0].get("id", "")
                    state.model_name = model_id
                    self._parse_model_id(state, model_id)

                # Check capabilities from /v1/models response
                for m in body.get("models", models):
                    caps = m.get("capabilities", [])
                    if "multimodal" in caps:
                        state.vision_enabled = True
                        break

                # Check /props for vision (llama.cpp extension)
                try:
                    props_resp = await client.get(f"{self._llama_url}/props")
                    if props_resp.status_code == 200:
                        props = props_resp.json()
                        if props.get("total_slots"):
                            state.inference_features.append("multi_slot")
                except Exception:
                    pass

                # Check /slots for vision mmproj (fallback)
                if not state.vision_enabled:
                    try:
                        slots_resp = await client.get(f"{self._llama_url}/slots")
                        if slots_resp.status_code == 200:
                            slots_data = slots_resp.json()
                            if isinstance(slots_data, list):
                                for slot in slots_data:
                                    if slot.get("has_vision"):
                                        state.vision_enabled = True
                                        break
                    except Exception:
                        pass

        except Exception:
            log.debug("llama.cpp model detection failed at %s", self._llama_url)

        # Fallback: check process cmdline for --mmproj
        if not state.vision_enabled:
            state.vision_enabled = await self._check_cmdline_vision()

        if state.model_name:
            state.inference_features.append("streaming")

    async def _detect_model_ollama(self, state: AnatomyState) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                if models:
                    state.model_name = models[0].get("name", "")
                    state.inference_features.append("streaming")
        except Exception:
            log.debug("Ollama model detection failed at %s", self._ollama_url)

    @staticmethod
    def _parse_model_id(state: AnatomyState, model_id: str) -> None:
        """Extract GGUF filename and quantization from model id string."""
        # llama.cpp model IDs are typically the GGUF filename
        state.model_file = model_id
        # Common quant patterns: Q4_K_M, Q4_K_XL, Q8_0, F16, BF16, IQ4_XS
        import re

        m = re.search(r"((?:I?Q\d+_\w+|[BF]F?\d+))", model_id, re.IGNORECASE)
        if m:
            state.model_quantization = m.group(1)

    @staticmethod
    async def _check_cmdline_vision() -> bool:
        """Check if any llama-server process was started with --mmproj."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-a", "llama-server",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return b"--mmproj" in stdout
        except Exception:
            return False

    async def _detect_tailscale(self, state: AnatomyState) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            data = json.loads(stdout.decode())

            self_ip = data.get("TailscaleIPs", [None])
            if self_ip:
                state.tailscale_ip = self_ip[0] if isinstance(self_ip, list) else str(self_ip)

            peers = data.get("Peer", {})
            for peer_info in peers.values():
                name = peer_info.get("HostName", "")
                if not name:
                    continue
                peer_ips = peer_info.get("TailscaleIPs", [])
                ip = peer_ips[0] if peer_ips else ""
                online = peer_info.get("Online", False)
                state.tailscale_peers.append({"name": name, "ip": ip, "online": online})
        except Exception:
            log.debug("Tailscale detection failed")

    def _detect_whisper(self, state: AnatomyState) -> None:
        try:
            import yaml

            cfg_path = Path(__file__).resolve().parents[3] / "config" / "prometheus.yaml"
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text())
                whisper_cfg = cfg.get("whisper", {})
                if whisper_cfg.get("enabled"):
                    state.whisper_model = whisper_cfg.get("model", "base")
        except Exception:
            pass
