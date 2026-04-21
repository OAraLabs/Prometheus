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
class AnatomyState:
    """Snapshot of the current infrastructure."""

    # Hardware
    hostname: str = ""
    platform: str = ""
    cpu: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpu_name: str | None = None
    gpu_vram_total_mb: int | None = None
    gpu_vram_used_mb: int | None = None
    gpu_vram_free_mb: int | None = None

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
        # Try local nvidia-smi first
        if await self._detect_gpu_local(state):
            return
        # Fall back to remote nvidia-smi via SSH on the inference host
        await self._detect_gpu_remote(state)

    async def _detect_gpu_local(self, state: AnatomyState) -> bool:
        """Try local nvidia-smi. Returns True if successful."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                return False
            return self._parse_nvidia_smi(state, stdout.decode())
        except Exception:
            log.debug("Local GPU detection failed (nvidia-smi not available)")
            return False

    async def _detect_gpu_remote(self, state: AnatomyState) -> bool:
        """Try nvidia-smi on the remote inference host via SSH."""
        from urllib.parse import urlparse
        url = self._llama_url if self._engine == "llama_cpp" else self._ollama_url
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host or host in ("localhost", "127.0.0.1", "::1"):
            return False

        # Build SSH command with optional user and key
        ssh_target = f"{self._ssh_user}@{host}" if self._ssh_user else host
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
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode != 0:
                log.debug("Remote GPU detection via SSH failed (rc=%d)", proc.returncode)
                return False
            return self._parse_nvidia_smi(state, stdout.decode())
        except Exception:
            log.debug("Remote GPU detection failed for host %s", host)
            return False

    @staticmethod
    def _parse_nvidia_smi(state: AnatomyState, output: str) -> bool:
        """Parse nvidia-smi CSV output into state. Returns True if successful."""
        line = output.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            state.gpu_name = parts[0]
            state.gpu_vram_total_mb = int(float(parts[1]))
            state.gpu_vram_used_mb = int(float(parts[2]))
            state.gpu_vram_free_mb = int(float(parts[3]))
            return True
        return False

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
