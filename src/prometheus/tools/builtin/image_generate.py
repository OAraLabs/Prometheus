# Provenance: Novel code for Prometheus
# License: MIT
# Modified: Multi-backend image generation — Pollinations.ai (free public)
#           and local ComfyUI (FLUX-schnell on host GPU). Backend selectable
#           via the tool's `backend` parameter or `image_generation` config.

"""Generate images via Pollinations.ai (free hosted) or local ComfyUI (sovereign).

Two backends behind one tool — the agent picks via input arg, or the
config's ``image_generation.default_backend`` decides. Same return
contract: a saved image on disk plus a short success blurb. Agents and
the Telegram gateway don't need to care which backend produced the
bytes.

Backend trade-offs:

- **pollinations** — free, no API key, no infra. Hosted by Pollinations.ai
  on shared infra; latency varies, prompts may be logged server-side.
- **comfyui** — local server (typically ~/comfyui, port 8188), FLUX-backed,
  air-gapped. Faster on a warm GPU, no quota, supports future img2img/
  inpaint/ControlNet extensions. Needs the model files on disk and
  competes with other workloads for VRAM.

When ``backend=auto``, the tool probes ComfyUI's ``/system_stats`` and
falls back to Pollinations if the local server is unreachable. The
probe is fast (a single HEAD-equivalent GET with 2s timeout) so the
auto path adds at most ~2s to the first call after a ComfyUI restart.

Image bytes always land in ``~/.prometheus/cache/images/`` via the
existing media-cache helper unless ``output_path`` overrides — this
matches how Telegram-side image *inputs* are cached, so cleanup hits
both.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, Field

from prometheus.security.path_guard import assert_path_under_roots
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend names
# ---------------------------------------------------------------------------

BackendName = Literal["auto", "pollinations", "comfyui"]

# Pollinations-side known model names (the endpoint accepts anything;
# unknown values fall through to flux server-side).
_POLLINATIONS_MODELS = ("flux", "turbo", "gptimage")

# Default timeouts in seconds for each backend.
_POLLINATIONS_TIMEOUT = 120.0
_COMFYUI_PROBE_TIMEOUT = 2.0
_COMFYUI_TOTAL_TIMEOUT = 180.0
_COMFYUI_POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class ImageGenerateInput(BaseModel):
    """Arguments for image generation."""

    prompt: str = Field(
        description=(
            "Text description of the image to generate. Be specific about "
            "subject, style, lighting, composition. Example: 'A red apple "
            "on a wooden table, soft afternoon light, photorealistic'."
        ),
        min_length=1,
        max_length=2000,
    )
    backend: BackendName = Field(
        default="auto",
        description=(
            "Which backend to use. 'auto' (default) tries local ComfyUI "
            "first and falls back to Pollinations if it's offline. "
            "'pollinations' forces the free hosted endpoint. 'comfyui' "
            "forces the local GPU backend (errors if unreachable)."
        ),
    )
    model: str = Field(
        default="flux",
        description=(
            "Model name. For Pollinations: 'flux' (default), 'turbo', "
            "'gptimage'. For ComfyUI: the .safetensors filename under "
            "models/checkpoints/ (config's default_model if omitted)."
        ),
    )
    width: int = Field(
        default=1024, ge=256, le=2048,
        description="Image width in pixels (256-2048).",
    )
    height: int = Field(
        default=1024, ge=256, le=2048,
        description="Image height in pixels (256-2048).",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Optional seed for reproducible output. Omit for random."
        ),
    )
    steps: int | None = Field(
        default=None, ge=1, le=50,
        description=(
            "Inference steps. ComfyUI only — Pollinations ignores. "
            "Defaults to 4 for FLUX-schnell (config.comfyui.default_steps)."
        ),
    )
    enhance: bool = Field(
        default=False,
        description=(
            "Pollinations-only: pass prompt through an LLM to expand. "
            "ComfyUI ignores this flag."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional absolute path to save the image. When omitted, "
            "saves to ~/.prometheus/cache/images/ with a generated name."
        ),
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class ImageGenerateTool(BaseTool):
    """Generate an image from text via Pollinations or local ComfyUI."""

    name = "image_generate"
    description = (
        "Generate an image from a text prompt. Two backends: 'pollinations' "
        "(free hosted, no key), 'comfyui' (local GPU, FLUX-backed, "
        "sovereign). Default 'auto' picks comfyui when reachable else "
        "pollinations. Returns the path to the saved image. Defaults: "
        "1024x1024, flux model. Set seed for reproducible output."
    )
    input_model = ImageGenerateInput

    def is_read_only(self, arguments: ImageGenerateInput) -> bool:
        # Only read-only when nothing is written to a model-chosen path.
        # With output_path set the tool writes a file, so it must NOT be
        # treated as read-only (which would skip permission handling and put
        # it on the parallel read-only dispatch path). Mirrors
        # YouTubeTranscriptTool's ``save_to is None`` pattern.
        return arguments.output_path is None

    async def execute(
        self,
        arguments: ImageGenerateInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        # Gate a model-supplied output_path BEFORE doing any work, so an
        # out-of-bounds path is refused without burning a generation call and
        # without ever reaching the write sink. The same guard fires again at
        # the sink (_save_image_bytes) as defense-in-depth.
        if arguments.output_path is not None:
            try:
                assert_path_under_roots(
                    arguments.output_path, _allowed_image_roots()
                )
            except ValueError as exc:
                return ToolResult(
                    output=(
                        f"image_generate: refusing to write outside the "
                        f"allowed image directories — {exc}"
                    ),
                    is_error=True,
                )

        cfg = _image_config(context)
        backend = await _resolve_backend(arguments.backend, cfg)
        logger.info(
            "image_generate: backend=%s model=%s size=%dx%d prompt_len=%d",
            backend, arguments.model, arguments.width, arguments.height,
            len(arguments.prompt),
        )

        if backend == "comfyui":
            return await _generate_via_comfyui(arguments, cfg)
        return await _generate_via_pollinations(arguments)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _image_config(context: ToolExecutionContext) -> dict[str, Any]:
    """Pull the ``image_generation`` config block from runtime context.

    Falls back to the bare prometheus.yaml file if context doesn't
    carry the loaded config (some tool-call entry points construct
    contexts without it).
    """
    meta = getattr(context, "metadata", {}) or {}
    config = meta.get("config") if isinstance(meta, dict) else None
    if isinstance(config, dict):
        block = config.get("image_generation", {})
        if isinstance(block, dict):
            return block
    # Fallback: re-read prometheus.yaml. Cheap enough — image gen is a
    # human-perceptible action, not a hot path.
    try:
        import yaml
        from prometheus.config.paths import get_config_dir
        for candidate in (
            Path(__file__).resolve().parents[4] / "config" / "prometheus.yaml",
            get_config_dir() / "prometheus.yaml",
        ):
            if candidate.is_file():
                cfg = yaml.safe_load(candidate.read_text()) or {}
                block = cfg.get("image_generation", {})
                if isinstance(block, dict):
                    return block
                break
    except Exception:
        pass
    return {}


async def _resolve_backend(
    requested: BackendName, cfg: dict[str, Any],
) -> Literal["pollinations", "comfyui"]:
    """Map ``requested`` to a concrete backend, probing on ``auto``.

    Explicit ``pollinations`` or ``comfyui`` are honored verbatim —
    the agent gets the failure it asked for. ``auto`` and a missing
    ``backend`` fall back through the config default; if that's also
    ``auto``, probe ComfyUI and pick.
    """
    if requested == "pollinations":
        return "pollinations"
    if requested == "comfyui":
        return "comfyui"
    # auto
    default = (cfg.get("default_backend") or "auto").lower()
    if default == "pollinations":
        return "pollinations"
    if default == "comfyui":
        return "comfyui"
    # Probe — auto means "use comfyui if it's up"
    if await _comfyui_reachable(cfg):
        return "comfyui"
    return "pollinations"


async def _comfyui_reachable(cfg: dict[str, Any]) -> bool:
    """Return True if ComfyUI's ``/system_stats`` endpoint responds quickly."""
    base = (cfg.get("comfyui", {}) or {}).get(
        "base_url", "http://127.0.0.1:8188",
    )
    try:
        async with httpx.AsyncClient(timeout=_COMFYUI_PROBE_TIMEOUT) as client:
            r = await client.get(f"{base.rstrip('/')}/system_stats")
            return r.status_code == 200
    except Exception as exc:
        logger.debug("ComfyUI probe failed at %s: %s", base, exc)
        return False


# ---------------------------------------------------------------------------
# Pollinations backend
# ---------------------------------------------------------------------------

async def _generate_via_pollinations(
    arguments: ImageGenerateInput,
) -> ToolResult:
    url = _pollinations_url(
        prompt=arguments.prompt,
        model=arguments.model,
        width=arguments.width,
        height=arguments.height,
        seed=arguments.seed,
        enhance=arguments.enhance,
    )
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=_POLLINATIONS_TIMEOUT,
        ) as client:
            response = await client.get(
                url, headers={"User-Agent": "Prometheus/0.1"},
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        return ToolResult(
            output=(
                f"image_generate: Pollinations timed out after "
                f"{_POLLINATIONS_TIMEOUT:.0f}s. Service may be under load — "
                f"retry, or use a smaller width/height, or switch to "
                f"backend=comfyui if local server is running."
            ),
            is_error=True,
        )
    except httpx.HTTPStatusError as exc:
        return ToolResult(
            output=(
                f"image_generate: Pollinations HTTP "
                f"{exc.response.status_code}. Body: "
                f"{exc.response.text[:200] if exc.response.text else '(empty)'}"
            ),
            is_error=True,
        )
    except httpx.HTTPError as exc:
        return ToolResult(
            output=f"image_generate: network error — {exc}",
            is_error=True,
        )

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        return ToolResult(
            output=(
                f"image_generate: unexpected Content-Type {content_type!r}. "
                f"Response prefix: "
                f"{response.text[:200] if response.text else '(binary)'}"
            ),
            is_error=True,
        )

    ext = _ext_from_content_type(content_type)
    try:
        saved_path = _save_image_bytes(
            response.content, ext=ext, override_path=arguments.output_path,
        )
    except ValueError as exc:
        return ToolResult(
            output=f"image_generate: refusing to save image — {exc}",
            is_error=True,
        )
    size_kb = len(response.content) / 1024.0
    return ToolResult(
        output=(
            f"Saved image to {saved_path}\n"
            f"  backend: pollinations\n"
            f"  size: {arguments.width}x{arguments.height}\n"
            f"  bytes: {size_kb:.1f} KB\n"
            f"  model: {arguments.model}\n"
            f"  format: {content_type}"
        ),
    )


def _pollinations_url(
    *,
    prompt: str,
    model: str,
    width: int,
    height: int,
    seed: int | None,
    enhance: bool,
) -> str:
    base = "https://image.pollinations.ai/prompt"
    encoded = quote(prompt, safe="")
    params = [
        f"width={width}",
        f"height={height}",
        f"model={quote(model, safe='')}",
        "nologo=true",
        "safe=false",
    ]
    if seed is not None:
        params.append(f"seed={seed}")
    if enhance:
        params.append("enhance=true")
    return f"{base}/{encoded}?{'&'.join(params)}"


# ---------------------------------------------------------------------------
# ComfyUI backend
# ---------------------------------------------------------------------------

# Minimal FLUX-schnell workflow (API format). Built for all-in-one
# checkpoints (Comfy-Org/flux1-schnell-fp8.safetensors and similar).
# Variables interpolated at call time: positive_prompt, ckpt_name,
# width, height, seed, steps. Node ids are stable strings — ComfyUI
# requires them but doesn't care about ordering.
_FLUX_SCHNELL_WORKFLOW: dict[str, Any] = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "prometheus", "images": ["8", 0]},
    },
}


async def _generate_via_comfyui(
    arguments: ImageGenerateInput,
    cfg: dict[str, Any],
) -> ToolResult:
    """Drive a local ComfyUI server end-to-end.

    Flow: POST workflow to /prompt → poll /history/{id} → GET /view
    for the saved image bytes → write to cache.
    """
    comfy = cfg.get("comfyui", {}) or {}
    base = (comfy.get("base_url") or "http://127.0.0.1:8188").rstrip("/")
    ckpt_name = arguments.model
    if ckpt_name in _POLLINATIONS_MODELS:
        # Caller passed a Pollinations alias — substitute the local default.
        ckpt_name = comfy.get("default_model") or "flux1-schnell-fp8.safetensors"
    steps = arguments.steps or int(comfy.get("default_steps", 4))
    seed = arguments.seed if arguments.seed is not None else (
        uuid.uuid4().int & 0xFFFFFFFF
    )

    workflow = _build_flux_workflow(
        prompt=arguments.prompt,
        ckpt_name=ckpt_name,
        width=arguments.width,
        height=arguments.height,
        seed=seed,
        steps=steps,
    )
    client_id = uuid.uuid4().hex
    payload = {"prompt": workflow, "client_id": client_id}

    try:
        async with httpx.AsyncClient(timeout=_COMFYUI_TOTAL_TIMEOUT) as client:
            r = await client.post(f"{base}/prompt", json=payload)
            r.raise_for_status()
            prompt_id = r.json().get("prompt_id")
            if not prompt_id:
                return ToolResult(
                    output=(
                        f"image_generate (comfyui): /prompt returned no "
                        f"prompt_id. Body: {r.text[:400]}"
                    ),
                    is_error=True,
                )

            history = await _poll_comfyui_history(
                client, base, prompt_id, _COMFYUI_TOTAL_TIMEOUT,
            )
            if history is None:
                return ToolResult(
                    output=(
                        f"image_generate (comfyui): timed out waiting for "
                        f"prompt {prompt_id} to complete after "
                        f"{_COMFYUI_TOTAL_TIMEOUT:.0f}s. Check ComfyUI logs."
                    ),
                    is_error=True,
                )

            image_meta = _extract_image_from_history(history)
            if image_meta is None:
                return ToolResult(
                    output=(
                        f"image_generate (comfyui): no image output in "
                        f"history for prompt {prompt_id}. The workflow "
                        f"may have hit a validation error — check "
                        f"history payload."
                    ),
                    is_error=True,
                )

            image_bytes = await _fetch_comfyui_image(client, base, image_meta)
    except httpx.HTTPStatusError as exc:
        return ToolResult(
            output=(
                f"image_generate (comfyui): HTTP {exc.response.status_code} — "
                f"{exc.response.text[:400] if exc.response.text else '(empty)'}"
            ),
            is_error=True,
        )
    except httpx.HTTPError as exc:
        return ToolResult(
            output=(
                f"image_generate (comfyui): network error at {base} — {exc}. "
                f"Is the local ComfyUI server running? "
                f"`systemctl --user status comfyui` or check ~/comfyui/main.py."
            ),
            is_error=True,
        )

    if not image_bytes:
        return ToolResult(
            output="image_generate (comfyui): /view returned empty bytes.",
            is_error=True,
        )
    try:
        saved_path = _save_image_bytes(
            image_bytes, ext=".png", override_path=arguments.output_path,
        )
    except ValueError as exc:
        return ToolResult(
            output=f"image_generate: refusing to save image — {exc}",
            is_error=True,
        )
    size_kb = len(image_bytes) / 1024.0
    return ToolResult(
        output=(
            f"Saved image to {saved_path}\n"
            f"  backend: comfyui ({base})\n"
            f"  size: {arguments.width}x{arguments.height}\n"
            f"  bytes: {size_kb:.1f} KB\n"
            f"  checkpoint: {ckpt_name}\n"
            f"  steps: {steps}\n"
            f"  seed: {seed}"
        ),
    )


def _build_flux_workflow(
    *,
    prompt: str,
    ckpt_name: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    """Deep-clone the template and substitute the per-call values."""
    wf = json.loads(json.dumps(_FLUX_SCHNELL_WORKFLOW))
    wf["3"]["inputs"]["seed"] = int(seed)
    wf["3"]["inputs"]["steps"] = int(steps)
    wf["4"]["inputs"]["ckpt_name"] = ckpt_name
    wf["5"]["inputs"]["width"] = int(width)
    wf["5"]["inputs"]["height"] = int(height)
    wf["6"]["inputs"]["text"] = prompt
    # negative stays empty for schnell (cfg=1.0 ignores it anyway)
    return wf


async def _poll_comfyui_history(
    client: httpx.AsyncClient,
    base: str,
    prompt_id: str,
    timeout: float,
) -> dict[str, Any] | None:
    """Poll /history/{id} until the prompt entry shows up. Returns the entry or None on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            r = await client.get(f"{base}/history/{prompt_id}")
            r.raise_for_status()
            data = r.json()
            entry = data.get(prompt_id)
            if entry and entry.get("outputs"):
                return entry
        except httpx.HTTPError as exc:
            logger.debug("history poll error: %s", exc)
        await asyncio.sleep(_COMFYUI_POLL_INTERVAL)
    return None


def _extract_image_from_history(history: dict[str, Any]) -> dict[str, str] | None:
    """Find the first image output in a /history entry.

    Returns a dict ``{filename, subfolder, type}`` matching ComfyUI's
    /view query schema, or None if no image is present.
    """
    outputs = history.get("outputs") or {}
    for node_output in outputs.values():
        images = node_output.get("images") or []
        for img in images:
            filename = img.get("filename")
            if not filename:
                continue
            return {
                "filename": str(filename),
                "subfolder": str(img.get("subfolder") or ""),
                "type": str(img.get("type") or "output"),
            }
    return None


async def _fetch_comfyui_image(
    client: httpx.AsyncClient, base: str, meta: dict[str, str],
) -> bytes:
    """Download the rendered image bytes via /view."""
    params = {
        "filename": meta["filename"],
        "subfolder": meta["subfolder"],
        "type": meta["type"],
    }
    r = await client.get(f"{base}/view", params=params)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ext_from_content_type(content_type: str) -> str:
    ct = content_type.split(";", 1)[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(ct, ".jpg")


def _allowed_image_roots() -> list[Path]:
    """Directories ``image_generate`` may write to when ``output_path`` is set.

    Deliberately tighter than SecurityGate's ``security.workspace_root`` — on
    this deployment that key is ``"~"`` (the whole home dir), i.e. the exact
    attack surface this guard exists to close (``~/.bashrc``, ``~/.ssh/...``).
    Writing image bytes to an arbitrary path is never a legitimate need, so
    outputs are confined to the Prometheus base dir ``~/.prometheus/`` (which
    contains ``cache/images/``, ``files/``, ``data/`` and the default
    ``workspace/``). The agent workspace is added explicitly in case
    ``PROMETHEUS_WORKSPACE_DIR`` relocates it outside the base.
    """
    from prometheus.config.paths import get_config_dir, get_workspace_dir

    roots: list[Path] = [get_config_dir()]
    try:
        roots.append(get_workspace_dir())
    except Exception:  # pragma: no cover - workspace dir is best-effort
        pass
    return roots


def _save_image_bytes(
    data: bytes, *, ext: str, override_path: str | None,
) -> str:
    if override_path:
        # Resolve-then-check (path_guard does both) so a ``../`` traversal
        # that escapes the allow-list is rejected even if the literal string
        # starts under an allowed root. Raises ValueError on violation — the
        # backend callers convert that into a ToolResult(is_error=True).
        path = assert_path_under_roots(override_path, _allowed_image_roots())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)
    from prometheus.gateway.media_cache import cache_image_from_bytes
    return cache_image_from_bytes(data, ext=ext)
