"""The backend registry — every local inference box this install knows about, probed.

WHY THIS MODULE EXISTS
----------------------
Until it did, "what is serving right now" was answered by curling three
endpoints by hand: the 4090's llama-server for the model and window, its
ollama for what was pulled, the mini's ollama for what was loaded. The
daemon knew ONE backend (``model.base_url``), probed it once at boot, and
every other surface either repeated that single fact or invented its own.

That shape produced the divergence class this codebase keeps meeting: two
consumers holding separate ideas of one fact (the ``24000`` in ``/api/lcm``
while the compactor enforced 32768; ``vision: false`` in the catalog while the
provider had detected true). A registry is the fix at the root: **one owner
of "which backends exist and what each one is serving", read by everything**
— the catalog, the router's override path, the fallback chain, Anatomy, and
``/api/status``. Nothing else may probe a backend or carry a backend URL.

WHAT IT IS, AND IS NOT
----------------------
It is a fact table with a probe. It **reports** what a box serves; it never
changes it. llama-server is one model per process and Prometheus does not
launch it (the same posture ``/api/status`` states for the KV-cache type:
*recorded, not governed*), so "swap the 4090's model" is a restart of that
box's service, outside this harness. Ollama backends swap per request, which
is why a spec may carry a vetted ``models`` list.

SOURCES OF TRUTH
----------------
* The boot primary (``model:``) is registered implicitly under the reserved
  name ``local`` — it is a backend like any other, probed the same way, so the
  ``local`` catalog row's window and vision come from this table too.
* ``backends:`` in ``prometheus.yaml`` — a map of *name* → spec::

      backends:
        4090:
          provider: llama_cpp
          base_url: http://gpu-box:8080
        mini:
          provider: ollama
          base_url: http://localhost:11434
          model: qwen2.5:14b-instruct
          models: [qwen2.5:14b-instruct, qwen2.5:7b-instruct]
          context_limit: 32768        # a HINT for when the probe cannot size it

  Names are slash-command names (``/4090``), so they obey Telegram's grammar
  (``[a-z0-9_]{1,32}``) and may not collide with a cloud preset or a built-in
  command. A bad entry is REFUSED with its reason — recorded on the registry
  as ``config_errors`` and logged — and the rest of the table still loads.
  Refusing the one entry beats refusing the boot, and a silently-dropped entry
  is exactly the config-dark shape the drift guards exist to prevent.

PROBES
------
* llama.cpp: ``GET /props`` (``default_generation_settings.n_ctx``,
  ``modalities.vision``, ``model_path``) and ``GET /v1/models`` (the id).
* ollama: ``GET /api/tags`` (pulled), ``GET /api/ps`` (loaded), and
  ``POST /api/show`` for the spec's default model (``*.context_length`` in
  ``model_info``, ``capabilities``).

Results are cached per backend with a TTL. A probe that fails records the
error and keeps the last good ``model``/``n_ctx`` visible as *stale*, so a
reader can tell "down" from "never asked". **Change detection:** when a
successful probe reports a different model than the previous one, the
backend's detected window and vision are invalidated, ``changed_at`` is set,
and one line is logged — so a llama-server restarted onto a different GGUF
is known at the next read, not at the next daemon restart.

Nothing here opens a connection at import or construction. The first probe
runs when the daemon asks for it at boot (bounded by ``timeout_s``, never
blocking boot beyond that) or when a reader asks with an expired TTL.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

#: The boot primary's name in the registry. Reserved: a config entry may not
#: use it, and the ``/local`` command keeps its meaning (clear the override).
PRIMARY_NAME = "local"

#: Telegram's command grammar — a backend name IS a slash command.
NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")

#: Provider names this registry knows how to probe. Anything else is refused
#: at load: a "backend" the registry cannot ask is a URL, not a backend.
LOCAL_PROVIDERS = frozenset({"llama_cpp", "ollama"})

#: Names a backend may not take even when the grammar allows them: the cloud
#: preset commands, their aliases, and the routing commands whose meaning
#: would silently change. Cloud presets are imported lazily (the router module
#: imports providers, not the other way round) and merged in at validation.
_BUILTIN_RESERVED = frozenset({
    PRIMARY_NAME, "grok", "route", "model", "backends", "status", "help",
    "start", "reset", "clear", "workspace", "context", "anatomy", "doctor",
})

# Shipped defaults — the literals in from_config() are kept EQUAL to
# config/prometheus.yaml.default (the defaults-equality guard compares them).
DEFAULT_TTL_S = 60
DEFAULT_TIMEOUT_S = 5.0


class BackendConfigError(ValueError):
    """One backend entry that cannot be used, with the reason."""


@dataclass(frozen=True)
class BackendSpec:
    """A configured backend: where it is and what it is expected to serve."""

    name: str
    provider: str
    base_url: str
    model: str | None = None
    models: tuple[str, ...] = ()
    context_limit: int | None = None   # operator HINT; a probe result outranks it
    is_primary: bool = False

    def provider_config(self, model: str | None = None) -> dict[str, Any]:
        """The provider-config tuple the router/registry build from — the same
        shape ``ProviderRegistry.create`` and ``ModelRouter.set_override`` take,
        plus ``backend`` so downstream knows which box a turn ran on."""
        cfg: dict[str, Any] = {
            "provider": self.provider,
            "base_url": self.base_url,
            "backend": self.name,
        }
        chosen = model or self.model
        if chosen:
            cfg["model"] = chosen
        return cfg


@dataclass(frozen=True)
class DetectedWindow:
    """A context window a backend REPORTED, and for which model."""

    backend: str
    model: str
    n_ctx: int
    probed_at: float   # monotonic seconds


@dataclass
class BackendStatus:
    """What the last probe found. ``ok`` is the last probe's verdict; the model /
    window / vision fields keep the last GOOD values so a reader can show
    "down since …, was serving X" rather than blanks."""

    name: str
    provider: str
    base_url: str
    ok: bool = False
    probed: bool = False              # False = never asked (distinct from "down")
    model: str | None = None          # served model id (ollama: the spec's default model)
    model_path: str | None = None     # llama.cpp: the GGUF path; ollama: None
    n_ctx: int | None = None
    vision: bool | None = None        # None = unknown / never detected
    latency_ms: float | None = None
    probed_at: float | None = None    # monotonic
    probed_at_iso: str | None = None
    error: str | None = None
    changed_at_iso: str | None = None  # last time the served model changed under us
    loaded_models: tuple[str, ...] = ()     # ollama /api/ps
    available_models: tuple[str, ...] = () # ollama /api/tags
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, stale: bool) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "ok": self.ok,
            "probed": self.probed,
            "stale": stale,
            "model": self.model,
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "vision": self.vision,
            "latency_ms": self.latency_ms,
            "probed_at": self.probed_at_iso,
            "error": self.error,
            "changed_at": self.changed_at_iso,
            "loaded_models": list(self.loaded_models),
            "available_models": list(self.available_models),
        }


# A probe is `async (spec, timeout_s) -> BackendStatus`. Injectable so the
# registry's caching / change-detection logic is testable without a server,
# and so a test can hand in a recorded payload.
ProbeFn = Callable[[BackendSpec, float], Awaitable[BackendStatus]]


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _reserved_names() -> frozenset[str]:
    try:
        from prometheus.router.model_router import OVERRIDE_PRESETS

        presets = frozenset(OVERRIDE_PRESETS)
    except Exception:  # noqa: BLE001 — validation must not depend on the router importing
        presets = frozenset()
    return _BUILTIN_RESERVED | presets


def _validate_spec(name: object, raw: object, reserved: frozenset[str]) -> BackendSpec:
    sname = str(name)
    if not NAME_RE.match(sname):
        raise BackendConfigError(
            f"backends.{sname}: name must match {NAME_RE.pattern} "
            "(it becomes a slash command, and Telegram's grammar is lowercase "
            "letters, digits and underscore)"
        )
    if sname in reserved:
        raise BackendConfigError(
            f"backends.{sname}: name collides with a built-in command or cloud preset"
        )
    if not isinstance(raw, Mapping):
        raise BackendConfigError(f"backends.{sname}: expected a mapping, got {type(raw).__name__}")
    provider = str(raw.get("provider") or "")
    if provider not in LOCAL_PROVIDERS:
        raise BackendConfigError(
            f"backends.{sname}: provider must be one of {sorted(LOCAL_PROVIDERS)}, got {provider!r} "
            "(cloud providers are slash-command presets, not backends)"
        )
    base_url = str(raw.get("base_url") or "").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise BackendConfigError(f"backends.{sname}: base_url must be an http(s) URL, got {base_url!r}")
    model = raw.get("model")
    model = str(model).strip() if model else None
    models_raw = raw.get("models") or ()
    if isinstance(models_raw, (str, bytes)):
        raise BackendConfigError(f"backends.{sname}: models must be a list, not a string")
    models = tuple(str(m).strip() for m in models_raw if str(m).strip())
    if model and model not in models:
        models = (model, *models)
    ctx_raw = raw.get("context_limit")
    context_limit: int | None = None
    if ctx_raw is not None:
        try:
            context_limit = int(ctx_raw)
        except (TypeError, ValueError):
            raise BackendConfigError(f"backends.{sname}: context_limit must be an integer") from None
        if context_limit <= 0:
            raise BackendConfigError(f"backends.{sname}: context_limit must be positive")
    return BackendSpec(
        name=sname, provider=provider, base_url=base_url, model=model,
        models=models, context_limit=context_limit,
    )


class BackendRegistry:
    """The table. Construct with :meth:`from_config`; probe on demand."""

    def __init__(
        self,
        specs: list[BackendSpec],
        *,
        ttl_s: float = DEFAULT_TTL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        config_errors: list[str] | None = None,
        probe: ProbeFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._specs: dict[str, BackendSpec] = {s.name: s for s in specs}
        self._status: dict[str, BackendStatus] = {
            s.name: BackendStatus(name=s.name, provider=s.provider, base_url=s.base_url)
            for s in specs
        }
        self.ttl_s = float(ttl_s)
        self.timeout_s = float(timeout_s)
        self.config_errors: list[str] = list(config_errors or [])
        self._probe: ProbeFn = probe or probe_backend
        self._clock = clock
        self._locks: dict[str, asyncio.Lock] = {}

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        probe: ProbeFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> "BackendRegistry":
        """Build from ``prometheus.yaml``: the primary (``model:``) as ``local``
        plus every valid ``backends.<name>``. Invalid entries are refused one by
        one with their reason; the registry still loads."""
        specs: list[BackendSpec] = []
        errors: list[str] = []

        model_cfg = config.get("model") or {}
        primary_provider = str(model_cfg.get("provider") or "llama_cpp")
        if primary_provider in LOCAL_PROVIDERS:
            default_url = "http://localhost:8080" if primary_provider == "llama_cpp" else "http://localhost:11434"
            specs.append(BackendSpec(
                name=PRIMARY_NAME,
                provider=primary_provider,
                base_url=str(model_cfg.get("base_url") or default_url).rstrip("/"),
                model=(str(model_cfg.get("model")).strip() or None) if model_cfg.get("model") else None,
                is_primary=True,
            ))
        # A cloud primary has no local backend to probe; the registry then holds
        # only the configured backends. `local` stays reserved either way.

        raw_backends = config.get("backends") or {}
        if not isinstance(raw_backends, Mapping):
            errors.append(f"backends: expected a mapping of name → spec, got {type(raw_backends).__name__}")
            raw_backends = {}
        reserved = _reserved_names()
        for name, raw in raw_backends.items():
            try:
                specs.append(_validate_spec(name, raw, reserved))
            except BackendConfigError as exc:
                errors.append(str(exc))
                log.error("backend registry: %s — entry refused", exc)

        probe_cfg = config.get("backend_probe") or {}
        if not isinstance(probe_cfg, Mapping):
            probe_cfg = {}
        return cls(
            specs,
            ttl_s=probe_cfg.get("ttl_s", 60),
            timeout_s=probe_cfg.get("timeout_s", 5.0),
            config_errors=errors,
            probe=probe,
            clock=clock,
        )

    # ── reads (no I/O) ──────────────────────────────────────────────────

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> BackendSpec | None:
        return self._specs.get(name)

    def specs(self) -> tuple[BackendSpec, ...]:
        return tuple(self._specs.values())

    def status(self, name: str) -> BackendStatus | None:
        return self._status.get(name)

    def is_stale(self, name: str) -> bool:
        st = self._status.get(name)
        if st is None or st.probed_at is None:
            return True
        return (self._clock() - st.probed_at) > self.ttl_s

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """JSON-ready view of every backend from the CACHE — no I/O, so a
        status endpoint never blocks on a dead box. ``stale`` says whether the
        TTL has lapsed; ``probed=false`` says nothing was ever asked."""
        return {
            name: st.as_dict(stale=self.is_stale(name))
            for name, st in self._status.items()
        }

    def detected_windows(self) -> dict[str, DetectedWindow]:
        """The windows backends REPORTED — the input the context resolver takes
        (one entry per backend, keyed by name). A backend whose model changed
        since its last good probe is absent until re-probed."""
        out: dict[str, DetectedWindow] = {}
        for name, st in self._status.items():
            if st.ok and st.n_ctx and st.model and st.probed_at is not None:
                out[name] = DetectedWindow(
                    backend=name, model=st.model, n_ctx=int(st.n_ctx), probed_at=st.probed_at,
                )
        return out

    # ── probes ──────────────────────────────────────────────────────────

    def _lock(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = self._locks[name] = asyncio.Lock()
        return lock

    async def probe(self, name: str, *, force: bool = False) -> BackendStatus:
        """Probe one backend, honouring the TTL unless ``force``. Concurrent
        callers for the same backend share one probe. Unknown name → KeyError."""
        spec = self._specs[name]
        async with self._lock(name):
            if not force and not self.is_stale(name):
                return self._status[name]
            previous = self._status[name]
            started = self._clock()
            try:
                fresh = await asyncio.wait_for(
                    self._probe(spec, self.timeout_s), timeout=self.timeout_s + 1.0,
                )
            except asyncio.TimeoutError:
                fresh = BackendStatus(
                    name=spec.name, provider=spec.provider, base_url=spec.base_url,
                    ok=False, error=f"probe timed out after {self.timeout_s:g}s",
                )
            except Exception as exc:  # noqa: BLE001 — a probe failure is a recorded state
                fresh = BackendStatus(
                    name=spec.name, provider=spec.provider, base_url=spec.base_url,
                    ok=False, error=f"{type(exc).__name__}: {exc}",
                )
            fresh.probed = True
            fresh.probed_at = self._clock()
            fresh.probed_at_iso = _iso_now()
            if fresh.latency_ms is None and fresh.ok:
                fresh.latency_ms = round((fresh.probed_at - started) * 1000.0, 1)
            self._status[name] = self._merge(previous, fresh)
            return self._status[name]

    async def probe_all(self, *, force: bool = False) -> dict[str, BackendStatus]:
        results = await asyncio.gather(
            *(self.probe(n, force=force) for n in self._specs), return_exceptions=True,
        )
        out: dict[str, BackendStatus] = {}
        for name, res in zip(self._specs, results):
            out[name] = res if isinstance(res, BackendStatus) else self._status[name]
        return out

    def _merge(self, previous: BackendStatus, fresh: BackendStatus) -> BackendStatus:
        """Change detection + keep-last-good.

        * A successful probe that reports a different model than the previous
          successful one marks ``changed_at`` and logs — the served model
          changed underneath the daemon.
        * A failed probe keeps the previous model/window/vision (so the
          operator sees what WAS there) but flips ``ok`` and records the error.
        """
        if fresh.ok:
            prev_identity = previous.model_path or previous.model
            new_identity = fresh.model_path or fresh.model
            if previous.ok and prev_identity and new_identity and prev_identity != new_identity:
                fresh.changed_at_iso = fresh.probed_at_iso
                log.warning(
                    "backend %s changed what it serves: %s → %s (window %s → %s, vision %s → %s)",
                    fresh.name, prev_identity, new_identity,
                    previous.n_ctx, fresh.n_ctx, previous.vision, fresh.vision,
                )
            else:
                fresh.changed_at_iso = previous.changed_at_iso
            return fresh
        # failed: carry the last good facts forward, mark down
        return BackendStatus(
            name=fresh.name, provider=fresh.provider, base_url=fresh.base_url,
            ok=False, probed=True,
            model=previous.model, model_path=previous.model_path,
            n_ctx=previous.n_ctx, vision=previous.vision,
            latency_ms=None,
            probed_at=fresh.probed_at, probed_at_iso=fresh.probed_at_iso,
            error=fresh.error, changed_at_iso=previous.changed_at_iso,
            loaded_models=(), available_models=previous.available_models,
        )

    # ── rendering (shared by /backends and the boot log) ─────────────────

    def render_table(self) -> str:
        """Plain-text table for chat surfaces and the boot log."""
        lines = ["Backends"]
        for name, st in self._status.items():
            spec = self._specs[name]
            tag = " (primary)" if spec.is_primary else ""
            if not st.probed:
                state = "not probed"
            elif st.ok:
                state = "up"
            else:
                state = f"DOWN — {st.error}"
            stale = "  [stale]" if st.probed and self.is_stale(name) else ""
            model = _short_model(st.model_path or st.model) or "?"
            window = f"{st.n_ctx // 1024}k" if st.n_ctx else "window ?"
            vision = "vision" if st.vision else ("no vision" if st.vision is False else "vision ?")
            lat = f"  {st.latency_ms:.0f} ms" if st.latency_ms is not None else ""
            lines.append(
                f"  {name}{tag}: {state}{stale} — {spec.provider} @ {spec.base_url}"
            )
            if st.probed:
                lines.append(f"      {model}, {window}, {vision}{lat}")
                if st.changed_at_iso:
                    lines.append(f"      served model changed at {st.changed_at_iso}")
            if spec.provider == "ollama" and st.available_models:
                loaded = set(st.loaded_models)
                pulled = ", ".join(
                    f"{m}*" if m in loaded else m for m in st.available_models[:8]
                )
                more = f" (+{len(st.available_models) - 8})" if len(st.available_models) > 8 else ""
                lines.append(f"      pulled: {pulled}{more}" + ("  (* = loaded)" if loaded else ""))
        if self.config_errors:
            lines.append("Refused config entries:")
            lines.extend(f"  - {e}" for e in self.config_errors)
        if len(self._specs) <= 1 and not self.config_errors:
            lines.append("  (only the primary; add named boxes under `backends:` in prometheus.yaml)")
        return "\n".join(lines)


# ── the process-wide handle ─────────────────────────────────────────────────
#
# The daemon builds ONE registry at boot and every consumer reads it: the web
# app (via create_app(backend_registry=…) / app.state), the chat commands (via
# get_registry()), Anatomy and the router in later PRs. Same pattern as the
# anatomy components — a module handle set once, read many times — so a
# gateway command needs no plumbing through three adapters to reach it.
_REGISTRY: BackendRegistry | None = None


def set_registry(registry: BackendRegistry | None) -> None:
    global _REGISTRY
    _REGISTRY = registry


def get_registry() -> BackendRegistry | None:
    return _REGISTRY


def _short_model(identity: str | None) -> str | None:
    """`/home/x/models/Qwen3.8-27B-UD-Q4_K_XL.gguf` → `Qwen3.8-27B-UD-Q4_K_XL`."""
    if not identity:
        return None
    tail = identity.rsplit("/", 1)[-1]
    return tail[:-5] if tail.endswith(".gguf") else tail


# ── the default probe ───────────────────────────────────────────────────────


async def probe_backend(spec: BackendSpec, timeout_s: float) -> BackendStatus:
    """Ask the box what it serves. HTTP only — no SSH, no credentials."""
    import httpx

    st = BackendStatus(name=spec.name, provider=spec.provider, base_url=spec.base_url)
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        if spec.provider == "llama_cpp":
            await _probe_llama_cpp(client, spec, st)
        elif spec.provider == "ollama":
            await _probe_ollama(client, spec, st)
        else:  # pragma: no cover - refused at validation
            st.error = f"no probe for provider {spec.provider!r}"
            return st
    st.latency_ms = round((time.monotonic() - started) * 1000.0, 1)
    return st


async def _probe_llama_cpp(client: Any, spec: BackendSpec, st: BackendStatus) -> None:
    props_resp = await client.get(f"{spec.base_url}/props")
    props_resp.raise_for_status()
    props = props_resp.json()
    dgs = props.get("default_generation_settings") or {}
    n_ctx = dgs.get("n_ctx")
    st.n_ctx = int(n_ctx) if n_ctx else None
    modalities = props.get("modalities") or {}
    # Vision is DETECTED here or unknown — never assumed. An older server with no
    # `modalities` key leaves it None rather than False.
    st.vision = bool(modalities.get("vision")) if isinstance(modalities, Mapping) and "vision" in modalities else None
    st.model_path = props.get("model_path") or None
    st.extra["total_slots"] = props.get("total_slots")
    try:
        models_resp = await client.get(f"{spec.base_url}/v1/models")
        if models_resp.status_code == 200:
            data = (models_resp.json() or {}).get("data") or []
            if data:
                st.model = str(data[0].get("id") or "") or None
    except Exception as exc:  # noqa: BLE001 — /props already answered; the id is a nicety
        log.debug("backend %s: /v1/models failed (%s); using model_path", spec.name, exc)
    if not st.model:
        st.model = st.model_path
    st.ok = True


async def _probe_ollama(client: Any, spec: BackendSpec, st: BackendStatus) -> None:
    tags_resp = await client.get(f"{spec.base_url}/api/tags")
    tags_resp.raise_for_status()
    tags = (tags_resp.json() or {}).get("models") or []
    st.available_models = tuple(str(m.get("name") or "") for m in tags if m.get("name"))
    try:
        ps_resp = await client.get(f"{spec.base_url}/api/ps")
        if ps_resp.status_code == 200:
            loaded = (ps_resp.json() or {}).get("models") or []
            st.loaded_models = tuple(str(m.get("name") or "") for m in loaded if m.get("name"))
    except Exception as exc:  # noqa: BLE001
        log.debug("backend %s: /api/ps failed (%s)", spec.name, exc)
    # The model this backend will serve for us: the spec's default, else the
    # one currently loaded, else the first pulled. Only then can /api/show size it.
    target = spec.model or (st.loaded_models[0] if st.loaded_models else None) \
        or (st.available_models[0] if st.available_models else None)
    st.model = target
    if target:
        if target not in st.available_models and not any(
            a.split(":")[0] == target.split(":")[0] for a in st.available_models
        ):
            st.error = f"configured model {target!r} is not pulled on this ollama"
        try:
            show_resp = await client.post(f"{spec.base_url}/api/show", json={"model": target})
            if show_resp.status_code == 200:
                show = show_resp.json() or {}
                info = show.get("model_info") or {}
                for key, value in info.items():
                    if str(key).endswith(".context_length"):
                        try:
                            st.n_ctx = int(value)
                        except (TypeError, ValueError):
                            pass
                        break
                caps = show.get("capabilities")
                if isinstance(caps, list):
                    st.vision = "vision" in caps
                st.extra["capabilities"] = caps
        except Exception as exc:  # noqa: BLE001
            log.debug("backend %s: /api/show failed (%s)", spec.name, exc)
    # `ok` = the server answered. A missing model is reported in `error` but the
    # box is up — the distinction /backends needs to show.
    st.ok = True
