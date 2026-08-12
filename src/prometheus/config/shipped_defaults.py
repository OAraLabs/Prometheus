"""Values a FRESH INSTALL ships with — imported by the readers AND the writer.

WHY THIS MODULE EXISTS (FIRSTLIGHT FL-2u)
-----------------------------------------
``config/prometheus.yaml.default`` is the documented shape of a config, but
it is **not installed**: ``pyproject.toml`` packages only ``src/prometheus``
and the template lives at the repo root, so a ``pip install`` has no copy of
it. Anything that needs a shipped default AT RUNTIME therefore cannot read
the template — it must read a constant, and that constant must live where
both the reader and the writer can import it without either depending on
the other.

That is this module: zero imports, no side effects, safe from
``cli/init.py`` (the stdlib-only fast setup path) and from the runtime
readers alike. ``tests/test_setup_advertised_defaults.py`` pins every
value here equal to the template, so "the constant" and "the documented
default" cannot drift.

THE RULE THIS ENCODES
---------------------
A key's fallback must be a WORKING value, not a degenerate one. FL-2u:
``setup`` learned to write ``tools.deferred_loading.always_loaded``, but an
install that UPGRADES keeps its old config — and the reader's fallback was
``[]``, which with deferred loading active means *advertise nothing*. The
fix is not to write config into an existing install; it is to make absence
safe (CROSS-CUTTING §5 — a property that cannot be violated beats a check
that must remember to run). See ``tests/test_absence_hostile_keys.py``.
"""

from __future__ import annotations

# tools.deferred_loading.always_loaded — the tool set a fresh install
# advertises. Absent from an upgraded config, and the old fallback of []
# meant the model was handed nothing it could call.
SHIPPED_ALWAYS_LOADED: tuple[str, ...] = (
    "bash", "task_create", "read_file", "write_file", "edit_file",
    "grep", "glob", "tool_search", "web_search", "web_fetch", "memory",
)
