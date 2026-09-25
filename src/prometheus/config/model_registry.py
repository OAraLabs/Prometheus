"""Locate the shipped model registry (``model_registry.yaml``).

The registry says which local models are trained for tool calling, and
``__main__._get_adapter_tier`` reads it to choose the adapter tier: a listed
model gets "light", an unlisted one "full". It lives at the repo root
(``config/model_registry.yaml``), outside ``packages = ["src/prometheus"]``,
so for the whole life of the project a ``pip install`` or a Homebrew install
had no copy of it. The reader looked only at ``<repo>/config/``, found
nothing, returned "no native tool calling" without a word, and every local
model on every install ran at tier "full". A git checkout had the file by
accident of the checkout, which is why nothing noticed.

Same defect and same fix as the config template (:mod:`prometheus.config.
template`): the wheel force-includes the file beside this module
(``[tool.hatch.build.targets.wheel.force-include]``), and this resolver looks
there first.
"""

from __future__ import annotations

from pathlib import Path

REGISTRY_NAME = "model_registry.yaml"


class ModelRegistryNotFound(FileNotFoundError):
    """The shipped model registry is missing: a packaging defect, not a config choice."""


def get_model_registry_path() -> Path:
    """Absolute path to the shipped model registry.

    Looks, in order:

    1. beside this module, where the wheel force-includes it;
    2. ``<repo>/config/``, for a source checkout or an editable install, where
       the file lives at its documented path and isn't copied into the
       package tree.

    Raises :class:`ModelRegistryNotFound` rather than returning ``None``: the
    caller that got nothing back is exactly how a missing registry turned into
    tier "full" for every model without anyone being told.
    """
    packaged = Path(__file__).resolve().parent / REGISTRY_NAME
    if packaged.is_file():
        return packaged

    # src/prometheus/config/model_registry.py -> repo root is four parents up.
    checkout = Path(__file__).resolve().parents[3] / "config" / REGISTRY_NAME
    if checkout.is_file():
        return checkout

    raise ModelRegistryNotFound(
        f"{REGISTRY_NAME} not found beside {packaged.parent} nor at {checkout}. "
        f"The wheel force-includes it via "
        f"[tool.hatch.build.targets.wheel.force-include]; if that stanza was "
        f"removed, installed packages lose the registry and every local model "
        f"falls back to adapter tier 'full'."
    )
