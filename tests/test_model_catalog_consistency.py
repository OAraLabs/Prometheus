"""The nine sites that re-type model facts must agree with each other.

WHY THIS FILE EXISTS. Model names, their prices and their selectability live in
nine places across this repo — OVERRIDE_PRESETS, PRESET_MODEL_CHOICES,
CLOUD_DEFAULTS, the PRICING table, the interactive wizard's menu, the
non-interactive setup table, the shipped prometheus.yaml, and the two media-tool
constants. Nothing held them together, so they drifted, and the drift was
INTERNAL — visible without any reference to what a provider currently ships:

  - cli/init.py shipped ``qwen3.7-max`` while every other site said
    ``qwen3.8-max``, so non-interactive cloud setup silently chose the older
    AND dearer model ($2.50/$7.50 vs $2.00/$6.00 per Mtok).
  - The wizard and cli/init.py both wrote ``claude-sonnet-4-6``, a model
    PRESET_MODEL_CHOICES could not select — setup produced a config the Models
    tab refused to show.
  - PRESET_MODEL_CHOICES offered ``claude-sonnet-5`` and ``claude-opus-5``
    while PRICING had neither, so choosing either from the Models tab billed
    $0.00. That is the same shape as the ``qwen3.8-max`` incident, in which 57%
    of this box's lifetime tokens priced at zero for months.

Consolidating the nine sites into one is a bigger change than this file.
Asserting that they agree is cheap, and it turns the next drift into a failing
test instead of a silent, months-long wrong number.

WHAT THIS FILE DELIBERATELY DOES NOT DO: assert that any model name is CURRENT.
Upstream moving is not drift — it is the world changing, and a test cannot see
it. These assertions are all internal-consistency ones, which is exactly the
class a test CAN hold. Staleness is downstream of inconsistency anyway: nine
sites are nine chances to refresh eight of them.
"""

from __future__ import annotations

import pytest

from prometheus.cli.init import _CLOUD_FAST_PROVIDERS
from prometheus.providers.registry import CLOUD_DEFAULTS
from prometheus.router.model_router import (
    OVERRIDE_PRESETS,
    resolve_model_choices,
)
from prometheus.setup_wizard import CLOUD_DEFAULT_ENV_VARS, CLOUD_PROVIDER_MODELS
from prometheus.telemetry.cost import price_for

PRESET_KEYS = tuple(OVERRIDE_PRESETS)

# preset key ↔ provider name. Built from the presets rather than re-typed, so a
# new preset joins every check below by existing.
PRESET_BY_PROVIDER: dict[str, str] = {
    spec["provider"]: key for key, spec in OVERRIDE_PRESETS.items()
}


def _default_model(key: str) -> str:
    return str(OVERRIDE_PRESETS[key]["model"])


# ─────────────────────────────────────────────────────────────────────────────
# The two invariants that matter most: a default you cannot pick, or cannot
# price, is a bug on the day it ships.
# ─────────────────────────────────────────────────────────────────────────────


class TestEveryPresetDefaultIsSelectableAndPriced:
    @pytest.mark.parametrize("key", PRESET_KEYS)
    def test_default_model_is_in_its_own_choices(self, key: str) -> None:
        """Selecting a preset by its bare key and picking its default from the
        Models tab must resolve to the same model."""
        default = _default_model(key)
        choices = resolve_model_choices(key, {})
        assert default in choices, (
            f"OVERRIDE_PRESETS[{key!r}] defaults to {default!r}, but "
            f"resolve_model_choices({key!r}) returns {choices!r} — the Models "
            f"tab cannot select the model the slash command actually uses. Add "
            f"it to PRESET_MODEL_CHOICES[{key!r}] or change the default."
        )

    @pytest.mark.parametrize("key", PRESET_KEYS)
    def test_default_model_has_a_price(self, key: str) -> None:
        """No PRICING row means every token on this model bills $0.00."""
        default = _default_model(key)
        assert price_for(default) is not None, (
            f"OVERRIDE_PRESETS[{key!r}] defaults to {default!r}, which has no "
            f"PRICING entry — every token spent through /{key} would be "
            f"counted at $0.00. This is the qwen3.8-max failure: add the row to "
            f"prometheus.telemetry.cost.PRICING, verified against the "
            f"provider's own pricing page."
        )

    @pytest.mark.parametrize("key", PRESET_KEYS)
    def test_default_price_is_positive(self, key: str) -> None:
        inp, out = price_for(_default_model(key))  # type: ignore[misc]
        assert inp > 0 and out > 0, (
            f"/{key} defaults to {_default_model(key)!r} priced at "
            f"({inp}, {out}) — a $0 row is indistinguishable from a missing "
            f"one at every surface that renders it."
        )


class TestEverySelectableModelIsPriced:
    """A model the Models tab offers is a model a user can pick. Picking it must
    not silently produce an unbillable session."""

    @pytest.mark.parametrize("key", PRESET_KEYS)
    def test_all_choices_have_prices(self, key: str) -> None:
        unpriced = [m for m in resolve_model_choices(key, {}) if price_for(m) is None]
        assert not unpriced, (
            f"PRESET_MODEL_CHOICES[{key!r}] offers {unpriced!r} with no PRICING "
            f"row. The Models tab would let a user select these and then report "
            f"their spend as $0.00."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Site-by-site agreement.
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistryDefaultsMatchPresets:
    """CLOUD_DEFAULTS is the model used when a provider is the PRIMARY; the
    preset is the model used when it is a slash command. A user should not get
    a different model depending on which door they came through."""

    @pytest.mark.parametrize("provider", sorted(PRESET_BY_PROVIDER))
    def test_cloud_default_model_matches_preset(self, provider: str) -> None:
        key = PRESET_BY_PROVIDER[provider]
        registry_model = CLOUD_DEFAULTS.get(provider, {}).get("model")
        assert registry_model == _default_model(key), (
            f"CLOUD_DEFAULTS[{provider!r}]['model'] is {registry_model!r} but "
            f"OVERRIDE_PRESETS[{key!r}]['model'] is {_default_model(key)!r}. "
            f"`model.provider: {provider}` and `/{key}` would run different "
            f"models from the same configuration."
        )

    @pytest.mark.parametrize("provider", sorted(PRESET_BY_PROVIDER))
    def test_cloud_default_env_var_matches_preset(self, provider: str) -> None:
        key = PRESET_BY_PROVIDER[provider]
        registry_env = CLOUD_DEFAULTS.get(provider, {}).get("default_env")
        preset_env = OVERRIDE_PRESETS[key].get("api_key_env")
        assert registry_env == preset_env, (
            f"CLOUD_DEFAULTS[{provider!r}] reads {registry_env!r} but "
            f"/{key} reads {preset_env!r} — one of them will look unconfigured "
            f"while the other works."
        )


# Providers where setup DELIBERATELY writes something other than the slash
# command's default, with the reason. The preset answers "what should /claude
# cost for one chat turn"; setup answers "what should run the whole agent loop".
# Those can legitimately differ — but the difference has to be DECLARED here,
# because an undeclared one is indistinguishable from the qwen3.7/3.8 drift.
_DELIBERATE_SETUP_DIVERGENCE: dict[str, str] = {
    "anthropic": (
        "/claude defaults to Haiku 4.5 — cheap and fast for an interactive "
        "chat override. Setup is choosing a PRIMARY model that will run the "
        "whole agent loop, where Sonnet 5 is the better default (and at "
        "$2/$10 is cheaper than the Sonnet 4.6 this used to write)."
    ),
}


class TestNonInteractiveSetupMatchesPresets:
    """`oara setup --noninteractive` writes a config the rest of the system has
    to agree with. It shipped qwen3.7-max against everyone else's 3.8 for two
    releases precisely because nothing checked."""

    @pytest.mark.parametrize("provider", sorted(_CLOUD_FAST_PROVIDERS))
    def test_fast_path_model_matches_preset(self, provider: str) -> None:
        key = PRESET_BY_PROVIDER.get(provider)
        if key is None:
            pytest.skip(f"{provider} has no slash-command preset")
        _env, model, _limit = _CLOUD_FAST_PROVIDERS[provider]
        if provider in _DELIBERATE_SETUP_DIVERGENCE:
            assert model != _default_model(key), (
                f"{provider!r} is listed in _DELIBERATE_SETUP_DIVERGENCE but "
                f"setup and the preset now agree on {model!r}. Delete the "
                f"exemption — a stale one hides the next real drift."
            )
            return
        assert model == _default_model(key), (
            f"cli/init.py _CLOUD_FAST_PROVIDERS[{provider!r}] writes {model!r} "
            f"but OVERRIDE_PRESETS[{key!r}] says {_default_model(key)!r}. "
            f"Non-interactive setup would silently configure a different model "
            f"from the one every other entry point uses. If the difference is "
            f"intentional, add {provider!r} to _DELIBERATE_SETUP_DIVERGENCE "
            f"with the reason — do not just change the number."
        )

    @pytest.mark.parametrize("provider", sorted(_CLOUD_FAST_PROVIDERS))
    def test_fast_path_model_is_selectable(self, provider: str) -> None:
        key = PRESET_BY_PROVIDER.get(provider)
        if key is None:
            pytest.skip(f"{provider} has no slash-command preset")
        _env, model, _limit = _CLOUD_FAST_PROVIDERS[provider]
        assert model in resolve_model_choices(key, {}), (
            f"setup writes {model!r} for {provider!r}, which the Models tab "
            f"cannot select. This is the claude-sonnet-4-6 case: setup produced "
            f"a working config that the UI refused to display."
        )


class TestWizardMenuMatchesPresets:
    """The interactive wizard re-types both model ids AND prices."""

    def test_wizard_covers_every_keyed_preset(self) -> None:
        keyed = {
            OVERRIDE_PRESETS[k]["provider"]
            for k in PRESET_KEYS
            if OVERRIDE_PRESETS[k].get("api_key_env")
        }
        missing = sorted(keyed - set(CLOUD_PROVIDER_MODELS))
        assert not missing, (
            f"setup_wizard.CLOUD_PROVIDER_MODELS has no entry for {missing!r}, "
            f"so the interactive wizard cannot configure a provider the rest of "
            f"the system fully supports. Qwen was missing here for two releases "
            f"while cli/init.py offered it."
        )

    def test_wizard_providers_have_env_vars(self) -> None:
        missing = sorted(set(CLOUD_PROVIDER_MODELS) - set(CLOUD_DEFAULT_ENV_VARS))
        assert not missing, (
            f"CLOUD_PROVIDER_MODELS offers {missing!r} but CLOUD_DEFAULT_ENV_VARS "
            f"has no env var for them — the wizard raises KeyError mid-run."
        )

    @pytest.mark.parametrize("provider", sorted(CLOUD_PROVIDER_MODELS))
    def test_wizard_models_are_priced(self, provider: str) -> None:
        unpriced = [
            model
            for model, _desc, _price in CLOUD_PROVIDER_MODELS[provider]
            if price_for(model) is None
        ]
        assert not unpriced, (
            f"The wizard offers {unpriced!r} for {provider!r} with no PRICING "
            f"row — a user who picks one gets a config whose spend reports as "
            f"$0.00 forever."
        )

    @pytest.mark.parametrize("provider", sorted(CLOUD_PROVIDER_MODELS))
    def test_wizard_models_are_selectable(self, provider: str) -> None:
        key = PRESET_BY_PROVIDER.get(provider)
        if key is None:
            pytest.skip(f"{provider} has no slash-command preset")
        choices = set(resolve_model_choices(key, {}))
        offered = {m for m, _d, _p in CLOUD_PROVIDER_MODELS[provider]}
        assert offered <= choices, (
            f"The wizard offers {sorted(offered - choices)!r} for {provider!r}, "
            f"which PRESET_MODEL_CHOICES[{key!r}] cannot select."
        )

    @pytest.mark.parametrize("provider", sorted(CLOUD_PROVIDER_MODELS))
    def test_wizard_env_var_matches_preset(self, provider: str) -> None:
        key = PRESET_BY_PROVIDER.get(provider)
        if key is None:
            pytest.skip(f"{provider} has no slash-command preset")
        assert CLOUD_DEFAULT_ENV_VARS[provider] == OVERRIDE_PRESETS[key]["api_key_env"], (
            f"The wizard reads {CLOUD_DEFAULT_ENV_VARS[provider]!r} for "
            f"{provider!r} but /{key} reads "
            f"{OVERRIDE_PRESETS[key]['api_key_env']!r}."
        )


class TestShippedConfigMatchesPresets:
    """config/prometheus.yaml.default is what a new install reads. It is the
    ninth site, and the only one a USER edits — so it drifting is the one that
    produces "I set it and nothing changed" bug reports."""

    @staticmethod
    def _shipped() -> dict:
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        text = (root / "config" / "prometheus.yaml.default").read_text()
        return yaml.safe_load(text).get("slash_commands") or {}

    @staticmethod
    def _canonical(model: str) -> str:
        """Anthropic aliases resolve server-side to a dated snapshot, so
        `claude-haiku-4-5` and `claude-haiku-4-5-20251001` are the SAME model.
        Comparing the raw strings would report a difference that does not
        exist — and, worse, would push someone to "fix" it by editing one side."""
        from prometheus.providers.anthropic import _MODEL_ALIASES

        return _MODEL_ALIASES.get(model, model)

    def test_shipped_config_covers_every_preset(self) -> None:
        missing = sorted(set(PRESET_KEYS) - set(self._shipped()))
        assert not missing, (
            f"config/prometheus.yaml.default has no slash_commands block for "
            f"{missing!r}. A user reading the shipped config would not know the "
            f"command exists — /qwen was undiscoverable this way for two releases."
        )

    @pytest.mark.parametrize("key", PRESET_KEYS)
    def test_shipped_model_matches_preset(self, key: str) -> None:
        shipped = self._shipped().get(key) or {}
        assert self._canonical(str(shipped.get("model", ""))) == self._canonical(
            _default_model(key)
        ), (
            f"config/prometheus.yaml.default ships "
            f"slash_commands.{key}.model = {shipped.get('model')!r} but "
            f"OVERRIDE_PRESETS[{key!r}] defaults to {_default_model(key)!r}. "
            f"A fresh install and an install that deleted the section would run "
            f"different models."
        )

    @pytest.mark.parametrize("key", PRESET_KEYS)
    def test_shipped_env_var_matches_preset(self, key: str) -> None:
        shipped = self._shipped().get(key) or {}
        assert shipped.get("api_key_env") == OVERRIDE_PRESETS[key].get("api_key_env"), (
            f"slash_commands.{key}.api_key_env is "
            f"{shipped.get('api_key_env')!r} in the shipped config but "
            f"{OVERRIDE_PRESETS[key].get('api_key_env')!r} in the preset."
        )


class TestTheEscapeHatchIsDocumented:
    """`slash_commands.<key>.models` is the reason the built-in list is not a
    limit. It existed, worked, and was documented ONLY in a Python docstring —
    so from a user's point of view it did not exist at all. This asserts the
    doc against the CODE's behaviour, not the reverse."""

    @staticmethod
    def _shipped_text() -> str:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        return (root / "config" / "prometheus.yaml.default").read_text()

    def test_models_key_is_mentioned_in_the_shipped_config(self) -> None:
        text = self._shipped_text()
        assert "models:" in text and "REPLACES the built-in list" in text, (
            "config/prometheus.yaml.default must document the "
            "slash_commands.<key>.models override. Without it the Models tab "
            "reads as a hard limit, and the only way to learn otherwise is to "
            "read resolve_model_choices()."
        )

    def test_defaults_not_limits_is_stated(self) -> None:
        assert "DEFAULTS, NOT LIMITS" in self._shipped_text(), (
            "The shipped config must say plainly that the model string is "
            "free-form. This is the single most load-bearing sentence for a "
            "user whose provider shipped something newer than our release."
        )

    def test_the_override_actually_works(self) -> None:
        """The doc is only honest if the behaviour it describes is real."""
        cfg = {"slash_commands": {"claude": {"models": ["some-model-we-never-shipped"]}}}
        choices = resolve_model_choices("claude", cfg)
        assert "some-model-we-never-shipped" in choices, (
            "The shipped config now tells users that slash_commands.<key>.models "
            f"replaces the built-in list, but resolve_model_choices returned "
            f"{choices!r}. Documenting a capability that does not work is worse "
            f"than not documenting it."
        )
