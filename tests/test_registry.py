"""Tests for the provider/model registry."""

from __future__ import annotations

import pytest

from openfusion.config import OpenFusionConfig, ProviderConfig, Tier
from openfusion.registry import ModelRegistry, ResolvedModel

# ---------------------------------------------------------------------------
# ModelRegistry.load — built-in catalog
# ---------------------------------------------------------------------------


def test_load_returns_registry() -> None:
    reg = ModelRegistry.load()
    assert isinstance(reg, ModelRegistry)


def test_builtin_providers_present() -> None:
    reg = ModelRegistry.load()
    for pid in ("openai", "anthropic", "groq", "mistral", "deepseek", "google"):
        assert reg.get_provider(pid) is not None, f"provider {pid!r} missing"


def test_builtin_models_present() -> None:
    reg = ModelRegistry.load()
    for mid in (
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4-5",
        "google/gemini-2.5-pro",
        "groq/llama-3.3-70b-versatile",
    ):
        assert reg.get_model(mid) is not None, f"model {mid!r} missing"


def test_model_tier_is_tier_enum() -> None:
    reg = ModelRegistry.load()
    model = reg.get_model("openai/gpt-4o-mini")
    assert model is not None
    assert model.tier == Tier.FAST


def test_model_context_is_positive() -> None:
    reg = ModelRegistry.load()
    for m in reg.list_models():
        assert m.context > 0, f"{m.id} has non-positive context"


# ---------------------------------------------------------------------------
# is_registered
# ---------------------------------------------------------------------------


def test_is_registered_known_provider() -> None:
    reg = ModelRegistry.load()
    assert reg.is_registered("openai/gpt-4o") is True


def test_is_registered_unknown_provider() -> None:
    reg = ModelRegistry.load()
    assert reg.is_registered("unknownprovider/some-model") is False


def test_is_registered_bare_model_name() -> None:
    reg = ModelRegistry.load()
    assert reg.is_registered("gpt-4o") is False


# ---------------------------------------------------------------------------
# resolve — key via explicit map
# ---------------------------------------------------------------------------


def test_resolve_with_explicit_key() -> None:
    reg = ModelRegistry.load()
    result = reg.resolve("openai/gpt-4o", api_keys={"openai": "sk-test"})
    assert isinstance(result, ResolvedModel)
    assert result.api_key == "sk-test"
    assert result.model_id == "gpt-4o"
    assert result.provider_id == "openai"
    assert result.format == "openai"


def test_resolve_anthropic_format() -> None:
    reg = ModelRegistry.load()
    result = reg.resolve("anthropic/claude-sonnet-4-5", api_keys={"anthropic": "sk-ant"})
    assert result is not None
    assert result.format == "anthropic"
    assert result.model_id == "claude-sonnet-4-5"


def test_resolve_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reg = ModelRegistry.load()
    result = reg.resolve("openai/gpt-4o", api_keys={})
    assert result is None


def test_resolve_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")
    reg = ModelRegistry.load()
    result = reg.resolve("openai/gpt-4o")
    assert result is not None
    assert result.api_key == "env-key-123"


def test_resolve_unknown_model_infers_provider() -> None:
    """A provider/model pair not in the catalog should still resolve if provider is known."""
    reg = ModelRegistry.load()
    result = reg.resolve("openai/gpt-5-turbo-preview", api_keys={"openai": "sk-x"})
    assert result is not None
    assert result.model_id == "gpt-5-turbo-preview"
    assert result.base_url == "https://api.openai.com/v1"


def test_resolve_unknown_provider_returns_none() -> None:
    reg = ModelRegistry.load()
    result = reg.resolve("noprovider/some-model", api_keys={"noprovider": "k"})
    assert result is None


# ---------------------------------------------------------------------------
# Extra providers / models via load()
# ---------------------------------------------------------------------------


def test_extra_provider_overrides_builtin() -> None:
    reg = ModelRegistry.load(
        extra_providers=[{"id": "openai", "base_url": "https://custom.proxy/v1", "format": "openai"}]  # noqa: E501
    )
    provider = reg.get_provider("openai")
    assert provider is not None
    assert provider.base_url == "https://custom.proxy/v1"


def test_extra_provider_adds_new() -> None:
    reg = ModelRegistry.load(
        extra_providers=[{"id": "myprovider", "base_url": "https://my.api/v1", "format": "openai"}]
    )
    assert reg.get_provider("myprovider") is not None


def test_extra_model_overrides_builtin() -> None:
    reg = ModelRegistry.load(
        extra_models=[
            {"id": "openai/gpt-4o", "provider": "openai", "tier": "fast", "context": 64000}
        ]
    )
    model = reg.get_model("openai/gpt-4o")
    assert model is not None
    assert model.tier == Tier.FAST
    assert model.context == 64000


# ---------------------------------------------------------------------------
# OpenFusionConfig.resolve_from_registry
# ---------------------------------------------------------------------------


def test_config_resolves_panel_member() -> None:
    cfg = OpenFusionConfig(
        providers=[ProviderConfig(id="openai", api_key="sk-test")],
        panel=[],  # will test via pass_through
        pass_through=None,
        judge=None,
    )
    # No panel/pass_through set so nothing to resolve — just check no error
    assert cfg.providers[0].api_key == "sk-test"


def test_config_resolves_pass_through_from_registry() -> None:
    from openfusion.config import PassThroughConfig

    cfg = OpenFusionConfig(
        providers=[ProviderConfig(id="groq", api_key="gsk-test")],
        pass_through=PassThroughConfig(
            base_url="",  # will be filled by registry
            api_key="",
            model="groq/llama-3.3-70b-versatile",
        ),
        judge=None,
    )
    assert cfg.pass_through is not None
    assert cfg.pass_through.base_url == "https://api.groq.com/openai/v1"
    assert cfg.pass_through.api_key == "gsk-test"
    assert cfg.pass_through.model == "llama-3.3-70b-versatile"
