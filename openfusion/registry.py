"""Provider and model registry.

Resolves ``provider/model`` identifiers to concrete endpoint configuration
(base_url, api_key, wire format). Ships with a built-in catalog of well-known
providers; user config can add providers or override API keys.

Usage::

    registry = ModelRegistry.load()
    resolved = registry.resolve("anthropic/claude-sonnet-4-5", api_keys={"anthropic": "sk-..."})
    # resolved.base_url, resolved.api_key, resolved.format, resolved.tier ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from openfusion.config import Tier

_CATALOG_PATH = Path(__file__).parent / "data" / "providers.yaml"

ProviderFormat = Literal["openai", "anthropic"]


@dataclass
class ProviderEntry:
    id: str
    base_url: str
    format: ProviderFormat
    env_key: str | None = None  # environment variable holding the API key


@dataclass
class ModelEntry:
    id: str                     # e.g. "anthropic/claude-sonnet-4-5"
    provider: str               # provider id, e.g. "anthropic"
    context: int = 128000       # max context tokens
    tier: Tier = Tier.BALANCED
    input_cost: float = 0.0     # USD per million input tokens
    output_cost: float = 0.0    # USD per million output tokens


@dataclass
class ResolvedModel:
    """Everything needed to make a call to a specific model."""

    model_id: str           # bare model name sent to the provider (no provider/ prefix)
    base_url: str
    api_key: str
    format: ProviderFormat
    tier: Tier
    context: int
    input_cost: float
    output_cost: float
    provider_id: str


def _parse_tier(value: Any) -> Tier:
    if isinstance(value, Tier):
        return value
    mapping = {"fast": Tier.FAST, "balanced": Tier.BALANCED, "strong": Tier.STRONG}
    return mapping.get(str(value).lower(), Tier.BALANCED)


class ModelRegistry:
    """Catalog of providers and models, plus resolution logic."""

    def __init__(
        self,
        providers: list[ProviderEntry],
        models: list[ModelEntry],
    ) -> None:
        self._providers: dict[str, ProviderEntry] = {p.id: p for p in providers}
        self._models: dict[str, ModelEntry] = {m.id: m for m in models}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        extra_providers: list[dict[str, Any]] | None = None,
        extra_models: list[dict[str, Any]] | None = None,
    ) -> ModelRegistry:
        """Load the built-in catalog and merge any user-supplied extras."""
        with _CATALOG_PATH.open() as f:
            catalog = yaml.safe_load(f)

        providers = [
            ProviderEntry(
                id=p["id"],
                base_url=p["base_url"].rstrip("/"),
                format=p.get("format", "openai"),
                env_key=p.get("env_key"),
            )
            for p in catalog.get("providers", [])
        ]
        models = [
            ModelEntry(
                id=m["id"],
                provider=m["provider"],
                context=int(m.get("context", 128000)),
                tier=_parse_tier(m.get("tier", "balanced")),
                input_cost=float(m.get("input_cost", 0.0)),
                output_cost=float(m.get("output_cost", 0.0)),
            )
            for m in catalog.get("models", [])
        ]

        # Merge user extras (overwrite by id)
        for raw in extra_providers or []:
            pe = ProviderEntry(
                id=raw["id"],
                base_url=raw["base_url"].rstrip("/"),
                format=raw.get("format", "openai"),
                env_key=raw.get("env_key"),
            )
            providers = [p for p in providers if p.id != pe.id]
            providers.append(pe)

        for raw in extra_models or []:
            me = ModelEntry(
                id=raw["id"],
                provider=raw["provider"],
                context=int(raw.get("context", 128000)),
                tier=_parse_tier(raw.get("tier", "balanced")),
                input_cost=float(raw.get("input_cost", 0.0)),
                output_cost=float(raw.get("output_cost", 0.0)),
            )
            models = [m for m in models if m.id != me.id]
            models.append(me)

        return cls(providers, models)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_provider(self, provider_id: str) -> ProviderEntry | None:
        return self._providers.get(provider_id)

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    def list_models(self) -> list[ModelEntry]:
        return list(self._models.values())

    def list_providers(self) -> list[ProviderEntry]:
        return list(self._providers.values())

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        model_id: str,
        api_keys: dict[str, str] | None = None,
    ) -> ResolvedModel | None:
        """Resolve a ``provider/model`` id to a :class:`ResolvedModel`.

        Returns ``None`` when the model or its provider is not in the catalog.
        ``api_keys`` is a mapping of provider id → API key; if absent, falls
        back to the provider's ``env_key`` environment variable.
        """
        model_entry = self._models.get(model_id)
        if model_entry is None:
            # Try inferring provider from the "provider/..." prefix
            if "/" in model_id:
                provider_id, bare_model = model_id.split("/", 1)
                provider_entry = self._providers.get(provider_id)
                if provider_entry is None:
                    return None
                # Unknown model — use defaults from provider
                model_entry = ModelEntry(
                    id=model_id,
                    provider=provider_id,
                )
            else:
                return None
        else:
            provider_entry = self._providers.get(model_entry.provider)
            if provider_entry is None:
                return None

        api_key = _resolve_key(provider_entry, api_keys)
        if not api_key:
            return None

        # Bare model name: strip the "provider/" prefix before sending to provider
        bare_model = model_id.split("/", 1)[1] if "/" in model_id else model_id

        return ResolvedModel(
            model_id=bare_model,
            base_url=provider_entry.base_url,
            api_key=api_key,
            format=provider_entry.format,
            tier=model_entry.tier,
            context=model_entry.context,
            input_cost=model_entry.input_cost,
            output_cost=model_entry.output_cost,
            provider_id=provider_entry.id,
        )

    def is_registered(self, model_id: str) -> bool:
        """Return True if model_id looks like a registry entry (provider/model pattern)."""
        if "/" not in model_id:
            return False
        provider_id = model_id.split("/", 1)[0]
        return provider_id in self._providers


def _resolve_key(provider: ProviderEntry, api_keys: dict[str, str] | None) -> str:
    """Pick API key: explicit map → env var → empty string."""
    if api_keys:
        key = api_keys.get(provider.id, "")
        if key:
            return key
    if provider.env_key:
        return os.environ.get(provider.env_key, "")
    return ""


# Module-level singleton, lazy-loaded.
_REGISTRY: ModelRegistry | None = None


def get_registry(
    extra_providers: list[dict[str, Any]] | None = None,
    extra_models: list[dict[str, Any]] | None = None,
) -> ModelRegistry:
    """Return the singleton registry, building it on first call."""
    global _REGISTRY
    if _REGISTRY is None or extra_providers or extra_models:
        _REGISTRY = ModelRegistry.load(extra_providers, extra_models)
    return _REGISTRY
