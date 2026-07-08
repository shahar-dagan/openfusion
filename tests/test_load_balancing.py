"""Tests for per-provider key round-robin (load balancing)."""

from __future__ import annotations

import pytest

from openfusion.health import KEY_REGISTRY, KeyPool, ProviderKeyRegistry

# ---------------------------------------------------------------------------
# KeyPool
# ---------------------------------------------------------------------------


def test_key_pool_single_key() -> None:
    pool = KeyPool(["sk-a"])
    assert pool.next_key() == "sk-a"
    assert pool.next_key() == "sk-a"


def test_key_pool_round_robins() -> None:
    pool = KeyPool(["sk-a", "sk-b", "sk-c"])
    keys = [pool.next_key() for _ in range(6)]
    assert keys == ["sk-a", "sk-b", "sk-c", "sk-a", "sk-b", "sk-c"]


def test_key_pool_empty_raises() -> None:
    with pytest.raises(ValueError):
        KeyPool([])


def test_key_pool_len() -> None:
    pool = KeyPool(["a", "b"])
    assert len(pool) == 2


# ---------------------------------------------------------------------------
# ProviderKeyRegistry
# ---------------------------------------------------------------------------


def test_registry_next_key_unknown_provider() -> None:
    reg = ProviderKeyRegistry()
    assert reg.next_key("openai") is None


def test_registry_has_pool_false_when_unregistered() -> None:
    reg = ProviderKeyRegistry()
    assert reg.has_pool("openai") is False


def test_registry_register_and_next_key() -> None:
    reg = ProviderKeyRegistry()
    reg.register("openai", ["sk-1", "sk-2"])
    assert reg.has_pool("openai") is True
    k1 = reg.next_key("openai")
    k2 = reg.next_key("openai")
    k3 = reg.next_key("openai")
    assert k1 == "sk-1"
    assert k2 == "sk-2"
    assert k3 == "sk-1"  # wraps back


def test_registry_replace_pool() -> None:
    reg = ProviderKeyRegistry()
    reg.register("openai", ["old-key"])
    reg.register("openai", ["new-a", "new-b"])
    assert reg.next_key("openai") == "new-a"


# ---------------------------------------------------------------------------
# ProviderConfig key consolidation
# ---------------------------------------------------------------------------


def test_provider_config_single_api_key() -> None:
    from openfusion.config import ProviderConfig

    pc = ProviderConfig(id="openai", api_key="sk-single")
    assert pc.api_keys == ["sk-single"]


def test_provider_config_multiple_keys() -> None:
    from openfusion.config import ProviderConfig

    pc = ProviderConfig(id="openai", api_keys=["sk-a", "sk-b"])
    assert len(pc.api_keys) == 2


def test_provider_config_api_key_merged_into_api_keys() -> None:
    from openfusion.config import ProviderConfig

    pc = ProviderConfig(id="openai", api_key="sk-x", api_keys=["sk-y"])
    assert "sk-x" in pc.api_keys
    assert "sk-y" in pc.api_keys


# ---------------------------------------------------------------------------
# Upstream round-robin integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_uses_pooled_key() -> None:
    """When KEY_REGISTRY has a pool for a provider, the upstream client should use it."""
    import httpx
    import respx

    from openfusion.config import PanelMember
    from openfusion.upstream import UpstreamClient

    # Isolate: temporarily replace KEY_REGISTRY pool for this provider
    KEY_REGISTRY.register("testprovider", ["pooled-key-1", "pooled-key-2"])

    captured_auth: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured_auth.append(request.headers.get("authorization", ""))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://testprovider.com/v1/chat/completions").mock(side_effect=capture)

        client = UpstreamClient()
        member = PanelMember(
            base_url="https://testprovider.com/v1",
            api_key="original-key",
            model="m",
        )
        await client.chat_completion(member, {"messages": []}, stream=False)
        await client.chat_completion(member, {"messages": []}, stream=False)
        await client.aclose()

    # Should have alternated between pooled-key-1 and pooled-key-2
    assert captured_auth[0] == "Bearer pooled-key-1"
    assert captured_auth[1] == "Bearer pooled-key-2"

    # Cleanup
    with KEY_REGISTRY._lock:
        KEY_REGISTRY._pools.pop("testprovider", None)
