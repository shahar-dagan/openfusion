"""HTTP-level tests for the OpenAI-compatible upstream call path's error handling.

The happy paths are exercised indirectly throughout the suite (panel, synthesize,
router tests all drive UpstreamClient.chat_completion against a mocked OpenAI-
compatible endpoint); this file targets the malformed-upstream-response branches
that had no direct coverage.
"""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import PanelMember
from openfusion.errors import UpstreamError
from openfusion.upstream import UpstreamClient


def _member(**overrides: object) -> PanelMember:
    defaults: dict[str, object] = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-test",
        "model": "some/model",
        "label": "panel",
    }
    defaults.update(overrides)
    return PanelMember(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_non_streaming_invalid_json_body_raises_upstream_error(mock_router) -> None:
    mock_router.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"not json at all")
    )
    client = UpstreamClient()

    with pytest.raises(UpstreamError, match="invalid JSON"):
        await client.chat_completion(
            _member(), {"messages": [{"role": "user", "content": "hi"}]}, stream=False
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_streaming_invalid_json_chunk_raises_upstream_error(mock_router) -> None:
    sse_body = "data: {not valid json\n\ndata: [DONE]\n\n"
    mock_router.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_body
        )
    )
    client = UpstreamClient()

    stream = await client.chat_completion(
        _member(), {"messages": [{"role": "user", "content": "hi"}]}, stream=True
    )
    with pytest.raises(UpstreamError, match="Invalid upstream SSE payload"):
        async for _ in stream:
            pass
    await client.aclose()
