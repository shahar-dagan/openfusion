"""Panel gather tests."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from openfusion.config import (
    CostControlsConfig,
    OpenFusionConfig,
    PanelMember,
    SelfFusionConfig,
    Strategy,
    TimeoutsConfig,
)
from openfusion.errors import UpstreamError
from openfusion.panel import gather_panel
from openfusion.upstream import UpstreamClient


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.asyncio
async def test_gather_panel_all_succeed(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("ok"))
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="k",
                model="m1",
                label="a",
            ),
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="k",
                model="m2",
                label="b",
            ),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        client,
    )

    assert len(result.responses) == 2
    assert not result.failures
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_degrades_on_member_failure(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_completion("good")),
            httpx.Response(500, json={"error": {"message": "boom"}}),
        ]
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        client,
    )

    assert len(result.responses) == 1
    assert len(result.failures) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_clamps_internal_max_tokens(mock_router) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 7
        return httpx.Response(200, json=_completion("ok"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=upstream_handler
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
        cost_controls=CostControlsConfig(panel_max_tokens=7),
    )
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99},
        config,
        client,
    )

    assert len(result.responses) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_all_fail_raises(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "down"}})
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    with pytest.raises(UpstreamError, match="All panel members failed"):
        await gather_panel(
            {"messages": [{"role": "user", "content": "hi"}]},
            config,
            client,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_cancels_in_flight_members_on_cancellation(mock_router) -> None:
    """Cancelling gather_panel (e.g. a client disconnect) must abort in-flight
    member calls, not let them run to completion in the background.

    gather_panel's `finally` block (openfusion/panel.py) cancels any panel-member
    task still running when `asyncio.gather(*tasks)` itself is cancelled. This is
    the mechanism that stops paying for upstream calls after a client disconnect;
    left untested, a regression here would silently keep running (and billing
    for) panel members no one is listening to anymore.
    """
    started = asyncio.Event()
    finished = {"n": 0}

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        finished["n"] += 1  # pragma: no cover - should never run
        return httpx.Response(200, json=_completion("too late"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=slow_handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    task = asyncio.create_task(
        gather_panel({"messages": [{"role": "user", "content": "hi"}]}, config, client)
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Let any orphaned background work run; the member's HTTP call should
    # already be cancelled rather than merely still in flight.
    await asyncio.sleep(0.05)
    assert finished["n"] == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_self_fusion_expands_members() -> None:
    from openfusion.panel import expand_panel_members

    config = OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        self_fusion=SelfFusionConfig(n=3),
        panel=[
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="k",
                model="solo",
                label="solo",
            ),
        ],
    )
    members = expand_panel_members(config)
    assert len(members) == 3
    assert members[0][1]["temperature"] == 0.3
    assert members[1][1]["temperature"] == 0.7
