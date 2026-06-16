"""Router gate: unit decisions, tool handling, and end-to-end SOLO routing."""

from __future__ import annotations

import httpx

from openfusion.config import OpenFusionConfig, RouterConfig, RouterMode
from openfusion.router import RouteDecision, route
from openfusion.server import _requires_pass_through_tools, create_app
from openfusion.tools import WEB_FETCH_TYPE, WEB_SEARCH_TYPE


def _body(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_short_prompt_routes_solo() -> None:
    assert route(_body("hi there"), RouterConfig(enabled=True)) == RouteDecision.SOLO


def test_keyword_routes_fuse() -> None:
    assert route(_body("compare A and B"), RouterConfig(enabled=True)) == RouteDecision.FUSE


def test_long_prompt_routes_fuse() -> None:
    assert route(_body("x " * 200), RouterConfig(enabled=True)) == RouteDecision.FUSE


def test_code_block_routes_fuse() -> None:
    decision = route(_body("fix ```py\nprint(1)\n```"), RouterConfig(enabled=True))
    assert decision == RouteDecision.FUSE


def test_mode_always_overrides_simple_prompt() -> None:
    assert route(_body("hi"), RouterConfig(mode=RouterMode.ALWAYS)) == RouteDecision.FUSE


def test_mode_never_overrides_hard_prompt() -> None:
    assert route(_body("compare " + "x " * 200), RouterConfig(mode=RouterMode.NEVER)) == (
        RouteDecision.SOLO
    )


def test_server_executable_tools_do_not_force_pass_through() -> None:
    body = {
        **_body("research this"),
        "tools": [{"type": WEB_SEARCH_TYPE}, {"type": WEB_FETCH_TYPE}],
    }
    assert _requires_pass_through_tools(body) is False


def test_function_tools_force_pass_through() -> None:
    body = {**_body("hi"), "tools": [{"type": "function", "function": {"name": "f"}}]}
    assert _requires_pass_through_tools(body) is True


def test_mixed_tools_force_pass_through() -> None:
    body = {**_body("hi"), "tools": [{"type": WEB_SEARCH_TYPE}, {"type": "function"}]}
    assert _requires_pass_through_tools(body) is True


def test_tool_role_message_forces_pass_through() -> None:
    assert _requires_pass_through_tools({"messages": [{"role": "tool", "content": "x"}]}) is True


async def test_router_solo_answers_with_single_call(
    test_config: OpenFusionConfig, mock_router
) -> None:
    test_config.router = RouterConfig(enabled=True, mode=RouterMode.NEVER)
    app = create_app(test_config)
    upstream = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "solo",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "single answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "single answer"
    # SOLO routing makes exactly one upstream call, not a panel fan-out.
    assert upstream.call_count == 1
