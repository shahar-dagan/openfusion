"""Tests for the sequential pipeline strategy."""

from __future__ import annotations

import pytest

from openfusion.config import PipelineStepConfig, PipelineStepUse
from openfusion.pipeline import _build_step_messages, _inject_step_outputs


# ---------------------------------------------------------------------------
# _inject_step_outputs
# ---------------------------------------------------------------------------


def test_inject_replaces_placeholder() -> None:
    result = _inject_step_outputs("Summary: {research}", {"research": "lots of facts"})
    assert result == "Summary: lots of facts"


def test_inject_multiple_placeholders() -> None:
    result = _inject_step_outputs(
        "{research} | {critique}",
        {"research": "R", "critique": "C"},
    )
    assert result == "R | C"


def test_inject_missing_placeholder_left_intact() -> None:
    result = _inject_step_outputs("hello {missing}", {"research": "R"})
    assert result == "hello {missing}"


def test_inject_empty_outputs() -> None:
    result = _inject_step_outputs("plain text", {})
    assert result == "plain text"


# ---------------------------------------------------------------------------
# _build_step_messages
# ---------------------------------------------------------------------------


def test_build_no_system_no_outputs() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    result = _build_step_messages(msgs, None, {})
    assert result == [{"role": "user", "content": "hi"}]


def test_build_strips_original_system() -> None:
    msgs = [
        {"role": "system", "content": "original"},
        {"role": "user", "content": "hi"},
    ]
    result = _build_step_messages(msgs, "new system", {})
    assert result[0] == {"role": "system", "content": "new system"}
    assert result[1] == {"role": "user", "content": "hi"}
    assert len(result) == 2


def test_build_injects_outputs_into_system() -> None:
    msgs = [{"role": "user", "content": "q"}]
    result = _build_step_messages(msgs, "Use {step1} here", {"step1": "answer1"})
    assert result[0]["content"] == "Use answer1 here"


def test_build_no_system_with_outputs_adds_context_block() -> None:
    msgs = [{"role": "user", "content": "q"}]
    result = _build_step_messages(msgs, None, {"step1": "prior output"})
    assert result[0]["role"] == "system"
    assert "prior output" in result[0]["content"]
    assert result[1] == {"role": "user", "content": "q"}


# ---------------------------------------------------------------------------
# PipelineStepConfig defaults
# ---------------------------------------------------------------------------


def test_step_default_use_is_solo() -> None:
    step = PipelineStepConfig(name="s1")
    assert step.use == PipelineStepUse.SOLO


def test_step_explicit_fuse() -> None:
    step = PipelineStepConfig(name="s1", use=PipelineStepUse.FUSE)
    assert step.use == PipelineStepUse.FUSE


def test_step_model_override() -> None:
    step = PipelineStepConfig(name="s1", model="gpt-4o-mini")
    assert step.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# run_pipeline: empty steps raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_empty_steps_raises() -> None:
    from openfusion.config import (
        JudgeConfig,
        OpenFusionConfig,
        PanelMember,
        PassThroughConfig,
        PipelineConfig,
        Strategy,
    )
    from openfusion.errors import InvalidRequestError
    from openfusion.pipeline import run_pipeline
    from openfusion.upstream import UpstreamClient

    cfg = OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(steps=[]),
        panel=[PanelMember(base_url="https://mock/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(base_url="https://mock/v1", api_key="k", model="m"),
    )
    client = UpstreamClient()
    body = {"messages": [{"role": "user", "content": "hi"}]}

    gen = run_pipeline(body, cfg, client)
    with pytest.raises(InvalidRequestError, match="no steps"):
        async for _ in gen:
            pass
    await client.aclose()


# ---------------------------------------------------------------------------
# run_pipeline: single SOLO step (mocked upstream)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_single_solo_step() -> None:
    import httpx
    import respx

    from openfusion.config import (
        JudgeConfig,
        OpenFusionConfig,
        PanelMember,
        PassThroughConfig,
        PipelineConfig,
        PipelineStepConfig,
        PipelineStepUse,
        Strategy,
    )
    from openfusion.pipeline import run_pipeline
    from openfusion.upstream import UpstreamClient

    cfg = OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(
            steps=[PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)]
        ),
        panel=[PanelMember(base_url="https://mock/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(base_url="https://mock/v1", api_key="k", model="m"),
    )

    sse_body = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://mock/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        client = UpstreamClient()
        parts: list[str] = []
        async for delta, _usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if delta:
                parts.append(delta)
        await client.aclose()

    assert "".join(parts) == "hello"
