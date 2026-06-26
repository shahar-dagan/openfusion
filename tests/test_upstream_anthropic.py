"""Tests for Anthropic native upstream support."""

from __future__ import annotations

from openfusion.upstream import (
    _anthropic_stream_event_to_openai,
    _anthropic_to_openai,
    _openai_to_anthropic,
)

# ---------------------------------------------------------------------------
# _openai_to_anthropic
# ---------------------------------------------------------------------------


def test_basic_user_message() -> None:
    body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 512}
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=False)
    assert result["model"] == "claude-3-haiku-20240307"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]
    assert result["max_tokens"] == 512
    assert result["stream"] is False
    assert "system" not in result


def test_system_message_extracted() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
    }
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=True)
    assert result["system"] == "You are a helpful assistant."
    assert result["messages"] == [{"role": "user", "content": "Hi"}]
    assert result["stream"] is True


def test_default_max_tokens_applied_when_missing() -> None:
    body = {"messages": [{"role": "user", "content": "x"}]}
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=False)
    assert result["max_tokens"] == 1024


def test_optional_fields_forwarded() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["\n"],
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
    assert result["stop"] == ["\n"]


def test_none_optional_fields_not_forwarded() -> None:
    body = {"messages": [{"role": "user", "content": "x"}], "temperature": None}
    result = _openai_to_anthropic(body, "m", stream=False)
    assert "temperature" not in result


# ---------------------------------------------------------------------------
# _anthropic_to_openai
# ---------------------------------------------------------------------------


def test_response_conversion() -> None:
    raw = {
        "id": "msg_abc",
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Hello there!"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = _anthropic_to_openai(raw)
    assert result["id"] == "msg_abc"
    assert result["object"] == "chat.completion"
    assert result["model"] == "claude-3-haiku-20240307"
    assert result["choices"][0]["message"]["content"] == "Hello there!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15


def test_max_tokens_stop_reason() -> None:
    raw = {
        "id": "x",
        "model": "m",
        "stop_reason": "max_tokens",
        "content": [{"type": "text", "text": "truncated"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    result = _anthropic_to_openai(raw)
    assert result["choices"][0]["finish_reason"] == "length"


def test_multiple_text_blocks_concatenated() -> None:
    raw = {
        "id": "x",
        "model": "m",
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": "Part 1 "},
            {"type": "text", "text": "Part 2"},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }
    result = _anthropic_to_openai(raw)
    assert result["choices"][0]["message"]["content"] == "Part 1 Part 2"


# ---------------------------------------------------------------------------
# _anthropic_stream_event_to_openai
# ---------------------------------------------------------------------------


def test_content_block_delta() -> None:
    event = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Hello"},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["choices"][0]["delta"]["content"] == "Hello"
    assert chunk["choices"][0]["finish_reason"] is None


def test_message_delta_with_stop() -> None:
    event = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 42},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["choices"][0]["finish_reason"] == "stop"
    assert chunk["usage"]["completion_tokens"] == 42


def test_message_start_with_usage() -> None:
    event = {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 100}},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["usage"]["prompt_tokens"] == 100
    assert chunk["usage"]["completion_tokens"] == 0


def test_ping_returns_none() -> None:
    assert _anthropic_stream_event_to_openai({"type": "ping"}) is None


def test_unknown_event_returns_none() -> None:
    assert _anthropic_stream_event_to_openai({"type": "content_block_start"}) is None


# ---------------------------------------------------------------------------
# Round-trip: config provider inference
# ---------------------------------------------------------------------------


def test_panel_member_infers_anthropic_provider() -> None:
    from openfusion.config import PanelMember

    m = PanelMember(
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        model="claude-3-haiku-20240307",
    )
    assert m.provider == "anthropic"


def test_panel_member_infers_openai_provider() -> None:
    from openfusion.config import PanelMember

    m = PanelMember(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="gpt-4o-mini",
    )
    assert m.provider == "openai"


def test_panel_member_explicit_provider_wins() -> None:
    from openfusion.config import PanelMember

    m = PanelMember(
        base_url="https://example.com/v1",
        api_key="key",
        model="custom-model",
        provider="anthropic",
    )
    assert m.provider == "anthropic"
