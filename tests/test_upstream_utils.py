"""Unit tests for upstream utility functions (extract_response_usage, member_from_dict)."""

from __future__ import annotations

import pytest

from openfusion.upstream import extract_response_usage, member_from_dict

# ---------------------------------------------------------------------------
# extract_response_usage
# ---------------------------------------------------------------------------


def test_returns_none_when_no_usage_key() -> None:
    assert extract_response_usage({}) is None
    assert extract_response_usage({"choices": []}) is None


def test_returns_none_when_usage_is_not_dict() -> None:
    assert extract_response_usage({"usage": None}) is None
    assert extract_response_usage({"usage": 42}) is None


def test_returns_none_when_usage_dict_is_empty() -> None:
    # An empty usage dict carries no countable fields → None.
    assert extract_response_usage({"usage": {}}) is None


def test_extracts_all_token_fields() -> None:
    payload = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    }
    result = extract_response_usage(payload)
    assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_extracts_cost_field() -> None:
    payload = {"usage": {"prompt_tokens": 5, "cost": 0.0001}}
    result = extract_response_usage(payload)
    assert result is not None
    assert result["cost"] == pytest.approx(0.0001)
    assert result["prompt_tokens"] == 5


def test_excludes_bool_token_values() -> None:
    # Booleans are ints in Python; they must be excluded to avoid counting True/False as tokens.
    payload = {"usage": {"prompt_tokens": True, "completion_tokens": 20}}
    result = extract_response_usage(payload)
    assert result is not None
    assert "prompt_tokens" not in result
    assert result["completion_tokens"] == 20


def test_excludes_bool_cost_value() -> None:
    payload = {"usage": {"cost": True}}
    result = extract_response_usage(payload)
    assert result is None  # bool cost excluded, nothing left


def test_partial_usage_dict() -> None:
    payload = {"usage": {"completion_tokens": 15}}
    result = extract_response_usage(payload)
    assert result == {"completion_tokens": 15}


def test_float_cost_converted() -> None:
    payload = {"usage": {"cost": 1}}  # int → float
    result = extract_response_usage(payload)
    assert result is not None
    assert isinstance(result["cost"], float)


# ---------------------------------------------------------------------------
# member_from_dict
# ---------------------------------------------------------------------------


def test_member_from_dict_basic() -> None:
    member = member_from_dict("https://api.example.com/v1", "sk-key", "gpt-4")
    assert member.base_url == "https://api.example.com/v1"
    assert member.api_key == "sk-key"
    assert member.model == "gpt-4"
    assert member.label is None


def test_member_from_dict_with_label() -> None:
    member = member_from_dict("https://api.example.com/v1", "sk-key", "gpt-key", label="my-panel")
    assert member.label == "my-panel"
