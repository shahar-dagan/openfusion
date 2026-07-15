"""CLI summary and config-error friendliness tests."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from openfusion import cli, pricing
from openfusion.cli import (
    _chat_turn,
    _print_estimate,
    _run_ask,
    _summarize_config,
    build_setup_yaml,
    run_ask,
)
from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_price_cache() -> None:
    pricing._cache.clear()


async def test_run_ask_prints_fused_answer(mock_router, capsys: pytest.CaptureFixture[str]) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "the answer"}}]}
        )
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,  # avoid the judge stream for a clean capture
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
    )

    await _run_ask("what is 2+2?", config)

    assert "the answer" in capsys.readouterr().out


async def test_print_estimate_reports_calls_tokens_and_cost(
    mock_router, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "m1",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    }
                ]
            },
        )
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
    )

    await _print_estimate("what is 2+2?", config)

    out = capsys.readouterr().out
    assert "openfusion: estimate for this prompt" in out
    assert "m1" in out
    assert "calls:             1" in out
    assert "cost (est.):       $" in out


async def test_print_estimate_reports_unknown_cost_without_pricing(
    mock_router, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
    )

    await _print_estimate("what is 2+2?", config)

    assert "cost (est.):       unknown (pricing unavailable)" in capsys.readouterr().out


def test_run_ask_estimate_flag_skips_execution_and_api_key_check(
    tmp_path: Path,
    mock_router,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        "panel:\n  - base_url: https://mock.upstream/v1\n    api_key: k\n    model: m1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    called = False

    async def fail_if_called(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_run_ask", fail_if_called)

    run_ask("what is 2+2?", str(config_path), None, estimate=True)

    assert called is False
    assert "openfusion: estimate for this prompt" in capsys.readouterr().out


async def test_chat_turn_streams_and_returns_answer(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "fused reply"}}]}
        )
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
    )

    answer = await _chat_turn([{"role": "user", "content": "hi"}], config)

    assert "fused reply" in answer


def test_setup_yaml_loads_into_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(build_setup_yaml("budget", "sk-xyz"), encoding="utf-8")

    config = load_config(config_path)

    assert len(config.panel) == 3
    assert all(member.api_key == "sk-xyz" for member in config.panel)
    assert config.judge is not None and config.judge.api_key == "sk-xyz"
    assert config.tools.web_search is True


def test_summarize_config_reports_preset_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text("preset: budget\n", encoding="utf-8")
    config = load_config(config_path)

    summary = _summarize_config(config, "0.0.0.0", 8000)

    assert "preset=budget" in summary
    assert "web search+fetch" in summary
    assert 'model="openfusion"' in summary
    assert "http://0.0.0.0:8000" in summary


def test_missing_config_file_has_actionable_hint(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="cp examples/preset.yaml.example"):
        load_config(missing)


def test_missing_env_var_hint_includes_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        """
panel:
  - base_url: https://example.com/v1
    api_key: ${OPENROUTER_API_KEY}
    model: test
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="export OPENROUTER_API_KEY"):
        load_config(config_path)
