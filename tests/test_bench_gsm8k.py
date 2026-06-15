"""Tests for numeric scoring and the GSM8K loader (offline)."""

from __future__ import annotations

from bench import datasets
from bench.datasets import _gold_answer, load_gsm8k
from bench.run import Task, _last_number, _score


def test_numeric_scoring_extracts_final_number() -> None:
    task = Task(id="g1", prompt="p", expected="18", match="numeric")
    assert _score(task, "First add 9 and 9. The answer is 18")
    assert _score(task, "... so the total is 18.")
    assert not _score(task, "The answer is 19")


def test_numeric_scoring_handles_commas_and_currency() -> None:
    task = Task(id="g2", prompt="p", expected="1200", match="numeric")
    assert _score(task, "The answer is $1,200")


def test_numeric_scoring_no_number_is_incorrect() -> None:
    task = Task(id="g3", prompt="p", expected="5", match="numeric")
    assert not _score(task, "I am not sure.")


def test_last_number_picks_the_final_value() -> None:
    assert _last_number("steps: 3, then 4, total 7") == 7.0
    assert _last_number("no digits here") is None


def test_gold_answer_strips_marker_and_commas() -> None:
    assert _gold_answer("Some reasoning here.\n#### 1,234") == "1234"
    assert _gold_answer("no marker 42") == "no marker 42"


def test_load_gsm8k_builds_numeric_tasks(monkeypatch) -> None:
    rows = [
        {"question": "What is 2+2?", "answer": "two plus two\n#### 4"},
        {"question": "What is 3+5?", "answer": "three plus five\n#### 8"},
        {"question": "extra", "answer": "x\n#### 9"},
    ]
    monkeypatch.setattr(datasets, "_fetch_jsonl", lambda url, name: rows)

    tasks = load_gsm8k(limit=2)
    assert len(tasks) == 2
    assert tasks[0].id == "gsm8k-1"
    assert tasks[0].expected == "4"
    assert tasks[0].match == "numeric"
    assert "The answer is N" in tasks[0].prompt
    assert tasks[0].prompt.startswith("What is 2+2?")
