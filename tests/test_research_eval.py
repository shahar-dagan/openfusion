"""Tests for the pairwise research eval logic (offline)."""

from __future__ import annotations

from bench.research_eval import fusion_is_a, winner_from_verdict


def test_fusion_is_a_is_deterministic() -> None:
    assert fusion_is_a("r1") == fusion_is_a("r1")
    # Mixed assignment across ids (not all the same slot).
    assignments = {fusion_is_a(f"r{i}") for i in range(10)}
    assert assignments == {True, False}


def test_winner_when_fusion_is_a() -> None:
    assert winner_from_verdict("A", fusion_was_a=True) == "fusion"
    assert winner_from_verdict("B", fusion_was_a=True) == "solo"
    assert winner_from_verdict("TIE", fusion_was_a=True) == "tie"


def test_winner_when_fusion_is_b() -> None:
    assert winner_from_verdict("A", fusion_was_a=False) == "solo"
    assert winner_from_verdict("B", fusion_was_a=False) == "fusion"


def test_winner_tolerates_verbose_verdicts() -> None:
    assert winner_from_verdict("A is better", fusion_was_a=True) == "fusion"
    assert winner_from_verdict("Answer B", fusion_was_a=False) == "fusion"
    assert winner_from_verdict("It's a tie", fusion_was_a=True) == "tie"


def test_winner_defaults_to_tie_when_unparseable() -> None:
    assert winner_from_verdict("???", fusion_was_a=True) == "tie"
