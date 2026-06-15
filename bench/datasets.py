#!/usr/bin/env python3
"""Dataset loaders for the benchmark.

GSM8K is fetched from the public openai/grade-school-math repo and cached
locally (gitignored). Only a deterministic prefix of the test split is used,
so runs are reproducible and cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

GSM8K_TEST_URL = (
    "https://github.com/openai/grade-school-math/raw/master/"
    "grade_school_math/data/test.jsonl"
)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# Appended to each question so both solo and fused answers end with a parseable
# final number; the numeric scorer reads the last number in the response.
GSM8K_INSTRUCTION = (
    "\n\nSolve this step by step. End your response with a line of the form "
    "'The answer is N' where N is the final number."
)


@dataclass
class RawTask:
    id: str
    prompt: str
    expected: str
    match: str


def _gold_answer(answer: str) -> str:
    """GSM8K gold answers end with a '#### <number>' marker."""
    marker = answer.rsplit("####", 1)
    tail = marker[1] if len(marker) == 2 else answer
    return tail.strip().replace(",", "")


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _fetch_jsonl(url: str, cache_name: str) -> list[dict]:
    cache = _cache_path(cache_name)
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        text = response.text
        cache.write_text(text, encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_gsm8k(limit: int) -> list[RawTask]:
    """Return the first ``limit`` GSM8K test problems as scorable tasks."""
    rows = _fetch_jsonl(GSM8K_TEST_URL, "gsm8k-test.jsonl")[:limit]
    tasks: list[RawTask] = []
    for index, row in enumerate(rows):
        tasks.append(
            RawTask(
                id=f"gsm8k-{index + 1}",
                prompt=str(row["question"]) + GSM8K_INSTRUCTION,
                expected=_gold_answer(str(row["answer"])),
                match="numeric",
            )
        )
    return tasks


LOADERS = {"gsm8k": load_gsm8k}
