"""Per-prompt router gate: decide whether a request is worth fusing.

Fusion runs N panel calls plus a judge, so it earns its cost on hard, open-ended
prompts and wastes it on trivial ones. When `router.enabled` is set, this module
makes that call up front from cheap prompt-shape signals (no model call), so the
proxy can answer simple prompts with a single pass-through completion and reserve
the panel for prompts that look like they benefit from it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from openfusion.config import RouterConfig, RouterMode


class RouteDecision(StrEnum):
    FUSE = "fuse"  # run the panel + aggregator
    SOLO = "solo"  # answer with a single pass-through call


def _user_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # OpenAI multimodal content blocks: collect text parts only.
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
    return "\n".join(parts)


def route(body: dict[str, Any], config: RouterConfig) -> RouteDecision:
    """Return whether to FUSE or answer SOLO for this request."""
    if config.mode == RouterMode.ALWAYS:
        return RouteDecision.FUSE
    if config.mode == RouterMode.NEVER:
        return RouteDecision.SOLO

    text = _user_text(body)
    lowered = text.lower()

    # A code block or an explicitly analytical ask is worth a panel.
    if "```" in text:
        return RouteDecision.FUSE
    if any(keyword in lowered for keyword in config.fuse_keywords):
        return RouteDecision.FUSE
    # Long prompts tend to carry enough substance to benefit from synthesis.
    if len(text) >= config.min_chars:
        return RouteDecision.FUSE
    return RouteDecision.SOLO
