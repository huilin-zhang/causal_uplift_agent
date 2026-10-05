"""Text providers for the two places where language matters: routing a question
and writing the final narrative.

DeterministicProvider is the default. It uses keyword rules and templates, so
tests and CI run without network or API keys. ClaudeProvider is optional
(AGENT_PROVIDER=anthropic plus credentials). It may only rephrase; every
number in its text is checked against the facts, and any mismatch falls
back to the template. Code guide: section "Providers" (sec:providers).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

INTENT_RULES = [
    ("customer", re.compile(r"\bcustomer\s*#?\s*\d+", re.I)),
    ("experiment", re.compile(r"\b(srm|balance|a/b|ab test|experiment|ate|significan|valid)", re.I)),
    ("method", re.compile(r"\b(what is|explain|why|how does|qini|pehe|x-learner|t-learner|causal forest|ipw)\b", re.I)),
]


class DeterministicProvider:
    name = "deterministic"

    def route(self, question: str) -> str:
        for intent, pattern in INTENT_RULES:
            if pattern.search(question or ""):
                return intent
        return "campaign"

    def narrate(self, facts: dict[str, Any], template_text: str) -> str:
        return template_text


class ClaudeProvider(DeterministicProvider):
    """Routing stays rule-based (cheap and testable); only the narrative uses the LLM."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("LLM_MODEL", "claude-opus-5-5")

    def narrate(self, facts: dict[str, Any], template_text: str) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=(
                "You write short briefs for a retention manager. Use only the facts given. "
                "Keep every number exactly as written. Plain sentences, no headings, under 180 words."
            ),
            messages=[{
                "role": "user",
                "content": "Facts (JSON):\n" + json.dumps(facts, default=str)
                + "\n\nDraft to improve:\n" + template_text,
            }],
        )
        if response.stop_reason == "refusal":
            return template_text
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text if numbers_match(text, template_text) else template_text


def numbers_match(text: str, reference: str) -> bool:
    """Every number in `text` must also appear in `reference` (the template built from state)."""
    allowed = set(re.findall(r"-?\d+(?:[.,]\d+)?", reference))
    return all(n in allowed for n in re.findall(r"-?\d+(?:[.,]\d+)?", text))


def get_provider(name: str):
    if name == "anthropic":
        try:
            return ClaudeProvider()
        except Exception:  # SDK missing or no credentials: stay deterministic
            return DeterministicProvider()
    return DeterministicProvider()
