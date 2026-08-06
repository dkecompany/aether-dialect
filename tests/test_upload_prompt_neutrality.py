"""Upload and CSV LLM system prompts must stay value-neutral."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable
from typing import Any

import pytest

from aetherdialect._constants import (
    CSV_IDENTIFIER_NAMING_SYSTEM,
    UPLOAD_COLUMN_TRANSFORMS_SYSTEM,
    UPLOAD_INTERPRET_SYSTEM,
    UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS,
    UPLOAD_SUMMARY_SYSTEM,
)
from aetherdialect._contracts_base import UploadColumnTransformId

_CONSTANTS = importlib.import_module("aetherdialect._constants")

_DENYLIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bUSD\b"),
    re.compile(r"\bEUR\b"),
    re.compile(r"\bGBP\b"),
    re.compile(r"\bAUD\b"),
    re.compile(r"\$"),
    re.compile(r"€"),
    re.compile(r"£"),
    re.compile(r"\bN/A\b"),
    re.compile(r"\bnull\b", re.IGNORECASE),
    re.compile(r"yyyy", re.IGNORECASE),
    re.compile(r"mm/dd", re.IGNORECASE),
    re.compile(r"\be\.g\.", re.IGNORECASE),
    re.compile(r"\bfor example\b", re.IGNORECASE),
    re.compile(r"\brevenue\b", re.IGNORECASE),
    re.compile(r"\bcustomer\b", re.IGNORECASE),
    re.compile(r"\bworkbook\b", re.IGNORECASE),
    re.compile(r"\bspreadsheet\b", re.IGNORECASE),
    re.compile(r"\bhigh\b", re.IGNORECASE),
    re.compile(r"\bmedium\b", re.IGNORECASE),
    re.compile(r"\blow\b", re.IGNORECASE),
)


def _iter_text_fragments(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_text_fragments(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_text_fragments(item)


def _audit_constant_text(name: str) -> str:
    value = getattr(_CONSTANTS, name)
    return "\n".join(_iter_text_fragments(value))


def _matching_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    return [pattern.pattern for pattern in patterns if pattern.search(text)]


@pytest.mark.fast
def test_upload_prompt_constants_avoid_denylist_tokens() -> None:
    violations: list[str] = []
    for name in sorted(UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS):
        matches = _matching_patterns(_audit_constant_text(name), _DENYLIST_PATTERNS)
        if matches:
            violations.append(f"{name}: {matches}")
    assert not violations


@pytest.mark.fast
def test_upload_transform_system_lists_closed_vocabulary_only() -> None:
    text = UPLOAD_COLUMN_TRANSFORMS_SYSTEM.lower()
    for transform_id in UploadColumnTransformId:
        assert transform_id.value in text
    assert "json" in text
    assert "verified" in text


@pytest.mark.fast
def test_upload_system_constants_are_registered() -> None:
    expected = {
        "UPLOAD_COLUMN_TRANSFORMS_SYSTEM",
        "UPLOAD_INTERPRET_SYSTEM",
        "UPLOAD_SUMMARY_SYSTEM",
        "CSV_IDENTIFIER_NAMING_SYSTEM",
    }
    assert expected <= UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS
    assert CSV_IDENTIFIER_NAMING_SYSTEM
    assert UPLOAD_SUMMARY_SYSTEM
    assert UPLOAD_INTERPRET_SYSTEM
