"""Regression guard against plan-only vocabulary in library and script code."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SCAN_ROOTS = (
    _REPO_ROOT / "src" / "aetherdialect",
    _REPO_ROOT / "scripts",
)

_PLAN_ID_RE = re.compile(r"\bT\d{2}\b")
_TIER_LETTER_RE = re.compile(r"\bTier [A-Z]\b")

_ALLOWED_TIER_PATHS = frozenset(
    {
        _REPO_ROOT / "src" / "aetherdialect" / "_constants.py",
    }
)


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_no_plan_step_ids_in_library_or_scripts() -> None:
    """Library and maintainer scripts must not embed plan step identifiers."""
    hits: list[str] = []
    for root in _SCAN_ROOTS:
        for path in _iter_python_files(root):
            text = path.read_text(encoding="utf-8")
            for match in _PLAN_ID_RE.finditer(text):
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{match.group(0)}")
    assert not hits, "plan step ids found:\n" + "\n".join(hits)


def test_no_tier_letter_metacommentary_outside_constants() -> None:
    """Docstrings and comments must not use ad-hoc tier-letter plan vocabulary."""
    hits: list[str] = []
    for root in _SCAN_ROOTS:
        for path in _iter_python_files(root):
            if path.resolve() in _ALLOWED_TIER_PATHS:
                continue
            text = path.read_text(encoding="utf-8")
            for match in _TIER_LETTER_RE.finditer(text):
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{match.group(0)}")
    assert not hits, "tier-letter metacommentary found:\n" + "\n".join(hits)
