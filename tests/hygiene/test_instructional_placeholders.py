"""Instructional table/column placeholders must use one canonical notation."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from aetherdialect._constants import (
    INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_TABLE_PLACEHOLDER,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "aetherdialect"

# Canonical instructional identifiers (plain English words like "tables" are not flagged).
_CANONICAL: frozenset[str] = frozenset(
    {
        INSTRUCTIONAL_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER,
        "column",
    }
)

# Alternate instructional forms that must not appear as live examples.
_FORBIDDEN: tuple[re.Pattern[str], ...] = (
    re.compile(r"<(?:table|column|col)(?:_\d+|\d+|_N|N)?>", re.IGNORECASE),
    re.compile(r"\btable_N\b"),
    re.compile(r"\bcolumn_N\b"),
    re.compile(r"\btable_\d+\b"),
    re.compile(r"\bcolumn_\d+\b"),
    re.compile(r"\btable\d+\b"),
    re.compile(r"\bcolumn\d+\b"),
    re.compile(r"\bcol\d+\b"),
)

_BAN_DOC = re.compile(
    r"(do not leave|never leave|angle-bracket|instructional tokens|"
    r"instructional placeholder|rewrite\s+`{0,2}table_N|map\s+`{0,2}table_N|"
    r"table_N[`\"]?-style leaks)",
    re.IGNORECASE,
)


def _is_ban_documentation(text: str) -> bool:
    """True when the string only documents forbidden forms (do-not-use / rewrite)."""
    return _BAN_DOC.search(text) is not None


def _forbidden_hits(text: str) -> list[str]:
    if _is_ban_documentation(text):
        return []
    found: list[str] = []
    for pat in _FORBIDDEN:
        for match in pat.finditer(text):
            token = match.group(0)
            if token in _CANONICAL:
                continue
            found.append(token)
    return found


def _scan_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        for token in _forbidden_hits(node.value):
            out.append(f"{path.name}:{getattr(node, 'lineno', 0)}: {token!r}")
    return out


@pytest.mark.fast
def test_instructional_placeholder_notation_is_canonical() -> None:
    """Only INSTRUCTIONAL_* snake tokens (table, table.column, …) may exemplify table/column shape."""
    assert "table" in _CANONICAL
    assert "table.column" in _CANONICAL
    violations: list[str] = []
    for path in sorted(_SRC.glob("*.py")):
        violations.extend(_scan_file(path))
    assert not violations, (
        "non-canonical instructional table/column notation "
        f"(canonical includes {sorted(_CANONICAL)[:8]}…):\n" + "\n".join(violations[:40])
    )


@pytest.mark.fast
def test_english_table_word_is_not_flagged() -> None:
    """Ordinary prose using the word table must not trip the instructional scanner."""
    assert _forbidden_hits("reorders the tables list before join planning") == []
    assert _forbidden_hits("Keep a table in join scope only via qualified tokens") == []
    assert _forbidden_hits("use table1 as the shape key") == ["table1"]
    assert _forbidden_hits("do not leave table_N or column_N instructional tokens") == []
