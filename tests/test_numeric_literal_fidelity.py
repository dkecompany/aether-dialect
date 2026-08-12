"""Tests for exact numeric literal parsing and rendering."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import pytest

from aetherdialect._dialect import Dialect
from aetherdialect._utils import parse_sql_numeric_literal, render_sql_numeric_literal


def _parameter_abstract_fingerprint(sql: str) -> str:
    abstracted, params = Dialect.parameter_abstract(sql, sqlglot_dialect="postgres")
    payload = abstracted + json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.fast
def test_exact_literals_survive_round_trip() -> None:
    """Parsed numeric literals keep their type and exact text through parse and render."""
    cases: list[tuple[str, type]] = [
        ("19.99", Decimal),
        ("0.1", Decimal),
        ("9007199254740993", int),
        ("1e10", float),
    ]
    for text, expected_type in cases:
        parsed = parse_sql_numeric_literal(text)
        assert isinstance(parsed, expected_type)
        assert render_sql_numeric_literal(text, parsed) == text

    sql_a = "SELECT 1.00000000000000001"
    sql_b = "SELECT 1.00000000000000002"
    assert _parameter_abstract_fingerprint(sql_a) != _parameter_abstract_fingerprint(sql_b)
