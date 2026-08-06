"""CSV ingest must preserve DECIMAL column values as Decimal, not float."""

from __future__ import annotations

from decimal import Decimal

import pytest

from aetherdialect._dialect_sqlglot_engines import CsvDialect


@pytest.mark.fast
def test_decimal_column_not_float() -> None:
    for duckdb_type in ("DECIMAL(10,2)", "NUMERIC(18,4)"):
        value = CsvDialect._coerce_typed_cell("123.45", duckdb_type)
        assert isinstance(value, Decimal)
        assert value == Decimal("123.45")
        assert not isinstance(value, float)

    float_value = CsvDialect._coerce_typed_cell("123.45", "DOUBLE")
    assert isinstance(float_value, float)
    assert not isinstance(float_value, Decimal)
