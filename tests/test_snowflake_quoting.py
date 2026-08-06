"""Snowflake identifier quoting is always double-quoted."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._dialect_sqlglot_engines import SnowflakeDialect


def _snowflake_uninit() -> SnowflakeDialect:
    return SnowflakeDialect.__new__(SnowflakeDialect)


@pytest.mark.fast
def test_special_char_identifier_quoted() -> None:
    """Special-char and plain uppercase identifiers render quoted in Snowflake SQL."""
    d = _snowflake_uninit()
    d.sqlglot_dialect = "snowflake"

    hyphen_col = d.quote_table_column("orders", "ship-date")
    assert '"ship-date"' in hyphen_col
    assert hyphen_col.startswith('"') and "." in hyphen_col

    mixed = d.quote_identifier("MyTable")
    assert mixed == '"MyTable"'

    uppercase = d.quote_table_column("orders", "status")
    assert '"' in uppercase
    assert uppercase == '"orders"."status"'

    d.config = SimpleNamespace(DATABASE="DVDRENTAL_NEW", SCHEMA="PUBLIC", DEBUG=False)
    qualified = d._qualify_tables_for_execution("SELECT title FROM film")
    assert '"DVDRENTAL_NEW"."PUBLIC"."FILM"' in qualified.replace(" ", "")
