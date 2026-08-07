"""Case-insensitive fallback wrapping when native ILIKE is unavailable."""

from __future__ import annotations

import pytest

from aetherdialect._dialect_sqlglot_engines import DatabricksDialect


@pytest.mark.fast
def test_databricks_lower_without_trim() -> None:
    """Databricks ILIKE fallback should use LOWER(expr) like other dialects."""
    dialect = DatabricksDialect.__new__(DatabricksDialect)
    wrapped = dialect.render_case_insensitive_wrap("customers.name")
    assert wrapped == "LOWER(customers.name)"
    assert "TRIM" not in wrapped.upper()
