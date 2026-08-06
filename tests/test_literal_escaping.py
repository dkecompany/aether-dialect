"""String literal escaping is dialect-owned and rejects hostile parameter shapes."""

from __future__ import annotations

import pytest

from aetherdialect._constants import UNSAFE_PARAM_LITERAL
from aetherdialect._dialect import DialectRegistry
from aetherdialect._dialect_sqlglot_engines import MariaDBDialect, MySQLDialect


@pytest.mark.fast
def test_backslash_escaped_on_mysql() -> None:
    """MySQL and MariaDB escape backslashes inside string literal bodies."""
    mysql = MySQLDialect.__new__(MySQLDialect)
    maria = MariaDBDialect.__new__(MariaDBDialect)
    raw = r"a\b"
    assert mysql.escape_string_literal(raw) == r"a\\b"
    assert mysql.quote_string_literal(raw) == r"'a\\b'"
    assert maria.escape_string_literal(raw) == r"a\\b"
    assert maria.quote_string_literal(raw) == r"'a\\b'"


@pytest.mark.fast
def test_terminator_in_value_refused() -> None:
    """Statement terminators and SQL comment sequences are refused, not escaped."""
    dialect_cls = DialectRegistry.get_class("duckdb")
    dialect = dialect_cls.__new__(dialect_cls)
    hostile_values = (
        "x; DROP TABLE users",
        "x--comment",
        "x/*y*/",
    )
    for value in hostile_values:
        with pytest.raises(ValueError, match=UNSAFE_PARAM_LITERAL):
            dialect.escape_string_literal(value)
        with pytest.raises(ValueError, match=UNSAFE_PARAM_LITERAL):
            dialect.quote_string_literal(value)
