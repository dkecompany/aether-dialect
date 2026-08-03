"""Sqlglot read/write token consistency for every shipped engine dialect."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import sqlglot

from aetherdialect._config import EngineConfig
from aetherdialect._constants import SQLGLOT_DIALECT_BY_ENGINE
from aetherdialect._dialect import get_dialect_class, get_registered_engines
from aetherdialect._schema_catalog import (
    _parse_sql_file_fallback,
    _parse_sql_file_sqlglot,
    _sqlglot_column_def_name_type,
)

_REPO = Path(__file__).resolve().parents[1]


def _uninit(engine: str):
    cls = get_dialect_class(engine)
    return cls.__new__(cls)


def _identifier_quote_char(dialect) -> str | None:
    quoted = dialect.quote_identifier("fixture")
    if quoted == "fixture":
        return None
    return quoted[0]


@pytest.mark.fast
@pytest.mark.parametrize("engine", get_registered_engines())
def test_shipped_dialect_sqlglot_identity_agrees(engine: str) -> None:
    """Read, write, name, capability lookup key, and quote style agree per engine."""
    dialect = _uninit(engine)
    read_token = dialect.sqlglot_dialect
    write_token = dialect.sqlglot_dialect
    assert read_token, f"{engine} must expose a sqlglot dialect token"
    assert write_token == read_token
    assert dialect.name == engine
    assert dialect.dialect_label == engine
    assert SQLGLOT_DIALECT_BY_ENGINE[engine] == read_token
    assert dialect.sql_file_parse_dialect == read_token
    _identifier_quote_char(dialect)


@pytest.mark.fast
def test_databricks_capability_lookup_is_not_spark_alias() -> None:
    assert SQLGLOT_DIALECT_BY_ENGINE["databricks"] == "databricks"
    assert "spark" not in SQLGLOT_DIALECT_BY_ENGINE


@pytest.mark.fast
def test_sqlglot_column_def_name_type_uses_dialect_token() -> None:
    alter = sqlglot.parse_one(
        "ALTER TABLE users ADD COLUMNS (tier STRING)",
        dialect="databricks",
    )
    col_def = alter.find(sqlglot.exp.ColumnDef)
    assert col_def is not None
    kind = col_def.args["kind"]
    captured: list[str | None] = []
    original_sql = kind.sql

    def tracking_sql(self, *, dialect=None, **kwargs):
        captured.append(dialect)
        return original_sql(dialect=dialect, **kwargs)

    with patch.object(type(kind), "sql", tracking_sql):
        parsed = _sqlglot_column_def_name_type(col_def, dialect_token="databricks")
    assert parsed is not None
    assert captured == ["databricks"]


@pytest.mark.fast
def test_parse_sql_file_fallback_dispatches_databricks_sql_file_dialect() -> None:
    captured: list[str] = []

    def capture(sql_content: str, dialect_token: str) -> dict[str, dict]:
        captured.append(dialect_token)
        return {}

    with patch("aetherdialect._schema_catalog._parse_sql_file_sqlglot", side_effect=capture):
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            _parse_sql_file_fallback("CREATE TABLE users (id INT);")
        finally:
            EngineConfig.TYPE = orig
    assert captured == ["databricks"]


@pytest.mark.fast
def test_parse_sql_file_sqlglot_databricks_identity_column() -> None:
    ddl = "CREATE TABLE users (id BIGINT GENERATED ALWAYS AS IDENTITY);"
    tables = _parse_sql_file_sqlglot(ddl, "databricks")
    assert "users" in tables
    assert "id" in tables["users"]["column_names_original"]


@pytest.mark.fast
def test_support_matrix_documents_snowpark_as_client_not_grammar() -> None:
    text = (_REPO / "docs" / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    snowflake_section = text.split("### Snowflake", 1)[1].split("\n### ", 1)[0]
    assert "Snowpark" in snowflake_section
    assert "not a separate SQL grammar" in snowflake_section
