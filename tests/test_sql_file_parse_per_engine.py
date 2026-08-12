"""Tests for per-engine SQL-file DDL parsing dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._schema_profile import _parse_sql_file_fallback, _parse_sql_file_sqlglot

MYSQL_DDL = """
CREATE TABLE `orders` (
  `order_id` INT NOT NULL PRIMARY KEY,
  `customer_id` INT NOT NULL,
  FOREIGN KEY (`customer_id`) REFERENCES `customers` (`customer_id`)
);
"""

TSQL_DDL = """
CREATE TABLE [orders] (
  [order_id] INT NOT NULL PRIMARY KEY,
  [customer_id] INT NOT NULL
);
"""

BIGQUERY_DDL = """
CREATE TABLE `proj.ds.orders` (
  order_id INT64 NOT NULL,
  customer_id INT64 NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES `proj.ds.customers` (customer_id) NOT ENFORCED
);
"""

DUCKDB_DDL = """
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER REFERENCES customers(customer_id)
);
"""

SQLITE_DDL = """
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER REFERENCES customers(customer_id)
);
"""

REDSHIFT_DDL = """
CREATE TABLE orders (
  order_id INTEGER NOT NULL PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id)
);
"""


def test_parse_sql_file_sqlglot_mysql_extracts_pk() -> None:
    tables = _parse_sql_file_sqlglot(MYSQL_DDL, "mysql")
    assert "orders" in tables
    assert tables["orders"]["primary_keys"] == ["order_id"]


def test_parse_sql_file_sqlglot_tsql_extracts_table() -> None:
    tables = _parse_sql_file_sqlglot(TSQL_DDL, "tsql")
    assert "orders" in tables
    assert "order_id" in tables["orders"]["column_names_original"]


@pytest.mark.parametrize(
    ("dialect_token", "ddl", "table", "fk_dst"),
    [
        ("bigquery", BIGQUERY_DDL, "orders", "customers"),
        ("duckdb", DUCKDB_DDL, "orders", "customers"),
        ("sqlite", SQLITE_DDL, "orders", "customers"),
        ("redshift", REDSHIFT_DDL, "orders", "customers"),
    ],
)
def test_parse_sql_file_sqlglot_extracts_foreign_keys_per_engine(
    dialect_token: str,
    ddl: str,
    table: str,
    fk_dst: str,
) -> None:
    tables = _parse_sql_file_sqlglot(ddl, dialect_token)
    assert table in tables
    assert tables[table]["foreign_keys"]
    assert tables[table]["foreign_keys"][0]["dst_table"] == fk_dst


@pytest.mark.parametrize(
    ("engine", "expected_parser"),
    [
        ("postgresql", "pglast"),
        ("mysql", "sqlglot"),
        ("databricks", "sqlglot"),
        ("sqlserver", "sqlglot"),
        ("oracle", "sqlglot"),
        ("bigquery", "sqlglot"),
        ("duckdb", "sqlglot"),
        ("sqlite", "sqlglot"),
        ("redshift", "sqlglot"),
    ],
)
def test_parse_sql_file_fallback_dispatches_by_engine(engine: str, expected_parser: str) -> None:
    ddl = MYSQL_DDL if engine != "postgresql" else "CREATE TABLE orders (id INT PRIMARY KEY);"
    with patch.object(EngineConfig, "TYPE", engine):
        with patch(
            "aetherdialect._schema_profile._parse_sql_file_pglast_postgres",
            return_value={"t": {}},
        ) as pg_mock:
            with patch(
                "aetherdialect._schema_profile._parse_sql_file_sqlglot",
                return_value={"t": {}},
            ) as sg_mock:
                _parse_sql_file_fallback(ddl)
                if expected_parser == "pglast":
                    pg_mock.assert_called_once()
                    sg_mock.assert_not_called()
                else:
                    sg_mock.assert_called_once()
                    pg_mock.assert_not_called()
