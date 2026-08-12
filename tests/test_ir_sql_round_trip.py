"""SQL ↔ intent ↔ SQL round-trip gate via abstract SQL fingerprints."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from aetherdialect._config import EngineConfig, SQLiteRuntimeConfig
from aetherdialect._contracts_base import MulGroup, NormalizedExpr, OrderByCol, PredicateGroup, WhereParam
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect import Dialect, DialectRegistry
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import SQLiteDialect
from aetherdialect._sql_gen import build_deterministic_sql
from aetherdialect._sql_to_intent import convert_sql_to_intent

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _sqlite_shell() -> SQLiteDialect:
    return SQLiteDialect.__new__(SQLiteDialect)


def _sqlglot_read(dialect: Any) -> str:
    return str(getattr(dialect, "sqlglot_dialect", "") or "postgres")


def _sql_abstract_fingerprint(sql: str, sqlglot_dialect: str) -> str:
    return Dialect.compute_sql_fp(sql, sqlglot_dialect=sqlglot_dialect)


def _assert_intent_sql_round_trip(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    *,
    verify_via_execute: bool = False,
) -> None:
    read = _sqlglot_read(dialect)
    sql1 = build_deterministic_sql(intent, schema=schema, dialect=dialect)
    cr = convert_sql_to_intent(sql1, schema, dialect, verify_via_execute=verify_via_execute)
    assert cr.failure_code is None, cr.failure_detail
    assert cr.intent is not None
    sql2 = build_deterministic_sql(cr.intent, schema=schema, dialect=dialect)
    assert _sql_abstract_fingerprint(sql1, read) == _sql_abstract_fingerprint(sql2, read)


def _concat_intent() -> RuntimeIntent:
    concat_expr = NormalizedExpr(
        add_groups=[
            MulGroup(
                multiply=[
                    NormalizedExpr.from_column("customers.name"),
                    NormalizedExpr.from_column("customers.email"),
                ],
                scalar_func="concat",
            )
        ]
    )
    return RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=concat_expr)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _simple_filter_intent() -> RuntimeIntent:
    where = PredicateGroup.from_list(
        [
            WhereParam(
                left_expr=NormalizedExpr.from_column("customers.email"),
                op="is null",
            )
        ]
    )
    return RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=where,
    )


def _distinct_on_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customers.id")),
            SelectCol(expr=NormalizedExpr.from_column("customers.name")),
        ],
        group_by_cols=[],
        order_by_cols=[
            OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC"),
            OrderByCol(expr=NormalizedExpr.from_column("customers.balance"), direction="DESC"),
        ],
        where=None,
        distinct_on=[NormalizedExpr.from_column("customers.id")],
    )


@pytest.mark.fast
def test_concat_intent_sql_round_trip(schema_graph: SchemaGraph) -> None:
    dialect = _sqlite_shell()
    _assert_intent_sql_round_trip(_concat_intent(), schema_graph, dialect)


@pytest.mark.fast
def test_simple_filter_intent_sql_round_trip(schema_graph: SchemaGraph) -> None:
    dialect = _sqlite_shell()
    _assert_intent_sql_round_trip(_simple_filter_intent(), schema_graph, dialect)


@pytest.mark.fast
def test_distinct_on_intent_sql_round_trip(simple_schema: SchemaGraph) -> None:
    dialect = _pg()
    _assert_intent_sql_round_trip(_distinct_on_intent(), simple_schema, dialect)


def _seed_schema_graph_customers(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE customers ("
        "customer_id INTEGER PRIMARY KEY, name TEXT, email TEXT, active INTEGER, created_at TEXT)"
    )
    connection.execute("INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com', 1, '2024-01-01')")
    connection.execute("INSERT INTO customers VALUES (2, 'Bob', NULL, 1, '2024-01-02')")
    connection.commit()


@pytest.fixture
def sqlite_memory_dialect() -> SQLiteDialect:
    orig_path = SQLiteRuntimeConfig.DATABASE_PATH
    orig_schema = SQLiteRuntimeConfig.SCHEMA
    orig_connection = SQLiteRuntimeConfig.NATIVE_CONNECTION
    EngineConfig.SCHEMA_JSON_PATH = ""
    EngineConfig.TYPE = "sqlite"
    EngineConfig.RUNTIME = SQLiteRuntimeConfig
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    _seed_schema_graph_customers(connection)
    SQLiteRuntimeConfig.DATABASE_PATH = ":memory:"
    SQLiteRuntimeConfig.SCHEMA = "main"
    SQLiteRuntimeConfig.NATIVE_CONNECTION = connection
    dialect = DialectRegistry.get("sqlite", SQLiteRuntimeConfig)
    yield dialect
    SQLiteRuntimeConfig.DATABASE_PATH = orig_path
    SQLiteRuntimeConfig.SCHEMA = orig_schema
    SQLiteRuntimeConfig.NATIVE_CONNECTION = orig_connection
    EngineConfig.TYPE = _ORIG_ENGINE_TYPE
    EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


@pytest.mark.fast
def test_simple_filter_verify_via_execute_sqlite_memory(
    schema_graph: SchemaGraph, sqlite_memory_dialect: SQLiteDialect
) -> None:
    _assert_intent_sql_round_trip(
        _simple_filter_intent(),
        schema_graph,
        sqlite_memory_dialect,
        verify_via_execute=True,
    )
