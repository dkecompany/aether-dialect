"""Q1 — execution binds parameters instead of interpolating literals. Q2 — identifier quoting routes through one sqlglot-backed helper. Q3 — DISTINCT ON wrapper parses/wraps/regenerates via dialect AST. Q4 — predicate/clause bool chains use sqlglot ``exp.And`` / ``exp.Or``. Q5 — string-builder SQL parses via ``parse_select`` on every shipped dialect."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlglot import exp

import aetherdialect._sql_gen
from aetherdialect import _dialect_sqlglot_engines, _federation
from aetherdialect._constants import DISTINCT_ON_RANK_COLUMN
from aetherdialect._contracts_base import NormalizedExpr, OrderByCol, PredicateGroup, WhereParam
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._core_utils import reconcile_execute_bind_params
from aetherdialect._dialect import (
    Dialect,
    finalize_executable_sql,
    get_dialect,
    get_dialect_class,
    get_registered_engines,
)
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_helper import SqlAlchemyResultBackend
from aetherdialect._schema_graph import SchemaGraph
from aetherdialect._sql_gen import (
    _join_clause_parts_with_bool_op,
    _quote_simple_qualified_mul_token,
    _render_predicate_group_sql,
    wrap_core_sql_with_distinct_on,
)


class _MinimalDialect(Dialect):
    """Dialect stub for finalize_render bind-preservation tests."""

    name = "minimal"
    sqlglot_dialect = "postgres"

    def parse_select(self, sql: str):
        return object()

    def ast_validate_full(self, sql: str, **kw):
        return []

    def explain_diagnose(self, sql: str, params=None, **kwargs):
        return True, [], ""


def _blank_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=[],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_execute_keeps_bind_tokens_when_params_present() -> None:
    """Finalize path keeps :pN placeholders or execute receives a non- empty bind map."""
    dialect = _MinimalDialect(object())
    sql_param = "SELECT 1 AS n WHERE col = :p1"
    params = {"p1": "safe_value"}
    exec_sql = dialect.finalize_render(
        sql_param,
        params,
        schema=SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={}),
        intent=_blank_intent(),
    )
    bind_map = reconcile_execute_bind_params(exec_sql, params)
    assert ":p1" in exec_sql or (bind_map is not None and bool(bind_map))


@pytest.mark.fast
def test_hostile_param_value_cannot_inject_semicolon() -> None:
    """A hostile string value must stay bound, not inlined into executable SQL."""
    hostile = "x'; DROP TABLE users; --"
    dialect = _MinimalDialect(object())
    sql_param = "SELECT 1 AS n WHERE col = :p1"
    params = {"p1": hostile}
    exec_sql = dialect.finalize_render(
        sql_param,
        params,
        schema=SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={}),
        intent=_blank_intent(),
    )
    assert "DROP TABLE" not in exec_sql.upper()
    bind_map = reconcile_execute_bind_params(exec_sql, params)
    assert bind_map is not None
    assert bind_map["p1"] == hostile


@pytest.mark.fast
def test_sqlalchemy_backend_passes_bind_map_to_execute() -> None:
    """SqlAlchemy result backend passes a non-empty bind map when SQL still has tokens."""
    engine = MagicMock()
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = [(1,)]
    engine.connect.return_value.__enter__.return_value = connection
    backend = SqlAlchemyResultBackend(engine, dialect_name="postgres")
    sql = "SELECT 1 WHERE col = :p1"
    params = {"p1": "bound"}
    backend.fetch_rows(sql, params)
    assert connection.execute.call_count == 1
    call_args = connection.execute.call_args
    passed_bind = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs
    assert passed_bind == {"p1": "bound"}


@pytest.mark.fast
def test_finalize_executable_sql_preserves_scalar_bind_token() -> None:
    """Execution finalize keeps scalar placeholders for driver binding."""
    out = finalize_executable_sql(
        "SELECT 1 WHERE id = :p1",
        {"p1": 42},
        None,
        sqlglot_dialect="postgres",
    )
    assert ":p1" in out
    assert "42" not in out.split()


@pytest.mark.fast
def test_finalize_executable_sql_inlines_allowlisted_numeric_comma_list() -> None:
    """Numeric comma-separated IN bypass remains an explicit allowlisted inline."""
    out = finalize_executable_sql(
        "SELECT 1 WHERE id IN (:p1)",
        {"p1": "1, 2, 3"},
        None,
        sqlglot_dialect="postgres",
    )
    assert ":p1" not in out
    assert "1, 2, 3" in out


@pytest.mark.fast
def test_finalize_executable_sql_inlines_allowlisted_prequoted_in_list() -> None:
    """Pre-quoted IN-list bypass remains an explicit allowlisted inline."""
    out = finalize_executable_sql(
        "SELECT 1 WHERE rating IN (:p1)",
        {"p1": "'R','PG-13'"},
        None,
        sqlglot_dialect="postgres",
    )
    assert ":p1" not in out
    assert "R" in out
    assert "PG-13" in out


def _pg_render() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _where_leaf(pred: WhereParam) -> str:
    return f'"{pred.left_expr.primary_column.replace(".", '"."')}" = :p1'


@pytest.mark.fast
def test_wrap_core_sql_with_distinct_on_multiline_select_appends_rank_via_ast() -> None:
    """ROW_NUMBER must join the SELECT list via parse/emit, not first- line text surgery."""
    core_sql = 'SELECT\n  "customers"."id",\n  "customers"."name"\nFROM "customers"\nWHERE "customers"."balance" > 0'
    dialect = _pg_render()
    wrapped = wrap_core_sql_with_distinct_on(
        core_sql,
        select_exprs=['"customers"."id"', '"customers"."name"'],
        distinct_on=[NormalizedExpr.from_column("customers.id")],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC")],
        limit=None,
        dialect=dialect,
    )
    assert "SELECT," not in wrapped
    assert DISTINCT_ON_RANK_COLUMN in wrapped
    inner_start = wrapped.index("FROM (\n") + len("FROM (\n")
    inner_end = wrapped.rindex("\n) AS _don_src")
    inner_sql = wrapped[inner_start:inner_end]
    assert dialect.parse_select(inner_sql) is not None
    assert f"AS {DISTINCT_ON_RANK_COLUMN}" in inner_sql or f"AS {DISTINCT_ON_RANK_COLUMN.lower()}" in inner_sql


@pytest.mark.fast
def test_wrap_core_sql_with_distinct_on_preserves_existing_projection() -> None:
    """AST wrap keeps original select columns alongside the rank expression."""
    core_sql = 'SELECT "customers"."id", "customers"."name" FROM "customers"'
    dialect = _pg_render()
    wrapped = wrap_core_sql_with_distinct_on(
        core_sql,
        select_exprs=['"customers"."id"', '"customers"."name"'],
        distinct_on=[NormalizedExpr.from_column("customers.id")],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC")],
        limit=5,
        dialect=dialect,
    )
    inner_start = wrapped.index("FROM (\n") + len("FROM (\n")
    inner_end = wrapped.rindex("\n) AS _don_src")
    inner_sql = wrapped[inner_start:inner_end]
    normalized_inner = inner_sql.replace('"', "").lower()
    assert "customers.id" in normalized_inner
    assert "customers.name" in normalized_inner
    assert DISTINCT_ON_RANK_COLUMN in inner_sql
    assert "LIMIT 5" in wrapped


@pytest.mark.fast
def test_render_clause_chain_uses_ast_for_mixed_connectors() -> None:
    """Mixed AND/OR chains must render with correct outer parentheses."""
    out = _join_clause_parts_with_bool_op([("a = 1", "OR"), ("b = 2", "AND"), ("c = 3", "AND")])
    assert out.startswith("(")
    assert " OR " in out
    assert " AND " in out
    assert out.endswith(")")


@pytest.mark.fast
def test_render_predicate_group_sql_or_chain_parentheses() -> None:
    """OR predicate groups keep child-group parentheses when rendered via AST bool chains."""
    group = PredicateGroup(
        op="or",
        groups=(
            PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b"))),
            PredicateGroup(op="and", predicates=(_fp("t.c"),)),
        ),
    )
    sql = _render_predicate_group_sql(group, _where_leaf)
    assert " OR " in sql
    assert sql.count("(") >= 2
    assert '"t"."a"' in sql and '"t"."c"' in sql


@pytest.mark.fast
def test_join_on_equality_sql_removed() -> None:
    """Dead _join_on_equality_sql helper must not remain in _sql_gen."""
    assert not hasattr(aetherdialect._sql_gen, "_join_on_equality_sql")


def _fp(col: str) -> WhereParam:
    return WhereParam(left_expr=NormalizedExpr.from_column(col), op="=", raw_value="x", param_key="p1")


def _expected_sqlglot_quote(ident: str, dialect_name: str, *, quoted: bool = True) -> str:
    s = str(ident).strip()
    if dialect_name == "snowflake" and not quoted:
        return exp.to_identifier(s.upper(), quoted=False).sql(dialect=dialect_name)
    return exp.to_identifier(s, quoted=quoted).sql(dialect=dialect_name)


@pytest.mark.fast
def test_sqlglot_quote_identifier_helper_exists_and_matches_generator() -> None:
    """Single sqlglot-backed helper is the canonical quote path."""
    from aetherdialect._dialect import sqlglot_quote_identifier

    for dialect_name in ("postgres", "duckdb", "mysql", "tsql"):
        for ident in ("order", "plain_table", 'has"quote'):
            assert sqlglot_quote_identifier(ident, dialect_name) == _expected_sqlglot_quote(ident, dialect_name)


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine,sqlglot_name,quoted",
    [
        ("postgresql", "postgres", True),
        ("duckdb", "duckdb", True),
        ("mysql", "mysql", True),
        ("redshift", "redshift", True),
        ("sqlserver", "tsql", True),
        ("bigquery", "bigquery", True),
        ("databricks", "databricks", True),
        ("sqlite", "sqlite", True),
        ("snowflake", "snowflake", False),
    ],
)
def test_dialect_quote_identifier_routes_through_sqlglot(engine: str, sqlglot_name: str, quoted: bool) -> None:
    """Registered dialects quote identifiers via the shared sqlglot helper."""
    from aetherdialect._dialect import sqlglot_quote_identifier

    dialect = get_dialect_class(engine).__new__(get_dialect_class(engine))
    for ident in ("order", "reserved", 'x"y'):
        expected = _expected_sqlglot_quote(ident, sqlglot_name, quoted=quoted)
        if engine == "snowflake":
            # Snowflake quote_identifier still uses forced quoting for profiling paths.
            expected = _expected_sqlglot_quote(ident, sqlglot_name, quoted=True)
        assert dialect.quote_identifier(ident) == expected
        assert dialect.quote_identifier(ident) == sqlglot_quote_identifier(ident, sqlglot_name, quoted=True)


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine,sqlglot_name,quoted",
    [
        ("postgresql", "postgres", True),
        ("duckdb", "duckdb", True),
        ("mysql", "mysql", True),
        ("redshift", "redshift", True),
        ("sqlserver", "tsql", True),
        ("bigquery", "bigquery", True),
        ("databricks", "databricks", True),
        ("sqlite", "sqlite", True),
        ("snowflake", "snowflake", False),
    ],
)
def test_dialect_quote_table_column_routes_through_sqlglot(engine: str, sqlglot_name: str, quoted: bool) -> None:
    """table.column emission composes the same sqlglot-backed identifier helper."""
    from aetherdialect._dialect import sqlglot_quote_table_column

    dialect = get_dialect_class(engine).__new__(get_dialect_class(engine))
    got = dialect.quote_table_column("orders", "status")
    expected = sqlglot_quote_table_column("orders", "status", sqlglot_name, quoted=quoted)
    assert got == expected


@pytest.mark.fast
def test_federation_and_engine_loaders_share_sqlglot_quote_helper() -> None:
    """Federation glue and DuckDB CSV loaders must not keep local _quote_ident copies."""
    from aetherdialect._dialect import sqlglot_quote_identifier

    assert _federation._quote_ident is sqlglot_quote_identifier
    assert _dialect_sqlglot_engines._quote_ident is sqlglot_quote_identifier


@pytest.mark.fast
def test_federation_quote_ident_quotes_reserved_words_via_sqlglot() -> None:
    """Coordinator glue quotes reserved identifiers through sqlglot, not a blocklist."""
    from aetherdialect._dialect import sqlglot_quote_identifier

    assert _federation._quote_ident("order") == '"order"'
    assert _federation._quote_ident("group") == '"group"'
    assert _federation._quote_ident("plain_table") == sqlglot_quote_identifier("plain_table", "duckdb")


@pytest.mark.fast
def test_mul_group_term_quoting_uses_dialect_quote_table_column() -> None:
    """Multiply/divide column refs delegate to dialect.quote_table_column."""
    dialect = get_dialect("duckdb")
    node = NormalizedExpr.from_column("line.order_id")
    rendered = _quote_simple_qualified_mul_token(node.column_ref or "", dialect)
    assert rendered == dialect.quote_table_column("line", "order_id")
    assert rendered == '"line"."order_id"'


def _uninit_dialect_for_parse(engine: str) -> Dialect:
    cls = get_dialect_class(engine)
    dialect = cls.__new__(cls)
    if engine == "databricks":
        dialect.config = SimpleNamespace(CATALOG="parse_catalog", SCHEMA="parse_schema")
    return dialect


@pytest.mark.parametrize("engine", get_registered_engines())
@pytest.mark.parametrize("case_id", sorted(importlib.import_module("tests.test_dialect_conformance")._CASE_BY_ID))
@pytest.mark.fast
def test_build_deterministic_sql_parses_for_dialect(engine: str, case_id: str) -> None:
    """Conformance-rendered SQL must parse as a single SELECT in its target dialect."""
    conformance = importlib.import_module("tests.test_dialect_conformance")
    case = conformance._CASE_BY_ID[case_id]
    sql = conformance._render_conformance_sql(engine, case)
    dialect = _uninit_dialect_for_parse(engine)
    parsed = dialect.parse_select(sql)
    assert parsed is not None, f"{engine}/{case_id} did not parse: {sql!r}"
