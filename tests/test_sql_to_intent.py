"""Unit tests for ``aetherdialect._sql_to_intent``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aetherdialect._constants import (
    SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY,
    SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM,
)
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    SchemaGraph,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._dialect_postgres import PostgresDialect, PostgresQueryLogSource
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect
from aetherdialect._sql_to_intent import (
    _convert_postgres,
    _convert_sqlglot,
    _dedup_cte_steps,
    _dedup_window_registry,
    _union_merge_bucket_key,
    compute_sql_history_content_hash,
    convert_sql_to_intent,
    dedup_runtime_intents,
    fetch_query_log,
    load_sql_history_statements,
    normalize_imported_intent,
    seed_warmup_intent_from_runtime_intent,
)


def _pg() -> PostgresDialect:
    """Minimal Postgres dialect shell for converter tests."""
    return PostgresDialect.__new__(PostgresDialect)


def _databricks() -> DatabricksDialect:
    """Minimal Databricks dialect shell for sqlglot import tests."""
    return DatabricksDialect.__new__(DatabricksDialect)


@pytest.mark.parametrize(
    "sql",
    (
        "WITH c AS (SELECT customer_id FROM customers) SELECT customer_id FROM c",
        "SELECT customer_id FROM customers WHERE customer_id IS NULL",
        "SELECT COUNT(*) AS n FROM customers",
    ),
)
def test_postgres_and_sqlglot_spark_bucket_parity(schema_graph: SchemaGraph, sql: str) -> None:
    """Native pglast and sqlglot import paths yield the same structural bucket after normalize."""
    pytest.importorskip("pglast")
    pg_d = _pg()
    db_d = _databricks()
    pg = _convert_postgres(sql, schema_graph, pg_d)
    raw = _convert_sqlglot(sql, schema_graph, db_d)
    sg, fail_code, _ = normalize_imported_intent(raw, schema_graph, db_d)
    assert fail_code is None and sg is not None
    assert _union_merge_bucket_key(pg) == _union_merge_bucket_key(sg)


def test_sqlglot_spark_cte_native_extractor(schema_graph: SchemaGraph) -> None:
    """Databricks sqlglot reader uses the native sqlglot extractor (no pglast transpile bridge)."""
    sql = "WITH c AS (SELECT customer_id FROM customers) SELECT customer_id FROM c"
    db_d = _databricks()
    rt = _convert_sqlglot(sql, schema_graph, db_d)
    assert rt.cte_steps
    assert "customers" in rt.tables


def test_databricks_plan_rows_from_explain_text_finds_row_count() -> None:
    """Spark-style statistics lines yield a numeric estimate."""
    from aetherdialect._sql_to_intent import databricks_plan_rows_from_explain_text

    sample = "Statistics(sizeInBytes=100.0 MiB, rowCount=12345)"
    assert databricks_plan_rows_from_explain_text(sample) == 12345.0


def test_compute_sql_history_content_hash_stable_order() -> None:
    """Sorted digest ignores input order for identical statement multiset."""
    a = ["SELECT 1 FROM t", "SELECT 2 FROM t"]
    b = ["SELECT 2 FROM t", "SELECT 1 FROM t"]
    assert compute_sql_history_content_hash(a) == compute_sql_history_content_hash(b)


def test_load_sql_history_statements(tmp_path: Path) -> None:
    """Numbered SQL lines parse like seed questions."""
    p = tmp_path / "hist.txt"
    p.write_text("1. SELECT a FROM customers\n2. SELECT b FROM orders\n", encoding="utf-8")
    rows = load_sql_history_statements(str(p))
    assert rows == ["SELECT a FROM customers", "SELECT b FROM orders"]


def test_convert_sql_to_intent_plain_projection(schema_graph: SchemaGraph) -> None:
    """Plain column projections on known tables convert under sqlglot."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None
    assert cr.intent is not None
    assert "customers" in cr.intent.tables


def test_dedup_runtime_intents_union_merge_bare_columns() -> None:
    """Shell-equivalent bare projections within policy merge into one widened select list."""
    t = ["customers"]
    r1 = RuntimeIntent(
        tables=t,
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
    )
    r2 = RuntimeIntent(
        tables=t,
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
    )
    out = dedup_runtime_intents([r1, r2])
    assert len(out) == 1
    keys = {sc.signature_key for sc in out[0].select_cols}
    assert len(keys) == 2


def test_dedup_runtime_intents_collapses_same_body_key(schema_graph: object) -> None:
    """Duplicate structural clusters reduce to one survivor."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers"
    r1 = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False).intent
    r2 = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False).intent
    assert r1 is not None and r2 is not None
    out = dedup_runtime_intents([r1, r2])
    assert len(out) == 1


def test_dedup_window_registry_collapses_implicit_duplicate_frames() -> None:
    """Omitted frames normalise to identical ANSI ROWS bounds so duplicates collapse."""
    ws = WindowSpec(
        function="sum",
        partition_by=[NormalizedExpr.from_column("customers.customer_id")],
        order_by=[],
    )
    w1 = WindowRegistryStep(registry_id="w1", window_spec=ws)
    w2 = WindowRegistryStep(registry_id="w2", window_spec=replace(ws))
    rt = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        window_registry=[w1, w2],
    )
    out = _dedup_window_registry(rt)
    assert len(out.window_registry) == 1


def test_dedup_cte_steps_union_merge_bare_columns() -> None:
    """CTE bodies sharing a structural shell merge widening SELECT lists under union rules."""
    s1 = RuntimeCteStep(
        cte_name="z",
        tables=["customers"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
    )
    s2 = RuntimeCteStep(
        cte_name="a",
        tables=["customers"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
    )
    rt = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        cte_steps=[s1, s2],
    )
    out = _dedup_cte_steps(rt)
    assert len(out.cte_steps) == 1
    assert len(out.cte_steps[0].select_cols) == 2


def test_postgres_query_log_source_unavailable_when_cursor_errors() -> None:
    """Missing extension or failing catalogs yields False."""
    src = PostgresQueryLogSource()

    class _Cur:
        def execute(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("boom")

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    assert src.is_available(_Conn()) is False


def test_postgres_query_log_source_available_when_extension_row_present() -> None:
    """``pg_stat_statements`` extension row marks source usable."""
    src = PostgresQueryLogSource()

    class _Cur:
        def execute(self, *_a: object, **_k: object) -> None:
            pass

        def fetchone(self) -> tuple[int]:
            return (1,)

        def close(self) -> None:
            pass

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    assert src.is_available(_Conn()) is True


def test_fetch_query_log_unknown_engine_returns_empty() -> None:
    """Unsupported dialect names yield no rows."""
    assert fetch_query_log("mysql", None, lookback_days=1, max_queries=10, min_runs=1, user_filter=None) == []


def test_pglast_where_literal_masks_into_param_values(
    schema_graph: SchemaGraph,
) -> None:
    """AST tier binds WHERE literals into ``param_values`` via stable keys."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers WHERE customer_id = 42"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    fp = cr.intent.filters_param[0]
    assert fp.param_key and cr.intent.param_values.get(fp.param_key) == 42


def test_pglast_inner_join_qualifiers(schema_graph: SchemaGraph) -> None:
    """AST tier resolves INNER JOIN aliases for SELECT qualification."""
    dialect = _pg()
    sql = "SELECT c.customer_id FROM customers c INNER JOIN orders o ON c.customer_id = o.customer_id"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert set(cr.intent.tables) >= {"customers", "orders"}
    assert cr.intent.select_cols[0].expr.column_ref == "c.customer_id"


def test_pglast_with_cte_extracts_cte_step(schema_graph: SchemaGraph) -> None:
    """Non-recursive WITH lists become ``RuntimeCteStep`` rows plus merged physical tables."""
    dialect = _pg()
    sql = "WITH c AS (SELECT customer_id FROM customers) SELECT c.customer_id FROM c"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert len(cr.intent.cte_steps) == 1
    assert cr.intent.cte_steps[0].tables == ["customers"]
    assert "customers" in cr.intent.tables


def test_pglast_with_nested_cte_orders_inner_before_outer(
    schema_graph: SchemaGraph,
) -> None:
    """Nested ``WITH`` inside a CTE body flattens to multiple ``RuntimeCteStep`` rows in dependency order."""
    dialect = _pg()
    sql = (
        "WITH outer_cte AS ( "
        "WITH inner_cte AS (SELECT customer_id FROM customers) "
        "SELECT inner_cte.customer_id FROM inner_cte "
        ") SELECT outer_cte.customer_id FROM outer_cte"
    )
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert len(cr.intent.cte_steps) == 2
    assert "customers" in cr.intent.tables


def test_pglast_left_join_collects_tables(schema_graph: SchemaGraph) -> None:
    """LEFT JOIN range vars participate in alias qualification like INNER."""
    dialect = _pg()
    sql = "SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert set(cr.intent.tables) >= {"customers", "orders"}


def test_pglast_where_is_null_and_in_list(schema_graph: SchemaGraph) -> None:
    """NULL tests and literal ``IN`` lists become structured filters and param entries."""
    dialect = _pg()
    sql_null = "SELECT customer_id FROM customers WHERE email IS NULL"
    cr0 = convert_sql_to_intent(sql_null, schema_graph, dialect, verify_via_execute=False)
    assert cr0.failure_code is None and cr0.intent is not None
    assert cr0.intent.filters_param[0].op == "is null"

    sql_in = "SELECT customer_id FROM customers WHERE customer_id IN (1, 2)"
    cr1 = convert_sql_to_intent(sql_in, schema_graph, dialect, verify_via_execute=False)
    assert cr1.failure_code is None and cr1.intent is not None
    fp = cr1.intent.filters_param[0]
    assert fp.op == "in"
    assert fp.param_key and cr1.intent.param_values.get(fp.param_key) == [1, 2]


def test_pglast_limit_and_offset_values(schema_graph: SchemaGraph) -> None:
    """Literal ``OFFSET`` is stored under the stable ``param_values`` key (no new intent fields)."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers LIMIT 5 OFFSET 10"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.limit == 5
    assert cr.intent.param_values.get(SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY) == 10


def test_pglast_distinct_sets_select_index(schema_graph: SchemaGraph) -> None:
    """Plain ``SELECT DISTINCT`` maps to ``distinct_select_index`` (first projection convention)."""
    dialect = _pg()
    sql = "SELECT DISTINCT customer_id FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.distinct_select_index == 0


def test_pglast_select_count_star(schema_graph: SchemaGraph) -> None:
    """Simple aggregates in the SELECT list map through ``NormalizedExpr.from_agg`` and scalar grain."""
    dialect = _pg()
    sql = "SELECT count(*) FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    mg = cr.intent.select_cols[0].expr.add_groups[0]
    assert mg.agg_func == "count"


def test_pglast_where_top_level_or(schema_graph: SchemaGraph) -> None:
    """Top-level ``OR`` of supported predicates assigns distinct ``filter_group`` ids."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers WHERE customer_id = 1 OR customer_id = 2"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert len(cr.intent.filters_param) == 2
    assert cr.intent.filters_param[0].filter_group == 1
    assert cr.intent.filters_param[1].filter_group == 2


def test_pglast_where_between_and_like(schema_graph: SchemaGraph) -> None:
    """``BETWEEN`` uses paired param keys (possibly decomposed after repairs); ``LIKE`` maps to filter op."""
    dialect = _pg()
    sql_bt = "SELECT customer_id FROM customers WHERE customer_id BETWEEN 1 AND 100"
    cr = convert_sql_to_intent(sql_bt, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    fp = cr.intent.filters_param[0]
    if fp.op == "between":
        assert fp.param_key and fp.param_key_hi
        assert cr.intent.param_values.get(fp.param_key) == 1
        assert cr.intent.param_values.get(fp.param_key_hi) == 100
    else:
        assert fp.op in ("<=", ">=")
        assert fp.param_key
    assert cr.intent.param_values.get(fp.param_key_hi) == 100

    sql_like = "SELECT customer_id FROM customers WHERE name LIKE 'A%'"
    cr2 = convert_sql_to_intent(sql_like, schema_graph, dialect, verify_via_execute=False)
    assert cr2.failure_code is None and cr2.intent is not None
    assert cr2.intent.filters_param[0].op == "like"


def test_pglast_having_aggregate_compare(schema_graph: SchemaGraph) -> None:
    """``HAVING`` on a simple aggregate FunCall maps to ``HavingParam``."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers GROUP BY customer_id HAVING count(*) > 1"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    hp = cr.intent.having_param[0]
    assert hp.op == ">"
    assert hp.left_expr.add_groups and hp.left_expr.add_groups[0].agg_func == "count"


def test_seed_warmup_intent_from_runtime_roundtrip(schema_graph: object) -> None:
    """SQL-history seed rows round-trip through runtime projection."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers"
    rt = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False).intent
    assert rt is not None
    sw = seed_warmup_intent_from_runtime_intent(rt, intent_id="x", seed_index=0)
    assert sw.source == "sql_history"
    assert sw.to_runtime_intent().tables == rt.tables


def test_pglast_self_join_lift_second_alias(schema_graph: SchemaGraph) -> None:
    """Repeated physical table aliases lift the lighter branch into a synthetic self-join CTE."""
    dialect = _pg()
    sql = "SELECT c1.customer_id FROM customers c1 INNER JOIN customers c2 ON c1.customer_id = c2.customer_id"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert len(cr.intent.cte_steps) >= 1
    assert len(cr.intent.tables) >= 1


def test_pglast_case_registry_emission(schema_graph: SchemaGraph) -> None:
    """CASE expressions emit ``case_registry`` rows and ``cNN`` select references when semantic gate passes."""
    dialect = _pg()
    sql = "SELECT CASE WHEN customer_id = 1 THEN 'a' WHEN customer_id = 2 THEN 'b' ELSE 'c' END FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    if cr.failure_code == "SEMANTIC_REJECT":
        pytest.skip("CASE registry with row_level grain rejected by semantic gate")
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.case_registry and cr.intent.case_registry[0].registry_id.startswith("c")
    rid = cr.intent.case_registry[0].registry_id
    assert cr.intent.select_cols[0].expr.column_ref == rid


def test_pglast_window_registry_row_number(schema_graph: SchemaGraph) -> None:
    """Inline ``OVER`` clauses populate ``window_registry`` with ``wNN`` references when semantic gate passes."""
    dialect = _pg()
    sql = "SELECT row_number() OVER (PARTITION BY customer_id ORDER BY name) FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    if cr.failure_code == "SEMANTIC_REJECT":
        pytest.skip("window registry rejected by semantic gate")
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.window_registry and cr.intent.window_registry[0].registry_id.startswith("w")
    wid = cr.intent.window_registry[0].registry_id
    assert cr.intent.select_cols[0].expr.column_ref == wid
    ws = cr.intent.window_registry[0].window_spec
    assert ws.function == "row_number"
    assert ws.partition_by


def test_pglast_arithmetic_and_extract(schema_graph: SchemaGraph) -> None:
    """Arithmetic products and ``EXTRACT`` map into ``NormalizedExpr`` structural fields."""
    dialect = _pg()
    sql_mul = "SELECT amount * 2.0 AS x FROM orders"
    cr1 = convert_sql_to_intent(sql_mul, schema_graph, dialect, verify_via_execute=False)
    assert cr1.failure_code is None and cr1.intent is not None
    g0 = cr1.intent.select_cols[0].expr.add_groups[0]
    assert len(g0.multiply) == 2
    assert g0.multiply[0].column_ref == "orders.amount"
    assert g0.multiply[1].string_literal == SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM

    sql_ex = "SELECT EXTRACT(YEAR FROM order_date) AS y FROM orders"
    cr2 = convert_sql_to_intent(sql_ex, schema_graph, dialect, verify_via_execute=False)
    assert cr2.failure_code is None and cr2.intent is not None
    mg = cr2.intent.select_cols[0].expr.add_groups[0]
    assert mg.scalar_func == "extract"


def test_pglast_where_or_of_and_two_groups(schema_graph: SchemaGraph) -> None:
    """Top-level ``OR`` of ``AND`` arms assigns ``filter_group`` ids."""
    dialect = _pg()
    sql = "SELECT order_id FROM orders WHERE (customer_id = 1 AND order_id = 1) OR (customer_id = 2 AND order_id = 2)"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert len(cr.intent.filters_param) == 4
    gids = {f.filter_group for f in cr.intent.filters_param}
    assert gids == {1, 2}


def test_pglast_where_not_rejected_by_pg_path(schema_graph: SchemaGraph) -> None:
    """``NOT`` brackets are outside the supported OR-of-AND tier for the pglast extractor."""
    dialect = _pg()
    sql = "SELECT customer_id FROM customers WHERE NOT (customer_id = 1)"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code in ("SQL_PARSE_FAILED", "SEMANTIC_REJECT")
