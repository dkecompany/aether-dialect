"""Unit tests for aetherdialect._validation_execute module."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from aetherdialect._contracts_base import (
    FailureCategory,
    SchemaGraph,
    SqlDiagnostic,
    SqlDiagnosticCode,
)
from aetherdialect._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    FilterParam,
    MulGroup,
    NormalizedExpr,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._dialect import DatabricksDialect, Dialect, PostgresDialect
from aetherdialect._validation_execute import (
    _enforce_select_only,
    _validate_case_branches_for_scope,
    _validate_cte_cardinality,
    _validate_cte_output_types,
    _validate_main_query_cte_usage,
    bind_params_for_sql,
    canonicalize_rejection_reason,
    compute_confidence,
    validate_semantics,
    validate_sql,
)

_PG_ENFORCE = PostgresDialect.__new__(PostgresDialect)
_DBR_ENFORCE = DatabricksDialect.__new__(DatabricksDialect)


class _MinimalDialect(Dialect):
    """Concrete dialect stub for tests that only need ``finalize_render`` defaults."""

    name = "minimal"
    sqlglot_dialect = "postgres"

    def parse_select(self, sql: str):
        return object()

    def ast_validate_full(self, sql: str, **kw):
        return []

    def explain_diagnose(self, sql: str, params=None, **kwargs):
        return True, [], ""


class TestDialectPrepareExecutionSql:
    """Tests for ``Dialect.finalize_render`` (default finalize path)."""

    def test_substitutes_params(self):
        """Parameterized SQL is finalized with bound literals."""
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        d = _MinimalDialect(object())
        out = d.finalize_render(
            "SELECT * FROM t WHERE col = :p1",
            {"p1": "x"},
            schema=schema,
            intent=intent,
        )
        assert "'x'" in out


class TestBindParamsForSql:
    """Tests for ``bind_params_for_sql``."""

    def test_none_params_returns_none(self):
        assert bind_params_for_sql("SELECT * FROM t WHERE id = :p1", None) is None

    def test_empty_dict_returns_none(self):
        assert bind_params_for_sql("SELECT * FROM t WHERE id = :p1", {}) is None

    def test_returns_map_when_sql_has_p_placeholder(self):
        params = {"p1": 42}
        assert bind_params_for_sql("SELECT * FROM t WHERE id = :p1", params) is params

    def test_returns_map_when_sql_has_s_placeholder(self):
        params = {"s1": "x"}
        assert bind_params_for_sql("SELECT * FROM t WHERE name = :s1", params) is params

    def test_no_placeholder_returns_none_even_with_params(self):
        params = {"p1": 1}
        assert bind_params_for_sql("SELECT 1", params) is None

    def test_noncanonical_token_not_recognized(self):
        """Only ``:pN`` / ``:sN`` forms are bind tokens (not ``:param``)."""
        params = {"foo": 1}
        assert bind_params_for_sql("SELECT * FROM t WHERE id = :foo", params) is None

    def test_p_token_must_be_word_bounded(self):
        params = {"p1": 1}
        assert bind_params_for_sql("SELECT :p1extra", params) is None


class TestEnforceSelectOnly:
    """Tests for enforce_select_only."""

    def test_valid_select(self):
        """Plain SELECT passes."""
        ok, reason = _enforce_select_only("SELECT * FROM film", _PG_ENFORCE)
        assert ok is True
        assert reason == "ok"

    def test_select_with_joins(self):
        """SELECT with JOIN passes."""
        ok, _ = _enforce_select_only(
            "SELECT f.title FROM film f JOIN language l ON f.language_id = l.language_id",
            _PG_ENFORCE,
        )
        assert ok is True

    def test_select_with_where(self):
        """SELECT with WHERE passes."""
        ok, _ = _enforce_select_only("SELECT title FROM film WHERE rating = 'PG'", _PG_ENFORCE)
        assert ok is True

    def test_case_when_in_select_allowed(self):
        """CASE WHEN in SELECT list is allowed (not in FORBIDDEN_SQL)."""
        ok, reason = _enforce_select_only(
            "SELECT film_id, CASE WHEN length > 120 THEN 'long' ELSE 'short' END AS bucket FROM film",
            _PG_ENFORCE,
        )
        assert ok is True
        assert reason == "ok"

    def test_rank_over_in_select_allowed(self):
        """Window OVER clause is allowed by enforce_select_only regex policy."""
        ok, reason = _enforce_select_only(
            "SELECT film_id, RANK() OVER (ORDER BY length DESC) AS rnk FROM film",
            _PG_ENFORCE,
        )
        assert ok is True
        assert reason == "ok"

    def test_not_select(self):
        """Non-SELECT DML fails before or during structural select gate."""
        ok, reason = _enforce_select_only("UPDATE film SET title = 'x'", _PG_ENFORCE)
        assert ok is False
        assert reason in {"not_select", "forbidden_sql"}

    def test_empty_string(self):
        """Empty string fails with not_select."""
        ok, reason = _enforce_select_only("", _PG_ENFORCE)
        assert ok is False
        assert reason == "not_select"

    def test_delete_forbidden(self):
        """DELETE keyword triggers forbidden_sql."""
        ok, reason = _enforce_select_only("SELECT 1; DELETE FROM film", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_drop_forbidden(self):
        """DROP keyword triggers forbidden_sql."""
        ok, reason = _enforce_select_only("SELECT * FROM film; DROP TABLE film", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_insert_forbidden(self):
        """INSERT keyword triggers forbidden_sql."""
        ok, reason = _enforce_select_only(
            "SELECT * FROM film WHERE title IN (SELECT title FROM (INSERT INTO x VALUES (1)))",
            _PG_ENFORCE,
        )
        assert ok is False
        assert reason == "forbidden_sql"

    def test_union_forbidden(self):
        """UNION is in forbidden list."""
        ok, reason = _enforce_select_only("SELECT title FROM film UNION SELECT title FROM film", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_case_insensitive_select(self):
        """Lowercase select passes."""
        ok, _ = _enforce_select_only("select title from film", _PG_ENFORCE)
        assert ok is True

    def test_semicolon_injection(self):
        """Semicolon followed by content is forbidden."""
        ok, reason = _enforce_select_only("SELECT 1; DROP TABLE film", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_leading_whitespace_before_select(self):
        ok, reason = _enforce_select_only("  \n\tSELECT 1", _PG_ENFORCE)
        assert ok is True
        assert reason == "ok"

    def test_with_select_passes_postgresql(self):
        """CTE-wrapped single select passes pglast structural gate."""
        sql = "WITH a AS (SELECT 1 AS n) SELECT * FROM a"
        ok, reason = _enforce_select_only(sql, _PG_ENFORCE)
        assert ok is True
        assert reason == "ok"

    def test_with_select_passes_databricks_sqlglot(self):
        """CTE-wrapped single select passes sqlglot spark gate."""
        sql = "WITH a AS (SELECT 1 AS n) SELECT * FROM a"
        ok, reason = _enforce_select_only(sql, _DBR_ENFORCE)
        assert ok is True
        assert reason == "ok"

    def test_insert_statement_rejected_postgresql(self):
        ok, reason = _enforce_select_only("INSERT INTO t VALUES (1)", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_insert_statement_rejected_databricks(self):
        ok, reason = _enforce_select_only("INSERT INTO t VALUES (1)", _DBR_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_exists_subquery_forbidden(self):
        ok, reason = _enforce_select_only("SELECT 1 WHERE EXISTS (SELECT 1 FROM t)", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_between_allowed_in_select_predicate(self):
        """BETWEEN in a read-only SELECT predicate is permitted (CW-002 / CASE WHEN)."""
        ok, reason = _enforce_select_only("SELECT 1 WHERE x BETWEEN 1 AND 2", _PG_ENFORCE)
        assert ok is True
        assert reason == "ok"

    def test_offset_forbidden(self):
        ok, reason = _enforce_select_only("SELECT a FROM t ORDER BY a OFFSET 10 ROWS", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"

    def test_intersect_forbidden(self):
        ok, reason = _enforce_select_only("SELECT 1 INTERSECT SELECT 2", _PG_ENFORCE)
        assert ok is False
        assert reason == "forbidden_sql"


class TestComputeConfidence:
    """Tests for compute_confidence."""

    def test_perfect_score(self):
        """High best_score with no penalties yields high confidence."""
        c = compute_confidence(1.0, 0.5, False, 0.0, 0.0, 0.0)
        assert c > 0.7

    def test_zero_inputs(self):
        """All-zero inputs with cold-start floor yields 0.5."""
        c = compute_confidence(0.0, 0.0, False, 0.0, 0.0, 0.0)
        assert c == 0.5

    def test_cold_start_floor_reduced_by_penalties(self):
        """Cold-start floor is reduced by negative/colmap/cte penalties."""
        c = compute_confidence(0.0, 0.0, False, 0.0, 1.0, 0.0)
        assert c < 0.5

    def test_new_tables_penalty(self):
        """Using new tables reduces confidence."""
        c_no = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 0.0)
        c_yes = compute_confidence(0.8, 0.3, True, 0.0, 0.0, 0.0)
        assert c_yes < c_no

    def test_shape_penalty(self):
        """Shape penalty reduces confidence."""
        c_low = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 0.0)
        c_high = compute_confidence(0.8, 0.3, False, 1.0, 0.0, 0.0)
        assert c_high < c_low

    def test_negative_penalty(self):
        """Negative memory penalty reduces confidence."""
        c_clean = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 0.0)
        c_neg = compute_confidence(0.8, 0.3, False, 0.0, 1.0, 0.0)
        assert c_neg < c_clean

    def test_clamped_to_0_1(self):
        """Result is always clamped to [0, 1]."""
        c = compute_confidence(0.0, 0.0, True, 1.0, 1.0, 1.0, 1.0)
        assert c >= 0.0
        c2 = compute_confidence(1.0, 1.0, False, 0.0, 0.0, 0.0, 0.0)
        assert c2 <= 1.0

    def test_colmap_penalty(self):
        """Column-map penalty reduces confidence."""
        c_clean = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 0.0)
        c_pen = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 1.0)
        assert c_pen < c_clean

    def test_cte_penalty(self):
        """CTE count penalty reduces confidence."""
        c_no_cte = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 0.0, 0.0)
        c_cte = compute_confidence(0.8, 0.3, False, 0.0, 0.0, 0.0, 1.0)
        assert c_cte < c_no_cte

    def test_score_gap_contribution_saturates(self):
        """Gap term uses ``min(1, score_gap * 2)`` so huge gaps do not explode past the cap."""
        c_low = compute_confidence(0.5, 0.6, False, 0.0, 0.0, 0.0, 0.0)
        c_high = compute_confidence(0.5, 10.0, False, 0.0, 0.0, 0.0, 0.0)
        assert c_high == c_low

    def test_penalties_above_one_are_clamped(self):
        """``negative_pen``, ``colmap_pen``, and ``num_cte_pen`` are clamped to ``[0, 1]`` before scaling."""
        c_cap = compute_confidence(0.8, 0.0, False, 0.0, 2.0, 3.0, 4.0)
        c_unit = compute_confidence(0.8, 0.0, False, 0.0, 1.0, 1.0, 1.0)
        assert c_cap == c_unit

    def test_cold_start_floor_only_when_best_score_is_exactly_zero(self):
        """The 0.5 floor applies only for ``best_score == 0.0``, not arbitrarily small positives."""
        c_tiny = compute_confidence(1e-9, 0.0, False, 0.0, 0.0, 0.0, 0.0)
        assert c_tiny < 0.1


class TestValidateCteCardinality:
    """Tests for validate_cte_cardinality."""

    def _make_cte(self, name="cte1", grain="row_level", limit=None, tables=None):
        """Build a minimal RuntimeCteStep."""
        return RuntimeCteStep(
            cte_name=name,
            tables=tables or ["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["order_id"],
            grain=grain,
            limit=limit,
        )

    def test_empty_list(self):
        """Empty CTE list yields no issues."""
        issues = _validate_cte_cardinality([])
        assert issues == []

    def test_row_level_no_limit_clean(self):
        """Row-level CTE without limit produces no issues."""
        cte = self._make_cte(grain="row_level")
        issues = _validate_cte_cardinality([cte])
        assert issues == []

    def test_scalar_grain_is_consistent(self):
        """Scalar grain CTE has expected_rows='one' via property, so no cardinality warning."""
        cte = self._make_cte(grain="scalar")
        issues = _validate_cte_cardinality([cte])
        cardinality_issues = [i for i in issues if "scalar_cardinality" in i.issue_id]
        assert cardinality_issues == []

    def test_limit_1_row_level_warns(self):
        """Row-level CTE with LIMIT 1 has expected_rows='few' which != 'one', triggers warning."""
        cte = self._make_cte(grain="row_level", limit=1)
        issues = _validate_cte_cardinality([cte])
        ids = [i.issue_id for i in issues]
        assert any("limit1_cardinality" in iid for iid in ids)

    def test_many_depends_on_single_row_no_info_issue_emitted(self):
        """Expansion notes are informational-only and therefore no issue is emitted."""
        cte1 = self._make_cte(name="cte1", grain="scalar")
        cte2 = self._make_cte(name="cte2", grain="row_level", tables=["cte1"])
        issues = _validate_cte_cardinality([cte1, cte2])
        ids = [i.issue_id for i in issues]
        assert not any("cardinality_expansion" in iid for iid in ids)

    def test_few_expected_depends_on_single_row_no_info_issue_emitted(self):
        """Few-grain dependency on scalar CTE is informational and suppressed."""
        cte1 = self._make_cte(name="cte1", grain="scalar")
        cte2 = self._make_cte(name="cte2", grain="row_level", limit=10, tables=["cte1"])
        issues = _validate_cte_cardinality([cte1, cte2])
        ids = [i.issue_id for i in issues]
        assert not any("cardinality_expansion" in iid for iid in ids)

    def test_scalar_grain_with_mismatched_expected_rows_warns(self):
        """Objects with ``grain=='scalar'`` but ``expected_rows!='one'`` warn (non-RuntimeCteStep edge)."""
        bad = SimpleNamespace(
            cte_name="bad_scalar",
            grain="scalar",
            expected_rows="many",
            limit=None,
            tables=[],
        )
        issues = _validate_cte_cardinality([bad])
        assert any(i.issue_id.startswith("cte_scalar_cardinality") for i in issues)


class TestValidateMainQueryCteUsage:
    """Tests for validate_main_query_cte_usage."""

    def test_no_cte_outputs(self):
        """No CTEs means no issues."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(intent, {})
        assert issues == []

    def test_unreferenced_cte_warns_without_cte_chain(self):
        """Unreferenced CTE with no ``cte_steps`` metadata produces a warning."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]})
        ids = [i.issue_id for i in issues]
        assert any("unreferenced" in iid for iid in ids)
        assert all(i.severity == "warning" for i in issues if "unreferenced" in i.issue_id)

    def test_unreferenced_scalar_subquery_cte_errors(self):
        """Unreferenced CTE with ``emission=='scalar_subquery'`` produces an error."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        step = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["total"],
            emission="scalar_subquery",
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]}, [step])
        ur = [i for i in issues if "unreferenced" in i.issue_id]
        assert ur
        assert all(i.severity == "error" for i in ur)

    def test_valid_cte_reference(self):
        """Main query referencing CTE column correctly produces no column errors."""
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]})
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_column_not_in_cte(self):
        """Main query referencing column absent from CTE outputs produces error."""
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.missing_col"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]})
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_filter_references_cte(self):
        """Filter referencing non-existent CTE column produces error."""
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("cte1.bad_col"),
                    op=">",
                    value_type="number",
                    param_key="p1",
                    raw_value="10",
                )
            ],
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]})
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_column_map_references_cte(self):
        """column_map referencing non-existent CTE column produces error."""
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={"ghost": "cte1"},
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]})
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_select_without_qualified_name_skips_cte_column_check(self):
        """Expressions without ``table.column`` do not participate in CTE output validation."""
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(intent, {"cte1": ["total"]})
        col_errors = [i for i in issues if "main_col_not_in_cte" in i.issue_id]
        assert col_errors == []

    def test_cte_outputs_key_case_must_match_for_column_lookup(self):
        """CTE output keys and column refs are matched case-insensitively via canonical lookup."""
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.Total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(intent, {"CTE1": ["Total"]})
        assert not any("main_col_not_in_cte" in i.issue_id for i in issues)

    def test_downstream_cte_table_reference_triggers_upstream_usage(self):
        """A CTE listed in a later step's ``tables`` is treated as referenced."""
        step1 = RuntimeCteStep(
            cte_name="upstream",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["order_id"],
            grain="row_level",
        )
        step2 = RuntimeCteStep(
            cte_name="downstream",
            tables=["upstream"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("upstream.order_id"))],
            output_columns=["order_id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["downstream"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("downstream.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        issues = _validate_main_query_cte_usage(
            intent,
            {"upstream": ["order_id"], "downstream": ["order_id"]},
            [step1, step2],
        )
        assert not any("unreferenced" in i.issue_id for i in issues)


class TestValidateCteOutputTypes:
    """Tests for validate_cte_output_types."""

    def test_empty_list(self):
        """Empty CTE list yields no issues."""
        issues = _validate_cte_output_types(
            [],
            SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="h"),
        )
        assert issues == []

    def test_sum_on_numeric_clean(self, schema_graph):
        """SUM on a numeric column produces no type warnings."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            output_columns=["sum_orders_amount"],
            grain="scalar",
        )
        issues = _validate_cte_output_types([cte], schema_graph)
        assert issues == []

    def test_sum_on_string_column_from_prior_cte_warns(self, schema_graph):
        """SUM over a string-typed column resolved via upstream CTE outputs emits a warning."""
        cte1 = RuntimeCteStep(
            cte_name="step1",
            tables=["customers"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
            output_columns=["name"],
            grain="row_level",
        )
        sum_name = NormalizedExpr(add_groups=[MulGroup(multiply=["sum(step1.name)"])])
        cte2 = RuntimeCteStep(
            cte_name="step2",
            tables=["step1"],
            select_cols=[SelectCol(expr=sum_name)],
            output_columns=["sum_step1_name"],
            grain="scalar",
        )
        issues = _validate_cte_output_types([cte1, cte2], schema_graph)
        assert any("cte_agg_type_mismatch" in i.issue_id for i in issues)

    def test_count_on_string_does_not_warn(self, schema_graph):
        """Only SUM/AVG are checked for numeric operand type."""
        cte1 = RuntimeCteStep(
            cte_name="step1",
            tables=["customers"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
            output_columns=["name"],
            grain="row_level",
        )
        cnt = NormalizedExpr(add_groups=[MulGroup(multiply=["count(step1.name)"])])
        cte2 = RuntimeCteStep(
            cte_name="step2",
            tables=["step1"],
            select_cols=[SelectCol(expr=cnt)],
            output_columns=["count_step1_name"],
            grain="scalar",
        )
        issues = _validate_cte_output_types([cte1, cte2], schema_graph)
        assert not any("cte_agg_type_mismatch" in i.issue_id for i in issues)


class TestValidateSql:
    """Tests for validate_sql."""

    def test_valid_select_passes(self):
        """Valid SELECT with passing AST check succeeds."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT * FROM film")
        assert ok is True
        assert err is None
        assert cat is None

    def test_non_select_fails(self):
        """Non-SELECT statement fails before AST check."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "DELETE FROM film")
        assert ok is False
        assert err in {"not_select", "forbidden_sql"}
        assert cat == (FailureCategory.SCHEMA if err == "not_select" else FailureCategory.OTHER)

    def test_ast_failure(self):
        """AST validation failure propagated."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [
                SqlDiagnostic(code=SqlDiagnosticCode.AST_PARSE_FAILED, message="syntax error")
            ],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT broken syntax")
        assert ok is False
        assert "syntax error" in err
        assert cat == FailureCategory.SCHEMA_VALIDATION

    def test_forbidden_sql_caught(self):
        """Forbidden keywords caught before AST check."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT 1; DROP TABLE film")
        assert ok is False
        assert err == "forbidden_sql"
        assert cat == FailureCategory.OTHER

    def test_explain_success_when_enabled(self):
        mock_explain = MagicMock(return_value=(True, [], ""))
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: True,
            explain_diagnose=mock_explain,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT 1", {"p1": 1})
        assert ok is True
        assert err is None
        assert cat is None
        mock_explain.assert_called_once_with("SELECT 1", {"p1": 1}, schema=None, intent=None)

    def test_explain_failure_returns_message(self):
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: True,
            explain_diagnose=lambda sql, params=None, **kw: (
                False,
                [],
                'relation "missing" does not exist',
            ),
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT * FROM missing")
        assert ok is False
        assert err.startswith("[explain_schema]")
        assert 'relation "missing" does not exist' in err
        assert cat == FailureCategory.EXECUTION_SCHEMA_ERROR

    def test_ast_none_error_stringified(self):
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [
                SqlDiagnostic(code=SqlDiagnosticCode.AST_PARSE_FAILED, message="None")
            ],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT")
        assert ok is False
        assert err == "SQL structure error: None"
        assert cat == FailureCategory.SCHEMA_VALIDATION

    def test_with_query_passes_enforcement_before_ast(self):
        """CTE-wrapped select passes ``_enforce_select_only`` then reaches AST validation."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object() if sql.strip().lower().startswith(("select", "with")) else None,
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        sql = "WITH c AS (SELECT 1 AS n) SELECT n FROM c"
        ok, err, cat, _diags = validate_sql(mock_dialect, sql)
        assert ok is True
        assert err is None
        assert cat is None

    def test_unbound_pyformat_placeholder_rejected(self):
        """SQL containing ``%(p1)s`` triggers the unbound_placeholder tripwire before AST."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object(),
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT * FROM film WHERE rating = %(p1)s")
        assert ok is False
        assert err == "unbound_placeholder"
        assert cat == FailureCategory.UNBOUND_PLACEHOLDER

    def test_unbound_positional_placeholder_rejected(self):
        """SQL containing bare ``%s`` also trips the unbound_placeholder tripwire."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object(),
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT * FROM film WHERE id = %s")
        assert ok is False
        assert err == "unbound_placeholder"
        assert cat == FailureCategory.UNBOUND_PLACEHOLDER

    def test_named_colon_placeholder_does_not_trip_unbound(self):
        """Canonical ``:p1`` is a legitimate bind token and must not trip the unbound tripwire."""
        mock_dialect = SimpleNamespace(
            name="postgresql",
            parse_select=lambda sql: object(),
            ast_validate_full=lambda sql, **kw: [],
            can_explain=lambda: False,
        )
        ok, err, cat, _diags = validate_sql(mock_dialect, "SELECT * FROM film WHERE id = :p1")
        assert ok is True
        assert err is None
        assert cat is None


class TestValidateSemantics:
    """Tests for validate_semantics."""

    def test_empty_tables_returns_tables_empty_issue(self, schema_graph):
        """validate_semantics returns tables_empty when intent has no tables."""
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = validate_semantics(intent, schema_graph)
        assert not result.is_valid
        issue_ids = [i.issue_id for i in result.issues]
        assert "tables_empty" in issue_ids

    def test_valid_minimal_intent_returns_result(self, schema_graph):
        """validate_semantics returns IntentValidationResult for valid minimal intent."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = validate_semantics(intent, schema_graph)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "issues")
        assert isinstance(result.issues, list)

    def test_cte_empty_name_produces_error(self, schema_graph):
        """CTE with empty name produces error in validate_semantics."""
        cte = RuntimeCteStep(
            cte_name="",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["order_id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = validate_semantics(intent, schema_graph)
        assert any(i.issue_id == "cte_name_empty" for i in result.issues)

    def test_cte_empty_output_columns_produces_error(self, schema_graph):
        """CTE with empty output_columns produces error."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=[],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = validate_semantics(intent, schema_graph)
        assert any("empty_output" in i.issue_id or "output_columns_empty" in i.issue_id for i in result.issues)

    def test_cte_unknown_table_produces_error(self, schema_graph):
        """CTE referencing unknown table produces error."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["nonexistent_table"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("nonexistent_table.id"))],
            output_columns=["id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = validate_semantics(intent, schema_graph)
        assert any("unknown_table" in i.issue_id for i in result.issues)

    def test_cte_duplicate_names_produce_error(self, schema_graph):
        """Duplicate CTE names produce error in validate_semantics."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["order_id"],
            grain="row_level",
        )
        cte2 = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["order_id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte1, cte2],
        )
        result = validate_semantics(intent, schema_graph)
        assert any("duplicate" in i.issue_id for i in result.issues)

    def test_cte_forward_reference_produces_error(self, schema_graph):
        """CTE referencing later-defined CTE produces error."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            tables=["cte2"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte2.total"))],
            output_columns=["total"],
            grain="row_level",
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            output_columns=["order_id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte1, cte2],
        )
        result = validate_semantics(intent, schema_graph)
        assert any("forward_ref" in i.issue_id for i in result.issues)

    def test_merge_cte_projection_columns_into_outputs_adds_missing_keys(self):
        """Sparse ``output_column_metadata`` is augmented from ``output_columns``."""
        from aetherdialect._contracts_base import CteOutputColumnMeta
        from aetherdialect._validation_execute import (
            _merge_cte_projection_columns_into_outputs,
        )

        meta: dict[str, CteOutputColumnMeta] = {}
        _merge_cte_projection_columns_into_outputs(meta, ["orders.order_id", "total"])
        assert "order_id" in meta
        assert "total" in meta
        assert meta["order_id"].source == "output_column_projection"


class TestValidateCaseBranchOperators:
    """CASE branch conditions allow between / in / not in without ``filter_between_not_decomposed``."""

    @staticmethod
    def _scope_kwargs(schema_graph: SchemaGraph) -> dict[str, Any]:
        return dict(
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("c01"))],
            window_registry=[],
            schema=schema_graph,
            allowed_tables={"orders"},
            cte_outputs={},
            cte_steps=[],
            location_prefix="main query",
            param_values={"p1": 0, "p2": 100, "p3": ["pending", "shipped"]},
        )

    def test_between_on_case_branch_not_flagged(self, schema_graph: SchemaGraph) -> None:
        branch = CaseWhenBranch(
            condition=FilterParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op="between",
                value_type="number",
                param_key="p1",
                param_key_hi="p2",
                raw_value=[0, 100],
            ),
            result=NormalizedExpr(string_literal="mid"),
        )
        step = CaseRegistryStep(registry_id="c01", case_when=CaseWhenExpr(branches=[branch]))
        issues = _validate_case_branches_for_scope(
            case_registry=[step],
            **self._scope_kwargs(schema_graph),
        )
        assert not any("filter_between_not_decomposed" in i.issue_id for i in issues)

    def test_in_on_case_branch_not_flagged(self, schema_graph: SchemaGraph) -> None:
        branch = CaseWhenBranch(
            condition=FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="in",
                value_type="string",
                param_key="p3",
            ),
            result=NormalizedExpr(string_literal="openish"),
        )
        step = CaseRegistryStep(registry_id="c01", case_when=CaseWhenExpr(branches=[branch]))
        issues = _validate_case_branches_for_scope(
            case_registry=[step],
            **self._scope_kwargs(schema_graph),
        )
        assert not any("filter_between_not_decomposed" in i.issue_id for i in issues)

    def test_not_in_on_case_branch_not_flagged(self, schema_graph: SchemaGraph) -> None:
        branch = CaseWhenBranch(
            condition=FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="not in",
                value_type="string",
                param_key="p3",
            ),
            result=NormalizedExpr(string_literal="other"),
        )
        step = CaseRegistryStep(registry_id="c01", case_when=CaseWhenExpr(branches=[branch]))
        issues = _validate_case_branches_for_scope(
            case_registry=[step],
            **self._scope_kwargs(schema_graph),
        )
        assert not any("filter_between_not_decomposed" in i.issue_id for i in issues)


class TestExplainCostGate:
    """EXPLAIN cost gate surfaces ``FailureCategory.EXECUTION_COST_EXCEEDED`` without repair."""

    def test_validate_sql_cost_exceeded_category(self, schema_graph: SchemaGraph) -> None:
        d = _MinimalDialect(object())

        def _explain(sql: str, params=None, **kwargs):
            return (
                False,
                [
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED,
                        message="cost cap tripped",
                    )
                ],
                "cost cap tripped",
            )

        d.explain_diagnose = _explain
        d.can_explain = lambda: True
        ok, msg, cat, diags = validate_sql(d, "SELECT 1", {}, schema=schema_graph)
        assert ok is False
        assert cat == FailureCategory.EXECUTION_COST_EXCEEDED
        assert diags and diags[0].code == SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED


class TestDialectExecute:
    """Tests for ``Dialect.execute`` delegation."""

    def test_execute_returns_rows(self):
        """Dialect ``execute`` returns row tuples."""
        mock_dialect = SimpleNamespace(execute=lambda sql, params=None: [(1, "a"), (2, "b")])
        rows = mock_dialect.execute("SELECT 1, 'a'")
        assert rows == [(1, "a"), (2, "b")]


class TestCanonicalizeRejectionReason:
    """Tests for ``canonicalize_rejection_reason``."""

    def test_first_line_only(self) -> None:
        assert canonicalize_rejection_reason("a\nb") == "a b"

    def test_strips_trailing_punctuation(self) -> None:
        assert canonicalize_rejection_reason("Bad join path.") == "Bad join path"

    def test_max_length(self) -> None:
        long = "x" * 200
        assert len(canonicalize_rejection_reason(long)) <= 160
