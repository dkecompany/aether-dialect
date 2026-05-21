"""Tests for sql_gen module: join topology, rendering, normalization."""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import (
    JOIN_CHOICE_SCOPE_MAIN,
)
from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    LlmJsonExhausted,
    SchemaGraph,
    TableMetadata,
    VirtualColumnSpec,
    VirtualTableSpec,
)
from aetherdialect._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ExprValue,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    WindowRegistryStep,
    WindowSpec,
    registry_render_scope,
)
from aetherdialect._core_utils import (
    _format_scalar_for_structural_sql_inline,
    reduce_structural_sql_placeholders,
)
from aetherdialect._dialect import (
    DatabricksDialect,
    PostgresDialect,
    finalize_executable_sql,
)
from aetherdialect._sql_gen import (
    JOIN_PRIOR_FEEDBACK_HEADING,
    ScopeClass,
    _analyze_join_topology,
    _build_deterministic_select_block,
    _candidate_join_paths_for_tables,
    _join_choice_payload_valid_final,
    _join_clause_from_signature,
    _join_clause_parts_with_bool_op,
    _join_kind_for_edge,
    _join_path_signature_for_path,
    _maybe_render_array_unnest_select,
    _orient_join_sig_for_from,
    _parse_join_choice_payload,
    _render_case_branch_sql,
    _render_case_when_sql,
    _render_group_sql,
    _render_window_over_sql,
    _serialize_join_candidate_row,
    _try_ast_inject_joins,
    _valid_cte_join_candidate_ids,
    _valid_main_join_candidate_ids,
    _wrap_for_case_insensitive,
    build_deterministic_sql,
    build_join_choice_prompt,
    classify_scope_candidates,
    cte_to_intent_for_ranking,
    enumerate_join_paths_base,
    generate_col_alias,
    get_join_choice_from_llm,
    inject_join_into_deterministic_sql,
    join_candidate_map,
    join_choice_scope_key_cte,
    join_hints_multi,
    merge_join_hints_for_na_scopes,
    physical_tables_for_join_hints,
    render_expr_sql,
    render_select_col_sql,
    select_col_prefers_llm_display_alias,
    tables_in_join_scope,
)


def _pg_render() -> PostgresDialect:
    """Uninitialized Postgres dialect for deterministic SQL rendering tests."""

    return PostgresDialect.__new__(PostgresDialect)


def _dbr_render() -> DatabricksDialect:
    """Uninitialized Databricks dialect for Spark-style ``OVER`` rendering tests."""

    return DatabricksDialect.__new__(DatabricksDialect)


class TestPhysicalTablesForJoinHints:
    """Tests for physical_tables_for_join_hints."""

    def test_filters_cte_and_unknown_names(self):
        """Keeps only names that exist as schema table keys."""
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                ),
                "payment": TableMetadata(
                    name="payment",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        got = physical_tables_for_join_hints(
            ["cte2", "customer", "payment", "cte1", "ghost"],
            schema,
        )
        assert got == ["customer", "payment"]

    def test_case_insensitive_match(self):
        """Resolves intent casing to canonical schema keys."""
        schema = SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        assert physical_tables_for_join_hints(["Film"], schema) == ["film"]

    def test_empty_input(self):
        """None or empty tables yields empty list."""
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="h")
        assert physical_tables_for_join_hints(None, schema) == []
        assert physical_tables_for_join_hints([], schema) == []

    def test_duplicate_physical_for_self_join(self):
        """Same physical table listed twice stays twice for join-hint enumeration."""
        schema = SchemaGraph(
            tables={
                "employee": TableMetadata(
                    name="employee",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        assert physical_tables_for_join_hints(["employee", "employee"], schema) == [
            "employee",
            "employee",
        ]


class TestJoinCandidateMap:
    """Tests for join_candidate_map."""

    def test_builds_map(self):
        """join_candidate_map builds candidate_id -> signature map."""
        hints = {
            "candidates": [
                {
                    "candidate_id": "J01",
                    "join_path_signature": ["orders.customer_id->customers.customer_id"],
                },
                {
                    "candidate_id": "J02",
                    "join_path_signature": ["orders.product_id->products.product_id"],
                },
            ]
        }
        result = join_candidate_map(hints)
        assert "J01" in result
        assert "J02" in result
        assert result["J01"] == ["orders.customer_id->customers.customer_id"]

    def test_empty_candidates(self):
        """join_candidate_map returns empty for no candidates."""
        assert join_candidate_map({"candidates": []}) == {}

    def test_missing_candidates_key(self):
        """join_candidate_map returns empty for missing key."""
        assert join_candidate_map({}) == {}


class TestAnalyzeJoinTopology:
    """Tests for analyze_join_topology."""

    def test_linear_topology(self):
        """analyze_join_topology detects linear chain."""
        sig = [
            "orders.customer_id->customers.customer_id",
        ]
        topo, anchor, leaves = _analyze_join_topology(sig)
        assert topo == "linear"
        assert len(leaves) == 2

    def test_star_topology(self):
        """analyze_join_topology detects star pattern with 3+ branches."""
        sig = [
            "orders.customer_id->customers.customer_id",
            "orders.product_id->products.product_id",
            "orders.store_id->stores.store_id",
        ]
        topo, anchor, leaves = _analyze_join_topology(sig)
        assert topo == "star"
        assert anchor == "orders"
        assert sorted(leaves) == ["customers", "products", "stores"]

    def test_empty_sig(self):
        """analyze_join_topology returns none for empty sig."""
        topo, anchor, leaves = _analyze_join_topology([])
        assert topo == "none"
        assert anchor == ""
        assert leaves == []


class TestRenderGroupSql:
    """Tests for _render_group_sql."""

    def test_bare_column(self):
        """_render_group_sql renders bare column."""
        g = MulGroup(multiply=["orders.amount"])
        assert _render_group_sql(g) == "orders.amount"

    def test_with_agg_func(self):
        """_render_group_sql renders with aggregation."""
        g = MulGroup(multiply=["orders.amount"], agg_func="sum")
        result = _render_group_sql(g)
        assert result == "SUM(orders.amount)"

    def test_with_coefficient(self):
        """_render_group_sql renders coefficient."""
        g = MulGroup(coefficient=2.0, multiply=["t.x"])
        result = _render_group_sql(g)
        assert "2.0" in result
        assert "t.x" in result

    def test_with_divide(self):
        """_render_group_sql renders division."""
        g = MulGroup(multiply=["t.a"], divide=["t.b"])
        result = _render_group_sql(g)
        assert "t.a" in result
        assert "t.b" in result

    def test_with_scalar_func(self):
        """_render_group_sql wraps with scalar function."""
        g = MulGroup(multiply=["t.x"], scalar_func="abs")
        result = _render_group_sql(g)
        assert result == "ABS(t.x)"

    def test_with_inner_scalar_func(self):
        """_render_group_sql wraps with inner scalar function."""
        g = MulGroup(multiply=["t.x"], inner_scalar_func="abs", agg_func="sum")
        result = _render_group_sql(g)
        assert "ABS(t.x)" in result
        assert "SUM" in result

    def test_empty_multiply(self):
        """_render_group_sql with empty multiply returns '1'."""
        g = MulGroup(multiply=[])
        assert _render_group_sql(g) == "1"

    def test_coeff_param_key(self):
        """_render_group_sql uses coeff_param_key."""
        g = MulGroup(multiply=["t.x"], coeff_param_key="s1")
        result = _render_group_sql(g)
        assert ":s1" in result

    def test_concat_two_columns_comma_joined(self):
        """CONCAT MulGroup joins multiply terms with commas inside CONCAT."""
        g = MulGroup(
            multiply=[
                NormalizedExpr.from_column("tbl_a.first_name"),
                NormalizedExpr.from_column("tbl_a.last_name"),
            ],
            scalar_func="concat",
        )
        result = _render_group_sql(g)
        assert "CONCAT" in result.upper()
        assert " * " not in result
        assert "tbl_a.first_name" in result
        assert "tbl_a.last_name" in result
        assert "," in result

    def test_concat_with_literal_between_columns(self):
        """CONCAT includes literal multiply term."""
        g = MulGroup(
            multiply=[
                NormalizedExpr.from_column("tbl_a.first_name"),
                NormalizedExpr(raw_sql="' '"),
                NormalizedExpr.from_column("tbl_a.last_name"),
            ],
            scalar_func="concat",
        )
        result = _render_group_sql(g)
        assert "CONCAT" in result.upper()
        assert "'" in result or '"' in result


class TestRenderExprSql:
    """Tests for render_expr_sql."""

    def test_single_column(self):
        """render_expr_sql renders single column expression."""
        expr = NormalizedExpr.from_column("t.x")
        assert render_expr_sql(expr) == "t.x"

    def test_addition(self):
        """render_expr_sql renders addition."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"]), MulGroup(multiply=["t.b"])],
        )
        result = render_expr_sql(expr)
        assert "+" in result

    def test_subtraction(self):
        """render_expr_sql renders subtraction."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"])],
            sub_groups=[MulGroup(multiply=["t.b"])],
        )
        result = render_expr_sql(expr)
        assert "-" in result


class TestExprGuidePlaceholders:
    """Expression guide strings use ``:param`` for keyed args and bare ``:`` for positional-only slots."""

    def test_scalar_func_keyed_args(self):
        g = MulGroup(
            multiply=["t.x"],
            scalar_func="round",
            scalar_func_args=[2],
            sarg_param_keys=["s1"],
        )
        out = _render_group_sql(g)
        assert ":s1" in out


class TestRenderExprSqlDialectQuoting:
    """Qualified refs in rendered SQL respect dialect quoting when a dialect is passed."""

    def test_postgres_quotes_reserved_column(self):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["orders.order"])])
        out = render_expr_sql(expr, _pg_render())
        assert '"orders"."order"' in out


class TestJoinPathSignatureForPath:
    """Tests for join_path_signature_for_path."""

    def test_single_edge(self, simple_schema):
        """Catalog FK edges use declared ``src_table`` to ``dst_table`` orientation."""
        path = [
            {
                "src_table": "orders",
                "src_cols": ["customer_id"],
                "dst_table": "customers",
                "dst_cols": ["id"],
            }
        ]
        result = _join_path_signature_for_path(path, simple_schema)
        assert result == ["orders.customer_id->customers.id"]

    def test_multiple_edges(self):
        """join_path_signature_for_path generates signature for multiple edges."""
        path = [
            {
                "src_table": "orders",
                "src_cols": ["customer_id"],
                "dst_table": "customers",
                "dst_cols": ["customer_id"],
            },
            {
                "src_table": "orders",
                "src_cols": ["product_id"],
                "dst_table": "products",
                "dst_cols": ["product_id"],
            },
        ]
        result = _join_path_signature_for_path(path)
        assert len(result) == 2

    def test_empty_path(self):
        """join_path_signature_for_path returns empty for empty path."""
        assert _join_path_signature_for_path([]) == []

    def test_multi_column_key(self):
        """join_path_signature_for_path handles composite keys."""
        path = [
            {
                "src_table": "t1",
                "src_cols": ["a", "b"],
                "dst_table": "t2",
                "dst_cols": ["c", "d"],
            }
        ]
        result = _join_path_signature_for_path(path)
        assert result == ["t1.a,b->t2.c,d"]


class TestCteToIntentForRanking:
    """Tests for cte_to_intent_for_ranking."""

    def test_basic_conversion(self):
        """cte_to_intent_for_ranking creates RuntimeIntent from CTE."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1", "t2"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[NormalizedExpr.from_column("t1.name")],
            output_columns=["id"],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("t1.name"))],
            filters_param=[FilterParam(left_expr=NormalizedExpr.from_column("t1.status"), op="=")],
        )
        intent = cte_to_intent_for_ranking(cte)
        assert intent.tables == ["t1", "t2"]
        assert intent.grain == "grouped"
        assert len(intent.select_cols) == 1
        assert len(intent.group_by_cols) == 1
        assert len(intent.filters_param) == 1
        assert intent.cte_steps == []

    def test_preserves_limit(self):
        """cte_to_intent_for_ranking preserves limit."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            limit=10,
        )
        intent = cte_to_intent_for_ranking(cte)
        assert intent.limit == 10

    def test_preserves_column_map(self):
        """cte_to_intent_for_ranking preserves column_map."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            column_map={"name": "t1"},
        )
        intent = cte_to_intent_for_ranking(cte)
        assert intent.column_map == {"name": "t1"}


class TestInjectJoinCanonicalStarOrder:
    """Star/tree join signatures are lexically ordered at inject time."""

    def test_star_sig_sorted_before_join_clause(self):
        det = "SELECT 1 FROM hub WHERE 1=1"
        sigs = [["zebra.z1->hub.h1", "apple.a1->hub.h1"]]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det, sigs, None, dialect=dialect)
        assert "apple" in out and "zebra" in out
        assert out.lower().index("apple") < out.lower().index("zebra")


class TestAnalyzeJoinTopologyEdgeCases:
    """Edge-case tests for analyze_join_topology."""

    def test_three_table_linear_chain(self):
        """analyze_join_topology detects 3-table linear chain."""
        sig = [
            "a.col->b.col",
            "b.col->c.col",
        ]
        topo, anchor, leaves = _analyze_join_topology(sig)
        assert topo == "linear"
        assert sorted(leaves) == ["a", "c"]

    def test_tree_topology(self):
        """analyze_join_topology detects tree pattern."""
        sig = [
            "a.col->b.col",
            "b.col->c.col",
            "b.col->d.col",
        ]
        topo, anchor, leaves = _analyze_join_topology(sig)
        assert topo in ("star", "tree")

    def test_single_edge(self):
        """analyze_join_topology handles single edge."""
        sig = ["orders.cid->customers.id"]
        topo, anchor, leaves = _analyze_join_topology(sig)
        assert len(leaves) == 2

    def test_no_arrow_items(self):
        """analyze_join_topology returns none for items without arrows."""
        sig = ["no_arrow"]
        topo, anchor, leaves = _analyze_join_topology(sig)
        assert topo == "none"


class TestJoinCandidateMapEdgeCases:
    """Edge-case tests for join_candidate_map."""

    def test_invalid_types_skipped(self):
        """join_candidate_map skips entries with invalid types."""
        hints = {
            "candidates": [
                {"candidate_id": 123, "join_path_signature": ["a.col->b.col"]},
                {"candidate_id": "J01", "join_path_signature": "not_a_list"},
            ]
        }
        result = join_candidate_map(hints)
        assert len(result) == 0

    def test_multiple_candidates(self):
        """join_candidate_map handles many candidates."""
        candidates = [
            {
                "candidate_id": f"J{i:02d}",
                "join_path_signature": [f"t{i}.col->t{i + 1}.col"],
            }
            for i in range(1, 6)
        ]
        result = join_candidate_map({"candidates": candidates})
        assert len(result) == 5

    def test_empty_signature_included(self):
        """join_candidate_map includes candidate with empty join_path_signature."""
        hints = {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]}
        result = join_candidate_map(hints)
        assert result == {"J00": []}


class TestRenderGroupSqlEdgeCases:
    """Edge-case tests for _render_group_sql."""

    def test_multiply_divide_combined(self):
        """_render_group_sql renders multiply and divide together."""
        g = MulGroup(multiply=["t.revenue"], divide=["t.count"])
        result = _render_group_sql(g)
        assert "t.revenue" in result
        assert "t.count" in result
        assert "/" in result

    def test_coefficient_and_agg(self):
        """_render_group_sql renders coefficient with aggregation."""
        g = MulGroup(coefficient=1.5, multiply=["t.x"], agg_func="sum")
        result = _render_group_sql(g)
        assert "1.5" in result
        assert "SUM" in result

    def test_multiple_multiply_terms(self):
        """_render_group_sql joins multiple multiply terms."""
        g = MulGroup(multiply=["t.a", "t.b"])
        result = _render_group_sql(g)
        assert "t.a * t.b" in result


class TestRenderExprSqlEdgeCases:
    """Edge-case tests for render_expr_sql."""

    def test_with_agg_func(self):
        """render_expr_sql renders agg_func on expr level."""
        expr = NormalizedExpr.from_agg("count", "t.id")
        result = render_expr_sql(expr)
        assert "COUNT" in result
        assert "t.id" in result

    def test_with_scalar_func(self):
        """render_expr_sql renders scalar_func on expr level."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.x"])],
            scalar_func="round",
            scalar_func_args=[2],
        )
        result = render_expr_sql(expr)
        assert "ROUND" in result

    def test_with_add_values(self):
        """render_expr_sql renders add_values."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.x"])],
            add_values=[ExprValue(value=10, param_key="p1")],
        )
        result = render_expr_sql(expr)
        assert ":p1" in result

    def test_with_sub_values(self):
        """render_expr_sql renders sub_values."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.x"])],
            sub_values=[ExprValue(value=5, param_key="p2")],
        )
        result = render_expr_sql(expr)
        assert ":p2" in result

    def test_empty_groups(self):
        """render_expr_sql renders 0 for empty groups."""
        expr = NormalizedExpr()
        result = render_expr_sql(expr)
        assert "0" in result


class TestCandidateJoinPathsForTables:
    """Tests for candidate_join_paths_for_tables."""

    def test_single_table_returns_empty_path(self):
        """Single table returns [[]] (no joins needed)."""
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        result = _candidate_join_paths_for_tables(schema, ["orders"])
        assert result == [[]]

    def test_two_tables_with_path(self):
        """Two tables with join path returns candidate edges."""
        edge = {
            "src_table": "orders",
            "src_cols": ["customer_id"],
            "dst_table": "customers",
            "dst_cols": ["id"],
        }
        schema = SchemaGraph(
            tables={
                "orders": TableMetadata(name="orders", columns={}, foreign_keys=[], primary_key=""),
                "customers": TableMetadata(name="customers", columns={}, foreign_keys=[], primary_key=""),
            },
            join_paths_multi={
                "orders": {"customers": [[edge]]},
                "customers": {
                    "orders": [
                        [
                            {
                                "src_table": "customers",
                                "src_cols": ["id"],
                                "dst_table": "orders",
                                "dst_cols": ["customer_id"],
                            }
                        ]
                    ]
                },
            },
            effective_structural_hash="",
        )
        result = _candidate_join_paths_for_tables(schema, ["orders", "customers"])
        assert len(result) >= 1
        assert any(len(path) > 0 for path in result)

    def test_empty_tables_list(self):
        """Empty tables list returns [[]]."""
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        result = _candidate_join_paths_for_tables(schema, [])
        assert result == [[]]

    def test_deduplicates_tables(self):
        """Duplicate tables are deduplicated."""
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        result = _candidate_join_paths_for_tables(schema, ["orders", "orders"])
        assert result == [[]]


class TestJoinHintsMulti:
    """Tests for join_hints_multi."""

    def test_single_table_returns_j00(self):
        """Single table returns J00 with empty signature."""
        schema = SchemaGraph(
            tables={"orders": TableMetadata(name="orders", columns={}, foreign_keys=[], primary_key="order_id")},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        result = join_hints_multi(schema, ["orders"])
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["candidate_id"] == "J00"
        assert result["candidates"][0]["edge_count"] == 0
        assert result["candidates"][0]["edge_kinds"] == []
        assert result["candidates"][0]["candidate_tier"] == "base"

    @patch(
        "aetherdialect._sql_gen._candidate_join_paths_for_tables",
        return_value=[
            [
                {
                    "src_table": "orders",
                    "dst_table": "customers",
                    "src_cols": ["cid"],
                    "dst_cols": ["cid"],
                }
            ]
        ],
    )
    def test_multi_table_includes_edge_kinds(self, _mock_candidates):
        """Multi-table candidates carry stable ``edge_kinds`` and ``candidate_tier``."""
        schema = SchemaGraph(
            tables={
                "orders": TableMetadata(name="orders", columns={}, foreign_keys=[], primary_key="order_id"),
                "customers": TableMetadata(
                    name="customers",
                    columns={},
                    foreign_keys=[],
                    primary_key="customer_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = join_hints_multi(schema, ["orders", "customers"], intent)
        assert result["candidates"][0]["candidate_id"] == "J01"
        assert result["candidates"][0]["edge_count"] == 1
        assert result["candidates"][0]["edge_kinds"] == ["catalog_fk"]
        assert result["candidates"][0]["candidate_tier"] == "base"

    def test_tables_in_join_scope_keeps_virtual(self):
        """Virtual CTE names listed in intent are retained for join enumeration."""
        schema = SchemaGraph(
            tables={"film": TableMetadata(name="film", columns={}, foreign_keys=[], primary_key="id")},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        virt = {
            "q": VirtualTableSpec(
                cte_name="q",
                columns={
                    "id": VirtualColumnSpec("film", "id", True, None, []),
                },
            )
        }
        assert tables_in_join_scope(["film", "q"], schema, virt) == ["film", "q"]


class TestBuildDeterministicSql:
    """Unit tests for build_deterministic_sql from RuntimeIntent."""

    def test_from_anchor_follows_join_signature_driver(self):
        """When a join path is supplied, FROM matches the first segment driver before row-count tie-break."""
        schema = SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                    row_count=1000,
                ),
                "language": TableMetadata(
                    name="language",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                    row_count=10,
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        intent = RuntimeIntent(
            tables=["film", "language"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["film.title"])], sub_groups=[])),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        sql_heuristic = build_deterministic_sql(intent, None, schema, _pg_render())
        sql_driver = build_deterministic_sql(
            intent,
            None,
            schema,
            _pg_render(),
            join_signature_for_from_anchor=["film.language_id->language.language_id"],
        )
        assert 'FROM "language"' in sql_heuristic
        assert 'FROM "film"' in sql_driver

    def test_row_level_select_from_single_table(self):
        """Row-level intent produces SELECT, FROM, no GROUP BY."""
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.a"])], sub_groups=[])),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "SELECT" in sql
        assert 'FROM "t1"' in sql
        assert "GROUP BY" not in sql
        assert "WHERE" not in sql

    def test_grouped_has_group_by_and_having(self):
        """Grouped intent produces GROUP BY and HAVING when present."""
        intent = RuntimeIntent(
            tables=["t1"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.x"])], sub_groups=[])),
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["t1.id"], agg_func="count")],
                        sub_groups=[],
                    )
                ),
            ],
            group_by_cols=[NormalizedExpr(add_groups=[MulGroup(multiply=["t1.x"])], sub_groups=[])],
            order_by_cols=[],
            filters_param=[],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["t1.id"], agg_func="count")],
                        sub_groups=[],
                    ),
                    op=">",
                    param_key="h1",
                ),
            ],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "GROUP BY" in sql
        assert "HAVING" in sql
        assert '"t1"."x"' in sql or "t1.x" in sql

    def test_date_window_filter_renders_range(self):
        """date_window filter with start/end renders two predicates."""
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.d"])], sub_groups=[])),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.d"])], sub_groups=[]),
                    value_type="date_window",
                    raw_value={"start": "2020-01-01", "end": "2020-12-31"},
                ),
            ],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert ">=" in sql and "2020-01-01" in sql
        assert "<=" in sql and "2020-12-31" in sql

    def test_date_window_emits_two_sided_predicate(self):
        """Relative date_window emits lower bound and inclusive upper anchor."""
        intent = RuntimeIntent(
            tables=["payment"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["payment.payment_id"])],
                        sub_groups=[],
                    )
                ),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["payment.payment_date"])],
                        sub_groups=[],
                    ),
                    value_type="date_window",
                    raw_value={"unit": "day", "amount": 30},
                ),
            ],
            having_param=[],
        )
        sql_pg = build_deterministic_sql(intent, dialect=_pg_render())
        assert ">=" in sql_pg and "<=" in sql_pg
        assert "CURRENT_DATE" in sql_pg
        sql_db = build_deterministic_sql(intent, dialect=_dbr_render())
        low = sql_db.lower()
        assert ">=" in sql_db and "<=" in sql_db
        assert "current_date()" in low

    def test_cte_with_output_aliases(self):
        """CTE step is emitted with WITH and output column aliases."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.id"])], sub_groups=[])),
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["t1.amt"], agg_func="sum")],
                        sub_groups=[],
                    )
                ),
            ],
            output_columns=["id", "total_amt"],
            group_by_cols=[NormalizedExpr(add_groups=[MulGroup(multiply=["t1.id"])], sub_groups=[])],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            column_map={},
            output_column_metadata={},
            description="",
            limit=None,
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["cte1.id"])], sub_groups=[])),
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["cte1.total_amt"])],
                        sub_groups=[],
                    )
                ),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            cte_steps=[cte],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "WITH" in sql
        assert "cte1 AS" in sql
        assert "AS id" in sql or "AS total_amt" in sql
        assert 'FROM "cte1"' in sql

    def test_limit_rendered(self):
        """Intent with limit produces LIMIT clause."""
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.a"])], sub_groups=[])),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            limit=10,
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "LIMIT 10" in sql

    def test_order_by_with_direction(self):
        """ORDER BY clause includes expression and direction."""
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.a"])], sub_groups=[])),
            ],
            group_by_cols=[],
            order_by_cols=[
                OrderByCol(
                    expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.a"])], sub_groups=[]),
                    direction="desc",
                ),
            ],
            filters_param=[],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "ORDER BY" in sql
        assert "DESC" in sql

    def test_lag_renders_with_argument_and_over(self):
        """LAG uses argument expression and ORDER BY inside OVER."""
        ws = WindowSpec(
            function="lag",
            order_by=[
                OrderByCol(
                    expr=NormalizedExpr.from_column("payment.payment_date"),
                    direction="asc",
                )
            ],
            argument=NormalizedExpr.from_column("payment.amount"),
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        with registry_render_scope(wr, None):
            lag_sql = render_select_col_sql(sc)
        compact = lag_sql.replace(" ", "").replace('"', "")
        assert "LAG(payment.amount)" in compact
        assert "OVER" in lag_sql.upper()

    def test_row_number_window_in_select(self):
        """Window ranking function appears in SELECT with OVER clause."""
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("t1.store_id")],
            order_by=[
                OrderByCol(expr=NormalizedExpr.from_column("t1.rental_date"), direction="DESC"),
            ],
        )
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(column_ref="w01")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=ws,
                )
            ],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "ROW_NUMBER()" in sql.replace(" ", "")
        assert "OVER" in sql.upper()
        assert "PARTITION BY" in sql.upper()


class TestJoinClauseFromSignature:
    """Tests for _join_clause_from_signature."""

    def test_single_edge(self):
        """One segment produces one JOIN clause."""
        sig = ["a.fk->b.pk"]
        out = _join_clause_from_signature(sig)
        assert "JOIN b ON" in out
        assert "a.fk = b.pk" in out

    def test_multiple_edges(self):
        """Multiple segments produce multiple JOINs."""
        sig = ["a.x->b.x", "b.y->c.y"]
        out = _join_clause_from_signature(sig)
        assert "JOIN b ON" in out
        assert "JOIN c ON" in out
        assert "a.x = b.x" in out
        assert "b.y = c.y" in out

    def test_empty_returns_empty(self):
        """Empty signature returns empty string."""
        assert _join_clause_from_signature([]) == ""

    def test_multi_column_edge(self):
        """Segment with comma-separated columns produces AND terms."""
        sig = ["a.c1,c2->b.c1,c2"]
        out = _join_clause_from_signature(sig)
        assert "JOIN b ON" in out
        assert "a.c1 = b.c1" in out
        assert "a.c2 = b.c2" in out
        assert " AND " in out

    def test_skips_redundant_join_when_table_already_in_chain(self):
        """Do not emit a second JOIN to a table already present in the chain."""
        sig = ["language.lang_id->film.lang_id", "film.title_id->language.lang_id"]
        out = _join_clause_from_signature(sig, from_table="language", schema=None)
        assert out.count("JOIN film") == 1

    def test_frontier_from_rental_joins_category_before_category_dot_on(self):
        """Long paths from rental emit JOIN category before any bare category. column ON term."""
        sig = [
            "rental.inventory_id->inventory.inventory_id",
            "inventory.film_id->film.film_id",
            "film.film_id->film_category.film_id",
            "category.category_id->film_category.category_id",
        ]
        clause = _join_clause_from_signature(sig, "rental", schema=None)
        jc = clause.find("JOIN category")
        assert jc != -1
        cat_dot = re.search(r"(?<![\w_])category\.", clause)
        assert cat_dot is not None
        assert jc < cat_dot.start()

    def test_dimension_join_target_is_inner_when_fk_not_nullable(self):
        """Dimension role does not bias join kind; INNER is used unless FK source column is nullable."""
        schema = SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                    role="fact",
                ),
                "customers": TableMetadata(
                    name="customers",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                    role="dimension",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        sig = ["orders.cid->customers.id"]
        out = _join_clause_from_signature(sig, from_table="orders", schema=schema)
        assert "INNER JOIN" in out.upper()
        assert "LEFT JOIN" not in out.upper()
        assert "customers" in out

    def test_self_join_emits_second_alias(self):
        """Same-table edge from an existing row adds ``JOIN tbl AS tbl__sjN``."""
        sig = ["film.parent_film_id->film.film_id"]
        out = _join_clause_from_signature(sig, from_table="film", schema=None)
        assert "JOIN film AS film__sj2" in out
        assert "film__sj2" in out
        assert "film.parent_film_id" in out
        assert "film__sj2.film_id" in out


class TestBuildJoinAstFromSignature:
    """Dialect adapter ``attach_joins`` reproduces the string join clause."""

    def test_attach_joins_postgres_emits_equivalent_join(self):
        from aetherdialect._dialect import JoinEdge
        from aetherdialect._sql_gen import _join_edges_from_signature

        sig = ["orders.cid->customers.id"]
        from_t = "orders"
        result = _join_edges_from_signature(sig, [""], from_t, None)
        assert result is not None
        edges, where_edges, extra_from = result
        assert edges and isinstance(edges[0], JoinEdge)
        assert where_edges == []
        assert extra_from == []
        pg = PostgresDialect.__new__(PostgresDialect)
        parsed = pg.parse_select(f"SELECT 1 FROM {from_t} WHERE TRUE")
        carriers = pg.ordered_join_carrier_froms(parsed)
        assert carriers
        assert pg.from_anchor_of(carriers[0]) == from_t
        assert pg.attach_joins(parsed, carriers[0], edges)
        emitted = pg.emit_sql(parsed).lower()
        assert "join customers" in emitted
        assert "orders.cid" in emitted
        assert "customers.id" in emitted

    def test_attach_joins_databricks_emits_equivalent_join(self):
        from aetherdialect._dialect import JoinEdge
        from aetherdialect._sql_gen import _join_edges_from_signature

        sig = ["orders.cid->customers.id"]
        from_t = "orders"
        result = _join_edges_from_signature(sig, [""], from_t, None)
        assert result is not None
        edges, where_edges, extra_from = result
        assert edges and isinstance(edges[0], JoinEdge)
        assert where_edges == []
        assert extra_from == []
        dx = DatabricksDialect.__new__(DatabricksDialect)
        parsed = dx.parse_select(f"SELECT 1 FROM {from_t} WHERE TRUE")
        carriers = dx.ordered_join_carrier_froms(parsed)
        assert carriers
        assert dx.from_anchor_of(carriers[0]) == from_t
        assert dx.attach_joins(parsed, carriers[0], edges)
        emitted = dx.emit_sql(parsed).lower()
        assert "join customers" in emitted
        assert "`orders`.`cid`" in emitted
        assert "`customers`.`id`" in emitted


class TestPartitionPathJoinVsWhere:
    """``_partition_path_join_vs_where`` routes by edge kind."""

    def test_fk_kinds_go_to_join_bucket(self):
        from aetherdialect._sql_gen import _partition_path_join_vs_where

        sig = ["a.fk->b.id", "b.fk->c.id"]
        kinds = ["catalog", "inferred"]
        join_b, where_b = _partition_path_join_vs_where(sig, kinds)
        assert len(join_b) == 2
        assert where_b == []

    def test_semantic_profile_goes_to_where_bucket(self):
        from aetherdialect._sql_gen import _partition_path_join_vs_where

        sig = ["a.fk->b.id", "b.x->c.x"]
        kinds = ["catalog", "semantic_profile"]
        join_b, where_b = _partition_path_join_vs_where(sig, kinds)
        assert len(join_b) == 1
        assert join_b[0][1] == "a"
        assert len(where_b) == 1
        assert where_b[0][1] == "b" and where_b[0][2] == "c"

    def test_semantic_profile_virtual_goes_to_where_bucket(self):
        from aetherdialect._sql_gen import _partition_path_join_vs_where

        sig = ["a.x->b.x"]
        join_b, where_b = _partition_path_join_vs_where(sig, ["semantic_profile_virtual"])
        assert join_b == []
        assert len(where_b) == 1

    def test_unknown_kind_defaults_to_join_bucket(self):
        from aetherdialect._sql_gen import _partition_path_join_vs_where

        sig = ["a.fk->b.id"]
        join_b, where_b = _partition_path_join_vs_where(sig, ["mystery_kind"])
        assert len(join_b) == 1
        assert where_b == []

    def test_missing_kind_defaults_to_join_bucket(self):
        from aetherdialect._sql_gen import _partition_path_join_vs_where

        sig = ["a.fk->b.id", "b.fk->c.id"]
        join_b, where_b = _partition_path_join_vs_where(sig, [])
        assert len(join_b) == 2
        assert where_b == []


class TestJoinEdgesFromSignatureMixed:
    """``_join_edges_from_signature`` mixed FK + semantic routing."""

    def test_fk_plus_semantic_returns_join_where_and_extra_from(self):
        from aetherdialect._sql_gen import _join_edges_from_signature

        sig = ["orders.cid->customers.id", "customers.email->ext.email"]
        kinds = ["catalog", "semantic_profile"]
        result = _join_edges_from_signature(sig, kinds, "orders", None)
        assert result is not None
        join_edges, where_edges, extra_from = result
        assert len(join_edges) == 1
        assert join_edges[0].table == "customers"
        assert len(where_edges) == 1
        assert "ext" in extra_from

    def test_pure_semantic_yields_empty_join_and_populated_where(self):
        from aetherdialect._sql_gen import _join_edges_from_signature

        sig = ["a.x->b.x"]
        result = _join_edges_from_signature(sig, ["semantic_profile"], "a", None)
        assert result is not None
        join_edges, where_edges, extra_from = result
        assert join_edges == []
        assert len(where_edges) == 1
        assert "b" in extra_from

    def test_self_join_in_join_bucket_raises(self):
        import pytest

        from aetherdialect._sql_gen import NoJoinPathError, _join_edges_from_signature

        sig = ["film.parent_film_id->film.film_id"]
        with pytest.raises(NoJoinPathError):
            _join_edges_from_signature(sig, ["catalog"], "film", None)


class TestInjectJoinDeterministicSqlMixed:
    """End-to-end inject with mixed FK + semantic kinds."""

    def test_fk_plus_semantic_emits_join_extra_from_and_where(self):
        det = "SELECT 1 FROM orders WHERE TRUE"
        sigs = [["orders.cid->customers.id", "customers.email->ext.email"]]
        kinds = [["catalog", "semantic_profile"]]
        out = inject_join_into_deterministic_sql(
            det,
            sigs,
            schema=None,
            edge_kinds_ordered=kinds,
            dialect=PostgresDialect.__new__(PostgresDialect),
        )
        low = out.lower()
        assert "join customers" in low
        assert "ext" in low
        assert "customers.email" in low and "ext.email" in low

    def test_single_join_attached_via_default_dialect(self):
        """A single signature attaches via the Postgres AST adapter."""
        det_sql = "SELECT t1.a\nFROM t1\nWHERE t1.a = :p1"
        sigs = [["t1.fk->t2.pk"]]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, None, dialect=dialect)
        low = out.lower()
        assert "join t2" in low
        assert "t1.fk" in low
        assert "t2.pk" in low
        assert "from t1" in low
        assert "where" in low

    def test_ast_path_with_postgres_dialect(self):
        """When dialect is set, the pglast adapter injects joins and preserves named placeholders."""
        det_sql = "SELECT t1.a\nFROM t1\nWHERE x = :p1"
        sigs = [["t1.fk->t2.pk"]]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, None, dialect=dialect)
        low = out.lower()
        assert "join t2" in low
        assert "t1.fk" in low
        assert "t2.pk" in low
        assert ":p1" in out
        assert "%(p1)s" not in out

    def test_ast_path_postgres_preserves_structural_placeholders(self):
        """Structural ``:sN`` and filter ``:pN`` placeholders survive the AST path verbatim."""
        det_sql = "SELECT SUM(:s1 * t1.amt) + :s2 AS total\nFROM t1\nWHERE LOWER(t1.cat) = :p1"
        sigs = [["t1.fk->t2.pk"]]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, None, dialect=dialect)
        assert ":s1" in out
        assert ":s2" in out
        assert ":p1" in out
        assert "%(s1)s" not in out
        assert "%(s2)s" not in out
        assert "%(p1)s" not in out

    def test_spark_dialect_emit_backticks_reserved_table(self):
        """Spark read/write may quote identifiers; join semantics preserved."""
        det_sql = "SELECT `user`.id\nFROM `user`\nWHERE 1=1"
        sigs = [["user.fk->other.pk"]]
        dialect = DatabricksDialect.__new__(DatabricksDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, None, dialect=dialect)
        assert "other" in out.lower()
        assert "fk" in out and "pk" in out

    def test_no_sigs_returns_unchanged(self):
        """Empty join_sigs_ordered leaves SQL untouched."""
        det_sql = "SELECT a\nFROM t1"
        out = inject_join_into_deterministic_sql(det_sql, [])
        assert out == det_sql

    def test_no_dialect_returns_unchanged(self):
        """Without a dialect adapter the function is a no-op."""
        det_sql = "SELECT a\nFROM t1\nWHERE 1=1"
        out = inject_join_into_deterministic_sql(det_sql, [["t1.fk->t2.pk"]])
        assert out == det_sql

    def test_parse_failure_raises_join_injection_failed(self):
        """When the dialect cannot parse deterministic SQL, injection fails loudly."""

        dialect = PostgresDialect.__new__(PostgresDialect)
        with pytest.raises(JoinInjectionFailedError) as exc_info:
            inject_join_into_deterministic_sql("NOT SQL", [["t1.fk->t2.pk"]], dialect=dialect)
        assert exc_info.value.det_sql == "NOT SQL"

    def test_empty_inner_sig_returns_unchanged(self):
        """An empty path for a slot causes the AST attach to skip; the result has no JOIN."""
        det_sql = "SELECT a\nFROM t1\nWHERE 1=1"
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, [[]], schema=None, dialect=dialect)
        assert "join" not in out.lower()
        assert "from t1" in out.lower()

    def test_multiple_carriers_cte_then_main(self):
        """Two carriers (CTE inner SELECT then outer SELECT) get two signatures."""
        det_sql = "WITH cte1 AS (\nSELECT x\nFROM a\n)\nSELECT cte1.x\nFROM cte1"
        sigs = [["a.fk->b.pk"], ["cte1.x->other.y"]]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, None, dialect=dialect)
        low = out.lower()
        assert "join b" in low
        assert "a.fk = b.pk" in low
        assert "join other" in low
        assert "cte1.x = other.y" in low

    def test_cte_alias_then_main_join_self_fk_target(self):
        """Main query joins a CTE alias to the base table for a self-referential FK."""

        det_sql = "WITH emp_mgr AS (SELECT id, manager_id FROM employee)\nSELECT emp_mgr.id\nFROM emp_mgr"
        sigs = [[], ["emp_mgr.manager_id->employee.id"]]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, schema=None, dialect=dialect)
        assert "join employee" in out.lower()
        assert "emp_mgr.manager_id" in out.replace(" ", "").lower()

    def test_inject_join_raises_on_carrier_alignment_mismatch(self) -> None:
        from unittest.mock import MagicMock

        dialect = MagicMock()
        dialect.parse_select.return_value = object()
        dialect.ordered_join_carrier_froms.return_value = ["c0", "c1", "c2"]
        with pytest.raises(JoinInjectionAlignmentError):
            _try_ast_inject_joins(
                "SELECT 1 FROM t",
                [["a->b"], []],
                [[], []],
                None,
                dialect,
            )

    def test_inject_join_into_deterministic_sql_emits_cte_inner_join_for_two_table_cte(
        self,
    ) -> None:
        det_sql = 'WITH cte1 AS (\nSELECT "actor"."actor_id"\nFROM "actor"\n)\nSELECT "cte1"."actor_id"\nFROM cte1'
        sigs = [["actor.actor_id->film_actor.actor_id"], []]
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det_sql, sigs, None, dialect=dialect)
        low = out.lower()
        assert "join" in low and "film_actor" in low
        assert "actor.actor_id" in low.replace('"', "").lower()


class TestReduceStructuralSqlPlaceholders:
    """Structural placeholder reduction for join-LLM / executable SQL."""

    def test_identity_inlines_s1(self):
        sql, rem = reduce_structural_sql_placeholders(
            "SELECT :s1 * amount AS x FROM t WHERE id = :p1",
            {"s1": 1, "p1": 5},
            None,
        )
        assert ":s1" not in sql
        assert "1 * amount" in sql or "* amount" in sql
        assert ":p1" in sql
        assert rem == {"p1": 5}

    def test_template_default_inlines_when_match(self):
        sql, rem = reduce_structural_sql_placeholders(
            "SELECT :s1 + col FROM t",
            {"s1": 7},
            {"s1": 7},
        )
        assert ":s1" not in sql
        assert "7" in sql
        assert rem == {}

    def test_template_default_keeps_when_mismatch(self):
        sql, rem = reduce_structural_sql_placeholders(
            "SELECT :s1 + col FROM t",
            {"s1": 3},
            {"s1": 7},
        )
        assert ":s1" in sql
        assert rem == {"s1": 3}

    def test_join_placeholder_preserved(self):
        marker = "-- <JOIN: integrate from join candidates>"
        det = "SELECT a FROM t1\n" + marker + "\nWHERE x = :s1 AND y = :p1"
        sql, rem = reduce_structural_sql_placeholders(det, {"s1": 0, "p1": "z"}, None)
        assert marker in sql
        assert ":p1" in sql

    def test_finalize_executable_sql_end_to_end(self):
        out = finalize_executable_sql(
            "SELECT :s1 * x WHERE id = :p1",
            {"s1": 1, "p1": 42},
            None,
            sqlglot_dialect="postgres",
        )
        assert ":p1" not in out
        assert "42" in out
        assert ":s1" not in out


class TestJoinClauseKindAndSyntheticPaths:
    """Join NULLability / role and synthetic CTE–physical paths."""

    def test_join_path_signature_dedupe_key_order_invariant(self):
        e1 = {"src_table": "a", "src_cols": ["x"], "dst_table": "b", "dst_cols": ["y"]}
        e2 = {"src_table": "b", "src_cols": ["y"], "dst_table": "c", "dst_cols": ["z"]}
        s1 = tuple(sorted(_join_path_signature_for_path([e1, e2])))
        s2 = tuple(sorted(_join_path_signature_for_path([e2, e1])))
        assert s1 == s2

    def test_nullable_fk_source_uses_left_join(self):
        schema = SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="int",
                            is_primary_key=True,
                            value_type="integer",
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                ),
                "lines": TableMetadata(
                    name="lines",
                    columns={
                        "oid": ColumnMetadata(
                            name="oid",
                            data_type="int",
                            is_foreign_key=True,
                            fk_target=("orders", "id"),
                            is_nullable=True,
                            value_type="integer",
                        ),
                    },
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        sig = ["orders.id->lines.oid"]
        clause = _join_clause_from_signature(sig, "orders", schema)
        assert "LEFT JOIN lines" in clause

    def test_non_nullable_fk_uses_inner_when_not_dimension(self):
        schema = SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="int",
                            is_primary_key=True,
                            value_type="integer",
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                ),
                "lines": TableMetadata(
                    name="lines",
                    columns={
                        "oid": ColumnMetadata(
                            name="oid",
                            data_type="int",
                            is_foreign_key=True,
                            fk_target=("orders", "id"),
                            is_nullable=False,
                            value_type="integer",
                        ),
                    },
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        sig = ["orders.id->lines.oid"]
        clause = _join_clause_from_signature(sig, "orders", schema)
        assert "INNER JOIN lines" in clause

    def test_virtual_pk_bridge_via_join_hints(self):
        film_cols = {
            "FilmId": ColumnMetadata(
                name="FilmId",
                data_type="integer",
                is_primary_key=True,
                value_type="integer",
            ),
        }
        schema = SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns=film_cols,
                    primary_key=["FilmId"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        cte_step = RuntimeCteStep(
            cte_name="sq",
            tables=["film"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.FilmId"))],
        )
        intent = RuntimeIntent(
            tables=["film", "sq"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte_step],
        )
        virt = {
            "sq": VirtualTableSpec(
                cte_name="sq",
                columns={"FilmId": VirtualColumnSpec("film", "FilmId", True, None, [])},
            )
        }
        hints = join_hints_multi(
            schema,
            ["film", "sq"],
            intent,
            virtual_specs=virt,
            include_semantic=False,
        )
        kinds = [tuple(c.get("edge_kinds", ())) for c in hints["candidates"] if c.get("candidate_id") != "J00"]
        assert any("virtual_pk_bridge" in k for k in kinds)

    def test_virtual_shared_pk_between_two_ctes_on_same_base_pk(self):
        """Two CTEs inheriting the same physical PK column emit virtual_shared_pk (plan D.2)."""
        film_cols = {
            "FilmId": ColumnMetadata(
                name="FilmId",
                data_type="integer",
                is_primary_key=True,
                value_type="integer",
            ),
        }
        schema = SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns=film_cols,
                    primary_key=["FilmId"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        virt = {
            "ca": VirtualTableSpec(
                cte_name="ca",
                columns={"fid": VirtualColumnSpec("film", "FilmId", True, None, [])},
            ),
            "cb": VirtualTableSpec(
                cte_name="cb",
                columns={"fid": VirtualColumnSpec("film", "FilmId", True, None, [])},
            ),
        }
        hints = join_hints_multi(
            schema,
            ["film", "ca", "cb"],
            None,
            virtual_specs=virt,
            include_semantic=False,
        )
        flat_kinds: list[str] = []
        for c in hints.get("candidates", []):
            flat_kinds.extend(list(c.get("edge_kinds") or []))
        assert "virtual_shared_pk" in flat_kinds


class TestJoinClausePartsWithBoolOpMatrix:
    """Matrix coverage for ``_join_clause_parts_with_bool_op`` (AND/OR wrapping)."""

    def test_empty_returns_empty_string(self) -> None:
        assert _join_clause_parts_with_bool_op([]) == ""

    def test_single_fragment_no_wrapping(self) -> None:
        assert _join_clause_parts_with_bool_op([("a = 1", "AND")]) == "a = 1"

    def test_two_and_chain_no_outer_parens(self) -> None:
        out = _join_clause_parts_with_bool_op([("a = 1", "AND"), ("b = 2", "OR")])
        assert out == "a = 1 AND b = 2"
        assert not out.startswith("(")

    def test_or_in_first_connector_wraps(self) -> None:
        out = _join_clause_parts_with_bool_op([("a = 1", "OR"), ("b = 2", "AND")])
        assert out.startswith("(")
        assert " OR " in out
        assert out.endswith(")")

    def test_three_fragments_middle_or_wraps(self) -> None:
        out = _join_clause_parts_with_bool_op(
            [
                ("a = 1", "AND"),
                ("b = 2", "OR"),
                ("c = 3", "AND"),
            ]
        )
        assert out.startswith("(")
        assert " OR " in out

    def test_all_and_three_parts_no_wrap(self) -> None:
        out = _join_clause_parts_with_bool_op(
            [
                ("a = 1", "AND"),
                ("b = 2", "AND"),
                ("c = 3", "AND"),
            ]
        )
        assert not out.startswith("(")
        assert out.count(" AND ") == 2


class TestWhereChainBoolOpRendering:
    """WHERE rendering preserves forward ``bool_op`` and ``filter_group`` parentheses."""

    def test_render_where_chain_emits_or_when_bool_op_is_or(self) -> None:
        dialect = PostgresDialect.__new__(PostgresDialect)
        sql = _build_deterministic_select_block(
            [SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            ["t"],
            [],
            [],
            [
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t.a"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                    bool_op="OR",
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t.b"),
                    op="=",
                    param_key="p2",
                    value_type="string",
                    bool_op="AND",
                ),
            ],
            [],
            None,
            "row_level",
            dialect,
        )
        assert "WHERE" in sql
        assert " OR " in sql

    def test_render_where_chain_groups_filter_group_with_parens(self) -> None:
        dialect = PostgresDialect.__new__(PostgresDialect)
        sql = _build_deterministic_select_block(
            [SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            ["t"],
            [],
            [],
            [
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t.a"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                    bool_op="AND",
                    filter_group=0,
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t.b"),
                    op="=",
                    param_key="p2",
                    value_type="string",
                    bool_op="OR",
                    filter_group=0,
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t.c"),
                    op="=",
                    param_key="p3",
                    value_type="string",
                    bool_op="AND",
                    filter_group=1,
                ),
            ],
            [],
            None,
            "row_level",
            dialect,
        )
        assert "WHERE" in sql
        assert "(" in sql
        assert " OR " in sql


class TestGetJoinChoiceFromLlm:
    """Tests for get_join_choice_from_llm validation and silent retry."""

    @patch("aetherdialect._sql_gen.llm_json")
    def test_second_call_used_when_first_id_invalid(self, mock_llm_json: MagicMock) -> None:
        cands = [
            {
                "candidate_id": "J01",
                "join_path_signature": ["a->b"],
                "candidate_tier": "base",
            },
            {
                "candidate_id": "J02",
                "join_path_signature": ["c->d"],
                "candidate_tier": "base",
            },
        ]
        scopes = [{"scope": JOIN_CHOICE_SCOPE_MAIN, "tables": [], "candidates": cands}]
        mock_llm_json.side_effect = [
            {"choices": {JOIN_CHOICE_SCOPE_MAIN: "J99"}},
            {"choices": {JOIN_CHOICE_SCOPE_MAIN: "J02"}},
        ]
        got = get_join_choice_from_llm(
            "question",
            "SELECT 1",
            llm_scopes=scopes,
            preset_choices={},
            accept_na_by_scope={JOIN_CHOICE_SCOPE_MAIN: False},
            require_final=False,
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "J02"
        assert mock_llm_json.call_count == 2

    @patch("aetherdialect._sql_gen.llm_json")
    def test_invalid_llm_choice_does_not_override_valid_preset(self, mock_llm_json: MagicMock) -> None:
        cands = [
            {
                "candidate_id": "J01",
                "join_path_signature": ["a->b"],
                "candidate_tier": "base",
            },
            {
                "candidate_id": "J02",
                "join_path_signature": ["c->d"],
                "candidate_tier": "base",
            },
        ]
        scopes = [{"scope": JOIN_CHOICE_SCOPE_MAIN, "tables": ["a", "b"], "candidates": cands}]
        mock_llm_json.return_value = {"choices": {JOIN_CHOICE_SCOPE_MAIN: "J99"}}
        got = get_join_choice_from_llm(
            "question",
            "SELECT 1",
            llm_scopes=scopes,
            preset_choices={JOIN_CHOICE_SCOPE_MAIN: "J02"},
            accept_na_by_scope={JOIN_CHOICE_SCOPE_MAIN: False},
            require_final=False,
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "J02"
        mock_llm_json.assert_called_once()

    @patch("aetherdialect._sql_gen.llm_json")
    def test_j00_when_both_attempts_invalid(self, mock_llm_json: MagicMock) -> None:
        cands = [{"candidate_id": "J01", "join_path_signature": [], "candidate_tier": "base"}]
        scopes = [{"scope": JOIN_CHOICE_SCOPE_MAIN, "tables": [], "candidates": cands}]
        mock_llm_json.side_effect = [
            {"choices": {JOIN_CHOICE_SCOPE_MAIN: "bogus"}},
            {"choices": {JOIN_CHOICE_SCOPE_MAIN: ""}},
        ]
        got = get_join_choice_from_llm(
            "q",
            "sql",
            llm_scopes=scopes,
            preset_choices={},
            accept_na_by_scope={JOIN_CHOICE_SCOPE_MAIN: False},
            require_final=False,
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "J01"
        assert mock_llm_json.call_count == 2

    @patch("aetherdialect._sql_gen.llm_json")
    def test_j00_when_llm_scopes_empty(self, mock_llm_json: MagicMock) -> None:
        got = get_join_choice_from_llm(
            "q",
            "sql",
            llm_scopes=[],
            preset_choices={JOIN_CHOICE_SCOPE_MAIN: "J01"},
            accept_na_by_scope={},
            require_final=False,
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "J01"
        mock_llm_json.assert_not_called()

    @patch("aetherdialect._sql_gen.llm_json")
    def test_j00_when_llm_json_exhausted_on_all_attempts(self, mock_llm_json: MagicMock) -> None:
        cands = [
            {
                "candidate_id": "J01",
                "join_path_signature": ["a->b"],
                "candidate_tier": "base",
            }
        ]
        scopes = [{"scope": JOIN_CHOICE_SCOPE_MAIN, "tables": [], "candidates": cands}]
        mock_llm_json.side_effect = [
            LlmJsonExhausted(task="join", attempts=2),
            LlmJsonExhausted(task="join", attempts=2),
        ]
        got = get_join_choice_from_llm(
            "q",
            "sql",
            llm_scopes=scopes,
            preset_choices={},
            accept_na_by_scope={JOIN_CHOICE_SCOPE_MAIN: False},
            require_final=False,
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "J01"
        assert mock_llm_json.call_count == 2

    @patch("aetherdialect._sql_gen.llm_json")
    def test_na_accepted_on_first_pass_when_enabled(self, mock_llm_json: MagicMock) -> None:
        main_c = [
            {
                "candidate_id": "J00",
                "join_path_signature": [],
                "candidate_tier": "base",
            },
            {
                "candidate_id": "J01",
                "join_path_signature": ["a.x->b.y"],
                "candidate_tier": "base",
            },
        ]
        c1_c = [{"candidate_id": "J00", "join_path_signature": [], "candidate_tier": "base"}]
        scopes = [
            {
                "scope": JOIN_CHOICE_SCOPE_MAIN,
                "tables": ["a", "b"],
                "candidates": main_c,
            },
            {
                "scope": join_choice_scope_key_cte("c1"),
                "tables": ["x"],
                "candidates": c1_c,
            },
        ]
        mock_llm_json.return_value = {
            "choices": {
                JOIN_CHOICE_SCOPE_MAIN: "NA",
                join_choice_scope_key_cte("c1"): "NA",
            },
        }
        got = get_join_choice_from_llm(
            "q",
            "sql",
            llm_scopes=scopes,
            preset_choices={},
            accept_na_by_scope={
                JOIN_CHOICE_SCOPE_MAIN: True,
                join_choice_scope_key_cte("c1"): True,
            },
            require_final=False,
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "NA"
        assert got[join_choice_scope_key_cte("c1")] == "NA"


class TestClassifyScopeCandidates:
    """Per-scope classification rows for join LLM policy."""

    def _c(self, cid: str, tier: str) -> dict:
        return {
            "candidate_id": cid,
            "join_path_signature": ["x"],
            "candidate_tier": tier,
        }

    def test_single_table_j00_only_needs_join_false(self) -> None:
        assert classify_scope_candidates([{"candidate_id": "J00", "join_path_signature": []}], needs_join=False) == (
            ScopeClass.single_table
        )

    def test_empty_when_needs_join_and_only_j00(self) -> None:
        assert classify_scope_candidates([{"candidate_id": "J00", "join_path_signature": []}], needs_join=True) == (
            ScopeClass.empty
        )

    def test_single_fk(self) -> None:
        assert classify_scope_candidates([{"candidate_id": "J00"}, self._c("J01", "base")], needs_join=True) == (
            ScopeClass.single_fk
        )

    def test_multi_fk(self) -> None:
        c = [{"candidate_id": "J00"}, self._c("J01", "base"), self._c("J02", "base")]
        assert classify_scope_candidates(c, needs_join=True) == ScopeClass.multi_fk

    def test_semantic_only(self) -> None:
        c = [{"candidate_id": "J00"}, self._c("J01", "extended")]
        assert classify_scope_candidates(c, needs_join=True) == ScopeClass.semantic_only

    def test_single_fk_with_semantic(self) -> None:
        c = [
            {"candidate_id": "J00"},
            self._c("J01", "base"),
            self._c("J02", "extended"),
        ]
        assert classify_scope_candidates(c, needs_join=True) == ScopeClass.single_fk_with_semantic

    def test_multi_fk_with_semantic(self) -> None:
        c = [
            {"candidate_id": "J00"},
            self._c("J01", "base"),
            self._c("J02", "base"),
            self._c("J03", "extended"),
        ]
        assert classify_scope_candidates(c, needs_join=True) == ScopeClass.multi_fk_with_semantic


class TestMergeJoinHintsNaScopesUsesMainKey:
    """NA merge must recognise the ``main`` scope key."""

    def test_main_key_triggers_regen(self) -> None:
        intent = RuntimeIntent(
            tables=["t1", "t2"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        schema = SchemaGraph(
            tables={
                "t1": TableMetadata(name="t1", columns={}, primary_key=[], foreign_keys=[]),
                "t2": TableMetadata(name="t2", columns={}, primary_key=[], foreign_keys=[]),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        pass1_main = {
            "candidates": [
                {
                    "candidate_id": "J00",
                    "join_path_signature": [],
                    "candidate_tier": "base",
                }
            ]
        }
        gen_main, _ = merge_join_hints_for_na_scopes(
            pass1_main,
            {},
            intent,
            schema,
            {},
            frozenset({JOIN_CHOICE_SCOPE_MAIN}),
        )
        assert isinstance(gen_main.get("candidates"), list)


class TestOrientJoinSigForFrom:
    """``_orient_join_sig_for_from`` flips segments when FROM matches RHS table."""

    def test_empty_from_returns_unchanged(self) -> None:
        sig = ["a.x->b.y"]
        assert _orient_join_sig_for_from(sig, "") == sig

    def test_flips_when_right_matches_from(self) -> None:
        sig = ["b.y->a.x"]
        out = _orient_join_sig_for_from(sig, "a")
        assert out == ["a.x->b.y"]

    def test_preserves_when_right_differs(self) -> None:
        sig = ["a.x->b.y"]
        assert _orient_join_sig_for_from(sig, "a") == sig

    def test_no_arrow_passthrough(self) -> None:
        assert _orient_join_sig_for_from(["nonsense"], "a") == ["nonsense"]


class TestJoinKindForEdge:
    """``_join_kind_for_edge`` LEFT vs INNER rules."""

    def test_no_schema_inner(self) -> None:
        assert _join_kind_for_edge("t", "u", ["c"], None) == " INNER"

    def test_unknown_table_inner(self) -> None:
        schema = SchemaGraph(
            tables={"only": TableMetadata(name="only", columns={}, primary_key=[], foreign_keys=[])},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        assert _join_kind_for_edge("missing", "only", ["x"], schema) == " INNER"

    def test_dimension_role_no_longer_biases_to_left(self) -> None:
        dim = TableMetadata(
            name="dim",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_nullable=False,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role="dimension",
        )
        schema = SchemaGraph(tables={"dim": dim}, join_paths_multi={}, effective_structural_hash="h")
        assert _join_kind_for_edge("dim", "fact", ["id"], schema) == " INNER"

    def test_nullable_fk_source_left(self) -> None:
        src = TableMetadata(
            name="src",
            columns={
                "fk": ColumnMetadata(
                    name="fk",
                    data_type="integer",
                    value_type="integer",
                    is_nullable=True,
                    is_foreign_key=True,
                    fk_target=("tgt", "id"),
                ),
            },
            primary_key=[],
            foreign_keys=[],
        )
        schema = SchemaGraph(tables={"src": src}, join_paths_multi={}, effective_structural_hash="h")
        assert _join_kind_for_edge("src", "tgt", ["fk"], schema) == " LEFT"

    def test_non_nullable_fk_source_inner(self) -> None:
        src = TableMetadata(
            name="src",
            columns={
                "fk": ColumnMetadata(
                    name="fk",
                    data_type="integer",
                    value_type="integer",
                    is_nullable=False,
                    is_foreign_key=True,
                    fk_target=("tgt", "id"),
                ),
            },
            primary_key=[],
            foreign_keys=[],
        )
        schema = SchemaGraph(tables={"src": src}, join_paths_multi={}, effective_structural_hash="h")
        assert _join_kind_for_edge("src", "tgt", ["fk"], schema) == " INNER"


class TestWrapCaseInsensitive:
    """Dialect hook used by WHERE rendering."""

    def test_delegates_to_dialect(self) -> None:
        dialect = SimpleNamespace(render_case_insensitive_wrap=lambda e: f"LOWER({e})")
        assert _wrap_for_case_insensitive("t.name", dialect) == "LOWER(t.name)"


class TestRenderGroupSqlLeadingArgFuncs:
    """``SCALAR_FUNCTIONS_LEADING_ARG`` branch (date_trunc, extract, …)."""

    def test_date_trunc_leading_arg_order(self) -> None:
        g = MulGroup(
            multiply=["t.created_at"],
            inner_scalar_func="date_trunc",
            isarg_param_keys=["unit"],
            inner_scalar_func_args=["day"],
        )
        out = _render_group_sql(g)
        assert "DATE_TRUNC" in out
        assert out.index("DATE_TRUNC") < out.index("t.created_at")

    def test_extract_outer_in_group(self) -> None:
        g = MulGroup(
            multiply=["t.d"],
            scalar_func="extract",
            sarg_param_keys=["part"],
            scalar_func_args=["year"],
        )
        out = _render_group_sql(g)
        assert "EXTRACT" in out
        assert "FROM" in out


class TestRenderExprSqlTopLevelFuncs:
    """Expression-level scalar/agg when not on MulGroups."""

    def test_expr_level_date_trunc_wraps_sum(self) -> None:
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.x"], agg_func="sum")],
            inner_scalar_func="date_trunc",
            isarg_param_keys=["u"],
            inner_scalar_func_args=["month"],
        )
        out = render_expr_sql(expr)
        assert "DATE_TRUNC" in out
        assert "SUM" in out

    def test_expr_level_extract(self) -> None:
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.d"])],
            scalar_func="extract",
            sarg_param_keys=["p"],
            scalar_func_args=["dow"],
        )
        out = render_expr_sql(expr)
        assert "EXTRACT" in out and "FROM" in out


class TestRenderCaseWhenSql:
    """CASE / WHEN rendering for SELECT."""

    def test_is_null_branch(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="is null",
            value_type="string",
        )
        assert _render_case_branch_sql(fp) == "t.x IS NULL"

    def test_param_comparison(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="integer",
            param_key="p1",
        )
        assert _render_case_branch_sql(fp) == "t.a = :p1"

    def test_in_branch_renders_param_in_parens(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.kind"),
            op="in",
            value_type="string",
            param_key="p1",
        )
        assert _render_case_branch_sql(fp) == "t.kind IN (:p1)"

    def test_not_in_branch_renders_param_in_parens(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.kind"),
            op="not in",
            value_type="string",
            param_key="p2",
        )
        assert _render_case_branch_sql(fp) == "t.kind NOT IN (:p2)"

    def test_case_when_string_literal_branch_quotes_apostrophe(self) -> None:
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch.from_dict(
                    {
                        "condition": FilterParam(
                            left_expr=NormalizedExpr.from_column("t.flag"),
                            op="=",
                            param_key="p1",
                            value_type="integer",
                        ).to_dict(),
                        "literal_string": "Bob's",
                    }
                )
            ],
            else_result=NormalizedExpr(string_literal="none"),
        )
        sql = _render_case_when_sql(cw)
        assert "THEN 'Bob''s'" in sql.replace("\n", " ")
        assert "ELSE 'none'" in sql.replace("\n", " ")

    def test_case_when_with_else(self) -> None:
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=FilterParam(
                        left_expr=NormalizedExpr.from_column("t.flag"),
                        op="=",
                        param_key="p1",
                        value_type="integer",
                    ),
                    result=NormalizedExpr.from_column("t.a"),
                )
            ],
            else_result=NormalizedExpr.from_column("t.b"),
        )
        sql = _render_case_when_sql(cw)
        assert sql.startswith("CASE ")
        assert "WHEN" in sql and "THEN" in sql
        assert "ELSE" in sql and sql.endswith("END")


class TestRenderSelectColCaseAndWindow:
    """``render_select_col_sql`` dispatches CASE vs window."""

    def test_select_col_prefers_case_over_expr(self) -> None:
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=FilterParam(
                        left_expr=NormalizedExpr.from_column("t.ok"),
                        op="is not null",
                        value_type="string",
                    ),
                    result=NormalizedExpr.from_column("t.v"),
                )
            ],
        )
        step = CaseRegistryStep(registry_id="c01", case_when=cw)
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        with registry_render_scope(None, [step]):
            assert render_select_col_sql(sc).startswith("CASE")

    def test_window_sum_with_partition_and_order(self) -> None:
        ws = WindowSpec(
            function="sum",
            argument=NormalizedExpr.from_column("o.amount"),
            partition_by=[NormalizedExpr.from_column("o.customer_id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("o.dt"), direction="asc")],
        )
        out = _render_window_over_sql(ws, _pg_render())
        compact = out.replace(" ", "").replace('"', "")
        assert "SUM(o.amount)" in compact
        assert "PARTITION BY" in out.upper()
        assert "ORDER BY" in out.upper()

    def test_window_row_number_no_args(self) -> None:
        ws = WindowSpec(function="row_number", order_by=[])
        assert "ROW_NUMBER()" in _render_window_over_sql(ws, _pg_render()).replace(" ", "")

    def test_window_first_value(self) -> None:
        ws = WindowSpec(
            function="first_value",
            argument=NormalizedExpr.from_column("t.price"),
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("t.t"), direction="DESC")],
        )
        out = _render_window_over_sql(ws, _pg_render())
        compact = out.replace(" ", "").replace('"', "")
        assert "FIRST_VALUE(t.price)" in compact

    def test_spark_window_sum_strips_table_qualifiers_on_agg_arg_only(self) -> None:
        """Databricks drops table qualifiers on the aggregate argument inside ``OVER``; partition/order stay qualified."""

        ws = WindowSpec(
            function="sum",
            argument=NormalizedExpr.from_column("tbl.amount"),
            partition_by=[NormalizedExpr.from_column("tbl.g")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("tbl.dt"), direction="asc")],
        )
        out = _render_window_over_sql(ws, _dbr_render()).replace(" ", "")
        assert "SUM(`amount`)" in out or "SUM(amount)" in out
        assert "`tbl`.`amount`" not in out
        assert "`tbl`.`g`" in out
        assert "`tbl`.`dt`" in out


class TestBuildDeterministicSqlEdgeCases:
    """Filters and anchors not covered elsewhere."""

    def test_string_like_uses_lower_on_param(self) -> None:
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.name"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t1.name"),
                    op="like",
                    value_type="string",
                    param_key="pat",
                )
            ],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "LOWER" in sql
        assert "LOWER(:pat)" in sql.replace(" ", "")

    def test_ilike_skips_lower_wrap(self) -> None:
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.name"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t1.name"),
                    op="ilike",
                    value_type="string",
                    param_key="p",
                )
            ],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "ILIKE" in sql.upper()
        assert "LOWER(t1.name)" not in sql

    def test_contains_renders_array_predicate(self) -> None:
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.tags"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t1.tags"),
                    op="contains",
                    value_type="array",
                    param_key="tag",
                )
            ],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert ":tag" in sql

    def test_where_or_triggers_outer_parens(self) -> None:
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t1.a"),
                    op="=",
                    value_type="integer",
                    param_key="p1",
                    bool_op="OR",
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t1.b"),
                    op="=",
                    value_type="integer",
                    param_key="p2",
                    bool_op="AND",
                ),
            ],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, dialect=_pg_render())
        assert "WHERE (" in sql
        assert " OR " in sql

    def test_order_by_resolves_from_anchor_case_insensitive(self) -> None:
        schema = SchemaGraph(
            tables={
                "Film": TableMetadata(
                    name="Film",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                    row_count=100,
                ),
                "Language": TableMetadata(
                    name="Language",
                    columns={},
                    primary_key=[],
                    foreign_keys=[],
                    row_count=10,
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        intent = RuntimeIntent(
            tables=["film", "Language"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("film.title"), direction="asc")],
            filters_param=[],
            having_param=[],
        )
        sql = build_deterministic_sql(intent, None, schema, _pg_render())
        assert 'FROM "film"' in sql


class TestInjectJoinStripsCommentWhenComplete:
    """Stray ``-- <JOIN ...>`` markers in legacy input pass straight through the AST emit."""

    def test_marker_is_not_re_emitted(self) -> None:
        det = "SELECT a FROM t\n-- <JOIN: integrate from join candidates>\nWHERE 1=1"
        dialect = PostgresDialect.__new__(PostgresDialect)
        out = inject_join_into_deterministic_sql(det, [["t.fk->u.pk"]], dialect=dialect)
        assert "<JOIN" not in out


class TestJoinChoicePromptAndValidation:
    """``build_join_choice_prompt`` and join-choice validators (no live LLM)."""

    def test_build_join_choice_prompt_shape(self) -> None:
        scopes = [
            {
                "scope": JOIN_CHOICE_SCOPE_MAIN,
                "tables": ["t1", "t2"],
                "candidates": [{"candidate_id": "J01", "join_path_signature": ["a.x->b.y"]}],
            },
            {
                "scope": join_choice_scope_key_cte("c1"),
                "tables": ["u"],
                "candidates": [{"candidate_id": "J01", "join_path_signature": []}],
            },
        ]
        system, user = build_join_choice_prompt("q?", "SELECT 1", scopes)
        assert "join" in system.lower()
        assert "scopes" in user
        assert "J01" in user
        assert join_choice_scope_key_cte("c1") in user

    def test_build_join_choice_prompt_prior_feedback_section(self) -> None:
        scopes = [
            {
                "scope": JOIN_CHOICE_SCOPE_MAIN,
                "tables": ["t1", "t2"],
                "candidates": [{"candidate_id": "J01", "join_path_signature": ["a.x->b.y"]}],
            },
        ]
        system, user = build_join_choice_prompt(
            "q?",
            "SELECT 1",
            scopes,
            prior_join_feedback=["joined wrong dimension"],
        )
        assert JOIN_PRIOR_FEEDBACK_HEADING in system
        assert "wrong dimension" in system
        assert "scopes" in user

    def test_valid_main_candidate_ids(self) -> None:
        assert _valid_main_join_candidate_ids(
            {"candidates": [{"candidate_id": " J01 "}, {"candidate_id": ""}]}
        ) == frozenset({"J01"})

    def test_valid_cte_join_candidate_ids(self) -> None:
        got = _valid_cte_join_candidate_ids({"a": {"candidates": [{"candidate_id": "J01"}]}, "b": {"candidates": []}})
        assert got["a"] == frozenset({"J01"})
        assert got["b"] == frozenset()

    def test_parse_join_choice_payload(self) -> None:
        got = _parse_join_choice_payload(
            {
                "choices": {
                    JOIN_CHOICE_SCOPE_MAIN: " J02 ",
                    join_choice_scope_key_cte("c1"): " J01 ",
                },
            }
        )
        assert got[JOIN_CHOICE_SCOPE_MAIN] == "J02"
        assert got[join_choice_scope_key_cte("c1")] == "J01"

    def test_parse_join_choice_payload_not_dict(self) -> None:
        assert _parse_join_choice_payload({"choices": "x"}) == {}

    def test_join_choice_payload_invalid_when_scope_missing(self) -> None:
        llm_scopes = [
            {
                "scope": JOIN_CHOICE_SCOPE_MAIN,
                "candidates": [{"candidate_id": "J01", "candidate_tier": "base"}],
            },
        ]
        assert not _join_choice_payload_valid_final(
            {join_choice_scope_key_cte("x"): "J01"},
            frozenset({JOIN_CHOICE_SCOPE_MAIN}),
            llm_scopes,
        )

    def test_join_choice_payload_invalid_id(self) -> None:
        llm_scopes = [
            {
                "scope": JOIN_CHOICE_SCOPE_MAIN,
                "candidates": [{"candidate_id": "J01", "candidate_tier": "base"}],
            },
        ]
        assert not _join_choice_payload_valid_final(
            {JOIN_CHOICE_SCOPE_MAIN: "J99"},
            frozenset({JOIN_CHOICE_SCOPE_MAIN}),
            llm_scopes,
        )


class TestClassifyScopeCandidatesMatrix:
    """``classify_scope_candidates`` coverage for plan D.8 disambiguation policy."""

    @staticmethod
    def _fk(cid: str) -> dict[str, str]:
        return {"candidate_id": cid, "candidate_tier": "base"}

    @staticmethod
    def _sem(cid: str) -> dict[str, str]:
        return {"candidate_id": cid, "candidate_tier": "extended"}

    def test_empty_candidates_needs_join_returns_empty(self) -> None:
        assert classify_scope_candidates([]) == ScopeClass.empty

    def test_empty_candidates_no_join_returns_single_table(self) -> None:
        assert classify_scope_candidates([], needs_join=False) == ScopeClass.single_table

    def test_only_j00_no_join_collapses_to_single_table(self) -> None:
        assert classify_scope_candidates([{"candidate_id": "J00"}], needs_join=False) == ScopeClass.single_table

    def test_only_j00_with_join_needed_is_empty(self) -> None:
        assert classify_scope_candidates([{"candidate_id": "J00"}]) == ScopeClass.empty

    def test_single_fk_only(self) -> None:
        assert classify_scope_candidates([self._fk("J01")]) == ScopeClass.single_fk

    def test_multi_fk_no_semantic(self) -> None:
        assert classify_scope_candidates([self._fk("J01"), self._fk("J02")]) == ScopeClass.multi_fk

    def test_single_fk_with_semantic(self) -> None:
        assert classify_scope_candidates([self._fk("J01"), self._sem("J02")]) == ScopeClass.single_fk_with_semantic

    def test_multi_fk_with_semantic(self) -> None:
        assert (
            classify_scope_candidates([self._fk("J01"), self._fk("J02"), self._sem("J03")])
            == ScopeClass.multi_fk_with_semantic
        )

    def test_semantic_only(self) -> None:
        assert classify_scope_candidates([self._sem("J01")]) == ScopeClass.semantic_only


class TestVirtualFkShadowPath:
    """Plan D.2: FK from a CTE column whose target is out of scope emits a shadow bridge."""

    def test_shadow_fk_when_target_out_of_scope(self) -> None:
        """A CTE FK to an out-of-scope table is emitted as ``virtual_fk_shadow_path`` when it bridges a real path."""
        from aetherdialect._contracts_base import FKEdge

        inventory_cols = {
            "inventory_id": ColumnMetadata(
                name="inventory_id",
                data_type="integer",
                is_primary_key=True,
                value_type="integer",
            ),
            "film_id": ColumnMetadata(
                name="film_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                fk_target=("film", "film_id"),
            ),
        }
        film_cols = {
            "film_id": ColumnMetadata(
                name="film_id",
                data_type="integer",
                is_primary_key=True,
                value_type="integer",
            ),
        }
        fk_inventory_film = FKEdge(
            src_table="inventory",
            src_cols=["film_id"],
            dst_table="film",
            dst_cols=["film_id"],
        )
        schema = SchemaGraph(
            tables={
                "inventory": TableMetadata(
                    name="inventory",
                    columns=inventory_cols,
                    primary_key=["inventory_id"],
                    foreign_keys=[fk_inventory_film],
                ),
                "film": TableMetadata(
                    name="film",
                    columns=film_cols,
                    primary_key=["film_id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={
                "inventory": {
                    "film": [
                        [
                            {
                                "src_table": "inventory",
                                "src_cols": ["film_id"],
                                "dst_table": "film",
                                "dst_cols": ["film_id"],
                                "edge_kind": "catalog_fk",
                            }
                        ]
                    ]
                },
                "film": {
                    "inventory": [
                        [
                            {
                                "src_table": "inventory",
                                "src_cols": ["film_id"],
                                "dst_table": "film",
                                "dst_cols": ["film_id"],
                                "edge_kind": "catalog_fk",
                            }
                        ]
                    ]
                },
            },
            effective_structural_hash="h",
        )
        virt = {
            "cte1": VirtualTableSpec(
                cte_name="cte1",
                columns={
                    "inventory_id": VirtualColumnSpec(
                        None,
                        None,
                        False,
                        ("inventory", "inventory_id"),
                        [],
                    ),
                },
            ),
        }
        hints = join_hints_multi(
            schema,
            ["cte1", "film"],
            None,
            virtual_specs=virt,
            include_semantic=False,
        )
        kinds: list[str] = []
        for c in hints.get("candidates") or []:
            kinds.extend(list(c.get("edge_kinds") or []))
        assert "virtual_fk_shadow_path" in kinds


class TestVirtualSharedBase:
    """Plan D.2: two CTEs with role (non-PK) over the same base column emit virtual_shared_base."""

    def test_shared_base_when_both_have_fk_to_same_column(self) -> None:
        customer_cols = {
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                is_primary_key=True,
                value_type="integer",
            ),
        }
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns=customer_cols,
                    primary_key=["customer_id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        virt = {
            "ca": VirtualTableSpec(
                cte_name="ca",
                columns={
                    "customer_id": VirtualColumnSpec(
                        "customer",
                        "customer_id",
                        False,
                        ("customer", "customer_id"),
                        [],
                    ),
                },
            ),
            "cb": VirtualTableSpec(
                cte_name="cb",
                columns={
                    "customer_id": VirtualColumnSpec(
                        "customer",
                        "customer_id",
                        False,
                        ("customer", "customer_id"),
                        [],
                    ),
                },
            ),
        }
        hints = join_hints_multi(
            schema,
            ["ca", "cb"],
            None,
            virtual_specs=virt,
            include_semantic=False,
        )
        kinds: list[str] = []
        for c in hints.get("candidates") or []:
            kinds.extend(list(c.get("edge_kinds") or []))
        assert "virtual_shared_base" in kinds or "virtual_shared_lineage" in kinds


class TestFormatScalarForStructuralSqlInline:
    """``_format_scalar_for_structural_sql_inline`` edge cases."""

    def test_bool_true_false(self) -> None:
        assert _format_scalar_for_structural_sql_inline(True) == "TRUE"
        assert _format_scalar_for_structural_sql_inline(False) == "FALSE"

    def test_numeric_list_string_passthrough(self) -> None:
        assert _format_scalar_for_structural_sql_inline("1, 2, 3") == "1, 2, 3"

    def test_string_single_quote_escaped(self) -> None:
        assert _format_scalar_for_structural_sql_inline("o'reilly") == "'o''reilly'"

    def test_preformatted_quoted_list_passthrough(self) -> None:
        assert _format_scalar_for_structural_sql_inline("'a','b'") == "'a','b'"


class TestReduceStructuralPlaceholdersExtended:
    """Longer structural keys and non-identity values."""

    def test_longer_s10_before_s1_replaced_correctly(self) -> None:
        sql, rem = reduce_structural_sql_placeholders(
            "SELECT :s10 + :s1 FROM t",
            {"s10": 0, "s1": 1},
            None,
        )
        assert ":s10" not in sql and ":s1" not in sql
        assert rem == {}

    def test_non_identity_value_not_inlined(self) -> None:
        sql, rem = reduce_structural_sql_placeholders("WHERE :s1 = 1", {"s1": 99}, None)
        assert ":s1" in sql
        assert rem == {"s1": 99}


class TestGenerateColAlias:
    """``generate_col_alias`` composite and DISTINCT prefix."""

    def test_two_multiply_groups_times_alias(self) -> None:
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(multiply=["t.a"]), MulGroup(multiply=["t.b"])],
            )
        )
        assert generate_col_alias(sc) == "a_times_b"

    def test_distinct_prefix_on_agg(self) -> None:
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(distinct=True, multiply=["t.id"], agg_func="count")],
            )
        )
        assert "distinct_" in generate_col_alias(sc)


class TestSelectColPrefersLlmDisplayAlias:
    """``select_col_prefers_llm_display_alias`` heuristics."""

    def test_simple_column_false(self) -> None:
        sc = SelectCol(expr=NormalizedExpr.from_column("t.id"))
        assert select_col_prefers_llm_display_alias(sc) is False

    def test_agg_true(self) -> None:
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"])], agg_func="sum"))
        assert select_col_prefers_llm_display_alias(sc) is True


class TestMaybeRenderArrayUnnestSelect:
    """CTE array column expansion via mocked column metadata."""

    def test_returns_unnest_sql_when_element_type(self) -> None:
        meta = ColumnMetadata(name="tags", data_type="text[]", element_type="text")
        sc = SelectCol(expr=NormalizedExpr.from_column("t1.tags"))
        dialect = _pg_render()
        with patch("aetherdialect._sql_gen.get_col_meta", return_value=meta):
            schema = SchemaGraph(
                tables={
                    "t1": TableMetadata(
                        name="t1",
                        columns={"tags": meta},
                        primary_key=[],
                        foreign_keys=[],
                    )
                },
                join_paths_multi={},
                effective_structural_hash="h",
            )
            out = _maybe_render_array_unnest_select(
                sc,
                schema,
                {},
                dialect,
                ["tag_col"],
                0,
                for_cte=True,
            )
        assert out is not None
        assert "UNNEST" in out and "tags" in out
        assert "AS tag_col" in out

    def test_skipped_when_not_cte(self) -> None:
        meta = ColumnMetadata(name="tags", data_type="text[]", element_type="text")
        sc = SelectCol(expr=NormalizedExpr.from_column("t1.tags"))
        with patch("aetherdialect._sql_gen.get_col_meta", return_value=meta):
            assert (
                _maybe_render_array_unnest_select(
                    sc,
                    SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="h"),
                    {},
                    _pg_render(),
                    None,
                    0,
                    for_cte=False,
                )
                is None
            )


class TestCandidateJoinPathsBridgeFallback:
    """``_candidate_join_paths_for_tables`` falls back when bridge tables are required."""

    def test_bridge_path_when_direct_paths_exclude_extra_tables(self) -> None:
        bridge_edge = {
            "src_table": "a",
            "src_cols": ["id"],
            "dst_table": "bridge",
            "dst_cols": ["aid"],
        }
        second = {
            "src_table": "bridge",
            "src_cols": ["cid"],
            "dst_table": "c",
            "dst_cols": ["id"],
        }
        path = [bridge_edge, second]
        schema = SchemaGraph(
            tables={
                "a": TableMetadata(name="a", columns={}, primary_key=[], foreign_keys=[]),
                "c": TableMetadata(name="c", columns={}, primary_key=[], foreign_keys=[]),
                "bridge": TableMetadata(name="bridge", columns={}, primary_key=[], foreign_keys=[]),
            },
            join_paths_multi={"a": {"c": [path]}},
            effective_structural_hash="h",
        )
        result = _candidate_join_paths_for_tables(schema, ["a", "c"])
        assert len(result) >= 1
        sigs = {tuple(_join_path_signature_for_path(p)) for p in result}
        assert any(len(s) == 2 for s in sigs)


class TestJoinClauseFromSignatureSkipsMalformed:
    """Invalid signature segments are skipped."""

    def test_skips_segment_without_arrow(self) -> None:
        assert _join_clause_from_signature(["noarrow", "a.x->b.y"], from_table="a") != ""
        assert "JOIN b" in _join_clause_from_signature(["noarrow", "a.x->b.y"], from_table="a")

    def test_skips_segment_without_dot(self) -> None:
        assert _join_clause_from_signature(["a->b.y"], from_table="a") == ""


class TestRegistrySqlParity:
    """Window registry rendering produces expected OVER SQL."""

    def test_row_number_via_registry_renders(self) -> None:
        ob = OrderByCol(expr=NormalizedExpr.from_column("orders.amount"), direction="ASC")
        ws = WindowSpec(function="row_number", order_by=[ob])
        reg = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=ws,
                )
            ],
        )
        sql = build_deterministic_sql(reg, None, None, _pg_render())
        norm = sql.replace(" ", "").replace('"', "")
        assert "ROW_NUMBER()" in norm
        assert "OVER" in sql.upper()
        assert "orders.amount" in norm


class TestPromotedSemanticEdgeFlowsThroughJoinHints:
    """
    Integration: semantic→FK promotion + two-pass disambiguation surfaces.

    Verifies the Turn A promotion path so the existing two-pass LLM join machinery (pass 1 base FKs, pass 2 with semantics) sees promoted edges as first-class FK candidates and only falls back to semantic candidates for genuinely non-promotable edges.
    """

    def _build_pair(
        self,
        *,
        right_is_pk: bool,
    ) -> SchemaGraph:
        """Build a two-table schema with a semantic neighbor pair."""

        left_cols = {
            "right_tbl_id": ColumnMetadata(
                name="right_tbl_id",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
                row_count=200,
                null_ratio=0.0,
                semantic_join_neighbors=[("right_tbl", "id" if right_is_pk else "other")],
            ),
        }
        right_cols = {
            "id": ColumnMetadata(
                name="id",
                data_type="varchar",
                value_type="string",
                is_primary_key=True,
                distinct_count=100,
                row_count=100,
                null_ratio=0.0,
                semantic_join_neighbors=([("left_tbl", "right_tbl_id")] if right_is_pk else []),
            ),
            "other": ColumnMetadata(
                name="other",
                data_type="varchar",
                value_type="string",
                distinct_count=50,
                row_count=100,
                null_ratio=0.0,
                semantic_join_neighbors=([] if right_is_pk else [("left_tbl", "right_tbl_id")]),
            ),
        }
        return SchemaGraph(
            tables={
                "left_tbl": TableMetadata(
                    name="left_tbl",
                    columns=left_cols,
                    primary_key=[],
                    foreign_keys=[],
                ),
                "right_tbl": TableMetadata(
                    name="right_tbl",
                    columns=right_cols,
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_promoted_edge_appears_as_base_fk_candidate(self):
        """A semantic neighbor anchored on a PK is promoted and surfaces as catalog_fk in pass 1."""
        from aetherdialect._schema import (
            _promote_semantic_edges_to_fks,
            _recompute_join_paths_multi,
        )

        sg = self._build_pair(right_is_pk=True)
        promoted = _promote_semantic_edges_to_fks(sg)
        assert promoted == 1
        assert sg.tables["left_tbl"].columns["right_tbl_id"].semantic_join_neighbors == []
        assert sg.tables["right_tbl"].columns["id"].semantic_join_neighbors == []

        sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
        hints = join_hints_multi(
            sg,
            ["left_tbl", "right_tbl"],
            None,
            virtual_specs={},
            include_semantic=False,
        )
        non_anchor = [c for c in hints["candidates"] if c["candidate_id"] != "J00"]
        assert len(non_anchor) >= 1
        assert any("inferred_semantic_fk" in (c.get("edge_kinds") or []) for c in non_anchor)
        assert all("semantic_profile" not in (c.get("edge_kinds") or []) for c in non_anchor)

    def test_non_promotable_edge_only_visible_in_pass2(self):
        """A semantic neighbor with no PK endpoint is NOT promoted; pass 1 has no candidate, pass 2 surfaces it."""
        from aetherdialect._schema import (
            _promote_semantic_edges_to_fks,
            _recompute_join_paths_multi,
        )

        sg = self._build_pair(right_is_pk=False)
        promoted = _promote_semantic_edges_to_fks(sg)
        assert promoted == 0
        assert sg.tables["left_tbl"].columns["right_tbl_id"].semantic_join_neighbors == [("right_tbl", "other")]

        sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
        pass1 = join_hints_multi(
            sg,
            ["left_tbl", "right_tbl"],
            None,
            virtual_specs={},
            include_semantic=False,
        )
        pass1_non_anchor = [c for c in pass1["candidates"] if c["candidate_id"] != "J00"]
        assert pass1_non_anchor == []

        pass2 = join_hints_multi(
            sg,
            ["left_tbl", "right_tbl"],
            None,
            virtual_specs={},
            include_semantic=True,
        )
        pass2_non_anchor = [c for c in pass2["candidates"] if c["candidate_id"] != "J00"]
        assert len(pass2_non_anchor) >= 1
        assert any("semantic_profile" in (c.get("edge_kinds") or []) for c in pass2_non_anchor)

    def test_promoted_edge_canonicalized_via_full_pipeline_helpers(self):
        """End-to-end: profile→promote→recompute yields a stable J01 with catalog_fk edge_kind."""
        from aetherdialect._schema import (
            _promote_semantic_edges_to_fks,
            _recompute_join_paths_multi,
        )

        sg = self._build_pair(right_is_pk=True)
        _promote_semantic_edges_to_fks(sg)
        sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
        hints = join_hints_multi(
            sg,
            ["left_tbl", "right_tbl"],
            None,
            virtual_specs={},
            include_semantic=True,
        )
        ids = [c["candidate_id"] for c in hints["candidates"]]
        assert "J01" in ids
        j01 = next(c for c in hints["candidates"] if c["candidate_id"] == "J01")
        assert j01["candidate_tier"] == "base"
        assert "inferred_semantic_fk" in (j01.get("edge_kinds") or [])


class TestVisibilityFilterInJoinPaths:
    """Non-visible columns must not appear in any LLM-facing join candidate."""

    def _build_two_tables(
        self,
        *,
        fk_denied: bool = False,
        fk_sensitivity: str | None = None,
    ) -> SchemaGraph:
        """Two tables (orders, customers) with orders.customer_id FK to customers.id."""
        orders_cols = {
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                distinct_count=100,
                row_count=100,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                fk_target=("customers", "id"),
                distinct_count=50,
                row_count=100,
                is_denied=fk_denied,
                sensitivity=fk_sensitivity,
            ),
        }
        customers_cols = {
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                distinct_count=50,
                row_count=50,
            ),
        }
        from aetherdialect._contracts_base import FKEdge

        fk = FKEdge(
            src_table="orders",
            src_cols=["customer_id"],
            dst_table="customers",
            dst_cols=["id"],
        )
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns=orders_cols,
                    primary_key=["id"],
                    foreign_keys=[fk],
                ),
                "customers": TableMetadata(
                    name="customers",
                    columns=customers_cols,
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={
                "orders": {
                    "customers": [
                        [
                            {
                                "src_table": "orders",
                                "src_cols": ["customer_id"],
                                "dst_table": "customers",
                                "dst_cols": ["id"],
                                "edge_kind": "catalog_fk",
                            }
                        ]
                    ]
                },
                "customers": {
                    "orders": [
                        [
                            {
                                "src_table": "orders",
                                "src_cols": ["customer_id"],
                                "dst_table": "customers",
                                "dst_cols": ["id"],
                                "edge_kind": "catalog_fk",
                            }
                        ]
                    ]
                },
            },
            effective_structural_hash="h",
        )

    def test_baseline_visible_fk_yields_path(self) -> None:
        """Sanity: with visible FK, enumerate_join_paths_base yields the FK path."""
        sg = self._build_two_tables()
        paths = enumerate_join_paths_base(["orders", "customers"], sg, virtual_specs={})
        non_empty = [p for p in paths if p]
        assert len(non_empty) >= 1
        assert any(e.get("src_table") == "orders" and e.get("dst_table") == "customers" for p in non_empty for e in p)

    def test_denied_fk_column_filtered_from_paths(self) -> None:
        """is_denied on FK source column drops the path from enumerate_join_paths_base."""
        sg = self._build_two_tables(fk_denied=True)
        paths = enumerate_join_paths_base(["orders", "customers"], sg, virtual_specs={})
        assert paths == [[]]

    def test_pii_sensitivity_fk_column_filtered(self) -> None:
        """sensitivity in HIDDEN_SENSITIVITIES on FK source column drops the path."""
        sg = self._build_two_tables(fk_sensitivity="pii")
        paths = enumerate_join_paths_base(["orders", "customers"], sg, virtual_specs={})
        assert paths == [[]]

    def test_serialize_join_candidate_row_no_schema_no_assert(self) -> None:
        """Without schema kwarg, assertion is dormant even for hidden col signatures."""
        row = {
            "candidate_id": "J01",
            "join_path_signature": ["orders.customer_id->customers.id"],
            "edge_kinds": ["catalog_fk"],
            "candidate_tier": "base",
        }
        out = _serialize_join_candidate_row(row)
        assert out["candidate_id"] == "J01"

    def test_serialize_join_candidate_row_with_schema_visible_passes(self) -> None:
        """With schema kwarg, visible columns serialize normally."""
        sg = self._build_two_tables()
        row = {
            "candidate_id": "J01",
            "join_path_signature": ["orders.customer_id->customers.id"],
            "edge_kinds": ["catalog_fk"],
            "candidate_tier": "base",
        }
        out = _serialize_join_candidate_row(row, schema=sg)
        assert out["join_path_signature"] == ["orders.customer_id->customers.id"]

    def test_serialize_join_candidate_row_with_schema_hidden_raises(self) -> None:
        """With schema kwarg, a non-visible column in any signature raises AssertionError."""
        import pytest

        sg = self._build_two_tables(fk_denied=True)
        row = {
            "candidate_id": "J01",
            "join_path_signature": ["orders.customer_id->customers.id"],
            "edge_kinds": ["catalog_fk"],
            "candidate_tier": "base",
        }
        with pytest.raises(AssertionError, match="non-visible column"):
            _serialize_join_candidate_row(row, schema=sg)
