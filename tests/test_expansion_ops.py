"""Unit tests for aetherdialect._expansion_ops module."""

import copy
from dataclasses import replace
from unittest import mock

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._config import ExpansionOperatorId as Op
from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    ExpansionMetadata,
    FKEdge,
    SchemaGraph,
    TableMetadata,
    TableRole,
)
from aetherdialect._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ExprValue,
    FilterParam,
    HavingParam,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SeedWarmupIntent,
    SelectCol,
    WindowRegistryStep,
    WindowSpec,
    expr_registry_ref,
)
from aetherdialect._expansion_ops import (
    _add_expansion_metadata,
    _agg_change,
    _bridge_intermediate_add,
    _build_column_metadata,
    _build_fk_map,
    _build_operator_registry,
    _deterministic_repair_warmup_seed,
    _dimension_swap,
    _distinct_add,
    _distinct_remove,
    _expand_single_depth,
    _finalize_registry_touch_seed,
    _filter_add,
    _filter_array_contains_add,
    _filter_expr_add,
    _filter_ilike_add,
    _filter_or_group,
    _filter_remove,
    _get_dimension_tables,
    _get_filterable_columns,
    _get_groupable_columns,
    _get_temporal_columns,
    _groupby_add,
    _groupby_remove,
    _having_expr_add,
    _having_remove,
    _having_value_add,
    _include_gold,
    _join_dimension_add,
    _join_fact_add,
    _limit_add,
    _limit_remove,
    _num_abs_filter,
    _num_round_select,
    _orderby_add,
    _orderby_remove,
    _select_case_label_add,
    _select_col_trim,
    _select_expr_pair_multiply,
    _swap_agg_func,
    _table_remove,
    _tables_are_connected,
    _temp_date_diff_filter,
    _temp_date_trunc_groupby,
    _temp_date_window_filter,
    _temp_extract_groupby,
    _window_lag_add,
    _window_lead_add,
    _window_rank_add,
    _window_strip,
    _window_sum_partition_add,
    expand_gold_intents,
)
from aetherdialect._schema import assign_column_ops
from aetherdialect._utils import intent_key


def _all_operator_id_strings() -> set[str]:
    """Return every public string constant on `ExpansionOperatorId`."""
    return {getattr(Op, name) for name in dir(Op) if not name.startswith("_") and isinstance(getattr(Op, name), str)}


def _warmup_intent(**overrides) -> SeedWarmupIntent:
    """Build a minimal SeedWarmupIntent with optional overrides."""
    defaults = dict(
        intent_id="test_001",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        param_values={},
        expansion_metadata=None,
        limit=None,
    )
    defaults.update(overrides)
    return SeedWarmupIntent(**defaults)


class TestAddExpansionMetadata:
    """Tests for add_expansion_metadata."""

    def test_first_expansion(self):
        """First expansion sets depth=1 and single-element path."""
        intent = _warmup_intent()
        _add_expansion_metadata(intent, Op.FILTER_ADD)
        meta = intent.expansion_metadata
        assert meta is not None
        assert meta.operator == Op.FILTER_ADD
        assert meta.depth == 1
        assert meta.expansion_path == [Op.FILTER_ADD]

    def test_second_expansion(self):
        """Second expansion increments depth and appends to path."""
        intent = _warmup_intent()
        _add_expansion_metadata(intent, Op.FILTER_ADD)
        _add_expansion_metadata(intent, Op.GROUPBY_ADD)
        meta = intent.expansion_metadata
        assert meta.depth == 2
        assert meta.expansion_path == [Op.FILTER_ADD, Op.GROUPBY_ADD]
        assert meta.operator == Op.GROUPBY_ADD

    def test_parent_intent_id_propagated(self):
        """Second expansion inherits intent_id as parent when first parent is empty."""
        intent = _warmup_intent()
        _add_expansion_metadata(intent, Op.FILTER_ADD)
        assert intent.expansion_metadata.parent_intent_id == ""
        _add_expansion_metadata(intent, Op.AGG_CHANGE)
        assert intent.expansion_metadata.parent_intent_id == "test_001"

    def test_third_expansion(self):
        """Three-deep expansion has correct depth and path length."""
        intent = _warmup_intent()
        _add_expansion_metadata(intent, Op.FILTER_ADD)
        _add_expansion_metadata(intent, Op.FILTER_EXPR_ADD)
        _add_expansion_metadata(intent, Op.JOIN_DIMENSION_ADD)
        meta = intent.expansion_metadata
        assert meta.depth == 3
        assert len(meta.expansion_path) == 3

    def test_existing_metadata_uses_original_parent(self):
        """Existing metadata with parent_intent_id preserves it."""
        intent = _warmup_intent(
            expansion_metadata=ExpansionMetadata(
                parent_intent_id="original_parent",
                operator=Op.FILTER_ADD,
                depth=1,
                expansion_path=[Op.FILTER_ADD],
            )
        )
        _add_expansion_metadata(intent, Op.JOIN_DIMENSION_ADD)
        assert intent.expansion_metadata.parent_intent_id == "original_parent"
        assert intent.expansion_metadata.depth == 2


class TestGetDimensionTables:
    """Tests for get_dimension_tables."""

    def test_returns_dimensions(self, schema_graph):
        """Returns only dimension tables."""
        dims = _get_dimension_tables(schema_graph)
        assert "customers" in dims
        assert "products" in dims
        assert "orders" not in dims

    def test_empty_schema(self):
        """Empty schema returns empty list."""
        empty = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="empty")
        assert _get_dimension_tables(empty) == []


class TestTablesAreConnected:
    """Tests for tables_are_connected."""

    def test_single_table(self):
        """Single table is always connected."""
        assert _tables_are_connected(["orders"], {}) is True

    def test_empty_list(self):
        """Empty list is trivially connected."""
        assert _tables_are_connected([], {}) is True

    def test_two_connected(self):
        """Two tables with FK edge are connected."""
        fk_map = {
            "orders": [
                {
                    "source_column": "customer_id",
                    "target_table": "customers",
                    "target_column": "customer_id",
                }
            ]
        }
        assert _tables_are_connected(["orders", "customers"], fk_map) is True

    def test_two_disconnected(self):
        """Two tables with no FK edge are disconnected."""
        assert _tables_are_connected(["orders", "unrelated"], {}) is False

    def test_three_chain(self):
        """Three tables forming a chain are connected."""
        fk_map = {
            "orders": [
                {
                    "source_column": "customer_id",
                    "target_table": "customers",
                    "target_column": "customer_id",
                }
            ],
            "customers": [
                {
                    "source_column": "region_id",
                    "target_table": "regions",
                    "target_column": "region_id",
                }
            ],
        }
        assert (
            _tables_are_connected(
                ["orders", "customers", "regions"],
                fk_map,
            )
            is True
        )


class TestGetFilterableColumns:
    """Tests for get_filterable_columns."""

    def test_returns_filterable_roles(self, schema_graph):
        """Returns columns with CATEGORICAL, TEMPORAL, or IDENTIFIER roles."""
        cols = _get_filterable_columns(schema_graph, "orders")
        qualified_names = [c.split(".")[1] for c in cols]
        assert "order_id" in qualified_names

    def test_unknown_table(self, schema_graph):
        """Unknown table returns empty list."""
        assert _get_filterable_columns(schema_graph, "nonexistent") == []

    def test_format(self, schema_graph):
        """Returns fully-qualified table.column format."""
        cols = _get_filterable_columns(schema_graph, "orders")
        for c in cols:
            assert c.startswith("orders.")


class TestGetGroupableColumns:
    """Tests for get_groupable_columns."""

    def test_returns_dimension_and_date_roles(self, schema_graph):
        """Returns columns with CATEGORICAL or TEMPORAL roles."""
        cols = _get_groupable_columns(schema_graph, "orders")
        qualified_names = [c.split(".")[1] for c in cols]
        for name in qualified_names:
            role = schema_graph.tables["orders"].columns[name].role
            assert role in (
                ColumnRole.CATEGORICAL.value,
                ColumnRole.TEMPORAL.value,
            )


class TestBuildColumnMetadata:
    """Tests for _build_column_metadata."""

    def test_basic_structure(self, schema_graph):
        """Returns nested dict of table -> column -> metadata."""
        meta = _build_column_metadata(schema_graph)
        assert "orders" in meta
        assert "order_id" in meta["orders"]
        assert "data_type" in meta["orders"]["order_id"]

    def test_all_tables_present(self, schema_graph):
        """Every schema table appears in the result."""
        meta = _build_column_metadata(schema_graph)
        for table_name in schema_graph.tables:
            assert table_name in meta

    def test_empty_schema_returns_empty_dict(self):
        """Empty schema produces empty metadata dict."""
        empty = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        assert _build_column_metadata(empty) == {}


class TestBuildFkMap:
    """Tests for _build_fk_map."""

    def test_basic_fk(self, schema_graph):
        """FK edges are grouped by source table."""
        fk_map = _build_fk_map(schema_graph)
        found = False
        for fks in fk_map.values():
            for fk in fks:
                if fk["target_table"] in schema_graph.tables:
                    found = True
        assert found or not schema_graph.fk_edges

    def test_empty_schema(self):
        """Empty schema produces empty map."""
        empty = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="empty")
        assert _build_fk_map(empty) == {}


class TestSwapAggFunc:
    """Tests for _swap_agg_func."""

    def test_swap_on_agg_expr(self):
        """Swaps agg_func on an aggregated expression."""
        expr = NormalizedExpr.from_agg("count", "orders.order_id")
        swapped = _swap_agg_func(expr, "sum")
        assert swapped.add_groups[0].agg_func == "sum"

    def test_swap_preserves_column(self):
        """Column reference is preserved after swap."""
        expr = NormalizedExpr.from_agg("avg", "orders.amount")
        swapped = _swap_agg_func(expr, "max")
        assert swapped.primary_column == "orders.amount"


class TestFilterAdd:
    """Tests for filter-add expansion (fully deterministic)."""

    def test_produces_variants(self, schema_graph):
        """Returns filter variants for filterable columns."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _filter_add(intent, schema_graph, cm)
        assert len(results) > 0
        for r in results:
            assert len(r.filters_param) == 1

    def test_skips_max_filters(self, schema_graph):
        """Skips when max filters reached."""
        filters = [
            FilterParam(
                left_expr=NormalizedExpr.from_column(f"orders.col{i}"),
                op="=",
                value_type="string",
                param_key=f"p{i}",
            )
            for i in range(20)
        ]
        intent = _warmup_intent(tables=["orders"], filters_param=filters)
        cm = _build_column_metadata(schema_graph)
        results = _filter_add(intent, schema_graph, cm)
        assert results == []


class TestAggChange:
    """Tests for aggregation swap expansion."""

    def test_no_aggregation_skips(self, schema_graph):
        """Non-aggregated intent returns empty."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _agg_change(intent, schema_graph, cm)
        assert results == []

    def test_swaps_aggregations(self, schema_graph):
        """Aggregated intent gets alternative aggregations."""
        intent = _warmup_intent(
            tables=["orders"],
            grain="scalar",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr.from_agg("count", "orders.order_id"),
                )
            ],
        )
        cm = _build_column_metadata(schema_graph)
        results = _agg_change(intent, schema_graph, cm)
        assert len(results) > 0


class TestGroupbyAdd:
    """Tests for group-by add expansion."""

    def test_adds_groupby(self, schema_graph):
        """Adds groupable columns."""

        intent = _warmup_intent(
            tables=["orders"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id")),
            ],
        )
        cm = _build_column_metadata(schema_graph)
        results = _groupby_add(intent, schema_graph, cm)
        assert len(results) > 0
        for r in results:
            assert len(r.group_by_cols) >= 1
            assert r.grain == "grouped"


class TestFilterRemove:
    """Tests for filter removal expansion."""

    def test_no_filters_skips(self, schema_graph):
        """No filters returns empty."""
        intent = _warmup_intent(tables=["orders"], filters_param=[])
        cm = _build_column_metadata(schema_graph)
        results = _filter_remove(intent, schema_graph, cm)
        assert results == []

    def test_removes_each(self, schema_graph):
        """Removes each filter one at a time."""
        filters = [
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="=",
                value_type="string",
                param_key="p1",
            ),
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">",
                value_type="number",
                param_key="p2",
            ),
        ]
        intent = _warmup_intent(tables=["orders"], filters_param=filters)
        cm = _build_column_metadata(schema_graph)
        results = _filter_remove(intent, schema_graph, cm)
        assert len(results) == 2
        for r in results:
            assert len(r.filters_param) == 1


class TestJoinDimensionAdd:
    """Tests for dimension join expansion."""

    def test_joins_dimension(self, schema_graph):
        """Joins a connected dimension table."""
        intent = _warmup_intent(tables=["orders"])
        fk_map = _build_fk_map(schema_graph)
        cm = _build_column_metadata(schema_graph)
        results = _join_dimension_add(intent, schema_graph, fk_map, cm)
        for r in results:
            assert len(r.tables) > 1


class TestTableRemove:
    """Tests for table removal expansion."""

    def test_single_table_skips(self, schema_graph):
        """Single table intent returns empty."""
        intent = _warmup_intent(tables=["orders"])
        fk_map = _build_fk_map(schema_graph)
        cm = _build_column_metadata(schema_graph)
        results = _table_remove(intent, schema_graph, fk_map, cm)
        assert results == []


class TestIncludeGold:
    """Tests for include-gold expansion."""

    def test_returns_single_copy(self, schema_graph):
        """Returns a single-element list."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _include_gold(intent, schema_graph, cm)
        assert len(results) == 1

    def test_copy_has_metadata(self, schema_graph):
        """Copy is stamped with INCLUDE_GOLD metadata."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _include_gold(intent, schema_graph, cm)
        assert results[0].expansion_metadata.operator == Op.INCLUDE_GOLD


class TestTempExtractGroupby:
    """Tests for temporal EXTRACT groupby expansion."""

    def test_produces_extract_variants(self, schema_graph):
        """Temporal columns get EXTRACT variants."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        temporal = _get_temporal_columns(schema_graph, "orders")
        results = _temp_extract_groupby(intent, schema_graph, cm)
        if temporal:
            assert len(results) > 0
            for r in results:
                assert r.grain == "grouped"


class TestTempDateWindowFilter:
    """Tests for date_window filter expansion."""

    def test_produces_date_window_variants(self, schema_graph):
        """Temporal columns get date_window filter variants."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        temporal = _get_temporal_columns(schema_graph, "orders")
        results = _temp_date_window_filter(intent, schema_graph, cm)
        if temporal:
            assert len(results) > 0
            for r in results:
                added = [f for f in r.filters_param if f.value_type == "date_window"]
                assert len(added) == 1


class TestDistinctAdd:
    """Tests for DISTINCT add expansion."""

    def test_adds_distinct(self, schema_graph):
        """Produces one variant with distinct set."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _distinct_add(intent, schema_graph, cm)
        assert len(results) == 1
        assert results[0].distinct_select_index == 0


class TestLimitAdd:
    """Tests for LIMIT expansion."""

    def test_adds_limit_variants(self, schema_graph):
        """Produces variants with different LIMIT values."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _limit_add(intent, schema_graph, cm)
        assert len(results) == len(
            __import__("aetherdialect._config", fromlist=["SeedWarmupConfig"]).SeedWarmupConfig.LIMIT_EXPANSION_VALUES
        )
        for r in results:
            assert r.limit is not None

    def test_skips_when_limit_set(self, schema_graph):
        """Skips when limit already set."""
        intent = _warmup_intent(tables=["orders"], limit=10)
        cm = _build_column_metadata(schema_graph)
        results = _limit_add(intent, schema_graph, cm)
        assert results == []


class TestFilterOrGroup:
    """Tests for OR filter group expansion."""

    def test_creates_or_groups(self, schema_graph):
        """Creates OR groups from pairs of existing filters."""
        filters = [
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="=",
                value_type="string",
                param_key="p1",
            ),
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">",
                value_type="number",
                param_key="p2",
            ),
        ]
        intent = _warmup_intent(tables=["orders"], filters_param=filters)
        cm = _build_column_metadata(schema_graph)
        results = _filter_or_group(intent, schema_graph, cm)
        assert len(results) == 1
        or_filters = [f for f in results[0].filters_param if f.bool_op == "OR"]
        assert len(or_filters) == 2

    def test_needs_two_filters(self, schema_graph):
        """Returns empty when fewer than 2 filters."""
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        results = _filter_or_group(intent, schema_graph, cm)
        assert results == []


class TestExpandGoldIntents:
    """Tests for expand_gold_intents multi-depth."""

    def test_returns_list(self, schema_graph):
        """Returns a list."""
        gold = [_warmup_intent(tables=["orders"])]
        results = expand_gold_intents(gold, schema_graph, max_depth=1)
        assert isinstance(results, list)

    def test_cte_intents_included(self, schema_graph):
        """CTE intents are expanded (not skipped)."""
        cte_step = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr.from_column("orders.order_id"),
                )
            ],
            group_by_cols=[],
            filters_param=[],
            having_param=[],
            order_by_cols=[],
        )
        gold = [_warmup_intent(tables=["orders"], cte_steps=[cte_step])]
        results = expand_gold_intents(gold, schema_graph, max_depth=1)
        assert len(results) > 0

    def test_deduplicates(self, schema_graph):
        """No duplicate intent keys in results."""
        gold = [_warmup_intent(tables=["orders"])]
        results = expand_gold_intents(gold, schema_graph, max_depth=1)
        keys = [intent_key(r.to_runtime_intent()) for r in results]
        assert len(keys) == len(set(keys))

    def test_depth_2_produces_more(self, schema_graph):
        """Depth 2 produces at least as many variants as depth 1."""
        gold = [_warmup_intent(tables=["orders"])]
        d1 = expand_gold_intents(gold, schema_graph, max_depth=1)
        d2 = expand_gold_intents(gold, schema_graph, max_depth=2)
        assert len(d2) >= len(d1)


class TestWindowExpansionOperators:
    """Window rank and partitioned sum expansions."""

    def test_window_rank_adds_row_number(self, schema_graph):
        gold = _warmup_intent(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
            ],
        )
        meta = _build_column_metadata(schema_graph)
        out = _window_rank_add(gold, schema_graph, meta)
        assert len(out) == 1
        win_steps = out[0].window_registry or []
        assert len(win_steps) == 1
        assert win_steps[0].window_spec.function == "row_number"

    def test_window_sum_partition_adds_sum_over(self, schema_graph):
        gold = _warmup_intent(
            grain="row_level",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        )
        meta = _build_column_metadata(schema_graph)
        out = _window_sum_partition_add(gold, schema_graph, meta)
        assert len(out) >= 1
        win_steps = out[0].window_registry or []
        assert win_steps
        assert win_steps[0].window_spec.function == "sum"


class TestBuildOperatorRegistry:
    """Registry completeness and smoke coverage for every operator id."""

    def test_registry_keys_match_expansion_operator_id_constants(self, schema_graph):
        """Every `Op` string constant has exactly one registry entry and vice versa."""
        cm = _build_column_metadata(schema_graph)
        fk = _build_fk_map(schema_graph)
        reg = _build_operator_registry(cm, fk)
        assert set(reg.keys()) == _all_operator_id_strings()

    def test_each_operator_returns_list_of_seed_warmup_intents(self, schema_graph):
        """No registered operator raises on a minimal orders intent."""
        cm = _build_column_metadata(schema_graph)
        fk = _build_fk_map(schema_graph)
        reg = _build_operator_registry(cm, fk)
        base = _warmup_intent(tables=["orders"])
        for op_id, fn in reg.items():
            out = fn(base, schema_graph)
            assert isinstance(out, list), op_id
            for item in out:
                assert isinstance(item, SeedWarmupIntent), op_id


class TestFilterExprAdd:
    """Column-vs-column filter expansion."""

    def test_produces_expr_filters_for_same_dtype_pairs(self, schema_graph):
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        out = _filter_expr_add(intent, schema_graph, cm)
        assert len(out) > 0
        assert any(f.right_expr is not None for r in out for f in r.filters_param)


class TestOrderbyAdd:
    """ORDER BY expansion."""

    def test_adds_order_variants(self, schema_graph):
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        out = _orderby_add(intent, schema_graph, cm)
        assert len(out) > 0
        for r in out:
            assert len(r.order_by_cols) >= 1


class TestHavingExpansions:
    """HAVING value and expression expansions."""

    def test_having_value_add_requires_grouped(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _having_value_add(intent, schema_graph, cm)
        assert len(out) > 0

    def test_having_expr_add_requires_grouped_with_agg_select(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _having_expr_add(intent, schema_graph, cm)
        assert len(out) > 0
        assert any(h.right_expr is not None for r in out for h in r.having_param)


class TestGroupbyRemove:
    """GROUP BY removal when multiple keys."""

    def test_removes_one_group_column(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            grain="grouped",
            group_by_cols=[
                NormalizedExpr.from_column("orders.customer_id"),
                NormalizedExpr.from_column("orders.status"),
            ],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _groupby_remove(intent, schema_graph, cm)
        assert len(out) == 2
        for r in out:
            assert len(r.group_by_cols) == 1


class TestHavingRemove:
    """HAVING removal."""

    def test_removes_each_having(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr.from_agg("sum", "orders.amount"),
                    op=">",
                    value_type="number",
                    param_key="h1",
                ),
                HavingParam(
                    left_expr=NormalizedExpr.from_agg("count", "*"),
                    op=">=",
                    value_type="number",
                    param_key="h2",
                ),
            ],
        )
        cm = _build_column_metadata(schema_graph)
        out = _having_remove(intent, schema_graph, cm)
        assert len(out) == 2


class TestJoinFactAdd:
    """Fact-table join from a dimension-only intent."""

    def test_adds_fact_from_dimension_parent(self, schema_graph):
        intent = _warmup_intent(tables=["customers"])
        fk = _build_fk_map(schema_graph)
        cm = _build_column_metadata(schema_graph)
        out = _join_fact_add(intent, schema_graph, fk, cm)
        assert len(out) >= 1
        assert any("orders" in r.tables for r in out)


class TestDimensionSwap:
    """Swap one dimension for another reachable from the fact."""

    def test_swaps_dimension_when_fact_present(self, schema_graph):
        intent = _warmup_intent(tables=["orders", "customers"])
        fk = _build_fk_map(schema_graph)
        cm = _build_column_metadata(schema_graph)
        out = _dimension_swap(intent, schema_graph, fk, cm)
        if out:
            for r in out:
                assert "orders" in r.tables
                assert len(r.tables) == 2


class TestBridgeIntermediateAdd:
    """Bridge table expansion."""

    def test_returns_list(self, schema_graph):
        intent = _warmup_intent(tables=["orders", "products"])
        fk = _build_fk_map(schema_graph)
        cm = _build_column_metadata(schema_graph)
        out = _bridge_intermediate_add(intent, schema_graph, fk, cm)
        assert isinstance(out, list)


class TestTempDateTruncAndDiff:
    """Temporal trunc and date_diff expansions."""

    def test_date_trunc_groupby_when_temporal(self, schema_graph):
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        temporal = _get_temporal_columns(schema_graph, "orders")
        out = _temp_date_trunc_groupby(intent, schema_graph, cm)
        if temporal:
            assert len(out) > 0

    def test_date_diff_when_temporal(self, schema_graph):
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        temporal = _get_temporal_columns(schema_graph, "orders")
        out = _temp_date_diff_filter(intent, schema_graph, cm)
        if temporal:
            assert len(out) > 0
            for r in out:
                assert any(f.value_type == "date_diff" for f in r.filters_param)


class TestNumericSelectAndFilterOps:
    """NUM_ROUND_SELECT and NUM_ABS_FILTER."""

    def test_num_round_wraps_measure(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _num_round_select(intent, schema_graph, cm)
        assert len(out) == 1
        assert out[0].select_cols[0].expr.scalar_func == "round"

    def test_num_abs_on_range_filter(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op=">=",
                    value_type="number",
                    param_key="p1",
                ),
            ],
        )
        cm = _build_column_metadata(schema_graph)
        out = _num_abs_filter(intent, schema_graph, cm)
        assert len(out) == 1
        assert out[0].filters_param[0].left_expr.scalar_func == "abs"


class TestSelectExprPairMultiply:
    """Multiply pair on numeric columns."""

    def test_adds_composed_select(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders", "products"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _select_expr_pair_multiply(intent, schema_graph, cm)
        assert len(out) > 0
        assert any(len(sc.expr.add_groups) > 0 for r in out for sc in r.select_cols if sc.expr.add_groups)


class TestSelectCaseLabelAdd:
    """CASE label in select list."""

    def test_adds_case_column(self, schema_graph):
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        out = _select_case_label_add(intent, schema_graph, cm)
        assert len(out) > 0
        for r in out:
            case_cols = [sc for sc in r.select_cols if (expr_registry_ref(sc.expr) or "").startswith("c")]
            assert len(case_cols) >= 1


class TestWindowLagLead:
    """LAG / LEAD window expansions."""

    def test_lag_adds_window_column(self, schema_graph):
        intent = _warmup_intent(tables=["orders"], grain="row_level")
        cm = _build_column_metadata(schema_graph)
        out = _window_lag_add(intent, schema_graph, cm)
        assert len(out) >= 1
        assert any(s.window_spec.function == "lag" for s in (out[0].window_registry or []))

    def test_lead_derived_from_lag(self, schema_graph):
        intent = _warmup_intent(tables=["orders"], grain="row_level")
        cm = _build_column_metadata(schema_graph)
        out = _window_lead_add(intent, schema_graph, cm)
        assert isinstance(out, list)
        if out:
            assert any(s.window_spec.function == "lead" for s in (out[0].window_registry or []))


class TestFilterIlikeAdd:
    """PostgreSQL ILIKE expansion."""

    def test_ilike_on_categorical_string_when_postgresql(self, schema_graph):
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "postgresql"
            intent = _warmup_intent(tables=["orders"])
            cm = _build_column_metadata(schema_graph)
            out = _filter_ilike_add(intent, schema_graph, cm)
            assert any(f.op == "ilike" for r in out for f in r.filters_param)
        finally:
            EngineConfig.TYPE = orig

    def test_empty_when_not_postgresql(self, schema_graph):
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            intent = _warmup_intent(tables=["orders"])
            cm = _build_column_metadata(schema_graph)
            assert _filter_ilike_add(intent, schema_graph, cm) == []
        finally:
            EngineConfig.TYPE = orig


class TestFilterArrayContainsAdd:
    """Array contains filter when column has element_type."""

    def test_adds_contains_filter(self):
        tags = TableMetadata(
            name="tags",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                    row_count=10,
                ),
                "labels": ColumnMetadata(
                    name="labels",
                    data_type="text[]",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    element_type="text",
                    distinct_count=5,
                    distinct_ratio=0.5,
                    row_count=10,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role=TableRole.DIMENSION.value,
            row_count=10,
        )
        sg = SchemaGraph(
            tables={"tags": tags},
            join_paths_multi={},
            effective_structural_hash="arr_test",
        )
        intent = _warmup_intent(tables=["tags"])
        cm = _build_column_metadata(sg)
        out = _filter_array_contains_add(intent, sg, cm)
        assert len(out) == 1
        assert out[0].filters_param[-1].value_type == "array"
        assert out[0].filters_param[-1].op == "contains"


class TestOrderbyLimitRemove:
    """Strip ORDER BY / LIMIT."""

    def test_orderby_remove_drops_last(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            order_by_cols=[
                OrderByCol(expr=NormalizedExpr.from_column("orders.order_id"), direction="ASC"),
            ],
        )
        cm = _build_column_metadata(schema_graph)
        out = _orderby_remove(intent, schema_graph, cm)
        assert len(out) == 1
        assert out[0].order_by_cols == []

    def test_limit_remove_clears_limit(self, schema_graph):
        intent = _warmup_intent(tables=["orders"], limit=10)
        cm = _build_column_metadata(schema_graph)
        out = _limit_remove(intent, schema_graph, cm)
        assert len(out) == 1
        assert out[0].limit is None


class TestSelectColTrim:
    """Drop one non-aggregated select column."""

    def test_trims_when_two_plain_columns(self, schema_graph):
        intent = _warmup_intent(
            tables=["orders"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("orders.amount")),
            ],
        )
        cm = _build_column_metadata(schema_graph)
        out = _select_col_trim(intent, schema_graph, cm)
        assert len(out) == 2
        for r in out:
            assert len(r.select_cols) == 1


class TestWindowStrip:
    """Remove pure window columns."""

    def test_strips_window_only_column(self, schema_graph):
        base = _warmup_intent(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
            ],
        )
        cm = _build_column_metadata(schema_graph)
        ranked = _window_rank_add(base, schema_graph, cm)
        assert ranked
        out = _window_strip(ranked[0], schema_graph, cm)
        assert len(out) == 1
        assert not (out[0].window_registry or [])


class TestDistinctRemove:
    """Turn off DISTINCT."""

    def test_clears_distinct_flag(self, schema_graph):
        intent = _warmup_intent(tables=["orders"], distinct_select_index=0)
        cm = _build_column_metadata(schema_graph)
        out = _distinct_remove(intent, schema_graph, cm)
        assert len(out) == 1
        assert out[0].distinct_select_index == -1


class TestDeterministicRepairWarmupSeed:
    """Warmup repair syncs canonical registry ids back onto the seed intent."""

    def test_deterministic_repair_warmup_seed_propagates_registries(self, schema_graph):
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.amount"), direction="DESC")],
        )
        branch = CaseWhenBranch(
            condition=FilterParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">",
                value_type="number",
                param_key="k",
            ),
            result=NormalizedExpr(add_values=[ExprValue(value=1.0)]),
        )
        cw = CaseWhenExpr(
            branches=[branch],
            else_result=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
        )
        seed = SeedWarmupIntent(
            intent_id="reg_seed",
            tables=["orders"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.amount")),
                SelectCol(expr=NormalizedExpr.from_column("w99")),
                SelectCol(expr=NormalizedExpr.from_column("c99")),
            ],
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            window_registry=[WindowRegistryStep(registry_id="w99", window_spec=ws)],
            case_registry=[CaseRegistryStep(registry_id="c99", case_when=cw)],
        )
        out = _deterministic_repair_warmup_seed(seed, schema_graph)
        assert out.window_registry[0].registry_id == "w01"
        assert out.case_registry[0].registry_id == "c01"


class TestSeedWarmupDistinctRuntime:
    """DISTINCT index propagation on seed and runtime intents."""

    def test_seed_warmup_intent_to_runtime_propagates_distinct(self):
        si = SeedWarmupIntent(
            intent_id="d1",
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.x"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            distinct_select_index=0,
        )
        rt = si.to_runtime_intent()
        assert rt.distinct_select_index == 0


class TestTableRemovePruning:
    """TABLE_REMOVE prunes dependent clauses."""

    def test_table_remove_prunes_having_referencing_dropped_table(self, schema_graph):
        intent = SeedWarmupIntent(
            intent_id="tr1",
            tables=["orders", "customers"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            order_by_cols=[],
            filters_param=[],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr.from_agg("count", "customers.customer_id"),
                    op=">",
                    value_type="number",
                    param_key="h1",
                )
            ],
        )
        cm = _build_column_metadata(schema_graph)
        fk = _build_fk_map(schema_graph)
        out = _table_remove(intent, schema_graph, fk, cm)
        assert out
        for v in out:
            assert not any("customers" in h.left_expr.primary_column for h in (v.having_param or []))

    def test_table_remove_prunes_filter_right_expr_referencing_dropped_table(self, schema_graph):
        intent = SeedWarmupIntent(
            intent_id="tr2",
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.order_id"),
                    op="=",
                    value_type="string",
                    param_key="a",
                    right_expr=NormalizedExpr.from_column("customers.customer_id"),
                )
            ],
            having_param=[],
        )
        cm = _build_column_metadata(schema_graph)
        fk = _build_fk_map(schema_graph)
        out = _table_remove(intent, schema_graph, fk, cm)
        assert out
        for v in out:
            for fp in v.filters_param or []:
                if fp.right_expr:
                    assert _table_from_column_ref(fp.right_expr.primary_column) != "customers"

    def test_table_remove_prunes_window_registry_partition_referencing_dropped_table(self, schema_graph):
        ws = WindowSpec(
            function="sum",
            partition_by=[NormalizedExpr.from_column("customers.customer_id")],
            order_by=[],
            argument=NormalizedExpr.from_column("orders.amount"),
        )
        intent = SeedWarmupIntent(
            intent_id="tr3",
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("w01")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            window_registry=[WindowRegistryStep(registry_id="w01", window_spec=ws)],
        )
        cm = _build_column_metadata(schema_graph)
        fk = _build_fk_map(schema_graph)
        out = _table_remove(intent, schema_graph, fk, cm)
        assert out
        for v in out:
            assert not (v.window_registry or [])
            assert not any((expr_registry_ref(sc.expr) or "").startswith("w") for sc in (v.select_cols or []))

    def test_table_remove_prunes_case_registry_branch_referencing_dropped_table(self, schema_graph):
        branch = CaseWhenBranch(
            condition=FilterParam(
                left_expr=NormalizedExpr.from_column("customers.customer_id"),
                op=">",
                value_type="number",
                param_key="c",
            ),
            result=NormalizedExpr(add_values=[ExprValue(value=1.0)]),
        )
        cw = CaseWhenExpr(
            branches=[branch],
            else_result=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
        )
        intent = SeedWarmupIntent(
            intent_id="tr4",
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("c01")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            case_registry=[CaseRegistryStep(registry_id="c01", case_when=cw)],
        )
        cm = _build_column_metadata(schema_graph)
        fk = _build_fk_map(schema_graph)
        out = _table_remove(intent, schema_graph, fk, cm)
        assert out
        for v in out:
            assert not (v.case_registry or [])
            assert not any((expr_registry_ref(sc.expr) or "").startswith("c") for sc in (v.select_cols or []))


def _table_from_column_ref(col_ref: str) -> str:
    if not col_ref or "." not in col_ref:
        return ""
    return col_ref.split(".", 1)[0]


class TestDimensionSwapRewrite:
    """DIMENSION_SWAP rewrites qualifiers when bare columns match."""

    def _two_dim_schema(self) -> SchemaGraph:
        dim_a = TableMetadata(
            name="dim_a",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                    row_count=10,
                ),
                "shared": ColumnMetadata(
                    name="shared",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                    row_count=10,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role=TableRole.DIMENSION.value,
            row_count=10,
        )
        dim_b = TableMetadata(
            name="dim_b",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                    row_count=10,
                ),
                "shared": ColumnMetadata(
                    name="shared",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                    row_count=10,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role=TableRole.DIMENSION.value,
            row_count=10,
        )
        fact = TableMetadata(
            name="fact",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=100,
                    distinct_ratio=1.0,
                    row_count=100,
                ),
                "a_id": ColumnMetadata(
                    name="a_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("dim_a", "id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=0.1,
                    row_count=100,
                ),
                "b_id": ColumnMetadata(
                    name="b_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("dim_b", "id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=0.1,
                    row_count=100,
                ),
            },
            primary_key=["id"],
            foreign_keys=[
                FKEdge(
                    src_table="fact",
                    src_cols=["a_id"],
                    dst_table="dim_a",
                    dst_cols=["id"],
                ),
                FKEdge(
                    src_table="fact",
                    src_cols=["b_id"],
                    dst_table="dim_b",
                    dst_cols=["id"],
                ),
            ],
            role=TableRole.FACT.value,
            row_count=100,
        )
        sg = SchemaGraph(
            tables={"dim_a": dim_a, "dim_b": dim_b, "fact": fact},
            join_paths_multi={
                "fact": {
                    "dim_a": [[{"src": "fact.a_id", "dst": "dim_a.id"}]],
                    "dim_b": [[{"src": "fact.b_id", "dst": "dim_b.id"}]],
                },
                "dim_a": {"fact": [[{"src": "fact.a_id", "dst": "dim_a.id"}]]},
                "dim_b": {"fact": [[{"src": "fact.b_id", "dst": "dim_b.id"}]]},
            },
            effective_structural_hash="swap_test",
        )
        assign_column_ops(sg)
        return sg

    def test_dimension_swap_rewrites_qualified_select_refs(self):
        sg = self._two_dim_schema()
        intent = SeedWarmupIntent(
            intent_id="ds1",
            tables=["fact", "dim_a"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("dim_a.shared"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        cm = _build_column_metadata(sg)
        fk = _build_fk_map(sg)
        out = _dimension_swap(intent, sg, fk, cm)
        assert out
        assert any(sc.expr.primary_column == "dim_b.shared" for v in out for sc in v.select_cols)

    def test_dimension_swap_skips_when_new_dim_missing_column(self):
        dim_only = TableMetadata(
            name="dim_only",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=1.0,
                    row_count=5,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role=TableRole.DIMENSION.value,
            row_count=5,
        )
        sg = self._two_dim_schema()
        sg.tables["dim_only"] = dim_only
        sg.join_paths_multi.setdefault("fact", {})["dim_only"] = [[{"src": "fact.id", "dst": "dim_only.id"}]]
        sg.join_paths_multi.setdefault("dim_only", {})["fact"] = [[{"src": "fact.id", "dst": "dim_only.id"}]]
        intent = SeedWarmupIntent(
            intent_id="ds2",
            tables=["fact", "dim_a"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("dim_a.shared"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        cm = _build_column_metadata(sg)
        fk = _build_fk_map(sg)
        out = _dimension_swap(intent, sg, fk, cm)
        swap_only = [v for v in out if "dim_only" in (v.tables or [])]
        assert not swap_only


class TestWindowStripOrderBy:
    """WINDOW_STRIP removes order-by rows tied to stripped window ids."""

    def test_window_strip_drops_order_by_referencing_stripped_registry(self, schema_graph):
        base = _warmup_intent(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        ranked = _window_rank_add(base, schema_graph, cm)
        assert ranked
        w0 = ranked[0].window_registry[0].registry_id
        ob_intent = replace(
            ranked[0],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column(w0), direction="ASC")],
        )
        out = _window_strip(ob_intent, schema_graph, cm)
        assert len(out) == 1
        for ob in out[0].order_by_cols or []:
            assert (expr_registry_ref(ob.expr) or "") != w0


class TestHavingStarTargets:
    """HAVING expansions avoid star targets for non-count aggregates."""

    def test_having_value_add_uses_numeric_measure_for_sum_avg(self, schema_graph):
        intent = _warmup_intent(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _having_value_add(intent, schema_graph, cm)
        for v in out:
            for hp in v.having_param or []:
                term = hp.left_expr.primary_term.lower()
                if "sum(" in term or "avg(" in term or "min(" in term or "max(" in term:
                    assert "*" not in term

    def test_having_expr_add_dedupe_uses_left_term_and_op(self, schema_graph):
        intent = _warmup_intent(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        out = _having_expr_add(intent, schema_graph, cm)
        pairs = [(h.left_expr.primary_term, h.op) for v in out for h in v.having_param]
        assert len(pairs) == len(set(pairs))


class TestFilterAddValueTypes:
    """FILTER_ADD maps semantic filter types from column metadata."""

    def test_filter_add_assigns_value_type_from_column(self, schema_graph):
        intent = _warmup_intent(tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        out = _filter_add(intent, schema_graph, cm)
        by_col = {f.left_expr.primary_column: f for v in out for f in v.filters_param}
        assert by_col["orders.customer_id"].value_type == "number"
        assert by_col["orders.status"].value_type == "string"
        assert by_col["orders.order_date"].value_type == "date"


class TestRemoveOperatorsDeepcopy:
    """REMOVE operators must not alias mutable rows from the source intent."""

    def test_remove_operators_do_not_alias_original_intent_objects(self, schema_graph):
        filters = [
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="=",
                value_type="string",
                param_key="p1",
            ),
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">",
                value_type="number",
                param_key="p2",
            ),
        ]
        intent = _warmup_intent(tables=["orders"], filters_param=filters)
        cm = _build_column_metadata(schema_graph)
        fr = _filter_remove(intent, schema_graph, cm)
        assert fr
        kept_idx = 1 if fr[0].filters_param[0].left_expr.primary_column == "orders.status" else 0
        fr[0].filters_param[kept_idx].op = "FAKE"
        assert intent.filters_param[kept_idx].op != "FAKE"

        gb_intent = _warmup_intent(
            grain="grouped",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            group_by_cols=[
                NormalizedExpr.from_column("orders.customer_id"),
                NormalizedExpr.from_column("orders.status"),
            ],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        gr = _groupby_remove(gb_intent, schema_graph, cm)
        assert gr
        gr[0].group_by_cols[0].column_ref = "mutated"
        assert gb_intent.group_by_cols[0].column_ref != "mutated"

        hv = [
            HavingParam(
                left_expr=NormalizedExpr.from_agg("count", "*"),
                op=">",
                value_type="number",
                param_key="h1",
            ),
            HavingParam(
                left_expr=NormalizedExpr.from_agg("count", "*"),
                op="<",
                value_type="number",
                param_key="h2",
            ),
        ]
        hint = replace(gb_intent, having_param=hv)
        hr = _having_remove(hint, schema_graph, cm)
        assert hr
        hr[0].having_param[0].op = "FAKE"
        assert hint.having_param[0].op != "FAKE"


class TestScalarGrainSkips:
    """Group-by temporal operators skip scalar grain."""

    @pytest.mark.parametrize(
        "op_fn",
        [_groupby_add, _temp_extract_groupby, _temp_date_trunc_groupby],
    )
    def test_group_add_operators_skip_scalar_grain(self, schema_graph, op_fn):
        intent = _warmup_intent(grain="scalar", tables=["orders"])
        cm = _build_column_metadata(schema_graph)
        assert op_fn(intent, schema_graph, cm) == []


class TestFilterExprAddPairing:
    """FILTER_EXPR_ADD pairs compatible columns only."""

    def test_filter_expr_add_pairs_only_same_value_type_and_role(self, schema_graph):
        intent = _warmup_intent(tables=["orders", "customers"])
        cm = _build_column_metadata(schema_graph)
        out = _filter_expr_add(intent, schema_graph, cm)
        for v in out:
            for fp in v.filters_param:
                if not fp.right_expr:
                    continue
                lt = fp.left_expr.primary_column
                rt = fp.right_expr.primary_column
                tl, cl = lt.split(".", 1)
                tr, cr = rt.split(".", 1)
                assert cm[tl][cl]["value_type"] == cm[tr][cr]["value_type"]
                assert cm[tl][cl]["role"] == cm[tr][cr]["role"]

    def test_filter_expr_add_pairs_fks_only_when_target_matches(self):
        dim = TableMetadata(
            name="dimt",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=1.0,
                    row_count=5,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role=TableRole.DIMENSION.value,
            row_count=5,
        )
        fact = TableMetadata(
            name="factt",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                    row_count=10,
                ),
                "u1": ColumnMetadata(
                    name="u1",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("dimt", "id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=0.5,
                    row_count=10,
                ),
                "u2": ColumnMetadata(
                    name="u2",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("dimt", "id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=0.5,
                    row_count=10,
                ),
                "z1": ColumnMetadata(
                    name="z1",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("dimt", "id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=0.5,
                    row_count=10,
                ),
                "z2": ColumnMetadata(
                    name="z2",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("other", "id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=0.5,
                    row_count=10,
                ),
            },
            primary_key=["id"],
            foreign_keys=[
                FKEdge(
                    src_table="factt",
                    src_cols=["u1"],
                    dst_table="dimt",
                    dst_cols=["id"],
                ),
                FKEdge(
                    src_table="factt",
                    src_cols=["u2"],
                    dst_table="dimt",
                    dst_cols=["id"],
                ),
                FKEdge(
                    src_table="factt",
                    src_cols=["z1"],
                    dst_table="dimt",
                    dst_cols=["id"],
                ),
                FKEdge(
                    src_table="factt",
                    src_cols=["z2"],
                    dst_table="other",
                    dst_cols=["id"],
                ),
            ],
            role=TableRole.FACT.value,
            row_count=10,
        )
        other = TableMetadata(
            name="other",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=1.0,
                    row_count=5,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            role=TableRole.DIMENSION.value,
            row_count=5,
        )
        sg = SchemaGraph(
            tables={"dimt": dim, "factt": fact, "other": other},
            join_paths_multi={},
            effective_structural_hash="fkpair",
        )
        assign_column_ops(sg)
        cm = _build_column_metadata(sg)
        intent = SeedWarmupIntent(
            intent_id="fkp",
            tables=["factt", "dimt", "other"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("factt.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        out = _filter_expr_add(intent, sg, cm)
        pairs = {
            (f.left_expr.primary_column, f.right_expr.primary_column)
            for v in out
            for f in v.filters_param
            if f.right_expr
        }
        assert ("factt.u1", "factt.u2") in pairs or ("factt.u2", "factt.u1") in pairs
        assert all("z2" not in p[0] and "z2" not in p[1] for p in pairs if "z1" in p[0] or "z1" in p[1])


class TestExpandSingleDepthPostRepair:
    """Post-repair qualified-ref validation."""

    def test_expand_single_depth_drops_variants_with_post_repair_orphan_refs(self, schema_graph):
        real = _deterministic_repair_warmup_seed

        def bad_repair(seed: SeedWarmupIntent, schema: SchemaGraph) -> SeedWarmupIntent:
            base = real(seed, schema)
            if seed.intent_id != "orph_marker":
                return base
            return replace(
                base,
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.typo_col"))],
            )

        def marker_op(i: SeedWarmupIntent, _s: SchemaGraph) -> list[SeedWarmupIntent]:
            v = copy.deepcopy(i)
            v.intent_id = "orph_marker"
            return [v]

        seen: set[str] = set()
        with mock.patch(
            "aetherdialect._expansion_ops._deterministic_repair_warmup_seed",
            side_effect=bad_repair,
        ):
            out = _expand_single_depth(
                [_warmup_intent(tables=["orders"])],
                schema_graph,
                {"m": marker_op},
                seen,
                "depth1",
            )
        assert not any(v.intent_id == "orph_marker" for v in out)


class TestFilterOrGroupUniqueId:
    """FILTER_OR_GROUP allocates a fresh filter_group id."""

    def test_filter_or_group_uses_unique_group_id(self, schema_graph):
        filters = [
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="=",
                value_type="string",
                param_key="p1",
                filter_group=1,
            ),
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">",
                value_type="number",
                param_key="p2",
            ),
        ]
        intent = _warmup_intent(tables=["orders"], filters_param=filters)
        cm = _build_column_metadata(schema_graph)
        results = _filter_or_group(intent, schema_graph, cm)
        assert results
        or_filters = [f for f in results[0].filters_param if f.bool_op == "OR"]
        assert {f.filter_group for f in or_filters} == {2}


class TestSelectGroupedSkips:
    """SELECT expansions that would break grouped grain are skipped."""

    @pytest.mark.parametrize("op_fn", [_select_expr_pair_multiply, _select_case_label_add])
    def test_select_grouped_skipping_operators_skip_grouped_grain(self, schema_graph, op_fn):
        intent = _warmup_intent(
            grain="grouped",
            tables=["orders"],
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
        )
        cm = _build_column_metadata(schema_graph)
        assert op_fn(intent, schema_graph, cm) == []


class TestFinalizeRegistryTouchSeed:
    """``_finalize_registry_touch_seed`` rejects dangling registry references after repairs."""

    def test_returns_none_when_select_refs_unknown_window_registry_id(self, schema_graph):
        ws = WindowSpec(function="row_number", partition_by=[], order_by=[])
        intent = _warmup_intent(
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("w99"))],
            window_registry=[WindowRegistryStep(registry_id="w01", window_spec=ws)],
        )
        assert _finalize_registry_touch_seed(intent, schema_graph) is None
