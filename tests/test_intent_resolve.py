"""Tests for intent_resolve module: normalization, sorting, simplification."""

import pytest

from aetherdialect._contracts_base import (
    ColumnRole,
    ExprValue,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    WhereParam,
    having_leaves,
    predicate_group_from_list,
    where_leaves,
)
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    CteOutputColumnMeta,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._intent_expr import (
    parse_expr_string,
    promote_date_subtraction_to_date_diff,
    replace_refs_in_expr,
)
from aetherdialect._intent_resolve import (
    _canonicalize_condition_order,
    _dedup_having,
    _dedup_where_predicates,
    _forward_links,
    _having_structural_key,
    _is_cte_output_groupable,
    _normalize_agg_to_agg_having,
    _normalize_col_to_col_where,
    _normalize_having_canonical,
    _normalize_where_canonical,
    _normalize_where_scalar_on_left,
    _shift_multi_group_representative_having_forward,
    _shift_multi_group_representative_where_forward,
    _simplify_expr,
    _where_structural_key,
    check_qualified_refs_exist,
    enforce_cte_grain_consistency,
    enforce_grain_consistency,
    normalize_count_star,
    normalize_cte_names,
    normalize_where_havings,
    qualify_count_star_mulgroups,
    resolve_column_map,
    resolve_cte_column_maps,
    resolve_window_registry_where_rhs,
    rewrite_cte_output_refs_to_aliases,
    rewrite_main_query_refs_to_final_cte_columns,
    simplify_exprs,
    sort_having,
    sort_select_cols,
    sort_where_predicates,
)
from aetherdialect._sql_gen import (
    _render_predicate_group_sql,
)
from tests.conftest import term_strs


class TestReplaceRefsInExpr:
    """Tests for replace_refs_in_expr."""

    def test_replaces_terms(self):
        """replace_refs_in_expr applies replacer to multiply/divide terms."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["old_table.col"])],
        )
        result = replace_refs_in_expr(expr, lambda s: s.replace("old_table", "new_table"))
        assert term_strs(result.add_groups[0].multiply) == ["new_table.col"]

    def test_preserves_add_values_and_sub_values(self):
        """replace_refs_in_expr does not modify add_values or sub_values."""
        from aetherdialect._contracts_base import ExprValue

        expr = NormalizedExpr(
            add_groups=[],
            sub_groups=[],
            add_values=[ExprValue(value=1.0)],
            sub_values=[ExprValue(value=0.5)],
        )
        result = replace_refs_in_expr(expr, lambda s: s.upper())
        assert result.add_values == expr.add_values
        assert result.sub_values == expr.sub_values

    def test_empty_groups_returns_empty_groups(self):
        """replace_refs_in_expr with no groups returns empty add/sub groups."""
        expr = NormalizedExpr(add_groups=[], sub_groups=[])
        result = replace_refs_in_expr(expr, lambda s: s)
        assert result.add_groups == []
        assert result.sub_groups == []


class TestNormalizeCountStarInIntent:
    """Tests for normalize_count_star."""

    def test_converts_count_1_to_count_star(self):
        """normalize_count_star converts COUNT(1) to COUNT(*)."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = normalize_count_star(intent)
        assert term_strs(result.select_cols[0].expr.add_groups[0].multiply) == ["COUNT(*)"]

    def test_no_count_1_unchanged(self):
        """normalize_count_star leaves intent unchanged when no COUNT(1)."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.col"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = normalize_count_star(intent)
        assert result.select_cols[0].expr.primary_column == "t.col"

    def test_cte_steps_normalized(self):
        """normalize_count_star converts COUNT(1) in CTE select_cols."""
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["t"],
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_count_star(intent)
        assert term_strs(result.cte_steps[0].select_cols[0].expr.add_groups[0].multiply) == ["COUNT(*)"]


class TestQualifyCountStarMulgroups:
    """Tests for qualify_count_star_mulgroups."""

    def test_rewrites_count_star_when_cte_reads_one_base_table(self):
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "rental": TableMetadata(
                    name="rental",
                    columns={
                        "rental_id": ColumnMetadata(
                            name="rental_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="rental_id",
                )
            },
        )
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["rental"],
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(agg_func="count", multiply=["*"])]))],
            group_by_cols=[NormalizedExpr.from_column("rental.customer_id")],
            grain="grouped",
        )
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        out = qualify_count_star_mulgroups(intent, schema)
        mul = out.cte_steps[0].select_cols[0].expr.add_groups[0]
        assert term_strs(mul.multiply) == ["rental.rental_id"]

    def test_rewrites_main_single_table_select_and_window(self):
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "rental": TableMetadata(
                    name="rental",
                    columns={
                        "rental_id": ColumnMetadata(
                            name="rental_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="rental_id",
                )
            },
        )
        intent = RuntimeIntent(
            tables=["rental"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(agg_func="count", multiply=["*"])]))],
            group_by_cols=[NormalizedExpr.from_column("rental.customer_id")],
            order_by_cols=[],
            where=None,
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=WindowSpec(
                        function="rank",
                        order_by=[
                            OrderByCol(expr=NormalizedExpr(add_groups=[MulGroup(agg_func="count", multiply=["*"])]))
                        ],
                    ),
                )
            ],
        )
        out = qualify_count_star_mulgroups(intent, schema)
        assert term_strs(out.select_cols[0].expr.add_groups[0].multiply) == ["rental.rental_id"]
        assert term_strs(out.window_registry[0].window_spec.order_by[0].expr.add_groups[0].multiply) == [
            "rental.rental_id"
        ]

    def test_leaves_multi_table_count_star_unchanged(self):
        from aetherdialect._intent_repair import reconcile_tables

        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "category": TableMetadata(
                    name="category",
                    columns={
                        "name": ColumnMetadata(name="name", data_type="text", value_type="string"),
                        "category_id": ColumnMetadata(
                            name="category_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="category_id",
                ),
                "rental": TableMetadata(
                    name="rental",
                    columns={
                        "rental_id": ColumnMetadata(
                            name="rental_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="rental_id",
                ),
            },
        )
        intent = RuntimeIntent(
            tables=["category", "rental"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("category.name")),
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(agg_func="count", multiply=["*"])])),
            ],
            group_by_cols=[NormalizedExpr.from_column("category.name")],
            order_by_cols=[],
            where=None,
        )
        out = qualify_count_star_mulgroups(intent, schema)
        assert term_strs(out.select_cols[1].expr.add_groups[0].multiply) == ["*"]
        qualified = RuntimeIntent(
            tables=["category", "rental"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("category.name")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "rental.rental_id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("category.name")],
            order_by_cols=[],
            where=None,
        )
        reconciled = reconcile_tables(qualified)
        assert "rental" in reconciled.tables
        assert "category" in reconciled.tables


class TestSortFunctions:
    """Tests for sort_select_cols, sort_where_predicates, sort_having."""

    def test_sort_select_cols_non_agg_first(self):
        """sort_select_cols places non-aggregated columns before aggregated."""
        cols = [
            SelectCol(expr=NormalizedExpr.from_agg("sum", "t.amount")),
            SelectCol(expr=NormalizedExpr.from_column("t.name")),
        ]
        sorted_cols = sort_select_cols(cols)
        assert sorted_cols[0].is_aggregated is False
        assert sorted_cols[1].is_aggregated is True

    def test_sort_select_cols_empty_list(self):
        """sort_select_cols returns empty list for empty input."""
        assert sort_select_cols([]) == []

    def test_sort_where_predicates_by_signature(self):
        """sort_where_predicates sorts by left_expr signature."""
        f1 = WhereParam(left_expr=NormalizedExpr.from_column("z.col"), op="=")
        f2 = WhereParam(left_expr=NormalizedExpr.from_column("a.col"), op="=")
        sorted_f = sort_where_predicates([f1, f2])
        assert sorted_f[0].left_expr.primary_term == "a.col"

    def test_sort_having_by_signature(self):
        """sort_having sorts by left_expr signature."""
        h1 = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "z.col"), op=">")
        h2 = HavingParam(left_expr=NormalizedExpr.from_agg("count", "a.col"), op=">")
        sorted_h = sort_having([h1, h2])
        assert sorted_h[0].left_expr.add_groups[0].agg_func == "count"


class TestCanonicalizeConditionOrder:
    """Tests for _canonicalize_condition_order structural-key sorting."""

    def test_single_item_unchanged(self):
        """Single-item list is returned unchanged."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        result = _canonicalize_condition_order([fp], _where_structural_key)
        assert len(result) == 1
        assert result[0].left_expr.primary_term == "t.a"

    def test_empty_list(self):
        """Empty list returns empty."""
        assert _canonicalize_condition_order([], _where_structural_key) == []

    def test_sorts_by_structural_key(self):
        """List is sorted by structural key regardless of input order."""
        fz = WhereParam(left_expr=NormalizedExpr.from_column("t.z"), op="=")
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        fm = WhereParam(left_expr=NormalizedExpr.from_column("t.m"), op="=")
        result = _canonicalize_condition_order([fz, fa, fm], _where_structural_key)
        assert [r.left_expr.primary_term for r in result] == ["t.a", "t.m", "t.z"]

    def test_two_items_sorted(self):
        """Two-item list sorts by structural key."""
        fz = WhereParam(left_expr=NormalizedExpr.from_column("t.z"), op="=")
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        result = _canonicalize_condition_order([fz, fa], _where_structural_key)
        assert result[0].left_expr.primary_term == "t.a"
        assert result[1].left_expr.primary_term == "t.z"

    def test_forward_links_returns_and_connectors(self):
        """``_forward_links`` is a legacy helper that always returns AND connectors."""
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        fb = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        fc = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        items = [fa, fb, fc]
        assert _forward_links(items) == ["AND", "AND", "AND"]
        result = _canonicalize_condition_order(items, _where_structural_key)
        assert [r.left_expr.primary_term for r in result] == ["t.a", "t.b", "t.c"]

    def test_having_variant(self):
        """Canonicalization works with HavingParam via _having_structural_key."""
        hz = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.z"), op=">")
        ha = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.a"), op=">")
        result = _canonicalize_condition_order([hz, ha], _having_structural_key)
        assert result[0].left_expr.add_groups[0].agg_func == "count"
        assert result[1].left_expr.add_groups[0].agg_func == "sum"

    def test_idempotent(self):
        """Applying canonicalization twice produces the same result."""
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        fc = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        fb = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        first = _canonicalize_condition_order([fa, fc, fb], _where_structural_key)
        second = _canonicalize_condition_order(first, _where_structural_key)
        assert [r.left_expr.primary_term for r in first] == [r.left_expr.primary_term for r in second]


class TestSortFiltersGroupAware:
    """Tests for sort_where_predicates flat structural-key sorting."""

    def test_sorts_by_structural_key(self):
        """Filters are sorted by structural key."""
        fz = WhereParam(left_expr=NormalizedExpr.from_column("t.z"), op="=")
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        result = sort_where_predicates([fz, fa])
        assert result[0].left_expr.primary_term == "t.a"
        assert result[1].left_expr.primary_term == "t.z"

    def test_sorts_three_items_by_structural_key(self):
        """Multiple filters sort by structural key regardless of input order."""
        fz = WhereParam(left_expr=NormalizedExpr.from_column("t.z"), op="=")
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        fb = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        result = sort_where_predicates([fz, fa, fb])
        assert [r.left_expr.primary_term for r in result] == ["t.a", "t.b", "t.z"]

    def test_render_or_predicate_group(self):
        """OR semantics are expressed via PredicateGroup trees, not flat bool_op."""
        g1 = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        g2 = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        result = sort_where_predicates([g1, g2])
        assert len(result) == 2
        or_group = PredicateGroup(
            op="or",
            groups=(
                PredicateGroup(op="and", predicates=(result[0],)),
                PredicateGroup(op="and", predicates=(result[1],)),
            ),
        )
        where_sql = _render_predicate_group_sql(or_group, lambda p: f"{p.left_expr.primary_term} = :p")
        assert " OR " in where_sql

    def test_render_or_predicate_group_after_sort(self):
        """Sorted flat filters can be wrapped in an OR PredicateGroup for rendering."""
        g1 = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        g2 = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        result = sort_where_predicates([g1, g2])
        assert len(result) == 2
        joined = _render_predicate_group_sql(
            PredicateGroup(
                op="or",
                groups=(
                    PredicateGroup(op="and", predicates=(result[0],)),
                    PredicateGroup(op="and", predicates=(result[1],)),
                ),
            ),
            lambda p: f"{p.left_expr.primary_term} = :p",
        )
        assert " OR " in joined

    def test_mixed_input_order_canonical(self):
        """Filters in arbitrary input order are canonicalized by structural key."""
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        fc = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        fb = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        result = sort_where_predicates([fa, fc, fb])
        assert [r.left_expr.primary_term for r in result] == ["t.a", "t.b", "t.c"]

    def test_sort_where_predicates_idempotent(self):
        """sort_where_predicates applied twice yields the same result."""
        fa = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        fb = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        fc = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        first = sort_where_predicates([fc, fa, fb])
        second = sort_where_predicates(first)
        assert [r.left_expr.primary_term for r in first] == [r.left_expr.primary_term for r in second]


class TestSortHavingGroupAware:
    """Tests for sort_having flat structural-key sorting."""

    def test_sorts_by_structural_key(self):
        """Having conditions are sorted by structural key."""
        hz = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.z"), op=">")
        ha = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.a"), op=">")
        result = sort_having([hz, ha])
        assert result[0].left_expr.add_groups[0].agg_func == "count"
        assert result[1].left_expr.add_groups[0].agg_func == "sum"

    def test_sorts_mixed_items_by_structural_key(self):
        """Multiple having conditions sort by structural key regardless of input order."""
        hz = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.z"), op=">")
        ha = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.a"), op=">")
        result = sort_having([hz, ha])
        assert result[0].left_expr.add_groups[0].agg_func == "count"
        assert result[1].left_expr.add_groups[0].agg_func == "sum"

    def test_intra_group_mixed_ops(self):
        """Having conditions in arbitrary input order are canonicalized by structural key."""
        ha = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.a"), op=">")
        hc = HavingParam(left_expr=NormalizedExpr.from_agg("max", "t.c"), op=">")
        hb = HavingParam(left_expr=NormalizedExpr.from_agg("min", "t.b"), op=">")
        result = sort_having([ha, hc, hb])
        funcs = [r.left_expr.add_groups[0].agg_func for r in result]
        assert funcs[1] < funcs[2]

    def test_sort_having_idempotent(self):
        """sort_having applied twice yields the same result."""
        ha = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.a"), op=">")
        hb = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.b"), op=">")
        first = sort_having([ha, hb])
        second = sort_having(first)
        assert [r.left_expr.primary_term for r in first] == [r.left_expr.primary_term for r in second]


class TestDedupFiltersGroupAware:
    """Tests for _dedup_where_predicates structural-signature dedup."""

    def test_same_signature_deduped(self):
        """Filters with identical signature_key are deduped."""
        fp1 = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="string",
        )
        fp2 = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="string",
        )
        result = _dedup_where_predicates([fp1, fp2])
        assert len(result) == 1

    def test_different_signatures_kept(self):
        """Filters with different signature_key are both kept."""
        fp1 = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="string",
        )
        fp2 = WhereParam(
            left_expr=NormalizedExpr.from_column("t.y"),
            op="=",
            value_type="string",
        )
        result = _dedup_where_predicates([fp1, fp2])
        assert len(result) == 2

    def test_identical_deduped(self):
        """Identical filters are deduped."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="string",
        )
        result = _dedup_where_predicates([fp, fp])
        assert len(result) == 1


class TestDedupHavingGroupAware:
    """Tests for _dedup_having structural-signature dedup."""

    def test_same_signature_deduped(self):
        """Havings with identical signature_key are deduped."""
        hp1 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="number",
        )
        hp2 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="number",
        )
        result = _dedup_having([hp1, hp2])
        assert len(result) == 1

    def test_different_signatures_kept(self):
        """Havings with different signature_key are both kept."""
        hp1 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="number",
        )
        hp2 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.y"),
            op="=",
            value_type="number",
        )
        result = _dedup_having([hp1, hp2])
        assert len(result) == 2

    def test_identical_deduped(self):
        """Identical havings are deduped."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="number",
        )
        result = _dedup_having([hp, hp])
        assert len(result) == 1


class TestNormalizeIntentFiltersHavingsGroupAware:
    """Tests for normalize_where_havings with group-aware sort and dedup."""

    def test_grouped_or_filters_sorted_within_group(self):
        """Grouped OR filters are sorted by structural key within the group."""
        fz = WhereParam(
            left_expr=NormalizedExpr.from_column("t.z"),
            op="=",
        )
        fa = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fz, fa]),
        )
        result = normalize_where_havings(intent)
        assert (result.where.leaves() if result.where else [])[0].left_expr.primary_term == "t.a"
        assert (result.where.leaves() if result.where else [])[1].left_expr.primary_term == "t.z"

    def test_grouped_having_sorted_within_group(self):
        """Grouped OR havings are sorted by structural key within the group."""
        hz = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "t.z"),
            op=">",
        )
        ha = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=predicate_group_from_list([hz, ha]),
        )
        result = normalize_where_havings(intent)
        assert (result.having.leaves() if result.having else [])[0].left_expr.add_groups[0].agg_func == "count"

    def test_cte_filters_group_aware_sort(self):
        """CTE step filters are group-aware sorted."""
        fz = WhereParam(
            left_expr=NormalizedExpr.from_column("t.z"),
            op="=",
        )
        fa = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fz, fa]),
            having=None,
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_where_havings(intent)
        assert (where_leaves(result.cte_steps[0].where) or [])[0].left_expr.primary_term == "t.a"

    def test_cte_having_group_aware_sort(self):
        """CTE step having conditions are group-aware sorted."""
        hz = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "t.z"),
            op=">",
        )
        ha = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=predicate_group_from_list([hz, ha]),
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_where_havings(intent)
        assert (having_leaves(result.cte_steps[0].having) or [])[0].left_expr.add_groups[0].agg_func == "count"

    def test_dedup_respects_signature_in_pipeline(self):
        """Pipeline dedup removes filters with identical structural signatures."""
        fp1 = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="string",
        )
        fp2 = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="string",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fp1, fp2]),
        )
        result = normalize_where_havings(intent)
        assert len(result.where.leaves() if result.where else []) == 1


class TestSimplifyExpr:
    """Tests for simplify_expr."""

    def test_constant_folding(self):
        """simplify_expr folds constant values."""
        expr = NormalizedExpr(
            add_values=[ExprValue(value=3.0), ExprValue(value=7.0)],
        )
        result = _simplify_expr(expr)
        assert len(result.add_values) == 1
        assert result.add_values[0].value == 10.0

    def test_zero_elimination(self):
        """simplify_expr eliminates zero-coefficient groups."""
        g = MulGroup(coefficient=1.0, multiply=["t.col"])
        expr = NormalizedExpr(
            add_groups=[g],
            sub_groups=[MulGroup(coefficient=1.0, multiply=["t.col"])],
        )
        result = _simplify_expr(expr)
        assert len(result.add_groups) == 0
        assert len(result.sub_groups) == 0

    def test_like_term_combining(self):
        """simplify_expr combines like terms."""
        g1 = MulGroup(coefficient=2.0, multiply=["t.col"])
        g2 = MulGroup(coefficient=3.0, multiply=["t.col"])
        expr = NormalizedExpr(add_groups=[g1, g2])
        result = _simplify_expr(expr)
        assert len(result.add_groups) == 1
        assert result.add_groups[0].coefficient == 5.0

    def test_negative_coefficient_to_sub(self):
        """simplify_expr moves negative coefficient groups to sub_groups."""
        g1 = MulGroup(coefficient=2.0, multiply=["t.a"])
        g2 = MulGroup(coefficient=5.0, multiply=["t.a"])
        expr = NormalizedExpr(add_groups=[g1], sub_groups=[g2])
        result = _simplify_expr(expr)
        assert len(result.sub_groups) == 1
        assert result.sub_groups[0].coefficient == 3.0

    def test_preserves_parameterized_values(self):
        """simplify_expr preserves parameterized ExprValues."""
        expr = NormalizedExpr(
            add_values=[ExprValue(value=5.0, param_key="p1"), ExprValue(value=3.0)],
        )
        result = _simplify_expr(expr)
        parameterized = [v for v in result.add_values if v.param_key]
        assert len(parameterized) == 1
        assert parameterized[0].param_key == "p1"

    def test_constant_only_group_folded(self):
        """simplify_expr folds MulGroup with no multiply/divide into constants."""
        g = MulGroup(coefficient=7.0, multiply=[], divide=[])
        expr = NormalizedExpr(add_groups=[g])
        result = _simplify_expr(expr)
        assert len(result.add_groups) == 0
        assert len(result.add_values) == 1
        assert result.add_values[0].value == 7.0

    def test_net_negative_constant(self):
        """simplify_expr handles net negative constants."""
        expr = NormalizedExpr(
            add_values=[ExprValue(value=3.0)],
            sub_values=[ExprValue(value=10.0)],
        )
        result = _simplify_expr(expr)
        assert len(result.sub_values) == 1
        assert result.sub_values[0].value == 7.0
        assert len(result.add_values) == 0

    def test_preserves_agg_func(self):
        """simplify_expr preserves expr-level agg_func."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.col"])],
            agg_func="sum",
        )
        result = _simplify_expr(expr)
        assert result.agg_func == "sum"


class TestSimplifyIntentExprs:
    """Tests for simplify_exprs."""

    def test_simplifies_select_cols(self):
        """simplify_exprs simplifies select column expressions."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[
                            MulGroup(coefficient=2.0, multiply=["t.x"]),
                            MulGroup(coefficient=3.0, multiply=["t.x"]),
                        ],
                    )
                ),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = simplify_exprs(intent)
        assert len(result.select_cols[0].expr.add_groups) == 1
        assert result.select_cols[0].expr.add_groups[0].coefficient == 5.0


class TestNormalizeFilterScalarOnLeft:
    """Tests for _normalize_where_scalar_on_left."""

    def test_swaps_scalar_left_column_right(self):
        """Swap when left is scalar and right is column, flipping operator."""
        fp = WhereParam(
            left_expr=parse_expr_string("CURRENT_DATE"),
            op="<=",
            right_expr=NormalizedExpr.from_column("payment.payment_date"),
            value_type="date",
        )
        result = _normalize_where_scalar_on_left(fp)
        assert result.left_expr.primary_column == "payment.payment_date"
        assert result.right_expr is not None
        assert "current_date" in result.right_expr.signature_key.lower()
        assert result.op == ">="

    def test_swaps_interval_left_column_right(self):
        """Swap when left is INTERVAL expression and right is column."""
        fp = WhereParam(
            left_expr=parse_expr_string("CURRENT_DATE - INTERVAL '90 days'"),
            op="<",
            right_expr=NormalizedExpr.from_column("rental.rental_date"),
            value_type="date",
        )
        result = _normalize_where_scalar_on_left(fp)
        assert result.left_expr.primary_column == "rental.rental_date"
        assert result.op == ">"

    def test_column_left_scalar_right_unchanged(self):
        """Leave column-on-left, scalar-on-right unchanged."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("rental.rental_date"),
            op=">=",
            right_expr=parse_expr_string("CURRENT_DATE - INTERVAL '90 days'"),
            value_type="date",
        )
        result = _normalize_where_scalar_on_left(fp)
        assert result.left_expr.primary_column == "rental.rental_date"
        assert result.op == ">="

    def test_column_left_column_right_unchanged(self):
        """Leave col-vs-col filters unchanged (handled by _normalize_col_to_col_where)."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            right_expr=NormalizedExpr.from_column("t.b"),
            value_type="string",
        )
        result = _normalize_where_scalar_on_left(fp)
        assert result.left_expr.primary_column == "t.a"
        assert result.right_expr.primary_column == "t.b"

    def test_no_right_expr_unchanged(self):
        """Leave filters without right_expr unchanged."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            raw_value="x",
        )
        result = _normalize_where_scalar_on_left(fp)
        assert result.left_expr.primary_column == "t.a"
        assert result.right_expr is None


class TestNormalizeFilterCanonical:
    """Tests for normalize_filter_canonical."""

    def test_swaps_empty_left_with_right(self):
        """normalize_filter_canonical swaps empty left with right."""
        fp = WhereParam(
            left_expr=NormalizedExpr(add_groups=[], add_values=[]),
            op=">",
            right_expr=NormalizedExpr.from_column("t.x"),
            value_type="number",
        )
        result = _normalize_where_canonical(fp)
        assert result.left_expr.primary_term == "t.x"
        assert result.op == "<"

    def test_nonempty_left_unchanged(self):
        """normalize_filter_canonical leaves non-empty left unchanged."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="number",
        )
        result = _normalize_where_canonical(fp)
        assert result.left_expr.primary_column == "t.x"
        assert result.op == "="


class TestNormalizeHavingCanonical:
    """Tests for normalize_having_canonical."""

    def test_swaps_empty_left_with_right(self):
        """normalize_having_canonical swaps empty left with right."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[], add_values=[]),
            op=">=",
            right_expr=NormalizedExpr.from_column("t.x"),
            value_type="number",
        )
        result = _normalize_having_canonical(hp)
        assert result.left_expr.primary_term == "t.x"
        assert result.op == "<="

    def test_nonempty_left_unchanged(self):
        """normalize_having_canonical leaves non-empty left unchanged."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("t.x"),
            op="=",
            value_type="number",
        )
        result = _normalize_having_canonical(hp)
        assert result.left_expr.primary_column == "t.x"


class TestNormalizeColToColFilter:
    """Tests for normalize_col_to_col_filter."""

    def test_smaller_sig_on_left(self):
        """normalize_col_to_col_filter puts smaller signature on left."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.z"),
            op=">",
            right_expr=NormalizedExpr.from_column("t.a"),
            value_type="number",
        )
        result = _normalize_col_to_col_where(fp)
        assert result.left_expr.primary_column == "t.a"
        assert result.op == "<"

    def test_already_ordered_unchanged(self):
        """normalize_col_to_col_filter leaves already-ordered unchanged."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            right_expr=NormalizedExpr.from_column("t.z"),
            value_type="number",
        )
        result = _normalize_col_to_col_where(fp)
        assert result.left_expr.primary_column == "t.a"
        assert result.op == ">"


class TestNormalizeAggToAggHaving:
    """Tests for normalize_agg_to_agg_having."""

    def test_smaller_sig_on_left(self):
        """normalize_agg_to_agg_having puts smaller signature on left."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("t.z"),
            op=">",
            right_expr=NormalizedExpr.from_column("t.a"),
            value_type="number",
        )
        result = _normalize_agg_to_agg_having(hp)
        assert result.left_expr.primary_column == "t.a"
        assert result.op == "<"

    def test_with_param_key_no_swap(self):
        """normalize_agg_to_agg_having does not swap when param_key set."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("t.z"),
            op=">",
            right_expr=NormalizedExpr.from_column("t.a"),
            value_type="number",
            param_key="p1",
        )
        result = _normalize_agg_to_agg_having(hp)
        assert result.left_expr.primary_column == "t.z"
        assert result.op == ">"


class TestDedupFilters:
    """Tests for _dedup_where_predicates."""

    def test_removes_duplicate(self):
        """_dedup_where_predicates removes duplicate by signature_key."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", value_type="string")
        result = _dedup_where_predicates([fp, fp])
        assert len(result) == 1

    def test_preserves_unique(self):
        """_dedup_where_predicates preserves unique filters."""
        fp1 = WhereParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", value_type="string")
        fp2 = WhereParam(left_expr=NormalizedExpr.from_column("t.y"), op=">", value_type="number")
        result = _dedup_where_predicates([fp1, fp2])
        assert len(result) == 2

    def test_empty_list(self):
        """_dedup_where_predicates returns empty for empty input."""
        assert _dedup_where_predicates([]) == []


class TestDedupHaving:
    """Tests for _dedup_having."""

    def test_removes_duplicate(self):
        """_dedup_having removes duplicate by signature_key."""
        hp = HavingParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", value_type="number")
        result = _dedup_having([hp, hp])
        assert len(result) == 1

    def test_preserves_unique(self):
        """_dedup_having preserves unique havings."""
        hp1 = HavingParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", value_type="number")
        hp2 = HavingParam(left_expr=NormalizedExpr.from_column("t.y"), op=">", value_type="number")
        result = _dedup_having([hp1, hp2])
        assert len(result) == 2


class TestNormalizeIntentFiltersHavings:
    """Tests for normalize_where_havings."""

    def test_deduplicates_filters(self):
        """normalize_where_havings deduplicates filters."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", value_type="string")
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fp, fp]),
        )
        result = normalize_where_havings(intent)
        assert len(result.where.leaves() if result.where else []) == 1

    def test_normalizes_ops(self):
        """normalize_where_havings normalizes operators."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.x"), op="==", value_type="string")
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fp]),
        )
        result = normalize_where_havings(intent)
        assert (result.where.leaves() if result.where else [])[0].op == "="


class TestEnforceGrainConsistency:
    """Tests for enforce_grain_consistency."""

    @pytest.fixture
    def grain_schema(self):
        """Schema for grain consistency tests."""
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                            distinct_count=100,
                            distinct_ratio=0.99,
                        ),
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                            distinct_count=50,
                        ),
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="numeric",
                            value_type="number",
                            role=ColumnRole.NUMERIC_MEASURE.value,
                            distinct_count=80,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )

    def test_infers_group_by_from_mixed_select(self, grain_schema):
        """enforce_grain_consistency infers group_by when select has agg and non-agg."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.customer_id")),
                SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, grain_schema)
        assert result.grain == "grouped"
        gb_terms = [g.primary_term for g in result.group_by_cols]
        assert "orders.customer_id" in gb_terms

    def test_no_change_when_no_agg(self, grain_schema):
        """enforce_grain_consistency returns unchanged when no aggregation."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, grain_schema)
        assert result.grain == "row_level"

    def test_no_change_all_agg(self, grain_schema):
        """enforce_grain_consistency returns unchanged when all columns are aggregated."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, grain_schema)
        assert result.grain == "row_level"

    def test_enforce_grain_consistency_preserves_window_select(self, grain_schema):
        """Partition keys on window specs do not promote grain or synthesise ``group_by_cols``."""
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.amount"))],
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.customer_id")),
                SelectCol(expr=NormalizedExpr(column_ref="w01")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=ws,
                )
            ],
        )
        result = enforce_grain_consistency(intent, grain_schema)
        assert result.grain == "row_level"
        assert result.group_by_cols == []
        assert result.window_registry and result.window_registry[0].window_spec.function == "row_number"


class TestEnforceCteGrainConsistency:
    """Tests for enforce_cte_grain_consistency."""

    def test_sets_grouped_when_group_by_present(self):
        """enforce_cte_grain_consistency sets grain to grouped."""
        cte = RuntimeCteStep(cte_name="cte1", group_by_cols=[NormalizedExpr.from_column("t.a")])
        result = enforce_cte_grain_consistency(cte)
        assert result.grain == "grouped"

    def test_unchanged_when_no_group_by(self):
        """enforce_cte_grain_consistency returns unchanged without group_by."""
        cte = RuntimeCteStep(cte_name="cte1")
        result = enforce_cte_grain_consistency(cte)
        assert result.grain == "row_level"


class TestResolveColumnMap:
    """Tests for resolve_column_map."""

    @pytest.fixture
    def map_schema(self):
        """Schema for column map tests."""
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "amount": ColumnMetadata(name="amount", data_type="numeric", value_type="number"),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "customers": TableMetadata(
                    name="customers",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "name": ColumnMetadata(name="name", data_type="varchar", value_type="string"),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )

    def test_bare_column_unique(self, map_schema):
        """resolve_column_map resolves unique bare column."""
        result, _issues = resolve_column_map(["name"], map_schema, ["orders", "customers"])
        assert result["name"] == "customers"

    def test_bare_column_ambiguous(self, map_schema):
        """resolve_column_map emits ``column_ambiguous`` instead of guessing for bare ``id``."""
        result, issues = resolve_column_map(["id"], map_schema, ["customers", "orders"])
        assert "id" not in result
        assert any(i.category == "column_ambiguous" for i in issues)

    def test_empty_columns(self, map_schema):
        """resolve_column_map returns empty for empty input."""
        result, _issues = resolve_column_map([], map_schema, ["orders"])
        assert result == {}

    def test_qualified_column(self, schema_graph):
        """Resolves table.column notation."""
        result, _issues = resolve_column_map(["orders.amount"], schema_graph, ["orders"])
        assert result["amount"] == "orders"

    def test_bare_column_single_match(self, schema_graph):
        """Resolves bare column name when unique across tables."""
        result, _issues = resolve_column_map(["amount"], schema_graph, ["orders"])
        assert result["amount"] == "orders"

    def test_bare_column_ambiguous_multi_table(self, schema_graph):
        """Ambiguous bare column yields an issue and no silent mapping."""
        schema = SchemaGraph(
            tables={
                "t1": TableMetadata(
                    name="t1",
                    columns={
                        "col": ColumnMetadata(
                            name="col",
                            data_type="integer",
                            value_type="integer",
                            role=ColumnRole.IDENTIFIER.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "t2": TableMetadata(
                    name="t2",
                    columns={
                        "col": ColumnMetadata(
                            name="col",
                            data_type="integer",
                            value_type="integer",
                            role=ColumnRole.IDENTIFIER.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )
        result, issues = resolve_column_map(["col"], schema, ["t1", "t2"])
        assert "col" not in result
        assert any(i.category == "column_ambiguous" for i in issues)

    def test_table_not_in_schema(self, schema_graph):
        """Skips tables not in schema graph."""
        result, _issues = resolve_column_map(["nonexistent.col"], schema_graph, ["nonexistent"])
        assert result == {}

    def test_qualified_column_wrong_table(self, schema_graph):
        """Qualified reference with table not in allowed list produces no mapping."""
        result, _issues = resolve_column_map(["products.name"], schema_graph, ["orders"])
        assert "name" not in result


class TestResolveCteColumnMaps:
    """Tests for resolve_cte_column_maps."""

    def test_single_cte_no_deps(self):
        """Single CTE with no upstream references produces empty column_map."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            output_columns=["amount"],
        )
        result = resolve_cte_column_maps([cte])
        assert len(result) == 1
        assert result[0].column_map.get("amount") is None or "orders" in result[0].column_map.get("amount", "")

    def test_second_cte_references_first(self):
        """Second CTE maps bare column to first CTE name."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            output_columns=["amount"],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("amount"))],
        )
        result = resolve_cte_column_maps([cte1, cte2])
        assert result[1].column_map.get("amount") == "cte1"

    def test_empty_list(self):
        """Returns empty list for empty input."""
        assert resolve_cte_column_maps([]) == []

    def test_qualified_ref_in_cte(self):
        """Qualified reference is split into column_map entry."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            output_columns=["amount"],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.amount"))],
        )
        result = resolve_cte_column_maps([cte1, cte2])
        assert result[1].column_map.get("amount") == "cte1"

    def test_maps_bare_columns_to_prior_cte(self):
        """resolve_cte_column_maps maps bare columns to earlier CTE output."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.total"))],
            output_columns=["total"],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("total"))],
        )
        result = resolve_cte_column_maps([cte1, cte2])
        assert result[1].column_map.get("total") == "cte1"

    def test_empty_cte_list(self):
        """resolve_cte_column_maps returns empty for empty input."""
        assert resolve_cte_column_maps([]) == []

    def test_qualified_column_uses_explicit_source(self):
        """resolve_cte_column_maps uses explicit table prefix for qualified column."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        )
        result = resolve_cte_column_maps([cte1])
        assert result[0].column_map.get("amount") == "orders"


class TestNormalizeCteNames:
    """Tests for normalize_cte_names."""

    def test_renames_ctes_sequentially(self):
        """normalize_cte_names renames CTE steps to cte1, cte2."""
        cte1 = RuntimeCteStep(
            cte_name="my_step",
            tables=["my_step"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("my_step.col"))],
        )
        cte2 = RuntimeCteStep(
            cte_name="other_step",
            tables=["my_step"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("my_step.val"))],
        )
        intent = RuntimeIntent(
            tables=["other_step"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("other_step.val"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte1, cte2],
        )
        result = normalize_cte_names(intent)
        assert result.cte_steps[0].cte_name == "cte1"
        assert result.cte_steps[1].cte_name == "cte2"
        assert "cte2" in result.tables

    def test_no_ctes_unchanged(self):
        """normalize_cte_names returns unchanged when no CTE steps."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = normalize_cte_names(intent)
        assert result.cte_steps == []

    def test_updates_main_query_refs(self):
        """normalize_cte_names updates references in main query expressions."""
        cte = RuntimeCteStep(
            cte_name="old_name",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.x"))],
        )
        intent = RuntimeIntent(
            tables=["old_name"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("old_name.x"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_cte_names(intent)
        assert result.select_cols[0].expr.primary_term == "cte1.x"
        assert "cte1" in result.tables

    def test_renames_to_canonical(self):
        """CTE steps are renamed to cte1, cte2, etc."""
        cte1 = RuntimeCteStep(cte_name="my_aggregation", tables=["orders"])
        cte2 = RuntimeCteStep(cte_name="my_filter", tables=["my_aggregation"])
        intent = RuntimeIntent(
            tables=["my_filter"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte1, cte2],
        )
        result = normalize_cte_names(intent)
        assert result.cte_steps[0].cte_name == "cte1"
        assert result.cte_steps[1].cte_name == "cte2"

    def test_updates_table_references(self):
        """Table references inside CTE steps are updated to new names."""
        cte1 = RuntimeCteStep(cte_name="old_cte", tables=["orders"])
        cte2 = RuntimeCteStep(cte_name="second_cte", tables=["old_cte"])
        intent = RuntimeIntent(
            tables=["second_cte"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte1, cte2],
        )
        result = normalize_cte_names(intent)
        assert "cte1" in result.cte_steps[1].tables
        assert "cte2" in result.tables

    def test_no_cte_steps_returns_unchanged(self, minimal_intent):
        """Intent without CTE steps is returned unchanged."""
        result = normalize_cte_names(minimal_intent)
        assert result.tables == minimal_intent.tables

    def test_updates_expressions_in_cte(self):
        """Expression terms referencing old CTE names are updated."""
        cte1 = RuntimeCteStep(
            cte_name="agg_step",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            output_columns=["amount"],
        )
        cte2 = RuntimeCteStep(
            cte_name="final_step",
            tables=["agg_step"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("agg_step.amount"))],
        )
        intent = RuntimeIntent(
            tables=["final_step"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("final_step.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte1, cte2],
        )
        result = normalize_cte_names(intent)
        main_term = result.select_cols[0].expr.primary_column
        assert "cte2" in main_term
        cte2_term = result.cte_steps[1].select_cols[0].expr.primary_column
        assert "cte1" in cte2_term

    def test_normalize_cte_names_preserves_scalar_output_alias(self):
        cte = RuntimeCteStep(
            cte_name="metrics",
            tables=["rental"],
            grain="scalar",
            emission="scalar_subquery",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "rental.rate"))],
            output_columns=["metrics"],
        )
        intent = RuntimeIntent(
            tables=["metrics"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("metrics.avg_rate"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_cte_names(intent)
        assert result.cte_steps[0].output_columns == ["rate"]

    def test_planner_cte_alias_rewrites_window_registry_refs(self):
        """Planner-only CTE aliases map to canonical cteN names (RS-004)."""
        cte = RuntimeCteStep(
            cte_name="customer_totals",
            tables=["customer"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "customer.amount"))],
            output_columns=["total"],
        )
        intent = RuntimeIntent(
            tables=["customer_totals"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer_totals.total"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
            planner_cte_names=["customer_totals"],
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=WindowSpec(
                        function="row_number",
                        partition_by=[NormalizedExpr.from_column("customer_totals.total")],
                        order_by=[],
                    ),
                )
            ],
        )
        result = normalize_cte_names(intent)
        assert result.cte_steps[0].cte_name == "cte1"
        part_col = result.window_registry[0].window_spec.partition_by[0].primary_column
        assert part_col == "cte1.total"

    def test_rewrite_main_query_refs_to_final_cte_columns_updates_filter(self):
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["rental"],
            grain="scalar",
            emission="scalar_subquery",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "rental.rate"))],
            output_columns=["rate"],
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("cte1.cte1"),
                        op=">",
                        raw_value=1,
                    )
                ]
            ),
            cte_steps=[cte],
        )
        out = rewrite_main_query_refs_to_final_cte_columns(intent)
        assert (out.where.leaves() if out.where else [])[0].left_expr.primary_term == "cte1.rate"

    def test_rewrite_main_query_refs_inferred_alias_to_explicit_output(self):
        """When output_columns disagree with inferred select aliases, remap by position."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_agg("avg", "orders.amount")),
                SelectCol(expr=NormalizedExpr.from_column("orders.status")),
            ],
            output_columns=["avg_amt", "order_status"],
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("cte1.avg_amount"),
                        op="=",
                        raw_value=1,
                    ),
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("cte1.status"),
                        op="=",
                        raw_value="x",
                    ),
                ]
            ),
            cte_steps=[cte],
        )
        out = rewrite_main_query_refs_to_final_cte_columns(intent)
        assert (out.where.leaves() if out.where else [])[0].left_expr.primary_term == "cte1.avg_amt"
        assert (out.where.leaves() if out.where else [])[1].left_expr.primary_term == "cte1.order_status"

    def test_rewrite_main_query_refs_raw_sql_fragment(self):
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["rental"],
            grain="scalar",
            emission="scalar_subquery",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "rental.rate"))],
            output_columns=["rate"],
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr(raw_sql="cte1.cte1 > 1"),
                        op=">",
                        raw_value=1,
                    )
                ]
            ),
            cte_steps=[cte],
        )
        out = rewrite_main_query_refs_to_final_cte_columns(intent)
        assert "cte1.rate" in ((out.where.leaves() if out.where else [])[0].left_expr.raw_sql or "")


class TestEnforceIntentSchema:
    """Tests for check_qualified_refs_exist."""

    @pytest.fixture
    def validate_schema(self):
        """Schema for validation tests."""
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "amount": ColumnMetadata(name="amount", data_type="numeric", value_type="number"),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )

    def test_valid_intent_no_errors(self, validate_schema):
        """check_qualified_refs_exist returns no errors for valid intent."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert errors == []

    def test_unknown_table_error(self, validate_schema):
        """check_qualified_refs_exist reports unknown table."""
        intent = RuntimeIntent(
            tables=["nonexistent"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert any("Unknown table" in e for e in errors)

    def test_unknown_column_error(self, validate_schema):
        """check_qualified_refs_exist reports unknown column."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.nonexistent"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert any("Unknown" in e and "nonexistent" in e for e in errors)

    def test_cte_name_accepted_as_table(self, validate_schema):
        """check_qualified_refs_exist accepts CTE names as valid tables."""
        cte = RuntimeCteStep(cte_name="cte1")
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert not any("Unknown table: cte1" in e for e in errors)


class TestReplaceRefsInExprEdgeCases:
    """Edge-case tests for replace_refs_in_expr."""

    def test_replaces_divide_terms(self):
        """replace_refs_in_expr applies replacer to divide terms."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=[], divide=["old.col"])])
        result = replace_refs_in_expr(expr, lambda s: s.replace("old", "new"))
        assert term_strs(result.add_groups[0].divide) == ["new.col"]

    def test_replaces_sub_groups(self):
        """replace_refs_in_expr applies replacer to sub_groups."""
        expr = NormalizedExpr(sub_groups=[MulGroup(multiply=["old.col"])])
        result = replace_refs_in_expr(expr, lambda s: s.replace("old", "new"))
        assert term_strs(result.sub_groups[0].multiply) == ["new.col"]

    def test_identity_replacer_no_change(self):
        """replace_refs_in_expr with identity function returns equivalent expr."""
        expr = NormalizedExpr.from_column("t.col")
        result = replace_refs_in_expr(expr, lambda s: s)
        assert result.primary_term == "t.col"

    def test_empty_groups_preserves_add_values(self):
        """replace_refs_in_expr with no groups preserves add_values."""
        expr = NormalizedExpr(add_groups=[], sub_groups=[], add_values=[ExprValue(value=1.0)])
        result = replace_refs_in_expr(expr, lambda s: s.upper())
        assert result.add_values == expr.add_values
        assert result.add_groups == []
        assert result.sub_groups == []


class TestNormalizeCountStarEdgeCases:
    """Edge-case tests for normalize_count_star."""

    def test_fixes_count_in_filters(self):
        """normalize_count_star fixes COUNT(1) in filter expressions."""
        fp = WhereParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]),
            op=">",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fp]),
        )
        result = normalize_count_star(intent)
        assert term_strs((result.where.leaves() if result.where else [])[0].left_expr.add_groups[0].multiply) == [
            "COUNT(*)"
        ]

    def test_no_count_unchanged(self):
        """normalize_count_star leaves non-count terms unchanged."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.name"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = normalize_count_star(intent)
        assert result.select_cols[0].expr.primary_term == "t.name"


class TestSortFunctionsEdgeCases:
    """Edge-case tests for sort functions."""

    def test_sort_select_cols_stable_on_tie(self):
        """sort_select_cols preserves order on signature tie."""
        c1 = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        c2 = SelectCol(expr=NormalizedExpr.from_column("t.b"))
        result = sort_select_cols([c2, c1])
        assert result[0].expr.primary_term == "t.a"

    def test_sort_where_predicates_empty(self):
        """sort_where_predicates returns empty for empty input."""
        assert sort_where_predicates([]) == []

    def test_sort_having_empty(self):
        """sort_having returns empty for empty input."""
        assert sort_having([]) == []


class TestSimplifyExprEdgeCases:
    """Edge-case tests for simplify_expr."""

    def test_empty_expr(self):
        """simplify_expr handles empty expression."""
        expr = NormalizedExpr()
        result = _simplify_expr(expr)
        assert result.add_groups == []
        assert result.add_values == []

    def test_sub_value_folding(self):
        """simplify_expr folds sub_values correctly."""
        expr = NormalizedExpr(
            sub_values=[ExprValue(value=5.0), ExprValue(value=3.0)],
        )
        result = _simplify_expr(expr)
        assert len(result.sub_values) == 1
        assert result.sub_values[0].value == 8.0

    def test_preserves_scalar_func(self):
        """simplify_expr preserves scalar_func on expression."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.col"])],
            scalar_func="upper",
        )
        result = _simplify_expr(expr)
        assert result.scalar_func == "upper"


class TestNormalizeCountStarCte:
    """CTE path tests for normalize_count_star."""

    def test_cte_count_1_normalised(self):
        """COUNT(1) in a CTE select_col is converted to COUNT(*)."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_count_star(intent)
        assert term_strs(result.cte_steps[0].select_cols[0].expr.add_groups[0].multiply) == ["COUNT(*)"]


class TestSimplifyIntentExprsCte:
    """CTE path tests for simplify_exprs."""

    def test_cte_constant_folded(self):
        """Constant values in CTE expressions are folded."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_values=[ExprValue(value=3.0), ExprValue(value=7.0)],
                    )
                ),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = simplify_exprs(intent)
        assert len(result.cte_steps[0].select_cols[0].expr.add_values) == 1
        assert result.cte_steps[0].select_cols[0].expr.add_values[0].value == 10.0


class TestNormalizeIntentFiltersHavingsCte:
    """CTE path tests for normalize_where_havings."""

    def test_cte_filters_deduped(self):
        """Duplicate filters in a CTE step are deduplicated."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", value_type="string")
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fp, fp]),
            having=None,
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_where_havings(intent)
        assert len(where_leaves(result.cte_steps[0].where) or []) == 1

    def test_cte_having_ops_normalised(self):
        """Operators in CTE havings are normalised."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op="==",
            value_type="integer",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=predicate_group_from_list([hp]),
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_where_havings(intent)
        assert (having_leaves(result.cte_steps[0].having) or [])[0].op == "="


class TestFilterAndHavingStructuralKeys:
    """Direct tests for _where_structural_key and _having_structural_key."""

    def test_filter_key_lowercases_op_and_value_type(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("T.X"),
            op="EQ",
            right_expr=NormalizedExpr.from_column("T.Y"),
            value_type="STRING",
        )
        key = _where_structural_key(fp)
        assert key[1] == "eq"
        assert key[3] == "string"

    def test_filter_key_empty_when_exprs_missing(self):
        fp = WhereParam(left_expr=NormalizedExpr(), op="=", value_type="number")
        key = _where_structural_key(fp)
        assert key[0] == ""
        assert key[2] == ""

    def test_having_key_matches_filter_shape(self):
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("SUM", "t.a"),
            op=">=",
            right_expr=None,
            value_type="NUMBER",
        )
        key = _having_structural_key(hp)
        assert key[1] == ">="
        assert key[3] == "number"
        assert key[2] == ""


class TestShiftMultiGroupRepresentatives:
    """Tests for legacy no-op group representative helpers."""

    def test_single_filter_unchanged(self):
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        assert _shift_multi_group_representative_where_forward([fp]) == [fp]

    def test_two_reps_unchanged(self):
        r1 = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=")
        r2 = WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=")
        out = _shift_multi_group_representative_where_forward([r1, r2])
        assert out == [r1, r2]

    def test_having_variant(self):
        h1 = HavingParam(left_expr=NormalizedExpr.from_column("t.a"), op=">")
        h2 = HavingParam(left_expr=NormalizedExpr.from_column("t.b"), op=">")
        out = _shift_multi_group_representative_having_forward([h1, h2])
        assert out == [h1, h2]


class TestIsCteOutputGroupable:
    """Tests for _is_cte_output_groupable."""

    def test_false_without_dot(self):
        assert _is_cte_output_groupable("col", []) is False

    def test_false_unknown_cte(self):
        cte = RuntimeCteStep(cte_name="c1", output_columns=["x"])
        assert _is_cte_output_groupable("other.x", [cte]) is False

    def test_true_when_output_lists_column_case_insensitive(self):
        cte = RuntimeCteStep(cte_name="MyCte", output_columns=["  Amount  "])
        assert _is_cte_output_groupable("mycte.amount", [cte]) is True

    def test_false_when_column_not_in_outputs(self):
        cte = RuntimeCteStep(cte_name="c1", output_columns=["a"])
        assert _is_cte_output_groupable("c1.b", [cte]) is False

    def test_empty_cte_steps(self):
        assert _is_cte_output_groupable("c1.x", []) is False


class TestRewriteCteOutputRefsToAliases:
    """Tests for rewrite_cte_output_refs_to_aliases."""

    def test_rewrites_main_select_to_output_alias(self):
        cte = RuntimeCteStep(
            cte_name="raw",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            output_columns=["revenue"],
        )
        intent = RuntimeIntent(
            tables=["raw"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("raw.orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = rewrite_cte_output_refs_to_aliases(intent)
        assert result.select_cols[0].expr.primary_term == "raw.revenue"

    def test_no_mapping_returns_same_intent_structure(self):
        cte = RuntimeCteStep(
            cte_name="raw",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["raw"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("raw.orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = rewrite_cte_output_refs_to_aliases(intent)
        assert result.select_cols[0].expr.primary_term == intent.select_cols[0].expr.primary_term

    def test_rewrites_filter_inside_cte(self):
        cte = RuntimeCteStep(
            cte_name="inner",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("inner.orders.amount"),
                        op=">",
                        right_expr=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
                    )
                ]
            ),
            output_columns=["amt"],
        )
        intent = RuntimeIntent(
            tables=["inner"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = rewrite_cte_output_refs_to_aliases(intent)
        left = (where_leaves(result.cte_steps[0].where) or [])[0].left_expr
        assert left.primary_term == "inner.amt"

    def test_maps_inferred_alias_to_output_column(self):
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["film_actor"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "film_actor.film_id"))],
            output_columns=["count_film_id"],
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("cte1.film_count"),
                        op=">",
                        right_expr=NormalizedExpr.from_column("w01"),
                    )
                ]
            ),
            cte_steps=[cte],
        )
        result = rewrite_cte_output_refs_to_aliases(intent)
        assert (result.where.leaves() if result.where else [])[0].left_expr.primary_term == "cte1.count_film_id"


class TestResolveWindowRegistryFilterRhs:
    """Tests for resolve_window_registry_where_rhs."""

    def test_moves_registry_token_from_raw_value_to_right_expr(self):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.amount"),
                        op=">",
                        raw_value="w01",
                        param_key="p1",
                    )
                ]
            ),
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=WindowSpec(function="avg", argument=NormalizedExpr.from_column("orders.amount")),
                )
            ],
        )
        out = resolve_window_registry_where_rhs(intent)
        fp = (out.where.leaves() if out.where else [])[0]
        assert fp.right_expr is not None
        assert fp.right_expr.column_ref == "w01"
        assert fp.raw_value is None
        assert fp.param_key == ""


class TestPromoteDateSubtractionToDateDiff:
    """Tests for promote_date_subtraction_to_date_diff."""

    def test_promotes_subtraction_filter_with_scalar_bound(self):
        left = parse_expr_string("rental.return_date - rental.rental_date")
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=left,
                        op=">",
                        raw_value=7,
                        value_type="number",
                    )
                ]
            ),
        )
        out = promote_date_subtraction_to_date_diff(intent)
        fp = (out.where.leaves() if out.where else [])[0]
        assert fp.value_type == "date_diff"
        assert fp.raw_value == {"unit": "day", "amount": 7}

    def test_expr_vs_expr_date_subtraction_unchanged(self):
        """Expr-vs-expr date subtraction vs integer column passes through unchanged."""
        left = parse_expr_string("rental.return_date - rental.rental_date")
        right = parse_expr_string("item.rental_duration")
        intent = RuntimeIntent(
            tables=["rental", "item"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=left,
                        op=">",
                        right_expr=right,
                    )
                ]
            ),
        )
        out = promote_date_subtraction_to_date_diff(intent)
        fp = (out.where.leaves() if out.where else [])[0]
        assert fp.right_expr is not None
        assert fp.value_type != "date_diff"


class TestEnforceGrainConsistencyExtended:
    """Additional grain-consistency scenarios."""

    @pytest.fixture
    def grain_schema(self):
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                            distinct_count=100,
                            distinct_ratio=0.99,
                        ),
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                            distinct_count=50,
                        ),
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="numeric",
                            value_type="number",
                            role=ColumnRole.NUMERIC_MEASURE.value,
                            distinct_count=80,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )

    def test_infers_group_by_from_cte_output_column(self, grain_schema):
        inner = RuntimeCteStep(
            cte_name="rollup",
            tables=["orders"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.customer_id")),
                SelectCol(expr=NormalizedExpr.from_column("orders.amount")),
            ],
            output_columns=["cid", "line_amt"],
        )
        intent = RuntimeIntent(
            tables=["rollup"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("rollup.cid")),
                SelectCol(expr=NormalizedExpr.from_agg("sum", "rollup.line_amt")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[inner],
        )
        result = enforce_grain_consistency(intent, grain_schema)
        assert result.grain == "grouped"
        assert any(g.primary_term == "rollup.cid" for g in result.group_by_cols)

    def test_excludes_non_groupable_column_from_inference(self, grain_schema):
        orders = grain_schema.tables["orders"]
        weird_col = ColumnMetadata(
            name="weird",
            data_type="varchar",
            value_type="string",
            is_groupable_override=False,
            distinct_ratio=1.0,
            distinct_count=10,
        )
        cols = dict(orders.columns)
        cols["weird"] = weird_col
        isolated = SchemaGraph(
            join_paths_multi=dict(grain_schema.join_paths_multi),
            effective_structural_hash=grain_schema.schema_hash,
            tables={
                **grain_schema.tables,
                "orders": TableMetadata(
                    name=orders.name,
                    columns=cols,
                    foreign_keys=list(orders.foreign_keys),
                    primary_key=orders.primary_key,
                ),
            },
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.weird")),
                SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, isolated)
        assert all(g.primary_term != "orders.weird" for g in result.group_by_cols)

    def test_existing_group_by_adds_missing_non_agg_select(self, schema_graph):
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.customer_id")),
                SelectCol(expr=NormalizedExpr.from_column("orders.status")),
                SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
            ],
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, schema_graph)
        terms = {g.primary_term for g in result.group_by_cols}
        assert "orders.status" in terms

    def test_primary_key_group_by_adds_descriptive_column(self, schema_graph):
        intent = RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("customers.customer_id")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "customers.customer_id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("customers.customer_id")],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, schema_graph)
        select_terms = {sc.expr.primary_term for sc in result.select_cols}
        assert "customers.name" in select_terms or "customers.email" in select_terms

    def test_foreign_key_group_by_adds_dst_descriptive_when_joined(self, schema_graph):
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            group_by_cols=[NormalizedExpr.from_column("orders.customer_id")],
            order_by_cols=[],
            where=None,
        )
        result = enforce_grain_consistency(intent, schema_graph)
        gb_terms = {g.primary_term for g in result.group_by_cols}
        assert any(t.startswith("customers.") for t in gb_terms)


class TestEnforceCteGrainConsistencyExtended:
    """Additional CTE grain tests."""

    def test_agg_without_group_by_sets_scalar(self):
        cte = RuntimeCteStep(
            cte_name="c",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            grain="row_level",
        )
        result = enforce_cte_grain_consistency(cte)
        assert result.grain == "scalar"

    def test_agg_without_group_by_idempotent_when_scalar(self):
        cte = RuntimeCteStep(
            cte_name="c",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t.id"))],
            grain="scalar",
        )
        result = enforce_cte_grain_consistency(cte)
        assert result is cte

    def test_sorts_group_by_signature(self):
        cte = RuntimeCteStep(
            cte_name="c",
            group_by_cols=[
                NormalizedExpr.from_column("t.z"),
                NormalizedExpr.from_column("t.a"),
            ],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        )
        result = enforce_cte_grain_consistency(cte)
        terms = [g.primary_term for g in result.group_by_cols]
        assert terms == sorted(terms)


class TestResolveColumnMapExtended:
    """Edge cases for resolve_column_map."""

    def test_qualified_ref_matches_table_suffix(self):
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "analytics.orders": TableMetadata(
                    name="analytics.orders",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        result, _issues = resolve_column_map(["orders.id"], sg, ["analytics.orders"])
        assert result["id"] == "analytics.orders"

    def test_bare_column_stripped_key(self, schema_graph):
        result, _issues = resolve_column_map(["  amount  "], schema_graph, ["orders"])
        assert "amount" in result
        assert result["amount"] == "orders"

    def test_unknown_bare_column_omitted(self, schema_graph):
        result, _issues = resolve_column_map(["not_a_real_column"], schema_graph, ["orders"])
        assert "not_a_real_column" not in result


class TestResolveCteColumnMapsExtended:
    """More coverage for resolve_cte_column_maps."""

    def test_extracts_columns_from_filters_order_by_having(self):
        cte1 = RuntimeCteStep(
            cte_name="c1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("a"), direction="ASC")],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("a"),
                        op=">",
                        right_expr=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
                    )
                ]
            ),
            having=predicate_group_from_list(
                [
                    HavingParam(
                        left_expr=NormalizedExpr.from_agg("count", "a"),
                        op=">",
                        right_expr=NormalizedExpr(add_values=[ExprValue(value=1.0)]),
                    )
                ]
            ),
            output_columns=["a"],
        )
        cte2 = RuntimeCteStep(
            cte_name="c2",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a"))],
        )
        result = resolve_cte_column_maps([cte1, cte2])
        assert result[1].column_map.get("a") == "c1"

    def test_later_cte_wins_on_duplicate_bare_name(self):
        cte1 = RuntimeCteStep(
            cte_name="first",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("x.a"))],
            output_columns=["v"],
        )
        cte2 = RuntimeCteStep(
            cte_name="second",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("y.b"))],
            output_columns=["v"],
        )
        cte3 = RuntimeCteStep(
            cte_name="third",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("v"))],
        )
        result = resolve_cte_column_maps([cte1, cte2, cte3])
        assert result[2].column_map.get("v") == "second"


class TestNormalizeCteNamesExtended:
    """Additional CTE rename and rewrite coverage."""

    def test_case_insensitive_ref_rewrite(self):
        cte = RuntimeCteStep(
            cte_name="MY_STEP",
            tables=["MY_STEP"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("my_step.col"))],
        )
        intent = RuntimeIntent(
            tables=["MY_STEP"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("My_Step.col"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = normalize_cte_names(intent)
        assert result.cte_steps[0].cte_name == "cte1"
        assert result.select_cols[0].expr.primary_term == "cte1.col"

    def test_column_map_and_output_metadata_keys_rewritten(self):
        cte = RuntimeCteStep(
            cte_name="old_cte",
            tables=["t"],
            select_cols=[],
            column_map={"old_cte.x": "old_cte"},
            output_columns=["old_cte.y"],
            output_column_metadata={"old_cte.meta": CteOutputColumnMeta(source="t")},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            column_map={"old_cte.k": "old_cte"},
            cte_steps=[cte],
        )
        result = normalize_cte_names(intent)
        assert any("cte1" in k for k in result.column_map)
        assert any("cte1" in k for k in result.cte_steps[0].column_map)
        assert any("cte1" in k for k in result.cte_steps[0].output_column_metadata)


class TestNormalizeCountStarExtended:
    """COUNT(1) normalization in more clauses."""

    def test_lowercase_count_1_normalized(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["count(1)"])]))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = normalize_count_star(intent)
        assert term_strs(result.select_cols[0].expr.add_groups[0].multiply) == ["COUNT(*)"]

    def test_order_by_having_and_main_having(self):
        fp = WhereParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]),
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
        )
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]),
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=1.0)]),
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[OrderByCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(1)"])]))],
            where=predicate_group_from_list([fp]),
            having=predicate_group_from_list([hp]),
        )
        result = normalize_count_star(intent)
        assert term_strs(result.order_by_cols[0].expr.add_groups[0].multiply) == ["COUNT(*)"]
        assert term_strs((result.where.leaves() if result.where else [])[0].left_expr.add_groups[0].multiply) == [
            "COUNT(*)"
        ]
        assert term_strs((result.having.leaves() if result.having else [])[0].left_expr.add_groups[0].multiply) == [
            "COUNT(*)"
        ]


class TestSimplifyExprsExtended:
    """Broader simplify_exprs coverage."""

    def test_simplifies_order_by_filters_and_having(self):
        dup = MulGroup(coefficient=1.0, multiply=["t.x"])
        expr = NormalizedExpr(add_groups=[dup, MulGroup(coefficient=2.0, multiply=["t.x"])])
        fp = WhereParam(
            left_expr=expr,
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
        )
        hp = HavingParam(
            left_expr=expr,
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=1.0)]),
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[OrderByCol(expr=expr, direction="DESC")],
            where=predicate_group_from_list([fp]),
            having=predicate_group_from_list([hp]),
        )
        result = simplify_exprs(intent)
        assert len(result.order_by_cols[0].expr.add_groups) == 1
        assert result.order_by_cols[0].expr.add_groups[0].coefficient == 3.0
        assert len((result.where.leaves() if result.where else [])[0].left_expr.add_groups) == 1
        assert len((result.having.leaves() if result.having else [])[0].left_expr.add_groups) == 1

    def test_net_zero_constant_drops_numeric_values(self):
        expr = NormalizedExpr(
            add_values=[ExprValue(value=5.0)],
            sub_values=[ExprValue(value=5.0)],
        )
        result = _simplify_expr(expr)
        assert result.add_values == []
        assert result.sub_values == []


class TestNormalizeColToColFilterExtended:
    """Edge cases for _normalize_col_to_col_where."""

    def test_param_key_prevents_swap(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.z"),
            op=">",
            right_expr=NormalizedExpr.from_column("t.a"),
            value_type="number",
            param_key="p",
        )
        result = _normalize_col_to_col_where(fp)
        assert result.left_expr.primary_column == "t.z"


class TestEnforceSchemaExtended:
    """Additional check_qualified_refs_exist violations."""

    @pytest.fixture
    def validate_schema(self):
        """Same shape as ``TestEnforceIntentSchema.validate_schema`` (class-local fixture)."""
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "amount": ColumnMetadata(name="amount", data_type="numeric", value_type="number"),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )

    def test_unknown_column_in_filter_right_side(self, validate_schema):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op="=",
            right_expr=NormalizedExpr.from_column("orders.not_a_col"),
            value_type="number",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list([fp]),
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert any("not_a_col" in e for e in errors)

    def test_unknown_column_in_group_by(self, validate_schema):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[NormalizedExpr.from_column("orders.bad_col")],
            order_by_cols=[],
            where=None,
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert any("bad_col" in e for e in errors)

    def test_cte_unknown_base_table(self, validate_schema):
        cte = RuntimeCteStep(cte_name="c1", tables=["ghost_table"], select_cols=[])
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert any("ghost_table" in e for e in errors)

    def test_cte_select_unknown_column(self, validate_schema):
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.no_such"))],
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        _, errors = check_qualified_refs_exist(intent, validate_schema)
        assert any("no_such" in e for e in errors)


class TestDedupHavingEmpty:
    def test_empty(self):
        assert _dedup_having([]) == []


def test_lift_distinct_select_from_raw_sql_promotes_structured_select(
    schema_graph: SchemaGraph,
) -> None:
    from aetherdialect._intent_resolve import lift_distinct_select_from_raw_sql

    cte = RuntimeCteStep(
        cte_name="c1",
        tables=[],
        select_cols=[SelectCol(expr=NormalizedExpr(raw_sql="DISTINCT customers.name"))],
    )
    intent = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr(raw_sql="DISTINCT customers.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customers.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    out = lift_distinct_select_from_raw_sql(intent, schema_graph)
    assert out.distinct_select_index == 0
    assert out.select_cols[0].expr.column_ref == "customers.customer_id"
    assert out.select_cols[1].expr.column_ref == "customers.name"
    assert out.cte_steps[0].distinct_select_index == 0
    assert out.cte_steps[0].select_cols[0].expr.column_ref == "customers.name"


def test_simplify_exprs_preserves_last_select_col(schema_graph: SchemaGraph) -> None:
    from aetherdialect._intent_resolve import simplify_exprs

    _ = schema_graph
    intent = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(raw_sql="DISTINCT customers.name"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    out = simplify_exprs(intent)
    assert out.select_cols[0].expr.raw_sql == "DISTINCT customers.name"


def test_prune_unused_cte_steps_drops_orphan() -> None:
    from aetherdialect._intent_resolve import prune_unused_cte_steps

    orphan = RuntimeCteStep(
        cte_name="dead",
        tables=[],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.x"))],
    )
    keeper = RuntimeCteStep(
        cte_name="live",
        tables=[],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.y"))],
    )
    intent = RuntimeIntent(
        tables=["live"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("live.a"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[orphan, keeper],
    )
    out = prune_unused_cte_steps(intent)
    assert len(out.cte_steps) == 1
    assert out.cte_steps[0].cte_name == "live"


class TestPruneUnusedCteOutputColumns:
    """Tests for prune_unused_cte_output_columns."""

    @staticmethod
    def _film_schema() -> SchemaGraph:
        film = TableMetadata(
            name="film",
            columns={
                "film_id": ColumnMetadata(
                    name="film_id",
                    data_type="integer",
                    role="identifier",
                    is_primary_key=True,
                ),
                "title": ColumnMetadata(name="title", data_type="varchar", role="categorical"),
                "description": ColumnMetadata(name="description", data_type="varchar", role="free_text"),
            },
            primary_key=["film_id"],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"film": film}, join_paths_multi={}, effective_structural_hash="h")

    @staticmethod
    def _payment_schema() -> SchemaGraph:
        payment = TableMetadata(
            name="payment",
            columns={
                "payment_id": ColumnMetadata(
                    name="payment_id",
                    data_type="integer",
                    role="identifier",
                    is_primary_key=True,
                ),
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    role="identifier",
                    is_foreign_key=True,
                    fk_target=("customer", "customer_id"),
                ),
            },
            primary_key=["payment_id"],
            foreign_keys=[],
        )
        return SchemaGraph(
            tables={"payment": payment},
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_downstream_reference_only_preserves_referenced_column(self) -> None:
        from aetherdialect._intent_resolve import prune_unused_cte_output_columns

        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["film"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film.title")),
                SelectCol(expr=NormalizedExpr.from_column("film.description")),
            ],
            output_columns=["title", "description"],
        )
        intent = RuntimeIntent(
            tables=["c1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("c1.title"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            natural_language="q",
            cte_steps=[cte],
        )
        out = prune_unused_cte_output_columns(intent, self._film_schema())
        assert len(out.cte_steps[0].select_cols) == 1
        assert out.cte_steps[0].output_columns == ["title"]

    def test_pk_passthrough_preserved_without_reference(self) -> None:
        from aetherdialect._intent_resolve import prune_unused_cte_output_columns

        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["film"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film.film_id")),
                SelectCol(expr=NormalizedExpr.from_column("film.title")),
            ],
            output_columns=["film_id", "title"],
        )
        intent = RuntimeIntent(
            tables=["c1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("c1.title"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            natural_language="q",
            cte_steps=[cte],
        )
        out = prune_unused_cte_output_columns(intent, self._film_schema())
        assert len(out.cte_steps[0].select_cols) == 2
        assert set(out.cte_steps[0].output_columns) == {"film_id", "title"}

    def test_fk_passthrough_preserved_without_reference(self) -> None:
        from aetherdialect._intent_resolve import prune_unused_cte_output_columns

        cte = RuntimeCteStep(
            cte_name="p1",
            tables=["payment"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("payment.customer_id")),
                SelectCol(expr=NormalizedExpr.from_column("payment.payment_id")),
            ],
            output_columns=["customer_id", "payment_id"],
        )
        intent = RuntimeIntent(
            tables=["p1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("p1.payment_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            natural_language="q",
            cte_steps=[cte],
        )
        out = prune_unused_cte_output_columns(intent, self._payment_schema())
        assert len(out.cte_steps[0].select_cols) == 2
        assert set(out.cte_steps[0].output_columns) == {"customer_id", "payment_id"}

    def test_computed_expression_not_auto_preserved_without_reference(self) -> None:
        from aetherdialect._intent_resolve import prune_unused_cte_output_columns

        expr_pk_plus = NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=[
                        NormalizedExpr.from_column("film.film_id"),
                        NormalizedExpr(raw_sql="1"),
                    ]
                ),
            ],
        )
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["film"],
            select_cols=[
                SelectCol(expr=expr_pk_plus),
                SelectCol(expr=NormalizedExpr.from_column("film.title")),
            ],
            output_columns=["expr1", "title"],
        )
        intent = RuntimeIntent(
            tables=["c1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("c1.title"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            natural_language="q",
            cte_steps=[cte],
        )
        out = prune_unused_cte_output_columns(intent, self._film_schema())
        assert len(out.cte_steps[0].select_cols) == 1
        assert out.cte_steps[0].output_columns == ["title"]


class TestCollectColumnRefsConcatMultiply:
    """collect_column_refs_for_post_processing walks CONCAT multiply parts."""

    def test_concat_multiply_column_refs(self) -> None:
        from aetherdialect._intent_resolve import (
            collect_column_refs_for_post_processing,
        )

        expr = NormalizedExpr(
            add_groups=[
                MulGroup(
                    scalar_func="concat",
                    multiply=[
                        NormalizedExpr.from_column("film.title"),
                        NormalizedExpr.from_column("film.description"),
                    ],
                )
            ],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=expr)],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={},
            natural_language="q",
        )
        refs = collect_column_refs_for_post_processing(intent)
        assert "film.title" in refs
        assert "film.description" in refs
