"""Tests for intent_expr module."""

import json
import re
from dataclasses import replace

import pytest

from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    SchemaGraph,
    TableMetadata,
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
    registry_render_scope,
)
from aetherdialect._core_utils import substitute_params
from aetherdialect._intent_expr import (
    _assign_structural_expr,
    _assign_structural_group,
    classify_cte_expr,
    _fill_expr_defaults,
    _fill_group_defaults,
    _is_date_or_interval_expr,
    _is_expr_numeric,
    _is_nontrivial_group,
    _normalize_order_direction,
    _order_by_col_from_obc,
    _parse_between_raw_value,
    _strip_angle_brackets,
    _strip_order_direction,
    _tag_single_expr,
    _validate_intent_schema,
    assign_param_keys,
    build_cte_output_metadata,
    canonicalize_temporal_unit_args,
    cleared_param_runtime_intent,
    collect_raw_param_values,
    decompose_between_params,
    derive_cte_output_columns,
    ensure_scalar_func_defaults,
    extract_columns_from_expr,
    extract_structural_params,
    normalize_date_diff_raw_values,
    normalize_in_raw_values,
    parse_expr_string,
    parse_intent_response,
    repair_misclassified_date_diff,
    replace_refs_in_expr,
    structural_s_key_assignment_order,
    tag_case_when_condition_scope,
    tag_expr_numeric,
)
from aetherdialect._sql_gen import classify_cte_emission, render_select_col_sql


class TestParseExprString:
    """Tests for parse_expr_string."""

    def test_bare_column(self):
        """Parse bare column to NormalizedExpr (canonical leaf, no MulGroup wrap)."""
        expr = parse_expr_string("table1.col1")
        assert (expr.column_ref, expr.add_groups) == ("table1.col1", [])
        assert expr.primary_column == "table1.col1"

    def test_count_star(self):
        """Parse COUNT(*)."""
        expr = parse_expr_string("COUNT(*)")
        assert expr.has_aggregation is True

    def test_additive_expression(self):
        """Parse SUM(a) - SUM(b) into add and sub groups."""
        expr = parse_expr_string("SUM(table1.a) - SUM(table1.b)")
        assert len(expr.add_groups) == 1
        assert len(expr.sub_groups) == 1

    def test_concat_and_dpipe_normalize_to_concat_scalar(self):
        """CONCAT(...) and PostgreSQL || both map to a concat MulGroup."""
        c = parse_expr_string("CONCAT(customers.first_name, ' ', customers.last_name)")
        assert len(c.add_groups) == 1
        g0 = c.add_groups[0]
        assert g0.scalar_func == "concat"
        assert len(g0.multiply) == 3
        d = parse_expr_string("customers.first_name || ' ' || customers.last_name")
        assert len(d.add_groups) == 1
        assert d.add_groups[0].scalar_func == "concat"
        assert len(d.add_groups[0].multiply) == 3

    def test_empty_string(self):
        """Parse empty string to empty NormalizedExpr."""
        expr = parse_expr_string("")
        assert expr == NormalizedExpr()

    def test_dict_with_expr_key(self):
        """Parse dict with 'expr' key uses value as expression."""
        expr = parse_expr_string({"expr": "orders.amount"})
        assert expr.primary_column == "orders.amount"

    def test_dict_missing_expr_returns_empty(self):
        """Parse dict without 'expr' key treats as empty."""
        expr = parse_expr_string({"other": "x"})
        assert expr == NormalizedExpr()

    def test_non_string_coerced_to_string(self):
        """Parse non-string input is coerced via str()."""
        expr = parse_expr_string(42)
        assert len(expr.add_values) == 1
        assert expr.add_values[0].value == 42.0

    def test_whitespace_only_returns_empty(self):
        """Parse whitespace-only string returns empty NormalizedExpr."""
        expr = parse_expr_string("   ")
        assert expr == NormalizedExpr()

    def test_single_negative_value_in_sub_values(self):
        """Parse single negative literal into sub_values."""
        expr = parse_expr_string("- 5")
        assert len(expr.sub_values) == 1
        assert expr.sub_values[0].value == 5.0

    def test_numeric_literal(self):
        """Parse standalone numeric literal."""
        expr = parse_expr_string("42")
        assert len(expr.add_values) == 1
        assert expr.add_values[0].value == 42.0


class TestStripAngleBrackets:
    """Tests for _strip_angle_brackets."""

    def test_string_with_brackets(self):
        """Strip angle brackets from string."""
        assert _strip_angle_brackets("<table1>") == "table1"

    def test_nested_dict(self):
        """Strip angle brackets in nested dict."""
        result = _strip_angle_brackets({"key": "<value>"})
        assert result == {"key": "value"}

    def test_list_values(self):
        """Strip angle brackets in list values."""
        result = _strip_angle_brackets(["<a>", "<b>"])
        assert result == ["a", "b"]

    def test_non_string(self):
        """Return non-string values unchanged."""
        assert _strip_angle_brackets(42) == 42


class TestAssignParamKeys:
    """Tests for assign_param_keys."""

    def test_sequential_keys(self):
        """
        Assign p1, p2, ...

        sequentially to filters.
        """
        fp1 = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        fp2 = FilterParam(left_expr=NormalizedExpr.from_column("t.b"), op=">", value_type="integer")
        new_fp, new_hp, new_cte, _, idx = assign_param_keys([fp1, fp2], [])
        assert new_fp[0].param_key == "p1"
        assert new_fp[1].param_key == "p2"
        assert idx == 3

    def test_is_null_skipped(self):
        """Skip param_key assignment for IS NULL filters."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="is null", value_type="null")
        new_fp, _, _, _, idx = assign_param_keys([fp], [])
        assert new_fp[0].param_key == ""
        assert idx == 1

    def test_having_gets_keys(self):
        """Assign param_keys to having conditions."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        _, new_hp, _, _, idx = assign_param_keys([], [hp])
        assert new_hp[0].param_key == "p1"

    def test_cte_before_main(self):
        """CTE filters get lower param_key indices than main query."""
        cte_fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        main_fp = FilterParam(left_expr=NormalizedExpr.from_column("t.b"), op="=", value_type="string")
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        new_fp, _, new_cte, _, _ = assign_param_keys([main_fp], [], [cte])
        assert new_cte[0].filters_param[0].param_key == "p1"
        assert new_fp[0].param_key == "p2"

    def test_case_registry_branch_conditions_get_param_keys(self):
        """Literals in ``case_registry`` CASE branches receive ``p*`` keys."""

        cond = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rental_rate"),
            op=">",
            value_type="number",
            raw_value=3,
        )
        branch = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("premium"))
        cw = CaseWhenExpr(
            branches=[branch],
            else_result=NormalizedExpr.from_column("standard"),
        )
        reg = CaseRegistryStep(registry_id="c01", case_when=cw)
        _, _, _, new_cr, idx = assign_param_keys([], [], None, [reg])
        assert new_cr[0].case_when.branches[0].condition.param_key == "p1"
        assert idx == 2


class TestDecomposeBetweenFilters:
    """Tests for decompose_between_params."""

    def test_between_with_list_value(self):
        """Decompose BETWEEN with [low, high] raw_value."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="between",
            value_type="integer",
            raw_value=[10, 20],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert len(result.filters_param) == 2
        assert result.filters_param[0].op == ">="
        assert result.filters_param[0].raw_value == 10
        assert result.filters_param[1].op == "<="
        assert result.filters_param[1].raw_value == 20

    def test_non_between_unchanged(self):
        """Non-BETWEEN filters pass through unchanged."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert len(result.filters_param) == 1
        assert result.filters_param[0].op == "="


def _make_intent_with_case(branches, group_by=None, cte=False):
    """Build a minimal RuntimeIntent (or one with a CTE) holding CASE in ``case_registry``."""

    cw = CaseWhenExpr(branches=list(branches), else_result=NormalizedExpr.from_column("'other'"))
    step = CaseRegistryStep(registry_id="c01", case_when=cw)
    sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
    if cte:
        cte_step = RuntimeCteStep(
            cte_name="c",
            description="",
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=list(group_by or []),
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            case_registry=[step],
        )
        return RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte_step],
            natural_language="t",
        )
    return RuntimeIntent(
        tables=["t"],
        grain="grouped" if group_by else "row_level",
        select_cols=[sc],
        group_by_cols=list(group_by or []),
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        param_values={},
        cte_steps=[],
        natural_language="t",
        case_registry=[step],
    )


class TestNormalizeInCaseBranches:
    """normalize_in_raw_values must canonicalise IN/NOT IN inside CASE branches."""

    def test_in_string_to_list_in_case_branch(self):
        cond = FilterParam(
            left_expr=NormalizedExpr.from_column("t.kind"),
            op="in",
            value_type="string",
            raw_value="'a','b','c'",
        )
        branch = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'X'"))
        intent = _make_intent_with_case([branch])
        out = normalize_in_raw_values(intent)
        new_cond = out.case_registry[0].case_when.branches[0].condition
        assert isinstance(new_cond.raw_value, list)
        assert new_cond.raw_value == ["a", "b", "c"]

    def test_not_in_string_to_list_in_cte_case_branch(self):
        cond = FilterParam(
            left_expr=NormalizedExpr.from_column("t.kind"),
            op="not in",
            value_type="string",
            raw_value="'a','b'",
        )
        branch = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'Y'"))
        intent = _make_intent_with_case([branch], cte=True)
        out = normalize_in_raw_values(intent)
        new_cond = out.cte_steps[0].case_registry[0].case_when.branches[0].condition
        assert new_cond.raw_value == ["a", "b"]

    def test_non_in_op_unchanged(self):
        cond = FilterParam(
            left_expr=NormalizedExpr.from_column("t.kind"),
            op="=",
            value_type="string",
            raw_value="a",
        )
        branch = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'X'"))
        intent = _make_intent_with_case([branch])
        out = normalize_in_raw_values(intent)
        assert out.case_registry[0].case_when.branches[0].condition.raw_value == "a"


class TestTagCaseWhenConditionScope:
    """tag_case_when_condition_scope must mark aggregated branch conditions as 'having'."""

    def test_row_level_branch_stays_filter(self):
        cond = FilterParam(left_expr=NormalizedExpr.from_column("t.x"), op="=", raw_value=1)
        br = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'low'"))
        intent = _make_intent_with_case([br])
        out = tag_case_when_condition_scope(intent)
        assert out.case_registry[0].case_when.condition_scope == "filter"

    def test_aggregated_branch_becomes_having(self):
        agg_left = NormalizedExpr.from_column("t.amount")
        agg_left = replace(agg_left, agg_func="sum")
        cond = FilterParam(left_expr=agg_left, op=">", value_type="number", raw_value=100)
        br = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'high'"))
        intent = _make_intent_with_case([br], group_by=[NormalizedExpr.from_column("t.k")])
        out = tag_case_when_condition_scope(intent)
        assert out.case_registry[0].case_when.condition_scope == "having"

    def test_aggregated_branch_in_cte(self):
        agg_left = NormalizedExpr.from_column("t.amount")
        agg_left = replace(agg_left, agg_func="sum")
        cond = FilterParam(left_expr=agg_left, op=">", value_type="number", raw_value=100)
        br = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'high'"))
        intent = _make_intent_with_case([br], cte=True)
        out = tag_case_when_condition_scope(intent)
        assert out.cte_steps[0].case_registry[0].case_when.condition_scope == "having"


class TestNormalizeDateDiffRawValues:
    """Tests for normalize_date_diff_raw_values."""

    def test_plural_days_to_singular(self):
        """Plural 'days' in date_diff raw_value becomes 'day'."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "days", "amount": 7},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_date_diff_raw_values(intent)
        assert result.filters_param[0].raw_value["unit"] == "day"

    def test_plural_weeks_to_singular(self):
        """Plural 'weeks' becomes 'week'."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "weeks", "amount": 2},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_date_diff_raw_values(intent)
        assert result.filters_param[0].raw_value["unit"] == "week"

    def test_date_window_plural_normalized(self):
        """date_window with plural unit is also normalized."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">=",
            value_type="date_window",
            raw_value={"unit": "months", "amount": 1},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_date_diff_raw_values(intent)
        assert result.filters_param[0].raw_value["unit"] == "month"

    def test_non_date_filter_unchanged(self):
        """Filters without date_window/date_diff pass through unchanged."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            raw_value="x",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_date_diff_raw_values(intent)
        assert result.filters_param[0].raw_value == "x"


class TestCanonicalizeTemporalUnitArgs:
    """Tests for canonicalize_temporal_unit_args (Phase 16)."""

    def _make_intent_with_select_expr(self, expr: NormalizedExpr) -> RuntimeIntent:
        return RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[SelectCol(expr=expr)],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )

    def test_canonicalizes_monthly_alias_in_select_expr(self):
        """`monthly` in date_trunc args is canonicalized to `month`."""
        expr = NormalizedExpr(
            scalar_func="date_trunc",
            scalar_func_args=["monthly"],
            add_groups=[MulGroup(multiply=["t.created_at"])],
        )
        out = canonicalize_temporal_unit_args(self._make_intent_with_select_expr(expr))
        assert out.select_cols[0].expr.scalar_func_args[0] == "month"

    def test_canonicalizes_quarterly_yearly_yyyy_aliases(self):
        """`quarterly`, `yearly`, `yyyy` all map to canonical units."""
        for alias, canonical in [
            ("quarterly", "quarter"),
            ("yearly", "year"),
            ("yyyy", "year"),
            ("daily", "day"),
        ]:
            expr = NormalizedExpr(
                scalar_func="date_trunc",
                scalar_func_args=[alias],
                add_groups=[MulGroup(multiply=["t.created_at"])],
            )
            out = canonicalize_temporal_unit_args(self._make_intent_with_select_expr(expr))
            assert out.select_cols[0].expr.scalar_func_args[0] == canonical, alias

    def test_already_canonical_unit_unchanged(self):
        """Canonical `month` stays `month`."""
        expr = NormalizedExpr(
            scalar_func="date_trunc",
            scalar_func_args=["month"],
            add_groups=[MulGroup(multiply=["t.created_at"])],
        )
        out = canonicalize_temporal_unit_args(self._make_intent_with_select_expr(expr))
        assert out.select_cols[0].expr.scalar_func_args[0] == "month"

    def test_unknown_token_passes_through(self):
        """An unrecognised token is left unchanged so validators can flag it."""
        expr = NormalizedExpr(
            scalar_func="date_trunc",
            scalar_func_args=["fortnight"],
            add_groups=[MulGroup(multiply=["t.created_at"])],
        )
        out = canonicalize_temporal_unit_args(self._make_intent_with_select_expr(expr))
        assert out.select_cols[0].expr.scalar_func_args[0] == "fortnight"

    def test_non_temporal_scalar_func_left_alone(self):
        """`scalar_func` outside SCALAR_FUNCTIONS_LEADING_ARG keeps its leading arg."""
        expr = NormalizedExpr(
            scalar_func="round",
            scalar_func_args=[2],
            add_groups=[MulGroup(multiply=["t.amount"])],
        )
        out = canonicalize_temporal_unit_args(self._make_intent_with_select_expr(expr))
        assert out.select_cols[0].expr.scalar_func_args[0] == 2

    def test_canonicalizes_inner_scalar_func_and_group_args(self):
        """Inner scalar args and per-group args also canonicalize."""
        group = MulGroup(
            multiply=["t.created_at"],
            scalar_func="extract",
            scalar_func_args=["yr"],
            inner_scalar_func="date_part",
            inner_scalar_func_args=["mo"],
        )
        expr = NormalizedExpr(
            inner_scalar_func="date_trunc",
            inner_scalar_func_args=["weekly"],
            add_groups=[group],
        )
        out = canonicalize_temporal_unit_args(self._make_intent_with_select_expr(expr))
        out_expr = out.select_cols[0].expr
        assert out_expr.inner_scalar_func_args[0] == "week"
        assert out_expr.add_groups[0].scalar_func_args[0] == "year"
        assert out_expr.add_groups[0].inner_scalar_func_args[0] == "month"

    def test_canonicalizes_in_cte_filter(self):
        """CTE-scoped filter exprs are also walked."""
        from aetherdialect._contracts_core import RuntimeCteStep

        cte_filter = FilterParam(
            left_expr=NormalizedExpr(
                scalar_func="date_trunc",
                scalar_func_args=["annually"],
                add_groups=[MulGroup(multiply=["t.created_at"])],
            ),
            op="=",
            value_type="date",
            raw_value="2024-01-01",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_filter],
            having_param=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        out = canonicalize_temporal_unit_args(intent)
        assert out.cte_steps[0].filters_param[0].left_expr.scalar_func_args[0] == "year"

    def test_date_diff_filter_alias_normalization_via_normalize_date_diff(self):
        """`normalize_date_diff_raw_values` also handles aliases like `monthly`."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "monthly", "amount": 3},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_date_diff_raw_values(intent)
        assert result.filters_param[0].raw_value["unit"] == "month"


class TestClassifyCteExpr:
    """Tests for classify_cte_expr."""

    def test_passthrough(self):
        """Bare column is passthrough."""
        expr = NormalizedExpr.from_column("t.a")
        assert classify_cte_expr(expr) == "passthrough"

    def test_aggregation(self):
        """COUNT(col) is aggregation."""
        expr = NormalizedExpr.from_agg("count", "t.a")
        assert classify_cte_expr(expr) == "aggregation"

    def test_scalar(self):
        """Scalar function without agg is scalar."""
        expr = NormalizedExpr.from_column("t.a")
        expr = replace(expr, scalar_func="upper")
        assert classify_cte_expr(expr) == "scalar"


class TestDeriveCteOutputColumns:
    """Tests for derive_cte_output_columns."""

    def test_passthrough_column(self):
        """Derive column name from passthrough expression."""
        sc = SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))
        result = derive_cte_output_columns([sc])
        assert result == ["customer_id"]

    def test_passthrough_preserves_mixed_case_name(self):
        """Passthrough uses the bare column token without lowercasing."""
        sc = SelectCol(expr=NormalizedExpr.from_column("film.FilmId"))
        assert derive_cte_output_columns([sc]) == ["FilmId"]

    def test_count_star(self):
        """Derive row_count from COUNT(*)."""
        expr = NormalizedExpr.from_agg("count", "*")
        sc = SelectCol(expr=expr)
        result = derive_cte_output_columns([sc])
        assert result == ["row_count"]

    def test_sum_column(self):
        """Derive sum_amount from SUM(orders.amount)."""
        expr = NormalizedExpr.from_agg("sum", "orders.amount")
        sc = SelectCol(expr=expr)
        result = derive_cte_output_columns([sc])
        assert result == ["sum_amount"]

    def test_duplicate_names_suffixed(self):
        """Duplicate derived names get numeric suffixes."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("t.id"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("t2.id"))
        result = derive_cte_output_columns([sc1, sc2])
        assert result[0] == "id"
        assert result[1] == "id_2"


class TestTagSingleExpr:
    """Tests for _tag_single_expr."""

    @pytest.fixture
    def num_schema(self):
        """Schema with numeric and non-numeric columns."""
        t = TableMetadata(
            name="t",
            columns={
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="numeric",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                ),
                "name": ColumnMetadata(name="name", data_type="varchar", role=ColumnRole.CATEGORICAL.value),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"t": t}, join_paths_multi={}, effective_structural_hash="test")

    def test_numeric_sets_is_numeric_true(self, num_schema):
        """_tag_single_expr sets is_numeric True for numeric column."""
        expr = NormalizedExpr.from_column("t.amount")
        result = _tag_single_expr(expr, num_schema)
        assert result.is_numeric is True

    def test_numeric_injects_offset(self, num_schema):
        """_tag_single_expr injects ExprValue(0.0) offset for numeric expression without add_values."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.amount"])])
        result = _tag_single_expr(expr, num_schema)
        assert len(result.add_values) == 1
        assert result.add_values[0].value == 0.0

    def test_numeric_preserves_existing_values(self, num_schema):
        """_tag_single_expr preserves existing add_values for numeric expression."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.amount"])],
            add_values=[ExprValue(value=5.0)],
        )
        result = _tag_single_expr(expr, num_schema)
        assert len(result.add_values) == 1
        assert result.add_values[0].value == 5.0

    def test_numeric_preserves_coefficient(self, num_schema):
        """_tag_single_expr preserves coefficient for numeric expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=3.5, multiply=["t.amount"])])
        result = _tag_single_expr(expr, num_schema)
        assert result.add_groups[0].coefficient == 3.5

    def test_non_numeric_sets_is_numeric_false(self, num_schema):
        """_tag_single_expr sets is_numeric False for non-numeric column."""
        expr = NormalizedExpr.from_column("t.name")
        result = _tag_single_expr(expr, num_schema)
        assert result.is_numeric is False

    def test_non_numeric_resets_coefficient(self, num_schema):
        """_tag_single_expr resets coefficient to 1.0 for non-numeric expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.5, multiply=["t.name"])])
        result = _tag_single_expr(expr, num_schema)
        assert result.add_groups[0].coefficient == 1.0

    def test_non_numeric_clears_coeff_param_key(self, num_schema):
        """_tag_single_expr clears coeff_param_key for non-numeric expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.name"], coeff_param_key="p1")])
        result = _tag_single_expr(expr, num_schema)
        assert result.add_groups[0].coeff_param_key == ""

    def test_non_numeric_clears_add_values(self, num_schema):
        """_tag_single_expr clears add_values for non-numeric expression."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.name"])],
            add_values=[ExprValue(value=10.0)],
        )
        result = _tag_single_expr(expr, num_schema)
        assert result.add_values == []

    def test_non_numeric_clears_sub_values(self, num_schema):
        """_tag_single_expr clears sub_values for non-numeric expression."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.name"])],
            sub_values=[ExprValue(value=5.0)],
        )
        result = _tag_single_expr(expr, num_schema)
        assert result.sub_values == []

    def test_non_numeric_preserves_multiply(self, num_schema):
        """_tag_single_expr preserves multiply list for non-numeric expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.name"])])
        result = _tag_single_expr(expr, num_schema)
        assert [m.column_ref for m in result.add_groups[0].multiply] == ["t.name"]

    def test_non_numeric_preserves_agg_func(self, num_schema):
        """_tag_single_expr preserves agg_func for non-numeric count expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.name"], agg_func="count")])
        result = _tag_single_expr(expr, num_schema)
        assert result.add_groups[0].agg_func == "count"

    def test_skip_value_injection_numeric(self, num_schema):
        """_tag_single_expr with skip_value_injection tags numeric but skips ExprValue injection."""
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.amount"])])
        result = _tag_single_expr(expr, num_schema, skip_value_injection=True)
        assert result.is_numeric is True
        assert result.add_values == []

    def test_skip_value_injection_preserves_existing_values(self, num_schema):
        """_tag_single_expr with skip_value_injection preserves pre- existing add_values."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.amount"])],
            add_values=[ExprValue(value=5.0)],
        )
        result = _tag_single_expr(expr, num_schema, skip_value_injection=True)
        assert result.is_numeric is True
        assert result.add_values == [ExprValue(value=5.0)]

    def test_skip_value_injection_non_numeric(self, num_schema):
        """_tag_single_expr with skip_value_injection still sanitizes non-numeric."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=2.0, multiply=["t.name"])],
            add_values=[ExprValue(value=10.0)],
        )
        result = _tag_single_expr(expr, num_schema, skip_value_injection=True)
        assert result.is_numeric is False
        assert result.add_groups[0].coefficient == 1.0
        assert result.add_values == []


class TestTagExprNumeric:
    """Tests for tag_expr_numeric."""

    @pytest.fixture
    def mixed_schema(self):
        """Schema with numeric and non-numeric columns."""
        t = TableMetadata(
            name="t",
            columns={
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="numeric",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                ),
                "name": ColumnMetadata(name="name", data_type="varchar", role=ColumnRole.CATEGORICAL.value),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"t": t}, join_paths_multi={}, effective_structural_hash="test")

    def test_tags_select_cols(self, mixed_schema):
        """tag_expr_numeric tags select column expressions."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.amount"])])),
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=3.0, multiply=["t.name"])])),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.select_cols[0].expr.is_numeric is True
        assert result.select_cols[0].expr.add_groups[0].coefficient == 2.0
        assert result.select_cols[1].expr.is_numeric is False
        assert result.select_cols[1].expr.add_groups[0].coefficient == 1.0

    def test_tags_filter_exprs(self, mixed_schema):
        """tag_expr_numeric tags filter expressions."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=5.0, multiply=["t.name"])]),
                    op="=",
                    value_type="string",
                ),
            ],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.filters_param[0].left_expr.is_numeric is False
        assert result.filters_param[0].left_expr.add_groups[0].coefficient == 1.0

    def test_clears_non_numeric_values_in_intent(self, mixed_schema):
        """tag_expr_numeric clears add_values on non-numeric select expression."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=1.0, multiply=["t.name"])],
                        add_values=[ExprValue(value=10.0)],
                    )
                ),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.select_cols[0].expr.add_values == []

    def test_numeric_agg_keeps_coefficient(self, mixed_schema):
        """tag_expr_numeric keeps coefficient for count aggregation (numeric result)."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.name"], agg_func="count")])
                ),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.select_cols[0].expr.is_numeric is True
        assert result.select_cols[0].expr.add_groups[0].coefficient == 2.0

    def test_filter_left_expr_skips_injection(self, mixed_schema):
        """tag_expr_numeric does not inject ExprValue offset into filter left_expr."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.amount"])]),
                    op=">",
                    value_type="number",
                ),
            ],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.filters_param[0].left_expr.is_numeric is True
        assert result.filters_param[0].left_expr.add_values == []

    def test_filter_right_expr_gets_injection(self, mixed_schema):
        """tag_expr_numeric injects ExprValue offset into filter right_expr when numeric."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["t.amount"])]),
                    op=">",
                    value_type="number",
                    right_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["t.amount"])]),
                ),
            ],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.filters_param[0].right_expr.is_numeric is True
        assert len(result.filters_param[0].right_expr.add_values) == 1
        assert result.filters_param[0].right_expr.add_values[0].value == 0.0

    def test_having_left_expr_skips_injection(self, mixed_schema):
        """tag_expr_numeric does not inject ExprValue offset into having left_expr."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=2.0, multiply=["t.amount"], agg_func="sum")]
                    ),
                    op=">",
                    value_type="number",
                ),
            ],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.having_param[0].left_expr.is_numeric is True
        assert result.having_param[0].left_expr.add_values == []

    def test_cte_filter_left_expr_skips_injection(self, mixed_schema):
        """tag_expr_numeric does not inject ExprValue offset into CTE filter left_expr."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.amount"])]),
                    op=">",
                    value_type="number",
                ),
            ],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = tag_expr_numeric(intent, mixed_schema)
        assert result.cte_steps[0].filters_param[0].left_expr.is_numeric is True
        assert result.cte_steps[0].filters_param[0].left_expr.add_values == []


class TestIsNontrivialGroup:
    """Tests for _is_nontrivial_group."""

    def test_trivial_group(self):
        """Bare column with coefficient 1.0 is trivial."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"])
        assert _is_nontrivial_group(g) is False

    def test_nontrivial_agg(self):
        """Group with agg_func is nontrivial."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], agg_func="count")
        assert _is_nontrivial_group(g) is True

    def test_nontrivial_coefficient(self):
        """Group with non-unit coefficient is nontrivial."""
        g = MulGroup(coefficient=2.0, multiply=["t.a"])
        assert _is_nontrivial_group(g) is True

    def test_nontrivial_scalar_func(self):
        """Group with scalar_func is nontrivial."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round")
        assert _is_nontrivial_group(g) is True

    def test_nontrivial_divide(self):
        """Group with divide list is nontrivial."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], divide=["t.b"])
        assert _is_nontrivial_group(g) is True


class TestAssignStructuralGroup:
    """Tests for _assign_structural_group."""

    def test_trivial_group_no_assignment(self):
        """Trivial group gets no structural param."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"])
        pv = {}
        idx = _assign_structural_group(g, 1, pv)
        assert idx == 1
        assert pv == {}

    def test_nontrivial_assigns_coeff_key(self):
        """Nontrivial group gets coefficient param key."""
        g = MulGroup(coefficient=2.5, multiply=["t.a"], agg_func="sum")
        pv = {}
        idx = _assign_structural_group(g, 1, pv)
        assert g.coeff_param_key == "s1"
        assert pv["s1"] == 2.5
        assert idx == 2

    def test_scalar_func_args(self):
        """Scalar func args get param keys; unit coefficient is skipped."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round", scalar_func_args=[2])
        pv = {}
        idx = _assign_structural_group(g, 1, pv)
        assert g.coeff_param_key == ""
        assert pv["s1"] == 2
        assert idx == 2

    def test_non_numeric_skips_coeff(self):
        """Non-numeric group skips coefficient assignment."""
        g = MulGroup(coefficient=3.0, multiply=["t.a"], agg_func="count")
        pv = {}
        _assign_structural_group(g, 1, pv, is_numeric=False)
        assert g.coeff_param_key == ""
        assert "s1" not in pv


class TestAssignStructuralExpr:
    """Tests for _assign_structural_expr."""

    def test_add_values_get_keys(self):
        """ExprValue in add_values gets param key."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"])],
            add_values=[ExprValue(value=5.0)],
            is_numeric=True,
        )
        pv = {}
        _assign_structural_expr(expr, 1, pv)
        assert expr.add_values[0].param_key == "s1"
        assert pv["s1"] == 5.0

    def test_sub_values_get_keys(self):
        """ExprValue in sub_values gets param key."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"])],
            sub_values=[ExprValue(value=3.0)],
            is_numeric=True,
        )
        pv = {}
        _assign_structural_expr(expr, 1, pv)
        assert expr.sub_values[0].param_key == "s1"
        assert pv["s1"] == 3.0

    def test_nontrivial_group_and_values(self):
        """Both group coefficient and value get sequential keys."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=2.0, multiply=["t.a"], agg_func="sum")],
            add_values=[ExprValue(value=10.0)],
            is_numeric=True,
        )
        pv = {}
        idx = _assign_structural_expr(expr, 1, pv)
        assert pv["s1"] == 2.0
        assert pv["s2"] == 10.0
        assert idx == 3

    def test_non_numeric_skips_expr_values(self):
        """Non-numeric expr skips ExprValue param assignment for add_values and sub_values."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"])],
            add_values=[ExprValue(value=5.0)],
            sub_values=[ExprValue(value=3.0)],
            is_numeric=False,
        )
        pv = {}
        idx = _assign_structural_expr(expr, 1, pv)
        assert pv == {}
        assert expr.add_values[0].param_key == ""
        assert expr.sub_values[0].param_key == ""
        assert idx == 1

    def test_scalar_func_args_skipped_when_not_numeric(self):
        """Non-numeric exprs do not get structural sN slots for scalar function args."""
        expr = NormalizedExpr(
            add_groups=[
                MulGroup(
                    coefficient=1.0,
                    multiply=["t.a"],
                    scalar_func="round",
                    scalar_func_args=[2],
                )
            ],
            is_numeric=False,
        )
        pv = {}
        idx = _assign_structural_expr(expr, 1, pv)
        assert pv == {}
        assert idx == 1


class TestIsExprNumeric:
    """Tests for _is_expr_numeric."""

    @pytest.fixture
    def num_schema(self):
        """Schema with numeric and non-numeric columns."""
        t = TableMetadata(
            name="t",
            columns={
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="numeric",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                    value_type="number",
                ),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    role=ColumnRole.CATEGORICAL.value,
                    value_type="string",
                ),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"t": t}, join_paths_multi={}, effective_structural_hash="test")

    def test_count_agg_is_numeric(self, num_schema):
        """COUNT agg returns numeric."""
        expr = NormalizedExpr.from_agg("count", "t.amount")
        assert _is_expr_numeric(expr, num_schema) is True

    def test_scalar_extract_is_numeric(self, num_schema):
        """EXTRACT scalar returns numeric."""
        expr = NormalizedExpr.from_column("t.amount")
        expr = replace(expr, scalar_func="extract")
        assert _is_expr_numeric(expr, num_schema) is True

    def test_numeric_column_by_value_type(self, num_schema):
        """Column with value_type number is numeric."""
        expr = NormalizedExpr.from_column("t.amount")
        assert _is_expr_numeric(expr, num_schema) is True

    def test_non_numeric_column(self, num_schema):
        """Column with value_type string is not numeric."""
        expr = NormalizedExpr.from_column("t.name")
        assert _is_expr_numeric(expr, num_schema) is False

    def test_multi_group_is_numeric(self, num_schema):
        """Multiple add/sub groups implies numeric when column unknown."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["unknown.col"])],
            sub_groups=[MulGroup(coefficient=1.0, multiply=["unknown.col2"])],
        )
        assert _is_expr_numeric(expr, num_schema) is True


class TestFillGroupDefaults:
    """Tests for _fill_group_defaults."""

    def test_round_gets_default_args(self):
        """Round scalar_func gets [2] default."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round")
        _fill_group_defaults(g)
        assert g.scalar_func_args == [2]

    def test_date_trunc_infers_unit(self):
        """date_trunc infers unit from column name."""
        g = MulGroup(coefficient=1.0, multiply=["t.created_year"], scalar_func="date_trunc")
        _fill_group_defaults(g)
        assert g.scalar_func_args == ["year"]

    def test_no_overwrite_existing_args(self):
        """Existing scalar_func_args are not overwritten."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round", scalar_func_args=[4])
        _fill_group_defaults(g)
        assert g.scalar_func_args == [4]

    def test_no_scalar_func_no_change(self):
        """Group without scalar_func unchanged."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"])
        _fill_group_defaults(g)
        assert g.scalar_func_args == []

    def test_inner_scalar_func_default(self):
        """Inner scalar func gets defaults."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], inner_scalar_func="round")
        _fill_group_defaults(g)
        assert g.inner_scalar_func_args == [2]


class TestFillExprDefaults:
    """Tests for _fill_expr_defaults."""

    def test_expr_scalar_func_default(self):
        """Expr-level scalar_func gets defaults."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"])],
            scalar_func="round",
        )
        _fill_expr_defaults(expr)
        assert expr.scalar_func_args == [2]

    def test_fills_child_groups(self):
        """Fills child MulGroup scalar_func defaults."""
        g = MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round")
        expr = NormalizedExpr(add_groups=[g])
        _fill_expr_defaults(expr)
        assert g.scalar_func_args == [2]

    def test_date_trunc_expr_level(self):
        """Expr-level date_trunc infers unit."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.order_month"])],
            scalar_func="date_trunc",
        )
        _fill_expr_defaults(expr)
        assert expr.scalar_func_args == ["month"]


class TestEnsureScalarFuncDefaults:
    """Tests for ensure_scalar_func_defaults."""

    def test_fills_main_select_defaults(self):
        """ensure_scalar_func_defaults fills select col defaults."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round")],
                    )
                )
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = ensure_scalar_func_defaults(intent)
        assert result.select_cols[0].expr.add_groups[0].scalar_func_args == [2]

    def test_fills_cte_defaults(self):
        """ensure_scalar_func_defaults fills CTE select col defaults."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"], scalar_func="round")],
                    )
                )
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = ensure_scalar_func_defaults(intent)
        assert result.cte_steps[0].select_cols[0].expr.add_groups[0].scalar_func_args == [2]

    def test_returns_same_intent(self):
        """ensure_scalar_func_defaults returns same intent object."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = ensure_scalar_func_defaults(intent)
        assert result is intent


class TestExtractStructuralParams:
    """Tests for extract_structural_params."""

    def test_limit_gets_param(self):
        """extract_structural_params assigns limit param key."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            limit=10,
        )
        result = extract_structural_params(intent)
        assert result.limit_param_key == "s1"
        assert result.param_values["s1"] == 10

    def test_no_limit_no_key(self):
        """extract_structural_params no limit_param_key when limit is None."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = extract_structural_params(intent)
        assert result.limit_param_key == ""

    def test_coefficient_in_select(self):
        """extract_structural_params assigns coefficient param in select."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=2.0, multiply=["t.a"], agg_func="sum")],
                        is_numeric=True,
                    )
                )
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = extract_structural_params(intent)
        assert "s1" in result.param_values
        assert result.param_values["s1"] == 2.0

    def test_case_when_literal_structural_keys_align_with_render_order(self):
        """CASE branch literals must not reuse structural keys from an unused ``sc.expr`` shell."""
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=FilterParam(
                        left_expr=NormalizedExpr.from_column("film.rental_rate"),
                        op=">",
                        right_expr=NormalizedExpr(add_values=[ExprValue(value=3)], is_numeric=True),
                        value_type="number",
                    ),
                    result=NormalizedExpr(add_values=[ExprValue(value="premium")], is_numeric=False),
                )
            ],
            else_result=NormalizedExpr(add_values=[ExprValue(value="standard")], is_numeric=False),
        )
        step = CaseRegistryStep(registry_id="c01", case_when=cw)
        case_col = SelectCol(expr=NormalizedExpr.from_column("c01"))
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[case_col],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[step],
        )
        out = extract_structural_params(intent)
        sc0 = out.select_cols[0]
        with registry_render_scope(None, out.case_registry):
            sql_frag = render_select_col_sql(sc0)
        final = substitute_params(sql_frag, dict(out.param_values))
        assert "> THEN" not in final
        compact = re.sub(r"\s+", "", final)
        assert "rental_rate>3" in compact


class TestDecomposeBetweenFiltersEdgeCases:
    """Edge-case tests for decompose_between_params."""

    def test_between_with_string_value(self):
        """Non-list raw_value between still decomposes into >= and <=."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="between",
            value_type="string",
            raw_value="invalid",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert len(result.filters_param) == 2
        assert result.filters_param[0].op == ">="
        assert result.filters_param[1].op == "<="

    def test_empty_filters(self):
        """Empty filters returns empty."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert result.filters_param == []


class TestAssignParamKeysEdgeCases:
    """Edge-case tests for assign_param_keys."""

    def test_empty_inputs(self):
        """Empty filters and having yield empty results."""
        fp, hp, cte, _, idx = assign_param_keys([], [])
        assert fp == []
        assert hp == []
        assert idx == 1

    def test_is_not_null_skipped(self):
        """IS NOT NULL filters skip param_key."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="is not null",
            value_type="null",
        )
        new_fp, _, _, _, idx = assign_param_keys([fp], [])
        assert new_fp[0].param_key == ""
        assert idx == 1

    def test_mixed_null_and_regular(self):
        """Mix of is null and regular filters."""
        fp1 = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="is null", value_type="null")
        fp2 = FilterParam(left_expr=NormalizedExpr.from_column("t.b"), op="=", value_type="string")
        new_fp, _, _, _, idx = assign_param_keys([fp1, fp2], [])
        assert new_fp[0].param_key == ""
        assert new_fp[1].param_key == "p1"
        assert idx == 2

    def test_expr_vs_expr_filter_skips_param_key(self):
        """Filter with right_expr skips param_key assignment."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="number",
            right_expr=NormalizedExpr.from_column("t.b"),
        )
        new_fp, _, _, _, idx = assign_param_keys([fp], [])
        assert new_fp[0].param_key == ""
        assert idx == 1

    def test_expr_vs_expr_having_skips_param_key(self):
        """Having with right_expr skips param_key assignment."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "t.a"),
            op=">",
            value_type="number",
            right_expr=NormalizedExpr.from_agg("sum", "t.b"),
        )
        _, new_hp, _, _, idx = assign_param_keys([], [hp])
        assert new_hp[0].param_key == ""
        assert idx == 1

    def test_expr_vs_expr_cte_filter_skips_param_key(self):
        """CTE filter with right_expr skips param_key assignment."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="number",
            right_expr=NormalizedExpr.from_column("t.b"),
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        _, _, new_ctes, _, idx = assign_param_keys([], [], cte_steps=[cte])
        assert new_ctes[0].filters_param[0].param_key == ""
        assert idx == 1

    def test_expr_vs_expr_mixed_with_regular(self):
        """Expr-vs-expr filter followed by regular filter gets correct keys."""
        fp1 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="number",
            right_expr=NormalizedExpr.from_column("t.b"),
        )
        fp2 = FilterParam(left_expr=NormalizedExpr.from_column("t.c"), op="=", value_type="string")
        new_fp, _, _, _, idx = assign_param_keys([fp1, fp2], [])
        assert new_fp[0].param_key == ""
        assert new_fp[1].param_key == "p1"
        assert idx == 2


class TestIsDateOrIntervalExpr:
    """Tests for _is_date_or_interval_expr."""

    def test_current_date(self):
        """Recognize CURRENT_DATE."""
        assert _is_date_or_interval_expr("CURRENT_DATE") is True

    def test_current_date_minus_interval(self):
        """Recognize CURRENT_DATE - INTERVAL expression."""
        assert _is_date_or_interval_expr("CURRENT_DATE - INTERVAL '90 days'") is True

    def test_interval_only(self):
        """Recognize INTERVAL expression."""
        assert _is_date_or_interval_expr("INTERVAL '7 days'") is True

    def test_now(self):
        """Recognize NOW()."""
        assert _is_date_or_interval_expr("NOW()") is True

    def test_current_timestamp(self):
        """Recognize CURRENT_TIMESTAMP."""
        assert _is_date_or_interval_expr("CURRENT_TIMESTAMP") is True

    def test_table_column_rejected(self):
        """Reject table.column references."""
        assert _is_date_or_interval_expr("rental.rental_date") is False

    def test_plain_literal_rejected(self):
        """Reject plain literals."""
        assert _is_date_or_interval_expr("42") is False
        assert _is_date_or_interval_expr("foo") is False

    def test_empty_rejected(self):
        """Reject empty or None."""
        assert _is_date_or_interval_expr("") is False
        assert _is_date_or_interval_expr(None) is False


class TestParseIntentResponse:
    """Tests for parse_intent_response."""

    def test_valid_minimal_json(self):
        """Parses minimal valid JSON into RuntimeIntent."""
        raw = '{"tables": ["orders"], "grain": "row_level", "select_cols": ["orders.order_id"]}'
        result = parse_intent_response(raw, "list orders")
        assert result is not None
        assert result.tables == ["orders"]
        assert len(result.select_cols) == 1

    def test_returns_none_for_garbage(self):
        """Returns None for unparseable text."""
        assert parse_intent_response("not json at all", "q") is None

    def test_returns_none_for_empty(self):
        """Returns None for empty string."""
        assert parse_intent_response("", "q") is None

    def test_parses_filters(self):
        """Parses filters_param with left_expr and op."""
        raw = '{"tables": ["orders"], "select_cols": ["orders.order_id"], "filters_param": [{"left_expr": "orders.status", "op": "=", "value": "shipped", "value_type": "string"}]}'
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert len(result.filters_param) == 1
        assert result.filters_param[0].op == "="

    def test_parses_having(self):
        """Parses having_param with left_expr and op."""
        raw = '{"tables": ["orders"], "select_cols": ["orders.order_id"], "having_param": [{"left_expr": "count(orders.order_id)", "op": ">", "value": 5, "value_type": "integer"}]}'
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert len(result.having_param) == 1

    def test_parses_order_by(self):
        """Parses order_by_cols with direction."""
        raw = '{"tables": ["orders"], "select_cols": ["orders.order_id"], "order_by_cols": [{"expr": "orders.amount", "direction": "desc"}]}'
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert len(result.order_by_cols) == 1
        assert result.order_by_cols[0].direction.lower() == "desc"

    def test_parses_order_by_expr_as_normalized_dict(self):
        """order_by_cols object with dict-shaped expr does not require a string strip path."""
        expr = {
            "add_groups": [{"multiply": ["orders.amount"], "divide": []}],
            "sub_groups": [],
        }
        raw = json.dumps(
            {
                "tables": ["orders"],
                "grain": "row_level",
                "select_cols": ["orders.order_id"],
                "order_by_cols": [{"expr": expr, "direction": "desc"}],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert len(result.order_by_cols) == 1
        assert result.order_by_cols[0].expr.primary_column == "orders.amount"
        assert result.order_by_cols[0].direction.lower() == "desc"

    def test_tables_as_string(self):
        """Handles tables provided as a single string instead of list."""
        raw = '{"tables": "orders", "select_cols": ["orders.order_id"]}'
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.tables == ["orders"]

    def test_skips_filter_without_left_expr(self):
        """Filters with no left_expr are skipped."""
        raw = '{"tables": ["t"], "select_cols": ["t.x"], "filters_param": [{"op": "=", "value": "a"}]}'
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert len(result.filters_param) == 0

    def test_filter_bool_op_parsed(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.a",
                        "op": "=",
                        "value": "1",
                        "value_type": "integer",
                    },
                    {
                        "left_expr": "t.b",
                        "op": ">",
                        "value": "2",
                        "value_type": "integer",
                        "bool_op": "OR",
                    },
                ],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.filters_param[0].bool_op == "AND"
        assert result.filters_param[1].bool_op == "OR"

    def test_filter_bool_op_defaults_when_missing(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.a",
                        "op": "=",
                        "value": "1",
                        "value_type": "integer",
                    },
                ],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.filters_param[0].bool_op == "AND"

    def test_filter_group_parsed(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.a",
                        "op": "=",
                        "value": "1",
                        "value_type": "integer",
                        "filter_group": 1,
                    },
                    {
                        "left_expr": "t.b",
                        "op": ">",
                        "value": "2",
                        "value_type": "integer",
                        "filter_group": 1,
                    },
                ],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.filters_param[0].filter_group == 1
        assert result.filters_param[1].filter_group == 1

    def test_filter_group_defaults_to_none(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.a",
                        "op": "=",
                        "value": "1",
                        "value_type": "integer",
                    },
                ],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.filters_param[0].filter_group is None

    def test_having_bool_op_parsed(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "having_param": [
                    {
                        "left_expr": "count(t.x)",
                        "op": ">",
                        "value": 5,
                        "value_type": "integer",
                    },
                    {
                        "left_expr": "sum(t.y)",
                        "op": "<",
                        "value": 10,
                        "value_type": "integer",
                        "bool_op": "OR",
                    },
                ],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.having_param[0].bool_op == "AND"
        assert result.having_param[1].bool_op == "OR"

    def test_having_filter_group_parsed(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "having_param": [
                    {
                        "left_expr": "count(t.x)",
                        "op": ">",
                        "value": 5,
                        "value_type": "integer",
                        "filter_group": 2,
                    },
                ],
            }
        )
        result = parse_intent_response(raw, "q")
        assert result is not None
        assert result.having_param[0].filter_group == 2

    def test_filter_preserves_date_interval_right_expr(self):
        """right_expr with date/interval expression is preserved, not cleared."""
        raw = json.dumps(
            {
                "tables": ["rental"],
                "select_cols": [{"expr": "rental.rental_id"}],
                "group_by_cols": [],
                "order_by_cols": [],
                "filters_param": [
                    {
                        "left_expr": "rental.rental_date",
                        "op": ">=",
                        "right_expr": "CURRENT_DATE - INTERVAL '90 days'",
                    }
                ],
                "having_param": [],
                "limit": None,
                "cte_steps": [],
                "natural_language": "rentals in last 90 days",
            }
        )
        result = parse_intent_response(raw, "rentals in last 90 days")
        assert result is not None
        assert len(result.filters_param) == 1
        fp = result.filters_param[0]
        assert fp.right_expr is not None

    def test_filter_clears_plain_literal_right_expr(self):
        """right_expr without table.column and not date/interval is cleared."""
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": [{"expr": "t.x"}],
                "group_by_cols": [],
                "order_by_cols": [],
                "filters_param": [
                    {
                        "left_expr": "t.a",
                        "op": "=",
                        "right_expr": "42",
                    }
                ],
                "having_param": [],
                "limit": None,
                "cte_steps": [],
                "natural_language": "test",
            }
        )
        result = parse_intent_response(raw, "test")
        assert result is not None
        assert len(result.filters_param) == 1
        assert result.filters_param[0].right_expr is None

    def test_parse_window_spec_on_select_col(self):
        """LLM JSON with ``window_registry`` and bare ``w01`` ref is parsed into select and registry."""
        raw = json.dumps(
            {
                "tables": ["orders"],
                "select_cols": [{"expr": "w01"}],
                "window_registry": [
                    {
                        "registry_id": "w01",
                        "label": "",
                        "window_spec": {
                            "function": "row_number",
                            "partition_by": ["orders.customer_id"],
                            "order_by": [{"expr": "orders.amount", "direction": "desc"}],
                        },
                    }
                ],
                "group_by_cols": [],
                "order_by_cols": [],
                "filters_param": [],
                "having_param": [],
                "limit": None,
                "cte_steps": [],
                "natural_language": "test",
            }
        )
        result = parse_intent_response(raw, "test")
        assert result is not None
        assert len(result.window_registry) == 1
        wr = result.window_registry[0]
        assert wr.registry_id == "w01"
        assert wr.window_spec.function == "row_number"
        assert len(wr.window_spec.partition_by) == 1
        assert len(wr.window_spec.order_by) == 1
        assert result.select_cols[0].expr.primary_term == "w01"

    def test_parse_case_registry_col(self):
        """LLM JSON with ``case_registry`` and bare ``c01`` select expr parses CASE bodies."""
        raw = json.dumps(
            {
                "tables": ["orders"],
                "select_cols": [{"expr": "c01"}],
                "case_registry": [
                    {
                        "registry_id": "c01",
                        "case_when": {
                            "branches": [
                                {
                                    "condition": {
                                        "left_expr": "orders.amount",
                                        "op": ">",
                                        "value_type": "number",
                                        "value": 100,
                                    },
                                    "result": "1",
                                }
                            ],
                            "else_result": "0",
                        },
                    }
                ],
                "group_by_cols": [],
                "order_by_cols": [],
                "filters_param": [],
                "having_param": [],
                "limit": None,
                "cte_steps": [],
                "natural_language": "test",
            }
        )
        result = parse_intent_response(raw, "test")
        assert result is not None
        cw = result.case_registry[0].case_when
        assert cw is not None
        assert len(cw.branches) == 1
        assert cw.else_result is not None


class TestBuildCteOutputMetadata:
    """Tests for build_cte_output_metadata."""

    def test_passthrough_column(self, schema_graph):
        """Passthrough column inherits role and type from source."""
        sc = SelectCol(expr=NormalizedExpr.from_column("orders.amount"))
        out_cols = ["amount"]
        result = build_cte_output_metadata([sc], out_cols, schema_graph)
        assert "amount" in result
        assert result["amount"].data_type == "numeric"

    def test_aggregation_column(self, schema_graph):
        """Aggregation column gets numeric_measure role."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id"))
        out_cols = ["order_count"]
        result = build_cte_output_metadata([sc], out_cols, schema_graph)
        assert "order_count" in result
        meta = result["order_count"]
        assert meta.role == "numeric_measure"
        assert meta.data_type == "integer"

    def test_empty_inputs(self, schema_graph):
        """Returns empty dict for empty inputs."""
        assert build_cte_output_metadata([], [], schema_graph) == {}

    def test_output_shorter_than_select(self, schema_graph):
        """Stops at output_columns length when shorter than select_cols."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.amount"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("orders.status"))
        result = build_cte_output_metadata([sc1, sc2], ["amount"], schema_graph)
        assert len(result) == 1

    def test_avg_aggregation_type(self, schema_graph):
        """AVG aggregation produces numeric data type."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("avg", "orders.amount"))
        out_cols = ["avg_amount"]
        result = build_cte_output_metadata([sc], out_cols, schema_graph)
        assert result["avg_amount"].data_type == "numeric"


class TestDecomposeBetweenHaving:
    """Tests for decompose_between_params on having_param."""

    def test_having_between_with_list(self):
        """BETWEEN having with [low, high] decomposes into >= and <=."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op="between",
            value_type="integer",
            raw_value=[5, 10],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert len(result.having_param) == 2
        assert result.having_param[0].op == ">="
        assert result.having_param[0].raw_value == 5
        assert result.having_param[1].op == "<="
        assert result.having_param[1].raw_value == 10

    def test_having_non_between_unchanged(self):
        """Non-BETWEEN having passes through unchanged."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            raw_value=5,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert len(result.having_param) == 1
        assert result.having_param[0].op == ">"

    def test_cte_having_between_decomposed(self):
        """BETWEEN having in a CTE step is also decomposed."""
        cte_hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "t.amount"),
            op="between",
            value_type="number",
            raw_value=[100, 500],
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[cte_hp],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        result = decompose_between_params(intent)
        assert len(result.cte_steps[0].having_param) == 2
        assert result.cte_steps[0].having_param[0].op == ">="
        assert result.cte_steps[0].having_param[1].op == "<="


class TestParseBetweenRawValue:
    """Tests for _parse_between_raw_value."""

    def test_list_of_two(self):
        """Two-element list returns (low, high)."""
        assert _parse_between_raw_value([10, 20]) == (10, 20)

    def test_string_with_and(self):
        """String with ' AND ' separator splits correctly."""
        assert _parse_between_raw_value("10 AND 20") == ("10", "20")

    def test_string_with_comma(self):
        """String with comma separator splits correctly."""
        assert _parse_between_raw_value("2020-01-01, 2020-12-31") == (
            "2020-01-01",
            "2020-12-31",
        )

    def test_string_with_dash_separator(self):
        """String with ' - ' separator splits correctly."""
        assert _parse_between_raw_value("5 - 10") == ("5", "10")

    def test_single_element_list_returns_none(self):
        """Single-element list returns None."""
        assert _parse_between_raw_value([42]) is None

    def test_integer_returns_none(self):
        """Plain integer returns None."""
        assert _parse_between_raw_value(42) is None

    def test_unparseable_string_returns_none(self):
        """Unsplittable string returns None."""
        assert _parse_between_raw_value("invalid") is None


class TestNormalizeInFilterRawValues:
    """Tests for normalize_in_raw_values."""

    def test_string_to_list_filter(self):
        """String IN value gets parsed to a list."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="in",
            value_type="string",
            raw_value="R, PG-13",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.filters_param[0].raw_value == ["R", "PG-13"]

    def test_quoted_string_to_list(self):
        """Quoted string IN value gets parsed correctly."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="in",
            value_type="string",
            raw_value="'R','PG-13','NC-17'",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.filters_param[0].raw_value == ["R", "PG-13", "NC-17"]

    def test_list_value_unchanged(self):
        """Already-list raw_value is not modified."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="in",
            value_type="string",
            raw_value=["R", "PG"],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.filters_param[0].raw_value == ["R", "PG"]

    def test_non_in_op_unchanged(self):
        """Non-IN ops are not touched even with string raw_value."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            raw_value="R, PG",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.filters_param[0].raw_value == "R, PG"

    def test_not_in_string_to_list(self):
        """NOT IN string value also gets parsed."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="not in",
            value_type="string",
            raw_value="R, G",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.filters_param[0].raw_value == ["R", "G"]

    def test_single_element_string_unchanged(self):
        """Single-element string is not split into a list."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="in",
            value_type="string",
            raw_value="PG-13",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.filters_param[0].raw_value == "PG-13"

    def test_cte_in_filter_normalised(self):
        """CTE step IN string value gets parsed too."""
        cte_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="in",
            value_type="string",
            raw_value="A, B, C",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        result = normalize_in_raw_values(intent)
        assert result.cte_steps[0].filters_param[0].raw_value == ["A", "B", "C"]


class TestExtractColumnsFromExpr:
    """Tests for extract_columns_from_expr."""

    def test_bare_column(self):
        """extract_columns_from_expr returns bare column."""
        expr = NormalizedExpr.from_column("orders.amount")
        cols = extract_columns_from_expr(expr)
        assert "orders.amount" in cols

    def test_function_wrapped(self):
        """extract_columns_from_expr strips function wrappers."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["UPPER(customers.name)"])])
        cols = extract_columns_from_expr(expr)
        assert "customers.name" in cols

    def test_star_excluded(self):
        """extract_columns_from_expr excludes *."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(*)"])])
        cols = extract_columns_from_expr(expr)
        assert "*" not in cols

    def test_multiple_groups(self):
        """extract_columns_from_expr handles add and sub groups."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"])],
            sub_groups=[MulGroup(multiply=["t.b"])],
        )
        cols = extract_columns_from_expr(expr)
        assert "t.a" in cols
        assert "t.b" in cols

    def test_only_add_values_returns_empty(self):
        """extract_columns_from_expr with only add_values returns empty list."""
        expr = NormalizedExpr(add_groups=[], sub_groups=[], add_values=[ExprValue(value=1.0)])
        cols = extract_columns_from_expr(expr)
        assert cols == []

    def test_nested_parens_stripped_to_column(self):
        """extract_columns_from_expr strips nested parens to get inner column."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["((t.col))"])])
        cols = extract_columns_from_expr(expr)
        assert "t.col" in cols

    def test_distinct_prefix_stripped_from_column_token(self):
        """DISTINCT inside a term normalizes to the qualified column name."""
        expr = NormalizedExpr(
            add_groups=[
                MulGroup(multiply=["DISTINCT customer.customer_id"], agg_func="count"),
            ],
        )
        cols = extract_columns_from_expr(expr)
        assert cols == ["customer.customer_id"]

    def test_unbalanced_open_paren_does_not_raise(self):
        """Malformed LLM terms without a closing parenthesis do not crash extraction."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["SUM(film.rental_rate"])])
        cols = extract_columns_from_expr(expr)
        assert isinstance(cols, list)

    def test_nested_function_still_extracts_inner_column(self):
        """Balanced nested calls still resolve the inner column ref."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["ROUND(AVG(film.rental_rate), 2)"])])
        cols = extract_columns_from_expr(expr)
        assert "film.rental_rate" in cols


class TestRepairMisclassifiedDateDiff:
    """Tests for repair_misclassified_date_diff."""

    def test_plain_column_converted_to_date_window(self):
        """date_diff with a plain column left_expr becomes date_window."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("rental.rental_date"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": 90},
        )
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        repaired = result.filters_param[0]
        assert repaired.value_type == "date_window"
        assert repaired.raw_value == {"unit": "day", "amount": 90}

    def test_subtraction_expr_stays_date_diff(self):
        """date_diff with a subtraction left_expr is not reclassified."""
        fp = FilterParam(
            left_expr=NormalizedExpr(
                add_groups=[MulGroup(multiply=["rental.return_date"])],
                sub_groups=[MulGroup(multiply=["rental.rental_date"])],
            ),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": 7},
        )
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        assert result.filters_param[0].value_type == "date_diff"

    def test_non_date_diff_unchanged(self):
        """Non-date_diff filters pass through unchanged."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            raw_value="hello",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        assert result.filters_param[0].value_type == "string"

    def test_cte_filters_also_repaired(self):
        """date_diff in CTE filters is also reclassified."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.order_date"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "month", "amount": 6},
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        repaired = result.cte_steps[0].filters_param[0]
        assert repaired.value_type == "date_window"
        assert repaired.raw_value == {"unit": "month", "amount": 6}

    def test_missing_amount_stays_date_diff(self):
        """date_diff without amount in raw_value is not rewritten."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("rental.rental_date"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "day"},
        )
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        assert result.filters_param[0].value_type == "date_diff"

    def test_non_dict_raw_value_unchanged(self):
        """Non-dict raw_value does not trigger reclassification."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op="=",
            value_type="date_diff",
            raw_value="nonsense",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        assert result.filters_param[0].value_type == "date_diff"
        assert result.filters_param[0].raw_value == "nonsense"

    def test_default_unit_when_missing_in_raw(self):
        """Reclassified date_window uses 'day' when unit is absent."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.event_date"),
            op=">=",
            value_type="date_diff",
            raw_value={"amount": 3},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            natural_language="test",
        )
        result = repair_misclassified_date_diff(intent)
        assert result.filters_param[0].value_type == "date_window"
        assert result.filters_param[0].raw_value == {"unit": "day", "amount": 3}


class TestNormalizeOrderDirection:
    """Tests for _normalize_order_direction."""

    def test_non_string_defaults_asc(self):
        assert _normalize_order_direction(None) == "asc"
        assert _normalize_order_direction(42) == "asc"

    def test_desc_substring(self):
        assert _normalize_order_direction("DESC") == "desc"
        assert _normalize_order_direction(" descending ") == "desc"

    def test_asc_without_desc(self):
        assert _normalize_order_direction("ASC") == "asc"
        assert _normalize_order_direction("ascending") == "asc"


class TestStripOrderDirection:
    """Tests for _strip_order_direction."""

    def test_non_string(self):
        assert _strip_order_direction(None) == ("", "asc")

    def test_empty_string(self):
        assert _strip_order_direction("   ") == ("", "asc")

    def test_suffix_desc_asc_case_insensitive(self):
        assert _strip_order_direction("orders.amount DESC") == ("orders.amount", "desc")
        assert _strip_order_direction("x asc") == ("x", "asc")

    def test_no_suffix(self):
        assert _strip_order_direction("orders.amount") == ("orders.amount", "asc")


class TestReplaceRefsInExpr:
    """Tests for replace_refs_in_expr."""

    def test_maps_all_multiply_and_divide_terms(self):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["a.x"], divide=["b.y"])],
            sub_groups=[MulGroup(multiply=["c.z"])],
        )
        out = replace_refs_in_expr(expr, lambda s: s.replace(".", "_"))
        assert [m.column_ref for m in out.add_groups[0].multiply] == ["a_x"]
        assert [m.column_ref for m in out.add_groups[0].divide] == ["b_y"]
        assert [m.column_ref for m in out.sub_groups[0].multiply] == ["c_z"]

    def test_remaps_qualified_columns_inside_raw_sql_leaf(self):
        expr = NormalizedExpr(raw_sql="cte1.cte1 > 1")
        out = replace_refs_in_expr(
            expr,
            lambda ref: "cte1.rate" if ref == "cte1.cte1" else ref,
        )
        assert out.raw_sql
        low = out.raw_sql.lower()
        assert "cte1" in low and "rate" in low


class TestValidateIntentSchema:
    """Tests for _validate_intent_schema."""

    def test_valid_minimal(self):
        assert _validate_intent_schema({"tables": ["t"]}) is True

    def test_missing_tables(self):
        assert _validate_intent_schema({"select_cols": ["t.x"]}) is False

    def test_invalid_limit_type_string(self):
        assert _validate_intent_schema({"tables": ["t"], "limit": "10"}) is False


class TestParseIntentResponseExtended:
    """Additional parse_intent_response edge cases (schema-valid JSON only)."""

    def test_returns_none_when_root_not_object(self):
        assert parse_intent_response("[]", "q") is None
        assert parse_intent_response("42", "q") is None

    def test_returns_none_when_schema_invalid(self):
        assert parse_intent_response("{}", "fallback") is None
        bad = '{"tables": ["t"], "limit": "not-an-int"}'
        assert parse_intent_response(bad, "q") is None

    def test_schema_invalid_records_jsonschema_detail(self):
        bad = '{"tables": ["t"], "limit": "not-an-int"}'
        buf: list[str] = []
        assert parse_intent_response(bad, "q", parse_detail_out=buf) is None
        assert buf
        assert "INTENT_SCHEMA validation" in buf[0]

    def test_null_right_expr_on_filter_is_schema_valid(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.y",
                        "op": "=",
                        "right_expr": None,
                        "value": "a",
                        "value_type": "string",
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert len(r.filters_param) == 1
        assert r.filters_param[0].right_expr is None

    def test_filter_group_list_rejected_by_schema(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.y",
                        "op": "=",
                        "value": 1,
                        "value_type": "integer",
                        "filter_group": [1, 2],
                    }
                ],
            }
        )
        buf: list[str] = []
        assert parse_intent_response(raw, "q", parse_detail_out=buf) is None
        assert buf
        assert "INTENT_SCHEMA" in buf[0]

    def test_natural_language_fallback_to_question(self):
        raw = '{"tables": ["t"], "select_cols": ["t.x"]}'
        r = parse_intent_response(raw, "my question text")
        assert r is not None
        assert r.natural_language == "my question text"

    def test_natural_language_from_json_when_present(self):
        raw = '{"tables": ["t"], "select_cols": ["t.x"], "natural_language": "from json"}'
        r = parse_intent_response(raw, "ignored")
        assert r is not None
        assert r.natural_language == "from json"

    def test_grain_grouped_when_group_by_present(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "group_by_cols": ["t.x"],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.grain == "grouped"

    def test_grain_scalar_when_agg_without_group_by(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["count(t.x)"],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.grain == "scalar"

    def test_grain_row_level_without_agg_or_group(self):
        raw = '{"tables": ["t"], "select_cols": ["t.x"]}'
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.grain == "row_level"

    def test_legacy_intent_status_schema_invalid_is_ignored(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "intent_status": "schema_invalid",
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.schema_invalid is False

    def test_formatter_schema_invalid_json_is_ignored(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "schema_invalid": True,
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.schema_invalid is False

    def test_limit_integer_preserved(self):
        raw = json.dumps({"tables": ["t"], "select_cols": ["t.x"], "limit": 25})
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.limit == 25

    def test_filter_left_col_alias(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [{"left_col": "t.a", "op": "=", "value_type": "string", "value": "v"}],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert len(r.filters_param) == 1
        assert r.filters_param[0].left_expr.primary_column == "t.a"

    def test_having_left_agg_and_right_agg_aliases(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "having_param": [
                    {
                        "left_agg": "count(t.x)",
                        "right_agg": "count(t.y)",
                        "op": ">",
                        "value_type": "integer",
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert len(r.having_param) == 1
        hp = r.having_param[0]
        assert hp.left_expr.has_aggregation
        assert hp.right_expr is not None
        assert hp.right_expr.has_aggregation

    def test_filter_preserves_qualified_right_expr(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": [{"expr": "t.x"}],
                "filters_param": [{"left_expr": "t.a", "op": "=", "right_expr": "t.b"}],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.filters_param[0].right_expr is not None
        assert r.filters_param[0].right_expr.primary_column == "t.b"

    def test_order_by_string_with_trailing_desc(self):
        raw = json.dumps(
            {
                "tables": ["orders"],
                "select_cols": ["orders.order_id"],
                "order_by_cols": ["orders.amount DESC"],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.order_by_cols[0].direction == "DESC"
        assert r.order_by_cols[0].expr.primary_column == "orders.amount"

    def test_order_by_dict_direction_overrides_expr_suffix(self):
        raw = json.dumps(
            {
                "tables": ["orders"],
                "select_cols": ["orders.order_id"],
                "order_by_cols": [{"expr": "orders.amount DESC", "direction": "asc"}],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.order_by_cols[0].direction == "ASC"

    def test_cte_explicit_alias_overrides_derived_name(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "cte_steps": [
                    {
                        "cte_name": "c1",
                        "select_cols": [{"expr": "t.a", "alias": "custom_alias"}],
                        "output_columns": ["custom_alias"],
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.cte_steps[0].output_columns == ["custom_alias"]

    def test_cte_output_columns_list_used_when_no_alias(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "cte_steps": [
                    {
                        "cte_name": "c1",
                        "select_cols": [{"expr": "t.col"}],
                        "output_columns": ["from_list"],
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.cte_steps[0].output_columns == ["from_list"]

    def test_select_col_empty_expr_string_yields_empty_normalized_expr(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": [{"expr": "   "}],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        sc = r.select_cols[0]
        assert sc.expr == NormalizedExpr()

    def test_window_registry_missing_spec_function_defaults_row_number(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": [{"expr": "w01"}],
                "window_registry": [
                    {
                        "registry_id": "w01",
                        "label": "",
                        "window_spec": {"partition_by": ["t.x"]},
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.window_registry[0].window_spec.function == "row_number"


class TestClassifyCteExprExtended:
    """More classify_cte_expr branches."""

    def test_computed_from_multiple_groups(self):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"])],
            sub_groups=[MulGroup(multiply=["t.b"])],
        )
        assert classify_cte_expr(expr) == "computed"

    def test_computed_from_add_values(self):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"])], add_values=[ExprValue(value=1.0)])
        assert classify_cte_expr(expr) == "computed"

    def test_computed_from_divide_operand(self):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"], divide=["t.b"])])
        assert classify_cte_expr(expr) == "computed"

    def test_computed_from_non_unit_coefficient(self):
        expr = NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["t.a"])])
        assert classify_cte_expr(expr) == "computed"

    def test_outer_agg_overrides_inner_scalar_class(self):
        expr = parse_expr_string("SUM(ROUND(t.a, 2))")
        assert classify_cte_expr(expr) == "aggregation"


class TestDeriveCteOutputColumnsExtended:
    """More derive_cte_output_columns cases."""

    def test_max_star_alias(self):
        expr = NormalizedExpr.from_agg("max", "*")
        assert derive_cte_output_columns([SelectCol(expr=expr)]) == ["max_star"]

    def test_computed_sequential_names(self):
        e1 = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"])],
            sub_groups=[MulGroup(multiply=["t.b"])],
        )
        e2 = NormalizedExpr(add_values=[ExprValue(value=1.0)], add_groups=[MulGroup(multiply=["t.c"])])
        out = derive_cte_output_columns([SelectCol(expr=e1), SelectCol(expr=e2)])
        assert out[0] == "expr1"
        assert out[1] == "expr2"


class TestBuildCteOutputMetadataExtended:
    """More build_cte_output_metadata branches."""

    def test_scalar_numeric_extract(self, schema_graph):
        expr = NormalizedExpr.from_column("orders.amount")
        expr = replace(expr, scalar_func="extract")
        sc = SelectCol(expr=expr)
        meta = build_cte_output_metadata([sc], ["ext_amt"], schema_graph)["ext_amt"]
        assert meta.role == "numeric_measure"
        assert meta.data_type == "integer"

    def test_computed_kind_numeric_defaults(self, schema_graph):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["orders.amount"])],
            sub_groups=[MulGroup(multiply=["orders.order_id"])],
        )
        sc = SelectCol(expr=expr)
        m = build_cte_output_metadata([sc], ["c1"], schema_graph)["c1"]
        assert m.source == "computed"
        assert m.role == "numeric_measure"
        assert m.data_type == "numeric"

    def test_sum_inherits_base_sql_type(self, schema_graph):
        sc = SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))
        m = build_cte_output_metadata([sc], ["s"], schema_graph)["s"]
        assert m.data_type == "numeric"


class TestIsExprNumericExtended:
    """Branches in _is_expr_numeric not covered elsewhere."""

    @pytest.fixture
    def schema_no_unknown(self):
        t = TableMetadata(
            name="t",
            columns={
                "x": ColumnMetadata(
                    name="x",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                ),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"t": t}, join_paths_multi={}, effective_structural_hash="x")

    def test_inner_scalar_func_numeric(self, schema_no_unknown):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.x"], inner_scalar_func="extract")],
        )
        assert _is_expr_numeric(expr, schema_no_unknown) is True

    def test_single_unknown_column_not_numeric(self, schema_no_unknown):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["other.y"])])
        assert _is_expr_numeric(expr, schema_no_unknown) is False


class TestAssignParamKeysDateWindow:
    """date_window dict payloads receive p* for the numeric offset and s* for unit after post-processing."""

    def test_date_window_skipped(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op=">=",
            value_type="date_window",
            raw_value={"unit": "day", "amount": 1},
        )
        new_fp, _, _, _, idx = assign_param_keys([fp], [])
        assert new_fp[0].param_key == "p1"
        assert idx == 2


class TestAssignStructuralGroupExtended:
    """Inner scalar args and key reuse."""

    def test_inner_scalar_func_args_get_keys(self):
        g = MulGroup(
            coefficient=1.0,
            multiply=["t.a"],
            inner_scalar_func="round",
            inner_scalar_func_args=[2],
        )
        pv = {}
        idx = _assign_structural_group(g, 1, pv)
        assert g.coeff_param_key == ""
        assert pv.get("s1") == 2
        assert idx == 2

    def test_reuses_existing_sarg_param_keys(self):
        g = MulGroup(
            coefficient=1.0,
            multiply=["t.a"],
            scalar_func="round",
            scalar_func_args=[3],
            sarg_param_keys=["old"],
        )
        pv = {}
        _assign_structural_group(g, 5, pv)
        assert g.coeff_param_key == ""
        assert g.sarg_param_keys[0] == "s5"
        assert pv["s5"] == 3


class TestAssignStructuralExprOuterScalarArgs:
    """Expr-level scalar_func_args in _assign_structural_expr."""

    def test_outer_scalar_and_inner_args(self):
        expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"])],
            scalar_func="round",
            scalar_func_args=[2],
            inner_scalar_func="abs",
            inner_scalar_func_args=[],
            is_numeric=True,
        )
        pv = {}
        idx = _assign_structural_expr(expr, 1, pv)
        assert pv["s1"] == 2
        assert idx == 2


class TestExtractStructuralParamsExtended:
    """CTE limit ordering and param_values merge."""

    def test_cte_limit_key_before_main_limit(self):
        cte = RuntimeCteStep(
            cte_name="c",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            limit=5,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.b"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            limit=10,
            cte_steps=[cte],
        )
        out = extract_structural_params(intent)
        assert cte.limit_param_key == "s1"
        assert out.cte_steps[0].limit_param_key == "s1"
        assert out.limit_param_key == "s2"
        assert out.param_values["s1"] == 5
        assert out.param_values["s2"] == 10

    def test_merges_existing_param_values(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            param_values={"p0": "keep"},
        )
        out = extract_structural_params(intent)
        assert out.param_values["p0"] == "keep"


class TestClearedParamAndStructuralOrder:
    """cleared_param_runtime_intent and structural_s_key_assignment_order."""

    def test_cleared_copies_and_clears_values(self):
        cte = RuntimeCteStep(
            cte_name="c",
            select_cols=[],
            param_values={"s9": 1},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            param_values={"s1": 2},
            cte_steps=[cte],
            limit=3,
            limit_param_key="s2",
        )
        cleared = cleared_param_runtime_intent(intent)
        assert cleared is not intent
        assert cleared.param_values == {}
        assert cleared.limit_param_key == ""
        assert cleared.cte_steps[0].param_values == {}

    def test_structural_order_matches_extract(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=2.0, multiply=["t.a"], agg_func="sum")],
                        is_numeric=True,
                    )
                )
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            limit=7,
        )
        keys = structural_s_key_assignment_order(intent)
        full = extract_structural_params(cleared_param_runtime_intent(intent))
        assert [k for k in full.param_values if k.startswith("s")] == keys


class TestCollectRawParamValues:
    """Tests for collect_raw_param_values."""

    def test_harvests_filters_and_clears_raw_value(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            param_key="p1",
            raw_value="hello",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = collect_raw_param_values(intent)
        assert out == {"p1": "hello"}
        assert intent.filters_param[0].raw_value is None

    def test_skips_empty_param_key(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            param_key="",
            raw_value="x",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        assert collect_raw_param_values(intent) == {}

    def test_cte_before_main(self):
        fp_cte = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            param_key="pc",
            raw_value=1,
        )
        fp_main = FilterParam(
            left_expr=NormalizedExpr.from_column("t.b"),
            op="=",
            value_type="string",
            param_key="pm",
            raw_value=2,
        )
        cte = RuntimeCteStep(cte_name="c", filters_param=[fp_cte])
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp_main],
            cte_steps=[cte],
        )
        out = collect_raw_param_values(intent)
        assert out == {"pc": 1, "pm": 2}


class TestParseBetweenRawValueExtended:
    """Extra _parse_between_raw_value cases."""

    def test_lowercase_and_separator(self):
        assert _parse_between_raw_value("1 and 2") == ("1", "2")


class TestNormalizeDateDiffSingularUnitPassthrough:
    """Valid singular units are left unchanged."""

    def test_day_unit_unchanged(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": 1},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        result = normalize_date_diff_raw_values(intent)
        assert result.filters_param[0].raw_value["unit"] == "day"


class TestParseExprStringTrivialZeros:
    """parse_expr_string drops trivial zero additive terms."""

    def test_plus_zero_removed(self):
        expr = parse_expr_string("t.a + 0")
        assert expr.primary_column == "t.a"

    def test_all_zeros_becomes_zero_literal(self):
        expr = parse_expr_string("0 + 0")
        assert len(expr.add_values) == 1
        assert expr.add_values[0].value == 0.0


class TestStripAngleBracketsMultiWord:
    """Angle-bracket pattern only replaces single-token placeholders."""

    def test_multi_word_placeholder_unchanged(self):
        assert _strip_angle_brackets("<not matched>") == "<not matched>"


class TestOrderByColFromObcLikely:
    """Realistic order_by_cols object shapes (dict expr, direction merging)."""

    def test_dict_expr_sort_key(self):
        obc = {
            "expr": {
                "add_groups": [{"multiply": ["sales.total"], "divide": []}],
                "sub_groups": [],
            },
            "direction": "DESC",
        }
        col = _order_by_col_from_obc(obc)
        assert col.expr.primary_column == "sales.total"
        assert col.direction == "DESC"

    def test_string_expr_direction_none_uses_trailing_suffix(self):
        obc = {"expr": "orders.amount DESC", "direction": None}
        col = _order_by_col_from_obc(obc)
        assert col.expr.primary_column == "orders.amount"
        assert col.direction == "DESC"

    def test_dict_expr_non_string_direction_normalizes(self):
        obc = {
            "expr": {
                "add_groups": [{"multiply": ["t.score"], "divide": []}],
                "sub_groups": [],
            },
            "direction": 1,
        }
        col = _order_by_col_from_obc(obc)
        assert col.expr.primary_column == "t.score"
        assert col.direction == "ASC"


class TestParseCaseWhenLikely:
    """CASE payloads with dict-shaped results (common when expr is structured)."""

    def test_branch_result_as_expr_dict(self):
        raw = {
            "branches": [
                {
                    "condition": {
                        "left_expr": "orders.status",
                        "op": "=",
                        "value_type": "string",
                        "value": "paid",
                    },
                    "result": {
                        "add_groups": [{"multiply": ["orders.amount"], "divide": []}],
                        "sub_groups": [],
                    },
                }
            ],
            "else_result": "0",
        }
        cw = CaseWhenExpr.from_dict(raw)
        assert cw is not None
        assert cw.branches[0].result.primary_column == "orders.amount"
        assert cw.else_result is not None
        assert cw.else_result.string_literal == "0"

    def test_else_result_dict(self):
        raw = {
            "branches": [
                {
                    "condition": {
                        "left_expr": "inventory.units",
                        "op": ">",
                        "value_type": "integer",
                        "value": 0,
                    },
                    "result": "inventory.units",
                }
            ],
            "else_result": {
                "add_groups": [{"multiply": ["inventory.reserved"], "divide": []}],
                "sub_groups": [],
            },
        }
        cw = CaseWhenExpr.from_dict(raw)
        assert cw is not None
        assert cw.else_result is not None
        assert cw.else_result.primary_column == "inventory.reserved"


class TestParseIntentResponseLikely:
    """Higher-frequency JSON shapes from intent parsing."""

    def test_limit_null_explicit(self):
        raw = json.dumps({"tables": ["t"], "select_cols": ["t.x"], "limit": None})
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.limit is None

    def test_filter_group_null(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.a",
                        "op": "=",
                        "value_type": "string",
                        "value": "x",
                        "filter_group": None,
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.filters_param[0].filter_group is None

    def test_string_filter_item_rejected_by_schema(self):
        """``filters_param`` items must be objects; string entries fail schema validation."""
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    "skip-me",
                    {
                        "left_expr": "t.ok",
                        "op": "=",
                        "value_type": "string",
                        "value": "1",
                    },
                ],
            }
        )
        buf: list[str] = []
        r = parse_intent_response(raw, "q", parse_detail_out=buf)
        assert r is None
        assert buf and "INTENT_SCHEMA" in buf[0]

    def test_mixed_string_and_object_select_cols(self):
        raw = json.dumps(
            {
                "tables": ["orders"],
                "select_cols": [
                    "orders.order_id",
                    {"expr": "sum(orders.amount)"},
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.select_cols[0].expr.primary_column == "orders.order_id"
        assert r.select_cols[1].expr.has_aggregation

    def test_group_by_wins_grain_over_scalar_select(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["count(t.id)", "t.region"],
                "group_by_cols": ["t.region"],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.grain == "grouped"

    def test_filter_right_col_qualified_preserved(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "filters_param": [
                    {
                        "left_expr": "t.price",
                        "op": ">=",
                        "right_col": "t.cost",
                        "value_type": "number",
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        assert r.filters_param[0].left_expr.primary_column == "t.price"
        assert r.filters_param[0].right_expr.primary_column == "t.cost"

    def test_cte_partial_output_columns_wrong_length_returns_none(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "cte_steps": [
                    {
                        "cte_name": "revenue",
                        "select_cols": ["t.region", "sum(t.amount)"],
                        "output_columns": ["region_only"],
                    }
                ],
            }
        )
        assert parse_intent_response(raw, "q") is None

    def test_cte_output_columns_mismatch_records_parse_detail(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "cte_steps": [
                    {
                        "cte_name": "revenue",
                        "select_cols": ["t.region", "sum(t.amount)"],
                        "output_columns": ["region_only"],
                    }
                ],
            }
        )
        buf: list[str] = []
        assert parse_intent_response(raw, "q", parse_detail_out=buf) is None
        assert buf
        assert "revenue" in buf[0]
        assert "output_columns length" in buf[0]

    def test_cte_invalid_output_column_name_gets_canonicalised(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "cte_steps": [
                    {
                        "cte_name": "c1",
                        "select_cols": ["t.id"],
                        "output_columns": ["1bad"],
                    }
                ],
            }
        )
        buf: list[str] = []
        r = parse_intent_response(raw, "q", parse_detail_out=buf)
        assert r is not None
        assert r.cte_steps[0].output_columns[0] == "_1bad"
        assert not buf

    def test_cte_order_by_string_in_step(self):
        raw = json.dumps(
            {
                "tables": ["t"],
                "select_cols": ["t.x"],
                "cte_steps": [
                    {
                        "cte_name": "ranked",
                        "select_cols": ["t.id"],
                        "output_columns": ["id"],
                        "order_by_cols": ["t.score DESC"],
                    }
                ],
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None
        ob = r.cte_steps[0].order_by_cols[0]
        assert ob.direction == "DESC"
        assert ob.expr.primary_column == "t.score"

    def test_aggregation_targets_allowed_by_schema(self):
        raw = json.dumps(
            {
                "tables": ["orders"],
                "select_cols": ["orders.order_id"],
                "aggregation_targets": {"orders.amount": "sum"},
            }
        )
        r = parse_intent_response(raw, "q")
        assert r is not None


class TestTagExprNumericLikely:
    """Expressions outside select list are tagged in real intents."""

    @pytest.fixture
    def schema_amt(self):
        t = TableMetadata(
            name="orders",
            columns={
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="numeric",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                    value_type="number",
                ),
                "region": ColumnMetadata(
                    name="region",
                    data_type="varchar",
                    role=ColumnRole.CATEGORICAL.value,
                    value_type="string",
                ),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"orders": t}, join_paths_multi={}, effective_structural_hash="t")

    def test_order_by_expr_tagged_numeric(self, schema_amt):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[
                OrderByCol(
                    expr=NormalizedExpr(add_groups=[MulGroup(coefficient=2.0, multiply=["orders.amount"])]),
                )
            ],
            filters_param=[],
        )
        out = tag_expr_numeric(intent, schema_amt)
        assert out.order_by_cols[0].expr.is_numeric is True
        assert out.order_by_cols[0].expr.add_values[0].value == 0.0

    def test_group_by_expr_tagged_non_numeric_text_column(self, schema_amt):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[NormalizedExpr(add_groups=[MulGroup(coefficient=3.0, multiply=["orders.region"])])],
            order_by_cols=[],
            filters_param=[],
        )
        out = tag_expr_numeric(intent, schema_amt)
        g = out.group_by_cols[0]
        assert g.is_numeric is False
        assert g.add_groups[0].coefficient == 1.0


class TestEnsureScalarFuncDefaultsLikely:
    """Defaults propagate into filter and having expressions."""

    def test_round_on_filter_left_expr(self):
        fp = FilterParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t.x"], scalar_func="round")]),
            op=">",
            value_type="number",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        ensure_scalar_func_defaults(intent)
        assert intent.filters_param[0].left_expr.add_groups[0].scalar_func_args == [2]

    def test_date_trunc_on_having_left(self):
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["orders.order_month"], scalar_func="date_trunc")]),
            op=">",
            value_type="integer",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
        )
        ensure_scalar_func_defaults(intent)
        assert intent.having_param[0].left_expr.add_groups[0].scalar_func_args == ["month"]


class TestExtractStructuralParamsLikely:
    """Structural keys on order_by and merged param_values."""

    def test_order_by_nontrivial_expr_gets_coeff_key(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[
                OrderByCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(coefficient=1.5, multiply=["t.score"], agg_func="avg")],
                        is_numeric=True,
                    ),
                )
            ],
            filters_param=[],
        )
        out = extract_structural_params(intent)
        assert any(v == 1.5 for v in out.param_values.values())

    def test_preserves_user_param_values_alongside_s_keys(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            param_values={"user_k": "keep", "s99": 1},
        )
        out = extract_structural_params(intent)
        assert out.param_values["user_k"] == "keep"
        assert out.param_values["s99"] == 1


class TestAssignParamKeysLikely:
    """date_diff dict payloads receive p* for amount."""

    def test_date_diff_filter_skipped(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.created_at"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": 30},
        )
        new_fp, _, _, _, idx = assign_param_keys([fp], [])
        assert new_fp[0].param_key == "p1"
        assert idx == 2


class TestExtractColumnsFromExprLikely:
    """Qualified refs inside divide operands show up in dependency lists."""

    def test_column_in_divisor(self):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["revenue.total"], divide=["revenue.days"])])
        cols = extract_columns_from_expr(expr)
        assert "revenue.total" in cols
        assert "revenue.days" in cols


class TestDecomposeBetweenLikely:
    """BETWEEN with parseable string bounds (common date or numeric range text)."""

    def test_string_and_separator_populates_bounds(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.order_date"),
            op="between",
            value_type="date",
            raw_value="2024-01-01 AND 2024-12-31",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="orders in 2024",
        )
        out = decompose_between_params(intent)
        assert out.filters_param[0].op == ">="
        assert out.filters_param[0].raw_value == "2024-01-01"
        assert out.filters_param[1].op == "<="
        assert out.filters_param[1].raw_value == "2024-12-31"


class TestNormalizeDateDiffLikely:
    """Plural units on having and inside CTEs."""

    def test_having_and_cte_normalized(self):
        hp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">=",
            value_type="date_window",
            raw_value={"unit": "months", "amount": 2},
        )
        cte_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.b"),
            op=">=",
            value_type="date_diff",
            raw_value={"unit": "years", "amount": 1},
        )
        cte = RuntimeCteStep(cte_name="c", filters_param=[cte_fp])
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
            param_values={},
            cte_steps=[cte],
            natural_language="q",
        )
        out = normalize_date_diff_raw_values(intent)
        assert out.having_param[0].raw_value["unit"] == "month"
        assert out.cte_steps[0].filters_param[0].raw_value["unit"] == "year"


class TestNormalizeInLikely:
    """Having clause IN lists are normalized like filters."""

    def test_having_in_string_split(self):
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op="in",
            value_type="string",
            raw_value="east, west, central",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
            param_values={},
            cte_steps=[],
            natural_language="regions",
        )
        out = normalize_in_raw_values(intent)
        assert out.having_param[0].raw_value == ["east", "west", "central"]


class TestStripOrderDirectionLikely:
    """Whitespace after DESC is not stripped by suffix logic (LLM slop)."""

    def test_desc_with_trailing_space_keeps_suffix_in_expr(self):
        expr, direction = _strip_order_direction("orders.amount DESC ")
        assert direction == "asc"
        assert "DESC" in expr


class TestIsExprNumericLikely:
    """Integer-typed columns and group-level numeric scalars."""

    @pytest.fixture
    def schema_int(self):
        t = TableMetadata(
            name="items",
            columns={
                "qty": ColumnMetadata(
                    name="qty",
                    data_type="integer",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                    value_type="integer",
                ),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"items": t}, join_paths_multi={}, effective_structural_hash="i")

    def test_integer_value_type_column_is_numeric(self, schema_int):
        expr = NormalizedExpr.from_column("items.qty")
        assert _is_expr_numeric(expr, schema_int) is True

    def test_length_scalar_on_group_is_numeric(self, schema_int):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["items.qty"], scalar_func="length")])
        assert _is_expr_numeric(expr, schema_int) is True


class TestDeriveCteOutputLikely:
    """Bare aggregation column naming (no table qualifier)."""

    def test_sum_bare_column_name(self):
        expr = NormalizedExpr.from_agg("sum", "amount")
        assert derive_cte_output_columns([SelectCol(expr=expr)]) == ["sum_amount"]

    def test_min_preserves_qualified_bare_part(self):
        expr = NormalizedExpr.from_agg("min", "inventory.sku_code")
        assert derive_cte_output_columns([SelectCol(expr=expr)]) == ["min_sku_code"]


class TestParseExprStringLikely:
    """Expressions that show up often in LLM output."""

    def test_coalesce_two_columns(self):
        expr = parse_expr_string("COALESCE(orders.discount, 0)")
        assert expr.scalar_func == "coalesce" or expr.add_groups[0].scalar_func == "coalesce"

    def test_line_total_multiplicative_terms(self):
        expr = parse_expr_string("order_lines.quantity * order_lines.unit_price")
        g = expr.add_groups[0]
        cols = [m.column_ref for m in g.multiply]
        assert "order_lines.quantity" in cols
        assert "order_lines.unit_price" in cols


class TestClassifyCteEmission:
    """Plan D.1 ``classify_cte_emission`` returns scalar_subquery only in safe shapes."""

    @staticmethod
    def _scalar_cte(name: str = "avg_rental") -> RuntimeCteStep:
        return RuntimeCteStep(
            cte_name=name,
            tables=["rental"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "rental.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=["avg_amount"],
            grain="scalar",
        )

    def test_non_scalar_grain_stays_join_table(self):
        cte = replace(self._scalar_cte(), grain="grouped")
        intent = RuntimeIntent(
            tables=[cte.cte_name],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{cte.cte_name}.avg_amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        assert classify_cte_emission(cte, intent, None) == "join_table"

    def test_multi_output_stays_join_table(self):
        cte = replace(
            self._scalar_cte(),
            output_columns=["avg_amount", "max_amount"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_agg("avg", "rental.amount")),
                SelectCol(expr=NormalizedExpr.from_agg("max", "rental.amount")),
            ],
        )
        intent = RuntimeIntent(
            tables=[cte.cte_name],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{cte.cte_name}.avg_amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        assert classify_cte_emission(cte, intent, None) == "join_table"

    def test_scalar_single_output_inlineable(self):
        cte = self._scalar_cte()
        intent = RuntimeIntent(
            tables=[cte.cte_name],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{cte.cte_name}.avg_amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        assert classify_cte_emission(cte, intent, None) == "scalar_subquery"

    def test_reaggregation_keeps_scalar_subquery(self):
        cte = self._scalar_cte()
        intent = RuntimeIntent(
            tables=[cte.cte_name],
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("max", f"{cte.cte_name}.avg_amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        assert classify_cte_emission(cte, intent, None) == "scalar_subquery"

    def test_filter_left_ref_keeps_scalar_subquery(self):
        cte = self._scalar_cte()
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column(f"{cte.cte_name}.avg_amount"),
            op=">",
            right_expr=None,
            value_type="numeric",
            param_key="p1",
        )
        intent = RuntimeIntent(
            tables=[cte.cte_name],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{cte.cte_name}.avg_amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            cte_steps=[cte],
        )
        assert classify_cte_emission(cte, intent, None) == "scalar_subquery"
