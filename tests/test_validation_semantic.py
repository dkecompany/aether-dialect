"""Tests for validation_semantic module."""

from aetherdialect._constants import COMPATIBLE_TYPE_PAIRS
from aetherdialect._contracts_base import (
    FailureCategory,
    FilterParam,
    HavingParam,
    LogicalIntent,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    SensitivityClassification,
)
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._validation_execute import (
    curated_warmup_post_binding_issues,
    curated_warmup_semantic_issues,
)
from aetherdialect._validation_semantic import (
    _check_mixed_aggregation_in_expr,
    _check_mixed_aggregation_in_group,
    _check_nested_aggregation,
    _english_plural_forms,
    _find_fk_column_for_target,
    _resolve_word_to_table,
    _term_has_aggregation,
    _validate_concat_group,
    _validate_single_expr_types,
    validate_agg_vs_agg_having,
    validate_arith_expression_semantics,
    validate_case_branch_aggregation_consistency,
    validate_concat_mulgroups_in_runtime,
    validate_count_threshold_missing_having,
    validate_cte_dependency_grains,
    validate_cte_grain_consistency,
    validate_denied_references,
    validate_deny_bare_select,
    validate_empty_window,
    validate_expr_vs_expr_filters,
    validate_filter_expr_types,
    validate_filter_no_aggregation,
    validate_for_each_grouping,
    validate_grain_consistency,
    validate_grouped_requires_aggregation,
    validate_having_expr_types,
    validate_having_requires_aggregation,
    validate_logical_intent_numeric_coverage,
    validate_mixed_aggregation_in_mulgroup,
    validate_no_nested_aggregation,
    validate_order_by_aggregation_context,
    validate_order_by_expr_types,
    validate_predicate_sidedness,
    validate_question_agg_keyword_coverage,
    validate_question_distinct_hint,
    validate_select_expr_types,
    validate_select_group_by_membership,
    validate_semantic_contradictions,
    validate_sensitivity_group_by,
    validate_sensitivity_order_by,
    validate_threshold_missing_having,
)


class TestValidateGrainConsistency:
    """Tests for validate_grain_consistency."""

    def test_grouped_with_group_by(self):
        """No issues for grouped grain with GROUP BY columns."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "t.a"))
        gb = NormalizedExpr.from_column("t.b")
        issues = validate_grain_consistency("grouped", [sc], [gb], [])
        assert len(issues) == 0

    def test_grouped_without_group_by(self):
        """Error for grouped grain without GROUP BY."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "t.a"))
        issues = validate_grain_consistency("grouped", [sc], [], [])
        assert any("without GROUP BY" in i.message for i in issues)

    def test_scalar_with_group_by(self):
        """Error for scalar grain with GROUP BY present."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "t.a"))
        gb = NormalizedExpr.from_column("t.b")
        issues = validate_grain_consistency("scalar", [sc], [gb], [])
        assert any("GROUP BY" in i.message for i in issues)

    def test_row_level_with_agg(self):
        """Error for row_level grain with aggregation."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("sum", "t.a"))
        issues = validate_grain_consistency("row_level", [sc], [], [])
        assert any("row_level" in i.message for i in issues)

    def test_invalid_grain_value(self):
        """Error for unrecognized grain value."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        issues = validate_grain_consistency("invalid", [sc], [], [])
        assert any("Invalid grain" in i.message for i in issues)

    def test_having_without_agg_grain(self):
        """Error for HAVING with row_level grain."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        issues = validate_grain_consistency("row_level", [sc], [], [hp])
        assert any("HAVING" in i.message for i in issues)


class TestValidateEmptyWindow:
    """Tests for validate_empty_window."""

    def test_rejects_identical_inequality_bounds(self):
        """Two ANDed predicates on one column with the same bound are an empty window."""
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        intent = RuntimeIntent(
            tables=["payment"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.payment_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op=">=",
                    value_type="date",
                    raw_value="2020-01-01",
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op="<=",
                    value_type="date",
                    raw_value="2020-01-01",
                ),
            ],
        )
        issues = validate_empty_window(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.INTENT_EMPTY_WINDOW
        assert "payment.payment_date" in issues[0].message

    def test_accepts_distinct_bounds(self):
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        intent = RuntimeIntent(
            tables=["payment"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.payment_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op=">=",
                    value_type="date",
                    raw_value="2020-01-01",
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op="<=",
                    value_type="date",
                    raw_value="2020-12-31",
                ),
            ],
        )
        assert validate_empty_window(intent, schema) == []

    def test_rejects_date_window_identical_start_end(self):
        schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        intent = RuntimeIntent(
            tables=["payment"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.payment_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op=">=",
                    value_type="date_window",
                    raw_value={"start": "2021-06-01", "end": "2021-06-01"},
                ),
            ],
        )
        issues = validate_empty_window(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.INTENT_EMPTY_WINDOW


class TestValidateSemanticContradictions:
    """Tests for validate_semantic_contradictions."""

    def test_no_contradiction(self):
        """No issues for non-contradictory intent."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "t.a"))
        issues = validate_semantic_contradictions([sc], "count of items", "scalar", "one")
        assert len(issues) == 0

    def test_nl_contradiction_pattern_never_total(self):
        """NL mentioning 'never' and 'total' triggers warning."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(t.a)"])]))
        issues = validate_semantic_contradictions(
            [sc],
            natural_language="Show me records that never have a total over 100",
            grain="grouped",
            expected_rows="few",
        )
        nl_issues = [i for i in issues if "never" in i.message.lower() or "contradiction" in i.message.lower()]
        assert len(nl_issues) >= 1

    def test_scalar_with_many_rows(self):
        """Error for scalar grain but expecting many rows."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "t.a"))
        issues = validate_semantic_contradictions([sc], "count", "scalar", "many")
        assert any("scalar" in i.message and "many" in i.message for i in issues)


class TestValidateExprVsExprFilters:
    """Tests for validate_expr_vs_expr_filters."""

    def test_self_comparison(self, typed_schema):
        """Error for self-comparison in filter."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.balance"),
            op=">",
            right_expr=NormalizedExpr.from_column("customers.balance"),
        )
        issues = validate_expr_vs_expr_filters([fp], typed_schema)
        assert any("Self-comparison" in i.message for i in issues)

    def test_type_mismatch(self, typed_schema):
        """Error for type mismatch in expr-vs-expr comparison."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.balance"),
            op=">",
            right_expr=NormalizedExpr.from_column("customers.name"),
        )
        issues = validate_expr_vs_expr_filters([fp], typed_schema)
        assert any("mismatch" in i.message.lower() for i in issues)

    def test_compatible_types_no_issue(self, typed_schema):
        """No issues for compatible type comparison."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.balance"),
            op=">",
            right_expr=NormalizedExpr.from_column("orders.amount"),
        )
        issues = validate_expr_vs_expr_filters([fp], typed_schema)
        type_issues = [i for i in issues if "mismatch" in i.message.lower()]
        assert len(type_issues) == 0

    def test_date_subtraction_vs_integer_duration_no_type_mismatch(self):
        """Date subtraction compared to an integer duration column is valid."""
        from aetherdialect._contracts_base import ColumnRole
        from aetherdialect._contracts_schema import (
            ColumnMetadata,
            SchemaGraph,
            TableMetadata,
        )

        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "rental": TableMetadata(
                    name="rental",
                    columns={
                        "return_date": ColumnMetadata(
                            name="return_date",
                            data_type="date",
                            value_type="date",
                            role=ColumnRole.TEMPORAL.value,
                        ),
                        "rental_date": ColumnMetadata(
                            name="rental_date",
                            data_type="date",
                            value_type="date",
                            role=ColumnRole.TEMPORAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "item": TableMetadata(
                    name="item",
                    columns={
                        "rental_duration": ColumnMetadata(
                            name="rental_duration",
                            data_type="integer",
                            value_type="integer",
                            role=ColumnRole.NUMERIC_MEASURE.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        from aetherdialect._intent_expr import parse_expr_string

        fp = FilterParam(
            left_expr=parse_expr_string("rental.return_date - rental.rental_date"),
            op=">",
            right_expr=parse_expr_string("item.rental_duration"),
        )
        issues = validate_expr_vs_expr_filters([fp], schema)
        type_issues = [i for i in issues if "mismatch" in i.message.lower() and i.severity == "error"]
        assert len(type_issues) == 0


class TestValidateFilterNoAggregation:
    """Tests for validate_filter_no_aggregation."""

    def test_bare_column_filter(self):
        """No issues for non-aggregated filter."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        issues = validate_filter_no_aggregation([fp])
        assert len(issues) == 0

    def test_aggregated_filter(self):
        """Error for aggregated filter expression."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        issues = validate_filter_no_aggregation([fp])
        assert len(issues) > 0


class TestValidateHavingRequiresAggregation:
    """Tests for validate_having_requires_aggregation."""

    def test_aggregated_having(self):
        """No issues for aggregated having expression."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        gb = [NormalizedExpr.from_column("t.a")]
        issues = validate_having_requires_aggregation([hp], group_by_cols=gb)
        assert len(issues) == 0

    def test_non_aggregated_having(self):
        """Error for non-aggregated having expression."""
        hp = HavingParam(left_expr=NormalizedExpr.from_column("t.a"), op=">", value_type="integer")
        gb = [NormalizedExpr.from_column("t.a")]
        issues = validate_having_requires_aggregation([hp], group_by_cols=gb)
        assert len(issues) > 0

    def test_having_without_group_by(self):
        """HAVING with no GROUP BY emits a single structural issue."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        issues = validate_having_requires_aggregation([hp], group_by_cols=[])
        assert len(issues) == 1
        assert issues[0].issue_id == "having_without_group_by"


class TestValidatePredicateSidedness:
    """Tests for validate_predicate_sidedness."""

    def test_clean_column_left_literal_right(self):
        """No issues when left is column-bearing and right is literal- only."""
        from aetherdialect._contracts_base import ExprValue

        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=5.0)]),
        )
        issues = validate_predicate_sidedness([fp], [])
        assert issues == []

    def test_flags_mutated_literal_left_column_right(self):
        """Error when a filter's sides are mutated so literal-only ends up on the left."""
        from aetherdialect._contracts_base import ExprValue

        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=5.0)]),
        )
        fp.left_expr = NormalizedExpr(add_values=[ExprValue(value=5.0)])
        fp.right_expr = NormalizedExpr.from_column("orders.amount")
        issues = validate_predicate_sidedness([fp], [])
        assert len(issues) == 1
        assert issues[0].category == "predicate_sidedness"
        assert issues[0].severity == "error"

    def test_flags_mutated_literal_left_agg_right_in_having(self):
        """Error when HAVING sides are mutated so literal-only ends up on the left."""
        from aetherdialect._contracts_base import ExprValue

        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.order_id"),
            op=">",
            right_expr=NormalizedExpr(add_values=[ExprValue(value=10.0)]),
        )
        hp.left_expr = NormalizedExpr(add_values=[ExprValue(value=10.0)])
        hp.right_expr = NormalizedExpr.from_agg("count", "orders.order_id")
        issues = validate_predicate_sidedness([], [hp])
        assert len(issues) == 1
        assert issues[0].context["clause"] == "having"

    def test_ignores_right_expr_none(self):
        """Filters with no right_expr are untouched by the sidedness check."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op="=",
            param_key="p1",
            raw_value=5,
            value_type="integer",
        )
        issues = validate_predicate_sidedness([fp], [])
        assert issues == []


class TestValidateLogicalIntentNumericCoverage:
    """Tests for validate_logical_intent_numeric_coverage."""

    def test_year_in_date_literal_covered(self):
        """Year substring in date literal (e.g. 2005 in 2005-08-01) is covered."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">=",
            value_type="date",
            raw_value="2005-08-01",
        )
        li = LogicalIntent(tables=("t",), select="*", filter="Show rentals from 2005")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [fp],
            [],
            None,
            "main",
        )
        assert len(issues) == 0

    def test_missing_numeric_without_date_covered(self):
        """Number in planner prose without matching filter produces issue."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            raw_value="x",
        )
        li = LogicalIntent(tables=("t",), select="*", filter="Show rows where value is 42")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [fp],
            [],
            None,
            "main",
        )
        assert len(issues) == 1
        assert "42" in issues[0].message


class TestCompatibleTypePairs:
    """Tests for COMPATIBLE_TYPE_PAIRS constant."""

    def test_int_integer_compatible(self):
        """Int and integer are compatible."""
        assert ("int", "integer") in COMPATIBLE_TYPE_PAIRS

    def test_varchar_text_compatible(self):
        """Varchar and text are compatible."""
        assert ("varchar", "text") in COMPATIBLE_TYPE_PAIRS

    def test_date_timestamp_compatible(self):
        """Date and timestamp are compatible."""
        assert ("date", "timestamp") in COMPATIBLE_TYPE_PAIRS

    def test_number_integer_compatible(self):
        """Number and integer are compatible."""
        assert ("number", "integer") in COMPATIBLE_TYPE_PAIRS

    def test_int_text_not_compatible(self):
        """Int and text are not compatible."""
        assert ("int", "text") not in COMPATIBLE_TYPE_PAIRS
        assert ("text", "int") not in COMPATIBLE_TYPE_PAIRS


class TestValidateNoNestedAggregation:
    """Tests for validate_no_nested_aggregation."""

    def test_nested_agg_in_select(self):
        """validate_no_nested_aggregation detects nested agg in select."""
        sc = SelectCol(
            expr=NormalizedExpr(
                agg_func="sum",
                add_groups=[MulGroup(coefficient=1.0, multiply=["orders.amount"], agg_func="count")],
            ),
        )
        issues = validate_no_nested_aggregation([sc], [], [], [])
        assert len(issues) >= 1
        assert any("nested" in i.message.lower() for i in issues)

    def test_no_nesting_passes(self):
        """validate_no_nested_aggregation passes for non-nested agg."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["orders.amount"], agg_func="sum")],
            ),
        )
        issues = validate_no_nested_aggregation([sc], [], [], [])
        assert len(issues) == 0


class TestValidateMixedAggregationInMulgroup:
    """Tests for validate_mixed_aggregation_in_mulgroup."""

    def test_mixed_agg_bare_errors(self):
        """validate_mixed_aggregation_in_mulgroup errors for mixed agg and bare terms."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(orders.id)", "customers.name"])],
            ),
        )
        issues = validate_mixed_aggregation_in_mulgroup([sc], [], [], [])
        assert len(issues) >= 1

    def test_all_agg_passes(self):
        """validate_mixed_aggregation_in_mulgroup passes for all-agg terms."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["orders.amount"], agg_func="sum")],
            ),
        )
        issues = validate_mixed_aggregation_in_mulgroup([sc], [], [], [])
        assert len(issues) == 0


class TestValidateOrderByAggregationContext:
    """Tests for validate_order_by_aggregation_context."""

    def test_agg_in_row_level_errors(self):
        """validate_order_by_aggregation_context errors for agg order_by at row_level."""
        obc = OrderByCol(
            expr=NormalizedExpr(
                agg_func="count",
                add_groups=[MulGroup(coefficient=1.0, multiply=["orders.id"])],
            ),
        )
        issues = validate_order_by_aggregation_context([obc], grain="row_level")
        assert len(issues) >= 1

    def test_agg_in_grouped_passes(self):
        """validate_order_by_aggregation_context passes for agg order_by at grouped."""
        obc = OrderByCol(
            expr=NormalizedExpr(
                agg_func="count",
                add_groups=[MulGroup(coefficient=1.0, multiply=["orders.id"])],
            ),
        )
        issues = validate_order_by_aggregation_context([obc], grain="grouped")
        assert len(issues) == 0

    def test_no_agg_row_level_passes(self):
        """validate_order_by_aggregation_context passes for non-agg at row_level."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("orders.id"))
        issues = validate_order_by_aggregation_context([obc], grain="row_level")
        assert len(issues) == 0


class TestValidateSelectGroupByMembership:
    """Tests for validate_select_group_by_membership."""

    def test_non_agg_not_in_group_by_errors(self):
        """validate_select_group_by_membership errors for non-agg select not in group_by when mixed aggregation is present."""
        sc_name = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        sc_count = SelectCol(
            expr=NormalizedExpr(
                agg_func="count",
                add_groups=[MulGroup(coefficient=1.0, multiply=["customers.id"])],
            ),
        )
        gb = [NormalizedExpr.from_column("customers.id")]
        issues = validate_select_group_by_membership([sc_name, sc_count], gb, grain="grouped")
        assert len(issues) >= 1

    def test_agg_select_passes(self):
        """validate_select_group_by_membership passes for aggregated select."""
        sc = SelectCol(
            expr=NormalizedExpr(
                agg_func="count",
                add_groups=[MulGroup(coefficient=1.0, multiply=["customers.id"])],
            ),
        )
        gb = [NormalizedExpr.from_column("customers.name")]
        issues = validate_select_group_by_membership([sc], gb, grain="grouped")
        assert len(issues) == 0

    def test_non_grouped_grain_passes(self):
        """validate_select_group_by_membership passes for row_level grain without mixed aggregation."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        issues = validate_select_group_by_membership([sc], [], grain="row_level")
        assert len(issues) == 0

    def test_mixed_aggregation_errors_when_identifier_missing_from_group_by(self):
        """Mixed agg/non-agg select triggers membership even when grain is not grouped."""
        sc_agg = SelectCol(
            expr=NormalizedExpr(
                agg_func="count",
                add_groups=[MulGroup(coefficient=1.0, multiply=["customers.id"])],
            ),
        )
        sc_id = SelectCol(expr=NormalizedExpr.from_column("customers.id"))
        gb = [NormalizedExpr.from_column("customers.name")]
        issues = validate_select_group_by_membership([sc_id, sc_agg], gb, grain="row_level")
        assert len(issues) == 1


class TestTermHasAggregation:
    """Tests for _term_has_aggregation."""

    def test_count_term(self):
        """Return True for COUNT term."""
        assert _term_has_aggregation("COUNT(t.a)") is True

    def test_sum_term(self):
        """Return True for SUM term."""
        assert _term_has_aggregation("SUM(t.a)") is True

    def test_bare_column(self):
        """Return False for bare column."""
        assert _term_has_aggregation("t.a") is False

    def test_upper_not_agg(self):
        """Return False for UPPER."""
        assert _term_has_aggregation("UPPER(t.a)") is False

    def test_min_term(self):
        """Return True for MIN term."""
        assert _term_has_aggregation("MIN(t.a)") is True


class TestValidateSingleExprTypes:
    """Tests for _validate_single_expr_types."""

    def test_no_arithmetic_no_issues(self, typed_schema):
        """No issues for non-arithmetic expression."""
        expr = NormalizedExpr.from_column("customers.name")
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        assert len(issues) == 0

    def test_arithmetic_with_non_numeric(self, typed_schema):
        """Error for non-numeric column in arithmetic."""
        g = MulGroup(multiply=["customers.name"], coefficient=2.0)
        expr = NormalizedExpr(add_groups=[g])
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1
        assert "Non-numeric" in errors[0].message

    def test_arithmetic_with_numeric(self, typed_schema):
        """No errors for numeric column in arithmetic."""
        g = MulGroup(multiply=["customers.balance"], coefficient=2.0)
        expr = NormalizedExpr(add_groups=[g])
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_division_with_non_numeric(self, typed_schema):
        """Error for non-numeric column in division."""
        g = MulGroup(multiply=["customers.balance"], divide=["customers.name"])
        expr = NormalizedExpr(add_groups=[g])
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_role_warning_for_identifier(self, typed_schema):
        """Warning for identifier role in arithmetic."""
        g = MulGroup(multiply=["customers.id"], coefficient=2.0)
        expr = NormalizedExpr(add_groups=[g])
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 1

    def test_date_like_columns_in_multiplicative_group_skip_non_numeric_errors(self, typed_schema):
        """Arithmetic context over ``date`` / ``timestamp`` columns does not emit type errors."""
        g = MulGroup(
            multiply=[
                NormalizedExpr.from_column("orders.order_date"),
                NormalizedExpr.from_column("customers.created_at"),
            ]
        )
        expr = NormalizedExpr(add_groups=[g])
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []


class TestValidateAggVsAggHaving:
    """Tests for validate_agg_vs_agg_having."""

    def test_self_comparison_detected(self, typed_schema):
        """Self-comparison in HAVING produces an issue."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["count(orders.id)"])])
        right = NormalizedExpr(add_groups=[MulGroup(multiply=["count(orders.id)"])])
        hp = HavingParam(
            left_expr=left,
            op=">",
            right_expr=right,
            value_type="integer",
        )
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert any("Self-comparison" in i.message for i in issues)

    def test_compatible_agg_no_issue(self, typed_schema):
        """Two numeric aggregations compared produce no issue."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["sum(orders.amount)"])])
        right = NormalizedExpr(add_groups=[MulGroup(multiply=["avg(orders.amount)"])])
        hp = HavingParam(
            left_expr=left,
            op=">",
            right_expr=right,
            value_type="number",
        )
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert len(issues) == 0

    def test_no_right_expr_skipped(self, typed_schema):
        """HAVING with no right_expr is skipped."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["count(orders.id)"])]),
            op=">",
            value_type="integer",
        )
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert len(issues) == 0

    def test_empty_having(self, typed_schema):
        """Empty having list produces no issues."""
        assert validate_agg_vs_agg_having([], typed_schema) == []

    def test_none_having(self, typed_schema):
        """None having list produces no issues."""
        assert validate_agg_vs_agg_having(None, typed_schema) == []

    def test_empty_having_no_issues(self, typed_schema):
        """No issues for empty having list."""
        issues = validate_agg_vs_agg_having([], typed_schema)
        assert len(issues) == 0

    def test_no_right_expr_no_issues(self, typed_schema):
        """No issues when having has no right expr."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
        )
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert len(issues) == 0

    def test_self_comparison_error(self, typed_schema):
        """Error for self-comparison in having."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(customers.id)"])])
        right = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(customers.id)"])])
        hp = HavingParam(left_expr=left, op=">", right_expr=right, value_type="integer")
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert any("Self-comparison" in i.message for i in issues)

    def test_type_mismatch_warning(self, typed_schema):
        """Warning for type mismatch between agg targets."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["MIN(customers.balance)"])])
        right = NormalizedExpr(add_groups=[MulGroup(multiply=["MIN(customers.name)"])])
        hp = HavingParam(left_expr=left, op=">", right_expr=right, value_type="number")
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 1

    def test_compatible_numeric_aggs_pass(self, typed_schema):
        """No issues for numeric-result agg functions on same type."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["SUM(customers.balance)"])])
        right = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(customers.id)"])])
        hp = HavingParam(left_expr=left, op=">", right_expr=right, value_type="number")
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert len(issues) == 0


class TestValidateArithExpressionSemantics:
    """Tests for validate_arith_expression_semantics."""

    def test_empty_inputs_no_issues(self, typed_schema):
        """No issues for empty filter and having lists."""
        issues = validate_arith_expression_semantics([], [], typed_schema)
        assert len(issues) == 0

    def test_no_arithmetic_no_issues(self, typed_schema):
        """Simple column filters with no arithmetic produce no issues."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op=">",
            value_type="number",
        )
        issues = validate_arith_expression_semantics([fp], [], typed_schema)
        assert len(issues) == 0

    def test_arithmetic_with_text_column(self, typed_schema):
        """Arithmetic involving text column produces issue."""
        arith_expr = NormalizedExpr(
            add_groups=[
                MulGroup(multiply=["customers.balance", "customers.description"]),
            ]
        )
        fp = FilterParam(left_expr=arith_expr, op=">", value_type="number", param_key="p1")
        issues = validate_arith_expression_semantics([fp], [], typed_schema)
        assert len(issues) >= 1

    def test_empty_filters_and_having(self, typed_schema):
        """Empty input lists produce no issues."""
        assert validate_arith_expression_semantics([], [], typed_schema) == []

    def test_non_arithmetic_filter_no_issues(self, typed_schema):
        """No issues for non-arithmetic filter."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_arith_expression_semantics([fp], [], typed_schema)
        assert len(issues) == 0

    def test_arithmetic_filter_with_non_numeric(self, typed_schema):
        """Error for non-numeric column in arithmetic filter."""
        g = MulGroup(multiply=["customers.name"], coefficient=2.0)
        fp = FilterParam(left_expr=NormalizedExpr(add_groups=[g]), op=">", value_type="number")
        issues = validate_arith_expression_semantics([fp], [], typed_schema)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_arithmetic_having_with_non_numeric(self, typed_schema):
        """Error for non-numeric column in arithmetic having."""
        g = MulGroup(multiply=["customers.name"], coefficient=2.0)
        hp = HavingParam(left_expr=NormalizedExpr(add_groups=[g]), op=">", value_type="number")
        issues = validate_arith_expression_semantics([], [hp], typed_schema)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1


class TestValidateSelectExprTypes:
    """Tests for validate_select_expr_types."""

    def test_empty_select_no_issues(self, typed_schema):
        """No issues for empty select list."""
        issues = validate_select_expr_types([], typed_schema)
        assert len(issues) == 0

    def test_bare_column_no_issues(self, typed_schema):
        """No issues for non-arithmetic select."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        issues = validate_select_expr_types([sc], typed_schema)
        assert len(issues) == 0

    def test_arithmetic_with_non_numeric(self, typed_schema):
        """Error for non-numeric in arithmetic select."""
        g = MulGroup(multiply=["customers.name"], coefficient=2.0)
        sc = SelectCol(expr=NormalizedExpr(add_groups=[g]))
        issues = validate_select_expr_types([sc], typed_schema)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_simple_column_no_issue(self, typed_schema):
        """Simple column expression produces no issue."""
        sc = SelectCol(expr=NormalizedExpr.from_column("orders.amount"))
        issues = validate_select_expr_types([sc], typed_schema)
        assert len(issues) == 0

    def test_arithmetic_text_produces_issue(self, typed_schema):
        """Arithmetic on text column produces an issue."""
        arith_expr = NormalizedExpr(
            add_groups=[
                MulGroup(multiply=["customers.name", "customers.balance"]),
            ]
        )
        sc = SelectCol(expr=arith_expr)
        issues = validate_select_expr_types([sc], typed_schema)
        assert len(issues) >= 1


class TestValidateOrderByExprTypes:
    """Tests for validate_order_by_expr_types."""

    def test_empty_order_by_no_issues(self, typed_schema):
        """No issues for empty order_by list."""
        issues = validate_order_by_expr_types([], typed_schema)
        assert len(issues) == 0

    def test_bare_column_no_issues(self, typed_schema):
        """No issues for non-arithmetic order_by."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="asc")
        issues = validate_order_by_expr_types([obc], typed_schema)
        assert len(issues) == 0

    def test_arithmetic_with_non_numeric(self, typed_schema):
        """Error for non-numeric in arithmetic order_by."""
        g = MulGroup(multiply=["customers.name"], coefficient=2.0)
        obc = OrderByCol(expr=NormalizedExpr(add_groups=[g]), direction="asc")
        issues = validate_order_by_expr_types([obc], typed_schema)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_simple_column_no_issue(self, typed_schema):
        """Simple column ORDER BY produces no issue."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("orders.amount"), direction="asc")
        issues = validate_order_by_expr_types([obc], typed_schema)
        assert len(issues) == 0

    def test_arithmetic_text_produces_issue(self, typed_schema):
        """Arithmetic on text column in ORDER BY produces an issue."""
        arith_expr = NormalizedExpr(
            add_groups=[
                MulGroup(multiply=["customers.name", "customers.balance"]),
            ]
        )
        obc = OrderByCol(expr=arith_expr, direction="desc")
        issues = validate_order_by_expr_types([obc], typed_schema)
        assert len(issues) >= 1

    def test_empty_order_by(self, typed_schema):
        """Empty order by list produces no issues."""
        assert validate_order_by_expr_types([], typed_schema) == []


class TestValidateFilterExprTypes:
    """Tests for validate_filter_expr_types."""

    def test_empty_filters_no_issues(self, typed_schema):
        """No issues for empty filter list."""
        issues = validate_filter_expr_types([], typed_schema)
        assert len(issues) == 0

    def test_non_arithmetic_filter_no_issues(self, typed_schema):
        """No issues for non-arithmetic filter."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_filter_expr_types([fp], typed_schema)
        assert len(issues) == 0

    def test_cross_type_mismatch(self, typed_schema):
        """Error for numeric vs non-numeric expr-vs-expr comparison."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["customers.balance"], coefficient=2.0)])
        right = NormalizedExpr.from_column("customers.name")
        fp = FilterParam(left_expr=left, op=">", right_expr=right, value_type="number")
        issues = validate_filter_expr_types([fp], typed_schema)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_arithmetic_op_mismatch(self, typed_schema):
        """Error for LIKE on arithmetic expression."""
        g = MulGroup(multiply=["customers.balance"], coefficient=2.0)
        fp = FilterParam(left_expr=NormalizedExpr(add_groups=[g]), op="like", value_type="string")
        issues = validate_filter_expr_types([fp], typed_schema)
        errors = [i for i in issues if "invalid" in i.message.lower() or "op" in i.message.lower()]
        assert len(errors) >= 1

    def test_simple_filter_no_issue(self, typed_schema):
        """Simple column filter produces no issue."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op=">",
            value_type="number",
            param_key="p1",
        )
        issues = validate_filter_expr_types([fp], typed_schema)
        assert len(issues) == 0

    def test_cross_type_mismatch_direct_columns(self, typed_schema):
        """Numeric vs non-numeric cross comparison produces issue."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op=">",
            right_expr=NormalizedExpr.from_column("customers.name"),
            value_type="number",
            param_key="p1",
        )
        issues = validate_filter_expr_types([fp], typed_schema)
        assert any("cross_type_mismatch" in i.issue_id or "mismatch" in i.message.lower() for i in issues)

    def test_empty_filters(self, typed_schema):
        """Empty filter list produces no issues."""
        assert validate_filter_expr_types([], typed_schema) == []

    def test_none_filters(self, typed_schema):
        """None filter list produces no issues."""
        assert validate_filter_expr_types(None, typed_schema) == []


class TestValidateHavingExprTypes:
    """Tests for validate_having_expr_types."""

    def test_empty_having_no_issues(self, typed_schema):
        """No issues for empty having list."""
        issues = validate_having_expr_types([], typed_schema)
        assert len(issues) == 0

    def test_non_arithmetic_having_no_issues(self, typed_schema):
        """No issues for non-arithmetic having."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
        )
        issues = validate_having_expr_types([hp], typed_schema)
        assert len(issues) == 0

    def test_cross_type_mismatch(self, typed_schema):
        """Error for numeric vs non-numeric having comparison."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["customers.balance"], coefficient=2.0)])
        right = NormalizedExpr.from_column("customers.name")
        hp = HavingParam(left_expr=left, op=">", right_expr=right, value_type="number")
        issues = validate_having_expr_types([hp], typed_schema)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_simple_having_no_issue(self, typed_schema):
        """Aggregated having with numeric type produces no issue."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            value_type="integer",
            param_key="p1",
        )
        issues = validate_having_expr_types([hp], typed_schema)
        assert len(issues) == 0

    def test_cross_type_mismatch_having(self, typed_schema):
        """Numeric vs non-numeric cross comparison in HAVING produces issue."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "orders.amount"),
            op=">",
            right_expr=NormalizedExpr.from_column("customers.name"),
            value_type="number",
            param_key="p1",
        )
        issues = validate_having_expr_types([hp], typed_schema)
        assert len(issues) >= 1

    def test_empty_having(self, typed_schema):
        """Empty having list produces no issues."""
        assert validate_having_expr_types([], typed_schema) == []

    def test_none_having(self, typed_schema):
        """None having list produces no issues."""
        assert validate_having_expr_types(None, typed_schema) == []


class TestCheckNestedAggregation:
    """Tests for _check_nested_aggregation."""

    def test_no_nesting(self):
        """No issues for non-nested expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"], agg_func="sum")])
        issues = _check_nested_aggregation(expr, "select_cols[0]", "main")
        assert len(issues) == 0

    def test_expr_wraps_group_agg(self):
        """Error for expr-level agg wrapping group-level agg."""
        expr = NormalizedExpr(agg_func="sum", add_groups=[MulGroup(multiply=["t.a"], agg_func="count")])
        issues = _check_nested_aggregation(expr, "select_cols[0]", "main")
        assert len(issues) >= 1
        assert any("nested" in i.message.lower() for i in issues)

    def test_expr_wraps_inline_agg(self):
        """Error for expr-level agg wrapping inline agg term."""
        expr = NormalizedExpr(agg_func="sum", add_groups=[MulGroup(multiply=["COUNT(t.a)"])])
        issues = _check_nested_aggregation(expr, "select_cols[0]", "main")
        assert len(issues) >= 1

    def test_group_wraps_inline_agg(self):
        """Error for group-level agg wrapping inline agg term."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.a)"], agg_func="sum")])
        issues = _check_nested_aggregation(expr, "select_cols[0]", "main")
        assert len(issues) >= 1


class TestCheckMixedAggregationInGroup:
    """Tests for _check_mixed_aggregation_in_group."""

    def test_all_agg_no_issues(self):
        """No issues when all terms are aggregated."""
        g = MulGroup(multiply=["COUNT(t.a)", "SUM(t.b)"])
        issues = _check_mixed_aggregation_in_group(g, "select_cols[0]_add[0]", "main")
        assert len(issues) == 0

    def test_mixed_agg_bare_error(self):
        """Error when mixing agg and bare terms."""
        g = MulGroup(multiply=["COUNT(t.a)", "t.b"])
        issues = _check_mixed_aggregation_in_group(g, "select_cols[0]_add[0]", "main")
        assert len(issues) == 1
        assert "mixes" in issues[0].message.lower()

    def test_group_level_agg_skips_check(self):
        """No issues when group has agg_func (covers all terms)."""
        g = MulGroup(multiply=["t.a", "t.b"], agg_func="sum")
        issues = _check_mixed_aggregation_in_group(g, "select_cols[0]_add[0]", "main")
        assert len(issues) == 0

    def test_single_term_no_issues(self):
        """No issues for single-term group."""
        g = MulGroup(multiply=["COUNT(t.a)"])
        issues = _check_mixed_aggregation_in_group(g, "select_cols[0]_add[0]", "main")
        assert len(issues) == 0


class TestCheckMixedAggregationInExpr:
    """Tests for _check_mixed_aggregation_in_expr."""

    def test_single_group_no_issues(self):
        """No issues for single-group expression."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.a)"])])
        issues = _check_mixed_aggregation_in_expr(expr, "select_cols[0]", "main")
        assert len(issues) == 0

    def test_mixed_across_groups_error(self):
        """Error when mixing agg and bare across groups."""
        g1 = MulGroup(multiply=["COUNT(t.a)"])
        g2 = MulGroup(multiply=["t.b"])
        expr = NormalizedExpr(add_groups=[g1, g2])
        issues = _check_mixed_aggregation_in_expr(expr, "select_cols[0]", "main")
        cross_issues = [
            i
            for i in issues
            if "across" in i.issue_id or "across" in i.message.lower() or "groups" in i.message.lower()
        ]
        assert len(cross_issues) >= 1

    def test_all_agg_groups_no_issues(self):
        """No issues when all groups have aggregation."""
        g1 = MulGroup(multiply=["COUNT(t.a)"])
        g2 = MulGroup(multiply=["SUM(t.b)"])
        expr = NormalizedExpr(add_groups=[g1, g2])
        issues = _check_mixed_aggregation_in_expr(expr, "select_cols[0]", "main")
        assert len(issues) == 0


class TestValidateCteGrainConsistency:
    """Tests for validate_cte_grain_consistency."""

    def test_grouped_with_group_by(self):
        """No issues for grouped CTE with group_by."""
        cte = RuntimeCteStep(
            cte_name="base",
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("t.a")],
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.a)"])]))],
        )
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        grain_errors = [i for i in issues if i.severity == "error"]
        assert len(grain_errors) == 0

    def test_grouped_without_group_by(self):
        """Error for grouped CTE without group_by."""
        cte = RuntimeCteStep(cte_name="base", grain="grouped")
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        assert any("no group_by_cols" in i.message for i in issues)

    def test_row_level_with_group_by(self):
        """Error for row_level CTE with group_by."""
        cte = RuntimeCteStep(
            cte_name="base",
            grain="row_level",
            group_by_cols=[NormalizedExpr.from_column("t.a")],
        )
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        assert any("group_by_cols" in i.message for i in issues)

    def test_row_level_with_agg(self):
        """Error for row_level CTE with aggregation."""
        cte = RuntimeCteStep(
            cte_name="base",
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.a)"])]))],
        )
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        assert any("aggregation" in i.message.lower() and "row_level" in i.message for i in issues)

    def test_scalar_with_group_by(self):
        """Error for scalar CTE with group_by."""
        cte = RuntimeCteStep(
            cte_name="base",
            grain="scalar",
            group_by_cols=[NormalizedExpr.from_column("t.a")],
        )
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        assert any("group_by_cols" in i.message for i in issues)

    def test_having_without_agg(self):
        """Error for CTE with having but no aggregation."""
        cte = RuntimeCteStep(
            cte_name="base",
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr.from_column("t.a"),
                    op=">",
                    value_type="integer",
                )
            ],
        )
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        assert any("HAVING" in i.message for i in issues)

    def test_agg_without_group_by_non_scalar_errors(self):
        """Error when grain is grouped but SELECT aggregates without GROUP BY columns."""
        cte = RuntimeCteStep(
            cte_name="base",
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.a)"])]))],
        )
        issues = validate_cte_grain_consistency(cte, "CTE 'base'")
        errors = [i for i in issues if i.severity == "error"]
        assert any("grouped" in i.message and "no group_by" in i.message for i in errors)

    def test_grouped_without_group_by_explicit(self):
        """Grouped CTE without group_by produces error."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t.a"))],
            group_by_cols=[],
        )
        issues = validate_cte_grain_consistency(cte, "cte1")
        assert any("grouped" in i.message and "no group_by" in i.message for i in issues)

    def test_row_level_with_aggregation(self):
        """Row-level CTE with aggregation produces error."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "t.a"))],
            group_by_cols=[],
        )
        issues = validate_cte_grain_consistency(cte, "cte1")
        assert any("row_level" in i.message for i in issues)

    def test_having_without_select_agg(self):
        """CTE with HAVING but no aggregation produces error."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[NormalizedExpr.from_column("t.a")],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr.from_agg("count", "t.b"),
                    op=">",
                    value_type="integer",
                )
            ],
        )
        issues = validate_cte_grain_consistency(cte, "cte1")
        assert any("HAVING" in i.message and "no aggregation" in i.message for i in issues)


class TestValidateCteDependencyGrains:
    """Tests for validate_cte_dependency_grains."""

    def test_no_ctes_no_issues(self):
        """No issues for empty CTE list."""
        issues = validate_cte_dependency_grains([], "row_level")
        assert len(issues) == 0

    def test_row_level_depends_on_grouped(self):
        """Warning when row_level CTE depends on grouped CTE."""
        cte1 = RuntimeCteStep(cte_name="agg", grain="grouped", tables=["customers"])
        cte2 = RuntimeCteStep(cte_name="detail", grain="row_level", tables=["agg"])
        issues = validate_cte_dependency_grains([cte1, cte2], "row_level")
        warnings = [i for i in issues if "incompatible" in i.issue_id.lower()]
        assert len(warnings) >= 1

    def test_main_row_level_uses_grouped_cte_no_grain_warning(self):
        """Main row_level with a grouped CTE is a join-path concern, not a cross-grain warning."""
        cte = RuntimeCteStep(cte_name="agg", grain="grouped", tables=["customers"])
        issues = validate_cte_dependency_grains([cte], "row_level")
        assert not any("main" in i.message.lower() for i in issues)

    def test_compatible_grains_no_issues(self):
        """No issues when grains are compatible."""
        cte1 = RuntimeCteStep(cte_name="base", grain="row_level", tables=["customers"])
        cte2 = RuntimeCteStep(cte_name="agg", grain="grouped", tables=["base"])
        issues = validate_cte_dependency_grains([cte1, cte2], "grouped")
        dep_issues = [i for i in issues if "cte_grain_incompatible" in i.issue_id]
        assert len(dep_issues) == 0

    def test_single_cte_grouped_main_grouped(self):
        """No issues when single CTE and main are both grouped."""
        cte = RuntimeCteStep(cte_name="base", grain="grouped", tables=["customers"])
        issues = validate_cte_dependency_grains([cte], "grouped")
        assert len(issues) == 0

    def test_row_level_depends_on_grouped_message(self):
        """Row-level CTE depending on grouped CTE produces warning."""
        cte1 = RuntimeCteStep(cte_name="cte1", grain="grouped")
        cte2 = RuntimeCteStep(cte_name="cte2", grain="row_level", tables=["cte1"])
        issues = validate_cte_dependency_grains([cte1, cte2], "row_level")
        assert any("row_level" in i.message and "aggregated" in i.message for i in issues)

    def test_row_level_on_grouped_no_warning_when_consumer_has_window(self):
        cte1 = RuntimeCteStep(cte_name="cte1", grain="grouped")
        ranked_ws = WindowSpec(
            function="rank",
            partition_by=[NormalizedExpr.from_column("cte1.g")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("cte1.n"), direction="desc")],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            grain="row_level",
            tables=["cte1"],
            select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=ranked_ws,
                )
            ],
        )
        issues = validate_cte_dependency_grains([cte1, cte2], "row_level")
        assert not any("depends on aggregated" in i.message for i in issues)

    def test_grouped_depends_on_grouped_no_issue(self):
        """Grouped CTE depending on grouped CTE produces no issue."""
        cte1 = RuntimeCteStep(cte_name="cte1", grain="grouped")
        cte2 = RuntimeCteStep(cte_name="cte2", grain="grouped", tables=["cte1"])
        issues = validate_cte_dependency_grains([cte1, cte2], "grouped")
        assert len(issues) == 0

    def test_main_row_level_with_grouped_final_cte_no_warning(self):
        """Main row_level with a grouped final CTE does not emit a main- query grain warning."""
        cte1 = RuntimeCteStep(cte_name="cte1", grain="grouped")
        issues = validate_cte_dependency_grains([cte1], "row_level")
        assert not any("Main query" in i.message for i in issues)

    def test_main_row_level_join_grouped_ctes_allowed_when_only_cte_outputs(self):
        """Outer SELECT may stay row_level when it projects only grouped CTE columns."""
        cte1 = RuntimeCteStep(cte_name="cte1", grain="grouped", tables=["customers"])
        cte2 = RuntimeCteStep(cte_name="cte2", grain="grouped", tables=["orders"])
        selects = [
            SelectCol(expr=NormalizedExpr.from_column("cte1.a")),
            SelectCol(expr=NormalizedExpr.from_column("cte2.b")),
        ]
        issues = validate_cte_dependency_grains(
            [cte1, cte2],
            "row_level",
            main_tables=["cte1", "cte2"],
            select_cols=selects,
        )
        assert not any("cte_main_grain_incompatible" in i.issue_id for i in issues)

    def test_empty_cte_steps(self):
        """Empty CTE list produces no issues."""
        assert validate_cte_dependency_grains([], "row_level") == []

    def test_no_cross_cte_dependency(self):
        """Independent CTEs produce no cross-dependency issues."""
        cte1 = RuntimeCteStep(cte_name="cte1", grain="grouped", tables=["orders"])
        cte2 = RuntimeCteStep(cte_name="cte2", grain="row_level", tables=["customers"])
        issues = validate_cte_dependency_grains([cte1, cte2], "grouped")
        assert not any("depends on aggregated" in i.message for i in issues)


class TestValidateFilterNoAggregationEdgeCases:
    """Edge-case tests for validate_filter_no_aggregation."""

    def test_right_expr_aggregated(self):
        """Error for aggregated right expression."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            right_expr=NormalizedExpr.from_agg("count", "t.b"),
            value_type="integer",
        )
        issues = validate_filter_no_aggregation([fp])
        assert len(issues) >= 1

    def test_empty_filters(self):
        """No issues for empty filter list."""
        issues = validate_filter_no_aggregation([])
        assert len(issues) == 0

    def test_none_filters(self):
        """No issues for None filter list."""
        issues = validate_filter_no_aggregation(None)
        assert len(issues) == 0


class TestValidateHavingRequiresAggregationEdgeCases:
    """Edge-case tests for validate_having_requires_aggregation."""

    def test_empty_having(self):
        """No issues for empty having list."""
        issues = validate_having_requires_aggregation([])
        assert len(issues) == 0

    def test_none_having(self):
        """No issues for None having list."""
        issues = validate_having_requires_aggregation(None)
        assert len(issues) == 0

    def test_multiple_non_agg_having(self):
        """Multiple errors for multiple non-aggregated having."""
        hp1 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="integer",
            param_key="p1",
        )
        hp2 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.b"),
            op="=",
            value_type="string",
            param_key="p2",
        )
        gb = [NormalizedExpr.from_column("t.a"), NormalizedExpr.from_column("t.b")]
        issues = validate_having_requires_aggregation([hp1, hp2], group_by_cols=gb)
        assert len(issues) == 2


class TestValidateHavingOperatorIsNumeric:
    """Tests for validate_having_operator_is_numeric."""

    def test_numeric_ops_clean(self):
        from aetherdialect._validation_semantic import (
            validate_having_operator_is_numeric,
        )

        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        assert validate_having_operator_is_numeric([hp]) == []

    def test_ilike_rejected(self):
        from aetherdialect._validation_semantic import (
            validate_having_operator_is_numeric,
        )

        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op="ilike",
            value_type="string",
        )
        issues = validate_having_operator_is_numeric([hp])
        assert len(issues) == 1


class TestValidateGroupedRequiresAggregation:
    def test_grouped_with_group_by_no_agg_error(self):
        select_cols = [SelectCol(expr=NormalizedExpr.from_column("orders.status"))]
        group_by_cols = [NormalizedExpr.from_column("orders.status")]
        issues = validate_grouped_requires_aggregation("grouped", select_cols, group_by_cols)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "aggregation" in issues[0].message.lower()

    def test_grouped_with_agg_no_issue(self):
        select_cols = [
            SelectCol(expr=NormalizedExpr.from_column("orders.status")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id")),
        ]
        group_by_cols = [NormalizedExpr.from_column("orders.status")]
        issues = validate_grouped_requires_aggregation("grouped", select_cols, group_by_cols)
        assert issues == []

    def test_grouped_having_agg_satisfies_rule(self):
        """HAVING with an aggregate expression counts as aggregation for grouped queries."""
        select_cols = [SelectCol(expr=NormalizedExpr.from_column("film.title"))]
        group_by_cols = [NormalizedExpr.from_column("film.title")]
        having = [
            HavingParam(
                left_expr=NormalizedExpr.from_agg("count", "inventory.store_id"),
                op="=",
                value_type="integer",
                param_key="p1",
            )
        ]
        issues = validate_grouped_requires_aggregation("grouped", select_cols, group_by_cols, having_param=having)
        assert issues == []

    def test_row_level_no_issue(self):
        select_cols = [SelectCol(expr=NormalizedExpr.from_column("orders.status"))]
        issues = validate_grouped_requires_aggregation("row_level", select_cols, [])
        assert issues == []

    def test_grouped_without_group_by_no_issue(self):
        select_cols = [SelectCol(expr=NormalizedExpr.from_column("orders.status"))]
        issues = validate_grouped_requires_aggregation("grouped", select_cols, [])
        assert issues == []

    def test_scalar_no_issue(self):
        select_cols = [SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id"))]
        issues = validate_grouped_requires_aggregation("scalar", select_cols, [])
        assert issues == []


class TestValidateCaseBranchAggregationConsistency:
    """Tests for validate_case_branch_aggregation_consistency."""

    @staticmethod
    def _having_case() -> CaseWhenExpr:
        agg = NormalizedExpr.from_agg("sum", "orders.amount")
        cond = FilterParam(left_expr=agg, op=">", value_type="number", param_key="p1")
        br = CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'high'"))
        return CaseWhenExpr(
            branches=[br],
            else_result=NormalizedExpr.from_column("'low'"),
            condition_scope="having",
        )

    def test_having_case_without_group_by_errors(self):
        step = CaseRegistryStep(registry_id="c01", case_when=self._having_case())
        issues = validate_case_branch_aggregation_consistency([step], [], context="main")
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "GROUP BY" in issues[0].message
        assert "case_registry[c01]" in issues[0].context.get("where", "")

    def test_having_case_with_group_by_passes(self):
        step = CaseRegistryStep(registry_id="c01", case_when=self._having_case())
        gb = [NormalizedExpr.from_column("orders.customer_id")]
        issues = validate_case_branch_aggregation_consistency([step], gb, context="main")
        assert issues == []

    def test_filter_scope_case_ignored(self):
        cond = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.status"),
            op="=",
            value_type="string",
            param_key="p1",
        )
        cw = CaseWhenExpr(
            branches=[CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("'x'"))],
            condition_scope="filter",
        )
        step = CaseRegistryStep(registry_id="c01", case_when=cw)
        issues = validate_case_branch_aggregation_consistency([step], [], context="main")
        assert issues == []


class TestValidateThresholdMissingHaving:
    """Tests for validate_threshold_missing_having."""

    def test_fires_when_grouped_agg_threshold_no_having(self):
        """Issue raised when grain='grouped', agg exists, threshold phrase matches, but no HAVING is defined."""
        issues = validate_threshold_missing_having(
            "Customers with more than 5 orders",
            [
                SelectCol(expr=NormalizedExpr.from_column("customer.name")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id")),
            ],
            [],
            "grouped",
        )
        assert len(issues) == 1
        assert issues[0].category == "threshold_missing_having"
        assert issues[0].severity == "error"

    def test_silent_when_having_present(self):
        """No issue when HAVING is already defined."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            value_type="number",
            raw_value=5,
        )
        issues = validate_threshold_missing_having(
            "Customers with more than 5 orders",
            [
                SelectCol(expr=NormalizedExpr.from_column("customer.name")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id")),
            ],
            [hp],
            "grouped",
        )
        assert issues == []

    def test_silent_when_not_grouped(self):
        """No issue when grain is row_level."""
        issues = validate_threshold_missing_having(
            "Orders with more than 5 items",
            [SelectCol(expr=NormalizedExpr.from_agg("count", "items.id"))],
            [],
            "row_level",
        )
        assert issues == []

    def test_silent_when_no_aggregation(self):
        """No issue when no aggregated select column exists."""
        issues = validate_threshold_missing_having(
            "Customers with more than 5 orders",
            [SelectCol(expr=NormalizedExpr.from_column("customer.name"))],
            [],
            "grouped",
        )
        assert issues == []

    def test_silent_when_no_threshold_phrase(self):
        """No issue when question has no threshold phrase."""
        issues = validate_threshold_missing_having(
            "List all customers and their order counts",
            [
                SelectCol(expr=NormalizedExpr.from_column("customer.name")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id")),
            ],
            [],
            "grouped",
        )
        assert issues == []

    def test_silent_when_empty_question(self):
        """No issue when natural_language is empty."""
        issues = validate_threshold_missing_having(
            "",
            [SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id"))],
            [],
            "grouped",
        )
        assert issues == []

    def test_at_least_phrase_fires(self):
        """'at least' threshold phrase triggers issue."""
        issues = validate_threshold_missing_having(
            "Actors in at least 10 films",
            [
                SelectCol(expr=NormalizedExpr.from_column("actor.name")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "film.id")),
            ],
            [],
            "grouped",
        )
        assert len(issues) == 1

    def test_context_label_propagated(self):
        """Context label appears in issue_id and context dict."""
        issues = validate_threshold_missing_having(
            "Items with more than 3 reviews",
            [
                SelectCol(expr=NormalizedExpr.from_column("items.name")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "reviews.id")),
            ],
            [],
            "grouped",
            context="cte_step",
        )
        assert len(issues) == 1
        assert "cte_step" in issues[0].issue_id
        assert issues[0].context["location"] == "cte_step"


class TestValidateForEachGrouping:
    """Tests for validate_for_each_grouping."""

    @staticmethod
    def _make_schema():
        """Build a minimal schema with customer and store tables."""
        return SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "first_name": ColumnMetadata(
                            name="first_name",
                            data_type="varchar",
                            value_type="string",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="customer_id",
                ),
                "store": TableMetadata(
                    name="store",
                    columns={
                        "store_id": ColumnMetadata(
                            name="store_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="store_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_fires_when_per_entity_missing_from_group_by(self):
        """Issue raised when 'per customer' without aggregation keyword prefix and customer not in group_by."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "show rentals per customer",
            [],
            schema,
            True,
            "main",
        )
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].category == "for_each_grouping"

    def test_silent_when_entity_in_group_by(self):
        """No issue when 'per store' and store column in group_by."""
        schema = self._make_schema()
        gb = [NormalizedExpr.from_column("store.store_id")]
        issues = validate_for_each_grouping(
            "show sales per store",
            gb,
            schema,
            True,
            "main",
        )
        assert issues == []

    def test_silent_when_no_for_each_phrase(self):
        """No issue when question has no 'for each' / 'per' phrase."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "show all customers",
            [],
            schema,
            True,
            "main",
        )
        assert issues == []

    def test_silent_when_noun_not_in_schema(self):
        """No issue when the noun after 'per' does not match any table or column."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "revenue per region",
            [],
            schema,
            True,
            "main",
        )
        assert issues == []

    def test_silent_when_no_aggregation(self):
        """No issue when question says 'for each' but intent has no aggregation context."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "show district for each customer",
            [],
            schema,
            False,
            "main",
        )
        assert issues == []

    def test_silent_on_by_phrase(self):
        """No issue when question uses 'by total payment' — removed 'by' from pattern to avoid false positives."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "top 5 customers by total payment",
            [],
            schema,
            True,
            "main",
        )
        assert issues == []


class TestForEachGroupingAggKeywordSkip:
    """Tests for validate_for_each_grouping skipping 'per X' after agg keywords."""

    @staticmethod
    def _make_schema():
        return SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="customer_id",
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
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_per_customer_after_average_keyword_skipped(self):
        """'average payment per customer' should NOT fire because 'per' is preceded by 'average'."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "average payment per customer",
            [],
            schema,
            True,
            "main",
        )
        assert issues == []

    def test_per_rental_after_total_keyword_skipped(self):
        """'total amount per rental' should NOT fire because 'per' is preceded by 'total'."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "total amount per rental",
            [],
            schema,
            True,
            "main",
        )
        assert issues == []

    def test_per_customer_after_count_keyword_skipped(self):
        """'count of films per customer' should NOT fire because 'per' is preceded by 'count'."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "count of films per customer",
            [],
            schema,
            True,
            "main",
        )
        assert issues == []

    def test_per_customer_without_agg_prefix_still_fires(self):
        """'show data per customer' fires because no agg keyword precedes 'per'."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "show data per customer",
            [],
            schema,
            True,
            "main",
        )
        assert len(issues) == 1
        assert issues[0].category == "for_each_grouping"

    def test_for_each_customer_still_fires(self):
        """'for each customer' always fires regardless of agg prefix (skip only applies to 'per')."""
        schema = self._make_schema()
        issues = validate_for_each_grouping(
            "average payment for each customer",
            [],
            schema,
            True,
            "main",
        )
        assert len(issues) == 1


class TestValidateQuestionAggKeywordCoverage:
    """Tests for validate_question_agg_keyword_coverage."""

    @staticmethod
    def _agg_select_col() -> SelectCol:
        """Return a SelectCol with an aggregated expression."""
        return SelectCol(expr=NormalizedExpr.from_agg("sum", "payment.amount"))

    def test_fires_when_total_keyword_and_no_agg(self):
        """'total' in question with no aggregation fires a warning."""
        issues = validate_question_agg_keyword_coverage(
            "total revenue per country",
            [],
            [],
        )
        assert len(issues) == 1
        assert issues[0].category == "agg_keyword_missing"
        assert issues[0].severity == "warning"

    def test_fires_when_count_keyword_and_no_agg(self):
        """'count' in question with no aggregation fires a warning."""
        issues = validate_question_agg_keyword_coverage(
            "count active customers",
            [],
            [],
        )
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_fires_when_average_keyword_and_no_agg(self):
        """'average' in question with no aggregation fires a warning."""
        issues = validate_question_agg_keyword_coverage(
            "average rental duration per rating",
            [],
            [],
        )
        assert len(issues) == 1

    def test_fires_when_how_many_keyword_and_no_agg(self):
        """'how many' in question with no aggregation fires a warning."""
        issues = validate_question_agg_keyword_coverage(
            "how many films is each actor in",
            [],
            [],
        )
        assert len(issues) == 1

    def test_silent_when_agg_present_in_select(self):
        """No issue when the intent already has an aggregated column."""
        col = self._agg_select_col()
        issues = validate_question_agg_keyword_coverage(
            "total revenue per country",
            [col],
            [],
        )
        assert issues == []

    def test_silent_when_having_present(self):
        """No issue when HAVING is present (aggregation implied)."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "rental.rental_id"),
            op=">",
            value_type="integer",
            raw_value=5,
        )
        issues = validate_question_agg_keyword_coverage(
            "total revenue per country",
            [],
            [hp],
        )
        assert issues == []

    def test_silent_when_no_agg_keyword(self):
        """No issue when the question has no aggregation keyword."""
        issues = validate_question_agg_keyword_coverage(
            "list all active customers",
            [],
            [],
        )
        assert issues == []

    def test_silent_when_question_empty(self):
        """No issue when question is empty string."""
        issues = validate_question_agg_keyword_coverage("", [], [])
        assert issues == []

    def test_issue_id_contains_context(self):
        """Issue id encodes the context label."""
        issues = validate_question_agg_keyword_coverage(
            "total revenue per country",
            [],
            [],
            context="cte_step_1",
        )
        assert "cte_step_1" in issues[0].issue_id

    def test_silent_when_aggregate_only_in_cte_steps(self):
        step = RuntimeCteStep(
            cte_name="rental_counts",
            grain="grouped",
            tables=["rental"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "rental.rental_id"))],
        )
        issues = validate_question_agg_keyword_coverage(
            "total rental count per customer",
            [],
            [],
            "main query",
            [step],
        )
        assert issues == []


class TestValidateCountThresholdMissingHaving:
    """Tests for validate_count_threshold_missing_having."""

    @staticmethod
    def _make_schema():
        """Schema with film, inventory, and store for threshold tests."""
        return SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "title": ColumnMetadata(
                            name="title",
                            data_type="varchar",
                            value_type="string",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="film_id",
                ),
                "inventory": TableMetadata(
                    name="inventory",
                    columns={
                        "inventory_id": ColumnMetadata(
                            name="inventory_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("film", "film_id"),
                        ),
                        "store_id": ColumnMetadata(
                            name="store_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("store", "store_id"),
                        ),
                    },
                    foreign_keys=[],
                    primary_key="inventory_id",
                ),
                "store": TableMetadata(
                    name="store",
                    columns={
                        "store_id": ColumnMetadata(
                            name="store_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="store_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_emits_error_when_having_missing(self):
        """Threshold phrase without HAVING produces an error."""
        schema = self._make_schema()
        issues = validate_count_threshold_missing_having(
            "list films available in exactly 2 stores",
            ["film", "inventory"],
            [],
            schema,
        )
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].category == "count_threshold_missing_having"
        assert "inventory.store_id" in issues[0].message

    def test_no_issue_when_having_present(self):
        """No issue when a HAVING clause already exists."""
        schema = self._make_schema()
        hp = HavingParam(
            left_expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(DISTINCT inventory.store_id)"])]
            ),
            op="=",
            value_type="integer",
            raw_value=2,
        )
        issues = validate_count_threshold_missing_having(
            "list films available in exactly 2 stores",
            ["film", "inventory"],
            [hp],
            schema,
        )
        assert issues == []

    def test_no_issue_when_pattern_absent(self):
        """No issue when question has no count-threshold pattern."""
        schema = self._make_schema()
        issues = validate_count_threshold_missing_having(
            "list all films",
            ["film"],
            [],
            schema,
        )
        assert issues == []

    def test_no_issue_when_word_not_in_schema(self):
        """No issue when threshold word does not resolve to a table."""
        schema = self._make_schema()
        issues = validate_count_threshold_missing_having(
            "list films in exactly 2 warehouses",
            ["film"],
            [],
            schema,
        )
        assert issues == []

    def test_fk_column_in_context(self):
        """Issue context includes the FK column reference."""
        schema = self._make_schema()
        issues = validate_count_threshold_missing_having(
            "films in exactly 3 stores",
            ["film", "inventory"],
            [],
            schema,
        )
        assert issues[0].context["fk_column"] == "inventory.store_id"
        assert issues[0].context["threshold_count"] == "3"


class TestValidateSemanticContradictionsExtended:
    """Extra branches for validate_semantic_contradictions."""

    def test_contradictory_max_and_min_agg_funcs(self):
        """MAX and MIN aggregate tokens in select trigger semantic_contradiction error."""
        sc_max = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MAX(orders.amount)"])]),
        )
        sc_min = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(orders.amount)"])]),
        )
        issues = validate_semantic_contradictions(
            [sc_max, sc_min],
            natural_language="range",
            grain="grouped",
            expected_rows="many",
        )
        assert any(i.category == "semantic_contradiction" and i.severity == "error" for i in issues)
        assert any("contradictory" in i.message.lower() for i in issues)

    def test_empty_natural_language_skips_nl_patterns(self):
        """Falsy NL still runs aggregate contradiction checks only."""
        sc_max = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MAX(orders.amount)"])]),
        )
        sc_min = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(orders.amount)"])]),
        )
        issues = validate_semantic_contradictions([sc_max, sc_min], "", "grouped", "many")
        assert any(i.category == "semantic_contradiction" for i in issues)
        assert not any(i.issue_id.startswith("nl_contradiction") for i in issues)

    def test_scalar_grain_with_few_expected_rows(self):
        """Scalar + expected_rows 'few' is treated like a contradiction."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id"))
        issues = validate_semantic_contradictions([sc], "count", "scalar", "few")
        assert any("scalar" in i.message and "few" in i.message for i in issues)

    def test_nl_zero_greater_than_warning(self):
        """'zero' and 'greater than' together yield a warning-level NL contradiction."""
        sc = SelectCol(expr=NormalizedExpr.from_column("orders.amount"))
        issues = validate_semantic_contradictions([sc], "Show zero rows greater than 10", "row_level", "many")
        assert any(i.severity == "warning" and "zero" in i.message.lower() for i in issues)

    def test_nl_empty_count_warning(self):
        """'empty' and 'count' together yield a warning."""
        sc = SelectCol(expr=NormalizedExpr.from_column("orders.amount"))
        issues = validate_semantic_contradictions([sc], "empty result count", "row_level", "many")
        assert any(i.severity == "warning" for i in issues)

    def test_nl_no_records_count_warning(self):
        """'no records' and 'count' together yield a warning."""
        sc = SelectCol(expr=NormalizedExpr.from_column("orders.amount"))
        issues = validate_semantic_contradictions([sc], "no records in count query", "row_level", "many")
        assert any("no records" in i.message.lower() or "count" in i.message.lower() for i in issues)


class TestValidateQuestionDistinctHint:
    """Tests for validate_question_distinct_hint."""

    def test_empty_question_no_issues(self):
        assert validate_question_distinct_hint("", [SelectCol(expr=NormalizedExpr.from_column("t.a"))]) == []

    def test_no_distinct_keyword_no_issues(self):
        issues = validate_question_distinct_hint(
            "list all customers",
            [SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
        )
        assert issues == []

    def test_fires_when_unique_in_question_no_distinct_in_expr(self):
        issues = validate_question_distinct_hint(
            "list unique customer names",
            [SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
        )
        assert len(issues) == 1
        assert issues[0].category == "missing_distinct"
        assert issues[0].severity == "warning"

    def test_silent_when_distinct_appears_in_multiply_term(self):
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(multiply=["DISTINCT customers.name"], coefficient=1.0)]),
        )
        issues = validate_question_distinct_hint("unique names", [sc])
        assert issues == []


class TestValidateLogicalIntentNumericCoverageExtended:
    """Additional branches for validate_logical_intent_numeric_coverage."""

    def test_top_n_number_excluded_from_missing_numeric(self):
        """Numbers only referenced in 'top N' phrases do not produce missing-numeric issues."""
        li = LogicalIntent(tables=("t",), select="*", filter="Show top 5 customers")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [],
            [],
            None,
            "main",
        )
        assert issues == []

    def test_number_covered_by_limit(self):
        li = LogicalIntent(tables=("t",), select="*", filter="Return 25 rows", limit="25")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [],
            [],
            25,
            "main",
        )
        assert issues == []

    def test_number_covered_by_having_raw_value(self):
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            value_type="integer",
            raw_value=100,
        )
        li = LogicalIntent(tables=("t",), select="*", having="Groups above 100 rentals")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [],
            [hp],
            None,
            "main",
        )
        assert not any("100" in i.issue_id for i in issues)

    def test_number_covered_by_having_param_values(self):
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            value_type="integer",
            raw_value=None,
            param_key="p1",
        )
        li = LogicalIntent(tables=("t",), select="*", having="Groups above 100 rentals")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [],
            [hp],
            None,
            "main",
            param_values={"p1": 100},
        )
        assert not any("100" in i.issue_id for i in issues)

    def test_number_covered_by_filter_param_values(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.balance"),
            op=">",
            value_type="number",
            raw_value=None,
            param_key="n1",
        )
        li = LogicalIntent(tables=("t",), select="*", filter="Customers with balance over 60")
        issues = validate_logical_intent_numeric_coverage(
            li,
            [fp],
            [],
            None,
            "main",
            param_values={"n1": 60},
        )
        assert not any("60" in i.issue_id for i in issues)

    def test_empty_question_returns_empty(self):
        assert validate_logical_intent_numeric_coverage(None, [], [], None, "main") == []

    def test_non_floatable_literal_skipped(self):
        """Invalid float in raw_value does not break scanning."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            raw_value="not-a-number",
        )
        li = LogicalIntent(tables=("t",), select="*", filter="value 99")
        issues = validate_logical_intent_numeric_coverage(li, [fp], [], None, "main")
        assert any("99" in i.message for i in issues)


class TestValidateExprVsExprFiltersExtended:
    """Branches for validate_expr_vs_expr_filters."""

    def test_empty_list(self, typed_schema):
        assert validate_expr_vs_expr_filters([], typed_schema) == []

    def test_pk_comparison_warning(self, typed_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.id"),
            op="=",
            right_expr=NormalizedExpr.from_column("orders.id"),
            value_type="integer",
        )
        issues = validate_expr_vs_expr_filters([fp], typed_schema)
        assert any(i.severity == "warning" and "primary key" in i.message.lower() for i in issues)

    def test_none_cte_outputs_defaults(self, typed_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customers.balance"),
            op=">",
            right_expr=NormalizedExpr.from_column("orders.amount"),
        )
        issues = validate_expr_vs_expr_filters([fp], typed_schema, cte_outputs=None)
        assert isinstance(issues, list)


class TestValidateFilterExprTypesIsNull:
    """Arithmetic + IS NULL should not trigger invalid-operator issues."""

    def test_is_null_on_arithmetic_allowed(self, typed_schema):
        g = MulGroup(multiply=["customers.balance"], coefficient=2.0)
        fp = FilterParam(
            left_expr=NormalizedExpr(add_groups=[g]),
            op="is null",
            value_type="string",
            param_key="pnull",
        )
        issues = validate_filter_expr_types([fp], typed_schema)
        op_issues = [i for i in issues if "invalid" in i.message.lower() and "operator" in i.message.lower()]
        assert len(op_issues) == 0


class TestValidateSingleExprTypesDateArithmetic:
    """Date columns in multiply with arithmetic skip strict numeric error."""

    def test_date_column_in_scaled_multiply_no_non_numeric_error(self, typed_schema):
        g = MulGroup(multiply=["orders.order_date"], coefficient=2.0)
        expr = NormalizedExpr(add_groups=[g])
        issues = _validate_single_expr_types(expr, typed_schema, {}, "select_cols[0]", "main")
        non_numeric = [i for i in issues if "Non-numeric" in i.message and "arithmetic" in i.message]
        assert len(non_numeric) == 0


class TestValidateGrainConsistencyExtended:
    """Additional grain / HAVING combinations."""

    def test_scalar_with_having_allowed(self):
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            value_type="integer",
        )
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id"))
        issues = validate_grain_consistency("scalar", [sc], [], [hp])
        having_grain_issues = [i for i in issues if "HAVING conditions without" in i.message]
        assert len(having_grain_issues) == 0

    def test_grouped_with_having_no_extra_having_grain_error(self):
        gb = NormalizedExpr.from_column("orders.customer_id")
        sc = SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id"))
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            value_type="integer",
        )
        issues = validate_grain_consistency("grouped", [sc], [gb], [hp])
        assert not any("HAVING conditions without aggregation" in i.message for i in issues)

    def test_context_label_in_invalid_grain_issue(self):
        issues = validate_grain_consistency("bogus", [], [], [], context="subquery")
        assert issues and issues[0].context.get("location") == "subquery"


class TestValidateSelectGroupByMembershipExtended:
    """Case-insensitivity and empty primary column."""

    def test_group_by_column_case_insensitive(self):
        sc = SelectCol(expr=NormalizedExpr.from_column("Customers.Name"))
        gb = [NormalizedExpr.from_column("customers.name")]
        issues = validate_select_group_by_membership([sc], gb, grain="grouped")
        assert issues == []

    def test_skips_empty_primary_column(self):
        sc = SelectCol(expr=NormalizedExpr())
        gb = [NormalizedExpr.from_column("t.a")]
        issues = validate_select_group_by_membership([sc], gb, grain="grouped")
        assert issues == []


class TestValidateSelectExprTypesNone:
    def test_none_select_cols_treated_as_empty(self, typed_schema):
        assert validate_select_expr_types(None, typed_schema) == []


class TestValidateNoNestedAggregationExtended:
    def test_nested_agg_in_filter_left(self):
        fp = FilterParam(
            left_expr=NormalizedExpr(
                agg_func="sum",
                add_groups=[MulGroup(coefficient=1.0, multiply=["t.a"], agg_func="count")],
            ),
            op=">",
            value_type="number",
        )
        issues = validate_no_nested_aggregation([], [], [fp], [])
        assert any("nested" in i.message.lower() for i in issues)


class TestValidateMixedAggregationInMulgroupExtended:
    def test_mixed_in_filter_expression(self):
        fp = FilterParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(t.a)", "t.b"])]),
            op=">",
            value_type="number",
        )
        issues = validate_mixed_aggregation_in_mulgroup([], [], [fp], [])
        assert len(issues) >= 1


class TestValidateOrderByAggregationContextExtended:
    def test_scalar_grain_allows_agg_order_by(self):
        obc = OrderByCol(
            expr=NormalizedExpr(
                agg_func="count",
                add_groups=[MulGroup(coefficient=1.0, multiply=["orders.id"])],
            ),
        )
        issues = validate_order_by_aggregation_context([obc], grain="scalar")
        assert issues == []


class TestValidateCteGrainConsistencyExtended:
    def test_scalar_agg_without_group_by_no_warning(self):
        cte = RuntimeCteStep(
            cte_name="totals",
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            group_by_cols=[],
        )
        issues = validate_cte_grain_consistency(cte, "main")
        warn_ids = [i.issue_id for i in issues if i.severity == "warning"]
        assert not any("agg_no_groupby" in wid for wid in warn_ids)


class TestValidateCteDependencyGrainsExtended:
    def test_main_row_level_final_scalar_cte_no_main_grain_warning(self):
        cte = RuntimeCteStep(cte_name="sum_cte", grain="scalar", tables=["customers"])
        issues = validate_cte_dependency_grains([cte], "row_level")
        assert not any("Main query" in i.message for i in issues)


class TestValidateAggVsAggHavingStarTarget:
    """COUNT(*) vs SUM(col) still runs type path when one side is star."""

    def test_count_star_vs_sum_amount_no_crash(self, typed_schema):
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(*)"])])
        right = NormalizedExpr(add_groups=[MulGroup(multiply=["SUM(orders.amount)"])])
        hp = HavingParam(left_expr=left, op=">", right_expr=right, value_type="number")
        issues = validate_agg_vs_agg_having([hp], typed_schema)
        assert isinstance(issues, list)


class TestPrivateHelpersValidationSemantic:
    """Direct tests for small private helpers."""

    def test_english_plural_forms_y_to_ies(self):
        forms = _english_plural_forms("category")
        assert "category" in forms
        assert "categories" in forms

    def test_english_plural_forms_sh(self):
        forms = _english_plural_forms("store")
        assert "store" in forms
        assert "stores" in forms

    def test_resolve_word_to_table_plural(self):
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="customer_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )
        assert _resolve_word_to_table("customers", schema) == "customer"

    def test_find_fk_column_for_target(self):
        schema = SchemaGraph(
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
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("customer", "customer_id"),
                        ),
                    },
                    foreign_keys=[],
                    primary_key="rental_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )
        assert _find_fk_column_for_target("customer", ["rental"], schema) == "rental.customer_id"


class TestValidateCountThresholdMissingHavingExtended:
    def test_generic_hint_when_no_fk_column(self):
        schema = SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="film_id",
                ),
                "store": TableMetadata(
                    name="store",
                    columns={
                        "store_id": ColumnMetadata(
                            name="store_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="store_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )
        issues = validate_count_threshold_missing_having(
            "films in exactly 2 stores",
            ["film"],
            [],
            schema,
        )
        assert len(issues) == 1
        assert issues[0].context.get("fk_column") == ""
        assert "<fk_column_referencing_store>" in issues[0].message


class TestValidateForEachGroupingExtended:
    def test_multi_word_noun_resolves_last_token_to_table(self):
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="customer_id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )
        issues = validate_for_each_grouping(
            "totals for each store customer",
            [],
            schema,
            True,
            "main",
        )
        assert len(issues) == 1
        assert issues[0].context.get("table") == "customer"


class TestTermHasAggregationExtended:
    def test_agg_with_leading_whitespace(self):
        assert _term_has_aggregation("  COUNT(t.a)") is True

    def test_agg_as_substring_not_at_start_without_space(self):
        assert _term_has_aggregation("xCOUNT(t.a)") is False

    def test_avg_term(self):
        assert _term_has_aggregation("AVG(orders.amount)") is True


class TestValidateDenyBareSelect:
    """Bare (non-aggregated) SELECT of denied columns is rejected; filters/aggregates allowed."""

    def _schema_with_deny(self) -> SchemaGraph:
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="numeric",
                            value_type="numeric",
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
            deny_columns={"orders": {"amount"}},
        )

    def test_empty_when_no_deny_list(self):
        schema = self._schema_with_deny()
        schema.deny_columns = {}
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        assert validate_deny_bare_select(intent, schema) == []

    def test_bare_denied_select_column_errors(self):
        schema = self._schema_with_deny()
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        issues = validate_deny_bare_select(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.DENY_BARE_SELECT

    def test_aggregated_denied_column_allowed(self):
        schema = self._schema_with_deny()
        intent = RuntimeIntent(
            tables=["orders"],
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        assert validate_deny_bare_select(intent, schema) == []

    def test_denied_column_in_filter_allowed(self):
        schema = self._schema_with_deny()
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op=">",
                    value_type="numeric",
                    raw_value=0,
                ),
            ],
            having_param=[],
        )
        assert validate_deny_bare_select(intent, schema) == []


class TestValidateDeniedReferences:
    """Denied columns are rejected anywhere they are referenced beyond bare SELECT."""

    def _schema_with_deny(self) -> SchemaGraph:
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="numeric",
                            value_type="numeric",
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
            deny_columns={"orders": {"amount"}},
        )

    def test_denied_in_filter_rejected(self):
        schema = self._schema_with_deny()
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op=">",
                    value_type="numeric",
                    raw_value=0,
                ),
            ],
            having_param=[],
        )
        issues = validate_denied_references(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.DENIED_REFERENCE

    def test_denied_in_aggregate_rejected(self):
        schema = self._schema_with_deny()
        intent = RuntimeIntent(
            tables=["orders"],
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        issues = validate_denied_references(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.DENIED_REFERENCE

    def test_denied_in_group_by_rejected(self):
        schema = self._schema_with_deny()
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            group_by_cols=[NormalizedExpr.from_column("orders.amount")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        issues = validate_denied_references(intent, schema)
        assert any(i.category == FailureCategory.DENIED_REFERENCE for i in issues)

    def test_denied_in_cte_order_by_rejected(self):
        schema = self._schema_with_deny()
        cte = RuntimeCteStep(
            cte_name="base",
            grain="row_level",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("orders.amount"), direction="asc")],
        )
        intent = RuntimeIntent(
            tables=["base"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("base.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            cte_steps=[cte],
        )
        issues = validate_denied_references(intent, schema)
        assert any(i.category == FailureCategory.DENIED_REFERENCE for i in issues)

    def test_no_deny_no_issue(self):
        schema = self._schema_with_deny()
        schema.deny_columns = {}
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op=">",
                    value_type="numeric",
                    raw_value=0,
                ),
            ],
            having_param=[],
        )
        assert validate_denied_references(intent, schema) == []


class TestValidateSensitiveGroupBy:
    """Sensitive columns may not be grouped on; WHERE / aggregates remain permitted."""

    def _schema_with_sensitivity(self, sensitivity: SensitivityClassification) -> SchemaGraph:
        return SchemaGraph(
            tables={
                "users": TableMetadata(
                    name="users",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                        "email": ColumnMetadata(
                            name="email",
                            data_type="text",
                            value_type="string",
                            sensitivity=sensitivity,
                        ),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_sensitive_in_group_by_rejected(self):
        schema = self._schema_with_sensitivity(SensitivityClassification.RESTRICTED)
        intent = RuntimeIntent(
            tables=["users"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "*"))],
            group_by_cols=[NormalizedExpr.from_column("users.email")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        issues = validate_sensitivity_group_by(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.SENSITIVE_GROUP_BY

    def test_hygiene_in_group_by_allowed(self):
        schema = self._schema_with_sensitivity(SensitivityClassification.NONE)
        intent = RuntimeIntent(
            tables=["users"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "*"))],
            group_by_cols=[NormalizedExpr.from_column("users.email")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        assert validate_sensitivity_group_by(intent, schema) == []

    def test_sensitive_in_filter_allowed_by_this_validator(self):
        schema = self._schema_with_sensitivity(SensitivityClassification.RESTRICTED)
        intent = RuntimeIntent(
            tables=["users"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("users.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("users.email"),
                    op="=",
                    value_type="string",
                    raw_value="x@y.com",
                ),
            ],
            having_param=[],
        )
        assert validate_sensitivity_group_by(intent, schema) == []

    def test_sensitive_in_order_by_rejected(self):
        schema = self._schema_with_sensitivity(SensitivityClassification.RESTRICTED)
        intent = RuntimeIntent(
            tables=["users"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("users.id"))],
            group_by_cols=[],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("users.email"), direction="asc")],
            filters_param=[],
            having_param=[],
        )
        issues = validate_sensitivity_order_by(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.ORDER_BY_VALIDITY

    def test_forbidden_in_order_by_blocked(self):
        """Hidden or restricted columns are blocked in ORDER BY."""
        schema = self._schema_with_sensitivity(SensitivityClassification.HIDDEN)
        intent = RuntimeIntent(
            tables=["users"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("users.id"))],
            group_by_cols=[],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("users.email"), direction="desc")],
            filters_param=[],
            having_param=[],
        )
        issues = validate_sensitivity_order_by(intent, schema)
        assert len(issues) == 1
        assert "sensitive column users.email cannot be used in ORDER BY" in issues[0].message

    def test_sensitive_in_cte_order_by_rejected(self):
        schema = self._schema_with_sensitivity(SensitivityClassification.RESTRICTED)
        cte = RuntimeCteStep(
            cte_name="base",
            grain="row_level",
            tables=["users"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("users.id"))],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("users.email"), direction="desc")],
        )
        intent = RuntimeIntent(
            tables=["base"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("base.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            cte_steps=[cte],
        )
        issues = validate_sensitivity_order_by(intent, schema)
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.ORDER_BY_VALIDITY


class TestCuratedWarmupSemanticParity:
    """Warmup curated helpers aggregate ``validate_semantics`` parity bundles."""

    def test_semantic_issues_non_empty_on_structural_violation(self, schema_graph):
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        msgs = curated_warmup_semantic_issues(intent, schema_graph)
        assert any("no tables" in m.lower() for m in msgs)

    def test_post_binding_includes_scope_registry_problems(self, schema_graph):
        ws = WindowSpec(function="row_number", partition_by=[], order_by=[])
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("w99"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            window_registry=[WindowRegistryStep(registry_id="w01", window_spec=ws)],
        )
        msgs = curated_warmup_post_binding_issues(intent, schema_graph, "SELECT 1")
        assert any("w99" in m or "registry" in m.lower() for m in msgs)


class TestValidateConcatMulgroup:
    """Structural validation for CONCAT MulGroup entries."""

    def test_divide_rejected(self) -> None:
        g = MulGroup(
            multiply=[NormalizedExpr.from_column("a.b")],
            divide=[NormalizedExpr.from_column("c.d")],
            scalar_func="concat",
        )
        issues = _validate_concat_group(g, "loc", "main query")
        assert issues

    def test_sum_outer_rejected(self) -> None:
        g = MulGroup(
            multiply=[
                NormalizedExpr.from_column("a.b"),
                NormalizedExpr.from_column("c.d"),
            ],
            scalar_func="concat",
            agg_func="sum",
        )
        issues = _validate_concat_group(g, "loc", "main query")
        assert any("COUNT" in i.message for i in issues)

    def test_count_outer_allowed(self) -> None:
        g = MulGroup(
            multiply=[
                NormalizedExpr.from_column("a.b"),
                NormalizedExpr.from_column("c.d"),
            ],
            scalar_func="concat",
            agg_func="count",
            distinct=True,
        )
        issues = _validate_concat_group(g, "loc", "main query")
        assert not issues

    def test_validate_concat_mulgroups_in_runtime_clean(self) -> None:
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
            filters_param=[],
            having_param=[],
            natural_language="q",
        )
        assert validate_concat_mulgroups_in_runtime(intent, "main query") == []
