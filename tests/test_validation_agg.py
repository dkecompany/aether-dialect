"""Tests for validation_agg module."""

from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    CteOutputColumnMeta,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._contracts_core import (
    ExprValue,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    SelectCol,
)
from aetherdialect._validation_agg import (
    expr_has_arithmetic,
    expr_result_is_numeric,
    strip_function_wrappers,
    term_result_is_numeric,
    validate_column_types,
    validate_having_agg_per_role,
    validate_order_by_agg_per_role,
    validate_order_by_agg_semantics,
    validate_pk_fk_aggregation,
    validate_scalar_expression_semantics,
    validate_scalar_func_type_semantics,
    validate_select_agg_per_role,
    validate_select_agg_semantics,
    validate_temporal_columns,
)


class TestValidateHavingAggPerRole:
    """Tests for validate_having_agg_per_role."""

    def test_valid_agg_passes(self, simple_schema):
        """validate_having_agg_per_role passes for valid agg."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(orders.amount)"])]),
            op=">",
            value_type="number",
        )
        issues = validate_having_agg_per_role([hp], simple_schema)
        assert len(issues) == 0

    def test_empty_having_no_issues(self, simple_schema):
        """validate_having_agg_per_role returns empty for no having."""
        assert validate_having_agg_per_role([], simple_schema) == []

    def test_invalid_agg_for_identifier_column_errors(self, simple_schema):
        """SUM on identifier role in HAVING produces an error."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.id)"])]),
            op=">",
            value_type="number",
        )
        issues = validate_having_agg_per_role([hp], simple_schema)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "having_agg_invalid_for_role" in issues[0].issue_id

    def test_cte_column_invalid_agg_errors(self, simple_schema):
        """Invalid aggregation against CTE output metadata in HAVING."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed", valid_aggregations=["count"])}}
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(cte1.total)"])]),
            op=">",
            value_type="number",
        )
        issues = validate_having_agg_per_role([hp], simple_schema, cte_outputs=cte_outputs)
        assert len(issues) == 1
        assert "having_agg_invalid_for_cte" in issues[0].issue_id

    def test_cte_column_name_matched_case_insensitively(self, simple_schema):
        """CTE output column keys match case-insensitively."""
        cte_outputs = {"cte1": {"Total": CteOutputColumnMeta(source="computed", valid_aggregations=["sum"])}}
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(cte1.total)"])]),
            op=">",
            value_type="number",
        )
        assert validate_having_agg_per_role([hp], simple_schema, cte_outputs=cte_outputs) == []

    def test_count_star_skipped(self, simple_schema):
        """COUNT(*) does not participate in per-role HAVING checks."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(*)"])]),
            op=">",
            value_type="number",
        )
        assert validate_having_agg_per_role([hp], simple_schema) == []

    def test_unqualified_target_skipped(self, simple_schema):
        """Unqualified agg argument is skipped."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(amount)"])]),
            op=">",
            value_type="number",
        )
        assert validate_having_agg_per_role([hp], simple_schema) == []

    def test_context_in_issue_id(self, simple_schema):
        """Custom context label appears in issue identifiers."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.id)"])]),
            op=">",
            value_type="number",
        )
        issues = validate_having_agg_per_role([hp], simple_schema, context="outer")
        assert issues and "outer" in issues[0].issue_id


class TestValidateSelectAggPerRole:
    """Tests for validate_select_agg_per_role."""

    def test_valid_count_passes(self, simple_schema):
        """validate_select_agg_per_role passes for valid COUNT."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]))
        issues = validate_select_agg_per_role([sc], simple_schema)
        assert len(issues) == 0

    def test_empty_select_no_issues(self, simple_schema):
        """validate_select_agg_per_role returns empty for no select cols."""
        assert validate_select_agg_per_role([], simple_schema) == []

    def test_invalid_agg_on_identifier_errors(self, simple_schema):
        """SUM on identifier column in SELECT is rejected."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.id)"])]))
        issues = validate_select_agg_per_role([sc], simple_schema)
        assert len(issues) == 1
        assert "select_agg_invalid_for_role" in issues[0].issue_id

    def test_bare_column_no_agg_no_issues(self, simple_schema):
        """Non-aggregated SELECT column is ignored."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        assert validate_select_agg_per_role([sc], simple_schema) == []

    def test_cte_invalid_agg_errors(self, simple_schema):
        """CTE output column restricts allowed aggregates in SELECT."""
        cte_outputs = {"cte1": {"x": CteOutputColumnMeta(source="computed", valid_aggregations=["count"])}}
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(cte1.x)"])]))
        issues = validate_select_agg_per_role([sc], simple_schema, cte_outputs=cte_outputs)
        assert len(issues) == 1
        assert "select_agg_invalid_for_cte" in issues[0].issue_id

    def test_cte_empty_valid_aggregations_skips_check(self, simple_schema):
        """When CTE meta has no valid_aggregations list, SELECT does not flag role mismatch."""
        cte_outputs = {"cte1": {"x": CteOutputColumnMeta(source="computed", valid_aggregations=[])}}
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(cte1.x)"])]))
        assert validate_select_agg_per_role([sc], simple_schema, cte_outputs=cte_outputs) == []


class TestValidateSelectAggSemantics:
    """Tests for validate_select_agg_semantics."""

    def test_sum_on_string_errors(self, simple_schema):
        """validate_select_agg_semantics errors for SUM on string column."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.name)"])]))
        issues = validate_select_agg_semantics([sc], simple_schema)
        error_issues = [i for i in issues if i.severity == "error"]
        assert len(error_issues) >= 1

    def test_sum_on_numeric_passes(self, simple_schema):
        """validate_select_agg_semantics passes for SUM on numeric."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(orders.amount)"])]))
        issues = validate_select_agg_semantics([sc], simple_schema)
        assert len(issues) == 0

    def test_min_on_free_text_warns(self):
        """validate_select_agg_semantics warns for MIN on free_text."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "desc": ColumnMetadata(
                            name="desc",
                            data_type="text",
                            value_type="string",
                            role=ColumnRole.FREE_TEXT.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(t.desc)"])]))
        issues = validate_select_agg_semantics([sc], schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 1

    def test_empty_select_no_issues(self, simple_schema):
        """Empty SELECT list yields no semantic issues."""
        assert validate_select_agg_semantics([], simple_schema) == []

    def test_avg_on_string_errors(self, simple_schema):
        """AVG on string column is an error."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["AVG(customers.name)"])]))
        issues = validate_select_agg_semantics([sc], simple_schema)
        assert any(i.severity == "error" for i in issues)

    def test_min_on_date_no_warning(self, typed_schema):
        """MIN on a date column does not produce the free-text warning."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(orders.order_date)"])]),
        )
        issues = validate_select_agg_semantics([sc], typed_schema)
        assert issues == []

    def test_unqualified_column_skipped(self, simple_schema):
        """Unqualified agg target is skipped (no crash, no issue)."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(amount)"])]))
        assert validate_select_agg_semantics([sc], simple_schema) == []

    def test_table_not_in_schema_skipped(self, simple_schema):
        """Unknown table is skipped."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(missing_tbl.col)"])]),
        )
        assert validate_select_agg_semantics([sc], simple_schema) == []


class TestValidateColumnTypes:
    """Tests for validate_column_types."""

    def test_numeric_agg_on_text_warns(self):
        """validate_column_types warns for numeric agg on text column."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(t.name)"])]))
        issues = validate_column_types([sc], schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 1

    def test_numeric_agg_on_numeric_no_issue(self):
        """validate_column_types no issue for numeric agg on numeric column."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="numeric",
                            value_type="number",
                            role=ColumnRole.NUMERIC_MEASURE.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(t.amount)"])]))
        issues = validate_column_types([sc], schema)
        assert len(issues) == 0

    def test_empty_select_cols(self, simple_schema):
        """Empty input returns no issues."""
        assert validate_column_types([], simple_schema) == []

    def test_count_on_text_no_numeric_warning(self):
        """COUNT is not treated as a numeric aggregation for text mismatch."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(t.name)"])]))
        assert validate_column_types([sc], schema) == []


class TestValidateOrderByAggPerRole:
    """Tests for validate_order_by_agg_per_role."""

    def test_empty_list_no_issues(self, simple_schema):
        """No issues for empty order_by list."""
        issues = validate_order_by_agg_per_role([], simple_schema)
        assert len(issues) == 0

    def test_bare_column_no_issues(self, simple_schema):
        """No issues for non-aggregated order by column."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="asc")
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0

    def test_valid_count_on_identifier(self, simple_schema):
        """No issues for COUNT on identifier column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0

    def test_sum_on_numeric_measure(self, simple_schema):
        """No issues for SUM on numeric measure column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(orders.amount)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0

    def test_sum_on_identifier_errors(self, simple_schema):
        """Error for SUM on identifier column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.id)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "not valid" in issues[0].message

    def test_avg_on_categorical_errors(self, simple_schema):
        """Error for AVG on categorical column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["AVG(orders.status)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 1
        assert "not valid" in issues[0].message

    def test_cte_column_valid_agg(self, simple_schema):
        """No issues for valid aggregation on CTE column."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed", valid_aggregations=["count", "sum"])}}
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(cte1.total)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema, cte_outputs=cte_outputs)
        assert len(issues) == 0

    def test_cte_column_invalid_agg(self, simple_schema):
        """Error for invalid aggregation on CTE column."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed", valid_aggregations=["count"])}}
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(cte1.total)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema, cte_outputs=cte_outputs)
        assert len(issues) == 1
        assert "not valid" in issues[0].message

    def test_count_star_skipped(self, simple_schema):
        """No issues for COUNT(*) in order by."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(*)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0

    def test_table_not_in_schema_skipped(self, simple_schema):
        """No issues when table not in schema (skipped)."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(unknown.col)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0


class TestValidateOrderByAggSemantics:
    """Tests for validate_order_by_agg_semantics."""

    def test_empty_list_no_issues(self, simple_schema):
        """No issues for empty order_by list."""
        issues = validate_order_by_agg_semantics([], simple_schema)
        assert len(issues) == 0

    def test_sum_on_numeric_passes(self, simple_schema):
        """No issues for SUM on numeric column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(orders.amount)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 0

    def test_sum_on_string_errors(self, simple_schema):
        """Error for SUM on string column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.name)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "requires numeric" in issues[0].message

    def test_avg_on_string_errors(self, simple_schema):
        """Error for AVG on string column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["AVG(customers.name)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_min_on_free_text_warns(self, simple_schema):
        """Warning for MIN on free text column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(customers.email)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) == 1
        assert "meaningless" in warnings[0].message

    def test_max_on_free_text_warns(self, simple_schema):
        """Warning for MAX on free text column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MAX(customers.email)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) == 1

    def test_min_on_numeric_passes(self, simple_schema):
        """No issues for MIN on numeric column."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(orders.amount)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 0

    def test_bare_column_no_issues(self, simple_schema):
        """No issues for non-aggregated order by column."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="asc")
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 0

    def test_count_skipped(self, simple_schema):
        """No issues for COUNT (not a numeric/minmax agg)."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.name)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 0

    def test_min_on_categorical_no_warning(self, simple_schema):
        """No warning for MIN on categorical (not free text)."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(orders.status)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], simple_schema)
        assert len(issues) == 0

    def test_min_on_date_no_warning(self, typed_schema):
        """MIN on a date column is allowed (temporal), no free-text warning."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["MIN(orders.order_date)"])]),
            direction="asc",
        )
        assert validate_order_by_agg_semantics([obc], typed_schema) == []

    def test_unqualified_column_skipped(self, simple_schema):
        """Unqualified agg target is skipped."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(amount)"])]),
            direction="asc",
        )
        assert validate_order_by_agg_semantics([obc], simple_schema) == []

    def test_unknown_table_skipped(self, simple_schema):
        """Missing table in schema is skipped without error."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(missing.col)"])]),
            direction="asc",
        )
        assert validate_order_by_agg_semantics([obc], simple_schema) == []


class TestValidateScalarFuncTypeSemantics:
    """Tests for validate_scalar_func_type_semantics."""

    def test_empty_lists_no_issues(self, simple_schema):
        """No issues for empty select and order_by lists."""
        issues = validate_scalar_func_type_semantics([], [], simple_schema)
        assert len(issues) == 0

    def test_no_scalar_no_issues(self, simple_schema):
        """No issues when no scalar function used."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 0

    def test_string_scalar_on_string_passes(self, simple_schema):
        """No issues for UPPER on string column."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(customers.name)"])]))
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 0

    def test_string_scalar_on_numeric_errors(self, simple_schema):
        """Error for UPPER on numeric column."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(customers.balance)"])])
        )
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "requires string" in issues[0].message

    def test_numeric_scalar_on_numeric_passes(self, simple_schema):
        """No issues for ABS on numeric column."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["ABS(customers.balance)"])]))
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 0

    def test_numeric_scalar_on_string_errors(self, simple_schema):
        """Error for ABS on string column."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["ABS(customers.name)"])]))
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 1
        assert "requires numeric" in issues[0].message

    def test_temporal_scalar_on_string_errors(self):
        """Error for YEAR on non-temporal column."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["YEAR(t.name)"])]))
        issues = validate_scalar_func_type_semantics([sc], [], schema)
        assert len(issues) == 1
        assert "requires temporal" in issues[0].message

    def test_temporal_scalar_on_date_passes(self):
        """No issues for YEAR on date column."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "created": ColumnMetadata(
                            name="created",
                            data_type="date",
                            value_type="date",
                            role=ColumnRole.TEMPORAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["YEAR(t.created)"])]))
        issues = validate_scalar_func_type_semantics([sc], [], schema)
        assert len(issues) == 0

    def test_agg_compatible_scalar_on_agg(self, simple_schema):
        """No issues for ROUND wrapping SUM."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["ROUND(SUM(orders.amount))"])])
        )
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 0

    def test_incompatible_scalar_on_agg(self, simple_schema):
        """Error for UPPER wrapping SUM."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(SUM(orders.amount))"])])
        )
        issues = validate_scalar_func_type_semantics([sc], [], simple_schema)
        assert len(issues) == 1
        assert "cannot wrap aggregation" in issues[0].message.lower()

    def test_order_by_scalar_type_mismatch(self, simple_schema):
        """Error for UPPER on numeric in order_by."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(customers.balance)"])]),
            direction="asc",
        )
        issues = validate_scalar_func_type_semantics([], [obc], simple_schema)
        assert len(issues) == 1
        assert "requires string" in issues[0].message

    def test_order_by_valid_scalar(self, simple_schema):
        """No issues for valid scalar in order_by."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["ABS(orders.amount)"])]),
            direction="asc",
        )
        issues = validate_scalar_func_type_semantics([], [obc], simple_schema)
        assert len(issues) == 0

    def test_none_lists_no_issues(self, simple_schema):
        """No issues when None passed for lists."""
        issues = validate_scalar_func_type_semantics(None, None, simple_schema)
        assert len(issues) == 0

    def test_multiple_select_mixed(self, simple_schema):
        """Multiple select cols with mixed validity."""
        sc1 = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(customers.name)"])]))
        sc2 = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(customers.balance)"])])
        )
        issues = validate_scalar_func_type_semantics([sc1, sc2], [], simple_schema)
        assert len(issues) == 1

    def test_column_name_resolved_case_insensitively(self, simple_schema):
        """Schema column lookup tolerates different casing in the SQL term."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["UPPER(customers.NAME)"])]))
        assert validate_scalar_func_type_semantics([sc], [], simple_schema) == []

    def test_unqualified_column_skipped(self, simple_schema):
        """Unqualified reference does not trigger type checks."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["ABS(balance)"])]))
        assert validate_scalar_func_type_semantics([sc], [], simple_schema) == []

    def test_unknown_table_skipped(self, simple_schema):
        """Unknown table causes no type issue (unresolved)."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["ABS(other_tbl.col)"])]),
        )
        assert validate_scalar_func_type_semantics([sc], [], simple_schema) == []

    def test_floor_numeric_scalar_allowed_on_agg(self, simple_schema):
        """FLOOR is in SCALAR_FUNCTIONS_NUMERIC and may wrap an aggregate."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["FLOOR(SUM(orders.amount))"])]),
        )
        assert validate_scalar_func_type_semantics([sc], [], simple_schema) == []


class TestValidateOrderByAggPerRoleEdgeCases:
    """Edge-case tests for validate_order_by_agg_per_role."""

    def test_unqualified_agg_target_skipped(self, simple_schema):
        """No issues when agg target is unqualified (skipped)."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(amount)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0

    def test_column_not_found_skipped(self, simple_schema):
        """No issues when column not found in table (skipped)."""
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.nonexistent)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc], simple_schema)
        assert len(issues) == 0

    def test_multiple_cols_mixed(self, simple_schema):
        """Multiple order by cols with mixed validity."""
        obc1 = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            direction="asc",
        )
        obc2 = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.id)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_per_role([obc1, obc2], simple_schema)
        assert len(issues) == 1


class TestValidateOrderByAggSemanticsEdgeCases:
    """Edge-case tests for validate_order_by_agg_semantics."""

    def test_sum_on_integer_passes(self):
        """No issues for SUM on integer column."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "qty": ColumnMetadata(
                            name="qty",
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
        obc = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(t.qty)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc], schema)
        assert len(issues) == 0

    def test_multiple_errors(self, simple_schema):
        """Multiple order by cols each producing errors."""
        obc1 = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.name)"])]),
            direction="asc",
        )
        obc2 = OrderByCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["AVG(customers.email)"])]),
            direction="asc",
        )
        issues = validate_order_by_agg_semantics([obc1, obc2], simple_schema)
        assert len(issues) == 2


class TestValidatePkFkAggregation:
    """Tests for validate_pk_fk_aggregation."""

    def test_sum_on_pk(self, typed_schema):
        """Warning for SUM on primary key column."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.id)"])]))
        issues = validate_pk_fk_aggregation([sc], typed_schema)
        assert any("PK/FK" in i.message for i in issues)

    def test_avg_on_fk(self, typed_schema):
        """Warning for AVG on foreign key column."""
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["AVG(orders.customer_id)"])])
        )
        issues = validate_pk_fk_aggregation([sc], typed_schema)
        assert any("PK/FK" in i.message for i in issues)

    def test_count_on_pk_no_issue(self, typed_schema):
        """No issues for COUNT on primary key column."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]))
        issues = validate_pk_fk_aggregation([sc], typed_schema)
        assert len(issues) == 0

    def test_bare_column_not_aggregated(self, typed_schema):
        """Non-aggregated columns are ignored."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.id"))
        assert validate_pk_fk_aggregation([sc], typed_schema) == []

    def test_unqualified_agg_target_skipped(self, typed_schema):
        """Unqualified column reference is skipped."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(id)"])]))
        assert validate_pk_fk_aggregation([sc], typed_schema) == []


class TestExprHasArithmetic:
    """Tests for expr_has_arithmetic."""

    def test_bare_column(self):
        """Bare column has no arithmetic."""
        expr = NormalizedExpr.from_column("t.a")
        assert expr_has_arithmetic(expr) is False

    def test_multiple_add_groups(self):
        """Multiple add groups indicates arithmetic."""
        g1 = MulGroup(multiply=["t.a"])
        g2 = MulGroup(multiply=["t.b"])
        expr = NormalizedExpr(add_groups=[g1, g2])
        assert expr_has_arithmetic(expr) is True

    def test_coefficient(self):
        """Non-unit coefficient indicates arithmetic."""
        g = MulGroup(multiply=["t.a"], coefficient=100.0)
        expr = NormalizedExpr(add_groups=[g])
        assert expr_has_arithmetic(expr) is True

    def test_division(self):
        """Division indicates arithmetic."""
        g = MulGroup(multiply=["t.a"], divide=["t.b"])
        expr = NormalizedExpr(add_groups=[g])
        assert expr_has_arithmetic(expr) is True

    def test_add_and_sub_group_counts_as_arithmetic(self):
        """One additive and one subtractive group implies arithmetic."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"])],
            sub_groups=[MulGroup(multiply=["t.b"])],
        )
        assert expr_has_arithmetic(expr) is True


class TestValidateTemporalColumns:
    """Tests for validate_temporal_columns."""

    def test_temporal_op_without_date_warns(self, typed_schema):
        """validate_temporal_columns warns for temporal op without date column."""
        sc = SelectCol(
            expr=NormalizedExpr(
                agg_func="latest",
                add_groups=[MulGroup(coefficient=1.0, multiply=["LATEST(customers.name)"])],
            ),
        )
        issues = validate_temporal_columns([sc], typed_schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 1

    def test_no_temporal_op_passes(self, typed_schema):
        """validate_temporal_columns passes with no temporal ops."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["customers.balance"], agg_func="sum")],
            ),
        )
        issues = validate_temporal_columns([sc], typed_schema)
        assert len(issues) == 0

    def test_temporal_op_with_date_column_passes(self, typed_schema):
        """A real date column in the SELECT satisfies the temporal check."""
        sc_latest = SelectCol(
            expr=NormalizedExpr(
                agg_func="latest",
                add_groups=[MulGroup(coefficient=1.0, multiply=["LATEST(customers.name)"])],
            ),
        )
        sc_date = SelectCol(expr=NormalizedExpr.from_column("orders.order_date"))
        assert validate_temporal_columns([sc_latest, sc_date], typed_schema) == []

    def test_temporal_op_satisfied_by_column_name_hint(self, typed_schema):
        """Column names containing date-like hints count as temporal evidence."""
        sc_latest = SelectCol(
            expr=NormalizedExpr(
                agg_func="latest",
                add_groups=[MulGroup(coefficient=1.0, multiply=["LATEST(customers.name)"])],
            ),
        )
        sc_hint = SelectCol(expr=NormalizedExpr.from_column("customers.created_at"))
        assert validate_temporal_columns([sc_latest, sc_hint], typed_schema) == []

    def test_temporal_op_on_unknown_table_name_hint(self, simple_schema):
        """Hints in the column name work even when the table is not in the schema."""
        sc_latest = SelectCol(
            expr=NormalizedExpr(
                agg_func="latest",
                add_groups=[MulGroup(coefficient=1.0, multiply=["LATEST(customers.name)"])],
            ),
        )
        sc_other = SelectCol(expr=NormalizedExpr.from_column("other.event_date"))
        assert validate_temporal_columns([sc_latest, sc_other], simple_schema) == []

    def test_empty_select_cols(self, typed_schema):
        """Empty SELECT list returns no issues."""
        assert validate_temporal_columns([], typed_schema) == []

    def test_latest_without_expr_agg_func_not_flagged(self, typed_schema):
        """LATEST in the term without expr-level agg metadata does not count as temporal agg."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["LATEST(customers.name)"])]))
        assert validate_temporal_columns([sc], typed_schema) == []


class TestValidateScalarExpressionSemantics:
    """Tests for validate_scalar_expression_semantics."""

    def test_numeric_scalar_on_string_warns(self, typed_schema):
        """validate_scalar_expression_semantics warns for numeric scalar on string column."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["round(customers.name)"])],
            ),
        )
        issues = validate_scalar_expression_semantics([sc], typed_schema)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 1

    def test_numeric_scalar_on_numeric_passes(self, typed_schema):
        """validate_scalar_expression_semantics passes for numeric scalar on numeric."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["round(customers.balance)"])],
            ),
        )
        issues = validate_scalar_expression_semantics([sc], typed_schema)
        assert len(issues) == 0

    def test_empty_select_cols(self, typed_schema):
        """Empty SELECT list returns no issues."""
        assert validate_scalar_expression_semantics([], typed_schema) == []

    def test_string_scalar_on_non_string_warns(self, typed_schema):
        """String scalars on numeric columns produce warnings."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["upper(customers.balance)"])],
            ),
        )
        issues = validate_scalar_expression_semantics([sc], typed_schema)
        assert any("non-string" in i.message.lower() for i in issues)

    def test_plain_aggregate_skipped(self, typed_schema):
        """Outer SQL aggregate functions are not validated as scalars."""
        sc = SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["SUM(customers.name)"])]))
        assert validate_scalar_expression_semantics([sc], typed_schema) == []

    def test_numeric_scalar_on_string_suppressed_when_aggregated(self, typed_schema):
        """ROUND(SUM(...)) on a string base does not warn: inner type check is skipped when aggregated."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(coefficient=1.0, multiply=["ROUND(SUM(customers.name))"])],
            ),
        )
        assert validate_scalar_expression_semantics([sc], typed_schema) == []


class TestStripFunctionWrappers:
    """Tests for strip_function_wrappers."""

    def test_bare_column(self):
        """Return bare column unchanged."""
        assert strip_function_wrappers("table1.col1") == "table1.col1"

    def test_single_wrapper(self):
        """Strip single function wrapper."""
        assert strip_function_wrappers("UPPER(table1.col1)") == "table1.col1"

    def test_nested_wrappers(self):
        """Strip nested function wrappers."""
        assert strip_function_wrappers("ABS(SUM(table1.col1))") == "table1.col1"

    def test_distinct_wrapper(self):
        """Strip DISTINCT keyword inside wrapper."""
        assert strip_function_wrappers("COUNT(DISTINCT table1.col1)") == "table1.col1"

    def test_empty_string(self):
        """Return empty string unchanged."""
        assert strip_function_wrappers("") == ""

    def test_star(self):
        """Return star from COUNT(*)."""
        assert strip_function_wrappers("COUNT(*)") == "*"

    def test_whitespace_inside_parens_stripped(self):
        """strip_function_wrappers strips inner whitespace."""
        assert strip_function_wrappers("  ABS(  t.col  )  ") == "t.col"

    def test_nested_scalar_and_agg(self):
        """Outermost wrapper is peeled until a bare column remains."""
        assert strip_function_wrappers("ROUND(UPPER(t.col))") == "t.col"


class TestTermResultIsNumeric:
    """Tests for term_result_is_numeric."""

    def test_count_is_numeric(self):
        """COUNT always returns numeric."""
        assert term_result_is_numeric("COUNT(t.a)") is True

    def test_sum_is_numeric(self):
        """SUM always returns numeric."""
        assert term_result_is_numeric("SUM(t.a)") is True

    def test_avg_is_numeric(self):
        """AVG always returns numeric."""
        assert term_result_is_numeric("AVG(t.a)") is True

    def test_abs_is_numeric(self):
        """ABS always returns numeric."""
        assert term_result_is_numeric("ABS(t.a)") is True

    def test_round_is_numeric(self):
        """ROUND always returns numeric."""
        assert term_result_is_numeric("ROUND(t.a)") is True

    def test_upper_not_numeric(self):
        """UPPER does not return numeric."""
        assert term_result_is_numeric("UPPER(t.a)") is False

    def test_bare_column_not_numeric(self):
        """Bare column is not guaranteed numeric."""
        assert term_result_is_numeric("t.a") is False

    def test_lower_returns_false(self):
        """LOWER is not a numeric-result function."""
        assert term_result_is_numeric("LOWER(t.col)") is False

    def test_nested_numeric(self):
        """ABS wrapping SUM is numeric."""
        assert term_result_is_numeric("ABS(SUM(t.a))") is True

    def test_min_not_guaranteed_numeric(self):
        """MIN returns False - result type depends on column."""
        assert term_result_is_numeric("MIN(t.amount)") is False

    def test_non_function_pattern_returns_false(self):
        """Term without function pattern returns False."""
        assert term_result_is_numeric("plain_column") is False

    def test_count_distinct_is_numeric(self):
        """COUNT(DISTINCT ...) is still numeric."""
        assert term_result_is_numeric("COUNT(DISTINCT t.a)") is True

    def test_ceil_is_numeric(self):
        """CEIL is a numeric-result scalar."""
        assert term_result_is_numeric("CEIL(t.a)") is True

    def test_whitespace_around_function(self):
        """Leading whitespace does not break detection."""
        assert term_result_is_numeric("  SUM( t.a )") is True

    def test_empty_inner_returns_false(self):
        """SUM() peels to empty and yields False."""
        assert term_result_is_numeric("SUM()") is False


class TestExprResultIsNumeric:
    """Tests for expr_result_is_numeric."""

    def test_count_agg_func(self, typed_schema):
        """Return True for count agg_func."""
        expr = NormalizedExpr(agg_func="count", add_groups=[MulGroup(multiply=["t.a"])])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_arithmetic_expr(self, typed_schema):
        """Return True for arithmetic expression."""
        g1 = MulGroup(multiply=["customers.balance"])
        g2 = MulGroup(multiply=["orders.amount"])
        expr = NormalizedExpr(add_groups=[g1, g2])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_numeric_column(self, typed_schema):
        """Return True for numeric primary column."""
        expr = NormalizedExpr.from_column("customers.balance")
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_string_column(self, typed_schema):
        """Return False for string primary column."""
        expr = NormalizedExpr.from_column("customers.name")
        assert expr_result_is_numeric(expr, typed_schema, {}) is False

    def test_group_level_agg(self, typed_schema):
        """Return True for group-level numeric agg."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"], agg_func="sum")])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_inline_count(self, typed_schema):
        """Return True for inline COUNT term."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.a)"])])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_unknown_column_returns_none(self, typed_schema):
        """Return None for column not in schema or CTE."""
        expr = NormalizedExpr.from_column("unknown_table.unknown_col")
        assert expr_result_is_numeric(expr, typed_schema, {}) is None

    def test_expr_level_scalar_func_numeric(self, typed_schema):
        """Expression-level scalar_func marks result numeric."""
        expr = NormalizedExpr(scalar_func="round", add_groups=[MulGroup(multiply=["customers.balance"])])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_literal_add_values_numeric(self, typed_schema):
        """Literal-only additive values are numeric."""
        expr = NormalizedExpr(add_values=[ExprValue(value=1.0)])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_mulgroup_inner_scalar_numeric(self, typed_schema):
        """Per-group inner_scalar_func can force numeric."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.balance"], inner_scalar_func="abs")],
        )
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_mulgroup_agg_sum_numeric(self, typed_schema):
        """Per-group agg_func SUM implies numeric."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["customers.balance"], agg_func="sum")])
        assert expr_result_is_numeric(expr, typed_schema, {}) is True

    def test_cte_column_numeric(self, typed_schema):
        """CTE output value_type resolves numeric."""
        cte_outputs = {
            "sq": {"n": CteOutputColumnMeta(source="computed", data_type="numeric", value_type="number")},
        }
        expr = NormalizedExpr.from_column("sq.n")
        assert expr_result_is_numeric(expr, typed_schema, cte_outputs) is True

    def test_cte_column_string_false(self, typed_schema):
        """CTE string column is not numeric."""
        cte_outputs = {
            "sq": {"lbl": CteOutputColumnMeta(source="computed", data_type="varchar", value_type="string")},
        }
        expr = NormalizedExpr.from_column("sq.lbl")
        assert expr_result_is_numeric(expr, typed_schema, cte_outputs) is False


class TestExprHasArithmeticEdgeCases:
    """Edge-case tests for expr_has_arithmetic."""

    def test_add_values(self):
        """Arithmetic when add_values present."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"])], add_values=[ExprValue(value=1.0)])
        assert expr_has_arithmetic(expr) is True

    def test_sub_values(self):
        """Arithmetic when sub_values present."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a"])], sub_values=[ExprValue(value=1.0)])
        assert expr_has_arithmetic(expr) is True

    def test_multiple_multiply_terms(self):
        """Arithmetic when multiple multiply terms."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["t.a", "t.b"])])
        assert expr_has_arithmetic(expr) is True

    def test_sub_groups(self):
        """Arithmetic when sub_groups present."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.a"])],
            sub_groups=[MulGroup(multiply=["t.b"])],
        )
        assert expr_has_arithmetic(expr) is True
