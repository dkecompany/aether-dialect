"""Tests for validation_schema module."""

import pytest

from aetherdialect._contracts_base import (
    ColumnRole,
    FailureCategory,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    WhereParam,
    predicate_group_from_list,
)
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ColumnMetadata,
    CteOutputColumnMeta,
    FKEdge,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
    registry_render_scope,
)
from aetherdialect._validation_schema import (
    _validate_agg_func_valid,
    _validate_having_agg,
    _validate_scalar_func_valid,
    _validate_where_col,
    extract_agg_col,
    extract_col_from_scalar_wrapper,
    extract_functions_from_term,
    get_col_meta,
    get_col_type,
    is_col_arithmetic_role,
    is_col_numeric,
    selectability_exempt_qualified_refs,
    validate_case_when_schema,
    validate_contains_array_filters,
    validate_date_diff_units,
    validate_date_window_units,
    validate_expr_no_extract_epoch,
    validate_filters_schema,
    validate_group_by_cols_schema,
    validate_having_ops_per_column,
    validate_having_schema,
    validate_no_between_ops,
    validate_null_filters,
    validate_order_by_cols_schema,
    validate_redundant_extract_year_column_literals,
    validate_select_cols_schema,
    validate_selectability,
    validate_where_ops_per_column,
    validate_where_value_type_alignment,
    validate_window_partition_group_by_alignment,
    validate_window_spec_schema,
)


class TestExtractColFromScalarWrapper:
    """Tests for extract_col_from_scalar_wrapper."""

    def test_bare_column(self):
        """Return bare column unchanged."""
        assert extract_col_from_scalar_wrapper("table1.col1") == "table1.col1"

    def test_abs_wrapper(self):
        """Extract column from ABS wrapper."""
        assert extract_col_from_scalar_wrapper("ABS(table1.col1)") == "table1.col1"

    def test_round_wrapper(self):
        """Extract column from ROUND wrapper."""
        assert extract_col_from_scalar_wrapper("ROUND(table1.col1)") == "table1.col1"

    def test_non_scalar_function(self):
        """Return expression unchanged for non-scalar function."""
        assert extract_col_from_scalar_wrapper("COUNT(table1.col1)") == "COUNT(table1.col1)"

    def test_empty_string(self):
        """Return empty string unchanged."""
        assert extract_col_from_scalar_wrapper("") == ""

    def test_distinct_prefix_on_bare_column(self):
        """Leading DISTINCT is stripped when not inside a scalar wrapper."""
        assert extract_col_from_scalar_wrapper("DISTINCT table1.col1") == "table1.col1"

    def test_scalar_wrapper_whitespace_and_case(self):
        """Regex tolerates spacing and ignores case on function name."""
        assert extract_col_from_scalar_wrapper("  abs (  table1.col1  )  ") == "table1.col1"

    def test_distinct_inside_scalar_wrapper(self):
        """DISTINCT inside allowed scalar unwraps to the qualified column."""
        assert extract_col_from_scalar_wrapper("ABS(DISTINCT table1.col1)") == "table1.col1"


class TestValidateScalarFuncValid:
    """Tests for validate_scalar_func_valid."""

    def test_valid_scalar(self):
        """No issues for valid scalar function."""
        issues = _validate_scalar_func_valid("round", "test", "select")
        assert len(issues) == 0

    def test_invalid_scalar(self):
        """Error for unknown scalar function."""
        issues = _validate_scalar_func_valid("foobar", "test", "select")
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_none_scalar(self):
        """No issues when scalar_func is None."""
        issues = _validate_scalar_func_valid(None, "test", "select")
        assert len(issues) == 0

    def test_empty_string_scalar_treated_as_absent(self):
        """Empty string is falsy; no validation issue (same as None)."""
        issues = _validate_scalar_func_valid("", "test", "select")
        assert issues == []


class TestValidateAggFuncValid:
    """Tests for validate_agg_func_valid."""

    def test_valid_agg(self):
        """No issues for valid aggregation function."""
        issues = _validate_agg_func_valid("count", "test", "select")
        assert len(issues) == 0

    def test_invalid_agg(self):
        """Error for unknown aggregation function."""
        issues = _validate_agg_func_valid("mode", "test", "select")
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_count_distinct_not_valid_agg(self):
        """count_distinct is not a valid aggregation function name."""
        issues = _validate_agg_func_valid("count_distinct", "test", "select")
        assert len(issues) == 1

    def test_empty_string_agg_treated_as_absent(self):
        issues = _validate_agg_func_valid("", "test", "select")
        assert issues == []


class TestExtractAggCol:
    """Tests for extract_agg_col."""

    def test_count_col(self):
        """Extract COUNT aggregation."""
        func, target, has_distinct = extract_agg_col("COUNT(table1.col1)")
        assert func == "count"
        assert target == "table1.col1"
        assert has_distinct is False

    def test_count_distinct(self):
        """Extract COUNT DISTINCT."""
        func, target, has_distinct = extract_agg_col("COUNT(DISTINCT table1.col1)")
        assert func == "count"
        assert target == "table1.col1"
        assert has_distinct is True

    def test_count_star(self):
        """Extract COUNT(*)."""
        func, target, _ = extract_agg_col("COUNT(*)")
        assert func == "count"
        assert target == "*"

    def test_no_agg(self):
        """Return None for bare column."""
        func, target, _ = extract_agg_col("table1.col1")
        assert func is None

    def test_agg_inner_scalar_unwraps(self):
        """Inner target runs through scalar wrapper stripping."""
        func, target, has_distinct = extract_agg_col("SUM(ABS(table1.col1))")
        assert func == "sum"
        assert target == "table1.col1"
        assert has_distinct is False

    def test_malformed_missing_paren_returns_none(self):
        assert extract_agg_col("COUNT(table1.col1") == (None, None, False)


class TestValidateExprNoExtractEpoch:
    """Tests for validate_expr_no_extract_epoch."""

    def test_top_level_extract_epoch(self):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"])],
            scalar_func="extract",
            scalar_func_args=["epoch", "ts"],
        )
        issues = validate_expr_no_extract_epoch(expr, "ctx", "main")
        assert len(issues) == 1
        assert issues[0].category == "extract_epoch"

    def test_inner_scalar_extract_epoch(self):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"])],
            inner_scalar_func="extract",
            inner_scalar_func_args=["EPOCH", "x"],
        )
        issues = validate_expr_no_extract_epoch(expr, "ctx", "main")
        assert any("inner" in i.issue_id for i in issues)

    def test_mul_group_scalar_extract_epoch(self):
        expr = NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=["customers.id"],
                    scalar_func="extract",
                    scalar_func_args=["epoch", "u"],
                )
            ],
        )
        issues = validate_expr_no_extract_epoch(expr, "ctx", "main")
        assert any("group" in i.issue_id for i in issues)

    def test_sub_group_inner_extract_epoch(self):
        expr = NormalizedExpr(
            sub_groups=[
                MulGroup(
                    multiply=["t.c"],
                    inner_scalar_func="extract",
                    inner_scalar_func_args=["epoch"],
                )
            ],
        )
        issues = validate_expr_no_extract_epoch(expr, "ctx", "main")
        assert any("inner_group" in i.issue_id for i in issues)

    def test_extract_day_allowed(self):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"])],
            scalar_func="extract",
            scalar_func_args=["day", "d"],
        )
        assert validate_expr_no_extract_epoch(expr, "ctx", "main") == []


class TestExtractFunctionsFromTerm:
    """Tests for extract_functions_from_term."""

    def test_scalar_wrapping_agg(self):
        """Extract scalar and agg from ROUND(SUM(t.a))."""
        scalar, agg = extract_functions_from_term("ROUND(SUM(table1.col1))")
        assert scalar == "round"
        assert agg == "sum"

    def test_bare_agg(self):
        """Extract agg only from COUNT(t.a)."""
        scalar, agg = extract_functions_from_term("COUNT(table1.col1)")
        assert scalar is None
        assert agg == "count"

    def test_no_function(self):
        """Return None for bare column."""
        scalar, agg = extract_functions_from_term("table1.col1")
        assert scalar is None
        assert agg is None

    def test_empty_string(self):
        """Return None for empty string."""
        scalar, agg = extract_functions_from_term("")
        assert scalar is None
        assert agg is None

    def test_outer_neither_valid_agg(self):
        """Unknown outer function wrapping a column yields outer name only."""
        scalar, agg = extract_functions_from_term("FOO(table1.col1)")
        assert scalar == "foo"
        assert agg is None


class TestValidateSelectColsSchema:
    """Tests for validate_select_cols_schema."""

    def test_valid_select(self, simple_schema):
        """No issues for valid qualified column."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        issues = validate_select_cols_schema([sc], simple_schema, {"customers", "orders"})
        assert len(issues) == 0

    def test_empty_select_cols(self, simple_schema):
        """Error for empty select_cols."""
        issues = validate_select_cols_schema([], simple_schema, {"customers"})
        assert len(issues) == 1
        assert "empty" in issues[0].message

    def test_unqualified_column(self, simple_schema):
        """Error for unqualified column reference."""
        sc = SelectCol(expr=NormalizedExpr.from_column("name"))
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"})
        assert any("qualified" in i.message for i in issues)

    def test_column_not_found(self, simple_schema):
        """Error for column that does not exist in table."""
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.nonexistent"))
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"})
        assert any("not in table" in i.message for i in issues)

    def test_table_not_allowed(self, simple_schema):
        """Error for table not in allowed set."""
        sc = SelectCol(expr=NormalizedExpr.from_column("products.name"))
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"})
        assert any("not in allowed" in i.message for i in issues)

    def test_cte_column_valid(self, simple_schema):
        cte = {"rollup": {"total": CteOutputColumnMeta(source="aggregation", data_type="numeric")}}
        sc = SelectCol(expr=NormalizedExpr.from_column("rollup.total"))
        assert validate_select_cols_schema([sc], simple_schema, {"customers"}, cte_outputs=cte) == []

    def test_cte_column_missing(self, simple_schema):
        cte = {"rollup": {"total": CteOutputColumnMeta(source="aggregation")}}
        sc = SelectCol(expr=NormalizedExpr.from_column("rollup.missing"))
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"}, cte_outputs=cte)
        assert any("not in CTE" in i.message for i in issues)

    def test_empty_primary_column_issue(self, simple_schema):
        sc = SelectCol(expr=NormalizedExpr())
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"})
        assert any("empty" in i.message.lower() for i in issues)

    def test_invalid_agg_on_select(self, simple_schema):
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["customers.id"], agg_func="mode")])
        sc = SelectCol(expr=expr)
        issues = validate_select_cols_schema([sc], simple_schema, {"customers", "orders"})
        assert any("Invalid aggregation" in i.message for i in issues)

    def test_extract_epoch_on_select(self, simple_schema):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"])],
            scalar_func="extract",
            scalar_func_args=["epoch", "e"],
        )
        sc = SelectCol(expr=expr)
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"})
        assert any(i.category == "extract_epoch" for i in issues)

    def test_case_insensitive_schema_column_match(self, simple_schema):
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.ID"))
        issues = validate_select_cols_schema([sc], simple_schema, {"customers"})
        assert issues == []


class TestValidateOrderByColsSchema:
    """Tests for validate_order_by_cols_schema."""

    def test_valid_order_by(self, simple_schema):
        """No issues for valid order by column."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="asc")
        issues = validate_order_by_cols_schema([obc], simple_schema, {"customers"})
        assert len(issues) == 0

    def test_invalid_direction(self, simple_schema):
        """Error for invalid direction."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="up")
        issues = validate_order_by_cols_schema([obc], simple_schema, {"customers"})
        assert any("direction" in i.message for i in issues)

    def test_empty_order_by_list(self, simple_schema):
        assert validate_order_by_cols_schema([], simple_schema, {"customers"}) == []

    def test_empty_order_expr(self, simple_schema):
        obc = OrderByCol(expr=NormalizedExpr(), direction="ASC")
        issues = validate_order_by_cols_schema([obc], simple_schema, {"customers"})
        assert any("empty" in i.message.lower() for i in issues)

    def test_desc_uppercased_ok(self, simple_schema):
        obc = OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="desc")
        assert validate_order_by_cols_schema([obc], simple_schema, {"customers"}) == []

    def test_cte_order_by_column(self, simple_schema):
        cte = {"x": {"n": CteOutputColumnMeta(source="computed")}}
        obc = OrderByCol(expr=NormalizedExpr.from_column("x.n"), direction="ASC")
        assert validate_order_by_cols_schema([obc], simple_schema, {"customers"}, cte_outputs=cte) == []

    def test_extract_epoch_in_order_by(self, simple_schema):
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"])],
            scalar_func="extract",
            scalar_func_args=["epoch", "e"],
        )
        obc = OrderByCol(expr=expr, direction="ASC")
        issues = validate_order_by_cols_schema([obc], simple_schema, {"customers"})
        assert any(i.category == "extract_epoch" for i in issues)


class TestValidateGroupByColsSchema:
    """Tests for validate_group_by_cols_schema."""

    def test_valid_group_by(self, simple_schema):
        """No issues for valid group by column."""
        gb = NormalizedExpr.from_column("customers.name")
        issues = validate_group_by_cols_schema([gb], simple_schema, {"customers"})
        assert len(issues) == 0

    def test_column_not_found(self, simple_schema):
        """Error for column not found in schema."""
        gb = NormalizedExpr.from_column("customers.nonexistent")
        issues = validate_group_by_cols_schema([gb], simple_schema, {"customers"})
        assert any("not in table" in i.message for i in issues)

    def test_unqualified_group_by(self, simple_schema):
        gb = NormalizedExpr.from_column("name")
        issues = validate_group_by_cols_schema([gb], simple_schema, {"customers"})
        assert any("qualified" in i.message for i in issues)

    def test_table_not_allowed_group_by(self, simple_schema):
        gb = NormalizedExpr.from_column("orders.amount")
        issues = validate_group_by_cols_schema([gb], simple_schema, {"customers"})
        assert any("not in allowed" in i.message for i in issues)

    def test_cte_column_not_found(self, simple_schema):
        cte = {"c": {"k": CteOutputColumnMeta(source="computed")}}
        gb = NormalizedExpr.from_column("c.missing")
        issues = validate_group_by_cols_schema([gb], simple_schema, {"customers"}, cte_outputs=cte)
        assert any("not in CTE" in i.message for i in issues)

    def test_cte_column_not_groupable_warning(self, simple_schema):
        cte = {"c": {"k": CteOutputColumnMeta(source="computed", groupable=False)}}
        gb = NormalizedExpr.from_column("c.k")
        issues = validate_group_by_cols_schema([gb], simple_schema, {"customers"}, cte_outputs=cte)
        assert any(i.severity == "warning" and "not recommended" in i.message for i in issues)

    def test_base_column_not_groupable_warning(self, typed_schema):
        gb = NormalizedExpr.from_column("customers.description")
        issues = validate_group_by_cols_schema([gb], typed_schema, {"customers"})
        assert any(i.severity == "warning" for i in issues)


class TestValidateFiltersSchema:
    """Tests for validate_filters_schema."""

    def test_valid_filter(self, simple_schema):
        """No issues for valid filter."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
            raw_value="Acme",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        assert len(issues) == 0

    def test_filter_cleared_raw_value_ok_when_param_bound(self, simple_schema):
        """Post-binding ``param_values`` satisfies FILTER when ``raw_value`` is cleared."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
            raw_value=None,
            param_key="s1",
        )
        issues = validate_filters_schema(
            [fp],
            simple_schema,
            {"customers"},
            param_values={"s1": "bound"},
        )
        missing = [i for i in issues if "missing_value" in i.issue_id]
        assert len(missing) == 0

    def test_invalid_op(self, simple_schema):
        """Error for invalid filter operator."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="<=>",
            value_type="string",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        assert any("Invalid filter operator" in i.message for i in issues)

    def test_empty_filters(self, simple_schema):
        """No issues for empty filter list."""
        issues = validate_filters_schema([], simple_schema, {"customers"})
        assert len(issues) == 0

    def test_valid_bool_op_and(self, simple_schema):
        """No issues for filter with bool_op 'AND'."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        bool_op_issues = [i for i in issues if "bool_op" in i.message]
        assert len(bool_op_issues) == 0

    def test_valid_bool_op_or(self, simple_schema):
        """No issues for filter with bool_op 'OR'."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        bool_op_issues = [i for i in issues if "bool_op" in i.message]
        assert len(bool_op_issues) == 0

    def test_invalid_bool_op_raises_issue(self, simple_schema):
        """Leaf params no longer carry bool_op; schema validation ignores removed fields."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        assert not any("bool_op" in i.message for i in issues)

    def test_invalid_value_type_when_literal_filter(self, simple_schema):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="not_a_valid_value_type",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        assert any("Invalid filter value_type" in i.message for i in issues)

    def test_is_null_skips_value_type_validation_in_filters_schema(self, simple_schema):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="is null",
            value_type="garbage",
        )
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        assert not any("Invalid filter value_type" in i.message for i in issues)

    def test_right_expr_validated(self, simple_schema):
        fp = WhereParam.__new__(WhereParam)
        object.__setattr__(fp, "left_expr", NormalizedExpr.from_column("customers.name"))
        object.__setattr__(fp, "right_expr", NormalizedExpr.from_column("orders.badcol"))
        object.__setattr__(fp, "op", "=")
        object.__setattr__(fp, "value", None)
        object.__setattr__(fp, "value_type", "string")
        object.__setattr__(fp, "raw_value", None)
        object.__setattr__(fp, "param_key", "")
        object.__setattr__(fp, "bool_op", "AND")
        object.__setattr__(fp, "where_group", None)
        issues = validate_filters_schema([fp], simple_schema, {"customers", "orders"})
        assert any("right_col" in i.issue_id and "not in table" in i.message for i in issues)

    def test_extract_epoch_on_filter_side(self, simple_schema):
        left = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"])],
            scalar_func="extract",
            scalar_func_args=["epoch", "e"],
        )
        fp = WhereParam(left_expr=left, op=">", value_type="integer")
        issues = validate_filters_schema([fp], simple_schema, {"customers"})
        assert any(i.category == "extract_epoch" for i in issues)


class TestValidateHavingSchema:
    """Tests for validate_having_schema."""

    def test_valid_having(self, simple_schema):
        """No issues for valid having clause."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
            raw_value=5,
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        assert len(issues) == 0

    def test_valid_having_bool_op_or(self, simple_schema):
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
            raw_value=5,
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        bool_op_issues = [i for i in issues if "bool_op" in i.message]
        assert len(bool_op_issues) == 0

    def test_invalid_having_bool_op_raises_issue(self, simple_schema):
        """Leaf params no longer carry bool_op; schema validation ignores removed fields."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        assert not any("bool_op" in i.message for i in issues)

    def test_having_missing_raw_value_emits_error(self, simple_schema):
        """HavingParam with no raw_value and no right_expr emits having_missing_value error."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
            raw_value=None,
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        missing = [i for i in issues if "missing_value" in i.issue_id]
        assert len(missing) == 1
        assert missing[0].severity == "error"

    def test_having_cleared_raw_value_ok_when_param_bound(self, simple_schema):
        """Post-binding ``param_values`` satisfies HAVING when ``raw_value`` is cleared."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="integer",
            raw_value=None,
            param_key="p1",
        )
        issues = validate_having_schema(
            [hp],
            simple_schema,
            {"customers"},
            param_values={"p1": 60},
        )
        missing = [i for i in issues if "missing_value" in i.issue_id]
        assert len(missing) == 0

    def test_having_is_null_no_value_required(self, simple_schema):
        """HavingParam with op='is null' and no raw_value does NOT emit a missing-value error."""
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op="is null",
            value_type="null",
            raw_value=None,
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        missing = [i for i in issues if "missing_value" in i.issue_id]
        assert len(missing) == 0

    def test_invalid_having_operator(self, simple_schema):
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op="like",
            value_type="integer",
            raw_value=1,
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        assert any("Invalid HAVING operator" in i.message for i in issues)

    def test_invalid_having_value_type(self, simple_schema):
        hp = HavingParam(
            left_expr=NormalizedExpr(add_groups=[MulGroup(coefficient=1.0, multiply=["COUNT(customers.id)"])]),
            op=">",
            value_type="not_valid",
            raw_value=1,
        )
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        assert any("Invalid HAVING value_type" in i.message for i in issues)

    def test_having_extract_epoch_on_expression(self, simple_schema):
        left = NormalizedExpr(
            add_groups=[MulGroup(multiply=["COUNT(customers.id)"])],
            scalar_func="extract",
            scalar_func_args=["epoch", "e"],
        )
        hp = HavingParam(left_expr=left, op=">", value_type="integer", raw_value=1)
        issues = validate_having_schema([hp], simple_schema, {"customers"})
        assert any(i.category == "extract_epoch" for i in issues)


class TestValidateNullFilters:
    """Tests for validate_null_filters."""

    def test_valid_null_filter(self):
        """No issues for IS NULL with null value_type."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="is null", value_type="null")
        issues = validate_null_filters([fp])
        assert len(issues) == 0

    def test_null_filter_wrong_value_type(self):
        """Error for IS NULL with non-null value_type."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="is null",
            value_type="string",
        )
        issues = validate_null_filters([fp])
        assert len(issues) == 1
        assert "value_type" in issues[0].message

    def test_non_null_filter_no_issue(self):
        """No issues for non-null filter with any value_type."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        issues = validate_null_filters([fp])
        assert len(issues) == 0

    def test_cte_filter_included(self):
        cte = RuntimeCteStep(
            cte_name="s1",
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t.a"),
                        op="is null",
                        value_type="string",
                    ),
                ]
            ),
        )
        issues = validate_null_filters([], cte_steps=[cte])
        assert len(issues) == 1

    def test_is_null_with_empty_value_type_ok(self):
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="is not null", value_type="")
        assert validate_null_filters([fp]) == []


class TestValidateFilterOpsPerColumn:
    """Tests for validate_where_ops_per_column."""

    def test_valid_op_passes(self, simple_schema):
        """validate_where_ops_per_column no issues for valid op."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_where_ops_per_column([fp], simple_schema)
        assert len(issues) == 0

    def test_invalid_op_errors(self, simple_schema):
        """validate_where_ops_per_column errors for invalid op."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.id"),
            op="like",
            value_type="string",
        )
        issues = validate_where_ops_per_column([fp], simple_schema)
        error_issues = [i for i in issues if i.severity == "error"]
        assert len(error_issues) >= 1

    def test_empty_filters_no_issues(self, simple_schema):
        """validate_where_ops_per_column returns empty for no filters."""
        assert validate_where_ops_per_column([], simple_schema) == []

    def test_cte_column_restricted_ops(self, simple_schema):
        cte = {
            "c": {
                "x": CteOutputColumnMeta(
                    source="computed",
                    valid_where_ops=["="],
                    data_type="varchar",
                    role=ColumnRole.CATEGORICAL.value,
                ),
            },
        }
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("c.x"),
            op="like",
            value_type="string",
        )
        issues = validate_where_ops_per_column([fp], simple_schema, cte_outputs=cte)
        assert any("not valid for CTE column" in i.message for i in issues)

    def test_cte_not_filterable_warns(self, simple_schema):
        cte = {
            "c": {
                "x": CteOutputColumnMeta(source="computed", filterable=False, data_type="varchar"),
            },
        }
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("c.x"),
            op="=",
            value_type="string",
        )
        issues = validate_where_ops_per_column([fp], simple_schema, cte_outputs=cte)
        assert any(i.severity == "warning" for i in issues)

    def test_base_column_not_filterable_warns(self, typed_schema):
        typed_schema.tables["customers"].columns["name"].is_filterable_override = False
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("customers.name"),
            op="=",
            value_type="string",
        )
        issues = validate_where_ops_per_column([fp], typed_schema)
        assert any("not recommended for filtering" in i.message for i in issues)


class TestValidateFilterCol:
    """Tests for _validate_where_col."""

    def test_empty_expr_no_issues(self, simple_schema):
        """No issues for empty column expression."""
        issues = _validate_where_col("", simple_schema, {"customers"}, {}, "main", "left_col", "pk1")
        assert len(issues) == 0

    def test_unqualified_column(self, simple_schema):
        """Error for unqualified column reference."""
        issues = _validate_where_col("name", simple_schema, {"customers"}, {}, "main", "left_col", "pk1")
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "qualified" in issues[0].message

    def test_table_not_allowed(self, simple_schema):
        """Error when table not in allowed set."""
        issues = _validate_where_col("orders.amount", simple_schema, {"customers"}, {}, "main", "left_col", "pk1")
        assert len(issues) == 1
        assert "not in allowed" in issues[0].message

    def test_table_not_in_schema(self, simple_schema):
        """Error when table not in schema."""
        issues = _validate_where_col("products.name", simple_schema, {"products"}, {}, "main", "left_col", "pk1")
        assert len(issues) == 1
        assert "not in schema" in issues[0].message

    def test_column_not_found(self, simple_schema):
        """Error when column not found in table."""
        issues = _validate_where_col(
            "customers.nonexistent",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_col",
            "pk1",
        )
        assert len(issues) == 1
        assert "not in table" in issues[0].message

    def test_valid_column(self, simple_schema):
        """No issues for valid qualified column."""
        issues = _validate_where_col(
            "customers.name",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_col",
            "pk1",
        )
        assert len(issues) == 0

    def test_cte_column_found(self, simple_schema):
        """No issues for valid CTE column reference."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed")}}
        issues = _validate_where_col(
            "cte1.total",
            simple_schema,
            {"customers"},
            cte_outputs,
            "main",
            "left_col",
            "pk1",
        )
        assert len(issues) == 0

    def test_cte_column_not_found(self, simple_schema):
        """Error when column not found in CTE outputs."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed")}}
        issues = _validate_where_col(
            "cte1.missing",
            simple_schema,
            {"customers"},
            cte_outputs,
            "main",
            "left_col",
            "pk1",
        )
        assert len(issues) == 1
        assert "not in CTE" in issues[0].message

    def test_scalar_wrapped_column(self, simple_schema):
        """Validate inner column when wrapped in scalar function."""
        issues = _validate_where_col(
            "ABS(customers.balance)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_col",
            "pk1",
        )
        assert len(issues) == 0

    def test_right_side_param(self, simple_schema):
        """Side parameter reflected in issue IDs."""
        issues = _validate_where_col("name", simple_schema, {"customers"}, {}, "main", "right_col", "pk1")
        assert "right_col" in issues[0].issue_id


class TestValidateHavingAgg:
    """Tests for _validate_having_agg."""

    def test_empty_expr_no_issues(self, simple_schema):
        """No issues for empty aggregation expression."""
        issues = _validate_having_agg("", simple_schema, {"customers"}, {}, "main", "left_agg", "pk1")
        assert len(issues) == 0

    def test_bare_column_invalid_format(self, simple_schema):
        """Error for bare column without aggregation."""
        issues = _validate_having_agg(
            "customers.name",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "Invalid aggregation format" in issues[0].message

    def test_invalid_agg_func(self, simple_schema):
        """Error for invalid aggregation function."""
        issues = _validate_having_agg(
            "MODE(customers.balance)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "Invalid aggregation function" in issues[0].message

    def test_distinct_on_non_count(self, simple_schema):
        """Error for DISTINCT with non-COUNT aggregation."""
        issues = _validate_having_agg(
            "SUM(DISTINCT customers.balance)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "DISTINCT only allowed with COUNT" in issues[0].message

    def test_star_on_non_count(self, simple_schema):
        """Error for star with non-COUNT aggregation."""
        issues = _validate_having_agg("SUM(*)", simple_schema, {"customers"}, {}, "main", "left_agg", "pk1")
        assert len(issues) == 1
        assert "'*' only allowed with COUNT" in issues[0].message

    def test_count_star_valid(self, simple_schema):
        """No issues for COUNT(*)."""
        issues = _validate_having_agg("COUNT(*)", simple_schema, {"customers"}, {}, "main", "left_agg", "pk1")
        assert len(issues) == 0

    def test_unqualified_target(self, simple_schema):
        """Error for unqualified aggregation target."""
        issues = _validate_having_agg("SUM(balance)", simple_schema, {"customers"}, {}, "main", "left_agg", "pk1")
        assert len(issues) == 1
        assert "qualified" in issues[0].message

    def test_table_not_allowed(self, simple_schema):
        """Error when target table not in allowed set."""
        issues = _validate_having_agg(
            "SUM(orders.amount)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "not in allowed" in issues[0].message

    def test_column_not_found(self, simple_schema):
        """Error when column not found in table."""
        issues = _validate_having_agg(
            "SUM(customers.nonexistent)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "not in table" in issues[0].message

    def test_type_mismatch(self, simple_schema):
        """Error for SUM on string column."""
        issues = _validate_having_agg(
            "SUM(customers.name)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "Cannot use SUM" in issues[0].message

    def test_valid_sum(self, simple_schema):
        """No issues for SUM on numeric column."""
        issues = _validate_having_agg(
            "SUM(customers.balance)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 0

    def test_count_on_any_type(self, simple_schema):
        """No issues for COUNT on any column type."""
        issues = _validate_having_agg(
            "COUNT(customers.name)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 0

    def test_cte_column_found(self, simple_schema):
        """No issues for valid CTE column in aggregation."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed")}}
        issues = _validate_having_agg(
            "SUM(cte1.total)",
            simple_schema,
            {"customers"},
            cte_outputs,
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 0

    def test_cte_column_not_found(self, simple_schema):
        """Error when CTE column not found."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed")}}
        issues = _validate_having_agg(
            "SUM(cte1.missing)",
            simple_schema,
            {"customers"},
            cte_outputs,
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "not in CTE" in issues[0].message

    def test_count_distinct_valid(self, simple_schema):
        """No issues for COUNT(DISTINCT col)."""
        issues = _validate_having_agg(
            "COUNT(DISTINCT customers.name)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 0


class TestValidateFilterColEdgeCases:
    """Edge-case tests for _validate_where_col."""

    def test_cte_case_insensitive_match(self, simple_schema):
        """CTE column match is case-insensitive."""
        cte_outputs = {"cte1": {"Total": CteOutputColumnMeta(source="computed")}}
        issues = _validate_where_col(
            "cte1.total",
            simple_schema,
            {"customers"},
            cte_outputs,
            "main",
            "left_col",
            "pk1",
        )
        assert len(issues) == 0

    def test_dotted_table_name(self, simple_schema):
        """Handles table.column split correctly."""
        issues = _validate_where_col("customers.id", simple_schema, {"customers"}, {}, "main", "left_col", "pk1")
        assert len(issues) == 0


class TestValidateHavingAggEdgeCases:
    """Edge-case tests for _validate_having_agg."""

    def test_max_on_string_passes(self, simple_schema):
        """No issues for MAX on string column (string allowed for max)."""
        issues = _validate_having_agg(
            "MAX(customers.name)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 0

    def test_avg_on_string_errors(self, simple_schema):
        """Error for AVG on string column."""
        issues = _validate_having_agg(
            "AVG(customers.name)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "left_agg",
            "pk1",
        )
        assert len(issues) == 1
        assert "Cannot use AVG" in issues[0].message

    def test_right_side_param(self, simple_schema):
        """Side parameter reflected in issues."""
        issues = _validate_having_agg(
            "MODE(customers.balance)",
            simple_schema,
            {"customers"},
            {},
            "main",
            "right_agg",
            "pk1",
        )
        assert "right_agg" in issues[0].issue_id


class TestValidateFilterValueTypeAlignment:
    @pytest.fixture
    def fk_schema(self):
        orders = TableMetadata(
            name="orders",
            columns={
                "order_id": ColumnMetadata(
                    name="order_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=500,
                ),
                "category_id": ColumnMetadata(
                    name="category_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("categories", "category_id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                ),
                "status": ColumnMetadata(
                    name="status",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=5,
                ),
            },
            primary_key=["order_id"],
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["category_id"],
                    dst_table="categories",
                    dst_cols=["category_id"],
                ),
            ],
        )
        categories = TableMetadata(
            name="categories",
            columns={
                "category_id": ColumnMetadata(
                    name="category_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                ),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=10,
                ),
            },
            primary_key=["category_id"],
            foreign_keys=[],
        )
        return SchemaGraph(
            tables={"orders": orders, "categories": categories},
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_string_on_fk_int_warns(self, fk_schema):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.category_id"),
                op="=",
                value_type="string",
                raw_value="Action",
            ),
        ]
        issues = validate_where_value_type_alignment(filters, fk_schema)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_integer_on_fk_int_no_issue(self, fk_schema):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.category_id"),
                op="=",
                value_type="integer",
                raw_value=5,
            ),
        ]
        issues = validate_where_value_type_alignment(filters, fk_schema)
        assert issues == []

    def test_string_on_string_col_no_issue(self, fk_schema):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="=",
                value_type="string",
                raw_value="shipped",
            ),
        ]
        issues = validate_where_value_type_alignment(filters, fk_schema)
        assert issues == []

    def test_no_raw_value_no_issue(self, fk_schema):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.category_id"),
                op="is null",
                value_type="string",
            ),
        ]
        issues = validate_where_value_type_alignment(filters, fk_schema)
        assert issues == []

    def test_unknown_column_no_issue(self, fk_schema):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.unknown_col"),
                op="=",
                value_type="string",
                raw_value="test",
            ),
        ]
        issues = validate_where_value_type_alignment(filters, fk_schema)
        assert issues == []

    def test_enum_on_fk_int_warns(self, fk_schema):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.category_id"),
                op="=",
                value_type="enum",
                raw_value="Action",
            ),
        ]
        issues = validate_where_value_type_alignment(filters, fk_schema)
        assert len(issues) == 1
        assert issues[0].severity == "warning"


class TestValidateNoBetweenOps:
    """Tests for validate_no_between_ops."""

    def test_no_between_no_issues(self):
        """No BETWEEN ops yields empty issue list."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op=">=", value_type="integer")
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
        )
        issues = validate_no_between_ops([fp], [hp])
        assert issues == []

    def test_filter_between_flagged(self):
        """BETWEEN on a filter produces an error issue."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="between",
            value_type="integer",
        )
        issues = validate_no_between_ops([fp], [])
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "where" in issues[0].issue_id.lower()

    def test_having_between_flagged(self):
        """BETWEEN on a having produces an error issue."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op="between",
            value_type="integer",
        )
        issues = validate_no_between_ops([], [hp])
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "having" in issues[0].issue_id.lower()

    def test_both_between_flagged(self):
        """BETWEEN on both filter and having produces two issues."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="between",
            value_type="integer",
        )
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op="between",
            value_type="integer",
        )
        issues = validate_no_between_ops([fp], [hp])
        assert len(issues) == 2

    def test_empty_lists_no_issues(self):
        """Empty filter and having lists yield no issues."""
        issues = validate_no_between_ops([], [])
        assert issues == []

    def test_case_insensitive_between_flagged(self):
        """BETWEEN in any case is flagged."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="BETWEEN",
            value_type="integer",
        )
        issues = validate_no_between_ops([fp], [])
        assert len(issues) == 1
        assert "between" in issues[0].message.lower()


class TestGetColType:
    """Tests for get_col_type."""

    def test_existing_column(self, typed_schema):
        """Return value_type for existing column."""
        result = get_col_type("customers.balance", typed_schema, {})
        assert result == "number"

    def test_unknown_column(self, typed_schema):
        """Return None for unknown column."""
        result = get_col_type("unknown.col", typed_schema, {})
        assert result is None

    def test_unqualified_column(self, typed_schema):
        """Return None for unqualified column."""
        result = get_col_type("balance", typed_schema, {})
        assert result is None


class TestGetColMeta:
    """Tests for get_col_meta."""

    def test_existing_column(self, typed_schema):
        """Return ColumnMetadata for existing column."""
        result = get_col_meta("customers.balance", typed_schema, {})
        assert result is not None
        assert result.name == "balance"

    def test_unknown_column(self, typed_schema):
        """Return None for unknown column."""
        result = get_col_meta("customers.nonexistent", typed_schema, {})
        assert result is None

    def test_unqualified_column(self, typed_schema):
        """Return None for unqualified column."""
        result = get_col_meta("balance", typed_schema, {})
        assert result is None

    def test_unknown_table(self, typed_schema):
        """Return None for unknown table."""
        result = get_col_meta("unknown.col", typed_schema, {})
        assert result is None

    def test_cte_column(self, typed_schema):
        """Return synthetic ColumnMetadata for CTE column."""
        cte_outputs = {
            "cte1": {
                "total": CteOutputColumnMeta(
                    source="computed",
                    data_type="numeric",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                )
            }
        }
        result = get_col_meta("cte1.total", typed_schema, cte_outputs)
        assert result is not None
        assert result.name == "total"

    def test_cte_column_not_found(self, typed_schema):
        """Return None for CTE column not found."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed")}}
        result = get_col_meta("cte1.missing", typed_schema, cte_outputs)
        assert result is None

    def test_scalar_wrapped_column(self, typed_schema):
        """Return metadata for scalar-wrapped column."""
        result = get_col_meta("ABS(customers.balance)", typed_schema, {})
        assert result is not None
        assert result.name == "balance"


class TestIsColNumeric:
    """Tests for is_col_numeric."""

    def test_numeric_column(self, typed_schema):
        """Return True for numeric column."""
        assert is_col_numeric("customers.balance", typed_schema, {}) is True

    def test_integer_column(self, typed_schema):
        """Return True for integer column."""
        assert is_col_numeric("customers.id", typed_schema, {}) is True

    def test_string_column(self, typed_schema):
        """Return False for string column."""
        assert is_col_numeric("customers.name", typed_schema, {}) is False

    def test_unknown_column(self, typed_schema):
        """Return None for unknown column."""
        assert is_col_numeric("unknown.col", typed_schema, {}) is None

    def test_unqualified_column(self, typed_schema):
        """Return None for unqualified column."""
        assert is_col_numeric("balance", typed_schema, {}) is None


class TestIsColArithmeticRole:
    """Tests for is_col_arithmetic_role."""

    def test_numeric_measure(self, typed_schema):
        """Return True for NUMERIC_MEASURE role."""
        assert is_col_arithmetic_role("customers.balance", typed_schema, {}) is True

    def test_identifier_role(self, typed_schema):
        """Return non-True for IDENTIFIER role."""
        result = is_col_arithmetic_role("customers.id", typed_schema, {})
        assert result is not True

    def test_categorical_role(self, typed_schema):
        """Return non-True for CATEGORICAL role."""
        result = is_col_arithmetic_role("customers.name", typed_schema, {})
        assert result is not True

    def test_unknown_column(self, typed_schema):
        """Return None for unknown column."""
        assert is_col_arithmetic_role("unknown.col", typed_schema, {}) is None

    def test_cte_numeric_role(self, typed_schema):
        """Return True for CTE column with NUMERIC_MEASURE role."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed", role=ColumnRole.NUMERIC_MEASURE.value)}}
        assert is_col_arithmetic_role("cte1.total", typed_schema, cte_outputs) is True


class TestExtractAggColSemantic:
    """Tests for extract_agg_col (validation_semantic version)."""

    def test_count_col(self):
        """Extract COUNT aggregation."""
        func, target, has_distinct = extract_agg_col("COUNT(table1.col1)")
        assert func == "count"
        assert target == "table1.col1"
        assert has_distinct is False

    def test_count_distinct(self):
        """Extract COUNT DISTINCT."""
        func, target, has_distinct = extract_agg_col("COUNT(DISTINCT table1.col1)")
        assert func == "count"
        assert has_distinct is True

    def test_empty_string(self):
        """Return None for empty string."""
        func, target, has_distinct = extract_agg_col("")
        assert func is None

    def test_bare_column(self):
        """Return None for bare column."""
        func, target, has_distinct = extract_agg_col("table1.col1")
        assert func is None


class TestGetColTypeEdgeCases:
    """Edge-case tests for get_col_type."""

    def test_cte_column_found(self, typed_schema):
        """Return value_type from CTE output."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed", data_type="numeric")}}
        result = get_col_type("cte1.total", typed_schema, cte_outputs)
        assert result is not None

    def test_cte_column_not_found(self, typed_schema):
        """Return None when CTE column not found."""
        cte_outputs = {"cte1": {"total": CteOutputColumnMeta(source="computed")}}
        result = get_col_type("cte1.missing", typed_schema, cte_outputs)
        assert result is None

    def test_date_column_type(self, typed_schema):
        """Return date value_type for temporal column."""
        result = get_col_type("customers.created_at", typed_schema, {})
        assert result == "date"

    def test_empty_col_expr_returns_none(self, typed_schema):
        """get_col_type returns None for empty column expression."""
        result = get_col_type("", typed_schema, {})
        assert result is None

    def test_scalar_wrapped_column_resolves(self, typed_schema):
        """get_col_type resolves column wrapped in scalar function."""
        result = get_col_type("ABS(customers.balance)", typed_schema, {})
        assert result == "number"


class TestValidateDateDiffUnits:
    """Tests for validate_date_diff_units."""

    def test_valid_date_diff_no_issues(self):
        """No issues for valid date_diff with day unit and numeric amount."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": 7},
        )
        issues = validate_date_diff_units([fp])
        assert len(issues) == 0

    def test_invalid_unit_raises_issue(self):
        """Invalid unit produces an issue."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "invalid_unit", "amount": 7},
        )
        issues = validate_date_diff_units([fp])
        assert len(issues) == 1
        assert "invalid unit" in issues[0].message

    def test_non_date_diff_ignored(self):
        """Filters with other value_types are ignored."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            raw_value="x",
        )
        issues = validate_date_diff_units([fp])
        assert len(issues) == 0


class TestContainsArrayValidator:
    """Tests for validate_contains_array_filters."""

    def test_contains_on_non_array(self, typed_schema):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op="contains",
            value_type="string",
            param_key="p1",
        )
        issues = validate_contains_array_filters([fp], typed_schema, {}, "main")
        assert len(issues) == 1

    def test_contains_on_array_ok(self, typed_schema):
        tags = typed_schema.tables["orders"].columns.get("tags")
        if tags is None:
            from aetherdialect._contracts_base import ColumnRole
            from aetherdialect._contracts_schema import ColumnMetadata

            typed_schema.tables["orders"].columns["tags"] = ColumnMetadata(
                name="tags",
                data_type="text[]",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
                element_type="text",
            )
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.tags"),
            op="contains",
            value_type="string",
            param_key="p1",
        )
        issues = validate_contains_array_filters([fp], typed_schema, {}, "main")
        assert len(issues) == 0


class TestWindowSpecValidator:
    """Tests for validate_window_spec_schema."""

    def test_ranking_without_order_by(self, typed_schema):
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[],
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("order_by" in i.message.lower() for i in issues)


class TestSelectabilityValidator:
    """Tests for validate_selectability."""

    def test_non_selectable_column(self, typed_schema):
        typed_schema.tables["customers"].columns["name"].sensitivity = "hidden"
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        issues = validate_selectability([sc], typed_schema, {}, "main")
        assert len(issues) >= 1

    def test_restricted_column_blocked(self, typed_schema):
        """Restricted columns are blocked in bare SELECT list."""
        typed_schema.tables["customers"].columns["name"].sensitivity = "restricted"
        sc = SelectCol(expr=NormalizedExpr.from_column("customers.name"))
        issues = validate_selectability([sc], typed_schema, {}, "main")
        assert len(issues) == 1
        assert issues[0].category == FailureCategory.ACCESS_POLICY
        assert "not selectable" in issues[0].message

    def test_count_star_allows_non_selectable_absent(self, typed_schema):
        """COUNT(*) has no column refs; no selectability issues."""
        typed_schema.tables["customers"].columns["name"].sensitivity = "hidden"
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["*"], agg_func="count")])
        sc = SelectCol(expr=expr)
        issues = validate_selectability([sc], typed_schema, {}, "main")
        assert issues == []

    def test_count_pk_exempt(self, typed_schema):
        """COUNT on primary key excuses non-selectable flag on that column."""
        typed_schema.tables["customers"].columns["id"].sensitivity = "hidden"
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"], agg_func="count")],
        )
        sc = SelectCol(expr=expr)
        issues = validate_selectability([sc], typed_schema, {}, "main")
        assert issues == []

    def test_sum_on_non_selectable_allowed_in_aggregate(self, typed_schema):
        """SUM (and other aggs) on a non-selectable column are allowed in SELECT."""
        typed_schema.tables["customers"].columns["name"].sensitivity = "hidden"
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.name"], agg_func="sum")],
        )
        sc = SelectCol(expr=expr)
        issues = validate_selectability([sc], typed_schema, {}, "main")
        assert issues == []

    def test_count_distinct_non_selectable_allowed(self, typed_schema):
        typed_schema.tables["customers"].columns["name"].sensitivity = "hidden"
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["DISTINCT customers.name"], agg_func="count")],
        )
        sc = SelectCol(expr=expr)
        issues = validate_selectability([sc], typed_schema, {}, "main")
        assert issues == []

    def test_window_partition_bare_non_selectable_errors(self, typed_schema):
        typed_schema.tables["customers"].columns["name"].sensitivity = "hidden"
        ws = WindowSpec(
            function="rank",
            partition_by=[NormalizedExpr.from_column("customers.name")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="asc")],
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        with registry_render_scope(wr, None):
            issues = validate_selectability([sc], typed_schema, {}, "main")
        assert len(issues) >= 1
        assert "partition" in issues[0].message.lower()

    def test_window_order_by_non_selectable_allowed(self, typed_schema):
        typed_schema.tables["customers"].columns["name"].sensitivity = "hidden"
        ws = WindowSpec(
            function="rank",
            partition_by=[NormalizedExpr.from_column("customers.id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("customers.name"), direction="asc")],
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        with registry_render_scope(wr, None):
            issues = validate_selectability([sc], typed_schema, {}, "main")
        assert issues == []

    def test_cte_output_bare_non_selectable_consumer_blocked(self, typed_schema):
        cte_outputs = {
            "inner_cte": {
                "secret": CteOutputColumnMeta(source="passthrough", sensitivity="hidden"),
            }
        }
        sc = SelectCol(expr=NormalizedExpr.from_column("inner_cte.secret"))
        issues = validate_selectability([sc], typed_schema, cte_outputs, "main")
        assert len(issues) >= 1

    def test_cte_output_non_selectable_wrapped_in_sum_allowed(self, typed_schema):
        cte_outputs = {
            "inner_cte": {
                "secret": CteOutputColumnMeta(source="passthrough", sensitivity="restricted"),
            }
        }
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["inner_cte.secret"], agg_func="sum")],
        )
        sc = SelectCol(expr=expr)
        issues = validate_selectability([sc], typed_schema, cte_outputs, "main")
        assert issues == []

    def test_selectability_exempt_refs_nonempty_for_count_pk(self, typed_schema):
        """Exempt set includes PK ref for COUNT(pk)."""
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["customers.id"], agg_func="count")],
        )
        ex = selectability_exempt_qualified_refs(expr, typed_schema)
        assert "customers.id" in ex


class TestValidateDateWindowUnits:
    """Tests for validate_date_window_units."""

    def test_valid_unit_no_issue(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op=">=",
            value_type="date_window",
            raw_value={"unit": "day", "amount": 1},
        )
        assert validate_date_window_units([fp]) == []

    def test_invalid_unit_issue(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op=">=",
            value_type="date_window",
            raw_value={"unit": "eon", "amount": 1},
        )
        issues = validate_date_window_units([fp])
        assert len(issues) == 1
        assert "invalid unit" in issues[0].message

    def test_non_dict_raw_value_skipped(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op=">=",
            value_type="date_window",
            raw_value="day",
        )
        assert validate_date_window_units([fp]) == []

    def test_none_unit_skipped(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op=">=",
            value_type="date_window",
            raw_value={"amount": 1},
        )
        assert validate_date_window_units([fp]) == []

    def test_cte_step_filters_scanned(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.d"),
            op=">=",
            value_type="date_window",
            raw_value={"unit": "bad", "amount": 1},
        )
        cte = RuntimeCteStep(cte_name="c1", where=predicate_group_from_list([fp]))
        issues = validate_date_window_units([], cte_steps=[cte])
        assert len(issues) == 1


class TestValidateDateDiffUnitsMore:
    """Extra edge cases for validate_date_diff_units."""

    def test_string_amount_parseable_as_int_ok(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.c"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": "3"},
        )
        assert validate_date_diff_units([fp]) == []

    def test_non_numeric_amount_errors(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.c"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "day", "amount": "x"},
        )
        issues = validate_date_diff_units([fp])
        assert any("non-numeric amount" in i.message for i in issues)

    def test_cte_date_diff_invalid_unit(self):
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.c"),
            op=">",
            value_type="date_diff",
            raw_value={"unit": "fortnight", "amount": 1},
        )
        cte = RuntimeCteStep(cte_name="inner", where=predicate_group_from_list([fp]))
        issues = validate_date_diff_units([], cte_steps=[cte])
        assert len(issues) == 1


class TestValidateHavingOpsPerColumn:
    """Tests for validate_having_ops_per_column."""

    def test_cte_restricted_having_op_errors(self, simple_schema):
        cte = {
            "x": {
                "amt": CteOutputColumnMeta(
                    source="aggregation",
                    valid_having_ops=["="],
                    data_type="numeric",
                    aggregatable=True,
                ),
            },
        }
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "x.amt"),
            op=">",
            value_type="integer",
            raw_value=1,
        )
        issues = validate_having_ops_per_column([hp], simple_schema, cte_outputs=cte)
        assert any("not valid for CTE column" in i.message for i in issues)

    def test_base_column_restricted_having_op(self, typed_schema):
        typed_schema.tables["orders"].columns["amount"].valid_having_ops = ["="]
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "orders.amount"),
            op=">",
            value_type="integer",
            raw_value=1,
        )
        issues = validate_having_ops_per_column([hp], typed_schema)
        assert any("not valid for column" in i.message for i in issues)

    def test_empty_having_list(self, simple_schema):
        assert validate_having_ops_per_column([], simple_schema) == []


class TestValidateHavingAggCteBareRef:
    """CTE alias path in _validate_having_agg."""

    def test_bare_cte_aggregation_column_skips_agg_parse(self, simple_schema):
        cte = {"agg_cte": {"total": CteOutputColumnMeta(source="aggregation", data_type="numeric")}}
        issues = _validate_having_agg(
            "agg_cte.total",
            simple_schema,
            {"customers"},
            cte,
            "main",
            "left_agg",
            "h1",
        )
        assert issues == []


class TestValidateCaseWhenSchema:
    """Tests for validate_case_when_schema."""

    def test_case_branch_unknown_table_in_result(self, simple_schema):
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op=">",
                        value_type="integer",
                        raw_value=0,
                    ),
                    result=NormalizedExpr.from_column("unknown_tbl.col"),
                ),
            ],
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, {}, "main", case_registry=cr)
        assert any("unknown table" in i.message for i in issues)

    def test_case_branch_valid_no_issues(self, simple_schema):
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op=">",
                        value_type="integer",
                        raw_value=0,
                    ),
                    result=NormalizedExpr.from_column("customers.name"),
                ),
            ],
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, {}, "main", case_registry=cr)
        assert issues == []

    def test_case_else_unknown_column(self, simple_schema):
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op=">",
                        value_type="integer",
                        raw_value=0,
                    ),
                    result=NormalizedExpr.from_column("customers.name"),
                ),
            ],
            else_result=NormalizedExpr.from_column("customers.nope"),
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, {}, "main", case_registry=cr)
        assert any("CASE ELSE" in i.message and "unknown column" in i.message for i in issues)

    def test_case_result_cte_column_allowed(self, simple_schema):
        """CASE THEN/ELSE may reference columns declared on prior CTE outputs (``cte_outputs``)."""
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op=">",
                        value_type="integer",
                        raw_value=0,
                    ),
                    result=NormalizedExpr.from_column("rollup.total"),
                ),
            ],
            else_result=NormalizedExpr.from_column("rollup.total"),
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        cte_outputs = {"rollup": {"total": CteOutputColumnMeta(source="aggregation", data_type="numeric")}}
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, cte_outputs, "main", case_registry=cr)
        assert issues == []

    def test_case_branch_filter_rejects_unknown_table(self, simple_schema):
        """Branch ``WHEN`` predicates use ``validate_filters_schema`` (same table allowlist as main/CTE)."""
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.amount"),
                        op=">",
                        value_type="integer",
                        raw_value=0,
                    ),
                    result=NormalizedExpr.from_column("customers.name"),
                ),
            ],
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, {}, "main", case_registry=cr)

        assert any(i.category == "where_validity" and "not in allowed tables" in i.message for i in issues)

    def test_case_else_unknown_table(self, simple_schema):
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op=">",
                        value_type="integer",
                        raw_value=0,
                    ),
                    result=NormalizedExpr.from_column("customers.name"),
                ),
            ],
            else_result=NormalizedExpr.from_column("ghost_tbl.x"),
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, {}, "main", case_registry=cr)
        assert any("CASE ELSE" in i.message and "unknown table" in i.message for i in issues)

    def test_case_two_branches_no_else_valid(self, simple_schema):
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op=">",
                        value_type="integer",
                        raw_value=10,
                    ),
                    result=NormalizedExpr.from_column("customers.name"),
                ),
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.id"),
                        op="<=",
                        value_type="integer",
                        raw_value=10,
                    ),
                    result=NormalizedExpr.from_column("customers.email"),
                ),
            ],
        )
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        cr = [CaseRegistryStep(registry_id="c01", case_when=cw)]
        issues = validate_case_when_schema([sc], simple_schema, {"customers"}, {}, "main", case_registry=cr)
        assert issues == []


class TestValidateFilterValueTypeAlignmentCte:
    def test_string_on_cte_numeric_warns(self, simple_schema):
        cte = {
            "v": {
                "n": CteOutputColumnMeta(
                    source="computed",
                    data_type="integer",
                    value_type="integer",
                ),
            },
        }
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("v.n"),
            op="=",
            value_type="string",
            raw_value="x",
        )
        issues = validate_where_value_type_alignment([fp], simple_schema, cte_outputs=cte)
        assert len(issues) == 1
        assert issues[0].severity == "warning"


class TestContainsArrayFiltersMore:
    def test_skips_when_not_exactly_one_column_ref(self, simple_schema):
        fp = WhereParam(
            left_expr=NormalizedExpr(
                add_groups=[MulGroup(multiply=["customers.id", "customers.name"])],
            ),
            op="contains",
            value_type="string",
        )
        issues = validate_contains_array_filters([fp], simple_schema, {}, "main")
        assert issues == []


class TestValidateWindowSpecSchemaMore:
    """More window validation branches."""

    def test_invalid_window_function(self, typed_schema):
        ws = WindowSpec(
            function="not_a_window_fn",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("invalid window function" in i.message for i in issues)

    def test_lag_missing_order_and_arg(self, typed_schema):
        ws = WindowSpec(
            function="lag",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[],
            argument=None,
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("order_by" in i.message.lower() for i in issues)
        assert any("argument" in i.message.lower() for i in issues)

    def test_lag_allows_non_numeric_offset_argument(self, typed_schema):
        """LEAD/LAG offset may reference non-numeric columns; schema does not enforce numeric-only."""
        ws = WindowSpec(
            function="lag",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
            argument=NormalizedExpr.from_column("customers.name"),
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert not any("window_offset_non_numeric" in i.issue_id for i in issues)

    def test_window_sum_non_numeric_argument(self, typed_schema):
        ws = WindowSpec(
            function="sum",
            partition_by=[NormalizedExpr.from_column("orders.customer_id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
            argument=NormalizedExpr.from_column("customers.name"),
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("numeric" in i.message.lower() for i in issues)

    def test_partition_non_groupable_column(self, typed_schema):
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("customers.description")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
        )
        wr = [
            WindowRegistryStep(
                registry_id="w01",
                window_spec=ws,
            )
        ]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("not groupable" in i.message.lower() for i in issues)


class TestExtendedWindowFunctionValidation:
    """Argument validation for ntile, percent_rank, cume_dist, and nth_value."""

    def test_ntile_requires_positive_numeric_argument(self, typed_schema):
        ws = WindowSpec(
            function="ntile",
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
            numeric_argument=None,
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("numeric_argument" in i.message for i in issues)

    def test_ntile_rejects_non_positive_numeric_argument(self, typed_schema):
        ws = WindowSpec(
            function="ntile",
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
            numeric_argument=0,
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("numeric_argument" in i.message for i in issues)

    def test_percent_rank_rejects_column_argument(self, typed_schema):
        ws = WindowSpec(
            function="percent_rank",
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
            argument=NormalizedExpr.from_column("orders.amount"),
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("must not carry an argument" in i.message for i in issues)

    def test_nth_value_requires_positive_numeric_argument(self, typed_schema):
        ws = WindowSpec(
            function="nth_value",
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
            argument=NormalizedExpr.from_column("orders.amount"),
            numeric_argument=-1,
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("numeric_argument" in i.message for i in issues)


class TestValidateGroupByColsEdgeCases:
    def test_empty_expr_unqualified(self, simple_schema):
        issues = validate_group_by_cols_schema([NormalizedExpr()], simple_schema, {"customers"})
        assert any("qualified" in i.message for i in issues)


class TestValidateWindowPartitionGroupByAlignment:
    def test_grouped_scope_missing_partition_col_errors(self):
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("tbl_a.col_k")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("tbl_a.col_a"), direction="asc")],
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        issues = validate_window_partition_group_by_alignment(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("tbl_a.col_a")],
            window_registry=wr,
            context="main query",
        )
        assert len(issues) == 1
        assert "window_partition_column_missing" in issues[0].issue_id

    def test_row_level_outer_scope_no_error(self):
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("tbl_a.col_k")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("tbl_a.col_a"), direction="asc")],
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        issues = validate_window_partition_group_by_alignment(
            grain="row_level",
            group_by_cols=[],
            window_registry=wr,
            context="main query",
        )
        assert issues == []

    def test_partition_col_present_in_group_by_ok(self):
        ws = WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("tbl_a.col_k")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("tbl_a.col_a"), direction="asc")],
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        issues = validate_window_partition_group_by_alignment(
            grain="grouped",
            group_by_cols=[
                NormalizedExpr.from_column("tbl_a.col_k"),
                NormalizedExpr.from_column("tbl_a.col_a"),
            ],
            window_registry=wr,
            context="main query",
        )
        assert issues == []


class TestValidateRedundantExtractYearLiterals:
    def test_bare_year_equality_with_extract_errors(self):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr(
                    add_groups=[MulGroup(multiply=[NormalizedExpr.from_column("tbl_a.date_a")])],
                    scalar_func="extract",
                    scalar_func_args=["year", "tbl_a.date_a"],
                ),
                op="=",
                raw_value="2026",
                value_type="integer",
            ),
            WhereParam(
                left_expr=NormalizedExpr.from_column("tbl_a.date_a"),
                op="=",
                raw_value="2026",
                value_type="integer",
            ),
        ]
        issues = validate_redundant_extract_year_column_literals(filters, [], "main")
        assert len(issues) == 1
        assert "redundant" in issues[0].message.lower()

    def test_bare_year_gt_with_extract_errors(self):
        filters = [
            WhereParam(
                left_expr=NormalizedExpr(
                    add_groups=[MulGroup(multiply=[NormalizedExpr.from_column("tbl_a.date_a")])],
                    scalar_func="extract",
                    scalar_func_args=["year", "tbl_a.date_a"],
                ),
                op="=",
                raw_value="2026",
                value_type="integer",
            ),
            WhereParam(
                left_expr=NormalizedExpr.from_column("tbl_a.date_a"),
                op=">=",
                raw_value="2025",
                value_type="integer",
            ),
        ]
        issues = validate_redundant_extract_year_column_literals(filters, [], "main")
        assert len(issues) == 1


def _deep_or_predicate_tree(depth: int):
    from aetherdialect._contracts_base import PredicateGroup, WhereParam

    def _leaf(col: str) -> WhereParam:
        return WhereParam(left_expr=NormalizedExpr.from_column(col), op="=", raw_value="x", param_key="p1")

    if depth <= 1:
        return PredicateGroup(op="and", predicates=(_leaf("t.a"), _leaf("t.b")))

    left = _deep_or_predicate_tree(depth - 1)
    right = PredicateGroup(op="and", predicates=(_leaf("t.c"),))
    return PredicateGroup(op="or", groups=(left, right))


@pytest.mark.fast
def test_where_predicate_depth_four_reports_max_nesting() -> None:
    from aetherdialect._constants import MAX_PREDICATE_NESTING_DEPTH
    from aetherdialect._contracts_base import coerce_predicate_group
    from aetherdialect._validation_schema import validate_predicate_nesting_depth

    nested = _deep_or_predicate_tree(MAX_PREDICATE_NESTING_DEPTH + 1)
    assert nested.depth() > MAX_PREDICATE_NESTING_DEPTH
    issues = validate_predicate_nesting_depth(nested, None)
    assert any(i.issue_id == "where_predicate_nesting_depth" for i in issues)
    assert any(i.severity == "error" for i in issues if i.issue_id == "where_predicate_nesting_depth")
    assert any(f"MAX_PREDICATE_NESTING_DEPTH={MAX_PREDICATE_NESTING_DEPTH}" in i.message for i in issues)
    coerced = coerce_predicate_group(nested)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH


@pytest.mark.fast
def test_having_predicate_depth_four_reports_max_nesting() -> None:
    from aetherdialect._constants import MAX_PREDICATE_NESTING_DEPTH
    from aetherdialect._contracts_base import HavingParam, PredicateGroup, coerce_predicate_group
    from aetherdialect._validation_schema import validate_predicate_nesting_depth

    def _having_leaf() -> HavingParam:
        return HavingParam(left_expr=NormalizedExpr.from_agg("count", "*"), op=">", raw_value=1)

    def _deep_having_tree(depth: int) -> PredicateGroup:
        if depth <= 1:
            return PredicateGroup(op="and", predicates=(_having_leaf(), _having_leaf()))
        left = _deep_having_tree(depth - 1)
        right = PredicateGroup(op="and", predicates=(_having_leaf(),))
        return PredicateGroup(op="or", groups=(left, right))

    nested = _deep_having_tree(MAX_PREDICATE_NESTING_DEPTH + 1)
    assert nested.depth() > MAX_PREDICATE_NESTING_DEPTH
    issues = validate_predicate_nesting_depth(None, nested)
    assert any(i.issue_id == "having_predicate_nesting_depth" for i in issues)
    assert any(i.severity == "error" for i in issues if i.issue_id == "having_predicate_nesting_depth")
    assert any(f"MAX_PREDICATE_NESTING_DEPTH={MAX_PREDICATE_NESTING_DEPTH}" in i.message for i in issues)
    coerced = coerce_predicate_group(nested)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH
