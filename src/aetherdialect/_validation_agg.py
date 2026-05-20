"""Aggregation and scalar typing rules for SELECT, ORDER BY, and HAVING, plus numeric and role heuristics."""

from __future__ import annotations

import re

from ._config import (
    NUMERIC_RESULT_AGGS,
    NUMERIC_RESULT_SCALARS,
    SCALAR_FUNCTIONS_NUMERIC,
    SCALAR_FUNCTIONS_STRING,
    SCALAR_FUNCTIONS_TEMPORAL,
    VALID_AGGREGATION_FUNCTIONS,
)
from ._contracts_base import (
    ColumnRole,
    CteOutputColumnMeta,
    FailureCategory,
    IntentIssue,
    SchemaGraph,
    make_intent_issue,
)
from ._contracts_core import HavingParam, NormalizedExpr, OrderByCol, SelectCol
from ._core_utils import debug
from ._validation_schema import (
    extract_agg_col,
    extract_col_from_scalar_wrapper,
    extract_functions_from_term,
    get_col_type,
    is_col_numeric,
)


def validate_having_agg_per_role(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that HAVING aggregation functions are valid for each column's role.

    Args:

        having_param: List of `HavingParam` instances to validate.

        schema: The `SchemaGraph`.

        cte_outputs: Dict of CTE name to output column metadata.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:
        agg_expr = hp.left_expr.primary_term
        if not agg_expr:
            continue
        result = extract_agg_col(agg_expr)
        if len(result) != 3:
            continue
        func, actual_target, _ = result
        if not func or actual_target == "*":
            continue
        if "." not in actual_target:
            continue
        table_name, col_name = actual_target.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                if cte_meta.valid_aggregations and func not in cte_meta.valid_aggregations:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"having_agg_invalid_for_cte_{context}_{actual_target}_{func}",
                            category=FailureCategory.HAVING_VALIDITY,
                            severity="error",
                            message=f"Aggregation '{func.upper()}' not valid for CTE column '{actual_target}' (role={cte_meta.role}) in HAVING for {context}. Valid: {sorted(cte_meta.valid_aggregations)}",
                            context={
                                "column": actual_target,
                                "function": func,
                                "role": cte_meta.role,
                                "valid_aggs": sorted(cte_meta.valid_aggregations),
                                "location": context,
                            },
                        )
                    )
            continue
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        valid_aggs = col_meta.get_valid_aggregations()
        if func not in valid_aggs:
            issues.append(
                make_intent_issue(
                    issue_id=f"having_agg_invalid_for_role_{context}_{actual_target}_{func}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"Aggregation '{func.upper()}' not valid for column '{actual_target}' (role={col_meta.role}) in HAVING for {context}. Valid: {sorted(valid_aggs)}",
                    context={
                        "column": actual_target,
                        "function": func,
                        "role": col_meta.role,
                        "valid_aggs": sorted(valid_aggs),
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_having_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_select_agg_per_role(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that SELECT aggregation functions are valid for each column's role.

    Args:

        select_cols: List of `SelectCol` instances to validate.

        schema: The `SchemaGraph`.

        cte_outputs: Dict of CTE name to output column metadata.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []
    if not select_cols:
        return []
    cte_outputs = cte_outputs or {}
    for sc in select_cols:
        _, agg_func = extract_functions_from_term(sc.expr.primary_term)
        if not agg_func:
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                if cte_meta.valid_aggregations:
                    func_lower = agg_func.lower()
                    if func_lower not in cte_meta.valid_aggregations:
                        issues.append(
                            make_intent_issue(
                                issue_id=f"select_agg_invalid_for_cte_{context}_{actual_col}_{agg_func}",
                                category=FailureCategory.AGGREGATION_VALIDITY,
                                severity="error",
                                message=f"Aggregation '{agg_func.upper()}' not valid for CTE column '{actual_col}' (role={cte_meta.role}) in {context}. Valid: {sorted(cte_meta.valid_aggregations)}",
                                context={
                                    "column": actual_col,
                                    "function": agg_func,
                                    "role": cte_meta.role,
                                    "valid_aggs": sorted(cte_meta.valid_aggregations),
                                    "location": context,
                                },
                            )
                        )
            continue
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        valid_aggs = col_meta.get_valid_aggregations()
        func_lower = agg_func.lower()
        if func_lower not in valid_aggs:
            issues.append(
                make_intent_issue(
                    issue_id=f"select_agg_invalid_for_role_{context}_{actual_col}_{agg_func}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"Aggregation '{agg_func.upper()}' not valid for column '{actual_col}' (role={col_meta.role}) in {context}. Valid: {sorted(valid_aggs)}",
                    context={
                        "column": actual_col,
                        "function": agg_func,
                        "role": col_meta.role,
                        "valid_aggs": sorted(valid_aggs),
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_select_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_select_agg_semantics(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that SELECT aggregation functions are semantically appropriate for column types.

    Errors for SUM/AVG on non-numeric columns; warns for MIN/MAX on FREE_TEXT columns.

    Args:

        select_cols: List of `SelectCol` instances to validate.

        schema: The `SchemaGraph`.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []
    if not select_cols:
        return []
    numeric_aggs = {"sum", "avg"}
    for sc in select_cols:
        _, agg_func = extract_functions_from_term(sc.expr.primary_term)
        if not agg_func:
            continue
        func_lower = agg_func
        if func_lower not in numeric_aggs and func_lower not in {"min", "max"}:
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        col_meta = schema.tables[table_name].columns.get(col_name)
        if not col_meta:
            continue
        vt = col_meta.value_type
        numeric = vt in ("integer", "number")
        temporal = vt == "date"
        if func_lower in numeric_aggs and not numeric:
            issues.append(
                make_intent_issue(
                    issue_id=f"invalid_agg_semantics_{func_lower}_{table_name}_{col_name}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="error",
                    message=f"Cannot {func_lower.upper()} on {actual_col} (type={col_meta.data_type}): {func_lower.upper()} requires numeric column",
                    context={
                        "aggregation": func_lower,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_schema.validate_select_agg_semantics] invalid {func_lower.upper()} on {actual_col}")
        elif func_lower in {"min", "max"} and not numeric and not temporal:
            col_role = col_meta.role if col_meta.role else None
            if col_role == ColumnRole.FREE_TEXT.value:
                issues.append(
                    make_intent_issue(
                        issue_id=f"questionable_agg_{func_lower}_{table_name}_{col_name}",
                        category=FailureCategory.AGGREGATION_SEMANTICS,
                        severity="warning",
                        message=f"Questionable {func_lower.upper()} on {actual_col} (type={col_meta.data_type}): {func_lower.upper()} on free text is semantically meaningless",
                        context={
                            "aggregation": func_lower,
                            "column": actual_col,
                            "data_type": col_meta.data_type,
                            "location": context,
                        },
                    )
                )
                debug(
                    f"[validation_schema.validate_select_agg_semantics] questionable {func_lower.upper()} on {actual_col}"
                )
    if issues:
        debug(f"[validation_schema.validate_select_agg_semantics] found {len(issues)} semantic issues")
    else:
        debug("[validation_schema.validate_select_agg_semantics] no semantic issues")
    return issues


def validate_order_by_agg_per_role(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that ORDER BY aggregation functions are valid for each column's role.

    Args:

        order_by_cols: List of `OrderByCol` instances to validate.

        schema: The `SchemaGraph`.

        cte_outputs: Dict of CTE name to output column metadata.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []
    if not order_by_cols:
        return []
    cte_outputs = cte_outputs or {}
    for obc in order_by_cols:
        _, agg_func = extract_functions_from_term(obc.expr.primary_term)
        if not agg_func:
            continue
        col_expr = obc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                if cte_meta.valid_aggregations:
                    func_lower = agg_func.lower()
                    if func_lower not in cte_meta.valid_aggregations:
                        issues.append(
                            make_intent_issue(
                                issue_id=f"order_by_agg_invalid_for_cte_{context}_{actual_col}_{agg_func}",
                                category=FailureCategory.AGGREGATION_VALIDITY,
                                severity="error",
                                message=f"Aggregation '{agg_func.upper()}' not valid for CTE column '{actual_col}' (role={cte_meta.role}) in order_by for {context}. Valid: {sorted(cte_meta.valid_aggregations)}",
                                context={
                                    "column": actual_col,
                                    "function": agg_func,
                                    "role": cte_meta.role,
                                    "valid_aggs": sorted(cte_meta.valid_aggregations),
                                    "location": context,
                                },
                            )
                        )
            continue
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        valid_aggs = col_meta.get_valid_aggregations()
        func_lower = agg_func.lower()
        if func_lower not in valid_aggs:
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_agg_invalid_for_role_{context}_{actual_col}_{agg_func}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"Aggregation '{agg_func.upper()}' not valid for column '{actual_col}' (role={col_meta.role}) in order_by for {context}. Valid: {sorted(valid_aggs)}",
                    context={
                        "column": actual_col,
                        "function": agg_func,
                        "role": col_meta.role,
                        "valid_aggs": sorted(valid_aggs),
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_order_by_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_order_by_agg_semantics(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that ORDER BY aggregation functions are semantically appropriate for column types.

    Errors for SUM/AVG on non-numeric columns; warns for MIN/MAX on FREE_TEXT columns.

    Args:

        order_by_cols: List of `OrderByCol` instances to validate.

        schema: The `SchemaGraph`.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []
    if not order_by_cols:
        return []
    numeric_aggs = {"sum", "avg"}
    for obc in order_by_cols:
        _, agg_func = extract_functions_from_term(obc.expr.primary_term)
        if not agg_func:
            continue
        func_lower = agg_func
        if func_lower not in numeric_aggs and func_lower not in {"min", "max"}:
            continue
        col_expr = obc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        col_meta = schema.tables[table_name].columns.get(col_name)
        if not col_meta:
            continue
        vt = col_meta.value_type
        numeric = vt in ("integer", "number")
        temporal = vt == "date"
        if func_lower in numeric_aggs and not numeric:
            issues.append(
                make_intent_issue(
                    issue_id=f"invalid_order_by_agg_semantics_{func_lower}_{table_name}_{col_name}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="error",
                    message=f"Cannot {func_lower.upper()} on {actual_col} (type={col_meta.data_type}) in ORDER BY: {func_lower.upper()} requires numeric column",
                    context={
                        "aggregation": func_lower,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_schema.validate_order_by_agg_semantics] invalid {func_lower.upper()} on {actual_col}")
        elif func_lower in {"min", "max"} and not numeric and not temporal:
            col_role = col_meta.role if col_meta.role else None
            if col_role == ColumnRole.FREE_TEXT.value:
                issues.append(
                    make_intent_issue(
                        issue_id=f"questionable_order_by_agg_{func_lower}_{table_name}_{col_name}",
                        category=FailureCategory.AGGREGATION_SEMANTICS,
                        severity="warning",
                        message=f"Questionable {func_lower.upper()} on {actual_col} (type={col_meta.data_type}) in ORDER BY: {func_lower.upper()} on free text is semantically meaningless",
                        context={
                            "aggregation": func_lower,
                            "column": actual_col,
                            "data_type": col_meta.data_type,
                            "location": context,
                        },
                    )
                )
                debug(
                    f"[validation_schema.validate_order_by_agg_semantics] questionable {func_lower.upper()} on {actual_col}"
                )
    if issues:
        debug(f"[validation_schema.validate_order_by_agg_semantics] found {len(issues)} semantic issues")
    else:
        debug("[validation_schema.validate_order_by_agg_semantics] no semantic issues")
    return issues


def validate_scalar_func_type_semantics(
    select_cols: list[SelectCol],
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that scalar functions are appropriate for column types and aggregation context.

    Errors when a non-aggregate-compatible scalar wraps an aggregation, or when a type-specific scalar (string, numeric, temporal) is applied to the wrong column type.

    Args:

        select_cols: List of `SelectCol` instances to validate.

        order_by_cols: List of `OrderByCol` instances to validate.

        schema: The `SchemaGraph`.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []

    def check_scalar_semantics(
        scalar_func: str, col_expr: str, agg_func: str | None, location: str
    ) -> list[IntentIssue]:
        """
        Check scalar function semantics for one term.

        Args:

            scalar_func: Outer scalar function name.

            col_expr: Column expression string.

            agg_func: Inner aggregation function if present.

            location: Location label for issues.

        Returns:

            List of `IntentIssue` objects for violations.
        """
        inner_issues = []
        func_lower = scalar_func.lower()
        if agg_func and func_lower not in SCALAR_FUNCTIONS_NUMERIC:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_on_agg_invalid_{location}_{func_lower}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' cannot wrap aggregation '{agg_func.upper()}' in {location}. Only {sorted(SCALAR_FUNCTIONS_NUMERIC)} allowed on aggregates",
                    context={
                        "scalar": scalar_func,
                        "aggregation": agg_func,
                        "location": location,
                        "allowed": sorted(SCALAR_FUNCTIONS_NUMERIC),
                    },
                )
            )
            return inner_issues
        if agg_func:
            return inner_issues
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if not actual_col or "." not in actual_col or actual_col == "*":
            return inner_issues
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            return inner_issues
        col_meta = schema.tables[table_name].columns.get(col_name) or schema.tables[table_name].columns.get(
            col_name.lower()
        )
        if not col_meta:
            return inner_issues
        vt = col_meta.value_type
        string = vt == "string"
        numeric = vt in ("integer", "number")
        temporal = vt == "date"
        if func_lower in SCALAR_FUNCTIONS_STRING and not string:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_type_mismatch_{location}_{func_lower}_{actual_col}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' requires string column, got '{actual_col}' (type={col_meta.data_type}) in {location}",
                    context={
                        "scalar": scalar_func,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "expected_type": "string",
                        "location": location,
                    },
                )
            )
        elif func_lower in SCALAR_FUNCTIONS_NUMERIC and not numeric:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_type_mismatch_{location}_{func_lower}_{actual_col}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' requires numeric column, got '{actual_col}' (type={col_meta.data_type}) in {location}",
                    context={
                        "scalar": scalar_func,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "expected_type": "numeric",
                        "location": location,
                    },
                )
            )
        elif func_lower in SCALAR_FUNCTIONS_TEMPORAL and not temporal:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_type_mismatch_{location}_{func_lower}_{actual_col}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' requires temporal column, got '{actual_col}' (type={col_meta.data_type}) in {location}",
                    context={
                        "scalar": scalar_func,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "expected_type": "date/timestamp",
                        "location": location,
                    },
                )
            )
        return inner_issues

    for idx, sc in enumerate(select_cols or []):
        sc_scalar, sc_agg = extract_functions_from_term(sc.expr.primary_term)
        if sc_scalar:
            issues.extend(check_scalar_semantics(sc_scalar, sc.expr.primary_column, sc_agg, f"select_cols[{idx}]"))
    for idx, obc in enumerate(order_by_cols or []):
        obc_scalar, obc_agg = extract_functions_from_term(obc.expr.primary_term)
        if obc_scalar:
            issues.extend(
                check_scalar_semantics(
                    obc_scalar,
                    obc.expr.primary_column,
                    obc_agg,
                    f"order_by_cols[{idx}]",
                )
            )
    if issues:
        debug(
            f"[validation_schema.validate_scalar_func_type_semantics] found {len(issues)} semantic issues in {context}"
        )
    else:
        debug(f"[validation_schema.validate_scalar_func_type_semantics] no semantic issues in {context}")
    return issues


def validate_column_types(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that operations match their column types (heuristic checks).

    Warns for numeric aggregations on text columns, date operations on non-date columns, and string operations on numeric columns.

    Args:

        select_cols: List of `SelectCol` instances to inspect.

        schema: The `SchemaGraph`.

        context: Label used in issue IDs and messages.

    Returns:

        List of `IntentIssue` objects.
    """
    issues = []
    debug("[validation_schema.validate_column_types] checking type consistency")
    numeric_aggs = {"sum", "avg", "average", "total", "mean"}
    date_ops = {"latest", "earliest", "recent", "oldest", "newest", "before", "after"}
    string_ops = {"contains", "starts", "ends", "like", "match"}
    for sc in select_cols:
        _, agg_func = extract_functions_from_term(sc.expr.primary_term)
        if not agg_func:
            continue
        func_lower = agg_func
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        vt = col_meta.value_type
        if vt:
            numeric = vt in ("integer", "number")
            date = vt == "date"
            text = vt == "string"
        else:
            numeric = any(
                hint in col_name.lower()
                for hint in [
                    "amount",
                    "price",
                    "total",
                    "count",
                    "qty",
                    "quantity",
                    "rate",
                    "cost",
                    "num",
                ]
            )
            date = any(
                hint in col_name.lower()
                for hint in [
                    "date",
                    "time",
                    "created",
                    "updated",
                    "at",
                    "day",
                    "year",
                    "month",
                ]
            )
            text = any(
                hint in col_name.lower() for hint in ["name", "title", "description", "email", "address", "text"]
            )
        if func_lower in numeric_aggs and text and not numeric:
            issues.append(
                make_intent_issue(
                    issue_id=f"numeric_on_text_{table_name}_{col_name}",
                    category=FailureCategory.TYPE_MISMATCH,
                    severity="warning",
                    message=f"Attempting numeric aggregation ({func_lower}) on text column '{col_name}' (type: {col_meta.data_type})",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "type": col_meta.data_type,
                        "agg": func_lower,
                        "location": context,
                    },
                )
            )
            debug("[validation_schema.validate_column_types] type_mismatch: numeric_on_text")
        if func_lower in date_ops and not date:
            issues.append(
                make_intent_issue(
                    issue_id=f"date_on_non_date_{table_name}_{col_name}",
                    category=FailureCategory.TYPE_MISMATCH,
                    severity="warning",
                    message=f"Attempting date operation ({func_lower}) on non-date column '{col_name}' (type: {col_meta.data_type})",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "type": col_meta.data_type,
                        "op": func_lower,
                        "location": context,
                    },
                )
            )
            debug("[validation_schema.validate_column_types] type_mismatch: date_on_non_date")
        if func_lower in string_ops and numeric and "_id" not in col_name.lower():
            issues.append(
                make_intent_issue(
                    issue_id=f"string_on_numeric_{table_name}_{col_name}",
                    category=FailureCategory.TYPE_MISMATCH,
                    severity="warning",
                    message=f"Attempting string operation ({func_lower}) on numeric column '{col_name}' (type: {col_meta.data_type})",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "type": col_meta.data_type,
                        "op": func_lower,
                        "location": context,
                    },
                )
            )
            debug("[validation_schema.validate_column_types] TYPE MISMATCH: string op on numeric column")
    if issues:
        debug(f"[validation_schema.validate_column_types] FAILED with {len(issues)} issues")
    else:
        debug("[validation_schema.validate_column_types] PASSED")
    return issues


def expr_has_arithmetic(expr: NormalizedExpr) -> bool:
    """
    Return `True` if a `NormalizedExpr` contains arithmetic operations.

    Args:

        expr: The normalised expression to inspect.

    Returns:

        `True` when the expression has multiple groups, add/sub constant values, a non-unit coefficient, division, or multiple multiply terms; `False` otherwise.
    """
    if len(expr.add_groups) + len(expr.sub_groups) > 1:
        return True
    if expr.add_values or expr.sub_values:
        return True
    for g in expr.add_groups + expr.sub_groups:
        if (g.scalar_func or "").lower() == "concat":
            continue
        if g.coefficient != 1.0 or g.divide or len(g.multiply) > 1:
            return True
    return False


def strip_function_wrappers(term: str) -> str:
    """
    Strip all nested function call wrappers to expose the innermost column reference.

    Args:

        term: SQL term string, possibly with wrappers such as `UPPER(table.col)` or `ABS(SUM(table.col))`.

    Returns:

        Bare column reference after removing wrapping functions.
    """
    while "(" in term:
        start = term.index("(")
        end = term.rindex(")")
        inner = term[start + 1 : end].strip()
        if inner.upper().startswith("DISTINCT "):
            inner = inner[9:].strip()
        term = inner
    return term


def term_result_is_numeric(term: str) -> bool:
    """
    Return `True` if function wrappers guarantee a numeric result regardless of column type.

    Args:

        term: SQL term string, possibly with nested function calls.

    Returns:

        `True` when the outermost function is a known numeric-result aggregation (`COUNT`, `SUM`, `AVG`) or numeric scalar (`ABS`, `ROUND`, etc.); `False` otherwise.
    """
    remaining = term.strip()
    while True:
        match = re.match(r"^\s*(\w+)\s*\(", remaining)
        if not match:
            return False
        func = match.group(1).lower()
        inner_start = remaining.index("(") + 1
        inner_end = remaining.rindex(")")
        inner = remaining[inner_start:inner_end].strip()
        if inner.upper().startswith("DISTINCT "):
            inner = inner[9:].strip()
        if func in NUMERIC_RESULT_AGGS or func in NUMERIC_RESULT_SCALARS:
            if not inner:
                return False
            return True
        remaining = inner


def expr_result_is_numeric(
    expr: NormalizedExpr,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> bool | None:
    """
    Return whether the result of a `NormalizedExpr` is numeric.

    Args:

        expr: The normalised expression to inspect.

        schema: Schema graph for resolving column types.

        cte_outputs: Map of CTE name to column output metadata.

    Returns:

        `True` if the expression provably produces a numeric result (aggregation, scalar, arithmetic, or numeric column). `False` if the primary column is known non-numeric. `None` if the result type cannot be determined.
    """
    if expr.agg_func and expr.agg_func in NUMERIC_RESULT_AGGS:
        return True
    if expr.scalar_func and expr.scalar_func in NUMERIC_RESULT_SCALARS:
        return True
    if expr.inner_scalar_func and expr.inner_scalar_func in NUMERIC_RESULT_SCALARS:
        return True
    if expr_has_arithmetic(expr):
        return True
    if expr.add_values or expr.sub_values:
        return True
    for g in expr.add_groups + expr.sub_groups:
        if g.agg_func and g.agg_func in NUMERIC_RESULT_AGGS:
            return True
        if g.scalar_func and g.scalar_func in NUMERIC_RESULT_SCALARS:
            return True
        if g.inner_scalar_func and g.inner_scalar_func in NUMERIC_RESULT_SCALARS:
            return True
    if expr.has_aggregation:
        primary = expr.primary_term
        result = extract_agg_col(primary)
        if len(result) == 3 and result[0] in {"count", "sum", "avg"}:
            return True
    col = expr.primary_column
    if col:
        return is_col_numeric(col, schema, cte_outputs)
    return None


def validate_scalar_expression_semantics(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that scalar functions are applied to semantically appropriate column types.

    Args:

        select_cols: SELECT column list to inspect for scalar misuse.

        schema: Schema graph for resolving column types and roles.

        context: Query context label for issue messages.

    Returns:

        List of `IntentIssue` instances describing scalar semantic violations.
    """
    issues = []
    debug("[validation_semantic.validate_scalar_expression_semantics] checking scalar semantics")
    numeric_scalars = {"abs", "round", "ceil", "floor", "sqrt"}
    string_scalars = {"upper", "lower", "trim", "ltrim", "rtrim", "length"}
    for sc in select_cols:
        outer_func, _, _ = extract_agg_col(sc.expr.primary_term)
        if not outer_func or outer_func in VALID_AGGREGATION_FUNCTIONS:
            continue
        func_lower = outer_func
        col_type = get_col_type(sc.expr.primary_column, schema, {})
        if col_type:
            numeric = col_type in ("integer", "number")
            text = col_type == "string"
            if func_lower in numeric_scalars and not numeric and not sc.is_aggregated:
                issues.append(
                    make_intent_issue(
                        issue_id=f"numeric_scalar_on_non_numeric_{sc.expr.primary_column}_{func_lower}",
                        category=FailureCategory.SCALAR_SEMANTIC,
                        severity="warning",
                        message=f"Numeric scalar '{func_lower}' on non-numeric column '{sc.expr.primary_column}' (type: {col_type})",
                        context={
                            "column": sc.expr.primary_column,
                            "scalar": func_lower,
                            "type": col_type,
                            "location": context,
                        },
                    )
                )
            if func_lower in string_scalars and not text:
                issues.append(
                    make_intent_issue(
                        issue_id=f"string_scalar_on_non_string_{sc.expr.primary_column}_{func_lower}",
                        category=FailureCategory.SCALAR_SEMANTIC,
                        severity="warning",
                        message=f"String scalar '{func_lower}' on non-string column '{sc.expr.primary_column}' (type: {col_type})",
                        context={
                            "column": sc.expr.primary_column,
                            "scalar": func_lower,
                            "type": col_type,
                            "location": context,
                        },
                    )
                )
    debug(f"[validation_semantic.validate_scalar_expression_semantics] {len(issues)} issues in {context}")
    return issues


def validate_temporal_columns(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate temporal aggregates against date-type columns in the intent.

    Args:

        select_cols: SELECT column list to inspect for temporal misuse.

        schema: Schema graph for resolving column types.

        context: Query context label for issue messages.

    Returns:

        List of `IntentIssue` instances when temporal ops lack an identifiable date column.
    """
    issues = []
    temporal_ops = {"latest", "recent", "last", "first", "earliest", "oldest", "newest"}
    agg_funcs = {extract_agg_col(sc.expr.primary_term)[0] for sc in select_cols if sc.is_aggregated} - {None}
    if not (agg_funcs & temporal_ops):
        return []
    debug("[validation_semantic.validate_temporal_columns] checking temporal column presence")
    has_date_column = False
    for sc in select_cols:
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in schema.tables:
            col_meta = schema.tables[table_name].columns.get(col_name)
            if col_meta:
                if col_meta.value_type == "date":
                    has_date_column = True
                    break
        if any(hint in col_name.lower() for hint in ["date", "time", "created", "updated", "at"]):
            has_date_column = True
            break
    if not has_date_column:
        issues.append(
            make_intent_issue(
                issue_id=f"temporal_no_date_col_{','.join(sorted(agg_funcs & temporal_ops))}",
                category=FailureCategory.MISSING_TEMPORAL_COLUMN,
                severity="warning",
                message=f"Intent uses temporal operation ({agg_funcs & temporal_ops}) but no date/time column identified",
                context={
                    "temporal_ops": list(agg_funcs & temporal_ops),
                    "location": context,
                },
            )
        )
        debug("[validation_semantic.validate_temporal_columns] AMBIGUITY: temporal ops but no date column")
    return issues


def validate_pk_fk_aggregation(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate that primary-key and foreign-key columns are not aggregated with SUM or AVG.

    Args:

        select_cols: SELECT column list to inspect for PK/FK aggregation misuse.

        schema: Schema graph for resolving column roles.

        context: Query context label for issue messages.

    Returns:

        List of `IntentIssue` instances where a PK or FK column uses `SUM` or `AVG`.
    """
    issues = []
    suspicious_aggs = {"sum", "avg"}
    debug("[validation_semantic.validate_pk_fk_aggregation] checking PK/FK aggregation")
    for sc in select_cols:
        if not sc.is_aggregated:
            continue
        func_lower, _, _ = extract_agg_col(sc.expr.primary_term)
        if not func_lower or func_lower not in suspicious_aggs:
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        col_meta = schema.tables[table_name].columns.get(col_name)
        if col_meta and (col_meta.is_primary_key or col_meta.is_foreign_key):
            issues.append(
                make_intent_issue(
                    issue_id=f"agg_on_pk_fk_{table_name}_{col_name}_{func_lower}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="warning",
                    message=f"{func_lower.upper()} on PK/FK column {actual_col} is suspicious",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "agg": func_lower,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_semantic.validate_pk_fk_aggregation] {func_lower.upper()} on PK/FK: {actual_col}")
    return issues
