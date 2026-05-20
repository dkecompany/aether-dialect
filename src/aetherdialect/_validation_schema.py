"""Validate intents against the schema for columns, operators, aggregates, scalars, and CTE-aware checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ._config import (
    AGGREGATION_ALLOWED_COLUMN_TYPES,
    ARITHMETIC_ROLES,
    DISALLOWED_EXTRACT_UNITS,
    REGISTRY_CASE_ID_RE,
    REGISTRY_WINDOW_ID_RE,
    SCALAR_FUNCTIONS_TEMPORAL,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_FILTER_OPS,
    VALID_HAVING_OPS,
    VALID_RELATIVE_DATE_UNITS,
    VALID_SCALAR_FUNCTIONS,
    VALID_VALUE_TYPES,
    VALID_WINDOW_FUNCTIONS,
    WINDOW_AGG_FUNCTIONS,
    WINDOW_FRAME_BOUNDS,
    WINDOW_OFFSET_FUNCTIONS,
    WINDOW_RANKING_FUNCTIONS,
    WINDOW_VALUE_FUNCTIONS,
)
from ._contracts_base import (
    ColumnMetadata,
    CteOutputColumnMeta,
    FailureCategory,
    IntentIssue,
    SchemaGraph,
    make_intent_issue,
)
from ._contracts_core import (
    CaseRegistryStep,
    CaseWhenExpr,
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
    effective_select_parts,
    expr_registry_ref,
)
from ._core_utils import debug
from ._intent_expr import (
    extract_columns_from_expr,
    strip_leading_distinct_from_column_ref,
)


def _strip_distinct_prefix(col: str) -> str:
    """
    Remove a leading `DISTINCT` keyword from a column reference string.

    Args:

        col: Column expression that may start with `DISTINCT`.

    Returns:

        The same expression with a leading `DISTINCT` removed, or `col` unchanged.
    """
    if col and col.upper().startswith("DISTINCT "):
        return col[9:].strip()
    return col


def extract_col_from_scalar_wrapper(col_expr: str) -> str:
    """
    Strip a scalar wrapper and leading `DISTINCT`, returning the inner column expression.

    Args:

        col_expr: Column expression, optionally wrapped in a scalar function and/or `DISTINCT`.

    Returns:

        The inner column reference, or `col_expr` if no recognised scalar wrapper applies.
    """
    if not col_expr:
        return col_expr
    match = re.match(r"^\s*(\w+)\s*\(\s*(.+)\s*\)\s*$", col_expr, re.IGNORECASE)
    if match:
        func_name = match.group(1).lower()
        if func_name in VALID_SCALAR_FUNCTIONS:
            return _strip_distinct_prefix(match.group(2).strip())
    return _strip_distinct_prefix(col_expr)


def _resolve_col_meta_from_expr(col_expr: str | None, schema: Any) -> Any | None:
    """Return ColumnMetadata for a qualified ``table.column`` expression, or None when unresolvable."""

    if not col_expr:
        return None
    actual = extract_col_from_scalar_wrapper(col_expr)
    if "." not in actual:
        return None
    table_name, col_name = actual.rsplit(".", 1)
    table_meta = getattr(schema, "tables", {}).get(table_name)
    if table_meta is None:
        return None
    cols = getattr(table_meta, "columns", {})
    return cols.get(col_name) or cols.get(col_name.lower())


def _validate_scalar_func_valid(
    scalar_func: str | None,
    context: str,
    location: str,
    col_meta: Any | None = None,
) -> list[IntentIssue]:
    """
    Validate that a scalar function name is allowed.

    Args:

        scalar_func: Scalar function name to validate, or `None`.

        context: Short label for the field (for example `select_0`).

        location: Human-readable location for error messages.

        col_meta: Optional resolved ColumnMetadata for the wrapped column. When provided, temporal scalar functions (``date_trunc``, ``date_part``, ``extract``, ``year``, ``month``, ``day``) are rejected if the column's ``value_type`` is not ``date``.

    Returns:

        `IntentIssue` instances; empty if valid or if `scalar_func` is `None`.
    """
    issues = []
    if not scalar_func:
        return issues
    func_lower = scalar_func.lower()
    if func_lower not in VALID_SCALAR_FUNCTIONS:
        issues.append(
            make_intent_issue(
                issue_id=f"invalid_scalar_func_{context}_{scalar_func}",
                category=FailureCategory.SCALAR_VALIDITY,
                severity="error",
                message=f"Invalid scalar function '{scalar_func}' in {location}. Allowed: {', '.join(sorted(VALID_SCALAR_FUNCTIONS))}",
                context={
                    "function": scalar_func,
                    "location": location,
                    "allowed": list(VALID_SCALAR_FUNCTIONS),
                },
            )
        )
        debug(f"[validation_schema.validate_scalar_func_valid] invalid scalar '{scalar_func}' in {location}")
        return issues
    if col_meta is not None and func_lower in SCALAR_FUNCTIONS_TEMPORAL:
        col_value_type = (getattr(col_meta, "value_type", "") or "").strip().lower()
        if col_value_type and col_value_type != "date":
            issues.append(
                make_intent_issue(
                    issue_id=f"temporal_scalar_on_non_date_{context}_{scalar_func}",
                    category=FailureCategory.SCALAR_VALIDITY,
                    severity="error",
                    message=(
                        f"Temporal scalar '{scalar_func}' applied to non-date column "
                        f"'{getattr(col_meta, 'name', '')}' (value_type={col_value_type!r}) in {location}"
                    ),
                    context={
                        "function": scalar_func,
                        "location": location,
                        "column": getattr(col_meta, "name", ""),
                        "value_type": col_value_type,
                    },
                )
            )
            debug(
                f"[validation_schema.validate_scalar_func_valid] temporal '{scalar_func}' on "
                f"non-date column {getattr(col_meta, 'name', '')!r} (value_type={col_value_type!r})"
            )
    return issues


def _first_arg_lower(args: list[Any]) -> str:
    """
    Return the first scalar argument lowercased, or an empty string.

    Args:

    args: Scalar argument list from an expression.

    Returns:

        Lowercased first argument, or `""` if `args` is empty.
    """
    if not args:
        return ""
    return str(args[0]).strip().lower()


def _is_extract_epoch(func: str | None, args: list[Any]) -> bool:
    """
    Return whether `func` is `extract` with a disallowed unit such as epoch.

    Args:

        func: Function name, or `None`.

    args: Argument list whose first entry is treated as the extract unit.

    Returns:

        `True` if the unit is disallowed; otherwise `False`.
    """
    if not func or func.lower() != "extract":
        return False
    unit = _first_arg_lower(args)
    return unit in DISALLOWED_EXTRACT_UNITS


def validate_expr_no_extract_epoch(
    expr: NormalizedExpr,
    context: str,
    location: str,
) -> list[IntentIssue]:
    """
    Flag `EXTRACT(EPOCH FROM ...)` in expressions because EPOCH is not supported.

    Args:

        expr: Normalized expression tree to walk (including add/sub groups).

        context: Label for issue identifiers.

        location: Human-readable location for messages.

    Returns:

        One `IntentIssue` per disallowed extract occurrence.
    """
    issues: list[IntentIssue] = []
    if _is_extract_epoch(expr.scalar_func, expr.scalar_func_args or []):
        issues.append(
            make_intent_issue(
                issue_id=f"extract_epoch_{context}",
                category=FailureCategory.EXTRACT_EPOCH,
                severity="error",
                message=(
                    "EXTRACT(EPOCH FROM ...) is not supported. Use date column "
                    "subtraction or other supported date functions."
                ),
                context={"location": location},
            )
        )
    if _is_extract_epoch(expr.inner_scalar_func, expr.inner_scalar_func_args or []):
        issues.append(
            make_intent_issue(
                issue_id=f"extract_epoch_inner_{context}",
                category=FailureCategory.EXTRACT_EPOCH,
                severity="error",
                message=(
                    "EXTRACT(EPOCH FROM ...) is not supported. Use date column "
                    "subtraction or other supported date functions."
                ),
                context={"location": location},
            )
        )
    for group in expr.add_groups + expr.sub_groups:
        if _is_extract_epoch(group.scalar_func, group.scalar_func_args or []):
            issues.append(
                make_intent_issue(
                    issue_id=f"extract_epoch_group_{context}",
                    category=FailureCategory.EXTRACT_EPOCH,
                    severity="error",
                    message=(
                        "EXTRACT(EPOCH FROM ...) is not supported. Use date column "
                        "subtraction or other supported date functions."
                    ),
                    context={"location": location},
                )
            )
        if _is_extract_epoch(group.inner_scalar_func, group.inner_scalar_func_args or []):
            issues.append(
                make_intent_issue(
                    issue_id=f"extract_epoch_inner_group_{context}",
                    category=FailureCategory.EXTRACT_EPOCH,
                    severity="error",
                    message=(
                        "EXTRACT(EPOCH FROM ...) is not supported. Use date column "
                        "subtraction or other supported date functions."
                    ),
                    context={"location": location},
                )
            )
    return issues


def _validate_agg_func_valid(agg_func: str | None, context: str, location: str) -> list[IntentIssue]:
    """
    Validate that an aggregation function name is allowed.

    Args:

        agg_func: Aggregation function name to validate, or `None`.

        context: Short label for the field being validated.

        location: Human-readable location for error messages.

    Returns:

        `IntentIssue` instances; empty if valid or if `agg_func` is `None`.
    """
    issues = []
    if not agg_func:
        return issues
    func_lower = agg_func.lower()
    if func_lower not in VALID_AGGREGATION_FUNCTIONS:
        issues.append(
            make_intent_issue(
                issue_id=f"invalid_agg_func_{context}_{agg_func}",
                category=FailureCategory.AGGREGATION_VALIDITY,
                severity="error",
                message=f"Invalid aggregation function '{agg_func}' in {location}. Allowed: {', '.join(sorted(VALID_AGGREGATION_FUNCTIONS))}",
                context={
                    "function": agg_func,
                    "location": location,
                    "allowed": list(VALID_AGGREGATION_FUNCTIONS),
                },
            )
        )
        debug(f"[validation_schema.validate_agg_func_valid] invalid agg '{agg_func}' in {location}")
    return issues


def _expr_has_registry_ref(expr: NormalizedExpr) -> bool:
    """Return True when *expr* is a bare registry-id reference."""

    return expr_registry_ref(expr) is not None


def _window_spec_has_registry_ref(ws: WindowSpec) -> bool:
    """Return True when any window sub-expression carries ``registry_ref``."""

    for e in ws.partition_by:
        if _expr_has_registry_ref(e):
            return True
    for o in ws.order_by:
        if _expr_has_registry_ref(o.expr):
            return True
    if ws.argument is not None and _expr_has_registry_ref(ws.argument):
        return True
    return False


def _case_when_has_registry_ref(cw: CaseWhenExpr) -> bool:
    """Return True when any CASE branch carries ``registry_ref``."""

    for br in cw.branches:
        if _filter_param_has_registry_ref(br.condition):
            return True
        if _expr_has_registry_ref(br.result):
            return True
    if cw.else_result is not None and _expr_has_registry_ref(cw.else_result):
        return True
    return False


def _filter_param_has_registry_ref(fp: FilterParam) -> bool:
    """Return True when a filter row references a registry id on an expression."""

    if _expr_has_registry_ref(fp.left_expr):
        return True
    if fp.right_expr is not None and _expr_has_registry_ref(fp.right_expr):
        return True
    return False


def validate_scope_registries(
    *,
    context: str,
    window_registry: list[WindowRegistryStep],
    case_registry: list[CaseRegistryStep],
    select_cols: list[SelectCol],
    group_by_cols: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    filters_param: list[FilterParam],
    having_param: list[HavingParam],
) -> list[IntentIssue]:
    """
    Validate per-scope window/case registry definitions and ``registry_ref`` uses.

    Args:

        context: Label for issue messages.

        window_registry: Window definitions for this scope.

        case_registry: Case definitions for this scope.

        select_cols: SELECT list for this scope.

        group_by_cols: GROUP BY expressions.

        order_by_cols: ORDER BY columns.

        filters_param: WHERE filters.

        having_param: HAVING clauses.

    Returns:

        Collected ``IntentIssue`` instances.
    """

    issues: list[IntentIssue] = []
    win_ids = [s.registry_id for s in window_registry or []]
    case_ids = [s.registry_id for s in case_registry or []]
    for wid in {i for i in win_ids if win_ids.count(i) > 1}:
        issues.append(
            make_intent_issue(
                issue_id=f"registry_duplicate_window_{context}_{wid}",
                category=FailureCategory.REGISTRY,
                severity="error",
                message=f"{context}: duplicate window registry_id '{wid}'",
                context={"registry_id": wid, "location": context},
            )
        )
    for cid in {i for i in case_ids if case_ids.count(i) > 1}:
        issues.append(
            make_intent_issue(
                issue_id=f"registry_duplicate_case_{context}_{cid}",
                category=FailureCategory.REGISTRY,
                severity="error",
                message=f"{context}: duplicate case registry_id '{cid}'",
                context={"registry_id": cid, "location": context},
            )
        )
    win_by = {s.registry_id: s for s in window_registry or []}
    case_by = {s.registry_id: s for s in case_registry or []}
    for s in window_registry or []:
        rid = s.registry_id
        if not REGISTRY_WINDOW_ID_RE.match(rid):
            issues.append(
                make_intent_issue(
                    issue_id=f"registry_invalid_window_id_{context}_{rid}",
                    category=FailureCategory.REGISTRY,
                    severity="error",
                    message=f"{context}: window registry_id must match ^w\\d{{2}}$, got '{rid}'",
                    context={"registry_id": rid, "location": context},
                )
            )
        if _window_spec_has_registry_ref(s.window_spec):
            issues.append(
                make_intent_issue(
                    issue_id=f"registry_recursion_window_{context}_{rid}",
                    category=FailureCategory.REGISTRY,
                    severity="error",
                    message=f"{context}: window registry body must not contain registry_ref (id '{rid}')",
                    context={"registry_id": rid, "location": context},
                )
            )
    for s in case_registry or []:
        rid = s.registry_id
        if not REGISTRY_CASE_ID_RE.match(rid):
            issues.append(
                make_intent_issue(
                    issue_id=f"registry_invalid_case_id_{context}_{rid}",
                    category=FailureCategory.REGISTRY,
                    severity="error",
                    message=f"{context}: case registry_id must match ^c\\d{{2}}$, got '{rid}'",
                    context={"registry_id": rid, "location": context},
                )
            )
        if _case_when_has_registry_ref(s.case_when):
            issues.append(
                make_intent_issue(
                    issue_id=f"registry_recursion_case_{context}_{rid}",
                    category=FailureCategory.REGISTRY,
                    severity="error",
                    message=f"{context}: case registry body must not contain registry_ref (id '{rid}')",
                    context={"registry_id": rid, "location": context},
                )
            )

    def _check_ref(ref: str, *, where: str) -> None:
        ref = ref.strip()
        if not ref:
            return
        if REGISTRY_WINDOW_ID_RE.match(ref):
            if ref not in win_by:
                issues.append(
                    make_intent_issue(
                        issue_id=f"registry_dangling_window_{context}_{ref}_{where}",
                        category=FailureCategory.REGISTRY,
                        severity="error",
                        message=f"{context}: undefined window registry_ref '{ref}' in {where}",
                        context={"registry_ref": ref, "location": context},
                    )
                )
            return
        if REGISTRY_CASE_ID_RE.match(ref):
            if ref not in case_by:
                issues.append(
                    make_intent_issue(
                        issue_id=f"registry_dangling_case_{context}_{ref}_{where}",
                        category=FailureCategory.REGISTRY,
                        severity="error",
                        message=f"{context}: undefined case registry_ref '{ref}' in {where}",
                        context={"registry_ref": ref, "location": context},
                    )
                )
            return
        issues.append(
            make_intent_issue(
                issue_id=f"registry_invalid_ref_format_{context}_{ref}_{where}",
                category=FailureCategory.REGISTRY,
                severity="error",
                message=f"{context}: registry_ref '{ref}' must match ^w\\d{{2}}$ or ^c\\d{{2}}$",
                context={"registry_ref": ref, "location": context},
            )
        )

    for si, sc in enumerate(select_cols or []):
        ref = expr_registry_ref(sc.expr) or ""
        if ref:
            _check_ref(ref, where=f"select_cols[{si}]")
    for gi, g in enumerate(group_by_cols or []):
        ref = expr_registry_ref(g) or ""
        if ref:
            _check_ref(ref, where=f"group_by_cols[{gi}]")
    for oi, obc in enumerate(order_by_cols or []):
        ref = expr_registry_ref(obc.expr) or ""
        if ref:
            _check_ref(ref, where=f"order_by_cols[{oi}]")
    for fi, fp in enumerate(filters_param or []):
        ref = expr_registry_ref(fp.left_expr) or ""
        if ref:
            _check_ref(ref, where=f"filters_param[{fi}].left_expr")
        if fp.right_expr is not None:
            ref_r = expr_registry_ref(fp.right_expr) or ""
            if ref_r:
                _check_ref(ref_r, where=f"filters_param[{fi}].right_expr")
    for hi, hp in enumerate(having_param or []):
        ref = expr_registry_ref(hp.left_expr) or ""
        if ref:
            _check_ref(ref, where=f"having_param[{hi}].left_expr")
        if hp.right_expr is not None:
            ref_r = expr_registry_ref(hp.right_expr) or ""
            if ref_r:
                _check_ref(ref_r, where=f"having_param[{hi}].right_expr")
    return issues


def runtime_scope_registry_error_messages(rt: RuntimeIntent) -> list[str]:
    """
    Collect error-level scope-registry validation strings for the main query and each CTE scope.

    Args:

        rt: Fully scoped runtime intent.

    Returns:

        De-duplicated human-readable messages for registry consistency failures.
    """

    msgs: list[str] = []
    for iss in validate_scope_registries(
        context="main query",
        window_registry=list(rt.window_registry or []),
        case_registry=list(rt.case_registry or []),
        select_cols=rt.select_cols or [],
        group_by_cols=rt.group_by_cols or [],
        order_by_cols=rt.order_by_cols or [],
        filters_param=rt.filters_param or [],
        having_param=rt.having_param or [],
    ):
        if (iss.severity or "").lower() == "error":
            msgs.append(iss.message)
    for cte in rt.cte_steps or []:
        ctx = f"CTE '{cte.cte_name}'"
        for iss in validate_scope_registries(
            context=ctx,
            window_registry=list(cte.window_registry or []),
            case_registry=list(cte.case_registry or []),
            select_cols=cte.select_cols or [],
            group_by_cols=cte.group_by_cols or [],
            order_by_cols=cte.order_by_cols or [],
            filters_param=cte.filters_param or [],
            having_param=cte.having_param or [],
        ):
            if (iss.severity or "").lower() == "error":
                msgs.append(iss.message)
    return list(dict.fromkeys(msgs))


def validate_select_cols_schema(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> list[IntentIssue]:
    """
    Validate `SelectCol` entries for column existence and qualification.

    Args:

        select_cols: `SelectCol` instances to validate.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this query context.

        cte_outputs: CTE name to output column metadata for cross-CTE lookup.

        context: Label for issue IDs and messages (for example `main`).

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not select_cols:
        issues.append(
            make_intent_issue(
                issue_id=f"select_cols_empty_{context}",
                category=FailureCategory.SELECT_VALIDITY,
                severity="error",
                message=f"select_cols cannot be empty in {context}",
                context={"location": context},
            )
        )
        return issues
    cte_outputs = cte_outputs or {}
    for idx, sc in enumerate(select_cols):
        if expr_registry_ref(sc.expr) is not None:
            parts = effective_select_parts(sc, window_registry, case_registry)
            ex = parts.expr
            issues.extend(_validate_agg_func_valid(ex.agg_func, f"select_{idx}", context))
            for g in ex.add_groups + ex.sub_groups:
                issues.extend(_validate_agg_func_valid(g.agg_func, f"select_{idx}", context))
            sc_scalar, sc_agg = extract_functions_from_term(ex.primary_term)
            issues.extend(_validate_agg_func_valid(sc_agg, f"select_{idx}", context))
            issues.extend(_validate_scalar_func_valid(sc_scalar, f"select_{idx}", context))
            issues.extend(validate_expr_no_extract_epoch(ex, f"select_{idx}", context))
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            issues.append(
                make_intent_issue(
                    issue_id=f"select_col_empty_{context}_{idx}",
                    category=FailureCategory.SELECT_VALIDITY,
                    severity="error",
                    message=f"SelectCol at index {idx} has empty col in {context}",
                    context={"index": idx, "location": context},
                )
            )
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            issues.append(
                make_intent_issue(
                    issue_id=f"select_unqualified_{context}_{actual_col}",
                    category=FailureCategory.SELECT_VALIDITY,
                    severity="error",
                    message=f"select_cols must be qualified as table.column, got '{actual_col}' in {context}",
                    context={"column": actual_col, "location": context},
                )
            )
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            if col_name.lower() not in [c.lower() for c in cte_outputs[table_name]]:
                issues.append(
                    make_intent_issue(
                        issue_id=f"select_cte_col_not_found_{context}_{table_name}_{col_name}",
                        category=FailureCategory.SELECT_VALIDITY,
                        severity="error",
                        message=f"Column '{col_name}' not in CTE '{table_name}' outputs for select in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "location": context,
                        },
                    )
                )
            continue
        if table_name not in allowed_tables:
            issues.append(
                make_intent_issue(
                    issue_id=f"select_table_not_allowed_{context}_{table_name}",
                    category=FailureCategory.SELECT_VALIDITY,
                    severity="error",
                    message=f"Table '{table_name}' not in allowed tables for select in {context}",
                    context={"table": table_name, "location": context},
                )
            )
            continue
        col_meta = None
        if table_name in schema.tables:
            table_meta = schema.tables[table_name]
            col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
            if not col_meta:
                issues.append(
                    make_intent_issue(
                        issue_id=f"select_col_not_found_{context}_{table_name}_{col_name}",
                        category=FailureCategory.SELECT_VALIDITY,
                        severity="error",
                        message=f"Column '{col_name}' not in table '{table_name}' for select in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "location": context,
                        },
                    )
                )
        issues.extend(_validate_agg_func_valid(sc.expr.agg_func, f"select_{idx}", context))
        for g in sc.expr.add_groups + sc.expr.sub_groups:
            issues.extend(_validate_agg_func_valid(g.agg_func, f"select_{idx}", context))
        sc_scalar, sc_agg = extract_functions_from_term(sc.expr.primary_term)
        issues.extend(_validate_agg_func_valid(sc_agg, f"select_{idx}", context))
        issues.extend(_validate_scalar_func_valid(sc_scalar, f"select_{idx}", context, col_meta=col_meta))
        issues.extend(validate_expr_no_extract_epoch(sc.expr, f"select_{idx}", context))
    debug(f"[validation_schema.validate_select_cols_schema] {len(issues)} issues in {context}")
    return issues


def validate_order_by_cols_schema(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate `OrderByCol` entries for column existence and sort direction.

    Args:

        order_by_cols: `OrderByCol` instances to validate.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this query context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not order_by_cols:
        return []
    cte_outputs = cte_outputs or {}
    for idx, obc in enumerate(order_by_cols):
        col_expr = obc.expr.primary_column
        if not col_expr:
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_col_empty_{context}_{idx}",
                    category=FailureCategory.ORDER_BY_VALIDITY,
                    severity="error",
                    message=f"OrderByCol at index {idx} has empty col in {context}",
                    context={"index": idx, "location": context},
                )
            )
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_unqualified_{context}_{actual_col}",
                    category=FailureCategory.ORDER_BY_VALIDITY,
                    severity="error",
                    message=f"order_by_cols must be qualified as table.column, got '{actual_col}' in {context}",
                    context={"column": actual_col, "location": context},
                )
            )
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            if col_name.lower() not in [c.lower() for c in cte_outputs[table_name]]:
                issues.append(
                    make_intent_issue(
                        issue_id=f"order_by_cte_col_not_found_{context}_{table_name}_{col_name}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=f"Column '{col_name}' not in CTE '{table_name}' outputs for order_by in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "location": context,
                        },
                    )
                )
            continue
        if table_name not in allowed_tables:
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_table_not_allowed_{context}_{table_name}",
                    category=FailureCategory.ORDER_BY_VALIDITY,
                    severity="error",
                    message=f"Table '{table_name}' not in allowed tables for order_by in {context}",
                    context={"table": table_name, "location": context},
                )
            )
            continue
        col_meta = None
        if table_name in schema.tables:
            table_meta = schema.tables[table_name]
            col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
            if not col_meta:
                issues.append(
                    make_intent_issue(
                        issue_id=f"order_by_col_not_found_{context}_{table_name}_{col_name}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=f"Column '{col_name}' not in table '{table_name}' for order_by in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "location": context,
                        },
                    )
                )
        if obc.direction not in ("ASC", "DESC"):
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_invalid_direction_{context}_{idx}",
                    category=FailureCategory.ORDER_BY_VALIDITY,
                    severity="error",
                    message=f"OrderByCol direction must be 'ASC' or 'DESC', got '{obc.direction}' in {context}",
                    context={"direction": obc.direction, "location": context},
                )
            )
        obc_scalar, obc_agg = extract_functions_from_term(obc.expr.primary_term)
        issues.extend(_validate_agg_func_valid(obc_agg, f"order_by_{idx}", context))
        issues.extend(_validate_scalar_func_valid(obc_scalar, f"order_by_{idx}", context, col_meta=col_meta))
        issues.extend(validate_expr_no_extract_epoch(obc.expr, f"order_by_{idx}", context))
    debug(f"[validation_schema.validate_order_by_cols_schema] {len(issues)} issues in {context}")
    return issues


def validate_group_by_cols_schema(
    group_by_cols: list[NormalizedExpr],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate `group_by_cols` against the schema and column groupability.

    Args:

        group_by_cols: `NormalizedExpr` instances to validate.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this query context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not group_by_cols:
        return []
    cte_outputs = cte_outputs or {}
    for g in group_by_cols:
        col = g.primary_column
        if "." not in col:
            issues.append(
                make_intent_issue(
                    issue_id=f"group_by_unqualified_{context}_{col}",
                    category=FailureCategory.GROUP_BY_VALIDITY,
                    severity="error",
                    message=f"group_by_cols must be qualified as table.column, got '{col}' in {context}",
                    context={"column": col, "location": context},
                )
            )
            continue
        table_name, col_name = col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if not matched_key:
                issues.append(
                    make_intent_issue(
                        issue_id=f"group_by_cte_col_not_found_{context}_{table_name}_{col_name}",
                        category=FailureCategory.GROUP_BY_VALIDITY,
                        severity="error",
                        message=f"Column '{col_name}' not in CTE '{table_name}' outputs for group_by in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "location": context,
                        },
                    )
                )
            elif not cte_cols[matched_key].groupable:
                issues.append(
                    make_intent_issue(
                        issue_id=f"group_by_cte_col_not_groupable_{context}_{table_name}_{col_name}",
                        category=FailureCategory.GROUP_BY_VALIDITY,
                        severity="warning",
                        message=f"CTE column '{table_name}.{col_name}' (role={cte_cols[matched_key].role}) is not recommended for GROUP BY in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "role": cte_cols[matched_key].role,
                            "location": context,
                        },
                    )
                )
            continue
        if table_name not in allowed_tables:
            issues.append(
                make_intent_issue(
                    issue_id=f"group_by_table_not_allowed_{context}_{table_name}",
                    category=FailureCategory.GROUP_BY_VALIDITY,
                    severity="error",
                    message=f"Table '{table_name}' not in allowed tables for group_by in {context}",
                    context={"table": table_name, "location": context},
                )
            )
            continue
        if table_name in schema.tables:
            table_meta = schema.tables[table_name]
            col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
            if not col_meta:
                issues.append(
                    make_intent_issue(
                        issue_id=f"group_by_col_not_found_{context}_{table_name}_{col_name}",
                        category=FailureCategory.GROUP_BY_VALIDITY,
                        severity="error",
                        message=f"Column '{col_name}' not in table '{table_name}' for group_by in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "location": context,
                        },
                    )
                )
            elif not col_meta.is_groupable:
                issues.append(
                    make_intent_issue(
                        issue_id=f"group_by_col_not_groupable_{context}_{table_name}_{col_name}",
                        category=FailureCategory.GROUP_BY_VALIDITY,
                        severity="warning",
                        message=f"Column '{col_name}' (role={col_meta.role}) is not recommended for grouping in {context}",
                        context={
                            "table": table_name,
                            "column": col_name,
                            "role": col_meta.role,
                            "location": context,
                        },
                    )
                )
    debug(f"[validation_schema.validate_group_by_cols_schema] {len(issues)} issues in {context}")
    return issues


def _validate_filter_col(
    col_expr: str,
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
    context: str,
    side: str,
    param_key: str,
) -> list[IntentIssue]:
    """
    Validate one filter column reference (left or right of a `FilterParam`).

    Args:

        col_expr: Column expression string to validate.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

        side: `left_col` or `right_col`.

        param_key: `FilterParam.param_key` for issue IDs.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not col_expr:
        return issues
    actual_col = extract_col_from_scalar_wrapper(col_expr)
    if "." not in actual_col:
        issues.append(
            make_intent_issue(
                issue_id=f"filter_{side}_unqualified_{context}_{actual_col}",
                category=FailureCategory.FILTER_VALIDITY,
                severity="error",
                message=f"Filter {side} must be qualified as table.column, got '{actual_col}' in {context}",
                context={
                    "column": actual_col,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    table_name, col_name = actual_col.rsplit(".", 1)
    if table_name in cte_outputs:
        if col_name.lower() not in [c.lower() for c in cte_outputs[table_name]]:
            issues.append(
                make_intent_issue(
                    issue_id=f"filter_{side}_cte_col_not_found_{context}_{table_name}_{col_name}",
                    category=FailureCategory.FILTER_VALIDITY,
                    severity="error",
                    message=f"Column '{col_name}' not in CTE '{table_name}' outputs for filter {side} in {context}",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "side": side,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
        return issues
    if table_name not in allowed_tables:
        issues.append(
            make_intent_issue(
                issue_id=f"filter_{side}_table_not_allowed_{context}_{table_name}",
                category=FailureCategory.FILTER_VALIDITY,
                severity="error",
                message=f"Table '{table_name}' not in allowed tables for filter {side} in {context}",
                context={
                    "table": table_name,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    if table_name not in schema.tables:
        issues.append(
            make_intent_issue(
                issue_id=f"filter_{side}_table_not_in_schema_{context}_{table_name}",
                category=FailureCategory.FILTER_VALIDITY,
                severity="error",
                message=f"Table '{table_name}' not in schema for filter {side} in {context}",
                context={
                    "table": table_name,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    table_meta = schema.tables[table_name]
    col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
    if not col_meta:
        issues.append(
            make_intent_issue(
                issue_id=f"filter_{side}_col_not_found_{context}_{table_name}_{col_name}",
                category=FailureCategory.FILTER_VALIDITY,
                severity="error",
                message=f"Column '{col_name}' not in table '{table_name}' for filter {side} in {context}",
                context={
                    "table": table_name,
                    "column": col_name,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
    return issues


def validate_filters_schema(
    filters_param: list[FilterParam],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """
    Validate `FilterParam` entries against the schema and allowed operators.

    Args:

        filters_param: `FilterParam` instances to validate.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

        param_values: Bound literals for this scope; used when ``raw_value`` was cleared after hoisting.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not filters_param:
        return []
    cte_outputs = cte_outputs or {}
    for fp in filters_param:
        param_key = fp.param_key or "unknown"
        issues.extend(
            _validate_filter_col(
                fp.left_expr.primary_column,
                schema,
                allowed_tables,
                cte_outputs,
                context,
                "left_col",
                param_key,
            )
        )
        if fp.right_expr:
            issues.extend(
                _validate_filter_col(
                    fp.right_expr.primary_column,
                    schema,
                    allowed_tables,
                    cte_outputs,
                    context,
                    "right_col",
                    param_key,
                )
            )
        if fp.op not in VALID_FILTER_OPS:
            issues.append(
                make_intent_issue(
                    issue_id=f"filter_invalid_op_{context}_{fp.op}",
                    category=FailureCategory.FILTER_VALIDITY,
                    severity="error",
                    message=f"Invalid filter operator '{fp.op}' in {context}",
                    context={
                        "operator": fp.op,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
        if not fp.right_expr and fp.op not in ("is null", "is not null"):
            if fp.value_type not in VALID_VALUE_TYPES:
                issues.append(
                    make_intent_issue(
                        issue_id=f"filter_invalid_value_type_{context}_{fp.value_type}",
                        category=FailureCategory.FILTER_VALIDITY,
                        severity="error",
                        message=f"Invalid filter value_type '{fp.value_type}' in {context}",
                        context={
                            "value_type": fp.value_type,
                            "param_key": param_key,
                            "location": context,
                        },
                    )
                )
            elif fp.resolved_value(param_values) is None:
                issues.append(
                    make_intent_issue(
                        issue_id=f"filter_missing_value_{context}_{param_key}",
                        category=FailureCategory.FILTER_VALIDITY,
                        severity="error",
                        message=(
                            f"Filter parameter '{param_key}' has no comparison value in {context}. "
                            "Provide a literal or bind ``param_key`` in param_values."
                        ),
                        context={
                            "param_key": param_key,
                            "op": fp.op,
                            "location": context,
                        },
                    )
                )
        fp_left_scalar, _ = extract_functions_from_term(fp.left_expr.primary_term)
        fp_right_scalar, _ = extract_functions_from_term(fp.right_expr.primary_term) if fp.right_expr else (None, None)
        fp_left_meta = _resolve_col_meta_from_expr(fp.left_expr.primary_column, schema)
        fp_right_meta = _resolve_col_meta_from_expr(fp.right_expr.primary_column, schema) if fp.right_expr else None
        issues.extend(
            _validate_scalar_func_valid(
                fp_left_scalar,
                f"filter_{param_key}_left",
                context,
                col_meta=fp_left_meta,
            )
        )
        issues.extend(
            _validate_scalar_func_valid(
                fp_right_scalar,
                f"filter_{param_key}_right",
                context,
                col_meta=fp_right_meta,
            )
        )
        issues.extend(validate_expr_no_extract_epoch(fp.left_expr, f"filter_{param_key}_left", context))
        if fp.right_expr:
            issues.extend(validate_expr_no_extract_epoch(fp.right_expr, f"filter_{param_key}_right", context))
        if fp.bool_op not in ("AND", "OR"):
            issues.append(
                make_intent_issue(
                    issue_id=f"filter_invalid_bool_op_{context}_{fp.bool_op}",
                    category=FailureCategory.FILTER_VALIDITY,
                    severity="error",
                    message=f"Invalid filter bool_op '{fp.bool_op}' in {context}. Must be 'AND' or 'OR'.",
                    context={
                        "bool_op": fp.bool_op,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_filters_schema] {len(issues)} issues in {context}")
    return issues


def extract_agg_col(agg_expr: str) -> tuple:
    """
    Parse `(func, target, has_distinct)` from an aggregation expression string.

    Args:

        agg_expr: Aggregation expression such as `COUNT(DISTINCT t.c)`.

    Returns:

        `(func, target, has_distinct)`, or `(None, None, False)` if the pattern does not match.
    """
    if not agg_expr:
        return (None, None, False)
    match = re.match(r"^\s*(\w+)\s*\(\s*(.*)\s*\)\s*$", agg_expr, re.IGNORECASE)
    if not match:
        return (None, None, False)
    func = match.group(1).lower()
    inner = match.group(2).strip()
    has_distinct = False
    actual_target = inner
    if inner.upper().startswith("DISTINCT "):
        has_distinct = True
        actual_target = inner[9:].strip()
    actual_target = extract_col_from_scalar_wrapper(actual_target)
    return (func, actual_target, has_distinct)


def extract_functions_from_term(term: str) -> tuple[str | None, str | None]:
    """
    Extract outer scalar and inner aggregation function names from a term.

    Args:

        term: Expression term string (possibly nested calls).

    Returns:

        `(scalar_func, agg_func)` as lowercase names, each `None` if absent.
    """
    result = extract_agg_col(term)
    if len(result) != 3 or not result[0]:
        return None, None
    outer = result[0]
    if outer in VALID_AGGREGATION_FUNCTIONS:
        return None, outer
    inner_result = extract_agg_col(result[1]) if result[1] else (None, None, False)
    if len(inner_result) == 3 and inner_result[0] and inner_result[0] in VALID_AGGREGATION_FUNCTIONS:
        return outer, inner_result[0]
    return outer, None


def _validate_having_agg(
    agg_expr: str,
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
    context: str,
    side: str,
    param_key: str,
) -> list[IntentIssue]:
    """
    Validate one HAVING aggregation expression (left or right side).

    Args:

        agg_expr: Aggregation string (for example `COUNT(t.c)`).

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

        side: `left_agg` or `right_agg`.

        param_key: `HavingParam.param_key` for issue IDs.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not agg_expr:
        return issues
    cte_col_match = re.match(r"^\s*(\w+)\s*\.\s*(\w+)\s*$", agg_expr.strip())
    if cte_col_match:
        cte_name, col_name = cte_col_match.group(1), cte_col_match.group(2)
        cte_cols = cte_outputs.get(cte_name, {})
        col_meta = next(
            (v for k, v in cte_cols.items() if k.lower() == col_name.lower()),
            None,
        )
        if col_meta and (col_meta.source == "aggregation" or (col_meta.agg_func or "").strip()):
            return issues
    result = extract_agg_col(agg_expr)
    if len(result) != 3:
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_invalid_format_{context}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"Invalid aggregation format in HAVING {side}: '{agg_expr}' in {context}",
                context={
                    "aggregation": agg_expr,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    func, actual_target, has_distinct = result
    if not func:
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_invalid_format_{context}_{agg_expr}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"Invalid aggregation format in HAVING {side}: '{agg_expr}' in {context}",
                context={
                    "aggregation": agg_expr,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    if func not in VALID_AGGREGATION_FUNCTIONS:
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_invalid_func_{context}_{func}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"Invalid aggregation function '{func}' in HAVING {side} for {context}",
                context={
                    "function": func,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    if has_distinct and func != "count":
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_distinct_not_count_{context}_{func}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"DISTINCT only allowed with COUNT in HAVING, not {func.upper()} in {context}",
                context={
                    "function": func,
                    "aggregation": agg_expr,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    if actual_target == "*":
        if func != "count":
            issues.append(
                make_intent_issue(
                    issue_id=f"having_{side}_star_not_count_{context}_{func}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"'*' only allowed with COUNT in HAVING, not {func.upper()} in {context}",
                    context={
                        "function": func,
                        "side": side,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
        return issues
    if "." not in actual_target:
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_unqualified_{context}_{actual_target}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"HAVING aggregation target must be qualified as table.column, got '{actual_target}' in {context}",
                context={
                    "target": actual_target,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    table_name, col_name = actual_target.rsplit(".", 1)
    if table_name in cte_outputs:
        if col_name.lower() not in [c.lower() for c in cte_outputs[table_name]]:
            issues.append(
                make_intent_issue(
                    issue_id=f"having_{side}_cte_col_not_found_{context}_{table_name}_{col_name}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"Column '{col_name}' not in CTE '{table_name}' outputs for HAVING {side} in {context}",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "side": side,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
        return issues
    if table_name not in allowed_tables:
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_table_not_allowed_{context}_{table_name}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"Table '{table_name}' not in allowed tables for HAVING {side} in {context}",
                context={
                    "table": table_name,
                    "side": side,
                    "param_key": param_key,
                    "location": context,
                },
            )
        )
        return issues
    if table_name in schema.tables:
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            issues.append(
                make_intent_issue(
                    issue_id=f"having_{side}_col_not_found_{context}_{table_name}_{col_name}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"Column '{col_name}' not in table '{table_name}' for HAVING {side} in {context}",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "side": side,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
        elif func != "count":
            value_type = col_meta.value_type or "string"
            allowed_types = AGGREGATION_ALLOWED_COLUMN_TYPES.get(func, [])
            if value_type not in allowed_types:
                issues.append(
                    make_intent_issue(
                        issue_id=f"having_{side}_type_mismatch_{context}_{func}_{col_name}",
                        category=FailureCategory.HAVING_VALIDITY,
                        severity="error",
                        message=f"Cannot use {func.upper()} on column '{actual_target}' of type '{col_meta.data_type}' in HAVING {side} for {context}",
                        context={
                            "function": func,
                            "column": actual_target,
                            "column_type": col_meta.data_type,
                            "side": side,
                            "param_key": param_key,
                            "location": context,
                        },
                    )
                )
    return issues


def _reconstruct_agg_expr(expr: NormalizedExpr) -> str:
    """
    Rebuild an aggregation expression string from a `NormalizedExpr`.

    Args:

        expr: `NormalizedExpr` from one side of a `HavingParam`.

    Returns:

        Canonical `FUNC(column)` text for `_validate_having_agg`, or `primary_term` if there is no aggregation.
    """
    agg_func = expr.agg_func
    if not agg_func and expr.add_groups:
        agg_func = expr.add_groups[0].agg_func
    column = expr.primary_term
    if not agg_func:
        return column
    return f"{agg_func.upper()}({column})"


def validate_having_schema(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """
    Validate `HavingParam` entries against the schema and allowed operators.

    Args:

        having_param: `HavingParam` instances to validate.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

        param_values: Bound literals for this scope; used when ``raw_value`` was cleared after hoisting.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:
        param_key = hp.param_key or "unknown"
        issues.extend(
            _validate_having_agg(
                _reconstruct_agg_expr(hp.left_expr),
                schema,
                allowed_tables,
                cte_outputs,
                context,
                "left_agg",
                param_key,
            )
        )
        if hp.right_expr:
            issues.extend(
                _validate_having_agg(
                    _reconstruct_agg_expr(hp.right_expr),
                    schema,
                    allowed_tables,
                    cte_outputs,
                    context,
                    "right_agg",
                    param_key,
                )
            )
        if hp.op not in VALID_HAVING_OPS:
            issues.append(
                make_intent_issue(
                    issue_id=f"having_invalid_op_{context}_{hp.op}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"Invalid HAVING operator '{hp.op}' in {context}",
                    context={
                        "operator": hp.op,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
        if not hp.right_expr:
            if hp.value_type not in VALID_VALUE_TYPES:
                issues.append(
                    make_intent_issue(
                        issue_id=f"having_invalid_value_type_{context}_{hp.value_type}",
                        category=FailureCategory.HAVING_VALIDITY,
                        severity="error",
                        message=f"Invalid HAVING value_type '{hp.value_type}' in {context}",
                        context={
                            "value_type": hp.value_type,
                            "param_key": param_key,
                            "location": context,
                        },
                    )
                )
            if hp.op not in ("is null", "is not null") and hp.resolved_value(param_values) is None:
                issues.append(
                    make_intent_issue(
                        issue_id=f"having_missing_value_{context}_{param_key}",
                        category=FailureCategory.HAVING_VALIDITY,
                        severity="error",
                        message=(
                            f"HAVING parameter '{param_key}' has no comparison value in {context}. "
                            "Provide a literal or bind ``param_key`` in param_values."
                        ),
                        context={
                            "param_key": param_key,
                            "op": hp.op,
                            "location": context,
                        },
                    )
                )
        hp_left_scalar, _ = extract_functions_from_term(hp.left_expr.primary_term)
        hp_right_scalar, _ = extract_functions_from_term(hp.right_expr.primary_term) if hp.right_expr else (None, None)
        hp_left_meta = _resolve_col_meta_from_expr(hp.left_expr.primary_column, schema)
        hp_right_meta = _resolve_col_meta_from_expr(hp.right_expr.primary_column, schema) if hp.right_expr else None
        issues.extend(
            _validate_scalar_func_valid(
                hp_left_scalar,
                f"having_{param_key}_left",
                context,
                col_meta=hp_left_meta,
            )
        )
        issues.extend(
            _validate_scalar_func_valid(
                hp_right_scalar,
                f"having_{param_key}_right",
                context,
                col_meta=hp_right_meta,
            )
        )
        issues.extend(validate_expr_no_extract_epoch(hp.left_expr, f"having_{param_key}_left", context))
        if hp.right_expr:
            issues.extend(validate_expr_no_extract_epoch(hp.right_expr, f"having_{param_key}_right", context))
        if hp.bool_op not in ("AND", "OR"):
            issues.append(
                make_intent_issue(
                    issue_id=f"having_invalid_bool_op_{context}_{hp.bool_op}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"Invalid HAVING bool_op '{hp.bool_op}' in {context}. Must be 'AND' or 'OR'.",
                    context={
                        "bool_op": hp.bool_op,
                        "param_key": param_key,
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_having_schema] {len(issues)} issues in {context}")
    return issues


def validate_filter_ops_per_column(
    filters_param: list[FilterParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate filter operators against each column's data type and role.

    Args:

        filters_param: `FilterParam` instances to validate.

        schema: Schema graph for base tables.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues = []
    if not filters_param:
        return []
    cte_outputs = cte_outputs or {}
    for fp in filters_param:
        col_expr = fp.left_expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                valid_ops = cte_meta.get_valid_filter_ops()
                if valid_ops and fp.op not in valid_ops:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"filter_op_invalid_for_cte_{context}_{actual_col}_{fp.op}",
                            category=FailureCategory.FILTER_VALIDITY,
                            severity="error",
                            message=f"Operator '{fp.op}' not valid for CTE column '{actual_col}' (role={cte_meta.role}, type={cte_meta.data_type}) in {context}. Valid: {valid_ops}",
                            context={
                                "column": actual_col,
                                "operator": fp.op,
                                "role": cte_meta.role,
                                "data_type": cte_meta.data_type,
                                "valid_ops": valid_ops,
                                "location": context,
                            },
                        )
                    )
                if not cte_meta.filterable and fp.op not in ("is null", "is not null"):
                    issues.append(
                        make_intent_issue(
                            issue_id=f"filter_cte_col_not_filterable_{context}_{actual_col}",
                            category=FailureCategory.FILTER_VALIDITY,
                            severity="warning",
                            message=f"CTE column '{actual_col}' (role={cte_meta.role}) is not recommended for filtering in {context}",
                            context={
                                "column": actual_col,
                                "role": cte_meta.role,
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
        valid_ops = col_meta.get_valid_filter_ops()
        if fp.op not in valid_ops:
            issues.append(
                make_intent_issue(
                    issue_id=f"filter_op_invalid_for_type_{context}_{actual_col}_{fp.op}",
                    category=FailureCategory.FILTER_VALIDITY,
                    severity="error",
                    message=f"Operator '{fp.op}' not valid for column '{actual_col}' (role={col_meta.role}, type={col_meta.data_type}) in {context}. Valid: {valid_ops}",
                    context={
                        "column": actual_col,
                        "operator": fp.op,
                        "role": col_meta.role,
                        "data_type": col_meta.data_type,
                        "valid_ops": valid_ops,
                        "location": context,
                    },
                )
            )
        if not col_meta.is_filterable and fp.op not in ("is null", "is not null"):
            issues.append(
                make_intent_issue(
                    issue_id=f"filter_col_not_filterable_{context}_{actual_col}",
                    category=FailureCategory.FILTER_VALIDITY,
                    severity="warning",
                    message=f"Column '{actual_col}' (role={col_meta.role}) is not recommended for filtering in {context}",
                    context={
                        "column": actual_col,
                        "role": col_meta.role,
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_filter_ops_per_column] {len(issues)} issues in {context}")
    return issues


def validate_having_ops_per_column(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate HAVING operators against each column's type and role.

    Args:

        having_param: `HavingParam` instances to validate.

        schema: Schema graph for base tables.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue IDs and messages.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues: list[IntentIssue] = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:

        def _check_expr(term: str, _hp: Any = hp) -> None:
            """
            Append issues if `_hp.op` is invalid for columns referenced in `term`.

            Args:

                term: Aggregation or column term from a HAVING side.

                _hp: `HavingParam` bound from the enclosing loop.

            Returns:

                None.
            """
            result = extract_agg_col(term)
            if len(result) == 3 and result[0] and result[1] and "." in result[1]:
                _, actual_target, _ = result
                table_name, col_name = actual_target.rsplit(".", 1)
            else:
                col = extract_col_from_scalar_wrapper(term)
                if "." not in col:
                    return
                table_name, col_name = col.rsplit(".", 1)
            if table_name in cte_outputs:
                cte_cols = cte_outputs[table_name]
                cte_meta = cte_cols.get(col_name) or cte_cols.get(col_name.lower())
                if cte_meta:
                    valid_ops = cte_meta.get_valid_having_ops()
                    if valid_ops and _hp.op not in valid_ops:
                        actual_col = f"{table_name}.{col_name}"
                        issues.append(
                            make_intent_issue(
                                issue_id=f"having_op_invalid_for_cte_{context}_{actual_col}_{_hp.op}",
                                category=FailureCategory.HAVING_VALIDITY,
                                severity="error",
                                message=f"Operator '{_hp.op}' not valid for CTE column '{actual_col}' (role={cte_meta.role}, type={cte_meta.data_type}) in {context}. Valid: {valid_ops}",
                                context={
                                    "column": actual_col,
                                    "operator": _hp.op,
                                    "role": cte_meta.role,
                                    "data_type": cte_meta.data_type,
                                    "valid_ops": valid_ops,
                                    "location": context,
                                },
                            )
                        )
            elif table_name in schema.tables:
                tbl = schema.tables[table_name]
                col_meta = tbl.columns.get(col_name) or tbl.columns.get(col_name.lower())
                if col_meta:
                    valid_ops = col_meta.get_valid_having_ops()
                    if valid_ops and _hp.op not in valid_ops:
                        actual_col = f"{table_name}.{col_name}"
                        issues.append(
                            make_intent_issue(
                                issue_id=f"having_op_invalid_for_column_{context}_{actual_col}_{_hp.op}",
                                category=FailureCategory.HAVING_VALIDITY,
                                severity="error",
                                message=f"Operator '{_hp.op}' not valid for column '{actual_col}' (role={col_meta.role}, type={col_meta.data_type}) in {context}. Valid: {valid_ops}",
                                context={
                                    "column": actual_col,
                                    "operator": _hp.op,
                                    "role": col_meta.role,
                                    "data_type": col_meta.data_type,
                                    "valid_ops": valid_ops,
                                    "location": context,
                                },
                            )
                        )

        _check_expr(hp.left_expr.primary_term)
        if hp.right_expr:
            _check_expr(hp.right_expr.primary_term)
    if issues:
        debug(f"[validation_schema.validate_having_ops_per_column] {len(issues)} issues in {context}")
    return issues


def validate_date_window_units(
    filters_param: list[FilterParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    context: str = "main",
    *,
    scope_param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """
    Validate `date_window` filters use an allowed relative-date unit.

    Args:

        filters_param: Main-query `FilterParam` list.

        cte_steps: Optional CTE steps whose filters are also checked.

        context: Label for issue IDs and messages.

        scope_param_values: Bound literals for ``filters_param`` when ``raw_value`` was hoisted.

    Returns:

        `IntentIssue` instances for invalid `date_window` units.
    """
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check(fp: FilterParam, loc: str, pv: Mapping[str, Any] | None) -> None:
        """
        Record an issue when `fp` is a `date_window` with a bad unit.

        Args:

            fp: Filter to inspect.

            loc: Location label for the message.

            pv: Parameter map for resolving hoisted window payloads.

        Returns:

            None.
        """
        if fp.value_type != "date_window":
            return
        rv = fp.resolved_value(pv)
        if not isinstance(rv, dict):
            return
        unit = rv.get("unit")
        if unit is None:
            return
        if unit not in VALID_RELATIVE_DATE_UNITS:
            col = fp.left_expr.primary_column
            issues.append(
                make_intent_issue(
                    issue_id=f"date_window_invalid_unit_{context}_{col}",
                    category=FailureCategory.FILTER_VALIDITY,
                    severity="error",
                    message=(
                        f"{loc}: date_window filter on '{col}' has invalid unit '{unit}'. "
                        f"Valid: {sorted(VALID_RELATIVE_DATE_UNITS)}"
                    ),
                    context={
                        "column": col,
                        "unit": unit,
                        "valid_units": sorted(VALID_RELATIVE_DATE_UNITS),
                        "location": context,
                    },
                )
            )

    for fp in filters_param:
        check(fp, f"{context} filter", scope_param_values)
    for cte in cte_steps:
        for fp in cte.filters_param or []:
            check(fp, f"CTE '{cte.cte_name}' filter", cte.param_values)

    if issues:
        debug(f"[validation_schema.validate_date_window_units] {len(issues)} invalid units")
    return issues


def validate_date_diff_units(
    filters_param: list[FilterParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    context: str = "main",
    *,
    scope_param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """
    Validate `date_diff` filters for allowed units and numeric amounts.

    Args:

        filters_param: Main-query `FilterParam` list.

        cte_steps: Optional CTE steps whose filters are also checked.

        context: Label for issue IDs and messages.

        scope_param_values: Bound literals for ``filters_param`` when ``raw_value`` was hoisted.

    Returns:

        `IntentIssue` instances for invalid `date_diff` configuration.
    """
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check(fp: FilterParam, loc: str, pv: Mapping[str, Any] | None) -> None:
        """
        Record issues for invalid `date_diff` unit or non-numeric amount.

        Args:

            fp: Filter to inspect.

            loc: Location label for the message.

            pv: Parameter map for resolving hoisted diff payloads.

        Returns:

            None.
        """
        if fp.value_type != "date_diff":
            return
        rv = fp.resolved_value(pv)
        if not isinstance(rv, dict):
            return
        unit = rv.get("unit")
        amount = rv.get("amount")
        if unit is not None and unit not in VALID_RELATIVE_DATE_UNITS:
            col = fp.left_expr.primary_column or fp.left_expr.primary_term or fp.param_key or "expr"
            issues.append(
                make_intent_issue(
                    issue_id=f"date_diff_invalid_unit_{context}_{col}",
                    category=FailureCategory.DATE_DIFF,
                    severity="error",
                    message=(
                        f"{loc}: date_diff filter has invalid unit '{unit}'. Valid: {sorted(VALID_RELATIVE_DATE_UNITS)}"
                    ),
                    context={
                        "column": col,
                        "unit": unit,
                        "valid_units": sorted(VALID_RELATIVE_DATE_UNITS),
                        "location": context,
                    },
                )
            )
        if amount is not None and not isinstance(amount, (int, float)):
            try:
                int(amount)
            except (TypeError, ValueError):
                col = fp.left_expr.primary_column or fp.left_expr.primary_term or fp.param_key or "expr"
                issues.append(
                    make_intent_issue(
                        issue_id=f"date_diff_invalid_amount_{context}_{col}",
                        category=FailureCategory.DATE_DIFF,
                        severity="error",
                        message=f"{loc}: date_diff filter has non-numeric amount '{amount}'",
                        context={"column": col, "amount": amount, "location": context},
                    )
                )

    for fp in filters_param:
        check(fp, f"{context} filter", scope_param_values)
    for cte in cte_steps:
        for fp in cte.filters_param or []:
            check(fp, f"CTE '{cte.cte_name}' filter", cte.param_values)

    if issues:
        debug(f"[validation_schema.validate_date_diff_units] {len(issues)} invalid configs")
    return issues


def validate_date_diff_left_expr_is_subtraction(
    filters_param: list[FilterParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Reject ``date_diff`` filters whose ``left_expr`` is not a subtraction.

    A ``date_diff`` filter must compare a duration to a numeric amount, so its ``left_expr`` is required to be a subtraction such as ``end_date - start_date`` or ``end_date - INTERVAL '1 day'``. A plain column reference is not a subtraction; the deterministic repair pipeline rewrites that case to ``date_window``. Anything that survives both repair and rewrite without containing ``sub_groups`` or ``sub_values`` is structurally invalid and is reported with :attr:`FailureCategory.DATE_DIFF`.
    """

    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check(fp: FilterParam, loc: str) -> None:
        """Append an issue when *fp* is a ``date_diff`` whose ``left_expr`` lacks subtraction."""

        if fp.value_type != "date_diff":
            return
        expr = fp.left_expr
        if expr.sub_groups or expr.sub_values:
            return
        col = expr.primary_column or expr.primary_term or fp.param_key or "expr"
        issues.append(
            make_intent_issue(
                issue_id=f"date_diff_left_expr_not_subtraction_{context}_{col}",
                category=FailureCategory.DATE_DIFF,
                severity="error",
                message=(
                    f"{loc}: date_diff filter on '{col}' must have left_expr as a subtraction "
                    "(e.g. end_date - start_date or end_date - INTERVAL '1 day'); use date_window "
                    "for relative date-window filters on a single date column"
                ),
                context={"column": col, "location": context},
            )
        )

    for fp in filters_param:
        check(fp, f"{context} filter")
    for cte in cte_steps:
        for fp in cte.filters_param or []:
            check(fp, f"CTE '{cte.cte_name}' filter")

    if issues:
        debug(f"[validation_schema.validate_date_diff_left_expr_is_subtraction] {len(issues)} invalid configs")
    return issues


def validate_null_filters(
    filters_param: list[FilterParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """
    Validate `IS NULL` / `IS NOT NULL` filters use `value_type` `null` or empty.

    Args:

        filters_param: Main-query `FilterParam` list.

        cte_steps: Optional CTE steps whose filters are also checked.

        context: Label for issue IDs and messages.

    Returns:

        `IntentIssue` instances when `value_type` disagrees with null semantics.
    """
    issues = []
    cte_steps = cte_steps or []

    def check_filter(fp: FilterParam, loc: str) -> IntentIssue | None:
        """
        Return an issue if a null filter has a non-null `value_type`.

        Args:

            fp: Filter to inspect.

            loc: Location label for the message.

        Returns:

            An `IntentIssue` or `None`.
        """
        if fp.op in ("is null", "is not null"):
            if fp.value_type and fp.value_type != "null":
                col = fp.left_expr.primary_column
                return make_intent_issue(
                    issue_id=f"null_filter_wrong_value_type_{col}",
                    category=FailureCategory.FILTER_STRUCTURE,
                    severity="error",
                    message=f"{loc}: IS NULL filter on '{col}' should have value_type='null' or empty, got '{fp.value_type}'",
                    context={
                        "column": col,
                        "op": fp.op,
                        "expected_value_type": "null",
                        "actual_value_type": fp.value_type,
                    },
                )
        return None

    for fp in filters_param:
        issue = check_filter(fp, f"{context} filter")
        if issue:
            issues.append(issue)

    for cte in cte_steps:
        for fp in cte.filters_param or []:
            issue = check_filter(fp, f"CTE '{cte.cte_name}' filter")
            if issue:
                issues.append(issue)

    if issues:
        debug(f"[validation_schema.validate_null_filters] FAILED with {len(issues)} issues")
    else:
        debug("[validation_schema.validate_null_filters] PASSED")
    return issues


def validate_filter_value_type_alignment(
    filters_param: list[FilterParam],
    schema: SchemaGraph,
    context: str = "main",
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """
    Warn when string or enum filter values target numeric FK or CTE columns.

    Args:

        filters_param: `FilterParam` instances to scan.

        schema: Schema graph for base tables.

        context: Label for issue IDs and messages.

        cte_outputs: CTE name to output column metadata.

        param_values: Bound literals for this scope when ``raw_value`` was hoisted.

    Returns:

        Collected `IntentIssue` instances (warnings only when applicable).
    """
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for fp in filters_param:
        if fp.value_type not in {"string", "enum"}:
            continue
        rv = fp.resolved_value(param_values)
        if rv is None:
            continue
        col = fp.left_expr.primary_column
        parts = col.split(".", 1) if "." in col else None
        if not parts:
            continue
        table_name, col_name = parts
        col_meta = schema.get_column(table_name, col_name)
        if col_meta:
            if col_meta.is_foreign_key and col_meta.value_type in {"integer", "number"}:
                issues.append(
                    make_intent_issue(
                        issue_id=f"filter_string_on_fk_int_{table_name}_{col_name}_{context}",
                        category=FailureCategory.TYPE_ALIGNMENT,
                        severity="warning",
                        message=f"Filter on {col} uses string value '{rv}' but column is a numeric FK in {context}. Filter should target the FK target table's descriptive column.",
                        context={
                            "column": col,
                            "value": str(rv),
                            "value_type": fp.value_type,
                            "column_type": col_meta.value_type,
                            "location": context,
                        },
                    )
                )
                debug(f"[validation_schema.validate_filter_value_type_alignment] string value on FK int column {col}")
            continue
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            cte_meta = cte_cols.get(col_name) or cte_cols.get(col_name.lower())
            if cte_meta and cte_meta.value_type in {"integer", "number"}:
                issues.append(
                    make_intent_issue(
                        issue_id=f"filter_string_on_cte_numeric_{table_name}_{col_name}_{context}",
                        category=FailureCategory.TYPE_ALIGNMENT,
                        severity="warning",
                        message=f"Filter on {col} uses string value '{rv}' but CTE column is numeric ({cte_meta.value_type}) in {context}.",
                        context={
                            "column": col,
                            "value": str(rv),
                            "value_type": fp.value_type,
                            "column_type": cte_meta.value_type,
                            "location": context,
                        },
                    )
                )
    return issues


def validate_no_between_ops(
    filters_param: list[FilterParam],
    having_param: list[HavingParam],
    context: str = "main query",
) -> list[IntentIssue]:
    """
    Flag surviving `BETWEEN` operators that should have been decomposed.

    Args:

        filters_param: Filter conditions to inspect.

        having_param: HAVING conditions to inspect.

        context: Query scope description for issue messages.

    Returns:

        One `IntentIssue` per remaining `BETWEEN` operator.
    """
    issues: list[IntentIssue] = []
    for fp in filters_param:
        if fp.op.lower() == "between":
            col = fp.left_expr.primary_column
            issues.append(
                make_intent_issue(
                    issue_id=f"filter_between_not_decomposed_{col}_{context}",
                    category=FailureCategory.OPERATOR,
                    severity="error",
                    message=(
                        f"Filter on {col} still uses BETWEEN in {context}. "
                        "Decompose into separate >= and <= conditions."
                    ),
                    context={"column": col, "op": fp.op, "location": context},
                )
            )
    for hp in having_param:
        if hp.op.lower() == "between":
            col = hp.left_expr.primary_column
            issues.append(
                make_intent_issue(
                    issue_id=f"having_between_not_decomposed_{col}_{context}",
                    category=FailureCategory.OPERATOR,
                    severity="error",
                    message=(
                        f"Having on {col} still uses BETWEEN in {context}. "
                        "Decompose into separate >= and <= conditions."
                    ),
                    context={"column": col, "op": hp.op, "location": context},
                )
            )
    if issues:
        debug(f"[validation_schema.validate_no_between_ops] FAILED with {len(issues)} issues in {context}")
    return issues


def _refs_from_filter_param(fp: FilterParam) -> list[str]:
    """
    Collect qualified column references from both sides of a filter.

    Args:

        fp: `FilterParam` whose expressions are scanned.

    Returns:

        List of column reference strings.
    """
    refs = extract_columns_from_expr(fp.left_expr)
    if fp.right_expr:
        refs.extend(extract_columns_from_expr(fp.right_expr))
    return refs


def _refs_from_having_param(hp: HavingParam) -> list[str]:
    """
    Collect qualified column references from both sides of a HAVING clause.

    Args:

        hp: `HavingParam` whose expressions are scanned.

    Returns:

        List of column reference strings.
    """
    refs = extract_columns_from_expr(hp.left_expr)
    if hp.right_expr:
        refs.extend(extract_columns_from_expr(hp.right_expr))
    return refs


def _refs_from_select_col_extended(
    sc: SelectCol,
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> list[str]:
    """
    Collect column refs from a select column, window spec, and CASE.

    Args:

        sc: `SelectCol` including resolved registry payloads via ``effective_select_parts``.

        window_registry: Optional window registry for resolving ``registry_ref``.

        case_registry: Optional case registry for resolving ``registry_ref``.

    Returns:

        List of column reference strings.
    """

    parts = effective_select_parts(sc, window_registry, case_registry)
    refs = extract_columns_from_expr(parts.expr)
    if parts.window_spec:
        ws = parts.window_spec
        for e in ws.partition_by:
            refs.extend(extract_columns_from_expr(e))
        for o in ws.order_by:
            refs.extend(extract_columns_from_expr(o.expr))
        if ws.argument:
            refs.extend(extract_columns_from_expr(ws.argument))
    if parts.case_when:
        for br in parts.case_when.branches:
            refs.extend(_refs_from_filter_param(br.condition))
            refs.extend(extract_columns_from_expr(br.result))
        if parts.case_when.else_result:
            refs.extend(extract_columns_from_expr(parts.case_when.else_result))
    return refs


def collect_column_refs_for_access_policy(
    select_cols: list[SelectCol],
    group_by_cols: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    filters_param: list[FilterParam],
    having_param: list[HavingParam],
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> list[str]:
    """
    Collect qualified column references for deny-column policy checks.

    Args:

        select_cols: SELECT list (may be empty).

        group_by_cols: GROUP BY expressions (may be empty).

        order_by_cols: ORDER BY columns (may be empty).

        filters_param: Filter list (may be empty).

        having_param: HAVING list (may be empty).

        window_registry: Optional window registry for resolving select ``registry_ref`` values.

        case_registry: Optional case registry for resolving select ``registry_ref`` values.

    Returns:

        Every column reference found across the intent parts, in traversal order.
    """
    refs: list[str] = []
    for sc in select_cols or []:
        refs.extend(_refs_from_select_col_extended(sc, window_registry=window_registry, case_registry=case_registry))
    for g in group_by_cols or []:
        refs.extend(extract_columns_from_expr(g))
    for ob in order_by_cols or []:
        refs.extend(extract_columns_from_expr(ob.expr))
    for fp in filters_param or []:
        refs.extend(_refs_from_filter_param(fp))
    for hp in having_param or []:
        refs.extend(_refs_from_having_param(hp))
    return refs


def validate_contains_array_filters(
    filters_param: list[FilterParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None,
    context: str,
) -> list[IntentIssue]:
    """
    Ensure `contains` is only used on array columns with `element_type` set.

    Args:

        filters_param: Filters to scan for `contains`.

        schema: Schema graph for base tables.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue messages.

    Returns:

        `IntentIssue` instances when `contains` targets a non-array column.
    """
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for i, fp in enumerate(filters_param or []):
        if fp.op != "contains":
            continue
        cols = extract_columns_from_expr(fp.left_expr)
        if len(cols) != 1:
            continue
        meta = get_col_meta(cols[0], schema, cte_outputs)
        if meta is None:
            continue
        if not getattr(meta, "element_type", None):
            issues.append(
                make_intent_issue(
                    issue_id=f"contains_non_array_{context}_{i}",
                    category=FailureCategory.FILTER_SEMANTIC,
                    severity="error",
                    message=f"{context}: operator 'contains' requires an array column; '{cols[0]}' is not array-typed",
                    context={"column": cols[0], "location": context},
                )
            )
    return issues


def validate_window_spec_schema(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None,
    context: str,
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> list[IntentIssue]:
    """
    Validate window function specifications on `SelectCol` entries.

    Args:

        select_cols: SELECT list entries that may carry `window_spec`.

        schema: Schema graph for base tables.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue messages.

    Returns:

        Collected `IntentIssue` instances.
    """
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for idx, sc in enumerate(select_cols or []):
        parts = effective_select_parts(sc, window_registry, case_registry)
        ws = parts.window_spec
        if ws is None:
            continue
        if ws.function not in VALID_WINDOW_FUNCTIONS:
            issues.append(
                make_intent_issue(
                    issue_id=f"window_invalid_fn_{context}_{idx}",
                    category=FailureCategory.SCHEMA,
                    severity="error",
                    message=f"{context}: invalid window function '{ws.function}'",
                    context={"function": ws.function, "location": context},
                )
            )
            continue
        if ws.function in WINDOW_RANKING_FUNCTIONS and not ws.order_by:
            issues.append(
                make_intent_issue(
                    issue_id=f"window_ranking_no_order_{context}_{idx}",
                    category=FailureCategory.SCHEMA,
                    severity="error",
                    message=f"{context}: ranking window '{ws.function}' requires order_by",
                    context={"function": ws.function, "location": context},
                )
            )
        if ws.function in WINDOW_OFFSET_FUNCTIONS:
            if not ws.order_by:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_offset_no_order_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=f"{context}: window '{ws.function}' requires order_by",
                        context={"function": ws.function, "location": context},
                    )
                )
            if ws.argument is None:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_offset_no_arg_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=f"{context}: window '{ws.function}' requires an argument expression",
                        context={"function": ws.function, "location": context},
                    )
                )
        if ws.function in WINDOW_VALUE_FUNCTIONS:
            if not ws.order_by:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_value_no_order_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=f"{context}: window '{ws.function}' requires order_by",
                        context={"function": ws.function, "location": context},
                    )
                )
            if ws.argument is None:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_value_no_arg_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=f"{context}: window '{ws.function}' requires an argument expression",
                        context={"function": ws.function, "location": context},
                    )
                )
        if ws.function in WINDOW_AGG_FUNCTIONS:
            if ws.argument is None:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_agg_no_arg_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=f"{context}: window '{ws.function}' requires an argument expression",
                        context={"function": ws.function, "location": context},
                    )
                )
            else:
                for cref in extract_columns_from_expr(ws.argument):
                    num = is_col_numeric(cref, schema, cte_outputs)
                    if num is False:
                        issues.append(
                            make_intent_issue(
                                issue_id=f"window_agg_non_numeric_{context}_{idx}_{cref}",
                                category=FailureCategory.SCHEMA,
                                severity="error",
                                message=f"{context}: window '{ws.function}' argument must be numeric; '{cref}' is not",
                                context={"column": cref, "location": context},
                            )
                        )
        for pe in ws.partition_by:
            for cref in extract_columns_from_expr(pe):
                meta = get_col_meta(cref, schema, cte_outputs)
                if meta is not None and not meta.is_groupable:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"window_partition_not_groupable_{context}_{idx}_{cref}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: PARTITION BY column '{cref}' is not groupable",
                            context={"column": cref, "location": context},
                        )
                    )
        if ws.frame_kind not in ("none", "rows", "range"):
            issues.append(
                make_intent_issue(
                    issue_id=f"window_bad_frame_kind_{context}_{idx}",
                    category=FailureCategory.SCHEMA,
                    severity="error",
                    message=f"{context}: invalid window frame_kind '{ws.frame_kind}'",
                    context={"location": context},
                )
            )
        elif ws.frame_kind != "none":
            if not ws.order_by:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_frame_no_order_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=f"{context}: {ws.frame_kind.upper()} frame requires order_by",
                        context={"location": context},
                    )
                )
            fs = " ".join(str(ws.frame_start or "").split()).strip().lower()
            fe = " ".join(str(ws.frame_end or "").split()).strip().lower()
            for label, bound, off in (
                ("frame_start", fs, ws.frame_start_offset),
                ("frame_end", fe, ws.frame_end_offset),
            ):
                if bound not in WINDOW_FRAME_BOUNDS:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"window_bad_{label}_{context}_{idx}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: window {label} '{bound}' is not an allowed frame bound",
                            context={"location": context},
                        )
                    )
                elif bound in ("n_preceding", "n_following") and (off is None or off < 0):
                    issues.append(
                        make_intent_issue(
                            issue_id=f"window_frame_offset_{context}_{idx}_{label}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: {label} uses {bound} and requires a non-negative integer offset",
                            context={"location": context},
                        )
                    )
    return issues


def filter_param_to_having_param(fp: FilterParam) -> HavingParam:
    """
    Convert a `FilterParam` into a `HavingParam` carrying the same predicate fields.

    Used when a `CaseWhenBranch` declares ``condition_scope == "having"`` so that
    HAVING-shaped validators can be applied to its filter-shaped condition.

    Args:

        fp: Source filter parameter to translate.

    Returns:

        A ``HavingParam`` with matching `left_expr`, `op`, `right_expr`, `value_type`,
        `param_key`, `raw_value`, `bool_op`, and `filter_group` fields.
    """
    return HavingParam(
        left_expr=fp.left_expr,
        op=fp.op,
        right_expr=fp.right_expr,
        value_type=fp.value_type,
        param_key=fp.param_key,
        raw_value=fp.raw_value,
        bool_op=fp.bool_op,
        filter_group=fp.filter_group,
    )


def iterate_case_branch_conditions(
    select_cols: list[SelectCol] | None,
    case_registry: list[CaseRegistryStep] | None,
    window_registry: list[WindowRegistryStep] | None,
    location_prefix: str,
) -> list[tuple[FilterParam, str, str]]:
    """
    Enumerate every CASE branch condition reachable from a query body.

    Collects conditions from ``case_registry`` entries referenced by bare ``cNN`` tokens in
    ``select_cols`` and from orphan registry rows not referenced by any select column.

    Args:

        select_cols: SELECT list entries that may reference CASE definitions via ``cNN``.

        case_registry: Standalone CASE registry steps for the same query body.

        window_registry: Window registry passed to ``effective_select_parts``.

        location_prefix: Scope label such as ``"main query"`` or
        ``"cte 'orders_agg'"``.

    Returns:

        A list of ``(FilterParam, str, str)`` tuples covering every branch in both
        the registry-driven layout.
    """
    out: list[tuple[FilterParam, str, str]] = []
    seen_registry_ids: set[str] = set()
    for _, sc in enumerate(select_cols or []):
        ref = expr_registry_ref(sc.expr) or "" if sc.expr is not None else ""
        if not ref.startswith("c"):
            continue
        parts = effective_select_parts(sc, window_registry, case_registry)
        cw = parts.case_when
        if cw is None:
            continue
        seen_registry_ids.add(ref)
        base_loc = f"{location_prefix} case_registry[{ref}]"
        scope = (cw.condition_scope or "filter").strip().lower() or "filter"
        for bi, br in enumerate(cw.branches or []):
            out.append((br.condition, scope, f"{base_loc}.branches[{bi}]"))
    for step in case_registry or []:
        if not step or step.case_when is None:
            continue
        rid = (step.registry_id or "").strip()
        if rid and rid in seen_registry_ids:
            continue
        cw = step.case_when
        scope = (cw.condition_scope or "filter").strip().lower() or "filter"
        base_loc = f"{location_prefix} case_registry[{rid or step.label or '?'}]"
        for bi, br in enumerate(cw.branches or []):
            out.append((br.condition, scope, f"{base_loc}.branches[{bi}]"))
    return out


def validate_case_when_schema(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None,
    context: str,
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
    param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """
    Validate CASE expressions for branch filters and result column references.

    Args:

        select_cols: SELECT list entries that reference CASE definitions via ``cNN`` when applicable.

        schema: Schema graph for base tables.

        allowed_tables: Table names permitted in this context.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue messages.

        param_values: Bound literals for the owning query body (branch filters share main or CTE scope).

    Returns:

        Collected `IntentIssue` instances.
    """
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for sc in select_cols or []:
        ref = expr_registry_ref(sc.expr) or ""
        cw = None
        if ref.startswith("c"):
            parts = effective_select_parts(sc, window_registry, case_registry)
            cw = parts.case_when
        if cw is None:
            continue
        for bi, br in enumerate(cw.branches):
            branch_scope = (cw.condition_scope or "filter").strip().lower() or "filter"
            if branch_scope == "having":
                issues.extend(
                    validate_having_schema(
                        [filter_param_to_having_param(br.condition)],
                        schema,
                        allowed_tables,
                        cte_outputs,
                        f"{context} case_when[{bi}]",
                        param_values=param_values,
                    )
                )
            else:
                issues.extend(
                    validate_filters_schema(
                        [br.condition],
                        schema,
                        allowed_tables,
                        cte_outputs,
                        f"{context} case_when[{bi}]",
                        param_values=param_values,
                    )
                )
            for cref in extract_columns_from_expr(br.result):
                ac = extract_col_from_scalar_wrapper(cref)
                if "." not in ac:
                    continue
                t_name, _c = ac.rsplit(".", 1)
                if t_name.lower() not in {x.lower() for x in allowed_tables} and t_name not in cte_outputs:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"case_result_bad_table_{context}_{bi}_{cref}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: CASE result references unknown table in '{cref}'",
                            context={"column": cref, "location": context},
                        )
                    )
                elif get_col_meta(cref, schema, cte_outputs) is None and t_name not in cte_outputs:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"case_result_unknown_col_{context}_{bi}_{cref}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: CASE result references unknown column '{cref}'",
                            context={"column": cref, "location": context},
                        )
                    )
        if cw.else_result:
            for cref in extract_columns_from_expr(cw.else_result):
                ac = extract_col_from_scalar_wrapper(cref)
                if "." not in ac:
                    continue
                t_name, _c = ac.rsplit(".", 1)
                if t_name.lower() not in {x.lower() for x in allowed_tables} and t_name not in cte_outputs:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"case_else_bad_table_{context}_{cref}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: CASE ELSE references unknown table in '{cref}'",
                            context={"column": cref, "location": context},
                        )
                    )
                elif get_col_meta(cref, schema, cte_outputs) is None and t_name not in cte_outputs:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"case_else_unknown_col_{context}_{cref}",
                            category=FailureCategory.SCHEMA,
                            severity="error",
                            message=f"{context}: CASE ELSE references unknown column '{cref}'",
                            context={"column": cref, "location": context},
                        )
                    )
    return issues


def _qualified_refs_under_aggregate_mulgroup(g: MulGroup) -> set[str]:
    """
    Collect qualified ``table.column`` refs that are direct arguments of an allowed aggregate.

    Args:

        g: Multiplicative group that may carry ``agg_func``.

    Returns:

        Qualified names for non-``COUNT(*)`` aggregate arguments, including ``COUNT(DISTINCT col)``.
    """

    af = (g.agg_func or "").lower()
    if af not in VALID_AGGREGATION_FUNCTIONS:
        return set()
    if af == "count" and _mul_group_is_count_star(g):
        return set()
    out: set[str] = set()
    wrapper = NormalizedExpr(add_groups=[g])
    for ref in extract_columns_from_expr(wrapper):
        cleaned = strip_leading_distinct_from_column_ref(ref.strip())
        if not cleaned or cleaned == "*":
            continue
        qc = extract_col_from_scalar_wrapper(cleaned)
        if "." in qc:
            out.add(qc)
    return out


def _mul_group_is_count_star(g: MulGroup) -> bool:
    """Return True when *g* is a row-count ``COUNT`` with no column reference."""

    if (g.agg_func or "").lower() != "count":
        return False
    if not g.multiply:
        return True
    if len(g.multiply) == 1:
        leaf = g.multiply[0]
        if leaf.star:
            return True
        if leaf.column_ref in {"*", "1"}:
            return True
        if leaf.add_values and not leaf.add_groups and not leaf.sub_groups and not leaf.sub_values:
            try:
                if len(leaf.add_values) == 1 and float(leaf.add_values[0].value) == 1.0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def selectability_exempt_qualified_refs(
    expr: NormalizedExpr,
    schema: SchemaGraph,
) -> set[str]:
    """
    Return qualified refs that appear only as arguments to allowed aggregate functions.

    Args:

        expr: SELECT-side expression for one column.

        schema: Base-table metadata (unused; kept for call-site stability).

    Returns:

        ``table.column`` names permitted inside ``COUNT`` (including ``DISTINCT``), ``SUM``, ``AVG``, ``MIN``, ``MAX``.
    """

    _ = schema
    exempt: set[str] = set()
    for g in expr.add_groups + expr.sub_groups:
        exempt |= _qualified_refs_under_aggregate_mulgroup(g)
    return exempt


def _selectability_issues_for_normalized_expr(
    expr: NormalizedExpr,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
    context: str,
    detail: str,
) -> list[IntentIssue]:
    """
    Build access-policy issues for bare non-selectable columns in one normalised expression.

    Args:

        expr: Expression to scan (SELECT fragment, partition, window argument, CASE branch).

        schema: Base tables.

        cte_outputs: CTE output column metadata for qualified names.

        context: Outer validation label.

        detail: Sublocation (for example ``select_cols[0]`` or ``window partition[1]``).

    Returns:

        ``IntentIssue`` list for disallowed bare sensitive columns.
    """

    issues: list[IntentIssue] = []
    col_refs = set(extract_columns_from_expr(expr))
    exempt = selectability_exempt_qualified_refs(expr, schema)
    seen_qc: set[str] = set()
    for cref in col_refs:
        qc = extract_col_from_scalar_wrapper(cref)
        if "." not in qc:
            continue
        if qc in exempt:
            continue
        if qc in seen_qc:
            continue
        seen_qc.add(qc)
        meta = get_col_meta(cref, schema, cte_outputs)
        if meta is not None and not getattr(meta, "is_selectable", True):
            loc = f"{context} {detail}".strip()
            issues.append(
                make_intent_issue(
                    issue_id=f"not_selectable_{loc.replace(' ', '_').replace('[', '_').replace(']', '_')}_{qc.replace('.', '_')}",
                    category=FailureCategory.ACCESS_POLICY,
                    severity="error",
                    message=f"{loc}: column '{cref}' is not selectable under sensitivity policy",
                    context={"column": cref, "location": loc},
                )
            )
    return issues


def validate_join_path_reachability_for_tables(
    tables: list[str],
    schema: SchemaGraph,
    context: str,
    *,
    extra_tables: set[str] | None = None,
) -> list[IntentIssue]:
    """Emit issues when two or more physical tables used together have no shortest-path join in ``join_paths_multi``."""

    combined: set[str] = set(tables or [])
    if extra_tables:
        combined |= extra_tables
    physical = {t for t in combined if t in schema.tables}
    if len(physical) <= 1:
        return []
    root = min(physical)
    jpm = schema.join_paths_multi or {}
    issues: list[IntentIssue] = []
    for t in physical:
        if t == root:
            continue
        fwd = (jpm.get(root) or {}).get(t) or []
        back = (jpm.get(t) or {}).get(root) or []
        if fwd or back:
            continue
        issues.append(
            make_intent_issue(
                issue_id=f"join_unreachable_{context.replace(' ', '_')}_{root}_{t}",
                category=FailureCategory.WRONG_JOIN,
                severity="error",
                message=(
                    f"{context}: no schema join path between '{root}' and '{t}' "
                    f"(disconnected FK groups; add a bridging foreign_keys_add)."
                ),
                context={"root": root, "target": t, "tables": sorted(physical)},
            )
        )
    return issues


def validate_join_path_reachability(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    context: str,
) -> list[IntentIssue]:
    """Like :func:`validate_join_path_reachability_for_tables` using ``intent.tables`` and ``intent.extra_tables``."""

    extra = getattr(intent, "extra_tables", None)
    et = set(extra) if extra else None
    return validate_join_path_reachability_for_tables(
        list(intent.tables or []),
        schema,
        context,
        extra_tables=et,
    )


def validate_selectability(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None,
    context: str,
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> list[IntentIssue]:
    """
    Reject bare non-selectable columns in SELECT expressions, window partitions, and CASE results.

    Args:

        select_cols: SELECT list entries to inspect.

        schema: Schema graph for base tables.

        cte_outputs: CTE name to output column metadata.

        context: Label for issue messages.

        window_registry: Registry used to resolve bare ``wNN`` tokens on select expressions.

        case_registry: Registry used to resolve bare ``cNN`` tokens on select expressions.

    Returns:

        `IntentIssue` instances for each disallowed bare sensitive column.
    """
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for idx, sc in enumerate(select_cols or []):
        detail = f"select_cols[{idx}]"
        parts = effective_select_parts(sc, window_registry, case_registry)
        issues.extend(_selectability_issues_for_normalized_expr(parts.expr, schema, cte_outputs, context, detail))
        if parts.window_spec:
            ws = parts.window_spec
            for pi, pe in enumerate(ws.partition_by):
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        pe,
                        schema,
                        cte_outputs,
                        context,
                        f"{detail} window partition[{pi}]",
                    )
                )
            if ws.argument is not None:
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        ws.argument,
                        schema,
                        cte_outputs,
                        context,
                        f"{detail} window argument",
                    )
                )
        if parts.case_when:
            cw = parts.case_when
            for bi, br in enumerate(cw.branches):
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        br.result,
                        schema,
                        cte_outputs,
                        context,
                        f"{detail} case_when[{bi}]",
                    )
                )
            if cw.else_result is not None:
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        cw.else_result,
                        schema,
                        cte_outputs,
                        context,
                        f"{detail} case_else",
                    )
                )
    return issues


def get_col_type(
    col_expr: str,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> str | None:
    """
    Resolve a column's `value_type` from the schema or CTE outputs.

    Args:

        col_expr: Column reference, optionally wrapped in scalar calls.

        schema: Schema graph with table and column metadata.

        cte_outputs: CTE name to column output metadata.

    Returns:

        The `value_type` string, or `None` if the column cannot be resolved.
    """
    actual_col = extract_col_from_scalar_wrapper(col_expr)
    if "." not in actual_col:
        return None
    table_name, col_name = actual_col.rsplit(".", 1)
    if table_name in cte_outputs:
        meta = cte_outputs[table_name].get(col_name) or cte_outputs[table_name].get(col_name.lower())
        return meta.value_type if meta else None
    if table_name not in schema.tables:
        return None
    table_meta = schema.tables[table_name]
    col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
    if not col_meta:
        return None
    return col_meta.value_type


def get_col_meta(
    col_expr: str,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> Any | None:
    """
    Resolve `ColumnMetadata` from the schema graph or synthesise it from CTE outputs.

    Args:

        col_expr: Column reference, optionally wrapped in scalar calls.

        schema: Schema graph with table and column metadata.

        cte_outputs: CTE name to column output metadata.

    Returns:

        A `ColumnMetadata` instance (real or synthetic), or `None` if unresolved.
    """
    actual_col = extract_col_from_scalar_wrapper(col_expr)
    if "." not in actual_col:
        return None
    table_name, col_name = actual_col.rsplit(".", 1)
    if table_name in cte_outputs:
        cte_meta = cte_outputs[table_name].get(col_name) or cte_outputs[table_name].get(col_name.lower())
        if not cte_meta:
            return None
        return ColumnMetadata(
            name=col_name,
            data_type=cte_meta.data_type or "unknown",
            role=cte_meta.role,
            is_filterable_override=cte_meta.filterable,
            is_aggregatable_override=cte_meta.aggregatable,
            is_groupable_override=cte_meta.groupable,
            valid_filter_ops=list(cte_meta.valid_filter_ops or []),
            valid_aggregations=list(cte_meta.valid_aggregations or []),
            valid_having_ops=list(cte_meta.valid_having_ops or []),
            sensitivity=cte_meta.sensitivity,
        )
    if table_name not in schema.tables:
        return None
    table_meta = schema.tables[table_name]
    return table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())


def is_col_numeric(
    col_ref: str,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> bool | None:
    """
    Return whether a column's value type is numeric.

    Args:

        col_ref: Qualified reference `table.column`.

        schema: Schema graph with column type information.

        cte_outputs: CTE name to column output metadata.

    Returns:

        `True` if numeric, `False` if known non-numeric, or `None` if unresolved.
    """
    col_type = get_col_type(col_ref, schema, cte_outputs)
    if col_type is None:
        return None
    return col_type in ("integer", "number")


def is_col_arithmetic_role(
    col_ref: str,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> bool | None:
    """
    Return whether a column's role allows use in arithmetic expressions.

    Args:

        col_ref: Qualified reference `table.column`.

        schema: Schema graph with column role information.

        cte_outputs: CTE name to column output metadata.

    Returns:

        `True` or `False` from role membership in `ARITHMETIC_ROLES`, or `None` if unresolved.
    """
    meta = get_col_meta(col_ref, schema, cte_outputs)
    if meta and meta.role:
        return meta.role in ARITHMETIC_ROLES
    actual_col = extract_col_from_scalar_wrapper(col_ref)
    if "." in actual_col:
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_meta = cte_outputs[table_name].get(col_name) or cte_outputs[table_name].get(col_name.lower())
            if cte_meta and cte_meta.role:
                return cte_meta.role in ARITHMETIC_ROLES
    return None
