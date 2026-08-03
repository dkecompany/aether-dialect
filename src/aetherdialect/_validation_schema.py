"""Validate intents against the schema for columns, operators, aggregates, scalars, and CTE-aware checks."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from ._config import PolicyConfig
from ._refusal_diagnostics import emit_session_refusal_diagnostic, refusal_diagnostic_code_for_intent_issue
from ._constants import (
    AGGREGATION_ALLOWED_COLUMN_TYPES,
    ANTI_JOIN_PRESENCE_COLUMN_SUFFIX,
    ARITHMETIC_ROLES,
    DATE_FRIENDLY_VALUE_TYPES,
    DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR,
    DISALLOWED_EXTRACT_UNITS,
    FAN_OUT_SENSITIVE_AGG_FUNCS,
    JOIN_PATH_EDGE_KIND_WHERE_BUCKET,
    MAX_PREDICATE_NESTING_DEPTH,
    NUMERIC_RESULT_AGGS,
    NUMERIC_RESULT_SCALARS,
    REGISTRY_CASE_ID_RE,
    REGISTRY_WINDOW_ID_RE,
    SCALAR_FUNCTIONS_TEMPORAL,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_HAVING_OPS,
    VALID_RELATIVE_DATE_UNITS,
    VALID_SCALAR_FUNCTIONS,
    VALID_VALUE_TYPES,
    VALID_WHERE_OPS,
    VALID_WINDOW_FUNCTIONS,
    WINDOW_AGG_FUNCTIONS,
    WINDOW_FRAME_BOUNDS,
    WINDOW_FUNCTIONS_WITHOUT_COLUMN_ARG,
    WINDOW_NUMERIC_ARG_FUNCTIONS,
    WINDOW_OFFSET_FUNCTIONS,
    WINDOW_RANKING_FUNCTIONS,
    WINDOW_VALUE_FUNCTIONS,
    YEAR_LITERAL_COMPARISON_OPS,
    YEAR_LITERAL_RE,
)
from ._contracts_base import (
    PROBE_CTE_EMISSION_KINDS,
    ComparisonJoinScopeExceededError,
    DatabaseFeatureCapability,
    FailureCategory,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    WhereParam,
    column_sensitivity_from_dict,
    expr_registry_ref,
    having_leaves,
    where_leaves,
)
from ._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol, effective_select_parts
from ._contracts_schema import (
    CaseRegistryStep,
    CaseWhenExpr,
    ColumnMetadata,
    CteOutputColumnMeta,
    IntentIssue,
    SchemaGraph,
    WindowRegistryStep,
    WindowSpec,
    make_intent_issue,
)
from ._core_utils import (
    cte_join_reachability_tables,
    debug,
    intent_join_reachability_tables,
    join_resolved_scope_tables,
    notify,
)
from ._dialect_sqlglot_helper import array_storage_kind
from ._intent_expr import expr_canonical_key, extract_columns_from_expr, strip_leading_distinct_from_column_ref
from ._schema_graph import fk_infer_value_types_compatible


def _strip_distinct_prefix(col: str) -> str:
    """Remove a leading `DISTINCT` keyword from a column reference. string."""
    if col and col.upper().startswith("DISTINCT "):
        return col[9:].strip()
    return col


def extract_col_from_scalar_wrapper(col_expr: str) -> str:
    """Strip a scalar wrapper and leading `DISTINCT`, returning the. inner column expression."""
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
    scalar_func: str | None, context: str, location: str, col_meta: Any | None = None
) -> list[IntentIssue]:
    """Validate that a scalar function name is allowed."""
    issues: list[IntentIssue] = []
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
    """Return the first scalar argument lowercased, or an empty string."""
    if not args:
        return ""
    return str(args[0]).strip().lower()


def _is_extract_epoch(func: str | None, args: list[Any]) -> bool:
    """Return whether `func` is `extract` with a disallowed unit such. as. epoch."""
    if not func or func.lower() != "extract":
        return False
    unit = _first_arg_lower(args)
    return unit in DISALLOWED_EXTRACT_UNITS


def validate_expr_no_extract_epoch(expr: NormalizedExpr, context: str, location: str) -> list[IntentIssue]:
    """Flag `EXTRACT(EPOCH FROM ...)` in expressions because EPOCH is. not supported."""
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
    """Validate that an aggregation function name is allowed."""
    issues: list[IntentIssue] = []
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


def _validate_string_agg_groups(
    groups: list[MulGroup],
    *,
    context: str,
    location: str,
    cap: DatabaseFeatureCapability | None,
) -> list[IntentIssue]:
    issues: list[IntentIssue] = []
    for g in groups:
        if (g.agg_func or "").strip().lower() != "string_agg":
            continue
        if not g.agg_sep_param_key:
            issues.append(
                make_intent_issue(
                    issue_id=f"string_agg_missing_sep_{context}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"string_agg in {location} requires agg_sep_param_key",
                    context={"location": location},
                )
            )
        if g.agg_order_by and cap is not None and not cap.supports_ordered_string_agg:
            issues.append(
                make_intent_issue(
                    issue_id=f"ordered_string_agg_unsupported_{context}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"ordered string_agg is not supported in {location}",
                    context={"location": location},
                )
            )
    return issues


def _validate_median_groups(
    groups: list[MulGroup],
    *,
    context: str,
    location: str,
    cap: DatabaseFeatureCapability | None,
) -> list[IntentIssue]:
    issues: list[IntentIssue] = []
    for g in groups:
        if (g.agg_func or "").strip().lower() != "median":
            continue
        if cap is not None and not cap.supports_median:
            issues.append(
                make_intent_issue(
                    issue_id=f"median_unsupported_{context}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"median is not supported in {location}",
                    context={"location": location},
                )
            )
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
        if _where_param_has_registry_ref(br.condition):
            return True
        if _expr_has_registry_ref(br.result):
            return True
    if cw.else_result is not None and _expr_has_registry_ref(cw.else_result):
        return True
    return False


def _where_param_has_registry_ref(fp: WhereParam) -> bool:
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
    where: PredicateGroup | None = None,
    having: PredicateGroup | None = None,
    where_params: list[WhereParam] | None = None,
    having_param: list[HavingParam] | None = None,
) -> list[IntentIssue]:
    """Validate per-scope window/case registry definitions and ``registry_ref`` uses."""
    filter_leaves = list(where_params or [])
    if not filter_leaves and where is not None:
        filter_leaves = [leaf for leaf in where.leaves() if isinstance(leaf, WhereParam)]
    having_leaves = list(having_param or [])
    if not having_leaves and having is not None:
        having_leaves = [leaf for leaf in having.leaves() if isinstance(leaf, HavingParam)]
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
    for win_step in window_registry or []:
        rid = win_step.registry_id
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
        if _window_spec_has_registry_ref(win_step.window_spec):
            issues.append(
                make_intent_issue(
                    issue_id=f"registry_recursion_window_{context}_{rid}",
                    category=FailureCategory.REGISTRY,
                    severity="error",
                    message=f"{context}: window registry body must not contain registry_ref (id '{rid}')",
                    context={"registry_id": rid, "location": context},
                )
            )
    for case_step in case_registry or []:
        rid = case_step.registry_id
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
        if _case_when_has_registry_ref(case_step.case_when):
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
    for fi, fp in enumerate(filter_leaves):
        ref = expr_registry_ref(fp.left_expr) or ""
        if ref:
            _check_ref(ref, where=f"where_params[{fi}].left_expr")
        if fp.right_expr is not None:
            ref_r = expr_registry_ref(fp.right_expr) or ""
            if ref_r:
                _check_ref(ref_r, where=f"where_params[{fi}].right_expr")
    for hi, hp in enumerate(having_leaves):
        ref = expr_registry_ref(hp.left_expr) or ""
        if ref:
            _check_ref(ref, where=f"having_param[{hi}].left_expr")
        if hp.right_expr is not None:
            ref_r = expr_registry_ref(hp.right_expr) or ""
            if ref_r:
                _check_ref(ref_r, where=f"having_param[{hi}].right_expr")
    return issues


def runtime_scope_registry_error_messages(rt: RuntimeIntent) -> list[str]:
    """Collect error-level scope-registry validation strings for the. main query and each CTE scope."""
    msgs: list[str] = []
    for iss in validate_scope_registries(
        context="main query",
        window_registry=list(rt.window_registry or []),
        case_registry=list(rt.case_registry or []),
        select_cols=rt.select_cols or [],
        group_by_cols=rt.group_by_cols or [],
        order_by_cols=rt.order_by_cols or [],
        where=rt.where,
        having=rt.having,
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
            where=cte.where,
            having=cte.having,
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
    """Validate `SelectCol` entries for column existence and. qualification."""
    issues: list[IntentIssue] = []
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
    cap = schema.database_feature_capability
    for idx, sc in enumerate(select_cols):
        issues.extend(
            _validate_string_agg_groups(
                sc.expr.add_groups + sc.expr.sub_groups,
                context=f"select_{idx}",
                location=context,
                cap=cap,
            )
        )
        issues.extend(
            _validate_median_groups(
                sc.expr.add_groups + sc.expr.sub_groups,
                context=f"select_{idx}",
                location=context,
                cap=cap,
            )
        )
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
    """Validate `OrderByCol` entries for column existence and sort. direction."""
    issues: list[IntentIssue] = []
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
        if obc.nulls is not None:
            if obc.nulls not in ("first", "last"):
                issues.append(
                    make_intent_issue(
                        issue_id=f"order_by_invalid_nulls_{context}_{idx}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=(f"OrderByCol nulls must be 'first' or 'last', got '{obc.nulls}' in {context}"),
                        context={"nulls": obc.nulls, "location": context},
                    )
                )
            elif not (obc.direction or "").strip():
                issues.append(
                    make_intent_issue(
                        issue_id=f"order_by_nulls_requires_direction_{context}_{idx}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=f"OrderByCol nulls requires an explicit direction in {context}",
                        context={"location": context},
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
        issues.extend(
            _selectability_issues_for_normalized_expr(obc.expr, schema, cte_outputs, context, f"order_by[{idx}]")
        )
    debug(f"[validation_schema.validate_order_by_cols_schema] {len(issues)} issues in {context}")
    return issues


def validate_group_by_cols_schema(
    group_by_cols: list[NormalizedExpr],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate `group_by_cols` against the schema and column. groupability."""
    issues: list[IntentIssue] = []
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
        issues.extend(_selectability_issues_for_normalized_expr(g, schema, cte_outputs, context, "group_by"))
    debug(f"[validation_schema.validate_group_by_cols_schema] {len(issues)} issues in {context}")
    return issues


def validate_distinct_on_schema(
    distinct_on: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Require ``order_by_cols`` when ``distinct_on`` is set and validate partition expressions."""
    issues: list[IntentIssue] = []
    if not distinct_on:
        return issues
    if not order_by_cols:
        issues.append(
            make_intent_issue(
                issue_id=f"distinct_on_missing_order_by_{context}",
                category=FailureCategory.ORDER_BY_VALIDITY,
                severity="error",
                message=f"{context}: distinct_on requires a non-empty order_by_cols for deterministic row selection",
                context={"location": context},
            )
        )
    elif distinct_on:
        for idx, do_expr in enumerate(distinct_on):
            if idx >= len(order_by_cols):
                issues.append(
                    make_intent_issue(
                        issue_id=f"distinct_on_order_prefix_{context}_{idx}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=(
                            f"{context}: order_by_cols must begin with the distinct_on partition "
                            f"expressions; missing ordering for partition {idx + 1}"
                        ),
                        context={"location": context, "partition_index": idx},
                    )
                )
                break
            if expr_canonical_key(do_expr) != expr_canonical_key(order_by_cols[idx].expr):
                issues.append(
                    make_intent_issue(
                        issue_id=f"distinct_on_order_prefix_{context}_{idx}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=(
                            f"{context}: order_by_cols must begin with the distinct_on partition "
                            f"expressions before any other sort keys"
                        ),
                        context={"location": context, "partition_index": idx},
                    )
                )
                break
    cte_outputs = cte_outputs or {}
    for idx, expr in enumerate(distinct_on):
        issues.extend(
            validate_group_by_cols_schema([expr], schema, allowed_tables, cte_outputs, f"{context} distinct_on[{idx}]")
        )
    return issues


def _expr_is_temporal_keyword_leaf(expr: NormalizedExpr | None) -> bool:
    """Return True when *expr* is a bare SQL temporal keyword leaf."""
    if expr is None:
        return False
    kw = (expr.keyword or "").strip().lower()
    return kw in ("current_timestamp", "current_date", "localtimestamp", "localtime", "sysdate")


def _validate_where_col(
    col_expr: str,
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
    context: str,
    side: str,
    param_key: str,
) -> list[IntentIssue]:
    """Validate one filter column reference (left or right of a. `WhereParam`)."""
    issues: list[IntentIssue] = []
    if not col_expr:
        return issues
    actual_col = extract_col_from_scalar_wrapper(col_expr)
    if "." not in actual_col:
        issues.append(
            make_intent_issue(
                issue_id=f"where_{side}_unqualified_{context}_{actual_col}",
                category=FailureCategory.WHERE_VALIDITY,
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
                    issue_id=f"where_{side}_cte_col_not_found_{context}_{table_name}_{col_name}",
                    category=FailureCategory.WHERE_VALIDITY,
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
                issue_id=f"where_{side}_table_not_allowed_{context}_{table_name}",
                category=FailureCategory.WHERE_VALIDITY,
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
                issue_id=f"where_{side}_table_not_in_schema_{context}_{table_name}",
                category=FailureCategory.WHERE_VALIDITY,
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
                issue_id=f"where_{side}_col_not_found_{context}_{table_name}_{col_name}",
                category=FailureCategory.WHERE_VALIDITY,
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
    where_params: list[WhereParam],
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """Validate `WhereParam` entries against the schema and allowed. operators."""
    issues: list[IntentIssue] = []
    if not where_params:
        return []
    cte_outputs = cte_outputs or {}
    for fp in where_params:
        param_key = fp.param_key or "unknown"
        if not _expr_is_temporal_keyword_leaf(fp.left_expr):
            issues.extend(
                _validate_where_col(
                    fp.left_expr.primary_column, schema, allowed_tables, cte_outputs, context, "left_col", param_key
                )
            )
        issues.extend(
            _selectability_issues_for_normalized_expr(
                fp.left_expr, schema, cte_outputs, context, f"filter {param_key} (left)"
            )
        )
        if fp.right_expr and not _expr_is_temporal_keyword_leaf(fp.right_expr):
            issues.extend(
                _validate_where_col(
                    fp.right_expr.primary_column, schema, allowed_tables, cte_outputs, context, "right_col", param_key
                )
            )
            issues.extend(
                _selectability_issues_for_normalized_expr(
                    fp.right_expr, schema, cte_outputs, context, f"filter {param_key} (right)"
                )
            )
        if fp.op not in VALID_WHERE_OPS:
            issues.append(
                make_intent_issue(
                    issue_id=f"where_invalid_op_{context}_{fp.op}",
                    category=FailureCategory.WHERE_VALIDITY,
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
                        issue_id=f"where_invalid_value_type_{context}_{fp.value_type}",
                        category=FailureCategory.WHERE_VALIDITY,
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
                        issue_id=f"where_missing_value_{context}_{param_key}",
                        category=FailureCategory.WHERE_VALIDITY,
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
        if fp.op == "=" and fp.value_type == "string" and not fp.right_expr:
            val = fp.resolved_value(param_values)
            if isinstance(val, str) and re.fullmatch(r"(19|20)\d{2}", val.strip()):
                meta = _resolve_col_meta_from_expr(fp.left_expr.primary_column, schema)
                if meta and getattr(meta, "role", "") == "temporal":
                    issues.append(
                        make_intent_issue(
                            issue_id=f"where_temporal_year_literal_{context}_{param_key}",
                            category=FailureCategory.WHERE_VALIDITY,
                            severity="error",
                            message=(
                                f"Filter parameter '{param_key}' compares temporal column '{fp.left_expr.primary_column}' "
                                f"to a four-digit year literal '{val}' using equality. "
                                "Use EXTRACT(year FROM column) = <year> or a full date string instead."
                            ),
                            context={
                                "param_key": param_key,
                                "column": fp.left_expr.primary_column,
                                "value": val,
                                "location": context,
                            },
                        )
                    )
        fp_left_scalar, _ = extract_functions_from_term(fp.left_expr.primary_term)
        fp_right_scalar, _ = extract_functions_from_term(fp.right_expr.primary_term) if fp.right_expr else (None, None)
        fp_left_meta = _resolve_col_meta_from_expr(fp.left_expr.primary_column, schema)
        fp_right_meta = _resolve_col_meta_from_expr(fp.right_expr.primary_column, schema) if fp.right_expr else None
        issues.extend(
            _validate_scalar_func_valid(fp_left_scalar, f"filter_{param_key}_left", context, col_meta=fp_left_meta)
        )
        issues.extend(
            _validate_scalar_func_valid(fp_right_scalar, f"filter_{param_key}_right", context, col_meta=fp_right_meta)
        )
        issues.extend(validate_expr_no_extract_epoch(fp.left_expr, f"filter_{param_key}_left", context))
        if fp.right_expr:
            issues.extend(validate_expr_no_extract_epoch(fp.right_expr, f"filter_{param_key}_right", context))
    debug(f"[validation_schema.validate_filters_schema] {len(issues)} issues in {context}")
    return issues


validate_where_schema = validate_filters_schema


def extract_agg_col(agg_expr: str) -> tuple[str | None, str | None, bool]:
    """Parse `(func, target, has_distinct)` from an aggregation. expression string."""
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


def _split_sql_comma_args(arg_str: str) -> list[str]:
    """Split comma-separated SQL args respecting single-quoted string literals."""
    parts: list[str] = []
    cur: list[str] = []
    in_quote = False
    for ch in arg_str:
        if ch == "'" and not in_quote:
            in_quote = True
            cur.append(ch)
        elif ch == "'" and in_quote:
            in_quote = False
            cur.append(ch)
        elif ch == "," and not in_quote:
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def extract_concat_agg_targets(target: str) -> list[str] | None:
    """Return CONCAT argument strings when *target* is a CONCAT expression or unwrap residue."""
    stripped = (target or "").strip()
    if not stripped:
        return None
    concat_match = re.match(r"^\s*CONCAT\s*\(\s*(.*)\s*\)\s*$", stripped, re.IGNORECASE | re.DOTALL)
    if concat_match:
        return _split_sql_comma_args(concat_match.group(1))
    if "," in stripped and not re.match(r"^\s*\w+\s*\(", stripped):
        return _split_sql_comma_args(stripped)
    return None


def extract_functions_from_term(term: str) -> tuple[str | None, str | None]:
    """Extract outer scalar and inner aggregation function names from a. term."""
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
    """Validate one HAVING aggregation expression (left or right side)."""
    issues: list[IntentIssue] = []
    if not agg_expr:
        return issues
    cte_col_match = re.match(r"^\s*(\w+)\s*\.\s*(\w+)\s*$", agg_expr.strip())
    if cte_col_match:
        cte_name, col_name = cte_col_match.group(1), cte_col_match.group(2)
        cte_cols = cte_outputs.get(cte_name, {})
        col_meta = next((v for k, v in cte_cols.items() if k.lower() == col_name.lower()), None)
        if col_meta and (col_meta.source == "aggregation" or (col_meta.agg_func or "").strip()):
            return issues
    result = extract_agg_col(agg_expr)
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
    if func == "count" and has_distinct and actual_target:
        concat_args = extract_concat_agg_targets(actual_target)
        if concat_args is not None:
            for arg in concat_args:
                arg_target = extract_col_from_scalar_wrapper(arg.strip())
                if not arg_target or arg_target.startswith("'") or arg_target.startswith('"'):
                    continue
                if "." not in arg_target:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"having_{side}_unqualified_{context}_{arg_target}",
                            category=FailureCategory.HAVING_VALIDITY,
                            severity="error",
                            message=(
                                f"HAVING COUNT(DISTINCT CONCAT(...)) argument must be qualified as "
                                f"table.column, got '{arg_target}' in {context}"
                            ),
                            context={
                                "target": arg_target,
                                "side": side,
                                "param_key": param_key,
                                "location": context,
                            },
                        )
                    )
                    continue
                table_name, col_name = arg_target.rsplit(".", 1)
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
                    continue
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
                    continue
                if table_name in schema.tables:
                    table_meta = schema.tables[table_name]
                    table_col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
                    if not table_col_meta:
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
            return issues
    if not actual_target:
        issues.append(
            make_intent_issue(
                issue_id=f"having_{side}_missing_target_{context}",
                category=FailureCategory.HAVING_VALIDITY,
                severity="error",
                message=f"HAVING aggregation target is missing in {context}",
                context={
                    "aggregation": agg_expr,
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
        table_col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not table_col_meta:
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
            value_type = table_col_meta.value_type or "string"
            allowed_types = AGGREGATION_ALLOWED_COLUMN_TYPES.get(func, [])
            if value_type not in allowed_types:
                issues.append(
                    make_intent_issue(
                        issue_id=f"having_{side}_type_mismatch_{context}_{func}_{col_name}",
                        category=FailureCategory.HAVING_VALIDITY,
                        severity="error",
                        message=f"Cannot use {func.upper()} on column '{actual_target}' of type '{table_col_meta.data_type}' in HAVING {side} for {context}",
                        context={
                            "function": func,
                            "column": actual_target,
                            "column_type": table_col_meta.data_type,
                            "side": side,
                            "param_key": param_key,
                            "location": context,
                        },
                    )
                )
    return issues


def _reconstruct_agg_expr(expr: NormalizedExpr) -> str:
    """Rebuild an aggregation expression string from a `NormalizedExpr`."""
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
    """Validate `HavingParam` entries against the schema and allowed. operators."""
    issues: list[IntentIssue] = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:
        param_key = hp.param_key or "unknown"
        issues.extend(
            _validate_having_agg(
                _reconstruct_agg_expr(hp.left_expr), schema, allowed_tables, cte_outputs, context, "left_agg", param_key
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
            _validate_scalar_func_valid(hp_left_scalar, f"having_{param_key}_left", context, col_meta=hp_left_meta)
        )
        issues.extend(
            _validate_scalar_func_valid(hp_right_scalar, f"having_{param_key}_right", context, col_meta=hp_right_meta)
        )
        issues.extend(validate_expr_no_extract_epoch(hp.left_expr, f"having_{param_key}_left", context))
        if hp.right_expr:
            issues.extend(validate_expr_no_extract_epoch(hp.right_expr, f"having_{param_key}_right", context))
    debug(f"[validation_schema.validate_having_schema] {len(issues)} issues in {context}")
    return issues


def validate_where_ops_per_column(
    where_params: list[WhereParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate filter operators against each column's data type and. role."""
    issues: list[IntentIssue] = []
    if not where_params:
        return []
    cte_outputs = cte_outputs or {}
    for fp in where_params:
        if fp.value_type == "date_diff":
            continue
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
                valid_ops = cte_meta.get_valid_where_ops()
                if valid_ops and fp.op not in valid_ops:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"where_op_invalid_for_cte_{context}_{actual_col}_{fp.op}",
                            category=FailureCategory.WHERE_VALIDITY,
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
                            issue_id=f"where_cte_col_not_filterable_{context}_{actual_col}",
                            category=FailureCategory.WHERE_VALIDITY,
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
        valid_ops = col_meta.get_valid_where_ops()
        if fp.op not in valid_ops:
            issues.append(
                make_intent_issue(
                    issue_id=f"where_op_invalid_for_type_{context}_{actual_col}_{fp.op}",
                    category=FailureCategory.WHERE_VALIDITY,
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
                    issue_id=f"where_col_not_filterable_{context}_{actual_col}",
                    category=FailureCategory.WHERE_VALIDITY,
                    severity="warning",
                    message=f"Column '{actual_col}' (role={col_meta.role}) is not recommended for filtering in {context}",
                    context={
                        "column": actual_col,
                        "role": col_meta.role,
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_where_ops_per_column] {len(issues)} issues in {context}")
    return issues


def validate_having_ops_per_column(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate HAVING operators against each column's type and role."""
    issues: list[IntentIssue] = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:

        def _check_expr(term: str, _hp: Any = hp) -> None:
            """Append issues if `_hp.op` is invalid for columns. referenced in `term`. Args: term: Aggregation or column term from a HAVING side. _hp: `HavingParam` bound from the enclosing loop. Returns: None."""
            result = extract_agg_col(term)
            actual_target = result[1]
            if result[0] and actual_target and "." in actual_target:
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
    where_params: list[WhereParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    context: str = "main",
    *,
    scope_param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """Validate `date_window` filters use an allowed relative-date unit."""
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check(fp: WhereParam, loc: str, pv: Mapping[str, Any] | None) -> None:
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
                    category=FailureCategory.WHERE_VALIDITY,
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

    for fp in where_params:
        check(fp, f"{context} where", scope_param_values)
    for cte in cte_steps:
        for fp in where_leaves(cte.where) or []:
            check(fp, f"CTE '{cte.cte_name}' filter", cte.param_values)

    if issues:
        debug(f"[validation_schema.validate_date_window_units] {len(issues)} invalid units")
    return issues


def validate_date_diff_units(
    where_params: list[WhereParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    context: str = "main",
    *,
    scope_param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """Validate `date_diff` filters for allowed units and numeric. amounts."""
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check(fp: WhereParam, loc: str, pv: Mapping[str, Any] | None) -> None:
        """
        Record issues for invalid `date_diff` unit or non-numeric.

        amount. Args: fp: Filter to inspect. loc: Location label for the

        message. pv: Parameter map for resolving hoisted diff payloads.

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

    for fp in where_params:
        check(fp, f"{context} where", scope_param_values)
    for cte in cte_steps:
        for fp in where_leaves(cte.where) or []:
            check(fp, f"CTE '{cte.cte_name}' filter", cte.param_values)

    if issues:
        debug(f"[validation_schema.validate_date_diff_units] {len(issues)} invalid configs")
    return issues


def validate_date_diff_left_expr_is_subtraction(
    where_params: list[WhereParam], cte_steps: list[RuntimeCteStep] | None = None, context: str = "main"
) -> list[IntentIssue]:
    """Reject ``date_diff`` filters whose ``left_expr`` is not a subtraction. A ``date_diff`` filter compares elapsed time to a scalar amount or to an integer duration column via ``right_expr``. Its ``left_expr`` must be a subtraction such as ``end_date - start_date`` or ``current_date - start_date``. A plain column reference is rewritten to ``date_window`` by the repair pipeline."""
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check(fp: WhereParam, loc: str) -> None:
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
                    f"{loc}: date_diff filter on '{col}' must have left_expr as a date subtraction "
                    "between two date columns; use date_window for relative date-window filters "
                    "on a single date column"
                ),
                context={"column": col, "location": context},
            )
        )

    for fp in where_params:
        check(fp, f"{context} where")
    for cte in cte_steps:
        for fp in where_leaves(cte.where) or []:
            check(fp, f"CTE '{cte.cte_name}' filter")

    if issues:
        debug(f"[validation_schema.validate_date_diff_left_expr_is_subtraction] {len(issues)} invalid configs")
    return issues


def validate_null_filters(
    where_params: list[WhereParam], cte_steps: list[RuntimeCteStep] | None = None, context: str = "main"
) -> list[IntentIssue]:
    """Validate `IS NULL` / `IS NOT NULL` filters use `value_type` `null` or empty."""
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check_filter(fp: WhereParam, loc: str) -> IntentIssue | None:
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
                    issue_id=f"null_where_wrong_value_type_{col}",
                    category=FailureCategory.WHERE_STRUCTURE,
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

    for fp in where_params:
        issue = check_filter(fp, f"{context} where")
        if issue:
            issues.append(issue)

    for cte in cte_steps:
        for fp in where_leaves(cte.where) or []:
            issue = check_filter(fp, f"CTE '{cte.cte_name}' filter")
            if issue:
                issues.append(issue)

    if issues:
        debug(f"[validation_schema.validate_null_filters] FAILED with {len(issues)} issues")
    else:
        debug("[validation_schema.validate_null_filters] PASSED")
    return issues


def validate_where_value_type_alignment(
    where_params: list[WhereParam],
    schema: SchemaGraph,
    context: str = "main",
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[IntentIssue]:
    """Warn when string or enum filter values target numeric FK or CTE. columns."""
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for fp in where_params:
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
                        issue_id=f"where_string_on_fk_int_{table_name}_{col_name}_{context}",
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
                debug(f"[validation_schema.validate_where_value_type_alignment] string value on FK int column {col}")
            continue
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            cte_meta = cte_cols.get(col_name) or cte_cols.get(col_name.lower())
            if cte_meta and cte_meta.value_type in {"integer", "number"}:
                issues.append(
                    make_intent_issue(
                        issue_id=f"where_string_on_cte_numeric_{table_name}_{col_name}_{context}",
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
    where_params: list[WhereParam], having_param: list[HavingParam], context: str = "main query"
) -> list[IntentIssue]:
    """Flag surviving `BETWEEN` operators that should have been. decomposed."""
    issues: list[IntentIssue] = []
    for fp in where_params:
        if fp.op.lower() == "between":
            col = fp.left_expr.primary_column
            issues.append(
                make_intent_issue(
                    issue_id=f"where_between_not_decomposed_{col}_{context}",
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


def _refs_from_where_param(fp: WhereParam) -> list[str]:
    """Collect qualified column references from both sides of a filter."""
    refs = extract_columns_from_expr(fp.left_expr)
    if fp.right_expr:
        refs.extend(extract_columns_from_expr(fp.right_expr))
    return refs


def _refs_from_having_param(hp: HavingParam) -> list[str]:
    """Collect qualified column references from both sides of a HAVING. clause."""
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
    """Collect column refs from a select column, window spec, and CASE."""
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
            refs.extend(_refs_from_where_param(br.condition))
            refs.extend(extract_columns_from_expr(br.result))
        if parts.case_when.else_result:
            refs.extend(extract_columns_from_expr(parts.case_when.else_result))
    return refs


def validate_contains_array_filters(
    where_params: list[WhereParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None,
    context: str,
) -> list[IntentIssue]:
    """Ensure `contains` is only used on array columns with. `element_type` set."""
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for i, fp in enumerate(where_params or []):
        if fp.op != "contains":
            continue
        cols = extract_columns_from_expr(fp.left_expr)
        if len(cols) != 1:
            continue
        meta = get_col_meta(cols[0], schema, cte_outputs)
        if meta is None:
            continue
        if array_storage_kind(meta) == "unknown":
            issues.append(
                make_intent_issue(
                    issue_id=f"contains_non_array_{context}_{i}",
                    category=FailureCategory.WHERE_SEMANTIC,
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
    """Validate window function specifications on `SelectCol` entries."""
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
        if ws.function in WINDOW_NUMERIC_ARG_FUNCTIONS:
            if ws.numeric_argument is None or ws.numeric_argument <= 0:
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_numeric_arg_required_{context}_{idx}",
                        category=FailureCategory.SCHEMA,
                        severity="error",
                        message=(f"{context}: window '{ws.function}' requires a positive integer numeric_argument"),
                        context={"function": ws.function, "location": context},
                    )
                )
        elif ws.numeric_argument is not None:
            issues.append(
                make_intent_issue(
                    issue_id=f"window_numeric_arg_forbidden_{context}_{idx}",
                    category=FailureCategory.SCHEMA,
                    severity="error",
                    message=f"{context}: window '{ws.function}' must not carry numeric_argument",
                    context={"function": ws.function, "location": context},
                )
            )
        if ws.function in WINDOW_FUNCTIONS_WITHOUT_COLUMN_ARG and ws.argument is not None:
            issues.append(
                make_intent_issue(
                    issue_id=f"window_column_arg_forbidden_{context}_{idx}",
                    category=FailureCategory.SCHEMA,
                    severity="error",
                    message=f"{context}: window '{ws.function}' must not carry an argument expression",
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


def _group_by_column_keys(group_by_cols: list[NormalizedExpr]) -> frozenset[str]:
    keys: set[str] = set()
    for g in group_by_cols or []:
        keys.add(expr_canonical_key(g))
        col = (g.primary_column or "").strip().lower()
        if col:
            keys.add(col)
    return frozenset(keys)


def validate_window_partition_group_by_alignment(
    *, grain: str, group_by_cols: list[NormalizedExpr], window_registry: list[WindowRegistryStep] | None, context: str
) -> list[IntentIssue]:
    """Require window PARTITION BY columns in GROUP BY when the scope is grouped."""
    issues: list[IntentIssue] = []
    grain = (grain or "row_level").strip().lower()
    group_by = list(group_by_cols or [])
    if grain == "row_level" and not group_by:
        return issues
    if not window_registry:
        return issues
    gb_keys = _group_by_column_keys(group_by)
    for wr_idx, wr in enumerate(window_registry):
        ws = wr.window_spec
        for part_idx, pe in enumerate(ws.partition_by or []):
            for cref in extract_columns_from_expr(pe):
                cref_key = expr_canonical_key(NormalizedExpr.from_column(cref))
                if cref_key in gb_keys or cref.lower() in gb_keys:
                    continue
                issues.append(
                    make_intent_issue(
                        issue_id=f"window_partition_column_missing_{context}_{wr_idx}_{part_idx}_{cref}",
                        category=FailureCategory.GROUP_BY_MEMBERSHIP,
                        severity="error",
                        message=(
                            f"{context}: window PARTITION BY column '{cref}' must appear in "
                            "group_by_cols when grain is grouped or GROUP BY is present"
                        ),
                        context={
                            "column": cref,
                            "location": context,
                            "registry_id": wr.registry_id,
                        },
                    )
                )
    return issues


def validate_redundant_extract_year_column_literals(
    where_params: list[WhereParam], cte_steps: list[RuntimeCteStep] | None = None, context: str = "main"
) -> list[IntentIssue]:
    """Reject bare four-digit year comparisons on a column that already has an EXTRACT(year FROM col) filter."""
    issues: list[IntentIssue] = []
    cte_steps = cte_steps or []

    def check_scope(filters: list[WhereParam], loc: str) -> None:
        extract_cols: set[str] = set()
        for fp in filters or []:
            if fp.left_expr.scalar_func == "extract" and fp.left_expr.scalar_func_args:
                if str(fp.left_expr.scalar_func_args[0]).lower() == "year":
                    ref = fp.left_expr.primary_column
                    if ref:
                        extract_cols.add(ref.lower())
        if not extract_cols:
            return
        for fp in filters or []:
            if fp.op not in YEAR_LITERAL_COMPARISON_OPS or fp.raw_value is None or fp.right_expr:
                continue
            val = str(fp.raw_value).strip()
            if not YEAR_LITERAL_RE.fullmatch(val):
                continue
            if fp.left_expr.scalar_func or fp.left_expr.agg_func or fp.left_expr.inner_scalar_func:
                continue
            ref = fp.left_expr.primary_column
            if ref and ref.lower() in extract_cols:
                issues.append(
                    make_intent_issue(
                        issue_id=f"redundant_extract_year_literal_{loc}_{ref}",
                        category=FailureCategory.WHERE_SEMANTIC,
                        severity="error",
                        message=(
                            f"{loc}: bare year literal comparison on '{ref}' is redundant with an "
                            "existing EXTRACT(year FROM column) filter on the same column"
                        ),
                        context={"column": ref, "location": loc},
                    )
                )

    check_scope(where_params, f"{context} where")
    for cte in cte_steps:
        check_scope(where_leaves(cte.where) or [], f"CTE '{cte.cte_name}' filter")
    return issues


def where_param_to_having_param(fp: WhereParam) -> HavingParam:
    """Convert a `WhereParam` into a `HavingParam` carrying the same. predicate fields. Used when a `CaseWhenBranch` declares ``condition_scope == "having"`` so that HAVING-shaped validators can be applied to its filter-shaped condition."""
    return HavingParam(
        left_expr=fp.left_expr,
        op=fp.op,
        right_expr=fp.right_expr,
        value_type=fp.value_type,
        param_key=fp.param_key,
        raw_value=fp.raw_value,
    )


def iterate_case_branch_conditions(
    select_cols: list[SelectCol] | None,
    case_registry: list[CaseRegistryStep] | None,
    window_registry: list[WindowRegistryStep] | None,
    location_prefix: str,
) -> list[tuple[WhereParam, str, str]]:
    """Enumerate every CASE branch condition reachable from a query. body. Collects conditions from ``case_registry`` entries referenced by bare ``cNN`` tokens in ``select_cols`` and from orphan registry rows not referenced by any select column."""
    out: list[tuple[WhereParam, str, str]] = []
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
        scope = (cw.condition_scope or "where").strip().lower() or "where"
        for bi, br in enumerate(cw.branches or []):
            out.append((br.condition, scope, f"{base_loc}.branches[{bi}]"))
    for step in case_registry or []:
        if not step or step.case_when is None:
            continue
        rid = (step.registry_id or "").strip()
        if rid and rid in seen_registry_ids:
            continue
        cw = step.case_when
        scope = (cw.condition_scope or "where").strip().lower() or "where"
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
    """Validate CASE expressions for branch filters and result column. references."""
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
            branch_scope = (cw.condition_scope or "where").strip().lower() or "where"
            if branch_scope == "having":
                issues.extend(
                    validate_having_schema(
                        [where_param_to_having_param(br.condition)],
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
    """Collect qualified ``table.column`` refs that are direct. arguments. of an allowed aggregate."""
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


def selectability_exempt_qualified_refs(expr: NormalizedExpr, schema: SchemaGraph) -> set[str]:
    """Return qualified refs that appear only as arguments to allowed. aggregate functions."""
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
    """Build access-policy issues for bare non-selectable columns in. one. normalised expression."""
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


def _cte_output_column_is_join_key(col_expr: str, schema: SchemaGraph) -> bool:
    """Return True when *col_expr* names a primary or foreign key column."""
    if "." not in col_expr:
        return False
    table_name, col_name = col_expr.rsplit(".", 1)
    tmeta = schema.tables.get(table_name)
    if tmeta is None:
        return False
    pk = {c.lower() for c in (tmeta.primary_key or [])}
    if col_name.lower() in pk:
        return True
    cm = tmeta.columns.get(col_name) or tmeta.columns.get(col_name.lower())
    return bool(cm and cm.is_foreign_key)


def _column_name_reserved_for_anti_join(col_name: str) -> bool:
    return col_name.endswith(ANTI_JOIN_PRESENCE_COLUMN_SUFFIX)


def _build_cte_outputs_map(intent: RuntimeIntent) -> dict[str, dict[str, CteOutputColumnMeta]]:
    return {c.cte_name: dict(c.output_column_metadata or {}) for c in (intent.cte_steps or []) if c.cte_name}


def _select_col_column_metadata(
    sc: SelectCol,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> ColumnMetadata | None:
    col = sc.expr.primary_column if sc.expr else ""
    if not col:
        return None
    return get_col_meta(col, schema, cte_outputs)


def _probe_output_column_metadata_at(
    probe_cte: RuntimeCteStep,
    index: int,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> ColumnMetadata | None:
    out_cols = probe_cte.output_columns or []
    if index >= len(out_cols):
        return None
    out_name = out_cols[index]
    ocm_map = probe_cte.output_column_metadata or {}
    ocm = ocm_map.get(out_name) or ocm_map.get(out_name.lower())
    if ocm is not None:
        return ColumnMetadata(
            name=out_name,
            data_type=ocm.data_type or "unknown",
            value_type=ocm.value_type or "",
        )
    select_cols = probe_cte.select_cols or []
    if index < len(select_cols):
        return _select_col_column_metadata(select_cols[index], schema, cte_outputs)
    return None


def _types_compatible_for_projection(
    left: ColumnMetadata | None,
    right: ColumnMetadata | None,
) -> bool | None:
    if left is None or right is None:
        return None
    return fk_infer_value_types_compatible(left, right)


def _semi_join_key_shape(cte: RuntimeCteStep, schema: SchemaGraph) -> bool:
    select_cols = cte.select_cols or []
    if not select_cols:
        return False
    for sc in select_cols:
        col = sc.expr.primary_column if sc.expr else ""
        if not col or not _cte_output_column_is_join_key(col, schema):
            return False
    return True


def _positional_projection_type_check(
    outer_cols: list[SelectCol],
    probe_cte: RuntimeCteStep,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
) -> tuple[bool, bool]:
    """Return whether positional types are compatible and whether any position was unresolvable."""
    if len(outer_cols) != len(probe_cte.output_columns or []):
        return False, False
    any_unresolvable = False
    for idx, sc in enumerate(outer_cols):
        outer_meta = _select_col_column_metadata(sc, schema, cte_outputs)
        probe_meta = _probe_output_column_metadata_at(probe_cte, idx, schema, cte_outputs)
        compat = _types_compatible_for_projection(outer_meta, probe_meta)
        if compat is None:
            any_unresolvable = True
        elif not compat:
            return False, any_unresolvable
    return True, any_unresolvable


def validate_table_reference_counts(
    tables: list[str],
    schema: SchemaGraph,
    context: str,
) -> list[IntentIssue]:
    """Refuse when a physical table appears more than twice in one scope."""
    issues: list[IntentIssue] = []
    schema_tables = set(schema.tables.keys())
    counts: dict[str, int] = {}
    for tbl in tables or []:
        if tbl in schema_tables:
            counts[tbl] = counts.get(tbl, 0) + 1
    max_refs = int(PolicyConfig.TABLE_REFERENCE_MAX_PER_SCOPE)
    for tbl in sorted(counts, key=str.lower):
        count = counts[tbl]
        if count > max_refs:
            issues.append(
                make_intent_issue(
                    issue_id=f"table_reference_count_{context.replace(' ', '_')}_{tbl}",
                    category=FailureCategory.WRONG_JOIN,
                    severity="error",
                    message=(
                        f"{context}: table '{tbl}' is referenced {count} times in one scope; "
                        f"at most {max_refs} references are allowed."
                    ),
                    context={"table": tbl, "count": count, "max": max_refs, "scope_label": context},
                )
            )
    return issues


def max_cte_reference_depth(cte_steps: list[RuntimeCteStep]) -> int:
    """Return the longest CTE-to-CTE dependency chain length in *cte_steps*."""
    if not cte_steps:
        return 0
    names = {c.cte_name for c in cte_steps if c.cte_name}
    deps: dict[str, set[str]] = {name: set() for name in names}
    for cte in cte_steps:
        if not cte.cte_name:
            continue
        deps[cte.cte_name] = {t for t in (cte.tables or []) if t in names and t != cte.cte_name}
    memo: dict[str, int] = {}

    def _depth(name: str, visiting: frozenset[str]) -> int:
        if name in memo:
            return memo[name]
        if name in visiting:
            return 0
        child_deps = deps.get(name) or set()
        if not child_deps:
            val = 1
        else:
            val = 1 + max(_depth(child, visiting | {name}) for child in sorted(child_deps))
        memo[name] = val
        return val

    if not names:
        return 0
    return max(_depth(name, frozenset()) for name in sorted(names, key=str.lower))


def validate_cte_limits(intent: RuntimeIntent, context: str = "main query") -> list[IntentIssue]:
    """Refuse intents that exceed configured CTE count or reference- depth caps."""
    issues: list[IntentIssue] = []
    count = len(intent.cte_steps or [])
    max_steps = int(PolicyConfig.MAX_CTE_STEPS)
    if count > max_steps:
        issues.append(
            make_intent_issue(
                issue_id="cte_step_count_exceeded",
                category=FailureCategory.CTE_STRUCTURE,
                severity="error",
                message=(f"{context}: intent defines {count} CTE steps; at most {max_steps} are permitted."),
                context={"count": count, "max": max_steps, "scope_label": context},
            )
        )
    depth = max_cte_reference_depth(list(intent.cte_steps or []))
    max_depth = int(PolicyConfig.MAX_CTE_REFERENCE_DEPTH)
    if depth > max_depth:
        issues.append(
            make_intent_issue(
                issue_id="cte_reference_depth_exceeded",
                category=FailureCategory.CTE_STRUCTURE,
                severity="error",
                message=(f"{context}: CTE reference depth is {depth}; at most {max_depth} is permitted."),
                context={"depth": depth, "max": max_depth, "scope_label": context},
            )
        )
    for issue in issues:
        code = refusal_diagnostic_code_for_intent_issue(issue)
        if code:
            emit_session_refusal_diagnostic(code, issue.message)
    return issues


def validate_predicate_nesting_depth(
    where: PredicateGroup | None,
    having: PredicateGroup | None,
    context: str = "main query",
) -> list[IntentIssue]:
    """Reject predicate trees whose nesting depth exceeds the configured maximum."""
    issues: list[IntentIssue] = []
    if where is not None and where.depth() > MAX_PREDICATE_NESTING_DEPTH:
        issues.append(
            make_intent_issue(
                issue_id="where_predicate_nesting_depth",
                category=FailureCategory.WHERE_SEMANTIC,
                severity="error",
                message=(f"WHERE predicate nesting exceeds MAX_PREDICATE_NESTING_DEPTH={MAX_PREDICATE_NESTING_DEPTH}."),
                context={"location": context},
            )
        )
    if having is not None and having.depth() > MAX_PREDICATE_NESTING_DEPTH:
        issues.append(
            make_intent_issue(
                issue_id="having_predicate_nesting_depth",
                category=FailureCategory.HAVING_SEMANTIC,
                severity="error",
                message=(
                    f"HAVING predicate nesting exceeds MAX_PREDICATE_NESTING_DEPTH={MAX_PREDICATE_NESTING_DEPTH}."
                ),
                context={"location": context},
            )
        )
    return issues


def qualifies_as_semi_join_probe(
    cte: RuntimeCteStep,
    intent: RuntimeIntent,
    schema: SchemaGraph,
) -> bool:
    """Return whether *cte* has a valid semi_join probe projection shape."""
    for sc in cte.select_cols or []:
        if sc.is_aggregated or (sc.expr and sc.expr.agg_func):
            return False
    if _semi_join_key_shape(cte, schema):
        return True
    cte_outputs = _build_cte_outputs_map(intent)
    intersection_shape, _ = _positional_projection_type_check(
        list(intent.select_cols or []),
        cte,
        schema,
        cte_outputs,
    )
    return intersection_shape


def validate_cte_emission_shapes(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Validate semi-join, anti-join, and set-difference CTE emission constraints."""
    issues: list[IntentIssue] = []
    cte_outputs = _build_cte_outputs_map(intent)
    cte_by_name = {c.cte_name: c for c in (intent.cte_steps or []) if c.cte_name}
    for cte in intent.cte_steps or []:
        emission = getattr(cte, "emission", "join_table") or "join_table"
        ctx = f"CTE '{cte.cte_name}'"
        for out_name in cte.output_columns or []:
            if _column_name_reserved_for_anti_join(out_name):
                issues.append(
                    make_intent_issue(
                        issue_id=f"cte_reserved_presence_col_{cte.cte_name}_{out_name}",
                        category=FailureCategory.CTE_STRUCTURE,
                        severity="error",
                        message=(
                            f"{ctx}: output column '{out_name}' uses the reserved anti-join "
                            f"presence suffix '{ANTI_JOIN_PRESENCE_COLUMN_SUFFIX}'"
                        ),
                        context={"cte_name": cte.cte_name, "column": out_name},
                    )
                )
        if emission == "semi_join":
            has_aggregate = False
            for idx, sc in enumerate(cte.select_cols or []):
                col = sc.expr.primary_column
                if sc.is_aggregated or (sc.expr.agg_func or ""):
                    has_aggregate = True
                    issues.append(
                        make_intent_issue(
                            issue_id=f"semi_join_payload_agg_{cte.cte_name}_{idx}",
                            category=FailureCategory.CTE_STRUCTURE,
                            severity="error",
                            message=f"{ctx}: semi_join select_cols[{idx}] must project keys only, not aggregates",
                            context={"index": idx},
                        )
                    )
            if not has_aggregate:
                key_shape = _semi_join_key_shape(cte, schema)
                intersection_shape = False
                intersection_unresolvable = False
                if not key_shape:
                    intersection_shape, intersection_unresolvable = _positional_projection_type_check(
                        list(intent.select_cols or []),
                        cte,
                        schema,
                        cte_outputs,
                    )
                if not key_shape and not intersection_shape:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"semi_join_projection_shape_{cte.cte_name}",
                            category=FailureCategory.CTE_STRUCTURE,
                            severity="error",
                            message=(
                                f"{ctx}: semi_join projection must either project declared join keys "
                                "only or match the outer select_cols in arity and positional type"
                            ),
                            context={"cte_name": cte.cte_name},
                        )
                    )
                elif intersection_unresolvable and not key_shape:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"semi_join_projection_type_unresolved_{cte.cte_name}",
                            category=FailureCategory.CTE_STRUCTURE,
                            severity="warning",
                            message=(
                                f"{ctx}: semi_join intersection projection has positions whose "
                                "types could not be resolved for compatibility checking"
                            ),
                            context={"cte_name": cte.cte_name},
                        )
                    )
        if emission == "anti_join":
            for fp in where_leaves(cte.where) if cte.where else []:
                left = fp.left_expr.primary_column if fp.left_expr else ""
                if left and _column_name_reserved_for_anti_join(left.rsplit(".", 1)[-1]):
                    issues.append(
                        make_intent_issue(
                            issue_id=f"anti_join_user_presence_where_{cte.cte_name}",
                            category=FailureCategory.CTE_STRUCTURE,
                            severity="error",
                            message=f"{ctx}: anti_join presence marker filters are renderer-owned",
                            context={"where": left},
                        )
                    )
    for fp in where_leaves(intent.where) if intent.where else []:
        left = fp.left_expr.primary_column if fp.left_expr else ""
        if not left or "." not in left:
            continue
        tbl, col = left.rsplit(".", 1)
        matched_cte = cte_by_name.get(tbl)
        if matched_cte is None:
            continue
        emission = getattr(matched_cte, "emission", "join_table") or "join_table"
        if emission != "anti_join":
            continue
        if (fp.op or "").strip().lower() in ("is null", "is not null"):
            issues.append(
                make_intent_issue(
                    issue_id=f"anti_join_user_null_where_{tbl}",
                    category=FailureCategory.WHERE_VALIDITY,
                    severity="error",
                    message=(
                        f"main query: anti_join null tests on '{left}' are renderer-owned; "
                        f"do not declare IS NULL on anti-join probe keys"
                    ),
                    context={"column": left, "op": fp.op},
                )
            )
    anti_in_main = [t for t in (intent.tables or []) if getattr(cte_by_name.get(t), "emission", "") == "anti_join"]
    if anti_in_main and intent.select_cols:
        for probe in anti_in_main:
            probe_cte = cte_by_name[probe]
            outer_cols = list(intent.select_cols or [])
            probe_out = list(probe_cte.output_columns or [])
            if len(probe_out) != len(outer_cols):
                issues.append(
                    make_intent_issue(
                        issue_id=f"set_difference_arity_{probe}",
                        category=FailureCategory.CTE_STRUCTURE,
                        severity="error",
                        message=(
                            f"set difference via anti_join '{probe}' requires probe keys to match "
                            f"the outer projection arity ({len(outer_cols)} vs "
                            f"{len(probe_out)})"
                        ),
                        context={"probe": probe},
                    )
                )
                continue
            type_unresolvable = False
            for idx, sc in enumerate(outer_cols):
                outer_meta = _select_col_column_metadata(sc, schema, cte_outputs)
                probe_meta = _probe_output_column_metadata_at(probe_cte, idx, schema, cte_outputs)
                compat = _types_compatible_for_projection(outer_meta, probe_meta)
                if compat is None:
                    type_unresolvable = True
                    continue
                if not compat:
                    outer_type = (outer_meta.value_type if outer_meta else "") or "unknown"
                    probe_type = (probe_meta.value_type if probe_meta else "") or "unknown"
                    outer_name = outer_meta.name if outer_meta else f"position_{idx}"
                    probe_name = (
                        probe_meta.name
                        if probe_meta
                        else (probe_out[idx] if idx < len(probe_out) else f"position_{idx}")
                    )
                    issues.append(
                        make_intent_issue(
                            issue_id=f"set_difference_type_{probe}_{idx}",
                            category=FailureCategory.CTE_STRUCTURE,
                            severity="error",
                            message=(
                                f"set difference via anti_join '{probe}' position {idx}: "
                                f"outer '{outer_name}' ({outer_type}) is incompatible with "
                                f"probe '{probe_name}' ({probe_type})"
                            ),
                            context={"probe": probe, "index": idx},
                        )
                    )
            if type_unresolvable:
                issues.append(
                    make_intent_issue(
                        issue_id=f"set_difference_type_unresolved_{probe}",
                        category=FailureCategory.CTE_STRUCTURE,
                        severity="warning",
                        message=(
                            f"set difference via anti_join '{probe}' has positions whose types "
                            "could not be resolved for compatibility checking"
                        ),
                        context={"probe": probe},
                    )
                )
    return issues


def validate_probe_cte_anchor_placement(
    intent: RuntimeIntent,
    *,
    join_signature: list[str] | None = None,
) -> list[IntentIssue]:
    """Reject semi-join and anti-join probe CTEs used as join anchors or left operands."""
    issues: list[IntentIssue] = []
    probe_names = {
        c.cte_name
        for c in (intent.cte_steps or [])
        if (getattr(c, "emission", "join_table") or "join_table") in PROBE_CTE_EMISSION_KINDS and c.cte_name
    }
    if not probe_names:
        return issues

    def _left_operand_issue(probe: str, signature: list[str], context: str) -> IntentIssue | None:
        left_tables = _join_sig_left_tables(signature)
        if probe.lower() not in left_tables:
            return None
        return make_intent_issue(
            issue_id=f"probe_cte_left_operand_{probe}",
            category=FailureCategory.CTE_STRUCTURE,
            severity="error",
            message=(f"{context}: semi_join and anti_join probe '{probe}' may not be the left operand of a join edge"),
            context={"cte_name": probe},
        )

    main_sig = list(join_signature or [])
    for probe in sorted(probe_names):
        issue = _left_operand_issue(probe, main_sig, "main query")
        if issue is not None:
            issues.append(issue)

    for cte in intent.cte_steps or []:
        cte_sig = list(cte.chosen_join_path_signature or [])
        cte_context = f"CTE '{cte.cte_name}'"
        for probe in sorted(probe_names):
            issue = _left_operand_issue(probe, cte_sig, cte_context)
            if issue is not None:
                issues.append(issue)
        tables = list(cte.tables or [])
        if len(tables) <= 1:
            continue
        anchor = tables[0]
        if anchor in probe_names:
            issues.append(
                make_intent_issue(
                    issue_id=f"probe_cte_anchor_{anchor}_{cte.cte_name}",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message=(f"{cte_context}: semi_join and anti_join probe '{anchor}' may not be the join anchor"),
                    context={"cte_name": anchor},
                )
            )
    main_tables = list(intent.tables or [])
    if len(main_tables) > 1 and main_tables[0] in probe_names:
        issues.append(
            make_intent_issue(
                issue_id=f"probe_cte_main_anchor_{main_tables[0]}",
                category=FailureCategory.CTE_STRUCTURE,
                severity="error",
                message=(f"main query: semi_join and anti_join probe '{main_tables[0]}' may not be the join anchor"),
                context={"cte_name": main_tables[0]},
            )
        )
    return issues


def _join_sig_left_tables(signature: list[str]) -> set[str]:
    lefts: set[str] = set()
    for seg in signature or []:
        seg = seg.strip()
        if "->" not in seg or "." not in seg:
            continue
        left_part = seg.split("->", 1)[0].strip()
        left_tbl = left_part.split(".", 1)[0].strip()
        if left_tbl:
            lefts.add(left_tbl.lower())
    return lefts


def _preservation_is_no_op_for_table(table: str, signature: list[str], schema: SchemaGraph) -> bool:
    """Return True when preservation on *table* would not change results."""
    table_l = table.lower()
    for seg in signature or []:
        seg = seg.strip()
        if "->" not in seg or "." not in seg:
            continue
        left_part, right_part = seg.split("->", 1)
        left_tbl = left_part.strip().split(".", 1)[0].strip()
        right_tbl = right_part.strip().split(".", 1)[0].strip()
        right_cols = [c.strip() for c in right_part.split(".", 1)[1].split(",")]
        if left_tbl.lower() != table_l:
            continue
        join_tbl, paired_tbl, cols_on_join = right_tbl, left_tbl, right_cols
        tmeta = schema.tables.get(join_tbl)
        if not tmeta:
            return False
        fk_any = False
        fk_nullable = False
        for jc in cols_on_join:
            cm = tmeta.columns.get(jc)
            if cm and cm.is_foreign_key and cm.fk_target and cm.fk_target[0] == paired_tbl:
                fk_any = True
                if cm.is_nullable:
                    fk_nullable = True
        if not fk_any or fk_nullable:
            return False
    return bool(signature)


def validate_preserve_tables(
    tables: list[str],
    preserve_tables: list[str],
    schema: SchemaGraph,
    context: str,
    *,
    join_signature: list[str] | None = None,
    probe_cte_names: frozenset[str] | None = None,
) -> list[IntentIssue]:
    """Validate table-scoped row-preservation declarations."""
    issues: list[IntentIssue] = []
    if not preserve_tables:
        return issues
    probes = probe_cte_names or frozenset()
    allowed = {t.lower() for t in (tables or [])}
    sig = list(join_signature or [])
    left_tables = _join_sig_left_tables(sig)
    anchor = tables[0] if tables else ""
    anchor_l = anchor.lower() if anchor else ""
    for raw in preserve_tables:
        name = (raw or "").strip()
        if not name:
            continue
        if name.lower() not in allowed:
            issues.append(
                make_intent_issue(
                    issue_id=f"preserve_tables_unknown_{context}_{name}",
                    category=FailureCategory.WRONG_JOIN,
                    severity="error",
                    message=(
                        f"{context}: preserve_tables entry '{name}' is not in the closed "
                        f"table vocabulary {sorted(tables or [])}"
                    ),
                    context={"table": name, "location": context},
                )
            )
            continue
        if name in probes:
            issues.append(
                make_intent_issue(
                    issue_id=f"preserve_tables_probe_{context}_{name}",
                    category=FailureCategory.WRONG_JOIN,
                    severity="error",
                    message=(
                        f"{context}: preserve_tables may not name probe CTE '{name}' because "
                        "its join kind is fixed by emission"
                    ),
                    context={"table": name, "location": context},
                )
            )
            continue
        reachable = name.lower() == anchor_l or name.lower() in left_tables
        if sig and not reachable:
            issues.append(
                make_intent_issue(
                    issue_id=f"preserve_tables_unreachable_{context}_{name}",
                    category=FailureCategory.WRONG_JOIN,
                    severity="error",
                    message=(
                        f"{context}: preserve_tables '{name}' is not reachable as anchor or "
                        "left operand on the resolved join path"
                    ),
                    context={"table": name, "location": context},
                )
            )
            continue
        if sig and _preservation_is_no_op_for_table(name, sig, schema):
            issues.append(
                make_intent_issue(
                    issue_id=f"preserve_tables_noop_{context}_{name}",
                    category=FailureCategory.WRONG_JOIN,
                    severity="error",
                    message=(
                        f"{context}: preserve_tables '{name}' would have no effect because "
                        "its only edges are many-to-one on non-nullable foreign keys"
                    ),
                    context={"table": name, "location": context},
                )
            )
    return issues


def validate_probe_cte_modifiers(intent: RuntimeIntent) -> list[IntentIssue]:
    """Reject intent fields on probe CTEs that conflict with existence semantics."""
    issues: list[IntentIssue] = []
    for cte in intent.cte_steps or []:
        emission = getattr(cte, "emission", "join_table") or "join_table"
        if emission not in PROBE_CTE_EMISSION_KINDS:
            continue
        ctx = f"CTE '{cte.cte_name}'"
        if cte.distinct_select_index >= 0:
            issues.append(
                make_intent_issue(
                    issue_id=f"probe_cte_distinct_select_index_{cte.cte_name}",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message=f"{ctx}: distinct_select_index conflicts with probe emission",
                    context={"cte_name": cte.cte_name},
                )
            )
        if cte.distinct_on:
            issues.append(
                make_intent_issue(
                    issue_id=f"probe_cte_distinct_on_{cte.cte_name}",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message=f"{ctx}: distinct_on conflicts with probe emission",
                    context={"cte_name": cte.cte_name},
                )
            )
        if cte.limit is not None:
            issues.append(
                make_intent_issue(
                    issue_id=f"probe_cte_limit_{cte.cte_name}",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message=f"{ctx}: limit conflicts with probe emission",
                    context={"cte_name": cte.cte_name},
                )
            )
    return issues


def validate_join_path_reachability_for_tables(
    tables: list[str], schema: SchemaGraph, context: str
) -> list[IntentIssue]:
    """Emit issues when physical tables in *tables* do not form one connected join component."""
    combined: set[str] = set(tables or [])
    physical = {t for t in combined if t in schema.tables}
    if len(physical) <= 1:
        return []
    jpm = schema.join_paths_multi or {}
    adj: dict[str, set[str]] = {t: set() for t in physical}

    def _connect(left: str, right: str) -> None:
        if left in physical and right in physical and left != right:
            adj[left].add(right)
            adj[right].add(left)

    sorted_physical = sorted(physical)
    for left_idx, src in enumerate(sorted_physical):
        for dst in sorted_physical[left_idx + 1 :]:
            paths = list((jpm.get(src) or {}).get(dst) or []) + list((jpm.get(dst) or {}).get(src) or [])
            for path in paths:
                prior: str | None = None
                for edge in path:
                    for tbl in (str(edge.get("src_table", "")), str(edge.get("dst_table", ""))):
                        if tbl not in physical:
                            continue
                        if prior is not None and prior != tbl:
                            _connect(prior, tbl)
                        prior = tbl

    for tbl in physical:
        tbl_meta = schema.tables.get(tbl)
        if tbl_meta is None:
            continue
        for _cn, col in tbl_meta.columns.items():
            for nt, _nc in col.semantic_join_neighbors:
                _connect(tbl, nt)

    root = min(physical)
    visited: set[str] = set()
    queue = [root]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        queue.extend(adj[cur] - visited)

    issues: list[IntentIssue] = []
    for target in sorted(physical - visited):
        issues.append(
            make_intent_issue(
                issue_id=f"join_unreachable_{context.replace(' ', '_')}_{root}_{target}",
                category=FailureCategory.WRONG_JOIN,
                severity="error",
                message=(
                    f"{context}: no schema join path between '{root}' and '{target}' "
                    f"(disconnected FK groups; add a bridging foreign_keys_add)."
                ),
                context={"root": root, "target": target, "tables": sorted(physical), "scope_label": context},
            )
        )
    return issues


def validate_join_path_reachability(intent: RuntimeIntent, schema: SchemaGraph, context: str) -> list[IntentIssue]:
    """Like :func:`validate_join_path_reachability_for_tables` using the resolved join scope when set."""
    tables = intent_join_reachability_tables(intent)
    return validate_join_path_reachability_for_tables(tables, schema, context)


def validate_intent_join_reachability(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Collect join-path reachability issues for the main query and every CTE scope."""
    issues: list[IntentIssue] = []
    issues.extend(validate_join_path_reachability(intent, schema, "main query"))
    for cte in intent.cte_steps or []:
        if not cte.cte_name:
            continue
        issues.extend(
            validate_join_path_reachability_for_tables(
                cte_join_reachability_tables(cte), schema, f"CTE '{cte.cte_name}'"
            )
        )
    return issues


def validate_selectability(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None,
    context: str,
    *,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> list[IntentIssue]:
    """Reject bare non-selectable columns in SELECT expressions, window. partitions, and CASE results."""
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
                        pe, schema, cte_outputs, context, f"{detail} window partition[{pi}]"
                    )
                )
            if ws.argument is not None:
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        ws.argument, schema, cte_outputs, context, f"{detail} window argument"
                    )
                )
        if parts.case_when:
            cw = parts.case_when
            for bi, br in enumerate(cw.branches):
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        br.result, schema, cte_outputs, context, f"{detail} case_when[{bi}]"
                    )
                )
            if cw.else_result is not None:
                issues.extend(
                    _selectability_issues_for_normalized_expr(
                        cw.else_result, schema, cte_outputs, context, f"{detail} case_else"
                    )
                )
    return issues


def get_col_type(
    col_expr: str, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> str | None:
    """Resolve a column's `value_type` from the schema or CTE outputs."""
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
    col_expr: str, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> Any | None:
    """Resolve `ColumnMetadata` from the schema graph or synthesise it. from CTE outputs."""
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
            valid_where_ops=list(cte_meta.valid_where_ops or []),
            valid_aggregations=list(cte_meta.valid_aggregations or []),
            valid_having_ops=list(cte_meta.valid_having_ops or []),
            sensitivity=column_sensitivity_from_dict({"sensitivity": cte_meta.sensitivity}),
        )
    if table_name not in schema.tables:
        return None
    table_meta = schema.tables[table_name]
    return table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())


def is_col_numeric(
    col_ref: str, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool | None:
    """Return whether a column's value type is numeric."""
    col_type = get_col_type(col_ref, schema, cte_outputs)
    if col_type is None:
        return None
    return col_type in ("integer", "number")


def is_col_arithmetic_role(
    col_ref: str, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool | None:
    """Return whether a column's role allows use in arithmetic. expressions."""
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


def expr_has_arithmetic(expr: NormalizedExpr) -> bool:
    """Return `True` if a `NormalizedExpr` contains arithmetic. operations."""
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


def expr_result_is_numeric(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool | None:
    """Return whether the result of a `NormalizedExpr` is numeric."""
    if expr.agg_func and expr.agg_func in NUMERIC_RESULT_AGGS:
        return True
    if expr.scalar_func and expr.scalar_func in NUMERIC_RESULT_SCALARS:
        return True
    if expr.inner_scalar_func and expr.inner_scalar_func in NUMERIC_RESULT_SCALARS:
        return True
    if is_date_integer_day_arithmetic(expr, schema, cte_outputs):
        return False
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


def group_is_simple_integer_column(
    group: MulGroup, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool:
    """Return True when *group* is a bare integer column reference."""
    if group.divide or group.coefficient != 1.0 or len(group.multiply) != 1:
        return False
    term = group.multiply[0]
    if term.add_groups or term.sub_groups or term.add_values or term.sub_values:
        return False
    col = term.primary_column
    if not col:
        return False
    col_type = get_col_type(col, schema, cte_outputs) or ""
    return col_type in {"integer", "int", "bigint", "smallint", "tinyint", "long", "short"}


def group_is_simple_date_column(
    group: MulGroup, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool:
    """Return True when *group* is a bare date or timestamp column reference."""
    if group.divide or group.coefficient != 1.0 or len(group.multiply) != 1:
        return False
    term = group.multiply[0]
    if term.add_groups or term.sub_groups or term.add_values or term.sub_values:
        return False
    col = term.primary_column
    if not col:
        return False
    col_type = get_col_type(col, schema, cte_outputs) or ""
    return col_type in DATE_FRIENDLY_VALUE_TYPES


def _extract_date_integer_day_base_column(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> str | None:
    """Return the date column when *expr* is ``date_col +/- integer days``."""
    for group in expr.add_groups + expr.sub_groups:
        if group_is_simple_date_column(group, schema, cte_outputs):
            return group.multiply[0].primary_column
    return None


def _has_integer_day_offset(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool:
    """Return True when *expr* carries an integer day offset via literal or column."""
    if _extract_date_integer_day_base_column(expr, schema, cte_outputs) is None:
        return False
    if expr.add_values or expr.sub_values:
        return len(expr.add_groups) + len(expr.sub_groups) == 1
    int_groups = [
        g for g in expr.add_groups + expr.sub_groups if group_is_simple_integer_column(g, schema, cte_outputs)
    ]
    return len(int_groups) == 1 and len(expr.add_groups) + len(expr.sub_groups) == 2


def is_date_column_subtraction(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool:
    """Return True when *expr* subtracts one date column from another."""
    if not (expr.sub_groups and expr.add_groups):
        return False
    if expr.add_values or expr.sub_values:
        return False
    if any(group_is_simple_integer_column(g, schema, cte_outputs) for g in expr.sub_groups):
        return False
    if any(group_is_simple_integer_column(g, schema, cte_outputs) for g in expr.add_groups):
        return False
    groups = expr.add_groups + expr.sub_groups
    return len(groups) == 2 and all(group_is_simple_date_column(g, schema, cte_outputs) for g in groups)


def is_date_integer_day_arithmetic(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool:
    """Return True when *expr* evaluates to a date via date-column +/- integer days."""
    if expr.agg_func or expr.scalar_func or expr.inner_scalar_func:
        return False
    return _extract_date_integer_day_base_column(expr, schema, cte_outputs) is not None and _has_integer_day_offset(
        expr, schema, cte_outputs
    )


def phys_table_key(tbl: str) -> str:
    return (tbl or "").strip().lower()


def _column_profiled_unique(schema: SchemaGraph, table: str, column: str) -> bool | None:
    """Return whether *column* on *table* is declared or profiled unique, or None when unknown."""
    tmeta = schema.tables.get(table)
    if tmeta is None:
        return None
    pk = list(tmeta.primary_key or [])
    if pk == [column]:
        return True
    cm = tmeta.columns.get(column)
    if cm is None:
        return None
    if cm.is_unique:
        return True
    row_count = int(tmeta.row_count or 0)
    distinct = int(cm.distinct_count or 0)
    if row_count > 0 and distinct > 0:
        return distinct >= row_count
    return None


def _join_columns_profiled_unique(schema: SchemaGraph, table: str, cols: list[str]) -> bool | None:
    """Return whether *cols* on *table* are jointly unique, or None when profiling is inconclusive."""
    clean = [c.strip() for c in cols if c.strip()]
    if not clean:
        return None
    tmeta = schema.tables.get(table)
    if tmeta is None:
        return None
    pk = list(tmeta.primary_key or [])
    if pk == clean:
        return True
    if len(clean) == 1:
        return _column_profiled_unique(schema, table, clean[0])
    profiled: list[bool | None] = [_column_profiled_unique(schema, table, col) for col in clean]
    if any(item is False for item in profiled):
        return False
    if all(item is True for item in profiled):
        return True
    return None


def _fk_edge_matches(
    child_tbl: str,
    parent_tbl: str,
    child_cols: list[str],
    parent_cols: list[str],
    schema: SchemaGraph | None,
) -> bool:
    """Return True when schema declares *child_cols* on *child_tbl* referencing *parent_cols* on *parent_tbl*."""
    if schema is None:
        return False
    child_cols_norm = [c.strip() for c in child_cols if c.strip()]
    parent_cols_norm = [c.strip() for c in parent_cols if c.strip()]
    if not child_cols_norm or not parent_cols_norm:
        return False
    tmeta = schema.tables.get(child_tbl)
    if tmeta is None:
        return False
    for edge in tmeta.foreign_keys:
        if (
            phys_table_key(edge.dst_table) == phys_table_key(parent_tbl)
            and list(edge.src_cols) == child_cols_norm
            and list(edge.dst_cols) == parent_cols_norm
        ):
            return True
    if len(child_cols_norm) == 1 and len(parent_cols_norm) == 1:
        cm = tmeta.columns.get(child_cols_norm[0])
        if cm and cm.is_foreign_key and cm.fk_target:
            dst_tbl, dst_col = cm.fk_target
            if phys_table_key(dst_tbl) == phys_table_key(parent_tbl) and dst_col == parent_cols_norm[0]:
                return True
    return False


def _fk_points_to_parent(child_tbl: str, parent_tbl: str, cols_on_child: list[str], schema: SchemaGraph | None) -> bool:
    if schema is None:
        return False
    child_cols_norm = [c.strip() for c in cols_on_child if c.strip()]
    if not child_cols_norm:
        return False
    tmeta = schema.tables.get(child_tbl)
    if tmeta is None:
        return False
    for edge in tmeta.foreign_keys:
        if phys_table_key(edge.dst_table) != phys_table_key(parent_tbl):
            continue
        if list(edge.src_cols) == child_cols_norm:
            return True
    if len(child_cols_norm) == 1:
        cm = tmeta.columns.get(child_cols_norm[0])
        if cm and cm.is_foreign_key and cm.fk_target:
            dst_tbl, _dst_col = cm.fk_target
            if phys_table_key(dst_tbl) == phys_table_key(parent_tbl):
                return True
    return False


def _parse_signature_segments(signature: list[str]) -> list[tuple[str, str, list[str], list[str]]]:
    segments: list[tuple[str, str, list[str], list[str]]] = []
    for seg in signature:
        item = str(seg).strip()
        if "->" not in item:
            continue
        left_part, right_part = item.split("->", 1)
        if "." not in left_part or "." not in right_part:
            continue
        left_tbl, left_cols = left_part.strip().split(".", 1)
        right_tbl, right_cols = right_part.strip().split(".", 1)
        lcols = [c.strip() for c in left_cols.split(",") if c.strip()]
        rcols = [c.strip() for c in right_cols.split(",") if c.strip()]
        if lcols and rcols:
            segments.append((left_tbl, right_tbl, lcols, rcols))
    return segments


def _join_step_multiplies_table(
    table: str,
    join_tbl: str,
    paired_tbl: str,
    join_cols: list[str],
    paired_cols: list[str],
    schema: SchemaGraph | None,
) -> bool:
    """Return True when attaching *join_tbl* to an existing row for *table* duplicates *table* rows."""
    if phys_table_key(table) != phys_table_key(paired_tbl):
        return False
    if _fk_edge_matches(join_tbl, paired_tbl, join_cols, paired_cols, schema):
        return True
    if schema is not None and _fk_edge_matches(paired_tbl, join_tbl, paired_cols, join_cols, schema):
        parent_unique = _join_columns_profiled_unique(schema, join_tbl, join_cols)
        if parent_unique is False:
            return True
        if parent_unique is True:
            return False
    if schema is not None:
        paired_unique = _join_columns_profiled_unique(schema, paired_tbl, paired_cols)
        if paired_unique is False:
            return True
    return True


def multiplying_edges_for_table(
    signature: list[str],
    table: str,
    schema: SchemaGraph | None,
    *,
    from_anchor: str | None = None,
) -> list[dict[str, str]]:
    """Return join signature segments that duplicate rows of *table* when traversed from *from_anchor*."""
    segments = _parse_signature_segments(signature)
    if not segments:
        return []
    anchor = from_anchor or segments[0][0]
    seen_logical: set[str] = {anchor}
    hits: list[dict[str, str]] = []
    for left_tbl, right_tbl, lcols, rcols in segments:
        li = left_tbl in seen_logical
        ri = right_tbl in seen_logical
        join_tbl: str
        paired_tbl: str
        join_cols: list[str]
        paired_cols: list[str]
        if li and not ri:
            join_tbl, paired_tbl, join_cols, paired_cols = right_tbl, left_tbl, rcols, lcols
            seen_logical.add(right_tbl)
        elif ri and not li:
            join_tbl, paired_tbl, join_cols, paired_cols = left_tbl, right_tbl, lcols, rcols
            seen_logical.add(left_tbl)
        elif li and ri:
            if phys_table_key(left_tbl) == phys_table_key(right_tbl):
                join_tbl, paired_tbl, join_cols, paired_cols = left_tbl, right_tbl, lcols, rcols
            elif phys_table_key(right_tbl) == phys_table_key(table):
                join_tbl, paired_tbl, join_cols, paired_cols = left_tbl, right_tbl, lcols, rcols
            else:
                join_tbl, paired_tbl, join_cols, paired_cols = right_tbl, left_tbl, rcols, lcols
        else:
            if left_tbl in seen_logical or phys_table_key(left_tbl) == phys_table_key(anchor):
                join_tbl, paired_tbl, join_cols, paired_cols = right_tbl, left_tbl, rcols, lcols
                seen_logical.add(right_tbl)
            elif right_tbl in seen_logical or phys_table_key(right_tbl) == phys_table_key(anchor):
                join_tbl, paired_tbl, join_cols, paired_cols = left_tbl, right_tbl, lcols, rcols
                seen_logical.add(left_tbl)
            elif phys_table_key(left_tbl) == phys_table_key(table):
                join_tbl, paired_tbl, join_cols, paired_cols = right_tbl, left_tbl, rcols, lcols
            elif phys_table_key(right_tbl) == phys_table_key(table):
                join_tbl, paired_tbl, join_cols, paired_cols = left_tbl, right_tbl, lcols, rcols
            else:
                continue
        if _join_step_multiplies_table(table, join_tbl, paired_tbl, join_cols, paired_cols, schema):
            hits.append(
                {
                    "edge": f"{left_tbl}.{','.join(lcols)}->{right_tbl}.{','.join(rcols)}",
                    "join_table": join_tbl,
                    "paired_table": paired_tbl,
                }
            )
    return hits


def iter_fan_out_sensitive_aggregates(expr: NormalizedExpr | None) -> list[tuple[str, str, str, bool]]:
    """Yield ``(agg_func, table, column, is_distinct_count)`` for aggregates sensitive to join fan-out."""
    if expr is None:
        return []
    out: list[tuple[str, str, str, bool]] = []

    def _from_group(group: MulGroup) -> None:
        func = (group.agg_func or "").strip().lower()
        if func not in FAN_OUT_SENSITIVE_AGG_FUNCS:
            return
        distinct = bool(group.distinct)
        if func == "count" and distinct:
            return
        for item in group.multiply or []:
            cols = extract_columns_from_expr(item) if isinstance(item, NormalizedExpr) else [str(item)]
            for col in cols:
                if "." not in col:
                    continue
                tbl, cn = col.split(".", 1)
                out.append((func, tbl, cn, distinct))

    for group in expr.add_groups or []:
        _from_group(group)
    return out


def _extend_fan_out_aggregates_from_expr(
    expr: NormalizedExpr | None,
    found: list[tuple[str, str, str, bool]],
) -> None:
    found.extend(iter_fan_out_sensitive_aggregates(expr))


def _collect_fan_out_from_case_registry(
    select_cols: list[SelectCol] | None,
    case_registry: list[CaseRegistryStep] | None,
    window_registry: list[WindowRegistryStep] | None,
    found: list[tuple[str, str, str, bool]],
) -> None:
    seen_registry_ids: set[str] = set()
    for sc in select_cols or []:
        ref = expr_registry_ref(sc.expr) or "" if sc.expr is not None else ""
        if not ref.startswith("c"):
            continue
        parts = effective_select_parts(sc, window_registry, case_registry)
        cw = parts.case_when
        if cw is None:
            continue
        seen_registry_ids.add(ref)
        for br in cw.branches or []:
            _extend_fan_out_aggregates_from_expr(br.result, found)
        if cw.else_result is not None:
            _extend_fan_out_aggregates_from_expr(cw.else_result, found)
    for step in case_registry or []:
        if step is None or step.case_when is None:
            continue
        rid = (step.registry_id or "").strip()
        if rid and rid in seen_registry_ids:
            continue
        cw = step.case_when
        for br in cw.branches or []:
            _extend_fan_out_aggregates_from_expr(br.result, found)
        if cw.else_result is not None:
            _extend_fan_out_aggregates_from_expr(cw.else_result, found)


def _collect_fan_out_from_query_body(
    *,
    select_cols: list[SelectCol] | None,
    having: PredicateGroup | None,
    order_by_cols: list[OrderByCol] | None,
    where: PredicateGroup | None,
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
    found: list[tuple[str, str, str, bool]],
) -> None:
    for sc in select_cols or []:
        _extend_fan_out_aggregates_from_expr(sc.expr, found)
    if having:
        for hp in having_leaves(having) or []:
            _extend_fan_out_aggregates_from_expr(hp.left_expr, found)
            _extend_fan_out_aggregates_from_expr(hp.right_expr, found)
    for ob in order_by_cols or []:
        _extend_fan_out_aggregates_from_expr(ob.expr, found)
    for fp in where_leaves(where) if where else []:
        _extend_fan_out_aggregates_from_expr(fp.left_expr, found)
        if fp.right_expr is not None:
            _extend_fan_out_aggregates_from_expr(fp.right_expr, found)
    for wr in window_registry or []:
        ws = wr.window_spec
        if ws.argument is not None:
            _extend_fan_out_aggregates_from_expr(ws.argument, found)
    _collect_fan_out_from_case_registry(select_cols, case_registry, window_registry, found)


def collect_fan_out_sensitive_aggregates(intent: RuntimeIntent) -> list[tuple[str, str, str, bool]]:
    """Collect fan-out-sensitive aggregates from the main query body and nested CTE steps."""
    found: list[tuple[str, str, str, bool]] = []
    _collect_fan_out_from_query_body(
        select_cols=intent.select_cols,
        having=intent.having,
        order_by_cols=intent.order_by_cols,
        where=intent.where,
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
        found=found,
    )
    for cte in intent.cte_steps or []:
        _collect_fan_out_from_query_body(
            select_cols=cte.select_cols,
            having=cte.having,
            order_by_cols=cte.order_by_cols,
            where=cte.where,
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            found=found,
        )
    return found


def _expr_has_unqualified_count_star(expr: NormalizedExpr | None) -> bool:
    """Return True when *expr* is ``COUNT(*)`` without ``DISTINCT``."""
    if expr is None:
        return False

    def _from_group(group: MulGroup) -> bool:
        func = (group.agg_func or "").strip().lower()
        if func != "count" or group.distinct:
            return False
        for item in group.multiply or []:
            if isinstance(item, NormalizedExpr) and (item.star or item.column_ref == "*"):
                return True
        return False

    return any(_from_group(group) for group in expr.add_groups or [])


def _anchor_table_multiplied(
    signature: list[str],
    anchor: str | None,
    schema: SchemaGraph | None,
) -> tuple[bool, str]:
    """Return whether *anchor* rows are duplicated on *signature* and the first multiplying edge."""
    if not signature or not anchor:
        return False, ""
    hits = multiplying_edges_for_table(signature, anchor, schema, from_anchor=anchor)
    if not hits:
        return False, ""
    return True, hits[0]["edge"]


def validate_clause_widened_rowset(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    context: str,
    *,
    join_signature: list[str] | None = None,
    from_anchor: str | None = None,
) -> list[IntentIssue]:
    """Emit diagnostics or refusals when clause results depend on a join-widened row set."""
    signature = list(join_signature or intent.chosen_join_path_signature or [])
    if not signature:
        return []
    if (intent.grain or "row_level") != "row_level" or intent.group_by_cols:
        return []
    anchor = from_anchor or (intent.tables[0] if intent.tables else None)
    multiplied, edge = _anchor_table_multiplied(signature, anchor, schema)
    if not multiplied:
        return []

    ctx_key = context.replace(" ", "_")
    issues: list[IntentIssue] = []
    main_tables = {phys_table_key(t) for t in intent_join_reachability_tables(intent)}

    if intent.limit is not None or (intent.limit_param_key or "").strip():
        issues.append(
            make_intent_issue(
                issue_id=f"clause_widened_rowset_limit_{ctx_key}",
                category=FailureCategory.WRONG_SORT_OR_LIMIT,
                severity="error",
                message=(
                    f"{context}: LIMIT applies to physical rows after join edge {edge!r} multiplies "
                    f"{anchor!r}, so the row cap does not reflect entity cardinality. Group or probe "
                    f"without multiplying before limiting."
                ),
                context={"edge": edge, "anchor": anchor, "location": context},
            )
        )

    if intent.order_by_cols:
        issues.append(
            make_intent_issue(
                issue_id=f"clause_widened_rowset_order_by_{ctx_key}",
                category=FailureCategory.WRONG_SORT_OR_LIMIT,
                severity="warning",
                message=(
                    f"{context}: ORDER BY runs on rows duplicated by join edge {edge!r}, so sort order is "
                    f"nondeterministic at the {anchor!r} grain."
                ),
                context={"edge": edge, "anchor": anchor, "location": context},
            )
        )

    if intent.distinct_select_index >= 0:
        issues.append(
            make_intent_issue(
                issue_id=f"clause_widened_rowset_select_distinct_{ctx_key}",
                category=FailureCategory.SELECT_VALIDITY,
                severity="warning",
                message=(
                    f"{context}: SELECT DISTINCT deduplicates the join-widened row set from edge {edge!r}; "
                    f"the distinct key may not match the intended {anchor!r} grain."
                ),
                context={"edge": edge, "anchor": anchor, "location": context},
            )
        )

    for idx, expr in enumerate(intent.distinct_on or []):
        multiplied_cols: list[str] = []
        for col in extract_columns_from_expr(expr):
            if "." not in col:
                continue
            tbl, _cn = col.split(".", 1)
            if phys_table_key(tbl) not in main_tables:
                continue
            if multiplying_edges_for_table(signature, tbl, schema, from_anchor=anchor):
                multiplied_cols.append(col)
        if multiplied_cols:
            issues.append(
                make_intent_issue(
                    issue_id=f"clause_widened_rowset_distinct_on_{ctx_key}_{idx}",
                    category=FailureCategory.ORDER_BY_VALIDITY,
                    severity="error",
                    message=(
                        f"{context}: distinct_on partition references {', '.join(multiplied_cols)} on rows "
                        f"duplicated by join edge {edge!r}. Partition on the non-multiplied side or group first."
                    ),
                    context={
                        "edge": edge,
                        "anchor": anchor,
                        "columns": multiplied_cols,
                        "partition_index": idx,
                        "location": context,
                    },
                )
            )

    count_star = any(_expr_has_unqualified_count_star(sc.expr) for sc in intent.select_cols or [])
    if not count_star and intent.having:
        for hp in having_leaves(intent.having) or []:
            if _expr_has_unqualified_count_star(hp.left_expr) or _expr_has_unqualified_count_star(hp.right_expr):
                count_star = True
                break
    if count_star:
        issues.append(
            make_intent_issue(
                issue_id=f"clause_widened_rowset_count_star_{ctx_key}",
                category=FailureCategory.AGGREGATION_SEMANTICS,
                severity="warning",
                message=(
                    f"{context}: COUNT(*) counts physical rows duplicated by join edge {edge!r}, which may not "
                    f"equal the number of {anchor!r} entities. Use COUNT(DISTINCT {anchor}.<pk>) when entity "
                    f"cardinality is intended."
                ),
                context={"edge": edge, "anchor": anchor, "location": context},
            )
        )

    return issues


def _parse_signature_edges(
    signature: list[str],
    edge_kinds: list[str],
) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for idx, seg in enumerate(signature):
        item = str(seg).strip()
        if "->" not in item:
            continue
        left_part, right_part = item.split("->", 1)
        if "." not in left_part or "." not in right_part:
            continue
        left_tbl = left_part.strip().split(".", 1)[0].strip()
        right_tbl = right_part.strip().split(".", 1)[0].strip()
        kind = edge_kinds[idx] if idx < len(edge_kinds) else ""
        edges.append((left_tbl, right_tbl, kind))
    return edges


def _shortest_path_tables(
    edges: list[tuple[str, str, str]],
    start: str,
    end: str,
) -> tuple[list[str], list[tuple[str, str, str]]] | None:
    if phys_table_key(start) == phys_table_key(end):
        return [start], []
    adj: dict[str, list[tuple[str, str, str]]] = {}
    for left_tbl, right_tbl, kind in edges:
        adj.setdefault(phys_table_key(left_tbl), []).append((left_tbl, right_tbl, kind))
        adj.setdefault(phys_table_key(right_tbl), []).append((right_tbl, left_tbl, kind))
    start_key = phys_table_key(start)
    end_key = phys_table_key(end)
    if start_key not in adj or end_key not in adj:
        return None
    queue: deque[tuple[str, list[str], list[tuple[str, str, str]]]] = deque([(start_key, [start], [])])
    visited: set[str] = {start_key}
    while queue:
        node_key, path_tables, path_edges = queue.popleft()
        for left_tbl, right_tbl, kind in adj.get(node_key, []):
            nxt_key = phys_table_key(right_tbl if phys_table_key(left_tbl) == node_key else left_tbl)
            nxt_tbl = right_tbl if phys_table_key(left_tbl) == node_key else left_tbl
            if nxt_key in visited:
                continue
            next_path = path_tables + [nxt_tbl]
            next_edges = path_edges + [(left_tbl, right_tbl, kind)]
            if nxt_key == end_key:
                return next_path, next_edges
            visited.add(nxt_key)
            queue.append((nxt_key, next_path, next_edges))
    return None


def _comparison_pairs_for_table(
    *,
    comparison_table: str,
    where_params: Sequence[Any] | None,
    having_params: Sequence[Any] | None,
) -> list[tuple[str, str, str, str]]:
    pairs: list[tuple[str, str, str, str]] = []
    comp_key = phys_table_key(comparison_table)

    def consider(left_expr: NormalizedExpr | None, right_expr: NormalizedExpr | None) -> None:
        if left_expr is None or right_expr is None:
            return
        left_cols = extract_columns_from_expr(left_expr)
        right_cols = extract_columns_from_expr(right_expr)
        for lc in left_cols:
            if "." not in lc:
                continue
            lt, lcol = lc.split(".", 1)
            for rc in right_cols:
                if "." not in rc:
                    continue
                rt, rcol = rc.split(".", 1)
                if phys_table_key(rt) == comp_key:
                    pairs.append((lt, lcol, rt, rcol))
                elif phys_table_key(lt) == comp_key:
                    pairs.append((rt, rcol, lt, lcol))

    for fp in where_params or []:
        consider(fp.left_expr, fp.right_expr)
    for hp in having_params or []:
        consider(hp.left_expr, hp.right_expr)
    return pairs


def _path_uses_semantic_overlap(path_edges: list[tuple[str, str, str]]) -> bool:
    for _left, _right, kind in path_edges:
        if kind in JOIN_PATH_EDGE_KIND_WHERE_BUCKET:
            return True
    return False


def emit_comparison_join_detour_diagnostics(
    *,
    scope_label: str,
    comparison_only: list[str],
    scope_tables: list[str],
    signature: list[str],
    edge_kinds: list[str],
    from_anchor: str | None,
    where_params: Sequence[Any] | None,
    having_params: Sequence[Any] | None,
) -> None:
    """Emit COMPARISON_JOIN_DETOUR when a comparison-only table joins through a short bridge."""
    if not comparison_only or not signature:
        return
    anchor = from_anchor or (scope_tables[0] if scope_tables else "")
    if not anchor:
        return
    edges = _parse_signature_edges(signature, edge_kinds)
    comp_set = {
        phys_table_key(t) for t in comparison_only if phys_table_key(t) in {phys_table_key(x) for x in scope_tables}
    }
    for comp_tbl in sorted(comparison_only, key=str.lower):
        if phys_table_key(comp_tbl) not in comp_set:
            continue
        resolved = _shortest_path_tables(edges, anchor, comp_tbl)
        if resolved is None:
            continue
        path_tables, path_edges = resolved
        hop_count = len(path_edges)
        max_hops = int(PolicyConfig.JOIN_COMPARISON_SCOPE_MAX_HOPS)
        if hop_count <= 1 or hop_count > max_hops:
            continue
        intermediates = [t for t in path_tables[1:-1] if t]
        if not intermediates:
            continue
        pairs = _comparison_pairs_for_table(
            comparison_table=comp_tbl,
            where_params=where_params,
            having_params=having_params,
        )
        pair_text = (
            ", ".join(f"{lt}.{lc} compared to {rt}.{rc}" for lt, lc, rt, rc in pairs[:2])
            if pairs
            else f"comparison involving {comp_tbl}"
        )
        notify(
            (
                f"Cross-table comparison in {scope_label} connects through "
                f"{hop_count} join hop(s) via {', '.join(intermediates)} ({pair_text}). "
                "The comparison does not assert this relationship."
            ),
            stage="join",
            code=DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR,
            level="info",
            details=(
                ("comparison_table", comp_tbl),
                ("hop_count", str(hop_count)),
                ("intermediates", ", ".join(intermediates)),
            ),
        )


def validate_comparison_join_scope(
    *,
    scope_label: str,
    scope_tables: list[str],
    comparison_only: list[str],
    signature: list[str],
    edge_kinds: list[str],
    from_anchor: str | None,
    where_params: Sequence[Any] | None,
    having_params: Sequence[Any] | None,
) -> list[IntentIssue]:
    """Refuse comparison-only tables joined through long or semantic- inferred paths."""
    issues: list[IntentIssue] = []
    if not comparison_only or len(scope_tables) < 2:
        return issues
    anchor = from_anchor or (scope_tables[0] if scope_tables else "")
    if not anchor or not signature:
        return issues
    edges = _parse_signature_edges(signature, edge_kinds)
    max_hops = int(PolicyConfig.JOIN_COMPARISON_SCOPE_MAX_HOPS)
    scope_keys = {phys_table_key(t) for t in scope_tables}
    for comp_tbl in sorted(comparison_only, key=str.lower):
        if phys_table_key(comp_tbl) not in scope_keys:
            continue
        resolved = _shortest_path_tables(edges, anchor, comp_tbl)
        if resolved is None:
            continue
        path_tables, path_edges = resolved
        hop_count = len(path_edges)
        intermediates = [t for t in path_tables[1:-1] if t]
        pairs = _comparison_pairs_for_table(
            comparison_table=comp_tbl,
            where_params=where_params,
            having_params=having_params,
        )
        pair_text = (
            ", ".join(f"{lt}.{lc} and {rt}.{rc}" for lt, lc, rt, rc in pairs[:2]) if pairs else f"table {comp_tbl}"
        )
        if _path_uses_semantic_overlap(path_edges):
            msg = (
                f"Cross-table comparison between {pair_text} can only be joined through a "
                f"profile-inferred relationship"
                + (f" via {', '.join(intermediates)}" if intermediates else "")
                + ". The comparison does not imply a relationship. Declare foreign_keys_add or a "
                "semantic override when the relationship is real."
            )
            issues.append(
                make_intent_issue(
                    issue_id="comparison_join_semantic_path",
                    message=msg,
                    severity="error",
                    category=FailureCategory.WRONG_JOIN,
                )
            )
            continue
        if hop_count > max_hops:
            msg = (
                f"Cross-table comparison between {pair_text} requires {hop_count} join hops"
                + (f" through {', '.join(intermediates)}" if intermediates else "")
                + f", exceeding the limit of {max_hops}. The comparison does not imply a relationship. "
                "Declare foreign_keys_add or a semantic override when the relationship is real."
            )
            issues.append(
                make_intent_issue(
                    issue_id="comparison_join_hop_ceiling",
                    message=msg,
                    severity="error",
                    category=FailureCategory.WRONG_JOIN,
                )
            )
    return issues


def validate_comparison_join_scope_or_raise(
    *,
    scope_label: str,
    scope_tables: list[str],
    comparison_only: list[str],
    signature: list[str],
    edge_kinds: list[str],
    from_anchor: str | None,
    where_params: Sequence[Any] | None = None,
    having_params: Sequence[Any] | None = None,
) -> None:
    """Raise ComparisonJoinScopeExceededError when comparison-only scope exceeds hop or semantic limits."""
    issues = validate_comparison_join_scope(
        scope_label=scope_label,
        scope_tables=scope_tables,
        comparison_only=comparison_only,
        signature=signature,
        edge_kinds=edge_kinds,
        from_anchor=from_anchor,
        where_params=where_params,
        having_params=having_params,
    )
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ComparisonJoinScopeExceededError(scope_label, errors[0].message)
    emit_comparison_join_detour_diagnostics(
        scope_label=scope_label,
        comparison_only=comparison_only,
        scope_tables=scope_tables,
        signature=signature,
        edge_kinds=edge_kinds,
        from_anchor=from_anchor,
        where_params=where_params,
        having_params=having_params,
    )
