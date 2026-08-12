"""Intent-level validation for grain, aggregations, WHERE/HAVING placement, CTE consistency, and related repairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._constants import (
    COMPATIBLE_TYPE_PAIRS,
    DATE_FRIENDLY_VALUE_TYPES,
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR,
    FAN_OUT_SENSITIVE_AGG_FUNCS,
    NUMERIC_COMPARE_OPS_ORDERED,
    NUMERIC_ONLY_AGGREGATIONS,
    NUMERIC_RESULT_AGGS,
    NUMERIC_RESULT_SCALARS,
    QUESTION_YEAR_IN_STRING_RE,
    SCALAR_FUNCTIONS_NUMERIC,
    SCALAR_FUNCTIONS_STRING,
    SCALAR_FUNCTIONS_TEMPORAL,
    SHAPE_FORM_NUM_REGEX,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_GRAINS,
    VALID_HAVING_OPS,
)
from ._constants_runtime import (
    REFUSAL_CATALOGUE,
    REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
)
from ._contracts_base import (
    FailureCategory,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    SensitivityClassification,
    WhereParam,
)
from ._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from ._contracts_schema import (
    CaseRegistryStep,
    ColumnRole,
    CteOutputColumnMeta,
    IntentIssue,
    LogicalIntent,
    SchemaGraph,
    WindowRegistryStep,
)
from ._intent_expr import (
    concat_logical_intent_prose,
    expr_has_arithmetic,
    extract_agg_col,
    extract_col_from_scalar_wrapper,
    extract_columns_from_expr,
    extract_functions_from_term,
    get_col_meta,
    get_col_type,
    intent_join_reachability_tables,
    is_col_arithmetic_role,
    is_col_numeric,
    strip_leading_distinct_from_column_ref,
)
from ._schema_graph import (
    fk_infer_value_types_compatible,
)
from ._utils import (
    column_metadata_timezone_awareness_mismatch,
    debug,
    emit_session_refusal_diagnostic,
    stable_json,
)
from ._validation_shape import (
    anchor_table_multiplied,
    expr_has_unqualified_count_star,
    is_date_column_subtraction,
    is_date_integer_day_arithmetic,
    multiplying_edges_for_table,
    parse_signature_segments,
    phys_table_key,
)


def _column_meta_or_none(schema: SchemaGraph, ref: str) -> Any:
    """Look up column metadata for a qualified ``table.column`` reference, or return None."""
    if "." not in ref:
        return None
    parts = ref.split(".", 1)
    if len(parts) != 2:
        return None
    return schema.get_column(parts[0], parts[1])


def _emit_denied_column_refusal(table: str, column: str, location: str) -> None:
    """Record a denied-column refusal with audit metadata hidden from the user message."""
    emit_session_refusal_diagnostic(
        DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
        REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
        details=(("table", table), ("column", column), ("location", location)),
        subject=f"{table}.{column}",
    )


def validate_deny_bare_select(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Reject bare (non-aggregated) ``select_cols`` entries that reference denied columns. Reads each column's ``is_denied`` flag (canonical source set during reflection from ``EngineContext.deny_columns``). Filters, ``group_by_cols``, and aggregated selects are not checked here — see ``validate_denied_references`` and ``validate_sensitivity_group_by`` for those gates. Every CTE body is scanned, including probe and intermediate CTEs not listed in ``intent.tables``."""
    issues: list[IntentIssue] = []

    def _scan_select_cols(select_cols: list[SelectCol], context: str) -> None:
        for idx, sc in enumerate(select_cols or []):
            if sc.is_aggregated:
                continue
            for ref in extract_columns_from_expr(sc.expr):
                meta = _column_meta_or_none(schema, ref)
                if meta is None or not meta.is_denied:
                    continue
                t, c = ref.split(".", 1)
                issues.append(
                    IntentIssue.make(
                        issue_id=f"deny_bare_select_{context}_{idx}",
                        category=FailureCategory.DENY_BARE_SELECT,
                        severity="error",
                        message=REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
                        context={"table": t, "column": c, "location": context},
                    )
                )
                _emit_denied_column_refusal(t, c, context)

    _scan_select_cols(intent.select_cols or [], "main query")
    for cte in intent.cte_steps or []:
        _scan_select_cols(cte.select_cols or [], f"CTE {cte.cte_name}")
    return issues


def _scan_norm_expr_refs(expr: NormalizedExpr | None) -> list[str]:
    """Return qualified column references from a normalized expression, empty when None."""
    if expr is None:
        return []
    return [ref for ref in extract_columns_from_expr(expr) if "." in ref]


def validate_denied_references(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Reject any reference to a denied column in filters, group-by, having, order-by, or aggregated selects (including each CTE body's ORDER BY list). Bare select projection is gated separately by ``validate_deny_bare_select``. This validator covers every other surface so a denied column never reaches generated SQL even when wrapped in COUNT, used as a WHERE predicate, or appears in GROUP BY / ORDER BY / HAVING. Every CTE body is scanned, including probe and intermediate CTEs not listed in ``intent.tables``."""
    issues: list[IntentIssue] = []

    def _emit(ref: str, location: str, idx: int) -> None:
        t, c = ref.split(".", 1)
        issues.append(
            IntentIssue.make(
                issue_id=f"denied_reference_{location}_{idx}_{t}_{c}",
                category=FailureCategory.DENIED_REFERENCE,
                severity="error",
                message=REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
                context={"table": t, "column": c, "location": location},
            )
        )
        _emit_denied_column_refusal(t, c, location)

    def _scan_intent(
        select_cols: list[SelectCol],
        filters: list[WhereParam],
        group_by: list[NormalizedExpr],
        having: list[HavingParam],
        order_by: list[OrderByCol],
        location: str,
    ) -> None:
        for idx, sc in enumerate(select_cols or []):
            if not sc.is_aggregated or sc.expr is None:
                continue
            for ref in extract_columns_from_expr(sc.expr):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} aggregated select", idx)
        for idx, fp in enumerate(filters or []):
            for ref in _scan_norm_expr_refs(fp.left_expr):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} filter", idx)
            for ref in _scan_norm_expr_refs(fp.right_expr):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} filter", idx)
        for idx, gb in enumerate(group_by or []):
            for ref in _scan_norm_expr_refs(gb):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} group_by", idx)
        for idx, hp in enumerate(having or []):
            for ref in _scan_norm_expr_refs(hp.left_expr):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} having", idx)
            for ref in _scan_norm_expr_refs(hp.right_expr):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} having", idx)
        for idx, ob in enumerate(order_by or []):
            obe = getattr(ob, "expr", None)
            for ref in _scan_norm_expr_refs(obe):
                meta = _column_meta_or_none(schema, ref)
                if meta is not None and meta.is_denied:
                    _emit(ref, f"{location} order_by", idx)

    _scan_intent(
        intent.select_cols or [],
        PredicateGroup.where_leaves(intent.where) or [],
        intent.group_by_cols or [],
        PredicateGroup.having_leaves(intent.having) or [],
        intent.order_by_cols or [],
        "main query",
    )
    for cte in intent.cte_steps or []:
        _scan_intent(
            cte.select_cols or [],
            PredicateGroup.where_leaves(cte.where) or [],
            cte.group_by_cols or [],
            PredicateGroup.having_leaves(cte.having) or [],
            cte.order_by_cols or [],
            f"CTE {cte.cte_name}",
        )
    return issues


def validate_sensitivity_group_by(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Reject GROUP BY entries that reference **hidden** columns. Grouping by such columns exposes distinct sensitive values. **Restricted** columns follow separate rules in selectability validation. WHERE, JOIN, and aggregation surfaces remain permitted where policy allows."""
    issues: list[IntentIssue] = []

    def _scan(group_by: list[NormalizedExpr], location: str) -> None:
        for idx, gb in enumerate(group_by or []):
            for ref in _scan_norm_expr_refs(gb):
                meta = _column_meta_or_none(schema, ref)
                if meta is None:
                    continue
                if meta.sensitivity not in (SensitivityClassification.RESTRICTED, SensitivityClassification.HIDDEN):
                    continue
                t, c = ref.split(".", 1)
                issues.append(
                    IntentIssue.make(
                        issue_id=f"sensitive_group_by_{location}_{idx}_{t}_{c}",
                        category=FailureCategory.SENSITIVE_GROUP_BY,
                        severity="error",
                        message=REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
                        context={"table": t, "column": c, "location": location},
                    )
                )
                _emit_denied_column_refusal(t, c, location)

    _scan(intent.group_by_cols or [], "main query")
    main_names = {t.lower() for t in intent.tables or []}
    for cte in intent.cte_steps or []:
        if cte.cte_name.lower() not in main_names:
            continue
        _scan(cte.group_by_cols or [], f"CTE {cte.cte_name}")
    return issues


def validate_sensitivity_order_by(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Reject ORDER BY entries that reference **restricted** or **hidden** columns when policy blocks bare ordering on them."""
    issues: list[IntentIssue] = []

    def _scan(order_by: list[OrderByCol], location: str) -> None:
        for idx, ob in enumerate(order_by or []):
            obe = getattr(ob, "expr", None)
            for ref in _scan_norm_expr_refs(obe):
                meta = _column_meta_or_none(schema, ref)
                if meta is None:
                    continue
                if meta.sensitivity not in (SensitivityClassification.RESTRICTED, SensitivityClassification.HIDDEN):
                    continue
                t, c = ref.split(".", 1)
                issues.append(
                    IntentIssue.make(
                        issue_id=f"sensitive_order_by_{location}_{idx}_{t}_{c}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
                        context={"table": t, "column": c, "location": location},
                    )
                )
                _emit_denied_column_refusal(t, c, location)

    _scan(intent.order_by_cols or [], "main query")
    main_names = {t.lower() for t in intent.tables or []}
    for cte in intent.cte_steps or []:
        if cte.cte_name.lower() not in main_names:
            continue
        _scan(cte.order_by_cols or [], f"CTE {cte.cte_name}")
    return issues


def _issues_for_non_selectable_expr(
    schema: SchemaGraph,
    expr: NormalizedExpr | None,
    *,
    location: str,
    surface: str,
    issue_tag: str,
    id_suffix: str = "",
) -> list[IntentIssue]:
    """Build terminal issues for qualified column references that are not selectable in *expr*."""
    issues: list[IntentIssue] = []
    if expr is None:
        return issues
    exempt = selectability_exempt_qualified_refs(expr, schema)
    for ref in _scan_norm_expr_refs(expr):
        if ref in exempt:
            continue
        meta = _column_meta_or_none(schema, ref)
        if meta is None or meta.is_selectable:
            continue
        t, c = ref.split(".", 1)
        msg = REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE
        category = FailureCategory.WHERE_VALIDITY if surface == "WHERE" else FailureCategory.HAVING_SEMANTIC
        suf = f"_{id_suffix}" if id_suffix else ""
        issues.append(
            IntentIssue.make(
                issue_id=f"{issue_tag}{suf}_{location}_{t}_{c}",
                category=category,
                severity="error",
                message=msg,
                context={
                    "table": t,
                    "column": c,
                    "location": location,
                    "surface": surface,
                },
            )
        )
        _emit_denied_column_refusal(t, c, location)
    return issues


def validate_non_selectable_predicates(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Emit terminal issues when ``WHERE`` or ``HAVING`` references columns that are not selectable."""
    issues: list[IntentIssue] = []

    def _scan_filters(filters: list[WhereParam], location: str) -> None:
        for fi, fp in enumerate(filters or []):
            op_cmp = (fp.op or "").strip().lower()
            if op_cmp not in ("is null", "is not null"):
                issues.extend(
                    _issues_for_non_selectable_expr(
                        schema,
                        fp.left_expr,
                        location=location,
                        surface="WHERE",
                        issue_tag="non_selectable_where",
                        id_suffix=f"{fi}_L",
                    )
                )
            if fp.right_expr is not None:
                issues.extend(
                    _issues_for_non_selectable_expr(
                        schema,
                        fp.right_expr,
                        location=location,
                        surface="WHERE",
                        issue_tag="non_selectable_where",
                        id_suffix=f"{fi}_R",
                    )
                )

    def _scan_having(having: list[HavingParam], location: str) -> None:
        for hi, hp in enumerate(having or []):
            issues.extend(
                _issues_for_non_selectable_expr(
                    schema,
                    hp.left_expr,
                    location=location,
                    surface="HAVING",
                    issue_tag="non_selectable_having",
                    id_suffix=f"{hi}_L",
                )
            )
            issues.extend(
                _issues_for_non_selectable_expr(
                    schema,
                    hp.right_expr,
                    location=location,
                    surface="HAVING",
                    issue_tag="non_selectable_having",
                    id_suffix=f"{hi}_R",
                )
            )

    _scan_filters(PredicateGroup.where_leaves(intent.where) or [], "main query")
    _scan_having(PredicateGroup.having_leaves(intent.having) or [], "main query")
    main_names = {t.lower() for t in intent.tables or []}
    for cte in intent.cte_steps or []:
        if cte.cte_name.lower() not in main_names:
            continue
        loc = f"CTE {cte.cte_name}"
        _scan_filters(PredicateGroup.where_leaves(cte.where) or [], loc)
        _scan_having(PredicateGroup.having_leaves(cte.having) or [], loc)
    return issues


def _where_conjunct_groups(where: PredicateGroup | list[WhereParam] | None) -> list[list[WhereParam]]:
    """Partition WHERE predicates into AND-conjuncts from a predicate tree or flat list."""
    if where is None:
        return []
    if isinstance(where, PredicateGroup):
        if where.op == "or":
            groups: list[list[WhereParam]] = []
            for pred in where.predicates:
                if isinstance(pred, WhereParam):
                    groups.append([pred])
            for child in where.groups:
                groups.extend(_where_conjunct_groups(child))
            return groups
        filters = [p for p in where.leaves() if isinstance(p, WhereParam)]
        if where.groups:
            groups = [filters] if filters else []
            for child in where.groups:
                groups.extend(_where_conjunct_groups(child))
            return groups
        return [filters] if filters else []
    filters = list(where)
    if not filters:
        return []
    return [filters]


def _primary_where_column_key(fp: WhereParam) -> str | None:
    """Return a single qualified ``table.column`` key for simple column filters, else ``None``."""
    refs = [r for r in extract_columns_from_expr(fp.left_expr) if "." in r]
    if len(refs) != 1:
        return None
    return refs[0].lower()


def _scalar_norm_for_window_bounds(val: Any) -> str:
    """Deterministic normalisation for comparing window endpoints."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, bool):
        return str(val)
    if val is None:
        return ""
    if isinstance(val, int | float):
        return repr(val)
    return stable_json(val) if isinstance(val, dict | list | tuple) else str(val).strip().lower()


def _where_bound_norm(fp: WhereParam, pv: Mapping[str, Any] | None) -> str | None:
    """Comparable bound payload for inequality / ``between`` window checks."""
    if fp.op == "between" and fp.param_key and fp.param_key_hi:
        store = pv or {}
        lo = store.get(fp.param_key)
        hi = store.get(fp.param_key_hi)
        return f"between:{_scalar_norm_for_window_bounds(lo)}:{_scalar_norm_for_window_bounds(hi)}"
    if fp.op == "between" and fp.raw_value is not None:
        rv = fp.raw_value
        if isinstance(rv, (list, tuple)) and len(rv) == 2:
            return f"between:{_scalar_norm_for_window_bounds(rv[0])}:{_scalar_norm_for_window_bounds(rv[1])}"
    if fp.right_expr is not None:
        return f"expr:{fp.right_expr.signature_key}"
    rv = fp.resolved_value(pv)
    if fp.value_type == "date_window" and isinstance(rv, dict) and "start" in rv and "end" in rv:
        return stable_json(
            {
                "vt": "date_window_range",
                "start": _scalar_norm_for_window_bounds(rv.get("start")),
                "end": _scalar_norm_for_window_bounds(rv.get("end")),
            }
        )
    if fp.value_type == "date_window" and isinstance(rv, dict):
        unit = str(rv.get("unit", "day") or "day").lower()
        amt = rv.get("amount")
        try:
            amount = int(amt) if amt is not None else 0
        except (TypeError, ValueError):
            amount = 0
        return stable_json({"vt": "date_window_rel", "unit": unit, "amount": amount})
    if rv is not None or fp.param_key:
        return f"val:{fp.value_type}:{_scalar_norm_for_window_bounds(rv)}"
    return None


def validate_empty_window(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Reject temporal windows whose lower and upper bounds collapse to the same normalised expression. Covers explicit ``start``/``end`` ``date_window`` payloads, ``BETWEEN`` with identical endpoints, and separate ``>=`` / ``<=`` filters on one column inside the same AND-conjunct."""
    _ = schema
    issues: list[IntentIssue] = []

    def _scan_filters(filters: list[WhereParam], pv: Mapping[str, Any] | None, loc: str) -> None:
        for fp in filters or []:
            if fp.value_type == "date_window":
                rv = fp.resolved_value(pv)
                if isinstance(rv, dict) and "start" in rv and "end" in rv:
                    if _scalar_norm_for_window_bounds(rv.get("start")) == _scalar_norm_for_window_bounds(rv.get("end")):
                        col = _primary_where_column_key(fp) or "unknown column"
                        issues.append(
                            IntentIssue.make(
                                issue_id=f"intent_empty_window_{loc}_date_window_range",
                                category=FailureCategory.INTENT_EMPTY_WINDOW,
                                severity="error",
                                message=f"{loc}: empty temporal window on {col} (identical start and end bounds)",
                                context={"location": loc, "column": col},
                            )
                        )
            if fp.op == "between":
                bn = _where_bound_norm(fp, pv)
                if bn and bn.startswith("between:"):
                    parts = bn.split(":", 2)
                    if len(parts) == 3 and parts[1] == parts[2] and parts[1] != "":
                        col = _primary_where_column_key(fp) or "unknown column"
                        issues.append(
                            IntentIssue.make(
                                issue_id=f"intent_empty_window_{loc}_between",
                                category=FailureCategory.INTENT_EMPTY_WINDOW,
                                severity="error",
                                message=f"{loc}: empty temporal window on {col} (identical BETWEEN bounds)",
                                context={"location": loc, "column": col},
                            )
                        )
        for conj in _where_conjunct_groups(list(filters or [])):
            by_col: dict[str, dict[str, set[str]]] = {}
            for fp in conj:
                col_key = _primary_where_column_key(fp)
                if col_key is None:
                    continue
                op = (fp.op or "").strip().lower()
                if op not in (">=", ">", "<=", "<"):
                    continue
                bn = _where_bound_norm(fp, pv)
                if bn is None:
                    continue
                bucket = by_col.setdefault(col_key, {"lower": set(), "upper": set()})
                if op in (">=", ">"):
                    bucket["lower"].add(bn)
                if op in ("<=", "<"):
                    bucket["upper"].add(bn)
            for col, sides in by_col.items():
                lows = sides["lower"]
                ups = sides["upper"]
                if any(low == up for low in lows for up in ups):
                    issues.append(
                        IntentIssue.make(
                            issue_id=f"intent_empty_window_{loc}_{col.replace('.', '_')}",
                            category=FailureCategory.INTENT_EMPTY_WINDOW,
                            severity="error",
                            message=f"{loc}: empty temporal window on {col} (identical lower and upper bound expressions)",
                            context={"location": loc, "column": col},
                        )
                    )

    pv_main = intent.param_values or {}
    _scan_filters(PredicateGroup.where_leaves(intent.where) or [], pv_main, "main query")
    main_names = {t.lower() for t in intent.tables or []}
    for cte in intent.cte_steps or []:
        if cte.cte_name.lower() not in main_names:
            continue
        pv_cte = cte.param_values or pv_main
        _scan_filters(PredicateGroup.where_leaves(cte.where) or [], pv_cte, f"CTE {cte.cte_name}")
    return issues


def validate_grain_consistency(
    grain: str,
    select_cols: list[SelectCol],
    group_by_cols: list[NormalizedExpr],
    having_param: list[HavingParam],
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that the declared grain is consistent with aggregation. and GROUP BY presence."""
    issues = []
    debug(
        f"[validation_rules.validate_grain_consistency] grain={grain}, group_by={len(group_by_cols)}, having={len(having_param)}"
    )
    if grain not in VALID_GRAINS:
        issues.append(
            IntentIssue.make(
                issue_id=f"invalid_grain_{grain}",
                category=FailureCategory.GRAIN_VALIDITY,
                severity="error",
                message=f"Invalid grain value '{grain}'. Allowed: {', '.join(sorted(VALID_GRAINS))}",
                context={"grain": grain, "location": context},
            )
        )
        return issues
    has_agg = any(sc.is_aggregated for sc in select_cols)
    has_group_by = bool(group_by_cols)
    has_having = bool(having_param)
    if grain == "grouped" and not has_group_by:
        issues.append(
            IntentIssue.make(
                issue_id=f"grouped_without_group_by_{context}",
                category=FailureCategory.GRAIN_CONSISTENCY,
                severity="error",
                message=f"Grouped grain without GROUP BY columns in {context}",
                context={"grain": grain, "location": context},
            )
        )
        debug("[validation_rules.validate_grain_consistency] grouped grain without group_by")
    if grain in {"scalar", "row_level"} and has_group_by:
        issues.append(
            IntentIssue.make(
                issue_id=f"group_by_with_{grain}_{context}",
                category=FailureCategory.GRAIN_CONSISTENCY,
                severity="error",
                message=f"GROUP BY columns present but grain={grain} in {context}",
                context={
                    "grain": grain,
                    "group_by": [g.primary_column for g in group_by_cols],
                    "location": context,
                },
            )
        )
        debug(f"[validation_rules.validate_grain_consistency] group_by present but grain={grain}")
    if has_agg and grain == "row_level":
        agg_funcs = [sc.expr.primary_term for sc in select_cols if sc.is_aggregated]
        issues.append(
            IntentIssue.make(
                issue_id=f"agg_with_row_level_{context}_{','.join(agg_funcs)}",
                category=FailureCategory.GRAIN_CONSISTENCY,
                severity="error",
                message=f"Aggregation functions {agg_funcs} with row_level grain in {context}",
                context={"agg_funcs": agg_funcs, "grain": grain, "location": context},
            )
        )
        debug("[validation_rules.validate_grain_consistency] agg funcs with row_level grain")
    if has_having and grain not in {"grouped", "scalar"}:
        issues.append(
            IntentIssue.make(
                issue_id=f"having_without_agg_{grain}_{context}",
                category=FailureCategory.GRAIN_CONSISTENCY,
                severity="error",
                message=f"HAVING conditions without aggregation: grain={grain} but HAVING present in {context}",
                context={
                    "grain": grain,
                    "having_count": len(having_param),
                    "location": context,
                },
            )
        )
        debug(f"[validation_rules.validate_grain_consistency] HAVING without aggregation: grain={grain}")
    debug(f"[validation_rules.validate_grain_consistency] {len(issues)} issues in {context}")
    return issues


def validate_grouped_requires_aggregation(
    grain: str,
    select_cols: list[SelectCol],
    group_by_cols: list[NormalizedExpr],
    context: str = "main",
    having_param: list[HavingParam] | None = None,
) -> list[IntentIssue]:
    """Ensure grouped grain with GROUP BY includes at least one. aggregation in SELECT."""
    issues: list[IntentIssue] = []
    if grain != "grouped":
        return issues
    if not group_by_cols:
        return issues
    has_agg = any(sc.is_aggregated for sc in select_cols)
    if has_agg:
        return issues
    hp = having_param or []
    if hp and any(h.left_expr.has_aggregation for h in hp):
        return issues
    issues.append(
        IntentIssue.make(
            issue_id=f"grouped_without_aggregation_{context}",
            category=FailureCategory.GRAIN_CONSISTENCY,
            severity="error",
            message=f"Grouped grain with GROUP BY but no aggregation in SELECT in {context}. Use row_level with DISTINCT instead.",
            context={
                "grain": grain,
                "group_by": [g.primary_column for g in group_by_cols],
                "location": context,
            },
        )
    )
    debug(f"[validation_rules.validate_grouped_requires_aggregation] grouped without aggregation in {context}")
    return issues


def validate_case_branch_aggregation_consistency(
    case_registry: list[Any] | None, group_by_cols: list[NormalizedExpr], context: str = "main"
) -> list[IntentIssue]:
    """Ensure CASE branch conditions that reference aggregates run in a. grouped scope. A branch like ``WHEN SUM(amount) > 1000 THEN ...`` is valid SQL only when the parent query aggregates (i.e. has at least one ``GROUP BY`` column). The intent_expr tagger sets ``CaseWhenExpr.condition_scope = "having"`` exactly when any branch's left/right expression has an aggregation; this validator emits an issue if that scope appears without ``GROUP BY``."""

    issues: list[IntentIssue] = []
    if group_by_cols:
        return issues

    def _check(cw: Any, label: str) -> None:
        if cw is None or not getattr(cw, "branches", None):
            return
        if getattr(cw, "condition_scope", "where") != "having":
            return
        issues.append(
            IntentIssue.make(
                issue_id=f"case_branch_aggregation_without_group_by_{context}_{label}",
                category=FailureCategory.HAVING_AGGREGATION,
                severity="error",
                message=(
                    f"CASE branch condition references aggregates in {context} ({label}) "
                    "but the scope has no GROUP BY. Add the appropriate group_by_cols or rewrite the branch condition to a row-level predicate."
                ),
                context={"location": context, "where": label},
            )
        )

    for step in case_registry or []:
        rid = getattr(step, "registry_id", "") or "?"
        _check(getattr(step, "case_when", None), f"case_registry[{rid}]")

    if issues:
        debug(
            f"[validation_rules.validate_case_branch_aggregation_consistency] "
            f"{len(issues)} aggregated CASE branch(es) without GROUP BY in {context}"
        )
    return issues


def validate_semantic_contradictions(
    select_cols: list[SelectCol], natural_language: str, grain: str, expected_rows: str, context: str = "main"
) -> list[IntentIssue]:
    """Check for contradictory operations in the intent."""
    issues = []
    debug("[validation_rules.validate_semantic_contradictions] checking for contradictions")
    agg_funcs = {extract_agg_col(sc.expr.primary_term)[0] for sc in select_cols if sc.is_aggregated} - {None}
    contradictory_pairs = [
        ({"highest", "max"}, {"lowest", "min"}),
        ({"most", "maximum"}, {"least", "minimum"}),
        ({"first", "earliest"}, {"last", "latest"}),
    ]
    for set1, set2 in contradictory_pairs:
        if agg_funcs & set1 and agg_funcs & set2:
            issues.append(
                IntentIssue.make(
                    issue_id=f"contradictory_ops_{','.join(sorted(set1 & agg_funcs))}_{','.join(sorted(set2 & agg_funcs))}",
                    category=FailureCategory.SEMANTIC_CONTRADICTION,
                    severity="error",
                    message=f"Intent contains contradictory operations: {set1 & agg_funcs} and {set2 & agg_funcs}",
                    context={
                        "ops1": list(set1 & agg_funcs),
                        "ops2": list(set2 & agg_funcs),
                        "location": context,
                    },
                )
            )
            debug(
                f"[validation_rules.validate_semantic_contradictions] CONTRADICTION: {set1 & agg_funcs} vs {set2 & agg_funcs}"
            )
    if grain == "scalar" and expected_rows in {"few", "many"}:
        issues.append(
            IntentIssue.make(
                issue_id=f"grain_expected_mismatch_scalar_{expected_rows}",
                category=FailureCategory.SEMANTIC_CONTRADICTION,
                severity="error",
                message=f"Intent expects a single value (scalar) but also expects multiple rows ({expected_rows})",
                context={
                    "grain": grain,
                    "expected_rows": expected_rows,
                    "location": context,
                },
            )
        )
        debug(
            f"[validation_rules.validate_semantic_contradictions] CONTRADICTION: grain=scalar but expected_rows={expected_rows}"
        )
    debug(f"[validation_rules.validate_semantic_contradictions] {len(issues)} issues")
    return issues


def validate_predicate_group_hints(
    natural_language: str, where: PredicateGroup | None, having: PredicateGroup | None, context: str = "main query"
) -> list[IntentIssue]:
    """Emit soft warnings for ambiguous boolean-composition signals in predicate trees."""
    _ = natural_language
    _ = context
    _ = where
    _ = having
    return []


def _validate_single_expr_types(
    expr: NormalizedExpr,
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
    location: str,
    context: str,
) -> list[IntentIssue]:
    """Validate that columns in an expression are type-appropriate, including in arithmetic or numeric-input aggregation and scalar contexts."""
    issues: list[IntentIssue] = []
    has_arith = expr_has_arithmetic(expr)
    for g in expr.add_groups + expr.sub_groups:
        if (g.scalar_func or "").lower() in SCALAR_FUNCTIONS_STRING:
            continue
        group_requires_numeric = (g.agg_func and g.agg_func in NUMERIC_ONLY_AGGREGATIONS) or (
            g.inner_scalar_func and g.inner_scalar_func in SCALAR_FUNCTIONS_NUMERIC
        )
        expr_requires_numeric = (expr.agg_func and expr.agg_func in NUMERIC_ONLY_AGGREGATIONS) or (
            expr.inner_scalar_func and expr.inner_scalar_func in SCALAR_FUNCTIONS_NUMERIC
        )
        needs_numeric = (
            has_arith
            or len(g.multiply) > 1
            or len(g.divide) > 0
            or g.coefficient != 1.0
            or group_requires_numeric
            or expr_requires_numeric
        )
        if not needs_numeric:
            continue
        if g.inner_scalar_func:
            col_check_skippable = (
                g.inner_scalar_func in NUMERIC_RESULT_SCALARS and g.inner_scalar_func not in SCALAR_FUNCTIONS_NUMERIC
            )
        elif g.agg_func:
            col_check_skippable = g.agg_func in NUMERIC_RESULT_AGGS and g.agg_func not in NUMERIC_ONLY_AGGREGATIONS
        elif expr.inner_scalar_func:
            col_check_skippable = (
                expr.inner_scalar_func in NUMERIC_RESULT_SCALARS
                and expr.inner_scalar_func not in SCALAR_FUNCTIONS_NUMERIC
            )
        elif expr.agg_func:
            col_check_skippable = (
                expr.agg_func in NUMERIC_RESULT_AGGS and expr.agg_func not in NUMERIC_ONLY_AGGREGATIONS
            )
        else:
            col_check_skippable = False
        for term in g.multiply:
            term_has_arith = has_arith or expr_has_arithmetic(term)
            if col_check_skippable:
                continue
            for ref in extract_columns_from_expr(term):
                if not ref or ref == "*" or "." not in ref:
                    continue
                is_num = is_col_numeric(ref, schema, cte_outputs)
                if is_num is False:
                    col_type = get_col_type(ref, schema, cte_outputs) or "unknown"
                    if term_has_arith and col_type in DATE_FRIENDLY_VALUE_TYPES:
                        continue
                    issues.append(
                        IntentIssue.make(
                            issue_id=f"expr_non_numeric_{location}_{ref}",
                            category=FailureCategory.EXPRESSION_TYPE,
                            severity="error",
                            message=f"Non-numeric column '{ref}' (type={col_type}) in arithmetic at {location} in {context}",
                            context={
                                "column": ref,
                                "data_type": col_type,
                                "location": location,
                            },
                        )
                    )
                role_ok = is_col_arithmetic_role(ref, schema, cte_outputs)
                if role_ok is False:
                    meta = get_col_meta(ref, schema, cte_outputs)
                    role = meta.role if meta else "unknown"
                    issues.append(
                        IntentIssue.make(
                            issue_id=f"expr_invalid_role_{location}_{ref}",
                            category=FailureCategory.EXPRESSION_TYPE,
                            severity="warning",
                            message=f"Column '{ref}' (role={role}) not suited for arithmetic at {location} in {context}",
                            context={"column": ref, "role": role, "location": location},
                        )
                    )
        for div_term in g.divide:
            div_has_arith = has_arith or expr_has_arithmetic(div_term)
            if col_check_skippable:
                continue
            for ref in extract_columns_from_expr(div_term):
                if not ref or ref == "*" or "." not in ref:
                    continue
                is_num = is_col_numeric(ref, schema, cte_outputs)
                if is_num is False:
                    col_type = get_col_type(ref, schema, cte_outputs) or "unknown"
                    if div_has_arith and col_type in DATE_FRIENDLY_VALUE_TYPES:
                        continue
                    issues.append(
                        IntentIssue.make(
                            issue_id=f"expr_non_numeric_div_{location}_{ref}",
                            category=FailureCategory.EXPRESSION_TYPE,
                            severity="error",
                            message=f"Non-numeric column '{ref}' (type={col_type}) in divide at {location} in {context}",
                            context={
                                "column": ref,
                                "data_type": col_type,
                                "location": location,
                            },
                        )
                    )
    return issues


def _is_date_column_subtraction(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> bool:
    """Return True when *expr* is a date-column minus date-column subtraction."""
    return is_date_column_subtraction(expr, schema, cte_outputs)


def _expr_effective_result_type(
    expr: NormalizedExpr, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> str | None:
    """Resolve the comparison type produced by *expr*."""
    if is_date_integer_day_arithmetic(expr, schema, cte_outputs):
        return "date"
    if _is_date_column_subtraction(expr, schema, cte_outputs):
        return "integer"
    col = expr.primary_column
    if not col:
        return None
    return get_col_type(col, schema, cte_outputs)


def _expr_vs_expr_effective_left_type(
    fp: WhereParam, schema: SchemaGraph, cte_outputs: dict[str, dict[str, CteOutputColumnMeta]]
) -> str | None:
    """Resolve the comparison type for the left side of an expr-vs-expr filter."""
    return _expr_effective_result_type(fp.left_expr, schema, cte_outputs)


def validate_expr_vs_expr_where(
    where_params: list[WhereParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate ``WhereParam`` expression-vs-expression comparisons for type compatibility. Date subtraction yields integer day counts; integer columns with temporal role compare as integers when matched against elapsed-day expressions or duration columns."""
    issues = []
    if not where_params:
        return []
    cte_outputs = cte_outputs or {}
    debug("[validation_rules.validate_expr_vs_expr_where] checking expr-vs-expr type compatibility")
    for fp in where_params:
        if not fp.right_expr:
            continue
        left_col = fp.left_expr.primary_column
        right_col = fp.right_expr.primary_column
        if left_col == right_col:
            issues.append(
                IntentIssue.make(
                    issue_id=f"self_comparison_where_{left_col}",
                    category=FailureCategory.WHERE_SEMANTIC,
                    severity="error",
                    message=f"Self-comparison in filter: {left_col} compared to itself",
                    context={
                        "column": left_col,
                        "param_key": fp.param_key,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_rules.validate_expr_vs_expr_where] self-comparison: {left_col}")
            continue
        left_type = _expr_vs_expr_effective_left_type(fp, schema, cte_outputs)
        right_type = _expr_effective_result_type(fp.right_expr, schema, cte_outputs)
        if left_type and right_type:
            if (left_type, right_type) not in COMPATIBLE_TYPE_PAIRS and (
                right_type,
                left_type,
            ) not in COMPATIBLE_TYPE_PAIRS:
                if left_type != right_type:
                    issues.append(
                        IntentIssue.make(
                            issue_id=f"where_type_mismatch_{left_col}_{right_col}",
                            category=FailureCategory.WHERE_SEMANTIC,
                            severity="error",
                            message=f"Type mismatch in filter: {left_col} ({left_type}) vs {right_col} ({right_type})",
                            context={
                                "left_col": left_col,
                                "left_type": left_type,
                                "right_col": right_col,
                                "right_type": right_type,
                                "param_key": fp.param_key,
                                "location": context,
                            },
                        )
                    )
                    debug(f"[validation_rules.validate_expr_vs_expr_where] type mismatch: {left_type} vs {right_type}")
        left_meta = get_col_meta(left_col, schema, cte_outputs)
        right_meta = get_col_meta(right_col, schema, cte_outputs)
        if left_meta and right_meta:
            if column_metadata_timezone_awareness_mismatch(left_meta, right_meta):
                issues.append(
                    IntentIssue.make(
                        issue_id=f"where_timezone_mismatch_{left_col}_{right_col}",
                        category=FailureCategory.WHERE_SEMANTIC,
                        severity="error",
                        message=(
                            f"Timezone awareness mismatch in filter: {left_col} ({left_meta.data_type}) "
                            f"vs {right_col} ({right_meta.data_type})"
                        ),
                        context={
                            "left_col": left_col,
                            "right_col": right_col,
                            "param_key": fp.param_key,
                            "location": context,
                        },
                    )
                )
                continue
            if left_meta.is_primary_key or right_meta.is_primary_key:
                issues.append(
                    IntentIssue.make(
                        issue_id=f"pk_comparison_where_{left_col}_{right_col}",
                        category=FailureCategory.WHERE_SEMANTIC,
                        severity="warning",
                        message=f"Comparing primary key in filter: {left_col} vs {right_col}",
                        context={
                            "left_col": left_col,
                            "right_col": right_col,
                            "param_key": fp.param_key,
                            "location": context,
                        },
                    )
                )
                debug(f"[validation_rules.validate_expr_vs_expr_where] PK comparison: {left_col}")
    debug(f"[validation_rules.validate_expr_vs_expr_where] {len(issues)} issues in {context}")
    return issues


def validate_agg_vs_agg_having(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate `HavingParam` expression-vs-expression comparisons for. numeric type compatibility."""
    issues = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    debug("[validation_rules.validate_agg_vs_agg_having] checking agg-vs-agg type compatibility")
    for hp in having_param:
        if not hp.right_expr:
            continue
        left_term = hp.left_expr.primary_term
        right_term = hp.right_expr.primary_term
        left_result = extract_agg_col(left_term)
        right_result = extract_agg_col(right_term)
        left_func, left_target, _ = left_result
        right_func, right_target, _ = right_result
        if not left_func or not right_func:
            continue
        if left_target == right_target and left_func == right_func:
            issues.append(
                IntentIssue.make(
                    issue_id=f"self_comparison_having_{left_term}",
                    category=FailureCategory.HAVING_SEMANTIC,
                    severity="error",
                    message=f"Self-comparison in HAVING: {left_term} compared to itself",
                    context={
                        "aggregation": left_term,
                        "param_key": hp.param_key,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_rules.validate_agg_vs_agg_having] self-comparison: {left_term}")
            continue
        if left_target and left_target != "*" and right_target and right_target != "*":
            left_type = get_col_type(left_target, schema, cte_outputs)
            right_type = get_col_type(right_target, schema, cte_outputs)
            left_meta = get_col_meta(left_target, schema, cte_outputs)
            right_meta = get_col_meta(right_target, schema, cte_outputs)
            if left_meta and right_meta and column_metadata_timezone_awareness_mismatch(left_meta, right_meta):
                issues.append(
                    IntentIssue.make(
                        issue_id=f"having_timezone_mismatch_{left_term}_{right_term}",
                        category=FailureCategory.HAVING_SEMANTIC,
                        severity="error",
                        message=(
                            f"Timezone awareness mismatch in HAVING: {left_term} ({left_meta.data_type}) "
                            f"vs {right_term} ({right_meta.data_type})"
                        ),
                        context={
                            "left_term": left_term,
                            "right_term": right_term,
                            "param_key": hp.param_key,
                            "location": context,
                        },
                    )
                )
                continue
            if left_type and right_type:
                numeric_funcs = {"sum", "avg", "count"}
                if left_func in numeric_funcs and right_func in numeric_funcs:
                    pass
                elif (left_type, right_type) not in COMPATIBLE_TYPE_PAIRS and (
                    right_type,
                    left_type,
                ) not in COMPATIBLE_TYPE_PAIRS:
                    if left_type != right_type:
                        issues.append(
                            IntentIssue.make(
                                issue_id=f"having_type_mismatch_{left_term}_{right_term}",
                                category=FailureCategory.HAVING_SEMANTIC,
                                severity="warning",
                                message=f"Type mismatch in HAVING: {left_term} ({left_type}) vs {right_term} ({right_type})",
                                context={
                                    "left_agg": left_term,
                                    "left_type": left_type,
                                    "right_agg": right_term,
                                    "right_type": right_type,
                                    "param_key": hp.param_key,
                                    "location": context,
                                },
                            )
                        )
                        debug(
                            f"[validation_rules.validate_agg_vs_agg_having] type mismatch: {left_type} vs {right_type}"
                        )
    debug(f"[validation_rules.validate_agg_vs_agg_having] {len(issues)} issues in {context}")
    return issues


def validate_select_expr_types(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that SELECT column arithmetic expressions reference. numeric columns with valid roles."""
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for idx, sc in enumerate(select_cols or []):
        issues.extend(_validate_single_expr_types(sc.expr, schema, cte_outputs, f"select_cols[{idx}]", context))
    if issues:
        debug(f"[validation_rules.validate_select_expr_types] {len(issues)} issues in {context}")
    return issues


def validate_order_by_expr_types(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that ORDER BY column arithmetic expressions reference. numeric columns with valid roles."""
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for idx, obc in enumerate(order_by_cols or []):
        issues.extend(_validate_single_expr_types(obc.expr, schema, cte_outputs, f"order_by_cols[{idx}]", context))
    if issues:
        debug(f"[validation_rules.validate_order_by_expr_types] {len(issues)} issues in {context}")
    return issues


def validate_where_expr_types(
    where_params: list[WhereParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate `WhereParam` expression types, cross-expression type. compatibility, and operator compatibility."""
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for fp in where_params or []:
        pk = fp.param_key or "unknown"
        issues.extend(_validate_single_expr_types(fp.left_expr, schema, cte_outputs, f"filter_{pk}_left", context))
        if fp.right_expr:
            issues.extend(
                _validate_single_expr_types(fp.right_expr, schema, cte_outputs, f"filter_{pk}_right", context)
            )
            left_num = expr_result_is_numeric(fp.left_expr, schema, cte_outputs)
            right_num = expr_result_is_numeric(fp.right_expr, schema, cte_outputs)
            left_eff = _expr_effective_result_type(fp.left_expr, schema, cte_outputs)
            right_eff = _expr_effective_result_type(fp.right_expr, schema, cte_outputs)
            if left_eff == "date" and right_eff == "date":
                pass
            elif left_num is not None and right_num is not None and left_num != right_num:
                issues.append(
                    IntentIssue.make(
                        issue_id=f"where_cross_type_mismatch_{pk}",
                        category=FailureCategory.EXPRESSION_TYPE,
                        severity="error",
                        message=f"Filter '{pk}' compares numeric expression to non-numeric expression in {context}",
                        context={
                            "param_key": pk,
                            "left_numeric": left_num,
                            "right_numeric": right_num,
                            "location": context,
                        },
                    )
                )
        left_arith = expr_has_arithmetic(fp.left_expr)
        right_arith = fp.right_expr and expr_has_arithmetic(fp.right_expr)
        left_eff = _expr_effective_result_type(fp.left_expr, schema, cte_outputs)
        right_eff = _expr_effective_result_type(fp.right_expr, schema, cte_outputs) if fp.right_expr else None
        date_arith = left_eff == "date" or right_eff == "date"
        if (
            (left_arith or right_arith)
            and not date_arith
            and fp.op not in NUMERIC_COMPARE_OPS_ORDERED
            and fp.op not in ("is null", "is not null")
        ):
            issues.append(
                IntentIssue.make(
                    issue_id=f"where_op_on_arith_{pk}_{fp.op}",
                    category=FailureCategory.EXPRESSION_TYPE,
                    severity="error",
                    message=f"Operator '{fp.op}' invalid on arithmetic expression in filter '{pk}' in {context}. Expected: {sorted(NUMERIC_COMPARE_OPS_ORDERED)}",
                    context={
                        "param_key": pk,
                        "operator": fp.op,
                        "valid_ops": sorted(NUMERIC_COMPARE_OPS_ORDERED),
                        "location": context,
                    },
                )
            )
    if issues:
        debug(f"[validation_rules.validate_where_expr_types] {len(issues)} issues in {context}")
    return issues


def validate_having_expr_types(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate `HavingParam` expression types and cross-expression. numeric type compatibility."""
    issues: list[IntentIssue] = []
    cte_outputs = cte_outputs or {}
    for hp in having_param or []:
        pk = hp.param_key or "unknown"
        issues.extend(_validate_single_expr_types(hp.left_expr, schema, cte_outputs, f"having_{pk}_left", context))
        if hp.right_expr:
            issues.extend(
                _validate_single_expr_types(hp.right_expr, schema, cte_outputs, f"having_{pk}_right", context)
            )
            left_num = expr_result_is_numeric(hp.left_expr, schema, cte_outputs)
            right_num = expr_result_is_numeric(hp.right_expr, schema, cte_outputs)
            if left_num is not None and right_num is not None and left_num != right_num:
                issues.append(
                    IntentIssue.make(
                        issue_id=f"having_cross_type_mismatch_{pk}",
                        category=FailureCategory.EXPRESSION_TYPE,
                        severity="error",
                        message=f"Having '{pk}' compares numeric expression to non-numeric expression in {context}",
                        context={
                            "param_key": pk,
                            "left_numeric": left_num,
                            "right_numeric": right_num,
                            "location": context,
                        },
                    )
                )
    if issues:
        debug(f"[validation_rules.validate_having_expr_types] {len(issues)} issues in {context}")
    return issues


def _validate_concat_group(
    group: MulGroup, location: str, context: str, *, parent_is_distinct_count: bool = False
) -> list[IntentIssue]:
    """Enforce structural rules for a MulGroup whose outer scalar. function is CONCAT."""
    issues: list[IntentIssue] = []
    if group.divide:
        issues.append(
            IntentIssue.make(
                issue_id=f"concat_divide_{location}",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message=f"CONCAT MulGroup at {location} in {context} must not carry divide terms.",
                context={"location": location},
            )
        )
    if group.coefficient != 1.0 or (group.coeff_param_key or "").strip():
        issues.append(
            IntentIssue.make(
                issue_id=f"concat_coeff_{location}",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message=f"CONCAT MulGroup at {location} in {context} must use coefficient 1.0 without coeff_param_key.",
                context={"location": location},
            )
        )
    if (group.inner_scalar_func or "").strip():
        issues.append(
            IntentIssue.make(
                issue_id=f"concat_inner_scalar_{location}",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message=f"CONCAT MulGroup at {location} in {context} must not set inner_scalar_func.",
                context={"location": location},
            )
        )
    if parent_is_distinct_count:
        if group.distinct:
            issues.append(
                IntentIssue.make(
                    issue_id=f"concat_distinct_under_count_{location}",
                    category=FailureCategory.STRUCTURAL,
                    severity="error",
                    message=(
                        f"CONCAT MulGroup at {location} in {context} must not set distinct when nested "
                        "under COUNT(DISTINCT CONCAT(...)); distinct belongs on the outer COUNT MulGroup."
                    ),
                    context={"location": location},
                )
            )
    agg = (group.agg_func or "").strip().lower()
    if parent_is_distinct_count and not agg:
        pass
    elif agg and agg != "count":
        issues.append(
            IntentIssue.make(
                issue_id=f"concat_agg_{location}",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message=f"CONCAT MulGroup at {location} in {context} allows only COUNT as outer aggregation, not {agg!r}.",
                context={"location": location, "agg_func": agg},
            )
        )
    for pi, part in enumerate(group.multiply):
        if part.has_aggregation:
            issues.append(
                IntentIssue.make(
                    issue_id=f"concat_nested_agg_{location}_{pi}",
                    category=FailureCategory.STRUCTURAL,
                    severity="error",
                    message=f"CONCAT part {pi} at {location} in {context} must not contain aggregation.",
                    context={"location": location, "part_index": pi},
                )
            )
    return issues


def _walk_expr_concat_mulgroups(
    expr: NormalizedExpr, location: str, context: str, *, parent_is_distinct_count: bool = False
) -> list[IntentIssue]:
    """Recursively validate CONCAT MulGroup entries nested under a. NormalizedExpr."""
    issues: list[IntentIssue] = []
    for gi, g in enumerate(expr.add_groups):
        loc_g = f"{location}_add[{gi}]"
        child_parent_distinct_count = parent_is_distinct_count
        if (g.agg_func or "").lower() == "count" and g.distinct:
            child_parent_distinct_count = True
        if (g.scalar_func or "").lower() == "concat":
            issues.extend(
                _validate_concat_group(g, loc_g, context, parent_is_distinct_count=child_parent_distinct_count)
            )
        for ti, t in enumerate(g.multiply + g.divide):
            issues.extend(
                _walk_expr_concat_mulgroups(
                    t, f"{loc_g}_m[{ti}]", context, parent_is_distinct_count=child_parent_distinct_count
                )
            )
    for gi, g in enumerate(expr.sub_groups):
        loc_g = f"{location}_sub[{gi}]"
        child_parent_distinct_count = parent_is_distinct_count
        if (g.agg_func or "").lower() == "count" and g.distinct:
            child_parent_distinct_count = True
        if (g.scalar_func or "").lower() == "concat":
            issues.extend(
                _validate_concat_group(g, loc_g, context, parent_is_distinct_count=child_parent_distinct_count)
            )
        for ti, t in enumerate(g.multiply + g.divide):
            issues.extend(
                _walk_expr_concat_mulgroups(
                    t, f"{loc_g}_m[{ti}]", context, parent_is_distinct_count=child_parent_distinct_count
                )
            )
    return issues


def validate_concat_mulgroups_in_runtime(intent: RuntimeIntent, context: str) -> list[IntentIssue]:
    """Validate every CONCAT MulGroup in the main query and each CTE. body."""
    issues: list[IntentIssue] = []
    for idx, sc in enumerate(intent.select_cols or []):
        issues.extend(_walk_expr_concat_mulgroups(sc.expr, f"{context} select_cols[{idx}]", context))
    for idx, g in enumerate(intent.group_by_cols or []):
        issues.extend(_walk_expr_concat_mulgroups(g, f"{context} group_by_cols[{idx}]", context))
    for idx, obc in enumerate(intent.order_by_cols or []):
        issues.extend(_walk_expr_concat_mulgroups(obc.expr, f"{context} order_by_cols[{idx}]", context))
    for fp in PredicateGroup.where_leaves(intent.where) or []:
        pk = fp.param_key or "unknown"
        issues.extend(_walk_expr_concat_mulgroups(fp.left_expr, f"{context} where_{pk}_left", context))
        if fp.right_expr:
            issues.extend(_walk_expr_concat_mulgroups(fp.right_expr, f"{context} where_{pk}_right", context))
    for hp in PredicateGroup.having_leaves(intent.having) or []:
        pk = hp.param_key or "unknown"
        issues.extend(_walk_expr_concat_mulgroups(hp.left_expr, f"{context} having_{pk}_left", context))
        if hp.right_expr:
            issues.extend(_walk_expr_concat_mulgroups(hp.right_expr, f"{context} having_{pk}_right", context))
    for cte in intent.cte_steps or []:
        cctx = f"{context} CTE '{cte.cte_name}'"
        for idx, sc in enumerate(cte.select_cols or []):
            issues.extend(_walk_expr_concat_mulgroups(sc.expr, f"{cctx} select_cols[{idx}]", cctx))
        for idx, g in enumerate(cte.group_by_cols or []):
            issues.extend(_walk_expr_concat_mulgroups(g, f"{cctx} group_by_cols[{idx}]", cctx))
        for idx, obc in enumerate(cte.order_by_cols or []):
            issues.extend(_walk_expr_concat_mulgroups(obc.expr, f"{cctx} order_by_cols[{idx}]", cctx))
        for fp in PredicateGroup.where_leaves(cte.where) or []:
            pk = fp.param_key or "unknown"
            issues.extend(_walk_expr_concat_mulgroups(fp.left_expr, f"{cctx} where_{pk}_left", cctx))
            if fp.right_expr:
                issues.extend(_walk_expr_concat_mulgroups(fp.right_expr, f"{cctx} where_{pk}_right", cctx))
        for hp in PredicateGroup.having_leaves(cte.having) or []:
            pk = hp.param_key or "unknown"
            issues.extend(_walk_expr_concat_mulgroups(hp.left_expr, f"{cctx} having_{pk}_left", cctx))
            if hp.right_expr:
                issues.extend(_walk_expr_concat_mulgroups(hp.right_expr, f"{cctx} having_{pk}_right", cctx))
    if issues:
        debug(f"[validation_rules.validate_concat_mulgroups_in_runtime] {len(issues)} issues in {context}")
    return issues


_OPAQUE_EXPR_REFUSAL_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR]["user_text"]


def _walk_opaque_raw_sql(expr: NormalizedExpr | None, location: str, context: str) -> list[IntentIssue]:
    """Collect issues for opaque ``raw_sql`` leaves nested under *expr*."""
    if expr is None:
        return []
    issues: list[IntentIssue] = []
    if expr.raw_sql:
        issues.append(
            IntentIssue.make(
                issue_id=f"opaque_raw_sql_{location}",
                category=FailureCategory.INTENT_PARSE_FAILED,
                severity="error",
                message=_OPAQUE_EXPR_REFUSAL_MESSAGE,
                context={"location": location, "context": context},
            )
        )
        emit_session_refusal_diagnostic(DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR, _OPAQUE_EXPR_REFUSAL_MESSAGE)
    for gi, g in enumerate(expr.add_groups):
        loc_g = f"{location}_add[{gi}]"
        for ti, t in enumerate(g.multiply + g.divide):
            issues.extend(_walk_opaque_raw_sql(t, f"{loc_g}_m[{ti}]", context))
    for gi, g in enumerate(expr.sub_groups):
        loc_g = f"{location}_sub[{gi}]"
        for ti, t in enumerate(g.multiply + g.divide):
            issues.extend(_walk_opaque_raw_sql(t, f"{loc_g}_m[{ti}]", context))
    return issues


def _scan_intent_opaque_raw_sql(
    *,
    select_cols: list[SelectCol],
    group_by_cols: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    where: PredicateGroup | None,
    having: PredicateGroup | None,
    distinct_on: list[NormalizedExpr],
    context: str,
) -> list[IntentIssue]:
    """Walk expression surfaces on one query scope for opaque ``raw_sql`` leaves."""
    issues: list[IntentIssue] = []
    for idx, sc in enumerate(select_cols or []):
        issues.extend(_walk_opaque_raw_sql(sc.expr, f"{context} select_cols[{idx}]", context))
    for idx, g in enumerate(group_by_cols or []):
        issues.extend(_walk_opaque_raw_sql(g, f"{context} group_by_cols[{idx}]", context))
    for idx, obc in enumerate(order_by_cols or []):
        issues.extend(_walk_opaque_raw_sql(obc.expr, f"{context} order_by_cols[{idx}]", context))
    for idx, expr in enumerate(distinct_on or []):
        issues.extend(_walk_opaque_raw_sql(expr, f"{context} distinct_on[{idx}]", context))
    for fp in PredicateGroup.where_leaves(where) or []:
        pk = fp.param_key or "unknown"
        issues.extend(_walk_opaque_raw_sql(fp.left_expr, f"{context} where_{pk}_left", context))
        if fp.right_expr:
            issues.extend(_walk_opaque_raw_sql(fp.right_expr, f"{context} where_{pk}_right", context))
    for hp in PredicateGroup.having_leaves(having) or []:
        pk = hp.param_key or "unknown"
        issues.extend(_walk_opaque_raw_sql(hp.left_expr, f"{context} having_{pk}_left", context))
        if hp.right_expr:
            issues.extend(_walk_opaque_raw_sql(hp.right_expr, f"{context} having_{pk}_right", context))
    return issues


def validate_no_opaque_raw_sql(intent: RuntimeIntent, schema: SchemaGraph) -> list[IntentIssue]:
    """Reject intents that carry opaque ``raw_sql`` expression leaves anywhere in the tree."""
    del schema
    issues = _scan_intent_opaque_raw_sql(
        select_cols=intent.select_cols or [],
        group_by_cols=intent.group_by_cols or [],
        order_by_cols=intent.order_by_cols or [],
        where=intent.where,
        having=intent.having,
        distinct_on=intent.distinct_on or [],
        context="main query",
    )
    for cte in intent.cte_steps or []:
        cctx = f"CTE '{cte.cte_name}'"
        issues.extend(
            _scan_intent_opaque_raw_sql(
                select_cols=cte.select_cols or [],
                group_by_cols=cte.group_by_cols or [],
                order_by_cols=cte.order_by_cols or [],
                where=cte.where,
                having=cte.having,
                distinct_on=cte.distinct_on or [],
                context=cctx,
            )
        )
    if issues:
        debug(f"[validation_rules.validate_no_opaque_raw_sql] {len(issues)} issues")
    return issues


def validate_arith_expression_semantics(
    where_params: list[WhereParam],
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that arithmetic expressions in filters and HAVING use. compatible operand types."""
    issues = []
    cte_outputs = cte_outputs or {}
    debug("[validation_rules.validate_arith_expression_semantics] checking arithmetic semantics")
    for fp in where_params:
        if expr_has_arithmetic(fp.left_expr):
            issues.extend(
                _validate_single_expr_types(
                    fp.left_expr, schema, cte_outputs, f"filter_{fp.param_key or 'unknown'}_left", context
                )
            )
        if fp.right_expr and expr_has_arithmetic(fp.right_expr):
            issues.extend(
                _validate_single_expr_types(
                    fp.right_expr, schema, cte_outputs, f"filter_{fp.param_key or 'unknown'}_right", context
                )
            )
    for hp in having_param:
        if expr_has_arithmetic(hp.left_expr):
            issues.extend(
                _validate_single_expr_types(
                    hp.left_expr, schema, cte_outputs, f"having_{hp.param_key or 'unknown'}_left", context
                )
            )
        if hp.right_expr and expr_has_arithmetic(hp.right_expr):
            issues.extend(
                _validate_single_expr_types(
                    hp.right_expr, schema, cte_outputs, f"having_{hp.param_key or 'unknown'}_right", context
                )
            )
    debug(f"[validation_rules.validate_arith_expression_semantics] {len(issues)} issues in {context}")
    return issues


def _term_has_aggregation(term: Any) -> bool:
    """Return whether a single multiply/divide term contains an aggregation call. Accepts a ``NormalizedExpr`` or a raw SQL string. Walks nested groups to detect any agg_func or raw_sql aggregation pattern."""
    if isinstance(term, NormalizedExpr):
        if term.agg_func:
            return True
        if term.raw_sql:
            return _term_has_aggregation(term.raw_sql)
        for g in term.add_groups + term.sub_groups:
            if g.agg_func:
                return True
            for child in g.multiply + g.divide:
                if _term_has_aggregation(child):
                    return True
        return False
    if not isinstance(term, str):
        return False
    upper = term.upper()
    return any(upper.startswith(f"{a.upper()}(") or f" {a.upper()}(" in upper for a in VALID_AGGREGATION_FUNCTIONS)


def validate_where_no_aggregation(where_params: list[WhereParam], context: str = "main") -> list[IntentIssue]:
    """Validate that filter (WHERE) conditions do not contain. aggregation functions."""
    issues: list[IntentIssue] = []
    debug("[validation_rules.validate_where_no_aggregation] checking filter aggregation ban")
    for fp in where_params or []:
        pk = fp.param_key or "unknown"
        if fp.left_expr.has_aggregation:
            issues.append(
                IntentIssue.make(
                    issue_id=f"where_has_aggregation_{pk}_left",
                    category=FailureCategory.WHERE_AGGREGATION,
                    severity="error",
                    message=f"Filter '{pk}' left expression contains aggregation in {context}; use HAVING instead of WHERE",
                    context={"param_key": pk, "side": "left", "location": context},
                )
            )
        if fp.right_expr and fp.right_expr.has_aggregation:
            issues.append(
                IntentIssue.make(
                    issue_id=f"where_has_aggregation_{pk}_right",
                    category=FailureCategory.WHERE_AGGREGATION,
                    severity="error",
                    message=f"Filter '{pk}' right expression contains aggregation in {context}; use HAVING instead of WHERE",
                    context={"param_key": pk, "side": "right", "location": context},
                )
            )
    if issues:
        debug(f"[validation_rules.validate_where_no_aggregation] {len(issues)} issues in {context}")
    return issues


def validate_having_operator_is_numeric(having_param: list[HavingParam], context: str = "main") -> list[IntentIssue]:
    """Validate that each HAVING operator is in ``VALID_HAVING_OPS``."""
    issues: list[IntentIssue] = []
    for hp in having_param or []:
        pk = hp.param_key or "unknown"
        op_norm = (hp.op or "=").strip().lower()
        if op_norm not in VALID_HAVING_OPS:
            issues.append(
                IntentIssue.make(
                    issue_id=f"having_invalid_op_{pk}",
                    category=FailureCategory.WRONG_HAVING,
                    severity="error",
                    message=(
                        f"HAVING predicate '{pk}' in {context} uses operator {hp.op!r}; "
                        f"allowed HAVING operators are {sorted(VALID_HAVING_OPS)}"
                    ),
                    context={"param_key": pk, "location": context, "op": hp.op},
                )
            )
    if issues:
        debug(f"[validation_rules.validate_having_operator_is_numeric] {len(issues)} issues in {context}")
    return issues


def validate_having_requires_aggregation(
    having_param: list[HavingParam], context: str = "main", *, group_by_cols: list[Any] | None = None
) -> list[IntentIssue]:
    """Validate that each HAVING condition contains at least one. aggregation (left or right expression). When ``group_by_cols`` is empty but ``having_param`` is non-empty, emit ``having_without_group_by``."""
    issues: list[IntentIssue] = []
    debug("[validation_rules.validate_having_requires_aggregation] checking having aggregation requirement")
    if having_param and not (group_by_cols or []):
        issues.append(
            IntentIssue.make(
                issue_id="having_without_group_by",
                category=FailureCategory.WRONG_HAVING,
                severity="error",
                message=f"HAVING is present in {context} but GROUP BY is empty; add GROUP BY or move predicates to WHERE",
                context={"location": context},
            )
        )
        return issues
    for hp in having_param or []:
        pk = hp.param_key or "unknown"
        has_agg = hp.left_expr.has_aggregation or (hp.right_expr is not None and hp.right_expr.has_aggregation)
        if not has_agg:
            issues.append(
                IntentIssue.make(
                    issue_id=f"having_missing_aggregation_{pk}",
                    category=FailureCategory.HAVING_AGGREGATION,
                    severity="error",
                    message=f"Having '{pk}' has no aggregation in {context}; belongs in WHERE not HAVING",
                    context={"param_key": pk, "location": context},
                )
            )
    if issues:
        debug(f"[validation_rules.validate_having_requires_aggregation] {len(issues)} issues in {context}")
    return issues


def _predicate_sidedness_issues(
    left: NormalizedExpr, right: NormalizedExpr | None, op: str, param_key: str | None, clause: str, context: str
) -> list[IntentIssue]:
    """Return an issue when a literal-only side appears on the left of a column-bearing side."""
    if right is None:
        return []
    if not left.is_literal_only:
        return []
    if not right.has_column_reference:
        return []
    pk = param_key or "unknown"
    return [
        IntentIssue.make(
            issue_id=f"predicate_literal_left_{clause}_{pk}",
            category=FailureCategory.PREDICATE_SIDEDNESS,
            severity="error",
            message=(
                f"{clause.upper()} predicate '{pk}' has a literal-only left expression and a column-bearing "
                f"right expression in {context}; column side must be on the left"
            ),
            context={
                "param_key": pk,
                "clause": clause,
                "op": op,
                "location": context,
            },
        )
    ]


def validate_predicate_sidedness(
    where_params: list[WhereParam], having_param: list[HavingParam], context: str = "main"
) -> list[IntentIssue]:
    """Validate that expr-vs-expr predicates place column-bearing sides on the left."""
    issues: list[IntentIssue] = []
    debug("[validation_rules.validate_predicate_sidedness] checking predicate sidedness")
    for fp in where_params or []:
        issues.extend(_predicate_sidedness_issues(fp.left_expr, fp.right_expr, fp.op, fp.param_key, "where", context))
    for hp in having_param or []:
        issues.extend(_predicate_sidedness_issues(hp.left_expr, hp.right_expr, hp.op, hp.param_key, "having", context))
    if issues:
        debug(f"[validation_rules.validate_predicate_sidedness] {len(issues)} issues in {context}")
    return issues


def _check_nested_aggregation(expr: NormalizedExpr, location: str, context: str) -> list[IntentIssue]:
    """Check a single `NormalizedExpr` for double-wrap (nested) aggregation."""
    issues: list[IntentIssue] = []
    for g in expr.add_groups + expr.sub_groups:
        group_inline_agg = any(_term_has_aggregation(t) for t in g.multiply + g.divide)
        if expr.agg_func and g.agg_func:
            issues.append(
                IntentIssue.make(
                    issue_id=f"nested_agg_expr_group_{location}",
                    category=FailureCategory.NESTED_AGGREGATION,
                    severity="error",
                    message=f"Nested aggregation {expr.agg_func.upper()}({g.agg_func.upper()}(...)) at {location} in {context}",
                    context={
                        "outer": expr.agg_func,
                        "inner": g.agg_func,
                        "location": location,
                    },
                )
            )
        if expr.agg_func and group_inline_agg:
            issues.append(
                IntentIssue.make(
                    issue_id=f"nested_agg_expr_inline_{location}",
                    category=FailureCategory.NESTED_AGGREGATION,
                    severity="error",
                    message=f"Nested aggregation: expr-level {expr.agg_func.upper()} wraps inline aggregation at {location} in {context}",
                    context={"outer": expr.agg_func, "location": location},
                )
            )
        if g.agg_func and group_inline_agg:
            issues.append(
                IntentIssue.make(
                    issue_id=f"nested_agg_group_inline_{location}",
                    category=FailureCategory.NESTED_AGGREGATION,
                    severity="error",
                    message=f"Nested aggregation: group-level {g.agg_func.upper()} wraps inline aggregation at {location} in {context}",
                    context={"outer": g.agg_func, "location": location},
                )
            )
    return issues


def validate_no_nested_aggregation(
    select_cols: list[SelectCol],
    order_by_cols: list[OrderByCol],
    where_params: list[WhereParam],
    having_param: list[HavingParam],
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that no expression across SELECT, ORDER BY, filters, or. HAVING contains nested aggregation."""
    issues: list[IntentIssue] = []
    debug("[validation_rules.validate_no_nested_aggregation] checking nested aggregation")
    for idx, sc in enumerate(select_cols or []):
        issues.extend(_check_nested_aggregation(sc.expr, f"select_cols[{idx}]", context))
    for idx, obc in enumerate(order_by_cols or []):
        issues.extend(_check_nested_aggregation(obc.expr, f"order_by_cols[{idx}]", context))
    for fp in where_params or []:
        pk = fp.param_key or "unknown"
        issues.extend(_check_nested_aggregation(fp.left_expr, f"filter_{pk}_left", context))
        if fp.right_expr:
            issues.extend(_check_nested_aggregation(fp.right_expr, f"filter_{pk}_right", context))
    for hp in having_param or []:
        pk = hp.param_key or "unknown"
        issues.extend(_check_nested_aggregation(hp.left_expr, f"having_{pk}_left", context))
        if hp.right_expr:
            issues.extend(_check_nested_aggregation(hp.right_expr, f"having_{pk}_right", context))
    if issues:
        debug(f"[validation_rules.validate_no_nested_aggregation] {len(issues)} issues in {context}")
    return issues


def _check_mixed_aggregation_in_group(group: MulGroup, location: str, context: str) -> list[IntentIssue]:
    """Check a single `MulGroup` for mixed aggregated and bare column. terms."""
    issues: list[IntentIssue] = []
    if group.agg_func:
        return issues
    all_terms = list(group.multiply) + list(group.divide)
    if len(all_terms) < 2:
        return issues
    agg_terms: list[str] = []
    bare_terms: list[str] = []
    for term in all_terms:
        rendered = term.raw_sql or term.column_ref or " ".join(extract_columns_from_expr(term)) or repr(term)
        if _term_has_aggregation(term):
            agg_terms.append(rendered)
        else:
            for ref in extract_columns_from_expr(term):
                if ref and ref != "*" and "." in ref:
                    bare_terms.append(rendered)
                    break
    if agg_terms and bare_terms:
        issues.append(
            IntentIssue.make(
                issue_id=f"mixed_agg_bare_{location}",
                category=FailureCategory.MIXED_AGGREGATION,
                severity="error",
                message=f"MulGroup at {location} in {context} mixes aggregated terms ({', '.join(agg_terms)}) with bare columns ({', '.join(bare_terms)})",
                context={
                    "agg_terms": agg_terms,
                    "bare_terms": bare_terms,
                    "location": location,
                },
            )
        )
    return issues


def _check_mixed_aggregation_in_expr(expr: NormalizedExpr, location: str, context: str) -> list[IntentIssue]:
    """Check all `MulGroup` entries in a `NormalizedExpr` for mixed. aggregation."""
    issues: list[IntentIssue] = []
    for idx, g in enumerate(expr.add_groups):
        issues.extend(_check_mixed_aggregation_in_group(g, f"{location}_add[{idx}]", context))
    for idx, g in enumerate(expr.sub_groups):
        issues.extend(_check_mixed_aggregation_in_group(g, f"{location}_sub[{idx}]", context))
    all_groups = list(expr.add_groups) + list(expr.sub_groups)
    if len(all_groups) >= 2 and not expr.agg_func:
        agg_groups: list[str] = []
        bare_groups: list[str] = []
        for g in all_groups:
            has_agg = g.agg_func or any(_term_has_aggregation(t) for t in g.multiply + g.divide)
            sig = g.signature_key
            if has_agg:
                agg_groups.append(sig)
            else:
                has_bare = False
                for t in g.multiply + g.divide:
                    if _term_has_aggregation(t):
                        continue
                    for ref in extract_columns_from_expr(t):
                        if "." in ref:
                            has_bare = True
                            break
                    if has_bare:
                        break
                if has_bare:
                    bare_groups.append(sig)
        if agg_groups and bare_groups:
            issues.append(
                IntentIssue.make(
                    issue_id=f"mixed_agg_across_groups_{location}",
                    category=FailureCategory.MIXED_AGGREGATION,
                    severity="error",
                    message=f"Expression at {location} in {context} mixes aggregated groups ({', '.join(agg_groups)}) with bare column groups ({', '.join(bare_groups)})",
                    context={
                        "agg_groups": agg_groups,
                        "bare_groups": bare_groups,
                        "location": location,
                    },
                )
            )
    return issues


def validate_mixed_aggregation_in_mulgroup(
    select_cols: list[SelectCol],
    order_by_cols: list[OrderByCol],
    where_params: list[WhereParam],
    having_param: list[HavingParam],
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that no `MulGroup` mixes aggregated terms with bare. column references."""
    issues: list[IntentIssue] = []
    debug("[validation_rules.validate_mixed_aggregation_in_mulgroup] checking mixed aggregation")
    for idx, sc in enumerate(select_cols or []):
        issues.extend(_check_mixed_aggregation_in_expr(sc.expr, f"select_cols[{idx}]", context))
    for idx, obc in enumerate(order_by_cols or []):
        issues.extend(_check_mixed_aggregation_in_expr(obc.expr, f"order_by_cols[{idx}]", context))
    for fp in where_params or []:
        pk = fp.param_key or "unknown"
        issues.extend(_check_mixed_aggregation_in_expr(fp.left_expr, f"filter_{pk}_left", context))
        if fp.right_expr:
            issues.extend(_check_mixed_aggregation_in_expr(fp.right_expr, f"filter_{pk}_right", context))
    for hp in having_param or []:
        pk = hp.param_key or "unknown"
        issues.extend(_check_mixed_aggregation_in_expr(hp.left_expr, f"having_{pk}_left", context))
        if hp.right_expr:
            issues.extend(_check_mixed_aggregation_in_expr(hp.right_expr, f"having_{pk}_right", context))
    if issues:
        debug(f"[validation_rules.validate_mixed_aggregation_in_mulgroup] {len(issues)} issues in {context}")
    return issues


def validate_order_by_aggregation_context(
    order_by_cols: list[OrderByCol], grain: str, context: str = "main"
) -> list[IntentIssue]:
    """Validate that ORDER BY aggregation expressions are compatible. with the query grain."""
    issues: list[IntentIssue] = []
    debug(f"[validation_rules.validate_order_by_aggregation_context] grain={grain}")
    if grain != "row_level":
        return issues
    for idx, obc in enumerate(order_by_cols or []):
        if obc.expr.has_aggregation:
            issues.append(
                IntentIssue.make(
                    issue_id=f"order_by_agg_row_level_{idx}",
                    category=FailureCategory.ORDER_BY_AGGREGATION,
                    severity="error",
                    message=f"Order-by[{idx}] contains aggregation but grain is row_level in {context}",
                    context={"index": idx, "grain": grain, "location": context},
                )
            )
    if issues:
        debug(f"[validation_rules.validate_order_by_aggregation_context] {len(issues)} issues in {context}")
    return issues


def validate_select_group_by_membership(
    select_cols: list[SelectCol], group_by_cols: list[NormalizedExpr], grain: str, context: str = "main"
) -> list[IntentIssue]:
    """Validate that every non-aggregated SELECT column appears in GROUP BY when mixed aggregation is present."""
    issues: list[IntentIssue] = []
    debug(f"[validation_rules.validate_select_group_by_membership] grain={grain}, group_by={len(group_by_cols)}")
    if not group_by_cols:
        return issues
    has_agg = any(sc.is_aggregated for sc in (select_cols or []))
    has_non_agg = any(not sc.is_aggregated for sc in (select_cols or []))
    if not (has_agg and has_non_agg):
        return issues
    group_by_set = frozenset(g.primary_column.lower() for g in group_by_cols)
    for idx, sc in enumerate(select_cols or []):
        if sc.is_aggregated:
            continue
        col = sc.expr.primary_column
        if not col:
            continue
        if col.lower() not in group_by_set:
            issues.append(
                IntentIssue.make(
                    issue_id=f"select_not_in_group_by_{idx}_{col}",
                    category=FailureCategory.GROUP_BY_MEMBERSHIP,
                    severity="error",
                    message=f"Non-aggregated select column '{col}' at index {idx} not in GROUP BY in {context}",
                    context={
                        "column": col,
                        "index": idx,
                        "group_by_cols": [g.primary_column for g in group_by_cols],
                        "location": context,
                    },
                )
            )
    if issues:
        debug(f"[validation_rules.validate_select_group_by_membership] {len(issues)} issues in {context}")
    return issues


def _validate_cte_grain_complete(cte: RuntimeCteStep, context: str) -> list[IntentIssue]:
    """Validate one CTE body against the grain state machine using. structural facts only. Emits errors only for impossible combinations of ``GROUP BY``, ``SELECT`` aggregation, ``HAVING``, and declared ``grain``. Wrong labels without structural conflict are left to deterministic repair."""
    with WindowRegistryStep.render_scope(cte.window_registry, cte.case_registry):
        issues: list[IntentIssue] = []
        grain = cte.grain or "row_level"
        group_by = cte.group_by_cols or []
        select_cols = cte.select_cols or []
        having_param = PredicateGroup.having_leaves(cte.having) or []
        has_agg = any(sc.is_aggregated for sc in select_cols)
        all_cols_agg = all(sc.is_aggregated for sc in select_cols) if select_cols else True
        if having_param and not has_agg:
            issues.append(
                IntentIssue.make(
                    issue_id=f"cte_{cte.cte_name}_having_no_agg",
                    category=FailureCategory.CTE_AGGREGATION,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has HAVING clause but no aggregation in {context}",
                    context={
                        "cte_name": cte.cte_name,
                        "having_count": len(having_param),
                        "location": context,
                    },
                )
            )
        if group_by and not has_agg:
            issues.append(
                IntentIssue.make(
                    issue_id=f"cte_group_by_without_agg_{cte.cte_name}",
                    category=FailureCategory.CTE_GRAIN_CONSISTENCY,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has GROUP BY columns but no aggregation in {context}",
                    context={"cte_name": cte.cte_name, "location": context},
                )
            )
        if has_agg and not group_by and not all_cols_agg:
            issues.append(
                IntentIssue.make(
                    issue_id=f"cte_agg_mixed_select_{cte.cte_name}",
                    category=FailureCategory.CTE_GRAIN_CONSISTENCY,
                    severity="error",
                    message=(
                        f"CTE '{cte.cte_name}' uses aggregation without GROUP BY but not every "
                        f"SELECT column is aggregated in {context}"
                    ),
                    context={
                        "cte_name": cte.cte_name,
                        "grain": grain,
                        "location": context,
                    },
                )
            )
        if grain == "grouped" and not group_by:
            issues.append(
                IntentIssue.make(
                    issue_id=f"cte_grouped_no_groupby_{cte.cte_name}",
                    category=FailureCategory.CTE_GRAIN_CONSISTENCY,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has grain=grouped but no group_by_cols in {context}",
                    context={
                        "cte_name": cte.cte_name,
                        "grain": grain,
                        "location": context,
                    },
                )
            )
        if grain in {"scalar", "row_level"} and group_by:
            issues.append(
                IntentIssue.make(
                    issue_id=f"cte_groupby_with_{grain}_{cte.cte_name}",
                    category=FailureCategory.CTE_GRAIN_CONSISTENCY,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has group_by_cols but grain={grain} in {context}",
                    context={
                        "cte_name": cte.cte_name,
                        "grain": grain,
                        "group_by": group_by,
                        "location": context,
                    },
                )
            )
        if has_agg and grain == "row_level":
            agg_funcs = [sc.expr.primary_term for sc in select_cols if sc.is_aggregated]
            issues.append(
                IntentIssue.make(
                    issue_id=f"cte_agg_row_level_{cte.cte_name}",
                    category=FailureCategory.CTE_GRAIN_CONSISTENCY,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has aggregation with row_level grain in {context}",
                    context={
                        "cte_name": cte.cte_name,
                        "agg_funcs": agg_funcs,
                        "grain": grain,
                        "location": context,
                    },
                )
            )
        debug(f"[validation_rules._validate_cte_grain_complete] {len(issues)} issues for CTE '{cte.cte_name}'")
        return issues


def validate_cte_grain_consistency(cte: RuntimeCteStep, context: str) -> list[IntentIssue]:
    """Validate CTE grain structure; delegates to. :func:`_validate_cte_grain_complete`."""
    return _validate_cte_grain_complete(cte, context)


def _cte_step_declares_window(cte: RuntimeCteStep) -> bool:
    """Return True when any SELECT column on the CTE step carries a. window specification."""
    if cte.window_registry:
        return True
    return any((sc.expr.registry_ref() or "").startswith("w") for sc in (cte.select_cols or []))


def validate_cte_dependency_grains(
    cte_steps: list[RuntimeCteStep],
    main_grain: str,
    *,
    main_tables: list[str] | None = None,
    select_cols: list[SelectCol] | None = None,
) -> list[IntentIssue]:
    """Validate that CTE grains are compatible with their upstream. dependencies and the main query grain."""
    issues = []
    cte_grains: dict[str, str] = {}
    debug(
        f"[validation_rules.validate_cte_dependency_grains] validating {len(cte_steps)} CTEs against main grain '{main_grain}'"
    )
    for cte in cte_steps:
        cte_grains[cte.cte_name] = cte.grain
    for cte in cte_steps:
        cte_name = cte.cte_name
        cte_grain = cte.grain
        for table in cte.tables:
            if table in cte_grains:
                dep_grain = cte_grains[table]
                if cte_grain == "row_level" and dep_grain in {"grouped", "scalar"}:
                    if _cte_step_declares_window(cte):
                        continue
                    issues.append(
                        IntentIssue.make(
                            issue_id=f"cte_grain_incompatible_{cte_name}_{table}",
                            category=FailureCategory.CTE_GRAIN_COMPATIBILITY,
                            severity="warning",
                            message=f"CTE '{cte_name}' (row_level) depends on aggregated CTE '{table}' ({dep_grain})",
                            context={
                                "cte_name": cte_name,
                                "cte_grain": cte_grain,
                                "dep_cte": table,
                                "dep_grain": dep_grain,
                            },
                        )
                    )
    debug(f"[validation_rules.validate_cte_dependency_grains] {len(issues)} grain compatibility issues")
    return issues


def _cte_exposes_join_key(cte: RuntimeCteStep) -> bool:
    """Return True when *cte* exposes at least one PK or FK output column suitable for joining."""
    meta_map = cte.output_column_metadata or {}
    for meta in meta_map.values():
        if meta.lineage_inherits_pk:
            return True
        if meta.lineage_fk_to_table and meta.lineage_fk_to_column:
            return True
        role = (meta.role or "").lower()
        if role in {"pk", "fk", "primary_key", "foreign_key"}:
            return True
    return False


def _level_table_count(tables: list[str] | None) -> int:
    """Count distinct, non-empty table identifiers at a single intent level."""
    return len({t for t in (tables or []) if t})


def validate_cte_join_key_exposure(intent: RuntimeIntent) -> list[IntentIssue]:
    """Reject ``join_table`` CTEs that participate in a multi-table level without exposing a join key. For every CTE in *intent*, locate every level that references it (the main scope or any other CTE body via its declared ``tables`` list). When a referencing level holds more than one entry — i.e. the CTE will need to be joined against at least one other table or CTE — and the CTE does not expose at least one PK or FK output column, an :class:`IntentIssue` of :attr:`FailureCategory.CTE_MISSING_JOIN_KEY` is emitted so the repair pipeline can ask the LLM to project a join key. CTEs that classify as ``scalar_subquery`` (single-row CROSS JOIN) are intentionally exempt because they do not require an explicit join key."""
    issues: list[IntentIssue] = []
    cte_steps = list(intent.cte_steps or [])
    if not cte_steps:
        return issues
    cte_names = {step.cte_name for step in cte_steps if step.cte_name}

    references: dict[str, list[tuple[str, int]]] = {name: [] for name in cte_names}
    main_count = _level_table_count(intent.tables)
    for tbl in intent.tables or []:
        if tbl in cte_names:
            references[tbl].append(("__main__", main_count))
    for step in cte_steps:
        step_count = _level_table_count(step.tables)
        for tbl in step.tables or []:
            if tbl in cte_names and tbl != step.cte_name:
                references[tbl].append((step.cte_name, step_count))

    for step in cte_steps:
        name = step.cte_name
        if not name:
            continue
        ref_levels = references.get(name) or []
        needs_join = any(count > 1 for _, count in ref_levels)
        if not needs_join:
            continue
        if (step.grain or "") == "scalar":
            continue
        if _cte_exposes_join_key(step):
            continue
        levels = sorted({lvl for lvl, count in ref_levels if count > 1})
        issues.append(
            IntentIssue.make(
                issue_id=f"cte_missing_join_key_{name}",
                category=FailureCategory.CTE_MISSING_JOIN_KEY,
                severity="error",
                message=(
                    f"CTE '{name}' is joined at level(s) {levels} but exposes no PK or FK column; "
                    "project at least one join key in the CTE's select list"
                ),
                context={
                    "cte_name": name,
                    "referencing_levels": levels,
                    "output_columns": list(step.output_columns or []),
                },
            )
        )
    return issues


def validate_question_agg_keyword_coverage(
    natural_language: str,
    select_cols: list[SelectCol],
    having_param: list[HavingParam],
    context: str = "main",
    cte_steps: list[RuntimeCteStep] | None = None,
) -> list[IntentIssue]:
    """Return no issues; NL keyword heuristics are not applied."""
    _ = (natural_language, select_cols, having_param, context, cte_steps)
    return []


def _runtime_intent_column_refs(intent: RuntimeIntent) -> set[str]:
    """Collect lowercased qualified column refs encoded in a runtime intent."""
    refs: set[str] = set()

    def add_expr(expr: NormalizedExpr) -> None:
        for ref in extract_columns_from_expr(expr):
            if "." in ref:
                refs.add(ref.lower())

    def add_filters(filters: list[WhereParam] | None) -> None:
        for fp in filters or []:
            add_expr(fp.left_expr)
            if fp.right_expr is not None:
                add_expr(fp.right_expr)

    def add_having(having: list[HavingParam] | None) -> None:
        for hp in having or []:
            add_expr(hp.left_expr)
            if hp.right_expr is not None:
                add_expr(hp.right_expr)

    add_filters(PredicateGroup.where_leaves(intent.where))
    add_having(PredicateGroup.having_leaves(intent.having))
    for sc in intent.select_cols or []:
        add_expr(sc.expr)
    for gb in intent.group_by_cols or []:
        add_expr(gb)
    for ob in intent.order_by_cols or []:
        add_expr(ob.expr)
    for cte in intent.cte_steps or []:
        add_filters(PredicateGroup.where_leaves(cte.where))
        add_having(PredicateGroup.having_leaves(cte.having))
        for sc in cte.select_cols or []:
            add_expr(sc.expr)
        for gb in cte.group_by_cols or []:
            add_expr(gb)
        for ob in cte.order_by_cols or []:
            add_expr(ob.expr)
    return refs


def validate_logical_intent_numeric_coverage(
    logical_intent: LogicalIntent | None,
    where_params: list[WhereParam],
    having_param: list[HavingParam],
    limit: int | None,
    context: str = "main",
    *,
    param_values: Mapping[str, Any] | None = None,
    case_registry: Sequence[Any] | None = None,
) -> list[IntentIssue]:
    """Flag digit runs in interpret prose that are missing from filters, HAVING, or limit."""
    issues: list[IntentIssue] = []
    if logical_intent is None:
        return issues
    coverage_source_text = concat_logical_intent_prose(logical_intent)
    if not coverage_source_text:
        return issues

    top_n_numbers: set[str] = set()
    if limit is not None:
        top_n_numbers.add(str(int(limit)) if float(limit).is_integer() else str(limit))

    all_numbers = SHAPE_FORM_NUM_REGEX.findall(coverage_source_text)
    if not all_numbers:
        return issues

    intent_values: set[float] = set()
    covered_number_strs: set[str] = set()
    for fp in where_params:
        rv = fp.resolved_value(param_values)
        if rv is not None:
            try:
                intent_values.add(float(rv))
            except (TypeError, ValueError):
                pass
            if isinstance(rv, str):
                for m in QUESTION_YEAR_IN_STRING_RE.finditer(rv):
                    covered_number_strs.add(m.group())
    for hp in having_param:
        hv = hp.resolved_value(param_values)
        if hv is not None:
            try:
                intent_values.add(float(hv))
            except (TypeError, ValueError):
                pass
            if isinstance(hv, str):
                for m in QUESTION_YEAR_IN_STRING_RE.finditer(hv):
                    covered_number_strs.add(m.group())
    if limit is not None:
        intent_values.add(float(limit))
    for step in case_registry or []:
        case_when = getattr(step, "case_when", None)
        if case_when is None:
            continue
        for branch in getattr(case_when, "branches", []) or []:
            cond = branch.condition
            rv = cond.resolved_value(param_values)
            if rv is not None:
                try:
                    intent_values.add(float(rv))
                except (TypeError, ValueError):
                    pass
                if isinstance(rv, str):
                    for m in QUESTION_YEAR_IN_STRING_RE.finditer(rv):
                        covered_number_strs.add(m.group())

    for num_str in all_numbers:
        if num_str in top_n_numbers:
            continue
        if num_str in covered_number_strs:
            continue
        try:
            val = float(num_str)
        except ValueError:
            continue
        if val in intent_values:
            continue
        issues.append(
            IntentIssue.make(
                issue_id=f"missing_numeric_{num_str}_{context}",
                category=FailureCategory.MISSING_NUMERIC_WHERE,
                severity="warning",
                message=(
                    f"Coverage text mentions number '{num_str}' which does not "
                    f"appear in any filter, having condition, or limit in "
                    f"{context}."
                ),
                context={"value": num_str, "location": context},
            )
        )
        debug(
            f"[validation_rules.validate_logical_intent_numeric_coverage] "
            f"number '{num_str}' not found in intent conditions"
        )
    return issues


def validate_question_distinct_hint(
    natural_language: str, select_cols: list[SelectCol], context: str = "main", distinct_select_index: int = -1
) -> list[IntentIssue]:
    """Return no issues; NL distinct/unique keyword heuristics are not applied."""
    _ = (natural_language, select_cols, context, distinct_select_index)
    return []


def validate_threshold_missing_having(
    natural_language: str,
    select_cols: list[SelectCol],
    having_param: list[HavingParam],
    grain: str,
    context: str = "main",
) -> list[IntentIssue]:
    """Return no issues; NL threshold phrase heuristics are not applied."""
    _ = (natural_language, select_cols, having_param, grain, context)
    return []


def validate_count_threshold_missing_having(
    natural_language: str,
    tables: list[str],
    having_param: list[HavingParam],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Return no issues; NL count-threshold phrase heuristics are not applied."""
    _ = (natural_language, tables, having_param, schema, context)
    return []


def validate_for_each_grouping(
    natural_language: str,
    group_by_cols: list[NormalizedExpr],
    schema: SchemaGraph,
    has_aggregation: bool,
    context: str = "main",
) -> list[IntentIssue]:
    """Return no issues; NL grouping heuristics are not applied."""
    _ = (natural_language, group_by_cols, schema, has_aggregation, context)
    return []


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
        ref = sc.expr.registry_ref() or "" if sc.expr is not None else ""
        if not ref.startswith("c"):
            continue
        parts = sc.effective_parts(window_registry, case_registry)
        cw = parts.case_when
        if cw is None:
            continue
        seen_registry_ids.add(ref)
        for br in cw.branches or []:
            _extend_fan_out_aggregates_from_expr(br.result, found)
        if cw.else_result is not None:
            _extend_fan_out_aggregates_from_expr(cw.else_result, found)
    for step in case_registry or []:
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
        for hp in PredicateGroup.having_leaves(having) or []:
            _extend_fan_out_aggregates_from_expr(hp.left_expr, found)
            _extend_fan_out_aggregates_from_expr(hp.right_expr, found)
    for ob in order_by_cols or []:
        _extend_fan_out_aggregates_from_expr(ob.expr, found)
    for fp in PredicateGroup.where_leaves(where) if where else []:
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


def validate_join_path_key_types(
    signature: list[str],
    schema: SchemaGraph,
    context: str,
) -> list[IntentIssue]:
    """Emit errors when paired join-key columns have incompatible value types."""
    issues: list[IntentIssue] = []
    ctx_key = context.replace(" ", "_")
    for seg_idx, (left_tbl, right_tbl, lcols, rcols) in enumerate(parse_signature_segments(signature)):
        if len(lcols) != len(rcols):
            continue
        for col_idx, (lcol, rcol) in enumerate(zip(lcols, rcols, strict=True)):
            left_meta = schema.tables.get(left_tbl)
            right_meta = schema.tables.get(right_tbl)
            if left_meta is None or right_meta is None:
                continue
            l_cm = left_meta.columns.get(lcol)
            r_cm = right_meta.columns.get(rcol)
            if l_cm is None or r_cm is None:
                continue
            if column_metadata_timezone_awareness_mismatch(l_cm, r_cm):
                issues.append(
                    IntentIssue.make(
                        issue_id=f"join_key_timezone_incompatible_{ctx_key}_{seg_idx}_{col_idx}",
                        category=FailureCategory.WRONG_JOIN,
                        severity="error",
                        message=(
                            f"{context}: join key timezone awareness incompatible for "
                            f"{left_tbl}.{lcol} ({l_cm.data_type}) vs {right_tbl}.{rcol} ({r_cm.data_type}) "
                            f"on the resolved join path."
                        ),
                        context={
                            "left": f"{left_tbl}.{lcol}",
                            "right": f"{right_tbl}.{rcol}",
                            "location": context,
                        },
                    )
                )
                continue
            if fk_infer_value_types_compatible(l_cm, r_cm):
                continue
            issues.append(
                IntentIssue.make(
                    issue_id=f"join_key_type_incompatible_{ctx_key}_{seg_idx}_{col_idx}",
                    category=FailureCategory.WRONG_JOIN,
                    severity="error",
                    message=(
                        f"{context}: join key {left_tbl}.{lcol} ({l_cm.data_type}) is not type-compatible "
                        f"with {right_tbl}.{rcol} ({r_cm.data_type}) on the resolved join path."
                    ),
                    context={
                        "left": f"{left_tbl}.{lcol}",
                        "right": f"{right_tbl}.{rcol}",
                        "location": context,
                    },
                )
            )
    return issues


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
    if (intent.grain or "row_level") != "row_level":
        return []
    anchor = from_anchor or (intent.tables[0] if intent.tables else None)
    multiplied, edge = anchor_table_multiplied(signature, anchor, schema)
    if not multiplied:
        return []

    ctx_key = context.replace(" ", "_")
    issues: list[IntentIssue] = []
    main_tables = {phys_table_key(t) for t in intent_join_reachability_tables(intent)}
    grouped = bool(intent.group_by_cols)

    if not grouped and (intent.limit is not None or (intent.limit_param_key or "").strip()):
        issues.append(
            IntentIssue.make(
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
            IntentIssue.make(
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

    if not grouped and intent.distinct_select_index >= 0:
        issues.append(
            IntentIssue.make(
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
                IntentIssue.make(
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

    count_star = any(expr_has_unqualified_count_star(sc.expr) for sc in intent.select_cols or [])
    if not count_star and intent.having:
        for hp in PredicateGroup.having_leaves(intent.having) or []:
            if expr_has_unqualified_count_star(hp.left_expr) or expr_has_unqualified_count_star(hp.right_expr):
                count_star = True
                break
    if count_star:
        issues.append(
            IntentIssue.make(
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


def validate_having_agg_per_role(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that HAVING aggregation functions are valid for each. column's role."""
    issues = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:
        agg_expr = hp.left_expr.primary_term
        if not agg_expr:
            continue
        func, actual_target, _ = extract_agg_col(agg_expr)
        if not func or not actual_target or actual_target == "*":
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
                        IntentIssue.make(
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
                IntentIssue.make(
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
    debug(f"[validation_rules.validate_having_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_select_agg_per_role(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that SELECT aggregation functions are valid for each. column's role."""
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
                            IntentIssue.make(
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
                IntentIssue.make(
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
    debug(f"[validation_rules.validate_select_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_select_agg_semantics(
    select_cols: list[SelectCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate that SELECT aggregation functions are semantically. appropriate for column types. Errors for SUM/AVG on non-numeric columns; warns for MIN/MAX on FREE_TEXT columns."""
    issues = []
    if not select_cols:
        return []
    numeric_aggs = {"sum", "avg", "stddev", "variance", "median"}
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
                IntentIssue.make(
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
            debug(f"[validation_rules.validate_select_agg_semantics] invalid {func_lower.upper()} on {actual_col}")
        elif func_lower in {"min", "max"} and not numeric and not temporal:
            col_role = col_meta.role if col_meta.role else None
            if col_role == ColumnRole.FREE_TEXT.value:
                issues.append(
                    IntentIssue.make(
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
                    f"[validation_rules.validate_select_agg_semantics] questionable {func_lower.upper()} on {actual_col}"
                )
    if issues:
        debug(f"[validation_rules.validate_select_agg_semantics] found {len(issues)} semantic issues")
    else:
        debug("[validation_rules.validate_select_agg_semantics] no semantic issues")
    return issues


def validate_order_by_agg_per_role(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that ORDER BY aggregation functions are valid for each. column's role."""
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
                            IntentIssue.make(
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
                IntentIssue.make(
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
    debug(f"[validation_rules.validate_order_by_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_order_by_agg_semantics(
    order_by_cols: list[OrderByCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate that ORDER BY aggregation functions are semantically. appropriate for column types. Errors for SUM/AVG on non-numeric columns; warns for MIN/MAX on FREE_TEXT columns."""
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
                IntentIssue.make(
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
            debug(f"[validation_rules.validate_order_by_agg_semantics] invalid {func_lower.upper()} on {actual_col}")
        elif func_lower in {"min", "max"} and not numeric and not temporal:
            col_role = col_meta.role if col_meta.role else None
            if col_role == ColumnRole.FREE_TEXT.value:
                issues.append(
                    IntentIssue.make(
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
                    f"[validation_rules.validate_order_by_agg_semantics] questionable {func_lower.upper()} on {actual_col}"
                )
    if issues:
        debug(f"[validation_rules.validate_order_by_agg_semantics] found {len(issues)} semantic issues")
    else:
        debug("[validation_rules.validate_order_by_agg_semantics] no semantic issues")
    return issues


def validate_scalar_func_type_semantics(
    select_cols: list[SelectCol], order_by_cols: list[OrderByCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate that scalar functions are appropriate for column types. and aggregation context. Errors when a non-aggregate-compatible scalar wraps an aggregation, or when a type-specific scalar (string, numeric, temporal) is applied to the wrong column type."""
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
                IntentIssue.make(
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
                IntentIssue.make(
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
                IntentIssue.make(
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
                IntentIssue.make(
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
            issues.extend(check_scalar_semantics(obc_scalar, obc.expr.primary_column, obc_agg, f"order_by_cols[{idx}]"))
    if issues:
        debug(
            f"[validation_rules.validate_scalar_func_type_semantics] found {len(issues)} semantic issues in {context}"
        )
    else:
        debug(f"[validation_rules.validate_scalar_func_type_semantics] no semantic issues in {context}")
    return issues


def validate_column_types(
    select_cols: list[SelectCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate that operations match their column types (heuristic. checks). Warns for numeric aggregations on text columns, date operations on non-date columns, and string operations on numeric columns."""
    issues = []
    debug("[validation_rules.validate_column_types] checking type consistency")
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
        if not vt:
            continue
        numeric = vt in ("integer", "number")
        date = vt == "date" or col_meta.role == ColumnRole.TEMPORAL.value
        text = vt == "string"
        if func_lower in numeric_aggs and text and not numeric:
            issues.append(
                IntentIssue.make(
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
            debug("[validation_rules.validate_column_types] type_mismatch: numeric_on_text")
        if func_lower in date_ops and not date:
            issues.append(
                IntentIssue.make(
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
            debug("[validation_rules.validate_column_types] type_mismatch: date_on_non_date")
        if func_lower in string_ops and numeric and "_id" not in col_name.lower():
            issues.append(
                IntentIssue.make(
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
            debug("[validation_rules.validate_column_types] TYPE MISMATCH: string op on numeric column")
    if issues:
        debug(f"[validation_rules.validate_column_types] FAILED with {len(issues)} issues")
    else:
        debug("[validation_rules.validate_column_types] PASSED")
    return issues


def validate_scalar_expression_semantics(
    select_cols: list[SelectCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate that scalar functions are applied to semantically. appropriate column types."""
    issues = []
    debug("[validation_rules.validate_scalar_expression_semantics] checking scalar semantics")
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
                    IntentIssue.make(
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
                    IntentIssue.make(
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
    debug(f"[validation_rules.validate_scalar_expression_semantics] {len(issues)} issues in {context}")
    return issues


def validate_temporal_columns(
    select_cols: list[SelectCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate temporal aggregates against date-type columns in the. intent."""
    issues = []
    temporal_ops = {"latest", "recent", "last", "first", "earliest", "oldest", "newest"}
    agg_funcs: set[str] = set()
    for sc in select_cols:
        if not sc.is_aggregated:
            continue
        fn, _, _ = extract_agg_col(sc.expr.primary_term)
        if fn:
            agg_funcs.add(fn)
    if not (agg_funcs & temporal_ops):
        return []
    debug("[validation_rules.validate_temporal_columns] checking temporal column presence")
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
                if col_meta.value_type == "date" or col_meta.role == ColumnRole.TEMPORAL.value:
                    has_date_column = True
                    break
    if not has_date_column:
        issues.append(
            IntentIssue.make(
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
        debug("[validation_rules.validate_temporal_columns] AMBIGUITY: temporal ops but no date column")
    return issues


def validate_pk_fk_aggregation(
    select_cols: list[SelectCol], schema: SchemaGraph, context: str = "main"
) -> list[IntentIssue]:
    """Validate that primary-key and foreign-key columns are not. aggregated with SUM or AVG."""
    issues = []
    suspicious_aggs = {"sum", "avg"}
    debug("[validation_rules.validate_pk_fk_aggregation] checking PK/FK aggregation")
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
                IntentIssue.make(
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
            debug(f"[validation_rules.validate_pk_fk_aggregation] {func_lower.upper()} on PK/FK: {actual_col}")
    return issues


def validate_aggregate_join_fan_out(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    context: str,
    *,
    join_signature: list[str] | None = None,
    from_anchor: str | None = None,
) -> list[IntentIssue]:
    """Refuse aggregates that would be computed over rows duplicated by the resolved join path."""
    signature = list(join_signature or intent.chosen_join_path_signature or [])
    if not signature:
        return []
    anchor = from_anchor or (intent.tables[0] if intent.tables else None)
    main_tables = {phys_table_key(t) for t in intent_join_reachability_tables(intent)}
    issues: list[IntentIssue] = []
    for func, tbl, col, _distinct in collect_fan_out_sensitive_aggregates(intent):
        if phys_table_key(tbl) not in main_tables:
            continue
        hits = multiplying_edges_for_table(signature, tbl, schema, from_anchor=anchor)
        if not hits:
            continue
        edge = hits[0]["edge"]
        issues.append(
            IntentIssue.make(
                issue_id=f"aggregate_join_fan_out_{context.replace(' ', '_')}_{tbl}_{func}",
                category=FailureCategory.AGGREGATION_SEMANTICS,
                severity="error",
                message=(
                    f"{context}: {func.upper()}({tbl}.{col}) would be computed over rows duplicated by "
                    f"join edge {edge!r}. Aggregate at the multiplied grain and group, or express the "
                    f"relationship as a semi_join probe that filters without multiplying."
                ),
                context={
                    "aggregate": func,
                    "table": tbl,
                    "column": col,
                    "edge": edge,
                    "tables": sorted(main_tables),
                },
            )
        )
    return issues


def validate_window_join_fan_out(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    context: str,
    *,
    join_signature: list[str] | None = None,
    from_anchor: str | None = None,
) -> list[IntentIssue]:
    """Refuse windows whose partition, ordering, or argument columns see join-multiplied rows."""
    signature = list(join_signature or intent.chosen_join_path_signature or [])
    if not signature or not intent.window_registry:
        return []
    anchor = from_anchor or (intent.tables[0] if intent.tables else None)
    main_tables = {phys_table_key(t) for t in intent_join_reachability_tables(intent)}
    row_level_ungrouped = (intent.grain or "row_level") == "row_level" and not intent.group_by_cols
    issues: list[IntentIssue] = []
    for wr in intent.window_registry or []:
        ws = wr.window_spec
        referenced: list[tuple[str, str, str]] = []
        if row_level_ungrouped:
            for pe in ws.partition_by or []:
                for col in extract_columns_from_expr(pe):
                    if "." in col:
                        tbl, cn = col.split(".", 1)
                        referenced.append(("partition", tbl, cn))
            for ob in ws.order_by or []:
                for col in extract_columns_from_expr(ob.expr):
                    if "." in col:
                        tbl, cn = col.split(".", 1)
                        referenced.append(("order", tbl, cn))
        if ws.argument is not None:
            for col in extract_columns_from_expr(ws.argument):
                if "." in col:
                    tbl, cn = col.split(".", 1)
                    referenced.append(("argument", tbl, cn))
        for role, tbl, cn in referenced:
            if phys_table_key(tbl) not in main_tables:
                continue
            hits = multiplying_edges_for_table(signature, tbl, schema, from_anchor=anchor)
            if not hits:
                continue
            edge = hits[0]["edge"]
            issues.append(
                IntentIssue.make(
                    issue_id=f"window_join_fan_out_{context}_{wr.registry_id}_{tbl}_{role}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="error",
                    message=(
                        f"{context}: window over {tbl}.{cn} would see rows duplicated by "
                        f"join edge {edge!r}. Aggregate at the multiplied grain and group, or "
                        f"express the relationship as a semi_join probe that filters without multiplying."
                    ),
                    context={
                        "table": tbl,
                        "column": cn,
                        "edge": edge,
                        "registry_id": wr.registry_id,
                        "role": role,
                    },
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
                if len(leaf.add_values) == 1 and float(cast(Any, leaf.add_values[0].value or 0)) == 1.0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


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
                IntentIssue.make(
                    issue_id=f"not_selectable_{loc.replace(' ', '_').replace('[', '_').replace(']', '_')}_{qc.replace('.', '_')}",
                    category=FailureCategory.ACCESS_POLICY,
                    severity="error",
                    message=f"{loc}: column '{cref}' is not selectable under sensitivity policy",
                    context={"column": cref, "location": loc},
                )
            )
    return issues


def selectability_exempt_qualified_refs(expr: NormalizedExpr, schema: SchemaGraph) -> set[str]:
    """Return qualified refs that appear only as arguments to allowed. aggregate functions."""
    _ = schema
    exempt: set[str] = set()
    for g in expr.add_groups + expr.sub_groups:
        exempt |= _qualified_refs_under_aggregate_mulgroup(g)
    return exempt


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
    if window_registry is None:
        window_registry = list(WindowRegistryStep.current_steps())
    if case_registry is None:
        case_registry = list(CaseRegistryStep.current_steps())
    for idx, sc in enumerate(select_cols or []):
        detail = f"select_cols[{idx}]"
        parts = sc.effective_parts(window_registry, case_registry)
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
