"""Intent-level validation for grain, aggregations, WHERE/HAVING placement, CTE consistency, and related repairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._constants import (
    AGG_QUANTITY_RE,
    COMPATIBLE_TYPE_PAIRS,
    COUNT_THRESHOLD_TABLE_RE,
    DATE_FRIENDLY_VALUE_TYPES,
    INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_BY_SENSITIVITY_VALUE,
    INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_DEFAULT,
    NUMERIC_ONLY_AGGREGATIONS,
    NUMERIC_RESULT_AGGS,
    NUMERIC_RESULT_OPS,
    NUMERIC_RESULT_SCALARS,
    QUESTION_DISTINCT_KEYWORD_RE,
    QUESTION_NUMERIC_LITERAL_RE,
    QUESTION_TOP_N_PHRASE_RE,
    QUESTION_YEAR_IN_STRING_RE,
    SCALAR_FUNCTIONS_NUMERIC,
    SCALAR_FUNCTIONS_STRING,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_GRAINS,
)
from ._contracts_base import (
    FailureCategory,
    HavingParam,
    LogicalIntent,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    SensitivityClassification,
    WhereParam,
    expr_registry_ref,
    having_leaves,
    where_leaves,
)
from ._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from ._contracts_schema import CteOutputColumnMeta, IntentIssue, SchemaGraph, make_intent_issue, registry_render_scope
from ._core_utils import debug, stable_json
from ._intent_expr import concat_logical_intent_prose, extract_columns_from_expr
from ._validation_schema import (
    expr_has_arithmetic,
    expr_result_is_numeric,
    extract_agg_col,
    get_col_meta,
    get_col_type,
    is_col_arithmetic_role,
    is_col_numeric,
    is_date_column_subtraction,
    is_date_integer_day_arithmetic,
    selectability_exempt_qualified_refs,
)


def _column_meta_or_none(schema: SchemaGraph, ref: str) -> Any:
    """Look up column metadata for a qualified ``table.column`` reference, or return None."""
    if "." not in ref:
        return None
    parts = ref.split(".", 1)
    if len(parts) != 2:
        return None
    return schema.get_column(parts[0], parts[1])


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
                    make_intent_issue(
                        issue_id=f"deny_bare_select_{context}_{idx}",
                        category=FailureCategory.DENY_BARE_SELECT,
                        severity="error",
                        message=(
                            f"{context}: denied column {t}.{c} cannot appear as a bare (non-aggregated) select column"
                        ),
                        context={"table": t, "column": c, "location": context},
                    )
                )

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
            make_intent_issue(
                issue_id=f"denied_reference_{location}_{idx}_{t}_{c}",
                category=FailureCategory.DENIED_REFERENCE,
                severity="error",
                message=f"{location}: denied column {t}.{c} cannot be referenced",
                context={"table": t, "column": c, "location": location},
            )
        )

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
        where_leaves(intent.where) or [],
        intent.group_by_cols or [],
        having_leaves(intent.having) or [],
        intent.order_by_cols or [],
        "main query",
    )
    for cte in intent.cte_steps or []:
        _scan_intent(
            cte.select_cols or [],
            where_leaves(cte.where) or [],
            cte.group_by_cols or [],
            having_leaves(cte.having) or [],
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
                    make_intent_issue(
                        issue_id=f"sensitive_group_by_{location}_{idx}_{t}_{c}",
                        category=FailureCategory.SENSITIVE_GROUP_BY,
                        severity="error",
                        message=f"{location}: sensitive column {t}.{c} cannot be used in GROUP BY",
                        context={"table": t, "column": c, "location": location},
                    )
                )

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
                    make_intent_issue(
                        issue_id=f"sensitive_order_by_{location}_{idx}_{t}_{c}",
                        category=FailureCategory.ORDER_BY_VALIDITY,
                        severity="error",
                        message=f"{location}: sensitive column {t}.{c} cannot be used in ORDER BY",
                        context={"table": t, "column": c, "location": location},
                    )
                )

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
        sk = str(meta.sensitivity.value)
        tpl = INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_BY_SENSITIVITY_VALUE.get(
            sk, INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_DEFAULT
        )
        msg = tpl.format(location=location, table=t, column=c, surface=surface)
        category = FailureCategory.WHERE_VALIDITY if surface == "WHERE" else FailureCategory.HAVING_SEMANTIC
        suf = f"_{id_suffix}" if id_suffix else ""
        issues.append(
            make_intent_issue(
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

    _scan_filters(where_leaves(intent.where) or [], "main query")
    _scan_having(having_leaves(intent.having) or [], "main query")
    main_names = {t.lower() for t in intent.tables or []}
    for cte in intent.cte_steps or []:
        if cte.cte_name.lower() not in main_names:
            continue
        loc = f"CTE {cte.cte_name}"
        _scan_filters(where_leaves(cte.where) or [], loc)
        _scan_having(having_leaves(cte.having) or [], loc)
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
                            make_intent_issue(
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
                            make_intent_issue(
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
                        make_intent_issue(
                            issue_id=f"intent_empty_window_{loc}_{col.replace('.', '_')}",
                            category=FailureCategory.INTENT_EMPTY_WINDOW,
                            severity="error",
                            message=f"{loc}: empty temporal window on {col} (identical lower and upper bound expressions)",
                            context={"location": loc, "column": col},
                        )
                    )

    pv_main = intent.param_values or {}
    _scan_filters(where_leaves(intent.where) or [], pv_main, "main query")
    main_names = {t.lower() for t in intent.tables or []}
    for cte in intent.cte_steps or []:
        if cte.cte_name.lower() not in main_names:
            continue
        pv_cte = cte.param_values or pv_main
        _scan_filters(where_leaves(cte.where) or [], pv_cte, f"CTE {cte.cte_name}")
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
        f"[validation_semantic.validate_grain_consistency] grain={grain}, group_by={len(group_by_cols)}, having={len(having_param)}"
    )
    if grain not in VALID_GRAINS:
        issues.append(
            make_intent_issue(
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
            make_intent_issue(
                issue_id=f"grouped_without_group_by_{context}",
                category=FailureCategory.GRAIN_CONSISTENCY,
                severity="error",
                message=f"Grouped grain without GROUP BY columns in {context}",
                context={"grain": grain, "location": context},
            )
        )
        debug("[validation_semantic.validate_grain_consistency] grouped grain without group_by")
    if grain in {"scalar", "row_level"} and has_group_by:
        issues.append(
            make_intent_issue(
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
        debug(f"[validation_semantic.validate_grain_consistency] group_by present but grain={grain}")
    if has_agg and grain == "row_level":
        agg_funcs = [sc.expr.primary_term for sc in select_cols if sc.is_aggregated]
        issues.append(
            make_intent_issue(
                issue_id=f"agg_with_row_level_{context}_{','.join(agg_funcs)}",
                category=FailureCategory.GRAIN_CONSISTENCY,
                severity="error",
                message=f"Aggregation functions {agg_funcs} with row_level grain in {context}",
                context={"agg_funcs": agg_funcs, "grain": grain, "location": context},
            )
        )
        debug("[validation_semantic.validate_grain_consistency] agg funcs with row_level grain")
    if has_having and grain not in {"grouped", "scalar"}:
        issues.append(
            make_intent_issue(
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
        debug(f"[validation_semantic.validate_grain_consistency] HAVING without aggregation: grain={grain}")
    debug(f"[validation_semantic.validate_grain_consistency] {len(issues)} issues in {context}")
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
        make_intent_issue(
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
    debug(f"[validation_semantic.validate_grouped_requires_aggregation] grouped without aggregation in {context}")
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
            make_intent_issue(
                issue_id=f"case_branch_aggregation_without_group_by_{context}_{label}",
                category=FailureCategory.HAVING_AGGREGATION,
                severity="error",
                message=(
                    f"CASE branch condition references aggregates in {context} ({label}) "
                    "but the scope has no GROUP BY. Add the appropriate group_by_cols or rewrite "
                    "the branch condition to a row-level predicate."
                ),
                context={"location": context, "where": label},
            )
        )

    for step in case_registry or []:
        rid = getattr(step, "registry_id", "") or "?"
        _check(getattr(step, "case_when", None), f"case_registry[{rid}]")

    if issues:
        debug(
            f"[validation_semantic.validate_case_branch_aggregation_consistency] "
            f"{len(issues)} aggregated CASE branch(es) without GROUP BY in {context}"
        )
    return issues


def validate_semantic_contradictions(
    select_cols: list[SelectCol], natural_language: str, grain: str, expected_rows: str, context: str = "main"
) -> list[IntentIssue]:
    """Check for contradictory operations in the intent."""
    issues = []
    debug("[validation_semantic.validate_semantic_contradictions] checking for contradictions")
    agg_funcs = {extract_agg_col(sc.expr.primary_term)[0] for sc in select_cols if sc.is_aggregated} - {None}
    contradictory_pairs = [
        ({"highest", "max"}, {"lowest", "min"}),
        ({"most", "maximum"}, {"least", "minimum"}),
        ({"first", "earliest"}, {"last", "latest"}),
    ]
    for set1, set2 in contradictory_pairs:
        if agg_funcs & set1 and agg_funcs & set2:
            issues.append(
                make_intent_issue(
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
                f"[validation_semantic.validate_semantic_contradictions] CONTRADICTION: {set1 & agg_funcs} vs {set2 & agg_funcs}"
            )
    if grain == "scalar" and expected_rows in {"few", "many"}:
        issues.append(
            make_intent_issue(
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
            f"[validation_semantic.validate_semantic_contradictions] CONTRADICTION: grain=scalar but expected_rows={expected_rows}"
        )
    nl = natural_language.lower() if natural_language else ""
    contradiction_patterns = [
        ("never", "total"),
        ("no records", "count"),
        ("zero", "greater than"),
        ("empty", "count"),
    ]
    for pattern1, pattern2 in contradiction_patterns:
        if pattern1 in nl and pattern2 in nl:
            issues.append(
                make_intent_issue(
                    issue_id=f"nl_contradiction_{pattern1.replace(' ', '_')}_{pattern2.replace(' ', '_')}",
                    category=FailureCategory.SEMANTIC_CONTRADICTION,
                    severity="warning",
                    message=f"Intent may contain contradiction: mentions '{pattern1}' and '{pattern2}'",
                    context={
                        "pattern1": pattern1,
                        "pattern2": pattern2,
                        "location": context,
                    },
                )
            )
            debug(
                f"[validation_semantic.validate_semantic_contradictions] POTENTIAL CONTRADICTION: '{pattern1}' and '{pattern2}'"
            )
    debug(f"[validation_semantic.validate_semantic_contradictions] {len(issues)} issues")
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
                        make_intent_issue(
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
                        make_intent_issue(
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
                        make_intent_issue(
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
    debug("[validation_semantic.validate_expr_vs_expr_where] checking expr-vs-expr type compatibility")
    for fp in where_params:
        if not fp.right_expr:
            continue
        left_col = fp.left_expr.primary_column
        right_col = fp.right_expr.primary_column
        if left_col == right_col:
            issues.append(
                make_intent_issue(
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
            debug(f"[validation_semantic.validate_expr_vs_expr_where] self-comparison: {left_col}")
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
                        make_intent_issue(
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
                    debug(
                        f"[validation_semantic.validate_expr_vs_expr_where] type mismatch: {left_type} vs {right_type}"
                    )
        left_meta = get_col_meta(left_col, schema, cte_outputs)
        right_meta = get_col_meta(right_col, schema, cte_outputs)
        if left_meta and right_meta:
            if left_meta.is_primary_key or right_meta.is_primary_key:
                issues.append(
                    make_intent_issue(
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
                debug(f"[validation_semantic.validate_expr_vs_expr_where] PK comparison: {left_col}")
    debug(f"[validation_semantic.validate_expr_vs_expr_where] {len(issues)} issues in {context}")
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
    debug("[validation_semantic.validate_agg_vs_agg_having] checking agg-vs-agg type compatibility")
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
                make_intent_issue(
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
            debug(f"[validation_semantic.validate_agg_vs_agg_having] self-comparison: {left_term}")
            continue
        if left_target and left_target != "*" and right_target and right_target != "*":
            left_type = get_col_type(left_target, schema, cte_outputs)
            right_type = get_col_type(right_target, schema, cte_outputs)
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
                            make_intent_issue(
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
                            f"[validation_semantic.validate_agg_vs_agg_having] type mismatch: {left_type} vs {right_type}"
                        )
    debug(f"[validation_semantic.validate_agg_vs_agg_having] {len(issues)} issues in {context}")
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
        debug(f"[validation_semantic.validate_select_expr_types] {len(issues)} issues in {context}")
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
        debug(f"[validation_semantic.validate_order_by_expr_types] {len(issues)} issues in {context}")
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
                    make_intent_issue(
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
            and fp.op not in NUMERIC_RESULT_OPS
            and fp.op not in ("is null", "is not null")
        ):
            issues.append(
                make_intent_issue(
                    issue_id=f"where_op_on_arith_{pk}_{fp.op}",
                    category=FailureCategory.EXPRESSION_TYPE,
                    severity="error",
                    message=f"Operator '{fp.op}' invalid on arithmetic expression in filter '{pk}' in {context}. Expected: {sorted(NUMERIC_RESULT_OPS)}",
                    context={
                        "param_key": pk,
                        "operator": fp.op,
                        "valid_ops": sorted(NUMERIC_RESULT_OPS),
                        "location": context,
                    },
                )
            )
    if issues:
        debug(f"[validation_semantic.validate_where_expr_types] {len(issues)} issues in {context}")
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
                    make_intent_issue(
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
        debug(f"[validation_semantic.validate_having_expr_types] {len(issues)} issues in {context}")
    return issues


def _validate_concat_group(
    group: MulGroup, location: str, context: str, *, parent_is_distinct_count: bool = False
) -> list[IntentIssue]:
    """Enforce structural rules for a MulGroup whose outer scalar. function is CONCAT."""
    issues: list[IntentIssue] = []
    if group.divide:
        issues.append(
            make_intent_issue(
                issue_id=f"concat_divide_{location}",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message=f"CONCAT MulGroup at {location} in {context} must not carry divide terms.",
                context={"location": location},
            )
        )
    if group.coefficient != 1.0 or (group.coeff_param_key or "").strip():
        issues.append(
            make_intent_issue(
                issue_id=f"concat_coeff_{location}",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message=f"CONCAT MulGroup at {location} in {context} must use coefficient 1.0 without coeff_param_key.",
                context={"location": location},
            )
        )
    if (group.inner_scalar_func or "").strip():
        issues.append(
            make_intent_issue(
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
                make_intent_issue(
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
            make_intent_issue(
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
                make_intent_issue(
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
    for fp in where_leaves(intent.where) or []:
        pk = fp.param_key or "unknown"
        issues.extend(_walk_expr_concat_mulgroups(fp.left_expr, f"{context} where_{pk}_left", context))
        if fp.right_expr:
            issues.extend(_walk_expr_concat_mulgroups(fp.right_expr, f"{context} where_{pk}_right", context))
    for hp in having_leaves(intent.having) or []:
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
        for fp in where_leaves(cte.where) or []:
            pk = fp.param_key or "unknown"
            issues.extend(_walk_expr_concat_mulgroups(fp.left_expr, f"{cctx} where_{pk}_left", cctx))
            if fp.right_expr:
                issues.extend(_walk_expr_concat_mulgroups(fp.right_expr, f"{cctx} where_{pk}_right", cctx))
        for hp in having_leaves(cte.having) or []:
            pk = hp.param_key or "unknown"
            issues.extend(_walk_expr_concat_mulgroups(hp.left_expr, f"{cctx} having_{pk}_left", cctx))
            if hp.right_expr:
                issues.extend(_walk_expr_concat_mulgroups(hp.right_expr, f"{cctx} having_{pk}_right", cctx))
    if issues:
        debug(f"[validation_semantic.validate_concat_mulgroups_in_runtime] {len(issues)} issues in {context}")
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
    debug("[validation_semantic.validate_arith_expression_semantics] checking arithmetic semantics")
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
    debug(f"[validation_semantic.validate_arith_expression_semantics] {len(issues)} issues in {context}")
    return issues


def _term_has_aggregation(term: Any) -> bool:
    """Return whether a single multiply/divide term contains an aggregation call. Accepts a ``NormalizedExpr`` (current contract) or a raw SQL string (legacy). Walks nested groups to detect any agg_func or raw_sql aggregation pattern."""
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
    debug("[validation_semantic.validate_where_no_aggregation] checking filter aggregation ban")
    for fp in where_params or []:
        pk = fp.param_key or "unknown"
        if fp.left_expr.has_aggregation:
            issues.append(
                make_intent_issue(
                    issue_id=f"where_has_aggregation_{pk}_left",
                    category=FailureCategory.WHERE_AGGREGATION,
                    severity="error",
                    message=f"Filter '{pk}' left expression contains aggregation in {context}; use HAVING instead of WHERE",
                    context={"param_key": pk, "side": "left", "location": context},
                )
            )
        if fp.right_expr and fp.right_expr.has_aggregation:
            issues.append(
                make_intent_issue(
                    issue_id=f"where_has_aggregation_{pk}_right",
                    category=FailureCategory.WHERE_AGGREGATION,
                    severity="error",
                    message=f"Filter '{pk}' right expression contains aggregation in {context}; use HAVING instead of WHERE",
                    context={"param_key": pk, "side": "right", "location": context},
                )
            )
    if issues:
        debug(f"[validation_semantic.validate_where_no_aggregation] {len(issues)} issues in {context}")
    return issues


def validate_having_operator_is_numeric(having_param: list[HavingParam], context: str = "main") -> list[IntentIssue]:
    """Validate that each HAVING operator is a numeric comparison or an. explicit null check."""
    allowed = frozenset({"=", "!=", ">", "<", ">=", "<=", "is null", "is not null"})
    issues: list[IntentIssue] = []
    for hp in having_param or []:
        pk = hp.param_key or "unknown"
        op_norm = (hp.op or "=").strip().lower()
        if op_norm not in allowed:
            issues.append(
                make_intent_issue(
                    issue_id=f"having_non_numeric_op_{pk}",
                    category=FailureCategory.WRONG_HAVING,
                    severity="error",
                    message=(
                        f"HAVING predicate '{pk}' in {context} uses operator {hp.op!r}; "
                        "only numeric comparisons and null checks (=, !=, >, <, >=, <=, is null, is not null) are allowed in HAVING"
                    ),
                    context={"param_key": pk, "location": context, "op": hp.op},
                )
            )
    if issues:
        debug(f"[validation_semantic.validate_having_operator_is_numeric] {len(issues)} issues in {context}")
    return issues


def validate_having_requires_aggregation(
    having_param: list[HavingParam], context: str = "main", *, group_by_cols: list[Any] | None = None
) -> list[IntentIssue]:
    """Validate that each HAVING condition contains at least one. aggregation (left or right expression). When ``group_by_cols`` is empty but ``having_param`` is non-empty, emit ``having_without_group_by``."""
    issues: list[IntentIssue] = []
    debug("[validation_semantic.validate_having_requires_aggregation] checking having aggregation requirement")
    if having_param and not (group_by_cols or []):
        issues.append(
            make_intent_issue(
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
                make_intent_issue(
                    issue_id=f"having_missing_aggregation_{pk}",
                    category=FailureCategory.HAVING_AGGREGATION,
                    severity="error",
                    message=f"Having '{pk}' has no aggregation in {context}; belongs in WHERE not HAVING",
                    context={"param_key": pk, "location": context},
                )
            )
    if issues:
        debug(f"[validation_semantic.validate_having_requires_aggregation] {len(issues)} issues in {context}")
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
        make_intent_issue(
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
    debug("[validation_semantic.validate_predicate_sidedness] checking predicate sidedness")
    for fp in where_params or []:
        issues.extend(_predicate_sidedness_issues(fp.left_expr, fp.right_expr, fp.op, fp.param_key, "where", context))
    for hp in having_param or []:
        issues.extend(_predicate_sidedness_issues(hp.left_expr, hp.right_expr, hp.op, hp.param_key, "having", context))
    if issues:
        debug(f"[validation_semantic.validate_predicate_sidedness] {len(issues)} issues in {context}")
    return issues


def _check_nested_aggregation(expr: NormalizedExpr, location: str, context: str) -> list[IntentIssue]:
    """Check a single `NormalizedExpr` for double-wrap (nested) aggregation."""
    issues: list[IntentIssue] = []
    for g in expr.add_groups + expr.sub_groups:
        group_inline_agg = any(_term_has_aggregation(t) for t in g.multiply + g.divide)
        if expr.agg_func and g.agg_func:
            issues.append(
                make_intent_issue(
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
                make_intent_issue(
                    issue_id=f"nested_agg_expr_inline_{location}",
                    category=FailureCategory.NESTED_AGGREGATION,
                    severity="error",
                    message=f"Nested aggregation: expr-level {expr.agg_func.upper()} wraps inline aggregation at {location} in {context}",
                    context={"outer": expr.agg_func, "location": location},
                )
            )
        if g.agg_func and group_inline_agg:
            issues.append(
                make_intent_issue(
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
    debug("[validation_semantic.validate_no_nested_aggregation] checking nested aggregation")
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
        debug(f"[validation_semantic.validate_no_nested_aggregation] {len(issues)} issues in {context}")
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
            make_intent_issue(
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
                make_intent_issue(
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
    debug("[validation_semantic.validate_mixed_aggregation_in_mulgroup] checking mixed aggregation")
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
        debug(f"[validation_semantic.validate_mixed_aggregation_in_mulgroup] {len(issues)} issues in {context}")
    return issues


def validate_order_by_aggregation_context(
    order_by_cols: list[OrderByCol], grain: str, context: str = "main"
) -> list[IntentIssue]:
    """Validate that ORDER BY aggregation expressions are compatible. with the query grain."""
    issues: list[IntentIssue] = []
    debug(f"[validation_semantic.validate_order_by_aggregation_context] grain={grain}")
    if grain != "row_level":
        return issues
    for idx, obc in enumerate(order_by_cols or []):
        if obc.expr.has_aggregation:
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_agg_row_level_{idx}",
                    category=FailureCategory.ORDER_BY_AGGREGATION,
                    severity="error",
                    message=f"Order-by[{idx}] contains aggregation but grain is row_level in {context}",
                    context={"index": idx, "grain": grain, "location": context},
                )
            )
    if issues:
        debug(f"[validation_semantic.validate_order_by_aggregation_context] {len(issues)} issues in {context}")
    return issues


def validate_select_group_by_membership(
    select_cols: list[SelectCol], group_by_cols: list[NormalizedExpr], grain: str, context: str = "main"
) -> list[IntentIssue]:
    """Validate that every non-aggregated SELECT column appears in GROUP BY when mixed aggregation is present."""
    issues: list[IntentIssue] = []
    debug(f"[validation_semantic.validate_select_group_by_membership] grain={grain}, group_by={len(group_by_cols)}")
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
                make_intent_issue(
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
        debug(f"[validation_semantic.validate_select_group_by_membership] {len(issues)} issues in {context}")
    return issues


def _validate_cte_grain_complete(cte: RuntimeCteStep, context: str) -> list[IntentIssue]:
    """Validate one CTE body against the grain state machine using. structural facts only. Emits errors only for impossible combinations of ``GROUP BY``, ``SELECT`` aggregation, ``HAVING``, and declared ``grain``. Wrong labels without structural conflict are left to deterministic repair."""
    with registry_render_scope(cte.window_registry, cte.case_registry):
        issues: list[IntentIssue] = []
        grain = cte.grain or "row_level"
        group_by = cte.group_by_cols or []
        select_cols = cte.select_cols or []
        having_param = having_leaves(cte.having) or []
        has_agg = any(sc.is_aggregated for sc in select_cols)
        all_cols_agg = all(sc.is_aggregated for sc in select_cols) if select_cols else True
        if having_param and not has_agg:
            issues.append(
                make_intent_issue(
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
                make_intent_issue(
                    issue_id=f"cte_group_by_without_agg_{cte.cte_name}",
                    category=FailureCategory.CTE_GRAIN_CONSISTENCY,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has GROUP BY columns but no aggregation in {context}",
                    context={"cte_name": cte.cte_name, "location": context},
                )
            )
        if has_agg and not group_by and not all_cols_agg:
            issues.append(
                make_intent_issue(
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
                make_intent_issue(
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
                make_intent_issue(
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
                make_intent_issue(
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
        debug(f"[validation_semantic._validate_cte_grain_complete] {len(issues)} issues for CTE '{cte.cte_name}'")
        return issues


def validate_cte_grain_consistency(cte: RuntimeCteStep, context: str) -> list[IntentIssue]:
    """Validate CTE grain structure; delegates to. :func:`_validate_cte_grain_complete`."""
    return _validate_cte_grain_complete(cte, context)


def _cte_step_declares_window(cte: RuntimeCteStep) -> bool:
    """Return True when any SELECT column on the CTE step carries a. window specification."""
    if cte.window_registry:
        return True
    return any((expr_registry_ref(sc.expr) or "").startswith("w") for sc in (cte.select_cols or []))


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
        f"[validation_semantic.validate_cte_dependency_grains] validating {len(cte_steps)} CTEs against main grain '{main_grain}'"
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
                        make_intent_issue(
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
    debug(f"[validation_semantic.validate_cte_dependency_grains] {len(issues)} grain compatibility issues")
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
            make_intent_issue(
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
    """Reserved hook; NL keyword heuristics are not used to drive validation."""
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

    add_filters(where_leaves(intent.where))
    add_having(having_leaves(intent.having))
    for sc in intent.select_cols or []:
        add_expr(sc.expr)
    for gb in intent.group_by_cols or []:
        add_expr(gb)
    for ob in intent.order_by_cols or []:
        add_expr(ob.expr)
    for cte in intent.cte_steps or []:
        add_filters(where_leaves(cte.where))
        add_having(having_leaves(cte.having))
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
    """Flag digit runs in planner prose that are missing from filters, HAVING, or limit."""
    issues: list[IntentIssue] = []
    if logical_intent is None:
        return issues
    coverage_source_text = concat_logical_intent_prose(logical_intent)
    if not coverage_source_text:
        return issues

    top_n_numbers: set[str] = set()
    for m in QUESTION_TOP_N_PHRASE_RE.finditer(coverage_source_text):
        for tok in m.group().split():
            if tok.isdigit():
                top_n_numbers.add(tok)

    all_numbers = QUESTION_NUMERIC_LITERAL_RE.findall(coverage_source_text)
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
            make_intent_issue(
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
            f"[validation_semantic.validate_logical_intent_numeric_coverage] "
            f"number '{num_str}' not found in intent conditions"
        )
    return issues


def validate_question_distinct_hint(
    natural_language: str, select_cols: list[SelectCol], context: str = "main", distinct_select_index: int = -1
) -> list[IntentIssue]:
    """Flag when the question requests distinct results but no. `DISTINCT` appears in any expression."""
    issues: list[IntentIssue] = []
    if not natural_language:
        return issues
    if not QUESTION_DISTINCT_KEYWORD_RE.search(natural_language):
        return issues
    if distinct_select_index >= 0:
        return issues
    for sc in select_cols:
        raw = sc.expr if isinstance(sc.expr, str) else str(sc.expr)
        if "DISTINCT" in raw.upper():
            return issues
    issues.append(
        make_intent_issue(
            issue_id=f"missing_distinct_{context}",
            category=FailureCategory.MISSING_DISTINCT,
            severity="warning",
            message=(
                f"Question explicitly requests distinct/unique results "
                f"but no DISTINCT keyword found in any expression in "
                f"{context}."
            ),
            context={"location": context},
        )
    )
    debug(
        "[validation_semantic.validate_question_distinct_hint] "
        "question has distinct/unique keyword but intent lacks DISTINCT"
    )
    return issues


def _english_plural_forms(word: str) -> list[str]:
    """Return plausible English plural forms for `word`, always. including the original token."""
    forms = [word]
    w = word.lower()
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        forms.append(w[:-1] + "ies")
    elif w.endswith(("s", "sh", "ch", "x", "z")):
        forms.append(w + "es")
    else:
        forms.append(w + "s")
    return forms


def validate_threshold_missing_having(
    natural_language: str,
    select_cols: list[SelectCol],
    having_param: list[HavingParam],
    grain: str,
    context: str = "main",
) -> list[IntentIssue]:
    """Detect threshold phrases where aggregation exists but HAVING is. absent."""
    issues: list[IntentIssue] = []
    if not natural_language:
        return issues
    if grain != "grouped":
        return issues
    has_agg = any(sc.is_aggregated for sc in select_cols)
    if not has_agg:
        return issues
    if having_param:
        return issues
    match = AGG_QUANTITY_RE.search(natural_language)
    if not match:
        return issues
    issues.append(
        make_intent_issue(
            issue_id=f"threshold_missing_having_{context}",
            category=FailureCategory.THRESHOLD_MISSING_HAVING,
            severity="error",
            message=(
                f"Question contains threshold phrase '{match.group()}' and "
                f"intent has aggregation, but no HAVING condition is defined. "
                f"Add a HAVING clause for the threshold."
            ),
            context={"matched_phrase": match.group(), "location": context},
        )
    )
    debug(f"[validation_semantic.validate_threshold_missing_having] threshold phrase '{match.group()}' without HAVING")
    return issues


def validate_count_threshold_missing_having(
    natural_language: str,
    tables: list[str],
    having_param: list[HavingParam],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Flag count-threshold phrases that lack a HAVING clause."""
    issues: list[IntentIssue] = []
    if not natural_language:
        return issues

    match = COUNT_THRESHOLD_TABLE_RE.search(natural_language)
    if not match:
        return issues

    threshold_count = match.group(1)
    threshold_word = match.group(2)
    threshold_table = _resolve_word_to_table(threshold_word, schema)
    if not threshold_table:
        return issues

    if having_param:
        return issues

    fk_col = _find_fk_column_for_target(threshold_table, tables, schema)

    hint = (
        f"COUNT(DISTINCT {fk_col}) = {threshold_count}"
        if fk_col
        else f"COUNT(DISTINCT <fk_column_referencing_{threshold_table}>) = {threshold_count}"
    )

    issues.append(
        make_intent_issue(
            issue_id=f"count_threshold_missing_having_{context}",
            category=FailureCategory.COUNT_THRESHOLD_MISSING_HAVING,
            severity="error",
            message=(
                f"Question implies a count threshold of "
                f"{threshold_count} for entity '{threshold_table}' "
                f"but no HAVING clause is defined. Add a HAVING "
                f"condition with {hint}."
            ),
            context={
                "threshold_count": threshold_count,
                "threshold_table": threshold_table,
                "fk_column": fk_col or "",
                "location": context,
            },
        )
    )
    debug(
        f"[validation_semantic.validate_count_threshold_missing_having] "
        f"count threshold {threshold_count} for '{threshold_table}' "
        f"without HAVING"
    )
    return issues


def _resolve_word_to_table(word: str, schema: SchemaGraph) -> str | None:
    """Resolve a natural-language word to a schema table name."""
    word_lower = word.lower()
    lower_tables = {t.lower(): t for t in schema.tables}
    if word_lower in lower_tables:
        return lower_tables[word_lower]
    for tbl_lower, tbl_canonical in lower_tables.items():
        if word_lower in _english_plural_forms(tbl_lower):
            return tbl_canonical
    return None


def _find_fk_column_for_target(target_table: str, candidate_tables: list[str], schema: SchemaGraph) -> str | None:
    """Find an FK column on `candidate_tables` referencing. `target_table`."""
    for tbl in candidate_tables:
        tbl_meta = schema.tables.get(tbl)
        if not tbl_meta:
            continue
        for col_name, col_meta in tbl_meta.columns.items():
            if not col_meta.is_foreign_key or not col_meta.fk_target:
                continue
            if col_meta.fk_target[0] == target_table:
                return f"{tbl}.{col_name}"
    return None


def validate_for_each_grouping(
    natural_language: str,
    group_by_cols: list[NormalizedExpr],
    schema: SchemaGraph,
    has_aggregation: bool,
    context: str = "main",
) -> list[IntentIssue]:
    """Reserved hook; NL grouping heuristics are not used to drive validation."""
    _ = (natural_language, group_by_cols, schema, has_aggregation, context)
    return []
