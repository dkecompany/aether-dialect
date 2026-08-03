"""Structural intent repairs and deterministic filter/select normalization."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from difflib import get_close_matches
from typing import Any, Literal, cast

from ._config import PolicyConfig
from ._constants import (
    ARRAY_REWRITABLE_OPS,
    BOOLEAN_FALSY_VALUES,
    BOOLEAN_TRUTHY_VALUES,
    CUMULATIVE_PHRASING_RE,
    DATE_COLUMN_VALUE_TYPES,
    DATE_RESULT_SCALARS,
    DESCRIPTIVE_ALLOWED_VALUE_TYPES,
    DESCRIPTIVE_EXCLUDED_VALUE_TYPES,
    DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED,
    DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_CAP_REACHED,
    DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_ELIMINATED,
    DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT,
    DIAGNOSTIC_FUZZY_CUTOFF,
    ELIMINATE_REDUNDANT_KEY_JOINS_MAX_ITERATIONS,
    IDENTIFIER_RE,
    IMPOSSIBLE_HAVING_RE,
    INSTRUCTIONAL_SHAPE_PLACEHOLDER_TOKENS,
    INTENT_PLACEHOLDER_ANGLE_RE,
    MAX_REPAIR_ATTEMPTS_PER_CODE,
    NON_NUMERIC_AGGS_FOR_DATES,
    NULL_OP_DOUBLE_NEGATED_ALIASES,
    NULL_OP_NEGATED_ALIASES,
    NULL_OP_PLAIN_ALIASES,
    NULL_SENSITIVE_ELIMINATION_OPS,
    NUMERIC_DATA_TYPES,
    NUMERIC_RESULT_AGGS,
    NUMERIC_RESULT_SCALARS,
    OP_FLIP,
    RANGE_OPS,
    SOFT_DIAGNOSTIC_CODES,
    SQL_KEYWORDS,
    STRING_COLUMN_VALUE_TYPES,
    STRING_OPS,
    TABLE_SCOPE_REPAIR_REASON_TEXT,
    UNKNOWN_DATEPART_TO_EXTRACT_UNIT,
    WINDOW_AGG_FUNCTIONS,
    YEAR_LITERAL_COMPARISON_OPS,
    YEAR_LITERAL_RE,
)
from ._contracts_base import (
    FailureCategory,
    FederationManifest,
    HavingParam,
    InferenceTag,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    RawValue,
    SensitivityClassification,
    SqlDiagnostic,
    SqlDiagnosticCode,
    WhereParam,
    expr_registry_ref,
    having_leaves,
    map_predicate_group,
    merge_predicate_groups,
    predicate_group_from_list,
    reapply_predicate_leaves,
    rebuild_predicate_group_from_leaves,
    where_leaves,
)
from ._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol, TableScopeRepair, effective_select_parts
from ._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ColumnMetadata,
    FKEdge,
    IntentIssue,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
    make_intent_issue,
)
from ._core_utils import debug, notify, pipeline_trace, stable_json
from ._dialect_sqlglot_helper import array_storage_kind
from ._intent_expr import (
    expr_canonical_key,
    extract_columns_from_expr,
    is_plain_column_expr,
    map_case_branch_conditions,
    replace_refs_in_expr,
)
from ._sql_gen import (
    cte_emission_map,
    inner_equality_pairs_from_resolved_join_path,
    probe_cte_names,
    render_expr_sql,
)
from ._validation_schema import (
    selectability_exempt_qualified_refs,
    validate_join_path_reachability_for_tables,
    where_param_to_having_param,
)
from ._validation_semantic import validate_having_operator_is_numeric, validate_having_requires_aggregation


def dedup_extract_year_vs_column_literal(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
    """Drop bare four-digit year comparisons on a column when an EXTRACT(year FROM that column) filter already exists."""
    if not filters:
        return filters, False

    extract_years: set[str] = set()
    for fp in filters:
        if fp.left_expr.scalar_func == "extract" and fp.left_expr.scalar_func_args:
            if str(fp.left_expr.scalar_func_args[0]).lower() == "year":
                ref = fp.left_expr.primary_column
                if ref:
                    extract_years.add(ref.lower())

    if not extract_years:
        return filters, False

    new_filters: list[WhereParam] = []
    changed = False
    for fp in filters:
        is_bare_year_lit = False
        if fp.op in YEAR_LITERAL_COMPARISON_OPS and fp.raw_value is not None and not fp.right_expr:
            val = str(fp.raw_value).strip()
            if YEAR_LITERAL_RE.fullmatch(val):
                is_bare_year_lit = True

        if is_bare_year_lit:
            if not fp.left_expr.scalar_func and not fp.left_expr.agg_func and not fp.left_expr.inner_scalar_func:
                ref = fp.left_expr.primary_column
                if ref and ref.lower() in extract_years:
                    changed = True
                    continue

        new_filters.append(fp)

    return new_filters, changed


def apply_where_to_main_and_ctes(
    intent: RuntimeIntent, process_fn: Callable[[list[WhereParam]], tuple[list[WhereParam], bool]]
) -> RuntimeIntent:
    """Apply a where processor to the main intent and each CTE, merging results. Also extends the processor to every CASE WHEN branch whose ``condition_scope`` is ``"where"`` so that branch-shaped predicates receive identical repairs as flat ``where`` leaves. A processor that returns zero or multiple predicates for a single- element branch input keeps the original branch condition because a CASE branch holds exactly one predicate."""
    new_fp, main_changed = process_fn(where_leaves(intent.where) or [])
    if not intent.cte_steps:
        result = (
            replace(intent, where=rebuild_predicate_group_from_leaves(intent.where, new_fp)) if main_changed else intent
        )
        return _apply_where_processor_to_case_branches(result, process_fn)
    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps:
        cte_fp, c = process_fn(where_leaves(cte.where) or [])
        if c:
            cte_changed = True
        new_cte_steps.append(
            replace(cte, where=rebuild_predicate_group_from_leaves(cte.where, cte_fp) if c else cte.where)
        )
    if not main_changed and not cte_changed:
        return _apply_where_processor_to_case_branches(intent, process_fn)
    result = replace(intent, where=rebuild_predicate_group_from_leaves(intent.where, new_fp))
    if cte_changed:
        result = replace(result, cte_steps=new_cte_steps)
    return _apply_where_processor_to_case_branches(result, process_fn)


def apply_having_to_main_and_ctes(
    intent: RuntimeIntent, process_fn: Callable[[list[HavingParam]], tuple[list[HavingParam], bool]]
) -> RuntimeIntent:
    """Apply a HAVING processor to the main intent and each CTE, merging results. Also extends the processor to every CASE WHEN branch whose ``condition_scope`` is ``"having"``. The branch condition is wrapped as a one-element ``HavingParam`` list via :func:`where_param_to_having_param`, processed, and converted back via :func:`_having_param_to_where_param`. A processor that returns zero or multiple predicates keeps the original branch because a CASE branch holds exactly one."""
    new_hp, main_changed = process_fn(having_leaves(intent.having) or [])
    if not intent.cte_steps:
        result = (
            replace(intent, having=rebuild_predicate_group_from_leaves(intent.having, new_hp))
            if main_changed
            else intent
        )
        return _apply_having_processor_to_case_branches(result, process_fn)
    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps:
        cte_hp, c = process_fn(having_leaves(cte.having) or [])
        if c:
            cte_changed = True
        new_cte_steps.append(
            replace(cte, having=rebuild_predicate_group_from_leaves(cte.having, cte_hp) if c else cte.having)
        )
    if not main_changed and not cte_changed:
        return _apply_having_processor_to_case_branches(intent, process_fn)
    result = replace(intent, having=rebuild_predicate_group_from_leaves(intent.having, new_hp))
    if cte_changed:
        result = replace(result, cte_steps=new_cte_steps)
    return _apply_having_processor_to_case_branches(result, process_fn)


def _having_param_to_where_param(hp: HavingParam) -> WhereParam:
    """Translate a :class:`HavingParam` back into the matching :class:`WhereParam`."""
    return WhereParam(
        left_expr=hp.left_expr,
        op=hp.op,
        right_expr=hp.right_expr,
        value_type=hp.value_type,
        param_key=hp.param_key,
        raw_value=hp.raw_value,
    )


def _apply_where_processor_to_case_branches(
    intent: RuntimeIntent, process_fn: Callable[[list[WhereParam]], tuple[list[WhereParam], bool]]
) -> RuntimeIntent:
    """Run *process_fn* against every filter-scope CASE branch via :func:`map_case_branch_conditions`."""

    def _branch_transform(conds: list[WhereParam]) -> list[WhereParam]:
        new_list, _ = process_fn(conds)
        return new_list

    return map_case_branch_conditions(intent, _branch_transform, scopes=frozenset({"where"}))


def _apply_having_processor_to_case_branches(
    intent: RuntimeIntent, process_fn: Callable[[list[HavingParam]], tuple[list[HavingParam], bool]]
) -> RuntimeIntent:
    """Run *process_fn* against every having-scope CASE branch via :func:`map_case_branch_conditions`."""

    def _branch_transform(conds: list[WhereParam]) -> list[WhereParam]:
        h_in = [where_param_to_having_param(c) for c in conds]
        new_h, _ = process_fn(h_in)
        return [_having_param_to_where_param(h) for h in new_h]

    return map_case_branch_conditions(intent, _branch_transform, scopes=frozenset({"having"}))


def _dedup_contradictory_where_list(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
    """Drop range filters on a column that also has an equality on that. column."""
    eq_columns: set[str] = set()
    for fp in filters:
        col = fp.left_expr.primary_column or ""
        if fp.op == "=" and col:
            eq_columns.add(col)

    if not eq_columns:
        return filters, False

    kept: list[WhereParam] = []
    changed = False
    for fp in filters:
        col = fp.left_expr.primary_column or ""
        if col in eq_columns and fp.op in RANGE_OPS:
            debug(f"[intent_repair.dedup_contradictory_where] dropping {fp.op} on '{col}' that contradicts =")
            changed = True
            continue
        kept.append(fp)
    return kept, changed


def _dedup_contradictory_having_list(having: list[HavingParam]) -> tuple[list[HavingParam], bool]:
    """Drop range HAVING predicates on an aggregation expression that also has an equality on that key. Keys use :func:`expr_canonical_key` on ``left_expr`` (aggregation expressions lack a single ``primary_column`` like WHERE filters)."""
    eq_keys: set[str] = set()
    for hp in having:
        key = expr_canonical_key(hp.left_expr)
        if hp.op == "=" and key:
            eq_keys.add(key)

    if not eq_keys:
        return having, False

    kept: list[HavingParam] = []
    changed = False
    for hp in having:
        key = expr_canonical_key(hp.left_expr)
        if key in eq_keys and hp.op in RANGE_OPS:
            debug(f"[intent_repair.dedup_contradictory_having] dropping {hp.op} on agg key {key!r} that contradicts =")
            changed = True
            continue
        kept.append(hp)
    return kept, changed


def dedup_contradictory_where(intent: RuntimeIntent) -> RuntimeIntent:
    """Remove contradictory range filters and HAVING predicates from main query and CTEs."""
    intent = apply_where_to_main_and_ctes(intent, _dedup_contradictory_where_list)
    return apply_having_to_main_and_ctes(intent, _dedup_contradictory_having_list)


def _rendered_expr_matches_param_raw(rendered: str, raw: Any) -> bool:
    if not rendered.strip():
        return False
    r = rendered.strip()
    if isinstance(raw, bool):
        return r.lower() in ("true", "false") and str(raw).lower() == r.lower()
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        try:
            if isinstance(raw, float):
                return abs(float(r) - float(raw)) < 1e-9
            if "." in r or "e" in r.lower():
                return abs(float(r) - float(raw)) < 1e-9
            return int(float(r)) == int(raw)
        except ValueError:
            return r == str(raw)
    if isinstance(raw, str):
        if len(r) >= 2 and r[0] == r[-1] and r[0] in "'\"":
            inner = r[1:-1].replace("''", "'")
            return inner == raw
        return r == raw
    return False


def _qualified_column_token(expr: NormalizedExpr) -> bool:
    col = (expr.column_ref or "").strip()
    return bool(col and "." in col)


def _normalized_expr_is_keyword_leaf(expr: NormalizedExpr) -> bool:
    """Return True when *expr* is a bare SQL keyword leaf such as current_timestamp."""
    return bool(
        expr.keyword
        and not expr.add_groups
        and not expr.sub_groups
        and not expr.add_values
        and not expr.sub_values
        and not expr.column_ref
        and not expr.star
        and not expr.agg_func
        and not expr.scalar_func
        and not expr.raw_sql
        and not expr.string_literal
    )


def _raw_value_to_temporal_keyword(raw: Any) -> str | None:
    """Map a filter/having raw literal to a canonical temporal keyword token, if applicable."""
    if not isinstance(raw, str):
        return None
    lower = raw.strip().lower().replace("()", "")
    if lower in ("current_timestamp", "current_date", "localtimestamp", "localtime", "sysdate"):
        return lower
    return None


def _promote_temporal_keyword_where(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
    out: list[WhereParam] = []
    changed = False
    for fp in filters:
        if fp.right_expr is not None and _normalized_expr_is_keyword_leaf(fp.right_expr):
            out.append(replace(fp, raw_value=None, param_key=""))
            continue
        kw = _raw_value_to_temporal_keyword(fp.raw_value)
        if kw is not None and fp.right_expr is None and fp.op not in ("is null", "is not null"):
            out.append(replace(fp, right_expr=NormalizedExpr(keyword=kw), raw_value=None, param_key=""))
            changed = True
        else:
            out.append(fp)
    return out, changed


def _promote_temporal_keyword_havings(havings: list[HavingParam]) -> tuple[list[HavingParam], bool]:
    out: list[HavingParam] = []
    changed = False
    for hp in havings:
        if hp.right_expr is not None and _normalized_expr_is_keyword_leaf(hp.right_expr):
            out.append(replace(hp, raw_value=None, param_key=""))
            continue
        kw = _raw_value_to_temporal_keyword(hp.raw_value)
        if kw is not None and hp.right_expr is None and hp.op not in ("is null", "is not null"):
            out.append(replace(hp, right_expr=NormalizedExpr(keyword=kw), raw_value=None, param_key=""))
            changed = True
        else:
            out.append(hp)
    return out, changed


def promote_temporal_keyword_rhs(intent: RuntimeIntent) -> RuntimeIntent:
    """Promote temporal keyword string literals on filter/having RHS into keyword ``right_expr`` leaves."""
    intent = apply_where_to_main_and_ctes(intent, _promote_temporal_keyword_where)
    return apply_having_to_main_and_ctes(intent, _promote_temporal_keyword_havings)


def _where_right_expr_redundant_with_value(fp: WhereParam) -> bool:
    if fp.right_expr is None or fp.raw_value is None:
        return False
    if _normalized_expr_is_keyword_leaf(fp.right_expr):
        return False
    if fp.op in ("is null", "is not null"):
        return False
    if isinstance(fp.raw_value, list | dict):
        return False
    if _qualified_column_token(fp.right_expr):
        return False
    try:
        rendered = render_expr_sql(fp.right_expr, None)
    except Exception:
        return False
    return _rendered_expr_matches_param_raw(rendered, fp.raw_value)


def _having_right_expr_redundant_with_value(hp: HavingParam) -> bool:
    if hp.right_expr is None or hp.raw_value is None:
        return False
    if _normalized_expr_is_keyword_leaf(hp.right_expr):
        return False
    if hp.op in ("is null", "is not null"):
        return False
    if isinstance(hp.raw_value, list | dict):
        return False
    if _qualified_column_token(hp.right_expr):
        return False
    try:
        rendered = render_expr_sql(hp.right_expr, None)
    except Exception:
        return False
    return _rendered_expr_matches_param_raw(rendered, hp.raw_value)


def _dedup_value_vs_right_expr_where(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
    out: list[WhereParam] = []
    changed = False
    for fp in filters:
        if _where_right_expr_redundant_with_value(fp):
            out.append(replace(fp, right_expr=None))
            changed = True
        else:
            out.append(fp)
    return out, changed


def _dedup_value_vs_right_expr_havings(having: list[HavingParam]) -> tuple[list[HavingParam], bool]:
    out: list[HavingParam] = []
    changed = False
    for hp in having:
        if _having_right_expr_redundant_with_value(hp):
            out.append(replace(hp, right_expr=None))
            changed = True
        else:
            out.append(hp)
    return out, changed


def dedup_value_vs_right_expr(intent: RuntimeIntent) -> RuntimeIntent:
    """Drop ``right_expr`` when it duplicates the bound ``value`` for parametric predicates."""
    intent = apply_where_to_main_and_ctes(intent, _dedup_value_vs_right_expr_where)
    return apply_having_to_main_and_ctes(intent, _dedup_value_vs_right_expr_havings)


def _is_null_value(raw_value: Any) -> bool:
    """Return True if the raw filter value represents NULL."""
    if raw_value is None:
        return True
    if isinstance(raw_value, str) and raw_value.strip().lower() == "null":
        return True
    return False


def _expr_has_resolvable_column(expr: NormalizedExpr) -> bool:
    """Return True when *expr* references at least one column-shaped token."""
    return bool(extract_columns_from_expr(expr))


def _canonicalize_null_op(op: str) -> str:
    """Return the canonical ``is null`` / ``is not null`` form for *op*, or *op* unchanged."""
    lowered: str = op.strip().lower()
    if lowered in NULL_OP_DOUBLE_NEGATED_ALIASES:
        return "is null"
    if lowered in NULL_OP_NEGATED_ALIASES:
        return "is not null"
    if lowered in NULL_OP_PLAIN_ALIASES:
        return "is null"
    return op


def repair_null_equality_where(intent: RuntimeIntent) -> RuntimeIntent:
    """Rewrite ``=`` / ``!=`` / ``<>`` against null into ``is null`` / ``is not null``."""
    intent = apply_where_to_main_and_ctes(intent, _repair_null_equality_list)
    return apply_having_to_main_and_ctes(intent, _repair_null_equality_having_list)


def _repair_null_equality_list(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
    repaired: list[WhereParam] = []
    changed = False
    for fp in filters:
        if fp.param_key:
            repaired.append(fp)
            continue
        if not _expr_has_resolvable_column(fp.left_expr):
            repaired.append(fp)
            continue
        canonical_op: str = _canonicalize_null_op(fp.op)
        if canonical_op != fp.op:
            repaired.append(replace(fp, op=canonical_op, raw_value=None, value_type="null"))
            changed = True
            continue
        if fp.right_expr is not None:
            repaired.append(fp)
            continue
        if fp.op == "=" and _is_null_value(fp.raw_value):
            repaired.append(replace(fp, op="is null", raw_value=None, value_type="null"))
            changed = True
        elif fp.op in ("!=", "<>") and _is_null_value(fp.raw_value):
            repaired.append(replace(fp, op="is not null", raw_value=None, value_type="null"))
            changed = True
        else:
            repaired.append(fp)
    return repaired, changed


def _repair_null_equality_having_list(having: list[HavingParam]) -> tuple[list[HavingParam], bool]:
    repaired: list[HavingParam] = []
    changed = False
    for hp in having:
        if hp.param_key:
            repaired.append(hp)
            continue
        if not _expr_has_resolvable_column(hp.left_expr):
            repaired.append(hp)
            continue
        canonical_op: str = _canonicalize_null_op(hp.op)
        if canonical_op != hp.op:
            repaired.append(replace(hp, op=canonical_op, raw_value=None, value_type="null"))
            changed = True
            continue
        if hp.right_expr is not None:
            repaired.append(hp)
            continue
        if hp.op == "=" and _is_null_value(hp.raw_value):
            repaired.append(replace(hp, op="is null", raw_value=None, value_type="null"))
            changed = True
        elif hp.op in ("!=", "<>") and _is_null_value(hp.raw_value):
            repaired.append(replace(hp, op="is not null", raw_value=None, value_type="null"))
            changed = True
        else:
            repaired.append(hp)
    return repaired, changed


def _descriptive_column_score(col_name: str, col_meta: ColumnMetadata) -> tuple[int, int, int, int]:
    """Return a sort key (higher is better) for descriptive-column. preference."""
    name_lower = col_name.lower()
    name_score = 0
    if "name" in name_lower or "title" in name_lower:
        name_score = 2
    elif "first_name" in name_lower or "last_name" in name_lower:
        name_score = 3
    dc = col_meta.distinct_count or 0
    uniq_boost = 1 if col_meta.is_unique else 0
    non_null_boost = 1 if not col_meta.is_nullable else 0
    return (non_null_boost, uniq_boost, name_score, dc)


def _best_descriptive_columns(
    table: str, schema_graph: SchemaGraph, exclude: set[str], max_count: int = 2
) -> list[str]:
    """Pick up to *max_count* descriptive columns for *table* (non- PK/FK, high cardinality)."""
    tbl_meta = schema_graph.tables.get(table)
    if not tbl_meta:
        return []
    candidates: list[tuple[str, ColumnMetadata]] = []
    for col_name, col_meta in tbl_meta.columns.items():
        if col_meta.is_primary_key or col_meta.is_foreign_key:
            continue
        if not col_meta.is_selectable:
            continue
        if f"{table}.{col_name}" in exclude:
            continue
        if (col_meta.role or "").strip().lower() == "free_text":
            continue
        vt = (col_meta.value_type or "").lower()
        if vt in DESCRIPTIVE_EXCLUDED_VALUE_TYPES:
            continue
        if vt not in DESCRIPTIVE_ALLOWED_VALUE_TYPES:
            continue
        ratio = col_meta.distinct_ratio
        if ratio is not None and ratio < 0.95:
            continue
        candidates.append((col_name, col_meta))
    if not candidates:
        return []
    candidates.sort(key=lambda p: _descriptive_column_score(p[0], p[1]), reverse=True)
    if max_count >= 2 and len(candidates) >= 2:
        pair = _best_composite_name_pair(tbl_meta, candidates)
        if pair is not None:
            return list(pair)
    return [col_name for col_name, _ in candidates[:max_count]]


def _best_composite_name_pair(
    tbl_meta: TableMetadata, candidates: list[tuple[str, ColumnMetadata]]
) -> tuple[str, str] | None:
    """Return two name-like columns when their composite distinct ratio. beats any single."""
    name_candidates = [(name, meta) for name, meta in candidates if _descriptive_column_score(name, meta)[1] >= 2]
    if len(name_candidates) < 2:
        return None
    best_single_ratio = max((m.distinct_ratio or 0.0) for _, m in candidates)
    ratios = tbl_meta.composite_descriptive_ratios
    for i in range(len(name_candidates)):
        for j in range(i + 1, len(name_candidates)):
            c1 = name_candidates[i][0]
            c2 = name_candidates[j][0]
            composite = ratios.get((c1, c2)) or ratios.get((c2, c1))
            if composite is not None and composite > best_single_ratio:
                return (c1, c2)
    return None


def best_descriptive_columns(table: str, schema_graph: SchemaGraph, exclude: set[str], max_count: int = 2) -> list[str]:
    """
    Pick up to *max_count* descriptive columns for *table* (non- PK/FK, high cardinality).

    Args:

        table: Table name.
        schema_graph: Schema graph.
        exclude: Fully-qualified columns already used elsewhere.
        max_count: Maximum columns to return (2 enables composite name pairs).

    Returns:

        Ordered column names, possibly a name pair when composite ratio wins.
    """
    tbl_meta = schema_graph.tables.get(table)
    if not tbl_meta:
        return []
    candidates: list[tuple[str, ColumnMetadata]] = []
    for col_name, col_meta in tbl_meta.columns.items():
        if col_meta.is_primary_key or col_meta.is_foreign_key:
            continue
        if not col_meta.is_selectable:
            continue
        if f"{table}.{col_name}" in exclude:
            continue
        if (col_meta.role or "").strip().lower() == "free_text":
            continue
        vt = (col_meta.value_type or "").lower()
        if vt in DESCRIPTIVE_EXCLUDED_VALUE_TYPES:
            continue
        if vt not in DESCRIPTIVE_ALLOWED_VALUE_TYPES:
            continue
        ratio = col_meta.distinct_ratio
        if ratio is not None and ratio < 0.95:
            continue
        candidates.append((col_name, col_meta))
    if not candidates:
        return []
    candidates.sort(key=lambda p: _descriptive_column_score(p[0], p[1]), reverse=True)
    if max_count >= 2 and len(candidates) >= 2:
        pair = _best_composite_name_pair(tbl_meta, candidates)
        if pair is not None:
            return list(pair)
    return [col_name for col_name, _ in candidates[:max_count]]


def best_descriptive_column(table: str, schema_graph: SchemaGraph, exclude: set[str]) -> str | None:
    """Return a single best descriptive column (wrapper around. ``max_count=1``)."""
    cols = _best_descriptive_columns(table, schema_graph, exclude, max_count=1)
    return cols[0] if cols else None


def _repair_fk_where(
    filters: list[WhereParam],
    select_cols: list[SelectCol],
    tables: list[str],
    schema_graph: SchemaGraph,
    label: str = "",
) -> tuple[list[WhereParam], list[str], bool]:
    """Scan filters for FK integer + string/enum value pairs (debug. only; no rewrite)."""
    new_filters: list[WhereParam] = []
    tables = list(tables)
    changed = False
    existing_terms = {sc.expr.primary_term for sc in select_cols or []}
    for fp in filters:
        if fp.value_type not in {"string", "enum"} or fp.raw_value is None:
            new_filters.append(fp)
            continue
        col = fp.left_expr.primary_column
        parts = col.split(".", 1) if "." in col else None
        if not parts:
            new_filters.append(fp)
            continue
        col_meta = schema_graph.get_column(parts[0], parts[1])
        if not col_meta or not col_meta.is_foreign_key or col_meta.value_type not in {"integer", "number"}:
            new_filters.append(fp)
            continue
        fk_target = col_meta.fk_target
        if not fk_target:
            new_filters.append(fp)
            continue
        target_table, _ = fk_target
        desc = best_descriptive_column(target_table, schema_graph, existing_terms)
        new_filters.append(fp)
        if desc:
            changed = True
            debug(
                f"[intent_resolve.repair_fk_where_type_mismatch{label}] detected fk filter {col} needing descriptive column"
            )
    return new_filters, tables, changed


def repair_fk_where_type_mismatch(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Detect string/enum values on numeric FK where predicates (debug trace; predicates not rewritten). Where-only by design; FK-type hints apply to ``where`` leaves, not HAVING aggregates."""
    main_filters, _, main_changed = _repair_fk_where(
        where_leaves(intent.where) or [], intent.select_cols or [], list(intent.tables or []), schema_graph
    )
    cte_changed = False
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_filters, _, c = _repair_fk_where(
            where_leaves(cte.where) or [],
            cte.select_cols or [],
            list(cte.tables or []),
            schema_graph,
            label=f" CTE '{cte.cte_name}'",
        )
        if c:
            new_cte_steps.append(replace(cte, where=predicate_group_from_list(cte_filters)))
            cte_changed = True
        else:
            new_cte_steps.append(cte)
    if not main_changed and not cte_changed:
        return intent
    result = intent
    if main_changed:
        result = replace(result, where=predicate_group_from_list(main_filters))
    if cte_changed:
        result = replace(result, cte_steps=new_cte_steps)
    return result


def _expand_fk_select_to_descriptive_tables_sel(
    select_cols: list[SelectCol], tables: list[str], schema_graph: SchemaGraph
) -> tuple[list[SelectCol], list[str], bool]:
    """Expand bare FK integer selects to descriptive columns and extend. ``tables``."""
    tables_out = list(tables or [])
    new_select: list[SelectCol] = []
    changed = False
    existing_terms = {sc.expr.primary_term for sc in select_cols or []}
    for sc in select_cols or []:
        if sc.is_aggregated:
            new_select.append(sc)
            continue
        col = sc.expr.primary_column
        parts = col.split(".", 1) if "." in col else None
        if not parts:
            new_select.append(sc)
            continue
        col_meta = schema_graph.get_column(parts[0], parts[1])
        if not col_meta or not col_meta.is_foreign_key or col_meta.value_type not in {"integer", "number"}:
            new_select.append(sc)
            continue
        tbl_meta = schema_graph.tables.get(parts[0])
        if tbl_meta is not None:
            fk_is_cross_source = any(
                parts[1] in (fk.src_cols or []) and fk.inference_tag == InferenceTag.CROSS_SOURCE
                for fk in tbl_meta.foreign_keys or []
            )
            if fk_is_cross_source:
                new_select.append(sc)
                continue
        fk_target = col_meta.fk_target
        if not fk_target:
            new_select.append(sc)
            continue
        target_table, _ = fk_target
        descs = _best_descriptive_columns(target_table, schema_graph, existing_terms, max_count=2)
        if not descs:
            new_select.append(sc)
            continue
        for desc in descs:
            fq = f"{target_table}.{desc}"
            new_expr = NormalizedExpr.from_column(fq)
            new_select.append(SelectCol(expr=new_expr))
            existing_terms.add(fq)
        if target_table not in tables_out:
            tables_out.append(target_table)
        changed = True
        debug(
            f"[intent_repair._expand_fk_select_to_descriptive_tables_sel] "
            f"rewired select {col} -> {[f'{target_table}.{d}' for d in descs]}"
        )
    return new_select, sorted(tables_out), changed


def expand_fk_select_to_descriptive(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Rewrite bare non-aggregated FK integer selects on the main query. to descriptive columns and add join tables. CTE bodies are not expanded so grouped CTE ``GROUP BY`` stays aligned with CTE ``select_cols``."""
    ns, nt, main_ch = _expand_fk_select_to_descriptive_tables_sel(
        list(intent.select_cols or []), list(intent.tables or []), schema_graph
    )
    if not main_ch:
        return intent
    return replace(intent, select_cols=ns, tables=nt)


def strip_spurious_group_by(intent: RuntimeIntent) -> RuntimeIntent:
    """Clear ``group_by_cols`` when select and having have no. aggregation (main and CTEs)."""
    main_changed = False
    new_grain = intent.grain
    new_gb = intent.group_by_cols or []
    main_distinct = intent.distinct_select_index
    if intent.group_by_cols:
        has_agg = any(sc.is_aggregated for sc in (intent.select_cols or []))
        has_agg = has_agg or any(hp.left_expr.has_aggregation for hp in (having_leaves(intent.having) or []))
        if not has_agg:
            debug(
                f"[intent_resolve.strip_spurious_group_by] group_by_cols present without aggregation — stripping {[g.primary_term for g in intent.group_by_cols]}"
            )
            new_grain = "row_level" if intent.grain == "grouped" else intent.grain
            if len(intent.group_by_cols) == 1 and main_distinct < 0:
                gb_sig = intent.group_by_cols[0].signature_key
                for idx, sc in enumerate(intent.select_cols or []):
                    if not sc.is_aggregated and sc.expr.signature_key == gb_sig:
                        main_distinct = idx
                        break
            new_gb = []
            main_changed = True

    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps or []:
        if not (cte.group_by_cols or []):
            new_cte_steps.append(cte)
            continue
        cte_has_agg = any(sc.is_aggregated for sc in (cte.select_cols or []))
        cte_has_agg = cte_has_agg or any(hp.left_expr.has_aggregation for hp in (having_leaves(cte.having) or []))
        if cte_has_agg:
            new_cte_steps.append(cte)
            continue
        debug(
            f"[intent_resolve.strip_spurious_group_by] CTE '{cte.cte_name}' group_by_cols present without aggregation — stripping {[g.primary_term for g in cte.group_by_cols]}"
        )
        cte_grain = "row_level" if cte.grain == "grouped" else cte.grain
        cte_distinct = cte.distinct_select_index
        if len(cte.group_by_cols) == 1 and cte_distinct < 0:
            gb_sig = cte.group_by_cols[0].signature_key
            for idx, sc in enumerate(cte.select_cols or []):
                if not sc.is_aggregated and sc.expr.signature_key == gb_sig:
                    cte_distinct = idx
                    break
        new_cte_steps.append(replace(cte, group_by_cols=[], grain=cte_grain, distinct_select_index=cte_distinct))
        cte_changed = True

    if not main_changed and not cte_changed:
        return intent
    return replace(
        intent,
        group_by_cols=new_gb,
        grain=new_grain,
        distinct_select_index=main_distinct,
        cte_steps=new_cte_steps if cte_changed else (intent.cte_steps or []),
    )


def _is_impossible_having(hp: HavingParam) -> bool:
    """Return True for impossible COUNT comparisons, not SUM."""
    left_expr = hp.left_expr
    if not left_expr:
        return False
    primary = left_expr.primary_term
    agg_func = ""
    if left_expr.agg_func:
        agg_func = left_expr.agg_func.upper()
    elif left_expr.add_groups and left_expr.add_groups[0].agg_func:
        agg_func = left_expr.add_groups[0].agg_func.upper()
    is_count = bool(IMPOSSIBLE_HAVING_RE.match(primary)) or agg_func == "COUNT"
    if not is_count:
        return False
    op = (hp.op or "").strip().lower()
    val = hp.raw_value
    if val is None:
        return False
    try:
        if isinstance(val, (int, float)):
            numeric_val = float(val)
        elif isinstance(val, str):
            numeric_val = float(val)
        else:
            return False
    except (ValueError, TypeError):
        return False
    if op in ("<", "<=") and numeric_val <= 0:
        return True
    if op == "=" and numeric_val < 0:
        return True
    return False


def strip_impossible_having(intent: RuntimeIntent) -> RuntimeIntent:
    """Drop HAVING clauses that ``_is_impossible_having`` flags (main. and CTEs)."""
    main_having = having_leaves(intent.having) or []
    kept_main = [hp for hp in main_having if not _is_impossible_having(hp)]
    main_changed = len(kept_main) != len(main_having)
    if main_changed:
        removed = len(main_having) - len(kept_main)
        debug(f"[strip_impossible_having] removed {removed} impossible HAVING condition(s)")

    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps or []:
        cte_having = having_leaves(cte.having) or []
        kept_cte = [hp for hp in cte_having if not _is_impossible_having(hp)]
        if len(kept_cte) != len(cte_having):
            cte_changed = True
            new_cte_steps.append(replace(cte, having=predicate_group_from_list(kept_cte)))
        else:
            new_cte_steps.append(cte)

    if not main_changed and not cte_changed:
        return intent
    return replace(
        intent,
        having=predicate_group_from_list(kept_main),
        cte_steps=new_cte_steps if cte_changed else (intent.cte_steps or []),
    )


def _federation_table_source(schema: SchemaGraph, table: str, manifest: FederationManifest | None) -> str:
    meta = schema.tables.get(table)
    if meta is not None and meta.source_id:
        return meta.source_id
    if manifest is not None:
        return str(manifest.table_namespace.get(table, "") or "")
    return ""


def _intent_source_scope(
    schema: SchemaGraph, intent: RuntimeIntent, manifest: FederationManifest | None = None
) -> frozenset[str]:
    scopes: set[str] = set()
    for table in intent.tables or []:
        source_id = _federation_table_source(schema, table, manifest)
        if source_id:
            scopes.add(source_id)
    return frozenset(scopes)


def _scoped_valid_tables(schema_graph: SchemaGraph, source_scope: frozenset[str]) -> dict[str, str]:
    if not source_scope:
        return {t.lower(): t for t in schema_graph.tables}
    scoped: dict[str, str] = {}
    for name, meta in schema_graph.tables.items():
        if meta.source_id and meta.source_id in source_scope:
            scoped[name.lower()] = name
        elif not meta.source_id and meta.member_source_ids and source_scope.intersection(meta.member_source_ids):
            scoped[name.lower()] = name
    return scoped


def _single_source_scope_from_schema(schema: SchemaGraph) -> frozenset[str]:
    sources = {meta.source_id for meta in schema.tables.values() if meta.source_id}
    return frozenset(sources) if len(sources) == 1 else frozenset()


def _repair_source_scope(
    schema: SchemaGraph,
    intent: RuntimeIntent,
    diag: SqlDiagnostic,
    *,
    manifest: FederationManifest | None = None,
) -> frozenset[str]:
    detail_source = (diag.details.get("source_id") or "").strip()
    if detail_source:
        return frozenset({detail_source})
    intent_scope = _intent_source_scope(schema, intent, manifest)
    if intent_scope:
        return intent_scope
    return _single_source_scope_from_schema(schema)


def _sanitize_table_names_list(
    tables: list[str], schema_graph: SchemaGraph, *, valid_tables: Mapping[str, str] | None = None
) -> tuple[list[str], bool]:
    """Return a copy of *tables* with SQL-keyword-prefixed hallucinations corrected when possible."""
    valid_tables = valid_tables or {t.lower(): t for t in schema_graph.tables}
    new_tables: list[str] = []
    changed = False
    for tbl in tables or []:
        if tbl.lower() in valid_tables:
            new_tables.append(tbl)
            continue
        parts = tbl.split()
        candidate = parts[-1].lower() if parts else ""
        if candidate in valid_tables and any(p.lower() in SQL_KEYWORDS for p in parts[:-1]):
            debug(f"[sanitize_table_names] corrected '{tbl}' → '{valid_tables[candidate]}'")
            new_tables.append(valid_tables[candidate])
            changed = True
        else:
            new_tables.append(tbl)
    return new_tables, changed


def sanitize_table_names(
    intent: RuntimeIntent, schema_graph: SchemaGraph, federation_manifest: FederationManifest | None = None
) -> RuntimeIntent:
    """Remove leading SQL keyword tokens from hallucinated multi-token. table names. Applies to the main ``tables`` list and each CTE ``tables`` list."""
    has_federation_tables = federation_manifest is not None or any(
        meta.source_id for meta in schema_graph.tables.values()
    )
    source_scope = (
        _intent_source_scope(schema_graph, intent, federation_manifest) if has_federation_tables else frozenset()
    )
    scoped_tables = _scoped_valid_tables(schema_graph, source_scope) if source_scope else None
    nt, main_ch = _sanitize_table_names_list(list(intent.tables or []), schema_graph, valid_tables=scoped_tables)
    out = replace(intent, tables=nt) if main_ch else intent
    new_cte_steps: list[RuntimeCteStep] = []
    cte_ch = False
    for cte in out.cte_steps or []:
        ctb, c = _sanitize_table_names_list(list(cte.tables or []), schema_graph, valid_tables=scoped_tables)
        if c:
            cte_ch = True
            new_cte_steps.append(replace(cte, tables=ctb))
        else:
            new_cte_steps.append(cte)
    if cte_ch:
        out = replace(out, cte_steps=new_cte_steps)
    if not main_ch and not cte_ch:
        return intent
    return out


def _table_source_id(schema: SchemaGraph, table: str) -> str:
    meta = schema.tables.get(table)
    return str(meta.source_id or "") if meta is not None else ""


def _same_federation_source(schema: SchemaGraph, left_table: str, right_table: str) -> bool:
    """Return True when both tables share a source id, or neither is source-stamped."""
    left_src = _table_source_id(schema, left_table)
    right_src = _table_source_id(schema, right_table)
    if left_src and right_src:
        return left_src == right_src
    return True


def _strip_join_condition_where(filters: list[WhereParam], schema_graph: SchemaGraph) -> list[WhereParam]:
    """Drop ``=`` filters that duplicate a schema FK edge (column-to- column)."""
    fk_pairs: set[tuple[str, str]] = set()
    for tbl in schema_graph.tables.values():
        for fk in tbl.foreign_keys:
            if fk.inference_tag == InferenceTag.CROSS_SOURCE:
                continue
            if len(fk.src_cols) == 1 and len(fk.dst_cols) == 1:
                left = f"{fk.src_table}.{fk.src_cols[0]}"
                right = f"{fk.dst_table}.{fk.dst_cols[0]}"
                fk_pairs.add((left, right))
                fk_pairs.add((right, left))
    result: list[WhereParam] = []
    for fp in filters:
        if fp.right_expr is None or fp.op not in ("=", "in"):
            result.append(fp)
            continue
        left_term = fp.left_expr.primary_term
        right_term = fp.right_expr.primary_term
        if (left_term, right_term) in fk_pairs:
            debug(
                f"[intent_resolve.strip_join_condition_where] dropping FK join filter: {left_term} {fp.op} {right_term}"
            )
            continue
        result.append(fp)
    return result


def strip_join_conditions(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Apply ``_strip_join_condition_where`` to main and each CTE."""
    new_filters = _strip_join_condition_where(where_leaves(intent.where) or [], schema_graph)
    new_cte_steps = [
        replace(
            cte,
            where=rebuild_predicate_group_from_leaves(
                cte.where, _strip_join_condition_where(where_leaves(cte.where) or [], schema_graph)
            ),
        )
        for cte in (intent.cte_steps or [])
    ]
    return replace(
        intent, where=rebuild_predicate_group_from_leaves(intent.where, new_filters), cte_steps=new_cte_steps
    )


def _where_param_is_bare_column_equality(fp: WhereParam) -> bool:
    """Return True when *fp* is ``table.col = table.col`` with no literal or param value."""
    if fp.op != "=" or fp.right_expr is None:
        return False
    if fp.param_key or fp.raw_value is not None:
        return False
    return is_plain_column_expr(fp.left_expr) and is_plain_column_expr(fp.right_expr)


def _drop_redundant_join_where_list(
    filters: list[WhereParam],
    *,
    droppable_pairs: set[frozenset[str]],
) -> tuple[list[WhereParam], bool]:
    """Remove ``where`` leaves duplicated by a resolved INNER join edge."""
    if not filters or not droppable_pairs:
        return filters, False
    kept: list[WhereParam] = []
    changed = False
    for fp in filters:
        if not _where_param_is_bare_column_equality(fp):
            kept.append(fp)
            continue
        left_term = fp.left_expr.primary_term
        right_term = fp.right_expr.primary_term if fp.right_expr is not None else ""
        if frozenset({left_term, right_term}) in droppable_pairs:
            changed = True
            notify(
                (
                    f"Dropped redundant join filter {left_term} = {right_term}; "
                    "the resolved join path already renders this equality in ON."
                ),
                stage="join",
                code=DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED,
                level="info",
                details=(
                    ("predicate", f"{left_term} = {right_term}"),
                    ("subsumed_by", "resolved_join_path"),
                ),
            )
            continue
        kept.append(fp)
    return kept, changed


def drop_redundant_resolved_join_where_predicates(
    intent: RuntimeIntent,
    schema: SchemaGraph | None,
    *,
    join_sigs_ordered: list[list[str]],
    edge_kinds_ordered: list[list[str]],
) -> tuple[RuntimeIntent, bool]:
    """Drop ``where`` leaves that repeat INNER join equalities from resolved path signatures."""
    emissions = cte_emission_map(intent.cte_steps)
    probes = probe_cte_names(intent.cte_steps)
    changed = False
    cte_steps = list(intent.cte_steps or [])
    for idx, cte in enumerate(cte_steps):
        sig = join_sigs_ordered[idx] if idx < len(join_sigs_ordered) else []
        kinds = edge_kinds_ordered[idx] if idx < len(edge_kinds_ordered) else []
        anchor = (cte.tables or [""])[0] if cte.tables else ""
        pairs = inner_equality_pairs_from_resolved_join_path(
            list(sig or []),
            list(kinds or []),
            anchor,
            schema,
            preserve_tables=list(cte.preserve_tables or []),
            cte_emissions=emissions,
            probe_cte_names=probes,
        )
        new_filters, scope_changed = _drop_redundant_join_where_list(
            where_leaves(cte.where) or [],
            droppable_pairs=pairs,
        )
        if scope_changed:
            changed = True
            cte_steps[idx] = replace(
                cte,
                where=rebuild_predicate_group_from_leaves(cte.where, new_filters),
            )
    main_sig = join_sigs_ordered[-1] if join_sigs_ordered else []
    main_kinds = edge_kinds_ordered[-1] if edge_kinds_ordered else []
    main_anchor = (intent.tables or [""])[0] if intent.tables else ""
    main_pairs = inner_equality_pairs_from_resolved_join_path(
        list(main_sig or []),
        list(main_kinds or []),
        main_anchor,
        schema,
        preserve_tables=list(intent.preserve_tables or []),
        cte_emissions=emissions,
        probe_cte_names=probes,
    )
    main_filters, main_changed = _drop_redundant_join_where_list(
        where_leaves(intent.where) or [],
        droppable_pairs=main_pairs,
    )
    if main_changed:
        changed = True
    if not changed:
        return intent, False
    return (
        replace(
            intent,
            where=rebuild_predicate_group_from_leaves(intent.where, main_filters),
            cte_steps=cte_steps,
        ),
        True,
    )


def _tables_in_join_signature(signature: list[str]) -> set[str]:
    """Return physical table names referenced by a stored join-path signature."""
    tables: set[str] = set()
    for seg in signature or []:
        seg = str(seg).strip()
        if "->" not in seg:
            continue
        left_part, right_part = seg.split("->", 1)
        for part in (left_part, right_part):
            part = part.strip()
            if "." in part:
                tables.add(part.split(".", 1)[0].strip())
    return tables


def _join_path_tables_locked(intent: RuntimeIntent) -> set[str]:
    """Return tables that must not be eliminated because a join path is already pinned."""
    signature = list(intent.chosen_join_path_signature or [])
    if not signature:
        return set()
    locked = _tables_in_join_signature(signature)
    for cte in intent.cte_steps or []:
        locked.update(_tables_in_join_signature(list(cte.chosen_join_path_signature or [])))
    return locked


def _federation_cross_source_endpoint_tables(manifest: FederationManifest | None) -> set[str]:
    """Return table names that participate in declared cross-source joins."""
    if manifest is None:
        return set()
    endpoints: set[str] = set()
    for join in manifest.cross_source_joins or ():
        for endpoint in (join.left, join.right):
            endpoint = str(endpoint or "").strip()
            if "." in endpoint:
                endpoints.add(endpoint.split(".", 1)[0].strip())
    return endpoints


def _cross_source_endpoint_tables(schema: SchemaGraph) -> set[str]:
    """Return table names stamped with a cross-source foreign-key edge."""
    endpoints: set[str] = set()
    for tbl in schema.tables.values():
        for fk in tbl.foreign_keys or []:
            if fk.inference_tag == InferenceTag.CROSS_SOURCE:
                endpoints.add(fk.src_table)
                endpoints.add(fk.dst_table)
    return endpoints


def _rewrite_term_map_for_fk(fk: FKEdge) -> dict[str, str]:
    """Build ``far.col`` to ``near.col`` replacements for one catalog foreign-key edge."""
    return {f"{fk.dst_table}.{dc}": f"{fk.src_table}.{sc}" for sc, dc in zip(fk.src_cols, fk.dst_cols, strict=True)}


def _column_ref_replacer(rewrites: dict[str, str]) -> Callable[[str], str]:
    """Return a column-reference replacer that applies *rewrites* case- insensitively."""

    def repl(term: str) -> str:
        direct = rewrites.get(term)
        if direct is not None:
            return direct
        lowered = term.lower()
        for old, new in rewrites.items():
            if old.lower() == lowered:
                return new
        return term

    return repl


def _rewrite_normalized_expr(expr: NormalizedExpr, rewrites: dict[str, str]) -> NormalizedExpr:
    """Rewrite column references inside one normalized expression."""
    if not rewrites:
        return expr
    return replace_refs_in_expr(expr, _column_ref_replacer(rewrites))


def _rewrite_where_params(params: list[WhereParam], rewrites: dict[str, str]) -> list[WhereParam]:
    """Rewrite column references in a flat ``where`` leaf list."""
    out: list[WhereParam] = []
    for fp in params:
        le = _rewrite_normalized_expr(fp.left_expr, rewrites)
        rexp = _rewrite_normalized_expr(fp.right_expr, rewrites) if fp.right_expr else None
        out.append(replace(fp, left_expr=le, right_expr=rexp))
    return out


def _rewrite_having_params(params: list[HavingParam], rewrites: dict[str, str]) -> list[HavingParam]:
    """Rewrite column references in a flat ``having`` leaf list."""
    out: list[HavingParam] = []
    for hp in params:
        le = _rewrite_normalized_expr(hp.left_expr, rewrites)
        rexp = _rewrite_normalized_expr(hp.right_expr, rewrites) if hp.right_expr else None
        out.append(replace(hp, left_expr=le, right_expr=rexp))
    return out


def _rewrite_window_spec(ws: WindowSpec, rewrites: dict[str, str]) -> WindowSpec:
    """Rewrite partition, order, and argument expressions in a window spec."""
    pb = [_rewrite_normalized_expr(e, rewrites) for e in ws.partition_by]
    ob = [replace(o, expr=_rewrite_normalized_expr(o.expr, rewrites)) for o in ws.order_by]
    arg = _rewrite_normalized_expr(ws.argument, rewrites) if ws.argument else None
    return replace(ws, partition_by=pb, order_by=ob, argument=arg)


def _rewrite_case_when(cw: CaseWhenExpr, rewrites: dict[str, str]) -> CaseWhenExpr:
    """Rewrite CASE branch conditions and results."""
    branches: list[CaseWhenBranch] = []
    for br in cw.branches:
        cond = _rewrite_where_params([br.condition], rewrites)[0]
        res = _rewrite_normalized_expr(br.result, rewrites)
        branches.append(CaseWhenBranch(condition=cond, result=res))
    er = _rewrite_normalized_expr(cw.else_result, rewrites) if cw.else_result else None
    return replace(cw, branches=branches, else_result=er)


def _scope_column_refs(
    *,
    select_cols: list[SelectCol] | None,
    order_by_cols: list[OrderByCol] | None,
    group_by_cols: list[NormalizedExpr] | None,
    where: PredicateGroup | None,
    having: PredicateGroup | None,
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
    distinct_on: list[NormalizedExpr] | None,
) -> set[str]:
    """Collect qualified column references from one intent scope."""
    refs: set[str] = set()
    for sc in select_cols or []:
        refs.update(cols_from_select_col(sc, window_registry, case_registry))
    for obc in order_by_cols or []:
        refs.update(extract_columns_from_expr(obc.expr))
    for g in group_by_cols or []:
        refs.update(extract_columns_from_expr(g))
    for expr in distinct_on or []:
        refs.update(extract_columns_from_expr(expr))
    for fp in where_leaves(where) or []:
        refs.update(extract_columns_from_expr(fp.left_expr))
        if fp.right_expr:
            refs.update(extract_columns_from_expr(fp.right_expr))
    for hp in having_leaves(having) or []:
        refs.update(extract_columns_from_expr(hp.left_expr))
        if hp.right_expr:
            refs.update(extract_columns_from_expr(hp.right_expr))
    for win_step in window_registry or []:
        for col in cols_from_named_registries([win_step], None):
            refs.add(col)
    for case_step in case_registry or []:
        for col in cols_from_named_registries(None, [case_step]):
            refs.add(col)
    return refs


def _refs_for_table(column_refs: set[str], table: str) -> set[str]:
    """Return qualified references belonging to *table*."""
    prefix = f"{table.lower()}."
    return {ref for ref in column_refs if ref.lower().startswith(prefix)}


def _predicate_touches_column(param: WhereParam | HavingParam, column_ref: str) -> bool:
    """Return whether a filter leaf references *column_ref*."""
    cols: list[str] = []
    cols.extend(extract_columns_from_expr(param.left_expr))
    if param.right_expr:
        cols.extend(extract_columns_from_expr(param.right_expr))
    target = column_ref.lower()
    return any(c.lower() == target for c in cols)


def _guard_catalog_foreign_key(fk: FKEdge) -> str | None:
    """Refuse elimination when the edge is not a catalog foreign key."""
    if fk.inference_tag is not None:
        return "foreign_key_not_catalog"
    return None


def _guard_complete_primary_key_target(fk: FKEdge, schema: SchemaGraph) -> str | None:
    """Refuse when the foreign key does not target the full primary key of the far table."""
    far_meta = schema.tables.get(fk.dst_table)
    if far_meta is None:
        return "far_table_missing"
    pk_cols = list(far_meta.primary_key or [])
    if not pk_cols:
        return "missing_primary_key"
    if list(fk.dst_cols) != pk_cols:
        return "incomplete_primary_key"
    if len(fk.src_cols) != len(fk.dst_cols):
        return "fk_column_count_mismatch"
    return None


def _guard_null_safe_foreign_key(
    fk: FKEdge,
    schema: SchemaGraph,
    where_params: list[WhereParam],
    having_params: list[HavingParam],
) -> str | None:
    """Refuse when a nullable foreign-key column has a null-sensitive predicate."""
    near_meta = schema.tables.get(fk.src_table)
    if near_meta is None:
        return "near_table_missing"
    for src_col in fk.src_cols:
        col_meta = near_meta.columns.get(src_col)
        if col_meta is None or not col_meta.is_nullable:
            continue
        fk_ref = f"{fk.src_table}.{src_col}"
        for fp in where_params:
            if _predicate_touches_column(fp, fk_ref) and fp.op.lower() in NULL_SENSITIVE_ELIMINATION_OPS:
                return "null_sensitive_predicate"
        for hp in having_params:
            if _predicate_touches_column(hp, fk_ref) and hp.op.lower() in NULL_SENSITIVE_ELIMINATION_OPS:
                return "null_sensitive_predicate"
    return None


def _guard_only_primary_key_references(far_table: str, pk_cols: list[str], column_refs: set[str]) -> str | None:
    """Refuse when any reference to the far table is not exactly its primary key."""
    allowed = {f"{far_table}.{col}".lower() for col in pk_cols}
    for ref in _refs_for_table(column_refs, far_table):
        if ref.lower() not in allowed:
            return "non_primary_key_reference"
    referenced_pk = {ref.lower() for ref in _refs_for_table(column_refs, far_table)}
    if len(pk_cols) > 1 and referenced_pk != allowed:
        return "composite_primary_key_incomplete"
    return None


def _guard_special_table(
    far_table: str,
    *,
    preserve_tables: list[str],
    cte_names: set[str],
    locked_tables: set[str],
    cross_source_tables: set[str],
) -> str | None:
    """Refuse when the far table is preserved, virtual, pinned, or federated."""
    if far_table in preserve_tables:
        return "preserve_tables"
    if far_table in cte_names:
        return "cte_name"
    if far_table in locked_tables:
        return "pinned_join_path"
    if far_table in cross_source_tables:
        return "cross_source_endpoint"
    return None


def _guard_remainder_connected(
    tables: list[str],
    far_table: str,
    schema: SchemaGraph,
) -> str | None:
    """Refuse when removing *far_table* disconnects the remaining scope tables."""
    remaining = [t for t in tables if t != far_table]
    if len(remaining) <= 1:
        return None
    issues = validate_join_path_reachability_for_tables(remaining, schema, "key_join_elimination")
    if any(i.severity == "error" for i in issues):
        return "disconnected_remainder"
    return None


def _elimination_probe_block(far_table: str, cte_steps: list[RuntimeCteStep] | None) -> str | None:
    """Refuse when the far table is a probe CTE."""
    for cte in cte_steps or []:
        if cte.cte_name != far_table:
            continue
        if getattr(cte, "emission", "join_table") in ("anti_join", "semi_join"):
            return "probe_emission"
    return None


def _rewrite_scope_fields(
    *,
    select_cols: list[SelectCol] | None,
    order_by_cols: list[OrderByCol] | None,
    group_by_cols: list[NormalizedExpr] | None,
    where: PredicateGroup | None,
    having: PredicateGroup | None,
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
    distinct_on: list[NormalizedExpr] | None,
    rewrites: dict[str, str],
) -> dict[str, Any]:
    """Return rewritten scope fields after applying *rewrites*."""
    new_select = [replace(sc, expr=_rewrite_normalized_expr(sc.expr, rewrites)) for sc in (select_cols or [])]
    new_order = [replace(obc, expr=_rewrite_normalized_expr(obc.expr, rewrites)) for obc in (order_by_cols or [])]
    new_group = [_rewrite_normalized_expr(g, rewrites) for g in (group_by_cols or [])]
    new_where = rebuild_predicate_group_from_leaves(where, _rewrite_where_params(where_leaves(where) or [], rewrites))
    new_having = rebuild_predicate_group_from_leaves(
        having, _rewrite_having_params(having_leaves(having) or [], rewrites)
    )
    new_windows = [
        replace(step, window_spec=_rewrite_window_spec(step.window_spec, rewrites)) for step in (window_registry or [])
    ]
    new_cases = [
        replace(step, case_when=_rewrite_case_when(step.case_when, rewrites)) for step in (case_registry or [])
    ]
    new_distinct = [_rewrite_normalized_expr(expr, rewrites) for expr in (distinct_on or [])]
    return {
        "select_cols": new_select,
        "order_by_cols": new_order,
        "group_by_cols": new_group,
        "where": new_where,
        "having": new_having,
        "window_registry": new_windows,
        "case_registry": new_cases,
        "distinct_on": new_distinct,
    }


def _try_eliminate_far_from_scope(
    *,
    tables: list[str],
    anchor: str | None,
    preserve_tables: list[str],
    select_cols: list[SelectCol] | None,
    order_by_cols: list[OrderByCol] | None,
    group_by_cols: list[NormalizedExpr] | None,
    where: PredicateGroup | None,
    having: PredicateGroup | None,
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
    distinct_on: list[NormalizedExpr] | None,
    schema: SchemaGraph,
    cte_names: set[str],
    locked_tables: set[str],
    cross_source_tables: set[str],
    cte_steps: list[RuntimeCteStep] | None,
) -> tuple[dict[str, Any] | None, FKEdge | None]:
    """Attempt one redundant key-join elimination within a single scope."""
    where_params = where_leaves(where) or []
    having_params = having_leaves(having) or []
    column_refs = _scope_column_refs(
        select_cols=select_cols,
        order_by_cols=order_by_cols,
        group_by_cols=group_by_cols,
        where=where,
        having=having,
        window_registry=window_registry,
        case_registry=case_registry,
        distinct_on=distinct_on,
    )
    table_set = set(tables or [])
    for near_table in sorted(table_set):
        near_meta = schema.tables.get(near_table)
        if near_meta is None:
            continue
        for fk in near_meta.foreign_keys or []:
            far_table = fk.dst_table
            if far_table not in table_set or fk.src_table != near_table:
                continue
            far_table_name = far_table
            guards = (
                _guard_catalog_foreign_key,
                lambda edge: _guard_complete_primary_key_target(edge, schema),
                lambda edge: _guard_null_safe_foreign_key(edge, schema, where_params, having_params),
                lambda _edge, ft=far_table_name: _guard_special_table(
                    ft,
                    preserve_tables=preserve_tables,
                    cte_names=cte_names,
                    locked_tables=locked_tables,
                    cross_source_tables=cross_source_tables,
                ),
                lambda _edge, ft=far_table_name: _elimination_probe_block(ft, cte_steps),
                lambda _edge, ft=far_table_name: _guard_only_primary_key_references(
                    ft,
                    list(
                        (
                            schema.tables.get(ft) or TableMetadata(name=ft, columns={}, primary_key=[], foreign_keys=[])
                        ).primary_key
                        or []
                    ),
                    column_refs,
                ),
                lambda _edge, ft=far_table_name: _guard_remainder_connected(list(tables), ft, schema),
            )
            blocked = False
            for guard in guards:
                reason = guard(fk)
                if reason is not None:
                    blocked = True
                    break
            if blocked:
                continue
            rewrites = _rewrite_term_map_for_fk(fk)
            rewritten = _rewrite_scope_fields(
                select_cols=select_cols,
                order_by_cols=order_by_cols,
                group_by_cols=group_by_cols,
                where=where,
                having=having,
                window_registry=window_registry,
                case_registry=case_registry,
                distinct_on=distinct_on,
                rewrites=rewrites,
            )
            rewritten["tables"] = sorted(t for t in tables if t != far_table)
            notify(
                (f"Eliminated redundant key join via {near_table}.{fk.src_cols[0]} -> {far_table}.{fk.dst_cols[0]}"),
                stage="intent",
                code=DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_ELIMINATED,
                level="info",
                details=(
                    ("near_table", near_table),
                    ("far_table", far_table),
                    ("near_column", f"{near_table}.{fk.src_cols[0]}"),
                    ("far_column", f"{far_table}.{fk.dst_cols[0]}"),
                ),
            )
            return rewritten, fk
    return None, None


def _eliminate_redundant_key_joins_once(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    *,
    locked_tables: set[str],
    cross_source_tables: set[str],
) -> tuple[RuntimeIntent, bool]:
    """Run one fixpoint iteration of redundant key-join elimination across all scopes."""
    cte_names = {cte.cte_name for cte in (intent.cte_steps or []) if cte.cte_name}
    changed = False
    new_cte_steps: list[RuntimeCteStep] = list(intent.cte_steps or [])
    for idx, cte in enumerate(new_cte_steps):
        rewritten, _fk = _try_eliminate_far_from_scope(
            tables=list(cte.tables or []),
            anchor=cte.tables[0] if cte.tables else "",
            preserve_tables=list(cte.preserve_tables or []),
            select_cols=cte.select_cols,
            order_by_cols=cte.order_by_cols,
            group_by_cols=cte.group_by_cols,
            where=cte.where,
            having=cte.having,
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            distinct_on=cte.distinct_on,
            schema=schema,
            cte_names=cte_names,
            locked_tables=locked_tables,
            cross_source_tables=cross_source_tables,
            cte_steps=intent.cte_steps,
        )
        if rewritten is None:
            continue
        changed = True
        new_cte_steps[idx] = replace(cte, **rewritten)
    main_rewritten, _fk = _try_eliminate_far_from_scope(
        tables=list(intent.tables or []),
        anchor=intent.tables[0] if intent.tables else "",
        preserve_tables=list(intent.preserve_tables or []),
        select_cols=intent.select_cols,
        order_by_cols=intent.order_by_cols,
        group_by_cols=intent.group_by_cols,
        where=intent.where,
        having=intent.having,
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
        distinct_on=intent.distinct_on,
        schema=schema,
        cte_names=cte_names,
        locked_tables=locked_tables,
        cross_source_tables=cross_source_tables,
        cte_steps=intent.cte_steps,
    )
    if main_rewritten is not None:
        changed = True
        intent = replace(intent, cte_steps=new_cte_steps, **main_rewritten)
    elif changed:
        intent = replace(intent, cte_steps=new_cte_steps)
    return intent, changed


def eliminate_redundant_key_joins(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    *,
    federation_manifest: FederationManifest | None = None,
) -> RuntimeIntent:
    """Remove far-side tables that are referenced only through their primary key."""
    if not PolicyConfig.ELIMINATE_REDUNDANT_KEY_JOINS:
        return intent
    locked_tables = _join_path_tables_locked(intent)
    cross_source_tables = _federation_cross_source_endpoint_tables(federation_manifest) | _cross_source_endpoint_tables(
        schema
    )
    current = intent
    for _ in range(ELIMINATE_REDUNDANT_KEY_JOINS_MAX_ITERATIONS):
        current, changed = _eliminate_redundant_key_joins_once(
            current,
            schema,
            locked_tables=locked_tables,
            cross_source_tables=cross_source_tables,
        )
        if not changed:
            return current
    notify(
        "Redundant key-join elimination reached its iteration cap.",
        stage="intent",
        code=DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_CAP_REACHED,
        level="warning",
    )
    return current


def _is_pk_column(col_ref: str, schema_graph: SchemaGraph) -> bool:
    """Return True when *col_ref* is a primary key column."""
    if "." not in col_ref:
        return False
    tbl, col = col_ref.split(".", 1)
    tbl_meta = schema_graph.tables.get(tbl)
    if not tbl_meta:
        return False
    col_meta = tbl_meta.columns.get(col)
    return col_meta.is_primary_key if col_meta else False


def _strip_distinct_prefix(term: str) -> str:
    """Strip a leading ``DISTINCT `` token from *term*."""
    if term.upper().startswith("DISTINCT "):
        return term[9:].strip()
    return term


def _count_wraps_multi_arg_concat(expr: NormalizedExpr) -> bool:
    """True when COUNT wraps a CONCAT MulGroup with more than one argument (Shape A)."""
    g0 = expr.add_groups[0] if expr.add_groups else None
    if not g0 or (g0.agg_func or "").lower() != "count" or not g0.multiply:
        return False
    child = g0.multiply[0]
    if not isinstance(child, NormalizedExpr) or not child.add_groups:
        return False
    inner = child.add_groups[0]
    return (inner.scalar_func or "").lower() == "concat" and len(inner.multiply) > 1


def _normalize_sc_pk_distinct(sc: SelectCol, schema_graph: SchemaGraph) -> SelectCol:
    """For COUNT on a PK, clear the redundant ``DISTINCT`` flag from the MulGroup."""
    e = sc.expr
    g0 = e.add_groups[0] if e.add_groups else None
    agg = (e.agg_func or (g0.agg_func if g0 else "") or "").lower()
    if agg != "count":
        return sc
    if not g0 or not g0.distinct:
        return sc
    if _count_wraps_multi_arg_concat(e):
        return sc
    col = e.primary_term
    if not _is_pk_column(col, schema_graph):
        return sc
    new_groups = list(e.add_groups)
    new_groups[0] = replace(g0, distinct=False)
    new_expr = replace(e, add_groups=new_groups)
    debug(f"[normalize_pk_distinct] cleared DISTINCT flag for COUNT on PK column: {col}")
    return replace(sc, expr=new_expr)


def normalize_pk_distinct(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Remove ``DISTINCT`` from ``COUNT`` on PK columns (main and CTE. selects)."""
    new_select = [_normalize_sc_pk_distinct(sc, schema_graph) for sc in (intent.select_cols or [])]
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_select = [_normalize_sc_pk_distinct(sc, schema_graph) for sc in (cte.select_cols or [])]
        new_cte_steps.append(replace(cte, select_cols=cte_select))
    return replace(intent, select_cols=new_select, cte_steps=new_cte_steps)


def _lift_distinct_from_select_col(sc: SelectCol) -> tuple[SelectCol, bool]:
    """Strip stray DISTINCT scalar wrappers from non-aggregate groups. With the parser-native schema, DISTINCT lives on ``MulGroup.distinct``; this helper only catches legacy-shaped ``DISTINCT(...)`` scalar wrappers that may slip in. Returns ``(sc, changed_flag)``."""
    e = sc.expr
    if not e.add_groups:
        return sc, False
    new_groups = list(e.add_groups)
    changed = False
    for i, g in enumerate(new_groups):
        if g.agg_func:
            continue
        if (g.scalar_func or "").lower() == "distinct" and not g.inner_scalar_func:
            new_groups[i] = replace(g, scalar_func="", scalar_func_args=[])
            changed = True
            debug("[lift_distinct_modifier_in_multiply] stripped DISTINCT(...) scalar wrapper")
    if not changed:
        return sc, False
    new_expr = replace(e, add_groups=new_groups)
    return replace(sc, expr=new_expr), True


def _expr_primary_column_value_type(
    expr: NormalizedExpr, schema: SchemaGraph, cte_steps: Sequence[RuntimeCteStep] | None = None
) -> str | None:
    col = expr.primary_column or ""
    if "." not in col:
        return None
    table, col_name = col.rsplit(".", 1)
    cte_vt = _cte_output_value_type(table, col_name, cte_steps)
    if cte_vt is not None:
        return cte_vt
    tmeta = schema.tables.get(table) if table in schema.tables else None
    if not tmeta:
        return None
    cmeta: ColumnMetadata | None = tmeta.columns.get(col_name) or tmeta.columns.get(col_name.lower())
    if not cmeta:
        return None
    return (cmeta.value_type or "").lower() or None


def _cte_output_value_type(qualifier: str, output_alias: str, cte_steps: Sequence[RuntimeCteStep] | None) -> str | None:
    """Return the value_type of a ``cte_name.output_alias`` reference, or None when not found."""
    if not cte_steps or not qualifier:
        return None
    qualifier_lc = qualifier.lower()
    alias_lc = output_alias.lower()
    for cte in cte_steps:
        if (cte.cte_name or "").lower() != qualifier_lc:
            continue
        ocm = cte.output_column_metadata or {}
        meta = ocm.get(output_alias) or ocm.get(alias_lc)
        if meta and (meta.value_type or "").strip():
            return meta.value_type.lower()
        for sc in cte.select_cols or []:
            sc_alias = (sc.output_alias or "").lower()
            if sc_alias and sc_alias != alias_lc:
                continue
            inferred = _infer_select_col_value_type(sc)
            if inferred:
                return inferred
        return None
    return None


def _infer_select_col_value_type(sc: SelectCol) -> str | None:
    """Infer a coarse value_type for a CTE select column from its expression shape."""
    expr = getattr(sc, "expr", None)
    if expr is None:
        return None
    agg = (expr.agg_func or "").lower()
    if agg in NUMERIC_RESULT_AGGS:
        return "number"
    scalar = (expr.scalar_func or "").lower()
    if scalar in NUMERIC_RESULT_SCALARS:
        return "number"
    inner_scalar = (expr.inner_scalar_func or "").lower()
    if inner_scalar in NUMERIC_RESULT_SCALARS:
        return "number"
    if scalar in DATE_RESULT_SCALARS or inner_scalar in DATE_RESULT_SCALARS:
        return "date"
    if getattr(expr, "is_numeric", False):
        return "number"
    return None


def _is_expr_date(
    expr: NormalizedExpr | None, schema: SchemaGraph, cte_steps: Sequence[RuntimeCteStep] | None = None
) -> bool:
    if expr is None:
        return False
    if (expr.scalar_func or "").lower() in DATE_RESULT_SCALARS:
        return True
    if (expr.inner_scalar_func or "").lower() in DATE_RESULT_SCALARS:
        return True
    if (expr.agg_func or "").lower() in NON_NUMERIC_AGGS_FOR_DATES:
        vt = _expr_primary_column_value_type(expr, schema, cte_steps)
        if vt and vt in DATE_COLUMN_VALUE_TYPES:
            return True
    vt = _expr_primary_column_value_type(expr, schema, cte_steps)
    if vt and vt in DATE_COLUMN_VALUE_TYPES:
        return True
    return False


def _is_expr_string(
    expr: NormalizedExpr | None, schema: SchemaGraph, cte_steps: Sequence[RuntimeCteStep] | None = None
) -> bool:
    if expr is None:
        return False
    vt = _expr_primary_column_value_type(expr, schema, cte_steps)
    if vt and vt in STRING_COLUMN_VALUE_TYPES:
        return True
    return False


def _align_pred_value_type(
    pred: WhereParam | HavingParam, schema: SchemaGraph, cte_steps: Sequence[RuntimeCteStep] | None = None
) -> WhereParam | HavingParam:
    left = pred.left_expr
    op = (pred.op or "").lower()
    current = pred.value_type or ""

    if op in ("is null", "is not null"):
        return pred

    right = pred.right_expr
    if right is not None and _normalized_expr_is_keyword_leaf(right):
        if current not in ("date", "date_window", "date_diff", "timestamp"):
            debug(
                f"[align_where_value_type_to_exprs] overriding value_type {current!r} -> 'date' on temporal keyword RHS"
            )
            return replace(pred, value_type="date")
        return pred

    if current in ("date_window", "date_diff"):
        return pred

    if op in STRING_OPS:
        if current != "string":
            debug(f"[align_where_value_type_to_exprs] overriding value_type {current!r} -> 'string' for op {op!r}")
            return replace(pred, value_type="string")
        return pred

    right = pred.right_expr
    left_numeric = bool(getattr(left, "is_numeric", False))
    right_numeric = right is None or bool(getattr(right, "is_numeric", False))
    if (
        left_numeric
        and right_numeric
        and not _is_expr_string(left, schema, cte_steps)
        and not _is_expr_string(right, schema, cte_steps)
    ):
        target = "number"
        if current != target:
            debug(
                f"[align_where_value_type_to_exprs] overriding value_type "
                f"{current!r} -> {target!r} on numeric predicate"
            )
            return replace(pred, value_type=target)
        return pred

    left_date = _is_expr_date(left, schema, cte_steps)
    right_date = right is None or _is_expr_date(right, schema, cte_steps)
    if left_date and right_date and right is not None:
        if current != "date":
            debug(f"[align_where_value_type_to_exprs] overriding value_type {current!r} -> 'date' on date predicate")
            return replace(pred, value_type="date")
        return pred

    if _is_expr_string(left, schema, cte_steps) or _is_expr_string(right, schema, cte_steps):
        if current not in ("string") and current not in ("date_window", "date_diff"):
            debug(
                f"[align_where_value_type_to_exprs] overriding value_type "
                f"{current!r} -> 'string' on string column predicate"
            )
            return replace(pred, value_type="string")
    return pred


def _collect_number_typed_predicate_param_keys(pred: WhereParam | HavingParam) -> set[str]:
    if (pred.value_type or "").lower() != "number":
        return set()
    keys: set[str] = set()
    pk = (pred.param_key or "").strip()
    if pk:
        keys.add(pk)
    if isinstance(pred, WhereParam):
        pkh = (pred.param_key_hi or "").strip()
        if pkh:
            keys.add(pkh)
    return keys


def _maybe_coerce_bool_literal_for_where(fp: WhereParam) -> WhereParam:
    if (fp.value_type or "").lower() != "number":
        return fp
    if isinstance(fp.raw_value, bool):
        return replace(fp, raw_value=1 if fp.raw_value else 0)
    return fp


def _maybe_coerce_bool_literal_for_having(hp: HavingParam) -> HavingParam:
    if (hp.value_type or "").lower() != "number":
        return hp
    if isinstance(hp.raw_value, bool):
        return replace(hp, raw_value=1 if hp.raw_value else 0)
    return hp


def _maybe_coerce_bool_literal_for_numeric_pred(pred: WhereParam | HavingParam) -> WhereParam | HavingParam:
    if isinstance(pred, WhereParam):
        return _maybe_coerce_bool_literal_for_where(pred)
    return _maybe_coerce_bool_literal_for_having(pred)


def _collect_numeric_predicate_param_keys_from_case_registry(registry: Sequence[CaseRegistryStep] | None) -> set[str]:
    keys: set[str] = set()
    if not registry:
        return keys
    for step in registry:
        cw = step.case_when
        if not cw or not cw.branches:
            continue
        for br in cw.branches:
            cond = br.condition
            keys.update(_collect_number_typed_predicate_param_keys(cond))
    return keys


def _coerce_boolean_bindings_for_number_typed_where(intent: RuntimeIntent) -> RuntimeIntent:
    keys: set[str] = set()
    for fp in where_leaves(intent.where) or []:
        keys.update(_collect_number_typed_predicate_param_keys(fp))
    for hp in having_leaves(intent.having) or []:
        keys.update(_collect_number_typed_predicate_param_keys(hp))
    for cte in intent.cte_steps or []:
        for fp in where_leaves(cte.where) or []:
            keys.update(_collect_number_typed_predicate_param_keys(fp))
        for hp in having_leaves(cte.having) or []:
            keys.update(_collect_number_typed_predicate_param_keys(hp))
        keys.update(_collect_numeric_predicate_param_keys_from_case_registry(cte.case_registry))
    keys.update(_collect_numeric_predicate_param_keys_from_case_registry(intent.case_registry))

    def _patch_param_map(pv: dict[str, Any] | None) -> dict[str, Any]:
        if not keys:
            return dict(pv or {})
        base = dict(pv or {})
        for k in keys:
            if k in base and isinstance(base[k], bool):
                base[k] = 1 if base[k] else 0
        return base

    new_pv = _patch_param_map(intent.param_values)
    new_filters = [_maybe_coerce_bool_literal_for_where(fp) for fp in where_leaves(intent.where) or []]
    new_having = [_maybe_coerce_bool_literal_for_having(hp) for hp in having_leaves(intent.having) or []]
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        nf = [_maybe_coerce_bool_literal_for_where(x) for x in where_leaves(cte.where) or []]
        nh = [_maybe_coerce_bool_literal_for_having(x) for x in having_leaves(cte.having) or []]
        new_ctes.append(
            replace(
                cte,
                where=predicate_group_from_list(nf),
                having=predicate_group_from_list(nh),
                param_values=_patch_param_map(cte.param_values),
            )
        )
    return replace(
        intent,
        where=predicate_group_from_list(new_filters),
        having=predicate_group_from_list(new_having),
        cte_steps=new_ctes,
        param_values=new_pv,
    )


def _align_where_value_type(
    fp: WhereParam, schema: SchemaGraph, cte_steps: Sequence[RuntimeCteStep] | None
) -> WhereParam:
    aligned = _align_pred_value_type(fp, schema, cte_steps)
    return aligned if isinstance(aligned, WhereParam) else fp


def _align_having_value_type(
    hp: HavingParam, schema: SchemaGraph, cte_steps: Sequence[RuntimeCteStep] | None
) -> HavingParam:
    aligned = _align_pred_value_type(hp, schema, cte_steps)
    return aligned if isinstance(aligned, HavingParam) else hp


def align_where_value_type_to_exprs(intent: RuntimeIntent, schema: SchemaGraph) -> RuntimeIntent:
    """Align ``WhereParam.value_type`` and ``HavingParam.value_type`` to the actual typing of the predicate sides. Decision order: ``is null``/``is not null`` is preserved; ``date_window``/``date_diff`` is preserved; LIKE/ILIKE/contains -> ``string``; both sides numeric (and not string columns) -> ``number``; both sides date -> ``date``; any side string column -> ``string``. Walks main + CTE filters/havings and case-branch conditions, consulting CTE output column metadata when a predicate references a ``cte_name.alias`` reference."""
    cte_steps_seq = intent.cte_steps or []
    new_filters = [_align_where_value_type(fp, schema, cte_steps_seq) for fp in (where_leaves(intent.where) or [])]
    new_having = [_align_having_value_type(hp, schema, cte_steps_seq) for hp in (having_leaves(intent.having) or [])]
    new_cte_steps = []
    for cte in cte_steps_seq:
        cte_filters = [_align_where_value_type(fp, schema, cte_steps_seq) for fp in (where_leaves(cte.where) or [])]
        cte_having = [_align_having_value_type(hp, schema, cte_steps_seq) for hp in (having_leaves(cte.having) or [])]
        new_cte_steps.append(
            replace(cte, where=predicate_group_from_list(cte_filters), having=predicate_group_from_list(cte_having))
        )
    intent = replace(
        intent,
        where=predicate_group_from_list(new_filters),
        having=predicate_group_from_list(new_having),
        cte_steps=new_cte_steps,
    )

    def _branch_align(conds: list[WhereParam]) -> list[WhereParam]:
        return [_align_where_value_type(c, schema, cte_steps_seq) for c in conds]

    intent = map_case_branch_conditions(intent, _branch_align)

    def _branch_coerce_bool_num(conds: list[WhereParam]) -> list[WhereParam]:
        if not conds:
            return conds
        return [_maybe_coerce_bool_literal_for_where(conds[0])]

    intent = map_case_branch_conditions(intent, _branch_coerce_bool_num, scopes=frozenset({"where", "having"}))
    return _coerce_boolean_bindings_for_number_typed_where(intent)


def lift_distinct_modifier_in_multiply(intent: RuntimeIntent) -> RuntimeIntent:
    """Strip standalone ``DISTINCT`` prefixes from multiply tokens. lacking an aggregate wrapper. Bare row-level ``DISTINCT col`` tokens emitted by the LLM cannot render to valid SQL when no surrounding aggregate consumes them; this repair removes the prefix so downstream rendering succeeds. Multiply tokens inside ``COUNT(...)`` or other aggregates are preserved because the deterministic SQL renderer already emits ``COUNT(DISTINCT col)`` for those. The first select column from which a bare ``DISTINCT`` is stripped records its index on ``intent.distinct_select_index`` (and on each ``RuntimeCteStep.distinct_select_index`` for CTE scopes); the renderer reads this to emit ``SELECT DISTINCT``. ``DISTINCT`` is a statement-level modifier so only the first stripped index is recorded per scope."""
    new_select: list[SelectCol] = []
    main_distinct_index = intent.distinct_select_index
    for i, sc in enumerate(intent.select_cols or []):
        new_sc, lifted = _lift_distinct_from_select_col(sc)
        new_select.append(new_sc)
        if lifted and main_distinct_index < 0:
            main_distinct_index = i
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_select: list[SelectCol] = []
        cte_distinct_index = cte.distinct_select_index
        for j, sc in enumerate(cte.select_cols or []):
            new_sc, lifted = _lift_distinct_from_select_col(sc)
            cte_select.append(new_sc)
            if lifted and cte_distinct_index < 0:
                cte_distinct_index = j
        new_cte_steps.append(replace(cte, select_cols=cte_select, distinct_select_index=cte_distinct_index))
    return replace(intent, select_cols=new_select, cte_steps=new_cte_steps, distinct_select_index=main_distinct_index)


def _rewrite_group_unknown_datepart_to_extract(group: MulGroup) -> MulGroup:
    """Rewrite ``YEAR(x)``-style scalar funcs (and inner) into canonical ``EXTRACT(unit FROM x)``."""
    sf = (group.scalar_func or "").lower()
    isf = (group.inner_scalar_func or "").lower()
    changed = False
    if sf in UNKNOWN_DATEPART_TO_EXTRACT_UNIT and not group.scalar_func_args:
        unit = UNKNOWN_DATEPART_TO_EXTRACT_UNIT[sf]
        group = replace(group, scalar_func="extract", scalar_func_args=[unit])
        changed = True
    if isf in UNKNOWN_DATEPART_TO_EXTRACT_UNIT and not group.inner_scalar_func_args:
        unit = UNKNOWN_DATEPART_TO_EXTRACT_UNIT[isf]
        group = replace(group, inner_scalar_func="extract", inner_scalar_func_args=[unit])
        changed = True
    if changed:
        debug(f"[replace_unknown_scalar_funcs] rewrote {sf!r}/{isf!r} to extract")
    return group


def _rewrite_expr_unknown_datepart_to_extract(expr: NormalizedExpr | None) -> NormalizedExpr | None:
    """Apply unknown-datepart rewrite across all groups of *expr*."""
    if expr is None:
        return expr
    new_add = [_rewrite_group_unknown_datepart_to_extract(g) for g in (expr.add_groups or [])]
    new_sub = [_rewrite_group_unknown_datepart_to_extract(g) for g in (expr.sub_groups or [])]
    sf = (expr.scalar_func or "").lower()
    isf = (expr.inner_scalar_func or "").lower()
    new_sf = expr.scalar_func
    new_sfa = list(expr.scalar_func_args or [])
    new_isf = expr.inner_scalar_func
    new_isfa = list(expr.inner_scalar_func_args or [])
    if sf in UNKNOWN_DATEPART_TO_EXTRACT_UNIT and not new_sfa:
        new_sf = "extract"
        new_sfa = [UNKNOWN_DATEPART_TO_EXTRACT_UNIT[sf]]
    if isf in UNKNOWN_DATEPART_TO_EXTRACT_UNIT and not new_isfa:
        new_isf = "extract"
        new_isfa = [UNKNOWN_DATEPART_TO_EXTRACT_UNIT[isf]]
    return replace(
        expr,
        add_groups=new_add,
        sub_groups=new_sub,
        scalar_func=new_sf,
        scalar_func_args=new_sfa,
        inner_scalar_func=new_isf,
        inner_scalar_func_args=new_isfa,
    )


def _rewrite_select_col_unknown_datepart(sc: SelectCol) -> SelectCol:
    """Apply rewrite to a SelectCol's expression."""
    new_expr = _rewrite_expr_unknown_datepart_to_extract(sc.expr)
    if new_expr is None or new_expr is sc.expr:
        return sc
    return replace(sc, expr=new_expr)


def replace_unknown_scalar_funcs(intent: RuntimeIntent) -> RuntimeIntent:
    """Rewrite ``YEAR``/``MONTH``/``DAY``/``QUARTER``/``DOW`` calls to canonical ``EXTRACT(unit FROM x)``. The renderer and validator only accept the ``extract`` scalar; LLMs often produce vendor-specific date-part functions that fail validation or execution. This deterministic step normalizes them in main and CTE select expressions, filters, having, group-by, and order-by exprs."""

    def _rewrite_filters(items: list[WhereParam]) -> list[WhereParam]:
        out: list[WhereParam] = []
        for fp in items:
            new_left = _rewrite_expr_unknown_datepart_to_extract(fp.left_expr) or fp.left_expr
            new_right = (
                _rewrite_expr_unknown_datepart_to_extract(fp.right_expr) or fp.right_expr
                if fp.right_expr
                else fp.right_expr
            )
            out.append(replace(fp, left_expr=new_left, right_expr=new_right))
        return out

    def _rewrite_havings(items: list[HavingParam]) -> list[HavingParam]:
        out: list[HavingParam] = []
        for hp in items:
            new_left = _rewrite_expr_unknown_datepart_to_extract(hp.left_expr) or hp.left_expr
            new_right = (
                _rewrite_expr_unknown_datepart_to_extract(hp.right_expr) or hp.right_expr
                if hp.right_expr
                else hp.right_expr
            )
            out.append(replace(hp, left_expr=new_left, right_expr=new_right))
        return out

    new_select = [_rewrite_select_col_unknown_datepart(sc) for sc in (intent.select_cols or [])]
    new_group_by = [_rewrite_expr_unknown_datepart_to_extract(g) or g for g in (intent.group_by_cols or [])]
    new_order_by = [
        replace(obc, expr=_rewrite_expr_unknown_datepart_to_extract(obc.expr) or obc.expr)
        for obc in (intent.order_by_cols or [])
    ]
    new_filters = _rewrite_filters(where_leaves(intent.where) or [])
    new_having = _rewrite_havings(having_leaves(intent.having) or [])
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_select = [_rewrite_select_col_unknown_datepart(sc) for sc in (cte.select_cols or [])]
        cte_group_by = [_rewrite_expr_unknown_datepart_to_extract(g) or g for g in (cte.group_by_cols or [])]
        cte_order_by = [
            replace(obc, expr=_rewrite_expr_unknown_datepart_to_extract(obc.expr) or obc.expr)
            for obc in (cte.order_by_cols or [])
        ]
        cte_filters = _rewrite_filters(where_leaves(cte.where) or [])
        cte_having = _rewrite_havings(having_leaves(cte.having) or [])
        new_cte_steps.append(
            replace(
                cte,
                select_cols=cte_select,
                group_by_cols=cte_group_by,
                order_by_cols=cte_order_by,
                where=reapply_predicate_leaves(cte.where, cte_filters),
                having=reapply_predicate_leaves(cte.having, cte_having),
            )
        )
    return replace(
        intent,
        select_cols=new_select,
        group_by_cols=new_group_by,
        order_by_cols=new_order_by,
        where=reapply_predicate_leaves(intent.where, new_filters),
        having=reapply_predicate_leaves(intent.having, new_having),
        cte_steps=new_cte_steps,
    )


def _append_expr_cols(buf: list[str], expr: NormalizedExpr | None) -> None:
    """Append qualified column strings referenced by *expr* to *buf*."""
    if expr is None:
        return
    buf.extend(extract_columns_from_expr(expr))


def _append_window_spec_cols(buf: list[str], ws: WindowSpec | None) -> None:
    """Append columns referenced inside a window specification."""
    if ws is None:
        return
    for pe in ws.partition_by or []:
        buf.extend(extract_columns_from_expr(pe))
    for obc in ws.order_by or []:
        buf.extend(extract_columns_from_expr(obc.expr))
    _append_expr_cols(buf, ws.argument)


def _append_case_when_cols(buf: list[str], cw: CaseWhenExpr | None) -> None:
    """Append columns referenced inside a CASE expression."""
    if cw is None:
        return
    for br in cw.branches or []:
        buf.extend(extract_columns_from_expr(br.condition.left_expr))
        if br.condition.right_expr:
            buf.extend(extract_columns_from_expr(br.condition.right_expr))
        buf.extend(extract_columns_from_expr(br.result))
    _append_expr_cols(buf, cw.else_result)


def cols_from_select_col(
    sc: SelectCol,
    window_registry: Sequence[WindowRegistryStep] | None,
    case_registry: Sequence[CaseRegistryStep] | None,
) -> list[str]:
    """Column references from a select column, including window, CASE, and registry resolution."""
    buf: list[str] = []
    resolved = effective_select_parts(sc, window_registry, case_registry)
    buf.extend(extract_columns_from_expr(resolved.expr))
    _append_window_spec_cols(buf, resolved.window_spec)
    _append_case_when_cols(buf, resolved.case_when)
    return buf


def cols_from_named_registries(
    window_registry: Sequence[WindowRegistryStep] | None, case_registry: Sequence[CaseRegistryStep] | None
) -> list[str]:
    """Column references from window and case registry definitions."""
    buf: list[str] = []
    for win_step in window_registry or []:
        _append_window_spec_cols(buf, win_step.window_spec)
    for case_step in case_registry or []:
        _append_case_when_cols(buf, case_step.case_when)
    return buf


def _tables_from_columns(cols: list[str]) -> set[str]:
    """Return distinct table prefixes from qualified ``table.column`` strings."""
    tables: set[str] = set()
    for col in cols:
        if "." in col:
            head = col.split(".", 1)[0]
            if IDENTIFIER_RE.match(head):
                tables.add(head)
    return tables


def collect_referenced_tables(
    select_cols: list[SelectCol],
    order_by_cols: list[OrderByCol],
    group_by_cols: list[NormalizedExpr],
    where_params: list[WhereParam],
    having_param: list[HavingParam],
    *,
    window_registry: Sequence[WindowRegistryStep] | None = None,
    case_registry: Sequence[CaseRegistryStep] | None = None,
    include_unreferenced_registries: bool = True,
) -> set[str]:
    """Union of tables referenced in select, order, group, filters, and. having."""
    all_cols: list[str] = []
    for sc in select_cols or []:
        all_cols.extend(cols_from_select_col(sc, window_registry, case_registry))
    if include_unreferenced_registries:
        all_cols.extend(cols_from_named_registries(window_registry, case_registry))
    for obc in order_by_cols or []:
        all_cols.extend(extract_columns_from_expr(obc.expr))
    for g in group_by_cols or []:
        all_cols.extend(extract_columns_from_expr(g))
    for fp in where_params or []:
        all_cols.extend(extract_columns_from_expr(fp.left_expr))
        if fp.right_expr:
            all_cols.extend(extract_columns_from_expr(fp.right_expr))
    for hp in having_param or []:
        all_cols.extend(extract_columns_from_expr(hp.left_expr))
        if hp.right_expr:
            all_cols.extend(extract_columns_from_expr(hp.right_expr))
    return _tables_from_columns(all_cols)


def collect_projected_tables(
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    *,
    distinct_on: Sequence[NormalizedExpr] | None = None,
    window_registry: Sequence[WindowRegistryStep] | None = None,
    case_registry: Sequence[CaseRegistryStep] | None = None,
    include_unreferenced_registries: bool = True,
) -> set[str]:
    """Return tables referenced by projection clauses rather than comparison operands."""
    all_cols: list[str] = []
    for sc in select_cols or []:
        all_cols.extend(cols_from_select_col(sc, window_registry, case_registry))
    if include_unreferenced_registries:
        all_cols.extend(cols_from_named_registries(window_registry, case_registry))
    for obc in order_by_cols or []:
        all_cols.extend(extract_columns_from_expr(obc.expr))
    for g in group_by_cols or []:
        all_cols.extend(extract_columns_from_expr(g))
    for expr in distinct_on or []:
        all_cols.extend(extract_columns_from_expr(expr))
    return _tables_from_columns(all_cols)


def collect_comparison_operand_tables(
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
) -> set[str]:
    """Return tables referenced only on the right-hand side of cross- table predicates."""
    all_cols: list[str] = []
    for fp in where_params or []:
        if fp.right_expr:
            all_cols.extend(extract_columns_from_expr(fp.right_expr))
    for hp in having_param or []:
        if hp.right_expr:
            all_cols.extend(extract_columns_from_expr(hp.right_expr))
    return _tables_from_columns(all_cols)


def derive_comparison_only_tables(
    *,
    select_cols: Sequence[SelectCol] | None,
    order_by_cols: Sequence[OrderByCol] | None,
    group_by_cols: Sequence[NormalizedExpr] | None,
    where_params: Sequence[WhereParam] | None,
    having_param: Sequence[HavingParam] | None,
    distinct_on: Sequence[NormalizedExpr] | None = None,
    window_registry: Sequence[WindowRegistryStep] | None = None,
    case_registry: Sequence[CaseRegistryStep] | None = None,
) -> list[str]:
    """Return tables brought into scope solely by cross-table comparison operands."""
    projected = collect_projected_tables(
        select_cols,
        order_by_cols,
        group_by_cols,
        distinct_on=distinct_on,
        window_registry=window_registry,
        case_registry=case_registry,
    )
    comparison_ops = collect_comparison_operand_tables(where_params, having_param)
    return sorted(comparison_ops - projected, key=str.lower)


def is_join_unreachable_issue(issue: IntentIssue) -> bool:
    """Return whether *issue* is a join-path reachability refusal."""
    return str(issue.issue_id or "").startswith("join_unreachable_")


def _scope_tables(intent: RuntimeIntent, scope_label: str) -> frozenset[str]:
    if scope_label == "main query":
        return frozenset(intent.tables or [])
    if scope_label.startswith("CTE '") and scope_label.endswith("'"):
        cte_name = scope_label[5:-1]
        for cte in intent.cte_steps or []:
            if cte.cte_name == cte_name:
                return frozenset(cte.tables or [])
    return frozenset(intent.tables or [])


def refusal_for_join_unreachable_table_removal(
    before: RuntimeIntent,
    after: RuntimeIntent,
    open_errors: Sequence[IntentIssue],
) -> str | None:
    """Refuse when repair drops tables while a join-unreachable error was still open."""
    unreachable = [issue for issue in open_errors if is_join_unreachable_issue(issue)]
    if not unreachable:
        return None
    for issue in unreachable:
        ctx = issue.context or {}
        scope_label = str(ctx.get("scope_label") or "main query")
        before_tables = _scope_tables(before, scope_label)
        after_tables = _scope_tables(after, scope_label)
        removed = before_tables - after_tables
        if not removed:
            continue
        root = str(ctx.get("root") or "")
        target = str(ctx.get("target") or "")
        if root and target:
            return (
                f"Tables '{root}' and '{target}' cannot be joined: no foreign key or semantic edge "
                "relates them. Repair removed a table to clear the error, which would answer a "
                "different question. Declare foreign_keys_add or a semantic neighbour override when "
                "the relationship is real."
            )
        return (
            "Repair removed a table while a join-unreachable error was open, which would answer a "
            "different question. Declare foreign_keys_add or a semantic neighbour override when "
            "the relationship is real."
        )
    return None


def referenced_registry_ids_in_scope(
    *,
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
) -> set[str]:
    """Return registry ids referenced by clause expressions in one query scope."""
    ids: set[str] = set()
    for sc in select_cols or []:
        ref = expr_registry_ref(sc.expr)
        if ref:
            ids.add(ref)
    for obc in order_by_cols or []:
        ref = expr_registry_ref(obc.expr)
        if ref:
            ids.add(ref)
    for group in group_by_cols or []:
        ref = expr_registry_ref(group)
        if ref:
            ids.add(ref)
    for fp in where_params or []:
        for expr in (fp.left_expr, fp.right_expr):
            if expr is None:
                continue
            ref = expr_registry_ref(expr)
            if ref:
                ids.add(ref)
    for hp in having_param or []:
        for expr in (hp.left_expr, hp.right_expr):
            if expr is None:
                continue
            ref = expr_registry_ref(expr)
            if ref:
                ids.add(ref)
    return ids


def where_scope_registries_to_referenced(
    *,
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
    window_registry: Sequence[WindowRegistryStep] | None = None,
    case_registry: Sequence[CaseRegistryStep] | None = None,
) -> tuple[list[WindowRegistryStep], list[CaseRegistryStep]]:
    """Keep only window/case registry rows referenced by the scope's clauses."""
    referenced = referenced_registry_ids_in_scope(
        select_cols=select_cols,
        order_by_cols=order_by_cols,
        group_by_cols=group_by_cols,
        where_params=where_params,
        having_param=having_param,
    )
    kept_wr = [step for step in (window_registry or []) if step.registry_id in referenced]
    kept_cr = [step for step in (case_registry or []) if step.registry_id in referenced]
    return kept_wr, kept_cr


def append_table_scope_repairs(
    intent: RuntimeIntent,
    *,
    scope_label: str,
    added: Sequence[str] | None = None,
    removed: Sequence[str] | None = None,
    add_reason: Literal["planner_align", "expression_reference", "unreferenced_table", "join_bridge"] = (
        "planner_align"
    ),
    remove_reason: Literal["planner_align", "expression_reference", "unreferenced_table", "join_bridge"] = (
        "unreferenced_table"
    ),
) -> RuntimeIntent:
    """Append engine table-scope repair records when tables are added or removed."""
    repairs = list(intent.table_scope_repairs)
    add_set = sorted({str(t) for t in (added or ()) if str(t)})
    rem_set = sorted({str(t) for t in (removed or ()) if str(t)})
    if add_set:
        repairs.append(TableScopeRepair(scope_label, tuple(add_set), "add", add_reason))
    if rem_set:
        repairs.append(TableScopeRepair(scope_label, tuple(rem_set), "remove", remove_reason))
    if repairs == list(intent.table_scope_repairs):
        return intent
    return replace(intent, table_scope_repairs=repairs)


def format_table_scope_repair_message(repair: TableScopeRepair) -> str:
    """Return one user-facing line describing a table-scope repair."""
    reason_text = TABLE_SCOPE_REPAIR_REASON_TEXT.get(repair.reason, repair.reason)
    verb = "added" if repair.action == "add" else "removed"
    table_list = ", ".join(repair.tables)
    return f"Engine {verb} table(s) {table_list} in {repair.scope_label} because they are {reason_text}."


def table_scope_repair_warning_messages(intent: RuntimeIntent) -> list[str]:
    """Return deduplicated user-facing warning strings for recorded table-scope repairs."""
    seen: set[str] = set()
    out: list[str] = []
    for repair in intent.table_scope_repairs or []:
        msg = format_table_scope_repair_message(repair)
        if msg in seen:
            continue
        seen.add(msg)
        out.append(msg)
    return out


def validate_table_scope_repairs(intent: RuntimeIntent) -> list[IntentIssue]:
    """Surface recorded table-scope repairs as semantic warnings."""
    issues: list[IntentIssue] = []
    for idx, repair in enumerate(intent.table_scope_repairs or []):
        issues.append(
            make_intent_issue(
                issue_id=f"table_scope_repair_{repair.scope_label.replace(' ', '_')}_{repair.action}_{idx}",
                category=FailureCategory.STRUCTURAL,
                severity="warning",
                message=format_table_scope_repair_message(repair),
                context={
                    "scope_label": repair.scope_label,
                    "tables": list(repair.tables),
                    "action": repair.action,
                    "reason": repair.reason,
                },
            )
        )
    return issues


def reconcile_tables(intent: RuntimeIntent) -> RuntimeIntent:
    """Set ``tables`` at every level to exactly the tables and CTEs. referenced at that level. For the main scope and each CTE step independently, this function recomputes the referenced table set from select, order, group, filter, having, window registry, and case registry expressions. The resulting ``tables`` list is the sorted reference set. No table is force-added because of CTE chain membership and no prior CTE name is inserted into a downstream CTE's tables list. Tables present in the input but not referenced are removed; tables referenced but missing from the input are added back."""
    main_referenced = collect_referenced_tables(
        intent.select_cols,
        intent.order_by_cols,
        intent.group_by_cols,
        where_leaves(intent.where),
        having_leaves(intent.having),
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    main_tables = sorted(main_referenced)
    main_comparison_only = derive_comparison_only_tables(
        select_cols=intent.select_cols,
        order_by_cols=intent.order_by_cols,
        group_by_cols=intent.group_by_cols,
        where_params=where_leaves(intent.where),
        having_param=having_leaves(intent.having),
        distinct_on=intent.distinct_on,
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    original_main = set(intent.tables or [])
    added_main = main_referenced - original_main
    removed_main = original_main - main_referenced
    if added_main:
        debug(f"[reconcile_tables] main added {sorted(added_main)}")
    if removed_main:
        debug(f"[reconcile_tables] main removed {sorted(removed_main)}")
    intent = append_table_scope_repairs(
        intent,
        scope_label="main query",
        added=sorted(added_main),
        removed=sorted(removed_main),
        add_reason="expression_reference",
        remove_reason="unreferenced_table",
    )

    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_referenced = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            where_leaves(cte.where),
            having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
        )
        cte_tables = sorted(cte_referenced)
        cte_comparison_only = derive_comparison_only_tables(
            select_cols=cte.select_cols,
            order_by_cols=cte.order_by_cols,
            group_by_cols=cte.group_by_cols,
            where_params=where_leaves(cte.where),
            having_param=having_leaves(cte.having),
            distinct_on=cte.distinct_on,
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
        )
        cte_original = set(cte.tables or [])
        cte_added = cte_referenced - cte_original
        cte_removed = cte_original - cte_referenced
        if cte_added:
            debug(f"[reconcile_tables] CTE '{cte.cte_name}' added {sorted(cte_added)}")
        if cte_removed:
            debug(f"[reconcile_tables] CTE '{cte.cte_name}' removed {sorted(cte_removed)}")
        cte_scope = f"CTE '{cte.cte_name}'" if cte.cte_name else "CTE"
        intent = append_table_scope_repairs(
            intent,
            scope_label=cte_scope,
            added=sorted(cte_added),
            removed=sorted(cte_removed),
            add_reason="expression_reference",
            remove_reason="unreferenced_table",
        )
        new_cte_steps.append(replace(cte, tables=cte_tables, comparison_only_tables=cte_comparison_only))

    return replace(
        intent,
        tables=main_tables,
        comparison_only_tables=main_comparison_only,
        cte_steps=new_cte_steps,
    )


def _fk_specialization_parent(scope_table: str, ref_table: str, schema: SchemaGraph) -> bool:
    """Return True when *scope_table* FK-targets *ref_table* on *ref_table*'s primary key."""
    if not _same_federation_source(schema, scope_table, ref_table):
        return False
    tbl = schema.tables.get(scope_table)
    dst_meta = schema.tables.get(ref_table)
    if tbl is None or dst_meta is None:
        return False
    pk_set = set(dst_meta.primary_key or [])
    if not pk_set:
        return False
    for fk in tbl.foreign_keys:
        if fk.inference_tag == InferenceTag.CROSS_SOURCE:
            continue
        if fk.dst_table != ref_table:
            continue
        if set(fk.dst_cols) <= pk_set and fk.src_cols:
            return True
    return False


def _expand_scope_tables_for_refs(tables: list[str], referenced: set[str], schema: SchemaGraph) -> list[str]:
    out_set = set(tables or [])
    for ref_tbl in referenced:
        if ref_tbl in out_set:
            continue
        for scope_tbl in list(out_set):
            if _fk_specialization_parent(scope_tbl, ref_tbl, schema):
                out_set.add(ref_tbl)
                debug(f"[expand_shared_pk_tables_for_refs] added {ref_tbl!r} via {scope_tbl!r}")
                break
    return sorted(out_set)


def expand_shared_pk_tables_for_refs(intent: RuntimeIntent, schema: SchemaGraph) -> RuntimeIntent:
    """Add parent/specialization tables when their columns are referenced but absent from scope."""
    main_ref = collect_referenced_tables(
        intent.select_cols,
        intent.order_by_cols,
        intent.group_by_cols,
        where_leaves(intent.where),
        having_leaves(intent.having),
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    main_tables = _expand_scope_tables_for_refs(list(intent.tables or []), main_ref, schema)
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_ref = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            where_leaves(cte.where),
            having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
        )
        cte_tables = _expand_scope_tables_for_refs(list(cte.tables or []), cte_ref, schema)
        new_cte_steps.append(replace(cte, tables=cte_tables))
    if main_tables == list(intent.tables or []) and new_cte_steps == list(intent.cte_steps or []):
        return intent
    return replace(intent, tables=main_tables, cte_steps=new_cte_steps)


def _coerce_element(val: Any, data_type: str) -> Any:
    """Coerce one IN-list element toward *data_type* (numeric columns. only)."""
    if data_type not in NUMERIC_DATA_TYPES:
        return val
    if isinstance(val, (int, float)):
        return val
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except (ValueError, OverflowError):
        return val


def _consolidate_in_list(vals: list[Any], data_type: str) -> str:
    """Join IN-list values into a comma-separated SQL fragment string."""
    if all(isinstance(v, (int, float)) for v in vals):
        return ", ".join(str(v) for v in vals)
    parts: list[str] = []
    for v in vals:
        if isinstance(v, str):
            parts.append(f"'{v}'")
        else:
            parts.append(str(v))
    return ", ".join(parts)


def _normalize_in_types_for_list(filters: list[WhereParam], schema_graph: SchemaGraph) -> tuple[list[WhereParam], bool]:
    """Coerce IN-list elements to column types, then consolidate to one. SQL string."""
    new_filters: list[WhereParam] = []
    changed = False
    for fp in filters:
        if fp.op.lower() not in {"in", "not in"} or not isinstance(fp.raw_value, list):
            new_filters.append(fp)
            continue
        col = fp.left_expr.primary_column
        parts = col.split(".", 1) if "." in col else None
        if not parts:
            new_filters.append(fp)
            continue
        col_meta = schema_graph.get_column(parts[0], parts[1])
        dtype = (col_meta.data_type or "").lower() if col_meta else ""
        coerced = [_coerce_element(v, dtype) for v in fp.raw_value]
        list_changed = any(a != b for a, b in zip(coerced, fp.raw_value, strict=True))
        consolidated = _consolidate_in_list(coerced, dtype)
        if list_changed:
            new_filters.append(replace(fp, raw_value=consolidated))
            changed = True
            debug(f"[intent_resolve_normalize_in_types_for_list] {col}: {fp.raw_value!r} -> {consolidated!r}")
        else:
            new_filters.append(fp)
    return new_filters, changed


def normalize_in_where_types(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Run IN coercion on main/CTEs, then. ``_decompose_in_not_in_where``. Filter-only by design; HAVING operates on aggregation results, not ``IN``-list expansion here."""

    def process(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
        return _normalize_in_types_for_list(filters, schema_graph)

    intent = apply_where_to_main_and_ctes(intent, process)
    return _decompose_in_not_in_where(intent)


def _decompose_in_list(filters: list[WhereParam], max_list_size: int = 10) -> PredicateGroup | None:
    """Split short IN/NOT IN lists into ``=``/``!=`` leaves under OR (IN) or AND (NOT IN) groups."""
    parts: list[PredicateGroup] = []
    for fp in filters:
        raw = fp.raw_value
        op_lower = (fp.op or "").lower()
        if op_lower in {"in", "not in"} and isinstance(raw, str):
            split_parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
            split_parts = [p for p in split_parts if p]
            value_type_lower = (fp.value_type or "").lower()
            if value_type_lower in {"integer", "int", "bigint", "smallint"}:
                coerced: list[Any] = []
                for p in split_parts:
                    try:
                        coerced.append(int(p))
                    except (TypeError, ValueError):
                        coerced.append(p)
                split_parts = coerced
            elif value_type_lower in {
                "number",
                "numeric",
                "float",
                "double",
                "decimal",
                "real",
            }:
                coerced_f: list[Any] = []
                for p in split_parts:
                    try:
                        coerced_f.append(float(p))
                    except (TypeError, ValueError):
                        coerced_f.append(p)
                split_parts = coerced_f
            if split_parts:
                fp = replace(fp, raw_value=cast(RawValue, split_parts))
                raw = cast(list[Any], split_parts)
        if not isinstance(raw, list) or op_lower not in {"in", "not in"} or len(raw) == 0 or len(raw) > max_list_size:
            parts.append(PredicateGroup(op="and", predicates=(fp,)))
            continue
        elems = list(raw)
        expanded = tuple(replace(fp, op="=" if op_lower == "in" else "!=", raw_value=val) for val in elems)
        connector: Literal["and", "or"] = "or" if op_lower == "in" else "and"
        parts.append(PredicateGroup(op=connector, predicates=expanded))
    return merge_predicate_groups("and", parts)


def _decompose_in_not_in_where(intent: RuntimeIntent) -> RuntimeIntent:
    """Apply ``_decompose_in_list`` to main and each CTE (filter-only; HAVING is out of scope)."""
    main_where = _decompose_in_list(where_leaves(intent.where) or [])
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        decomposed = _decompose_in_list(where_leaves(cte.where) or [])
        new_ctes.append(replace(cte, where=decomposed))
    out = replace(intent, where=main_where, cte_steps=new_ctes or intent.cte_steps)
    expanded = len((main_where.leaves() if main_where else []) or []) != len(where_leaves(intent.where) or [])
    if not expanded:
        for oc, nc in zip(intent.cte_steps or [], new_ctes, strict=True):
            if len(where_leaves(oc.where) or []) != len(where_leaves(nc.where) or []):
                expanded = True
                break
    if expanded:
        pipeline_trace(
            "intent_after_deterministic_repair.decompose_in_filters",
            lambda: stable_json(
                {
                    "main_filters": len(where_leaves(out.where) or []),
                    "cte_steps": len(out.cte_steps or []),
                }
            ),
        )
    return out


def _resolve_boolean_value(raw_value: Any, col_meta: ColumnMetadata) -> tuple[Any, str] | None:
    """Map *raw_value* to ``True``/``False`` when the column is a. native. boolean type."""
    dtype_lower = (col_meta.data_type or "").lower()
    if "bool" not in dtype_lower:
        return None
    if isinstance(raw_value, bool):
        return raw_value, "boolean"
    val_str = str(raw_value).lower().strip()
    if val_str in BOOLEAN_TRUTHY_VALUES:
        return True, "boolean"
    if val_str in BOOLEAN_FALSY_VALUES:
        return False, "boolean"
    return None


def _normalize_boolean_where_list(
    filters: list[WhereParam], schema_graph: SchemaGraph
) -> tuple[list[WhereParam], bool]:
    """Rewrite boolean-column filters to Python bool and ``value_type`` ``boolean``."""
    new_filters: list[WhereParam] = []
    changed = False
    for fp in filters:
        if fp.raw_value is None:
            new_filters.append(fp)
            continue
        col = fp.left_expr.primary_column
        parts = col.split(".", 1) if "." in col else None
        if not parts:
            new_filters.append(fp)
            continue
        col_meta = schema_graph.get_column(parts[0], parts[1])
        if not col_meta:
            new_filters.append(fp)
            continue
        resolved = _resolve_boolean_value(fp.raw_value, col_meta)
        if resolved is None:
            new_filters.append(fp)
            continue
        bool_val, vtype = resolved
        new_filters.append(replace(fp, raw_value=bool_val, value_type=vtype))
        changed = True
        debug(
            f"[intent_resolve_normalize_boolean_where_list] {col}: "
            f"{fp.raw_value!r} ({fp.value_type}) → {bool_val!r} ({vtype})"
        )
    return new_filters, changed


def normalize_boolean_where_values(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Apply ``_normalize_boolean_where_list`` to main and CTE where trees. Where-only by design; boolean coercion applies to ``where`` row literals, not HAVING."""

    def process(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
        return _normalize_boolean_where_list(filters, schema_graph)

    return apply_where_to_main_and_ctes(intent, process)


def _normalize_null_where_list(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
    """Force ``value_type="null"`` and ``raw_value=None`` for null. operators."""
    result: list[WhereParam] = []
    changed = False
    for fp in filters:
        if fp.op in ("is null", "is not null"):
            needs_fix = fp.value_type != "null" or fp.raw_value is not None
            if needs_fix:
                result.append(replace(fp, value_type="null", raw_value=None))
                changed = True
                continue
        result.append(fp)
    return result, changed


def _normalize_null_having_list(having: list[HavingParam]) -> tuple[list[HavingParam], bool]:
    """Force ``value_type="null"`` and ``raw_value=None`` for null operators on HAVING rows."""
    result: list[HavingParam] = []
    changed = False
    for hp in having:
        if hp.op in ("is null", "is not null"):
            needs_fix = hp.value_type != "null" or hp.raw_value is not None
            if needs_fix:
                result.append(replace(hp, value_type="null", raw_value=None))
                changed = True
                continue
        result.append(hp)
    return result, changed


def normalize_null_where_values(intent: RuntimeIntent) -> RuntimeIntent:
    """Apply null-operator normalization to main and CTE ``where`` and ``having`` trees."""
    intent = apply_where_to_main_and_ctes(intent, _normalize_null_where_list)
    return apply_having_to_main_and_ctes(intent, _normalize_null_having_list)


def _allocate_window_registry_id(registry: list[WindowRegistryStep]) -> str:
    """Return the next unused ``wNN`` id given existing window registry steps."""
    mx = 0
    for step in registry:
        m = re.fullmatch(r"w(\d{2})", (step.registry_id or "").strip())
        if m:
            mx = max(mx, int(m.group(1)))
    return f"w{mx + 1:02d}"


def _select_cols_have_aggregation(select_cols: Sequence[SelectCol], window_registry: list[WindowRegistryStep]) -> bool:
    """Return True when *select_cols* contains an aggregated expression without an enclosing window registry step."""
    for sc in select_cols:
        if effective_select_parts(sc, window_registry, None).window_spec is not None:
            continue
        if sc.expr.agg_func and sc.expr.agg_func.lower() in WINDOW_AGG_FUNCTIONS:
            return True
    return False


def _promote_aggregates_to_running_window(
    select_cols: list[SelectCol],
    order_by_cols: list[OrderByCol],
    window_registry: list[WindowRegistryStep],
    case_registry: list[CaseRegistryStep],
) -> tuple[list[SelectCol], list[WindowRegistryStep], bool]:
    """Promote plain aggregates to running-window definitions in ``window_registry``."""
    if not order_by_cols:
        return select_cols, window_registry, False
    registry = list(window_registry)
    promoted: list[SelectCol] = []
    changed = False
    for sc in select_cols:
        parts = effective_select_parts(sc, registry, case_registry)
        if parts.window_spec is not None or parts.case_when is not None:
            promoted.append(sc)
            continue
        agg = (sc.expr.agg_func or "").lower() if sc.expr.agg_func else None
        if not agg and sc.expr.add_groups and sc.expr.add_groups[0].agg_func:
            agg = sc.expr.add_groups[0].agg_func.lower()
        if agg not in WINDOW_AGG_FUNCTIONS:
            promoted.append(sc)
            continue
        argument = replace(sc.expr, agg_func=None)
        ws = WindowSpec(
            function=agg,
            partition_by=[],
            order_by=list(order_by_cols),
            argument=argument,
            frame_kind="rows",
            frame_start="unbounded_preceding",
            frame_end="current_row",
        )
        wid = _allocate_window_registry_id(registry)
        registry.append(WindowRegistryStep(registry_id=wid, window_spec=ws))
        promoted.append(SelectCol(expr=NormalizedExpr.from_column(wid)))
        changed = True
    return promoted, registry, changed


def repair_cumulative_phrasing_window_intent(intent: RuntimeIntent, question_norm: str) -> RuntimeIntent:
    """Promote plain aggregate select columns to running-window aggregates when *question_norm* contains a cumulative phrasing (``running total``, ``cumulative``, ``year-to-date``, ``rolling N``, ``moving sum``)."""
    haystack: str = (question_norm or "") + " " + (intent.natural_language or "")
    if not CUMULATIVE_PHRASING_RE.search(haystack):
        return intent
    main_select: list[SelectCol] = list(intent.select_cols or [])
    main_order: list[OrderByCol] = list(intent.order_by_cols or [])
    main_wr: list[WindowRegistryStep] = list(intent.window_registry or [])
    main_changed: bool = False
    if _select_cols_have_aggregation(main_select, main_wr):
        main_select, main_wr, main_changed = _promote_aggregates_to_running_window(
            main_select, main_order, main_wr, list(intent.case_registry or [])
        )
    if main_changed:
        intent = replace(intent, select_cols=main_select, window_registry=main_wr)
    if not intent.cte_steps:
        return intent
    new_ctes: list[RuntimeCteStep] = []
    cte_changed: bool = False
    for cte in intent.cte_steps:
        cte_select: list[SelectCol] = list(cte.select_cols or [])
        cte_order: list[OrderByCol] = list(cte.order_by_cols or [])
        cte_wr: list[WindowRegistryStep] = list(cte.window_registry or [])
        if _select_cols_have_aggregation(cte_select, cte_wr):
            cte_select, cte_wr, c = _promote_aggregates_to_running_window(
                cte_select, cte_order, cte_wr, list(cte.case_registry or [])
            )
            if c:
                cte_changed = True
                new_ctes.append(replace(cte, select_cols=cte_select, window_registry=cte_wr))
                continue
        new_ctes.append(cte)
    if cte_changed:
        intent = replace(intent, cte_steps=new_ctes)
    return intent


def drop_invalid_case_registry_entries(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Remove case-registry rows whose ``case_when`` has no branches. and. drop select columns that reference those ids."""
    _ = schema_graph

    def prune_scope(
        select_cols: list[SelectCol], case_registry: list[CaseRegistryStep]
    ) -> tuple[list[SelectCol], list[CaseRegistryStep], bool]:
        invalid_ids = {step.registry_id for step in case_registry if not step.case_when.branches}
        if not invalid_ids:
            return select_cols, case_registry, False
        kept_registry = [s for s in case_registry if s.registry_id not in invalid_ids]
        kept_select: list[SelectCol] = []
        for sc in select_cols:
            ref = expr_registry_ref(sc.expr)
            if ref is not None and ref in invalid_ids:
                continue
            kept_select.append(sc)
        return kept_select, kept_registry, True

    main_sel, main_cr, main_changed = prune_scope(list(intent.select_cols or []), list(intent.case_registry or []))
    result = intent
    if main_changed:
        result = replace(result, select_cols=main_sel, case_registry=main_cr)
    if not intent.cte_steps:
        return result
    new_ctes: list[RuntimeCteStep] = []
    cte_changed = False
    for cte in intent.cte_steps:
        s, cr, c = prune_scope(list(cte.select_cols or []), list(cte.case_registry or []))
        if c:
            cte_changed = True
            new_ctes.append(replace(cte, select_cols=s, case_registry=cr))
        else:
            new_ctes.append(cte)
    if cte_changed:
        result = replace(result, cte_steps=new_ctes)
    return result


def repair_case_when_intent(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Drop ``case_registry`` rows whose ``case_when`` has no branches."""
    _ = schema_graph

    def _strip_registry(regs: list[CaseRegistryStep] | None) -> list[CaseRegistryStep]:
        return [s for s in (regs or []) if s.case_when and s.case_when.branches]

    new_cr = _strip_registry(intent.case_registry)
    new_ctes = [replace(cte, case_registry=_strip_registry(cte.case_registry)) for cte in (intent.cte_steps or [])]
    return replace(intent, case_registry=new_cr, cte_steps=new_ctes)


def _column_meta_for_where_left(fp: WhereParam, schema_graph: SchemaGraph) -> ColumnMetadata | None:
    """Return metadata when ``left_expr`` references exactly one qualified column."""
    cols = extract_columns_from_expr(fp.left_expr)
    if len(cols) != 1:
        return None
    parts = cols[0].split(".", 1)
    if len(parts) != 2:
        return None
    return schema_graph.get_column(parts[0], parts[1])


def repair_array_where_intent(intent: RuntimeIntent, schema_graph: SchemaGraph, question: str = "") -> RuntimeIntent:
    """Normalise array-column filters: rewrite ``=``/``like`` on array columns to ``contains`` and remove ``contains`` on non-array columns."""

    def process(filters: list[WhereParam]) -> tuple[list[WhereParam], bool]:
        out: list[WhereParam] = []
        changed = False
        for fp in filters:
            meta = _column_meta_for_where_left(fp, schema_graph)
            if fp.op == "contains":
                kind = array_storage_kind(meta) if meta is not None else "unknown"
                if kind in ("native_array", "json_text_array"):
                    out.append(fp)
                    continue
                vt = (meta.value_type or "").lower() if meta else ""
                if meta is not None and vt in STRING_COLUMN_VALUE_TYPES:
                    rv = fp.raw_value
                    if isinstance(rv, str) and rv and "%" not in rv:
                        rv = f"%{rv}%"
                    new_fp = replace(fp, op="like", raw_value=rv, value_type="string")
                    out.append(new_fp)
                    changed = True
                    continue
                if meta is None or not meta.element_type:
                    debug(f"[intent_repair.repair_array_where] dropping contains on non-array column: {fp.param_key}")
                    changed = True
                    continue
            elif (
                fp.op in ARRAY_REWRITABLE_OPS
                and meta is not None
                and array_storage_kind(meta) in ("native_array", "json_text_array")
            ):
                debug(
                    f"[intent_repair.repair_array_where] rewriting {fp.op} to contains for array column: {fp.param_key}"
                )
                fp = replace(fp, op="contains", value_type="string")
                changed = True
            out.append(fp)
        return out, changed

    return apply_where_to_main_and_ctes(intent, process)


def _norm_expr_blocked_non_selectable_refs(expr: NormalizedExpr, schema: SchemaGraph) -> list[str]:
    """Return qualified ``table.column`` references in *expr* that are hidden under sensitivity policy."""
    blocked: list[str] = []
    exempt = selectability_exempt_qualified_refs(expr, schema)
    for ref in extract_columns_from_expr(expr):
        if ref in exempt:
            continue
        parts = ref.split(".", 1)
        if len(parts) != 2:
            continue
        meta = schema.get_column(parts[0], parts[1])
        if meta is not None and meta.sensitivity == SensitivityClassification.HIDDEN:
            blocked.append(ref)
    return blocked


def _select_col_selectable(sc: SelectCol, schema: SchemaGraph) -> bool:
    """Return False when the main expression projects blocked columns without an allowed ``COUNT`` form."""
    return not _norm_expr_blocked_non_selectable_refs(sc.expr, schema)


def _select_col_dropped_blocked_columns(sc: SelectCol, schema: SchemaGraph) -> list[str]:
    """Return the qualified ``table.column`` references in *sc* that fail the selectability gate."""
    return _norm_expr_blocked_non_selectable_refs(sc.expr, schema)


def enforce_sensitivity_policy_intent(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """Drop select and ``GROUP BY`` entries that reference sensitive columns and notify when any are dropped."""
    original_main = list(intent.select_cols or [])
    dropped_main: list[tuple[SelectCol, list[str]]] = []
    kept_main: list[SelectCol] = []
    for sc in original_main:
        blocked = _select_col_dropped_blocked_columns(sc, schema_graph)
        if blocked:
            dropped_main.append((sc, blocked))
        else:
            kept_main.append(sc)
    if dropped_main:
        for _sc, refs in dropped_main:
            notify(
                "Dropping select column(s) referencing sensitive fields: " + ", ".join(sorted(set(refs))),
                stage="intent",
                code=DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT,
            )
    if original_main and not kept_main:
        raise ValueError(
            f"{FailureCategory.SENSITIVITY_ALL_SELECT_DROPPED.value}: every requested select column "
            "references a sensitive field; no projectable output remains"
        )
    intent = replace(intent, select_cols=kept_main)

    original_gb = list(intent.group_by_cols or [])
    kept_gb: list[NormalizedExpr] = []
    dropped_gb_refs: list[str] = []
    for gb in original_gb:
        blocked = _norm_expr_blocked_non_selectable_refs(gb, schema_graph)
        if blocked:
            dropped_gb_refs.extend(blocked)
        else:
            kept_gb.append(gb)
    if dropped_gb_refs:
        notify(
            "Dropping GROUP BY expression(s) referencing sensitive fields: " + ", ".join(sorted(set(dropped_gb_refs))),
            stage="intent",
            code=DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT,
        )
    if original_gb and intent.grain == "grouped" and not kept_gb:
        raise ValueError(
            f"{FailureCategory.SENSITIVITY_ALL_GROUP_BY_DROPPED.value}: every GROUP BY expression "
            "references a sensitive field; no valid grouping keys remain"
        )
    intent = replace(intent, group_by_cols=kept_gb)

    if not intent.cte_steps:
        return intent
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps:
        original_cte_cols = list(cte.select_cols or [])
        kept_cte: list[SelectCol] = []
        for sc in original_cte_cols:
            blocked = _select_col_dropped_blocked_columns(sc, schema_graph)
            if blocked:
                notify(
                    f"Dropping CTE {cte.cte_name!r} select column(s) referencing sensitive fields: "
                    + ", ".join(sorted(set(blocked))),
                    stage="intent",
                    code=DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT,
                )
            else:
                kept_cte.append(sc)
        orig_cte_gb = list(cte.group_by_cols or [])
        kept_cte_gb: list[NormalizedExpr] = []
        dropped_cte_gb: list[str] = []
        for gb in orig_cte_gb:
            blocked = _norm_expr_blocked_non_selectable_refs(gb, schema_graph)
            if blocked:
                dropped_cte_gb.extend(blocked)
            else:
                kept_cte_gb.append(gb)
        if dropped_cte_gb:
            notify(
                f"Dropping CTE {cte.cte_name!r} GROUP BY expression(s) referencing sensitive fields: "
                + ", ".join(sorted(set(dropped_cte_gb))),
                stage="intent",
                code=DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT,
            )
        if orig_cte_gb and getattr(cte, "grain", "") == "grouped" and not kept_cte_gb:
            raise ValueError(
                f"{FailureCategory.SENSITIVITY_ALL_GROUP_BY_DROPPED.value}: CTE {cte.cte_name!r}: every GROUP BY "
                "expression references a sensitive field; no valid grouping keys remain"
            )
        new_ctes.append(replace(cte, select_cols=kept_cte, group_by_cols=kept_cte_gb))
    return replace(intent, cte_steps=new_ctes)


def intent_text_has_leakable_placeholder(text: str | None) -> bool:
    """Return True if *text* still has angle-bracket, numeric, or instructional shape tokens."""
    if not text:
        return False
    if INTENT_PLACEHOLDER_ANGLE_RE.search(text):
        return True
    if re.search(r"\btable_\d+\.", text, re.IGNORECASE):
        return True
    if re.search(r"\btable\d+\.", text, re.IGNORECASE):
        return True
    if re.search(r"\bcolumn_\d+\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bcol\d+\b", text, re.IGNORECASE):
        return True
    lowered = text.lower()
    for token in INSTRUCTIONAL_SHAPE_PLACEHOLDER_TOKENS:
        if "." in token:
            if token in lowered:
                return True
        elif re.search(rf"\b{re.escape(token)}\b", lowered):
            return True
    return False


def _yield_param_value_scan_strings(value: Any) -> Iterator[str]:
    """Yield string leaves from param or raw filter values for placeholder scans."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _yield_param_value_scan_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _yield_param_value_scan_strings(item)


def _yield_mul_group_instructional_strings(group: MulGroup) -> Iterator[str]:
    """Yield textual slots from one multiply/divide group (recursing into child expressions)."""
    for child in group.multiply + group.divide:
        if child.column_ref:
            yield child.column_ref
        if child.raw_sql:
            yield child.raw_sql
        if child.keyword:
            yield child.keyword
        if child.add_groups or child.sub_groups:
            yield from _yield_normalized_expr_instructional_strings(child)
    for arg in group.scalar_func_args:
        if isinstance(arg, str):
            yield arg
    for arg in group.inner_scalar_func_args:
        if isinstance(arg, str):
            yield arg


def _yield_normalized_expr_instructional_strings(expr: NormalizedExpr) -> Iterator[str]:
    """Yield all string-bearing slots from a normalized expression."""
    if expr.column_ref:
        yield expr.column_ref
    if expr.star:
        yield "*"
    if expr.keyword:
        yield expr.keyword
    if expr.raw_sql:
        yield expr.raw_sql
    for grp in expr.add_groups + expr.sub_groups:
        yield from _yield_mul_group_instructional_strings(grp)
    for arg in expr.scalar_func_args:
        if isinstance(arg, str):
            yield arg
    for arg in expr.inner_scalar_func_args:
        if isinstance(arg, str):
            yield arg


def _yield_window_spec_instructional_strings(spec: WindowSpec) -> Iterator[str]:
    """Yield strings from partition, order, and argument expressions."""
    for part in spec.partition_by:
        yield from _yield_normalized_expr_instructional_strings(part)
    for ob in spec.order_by:
        yield from _yield_normalized_expr_instructional_strings(ob.expr)
    if spec.argument is not None:
        yield from _yield_normalized_expr_instructional_strings(spec.argument)


def _yield_where_instructional_strings(fp: WhereParam) -> Iterator[str]:
    """Yield strings from filter expressions and inline raw value."""
    yield from _yield_normalized_expr_instructional_strings(fp.left_expr)
    if fp.right_expr is not None:
        yield from _yield_normalized_expr_instructional_strings(fp.right_expr)
    if fp.raw_value is not None:
        yield from _yield_param_value_scan_strings(fp.raw_value)


def _yield_having_instructional_strings(hp: HavingParam) -> Iterator[str]:
    """Yield strings from HAVING expressions and inline raw value."""
    yield from _yield_normalized_expr_instructional_strings(hp.left_expr)
    if hp.right_expr is not None:
        yield from _yield_normalized_expr_instructional_strings(hp.right_expr)
    if hp.raw_value is not None:
        yield from _yield_param_value_scan_strings(hp.raw_value)


def _yield_case_when_instructional_strings(case_when: CaseWhenExpr) -> Iterator[str]:
    """Yield strings from CASE branches and else clause."""
    for branch in case_when.branches:
        yield from _yield_where_instructional_strings(branch.condition)
        yield from _yield_normalized_expr_instructional_strings(branch.result)
    if case_when.else_result is not None:
        yield from _yield_normalized_expr_instructional_strings(case_when.else_result)


def _yield_select_col_instructional_strings(col: SelectCol) -> Iterator[str]:
    """Yield strings from a SELECT column expression."""
    yield from _yield_normalized_expr_instructional_strings(col.expr)


def _yield_window_registry_step_instructional_strings(step: WindowRegistryStep) -> Iterator[str]:
    """Yield strings from a window registry row (id/label and nested expressions)."""
    yield step.registry_id
    yield from _yield_window_spec_instructional_strings(step.window_spec)


def _yield_case_registry_step_instructional_strings(step: CaseRegistryStep) -> Iterator[str]:
    """Yield strings from a case registry row (id/label and CASE body)."""
    yield step.registry_id
    yield step.label
    yield from _yield_case_when_instructional_strings(step.case_when)


def _yield_runtime_cte_step_instructional_strings(step: RuntimeCteStep) -> Iterator[str]:
    """Yield strings from one CTE step relevant to instructional placeholders."""
    yield step.cte_name
    yield step.description
    yield from step.tables or []
    for col in step.select_cols or []:
        yield from _yield_select_col_instructional_strings(col)
    for gb in step.group_by_cols or []:
        yield from _yield_normalized_expr_instructional_strings(gb)
    for ob in step.order_by_cols or []:
        yield from _yield_normalized_expr_instructional_strings(ob.expr)
    for fp in where_leaves(step.where) or []:
        yield from _yield_where_instructional_strings(fp)
    for hp in having_leaves(step.having) or []:
        yield from _yield_having_instructional_strings(hp)
    for w in step.window_registry or []:
        yield from _yield_window_registry_step_instructional_strings(w)
    for c in step.case_registry or []:
        yield from _yield_case_registry_step_instructional_strings(c)
    for val in (step.param_values or {}).values():
        yield from _yield_param_value_scan_strings(val)
    yield from step.output_columns or []
    yield from step.chosen_join_path_signature or []


def _yield_runtime_intent_instructional_scan_strings(intent: RuntimeIntent) -> Iterator[str]:
    """Yield structured intent strings to scan for instructional placeholders."""
    yield from intent.tables or []
    for col in intent.select_cols or []:
        yield from _yield_select_col_instructional_strings(col)
    for gb in intent.group_by_cols or []:
        yield from _yield_normalized_expr_instructional_strings(gb)
    for ob in intent.order_by_cols or []:
        yield from _yield_normalized_expr_instructional_strings(ob.expr)
    for fp in where_leaves(intent.where) or []:
        yield from _yield_where_instructional_strings(fp)
    for hp in having_leaves(intent.having) or []:
        yield from _yield_having_instructional_strings(hp)
    for w in intent.window_registry or []:
        yield from _yield_window_registry_step_instructional_strings(w)
    for c in intent.case_registry or []:
        yield from _yield_case_registry_step_instructional_strings(c)
    for val in (intent.param_values or {}).values():
        yield from _yield_param_value_scan_strings(val)
    for step in intent.cte_steps or []:
        yield from _yield_runtime_cte_step_instructional_strings(step)
    for key, val in (intent.column_map or {}).items():
        yield key
        yield val
    yield from intent.chosen_join_path_signature or []
    if intent.limit_param_key:
        yield intent.limit_param_key


def runtime_intent_has_instructional_placeholders(intent: RuntimeIntent) -> bool:
    """Return True when any structured field still uses instructional placeholder tokens."""
    return any(
        intent_text_has_leakable_placeholder(s) for s in _yield_runtime_intent_instructional_scan_strings(intent)
    )


def _strip_intent_placeholder_angle_brackets(text: str) -> str:
    """Remove angle brackets around known instructional placeholder names."""
    return INTENT_PLACEHOLDER_ANGLE_RE.sub(r"\1", text)


def _intent_placeholder_table_alias_map(intent_tables: list[str]) -> dict[str, str]:
    """Map ``table_N`` / ``tableN`` tokens to real tables by sorted order."""
    if not intent_tables:
        return {}
    ordered = sorted(intent_tables)
    out: dict[str, str] = {}
    for i, t in enumerate(ordered, start=1):
        out[f"table_{i}"] = t
        out[f"table{i}"] = t
    return out


def _apply_intent_placeholder_table_rewrites(text: str, alias_map: dict[str, str]) -> str:
    """Rewrite ``table_N.`` (or trailing ``table_N``) using *alias_map*."""
    out = text
    for fake in sorted(alias_map.keys(), key=len, reverse=True):
        real = alias_map[fake]
        out = re.sub(rf"\b{re.escape(fake)}\b(\.|$)", rf"{real}\1", out)
    return out


def _rewrite_intent_placeholder_term(term: str, alias_map: dict[str, str]) -> str:
    """Strip brackets and rewrite table-alias tokens in one multiply/divide term."""
    s = _strip_intent_placeholder_angle_brackets(term.strip())
    if alias_map:
        s = _apply_intent_placeholder_table_rewrites(s, alias_map)
    return s


def _intent_expr_terms_blob(expr: NormalizedExpr) -> str:
    """Join leaf column refs (and raw_sql blobs) for a cheap placeholder scan."""
    parts: list[str] = list(extract_columns_from_expr(expr))

    def _collect_raw(node: NormalizedExpr) -> None:
        if node.raw_sql:
            parts.append(node.raw_sql)
        for grp in node.add_groups + node.sub_groups:
            for ch in grp.multiply + grp.divide:
                _collect_raw(ch)

    _collect_raw(expr)
    return " ".join(parts)


def _repair_intent_placeholder_normalized_expr(expr: NormalizedExpr, alias_map: dict[str, str]) -> NormalizedExpr:
    """Rewrite placeholder table tokens inside a ``NormalizedExpr``."""
    if not alias_map and not INTENT_PLACEHOLDER_ANGLE_RE.search(_intent_expr_terms_blob(expr)):
        return expr

    def repl(term: str) -> str:
        return _rewrite_intent_placeholder_term(term, alias_map)

    return replace_refs_in_expr(expr, repl)


def _repair_intent_placeholder_filters(params: list[WhereParam], alias_map: dict[str, str]) -> list[WhereParam]:
    """Repair filter left/right expressions for placeholder leaks."""
    out: list[WhereParam] = []
    for fp in params:
        le = _repair_intent_placeholder_normalized_expr(fp.left_expr, alias_map)
        rexp = _repair_intent_placeholder_normalized_expr(fp.right_expr, alias_map) if fp.right_expr else None
        out.append(replace(fp, left_expr=le, right_expr=rexp))
    return out


def _repair_intent_placeholder_having(params: list[HavingParam], alias_map: dict[str, str]) -> list[HavingParam]:
    """Repair HAVING left/right expressions for placeholder leaks."""
    out: list[HavingParam] = []
    for hp in params:
        le = _repair_intent_placeholder_normalized_expr(hp.left_expr, alias_map)
        rexp = _repair_intent_placeholder_normalized_expr(hp.right_expr, alias_map) if hp.right_expr else None
        out.append(replace(hp, left_expr=le, right_expr=rexp))
    return out


def _repair_intent_placeholder_window_registry_step(
    step: WindowRegistryStep, alias_map: dict[str, str]
) -> WindowRegistryStep:
    """Repair window spec inside one window registry row."""
    return replace(step, window_spec=_repair_intent_placeholder_window_spec(step.window_spec, alias_map))


def _repair_intent_placeholder_case_registry_step(
    step: CaseRegistryStep, alias_map: dict[str, str]
) -> CaseRegistryStep:
    """Repair CASE body inside one case registry row."""
    return replace(step, case_when=_repair_intent_placeholder_case_when(step.case_when, alias_map))


def _repair_intent_placeholder_window_spec(ws: WindowSpec, alias_map: dict[str, str]) -> WindowSpec:
    """Repair window partition, order, and argument expressions."""
    pb = [_repair_intent_placeholder_normalized_expr(e, alias_map) for e in ws.partition_by]
    ob = [replace(o, expr=_repair_intent_placeholder_normalized_expr(o.expr, alias_map)) for o in ws.order_by]
    arg = _repair_intent_placeholder_normalized_expr(ws.argument, alias_map) if ws.argument else None
    return replace(ws, partition_by=pb, order_by=ob, argument=arg)


def _repair_intent_placeholder_case_when(cw: CaseWhenExpr, alias_map: dict[str, str]) -> CaseWhenExpr:
    """Repair CASE branches and else for placeholder leaks."""
    branches: list[CaseWhenBranch] = []
    for br in cw.branches:
        cond = _repair_intent_placeholder_filters([br.condition], alias_map)[0]
        res = _repair_intent_placeholder_normalized_expr(br.result, alias_map)
        branches.append(CaseWhenBranch(condition=cond, result=res))
    er = _repair_intent_placeholder_normalized_expr(cw.else_result, alias_map) if cw.else_result else None
    return CaseWhenExpr(branches=branches, else_result=er)


def _repair_intent_placeholder_select_cols(cols: list[SelectCol], alias_map: dict[str, str]) -> list[SelectCol]:
    """Repair select list expressions."""
    out: list[SelectCol] = []
    for sc in cols:
        ex = _repair_intent_placeholder_normalized_expr(sc.expr, alias_map)
        out.append(replace(sc, expr=ex))
    return out


def _repair_intent_placeholder_order_by_cols(cols: list[OrderByCol], alias_map: dict[str, str]) -> list[OrderByCol]:
    """Repair ORDER BY expressions for placeholder leaks."""
    return [replace(obc, expr=_repair_intent_placeholder_normalized_expr(obc.expr, alias_map)) for obc in cols]


def _repair_intent_placeholder_cte_step(step: RuntimeCteStep, alias_map: dict[str, str]) -> RuntimeCteStep:
    """Repair one CTE step: selects, group/order, filters, having."""
    return replace(
        step,
        select_cols=_repair_intent_placeholder_select_cols(step.select_cols or [], alias_map),
        group_by_cols=[_repair_intent_placeholder_normalized_expr(g, alias_map) for g in (step.group_by_cols or [])],
        order_by_cols=_repair_intent_placeholder_order_by_cols(step.order_by_cols or [], alias_map),
        where=map_predicate_group(step.where, lambda fp: _repair_intent_placeholder_filters([fp], alias_map)[0]),
        having=map_predicate_group(step.having, lambda hp: _repair_intent_placeholder_having([hp], alias_map)[0]),
        window_registry=[
            _repair_intent_placeholder_window_registry_step(w, alias_map) for w in (step.window_registry or [])
        ],
        case_registry=[_repair_intent_placeholder_case_registry_step(c, alias_map) for c in (step.case_registry or [])],
    )


def repair_intent_placeholder_tokens(intent: RuntimeIntent, _schema_graph: SchemaGraph) -> RuntimeIntent:
    """Rewrite ``table_N``-style leaks using ``intent.tables`` sort. order."""
    tables = list(intent.tables or [])
    if not tables:
        return intent
    alias_map_main = _intent_placeholder_table_alias_map(tables)
    sel = _repair_intent_placeholder_select_cols(intent.select_cols or [], alias_map_main)
    gb = [_repair_intent_placeholder_normalized_expr(g, alias_map_main) for g in (intent.group_by_cols or [])]
    ob = _repair_intent_placeholder_order_by_cols(intent.order_by_cols or [], alias_map_main)
    ctes = []
    for c in intent.cte_steps or []:
        c_tables = list(c.tables or [])
        alias_map_cte = _intent_placeholder_table_alias_map(c_tables)
        ctes.append(_repair_intent_placeholder_cte_step(c, alias_map_cte))
    wr = [_repair_intent_placeholder_window_registry_step(w, alias_map_main) for w in (intent.window_registry or [])]
    cr = [_repair_intent_placeholder_case_registry_step(c, alias_map_main) for c in (intent.case_registry or [])]
    return replace(
        intent,
        select_cols=sel,
        group_by_cols=gb,
        order_by_cols=ob,
        where=map_predicate_group(intent.where, lambda fp: _repair_intent_placeholder_filters([fp], alias_map_main)[0]),
        having=map_predicate_group(
            intent.having, lambda hp: _repair_intent_placeholder_having([hp], alias_map_main)[0]
        ),
        cte_steps=ctes,
        window_registry=wr,
        case_registry=cr,
    )


def _flip_comparison_op(op: str) -> str:
    """Return the comparison operator to use after swapping left and right operands."""
    return OP_FLIP.get(op, op)


def _having_candidate_passes_numeric_and_group_rules(hp: HavingParam, *, group_by_cols: list[Any] | None) -> bool:
    """Return True when *hp* satisfies HAVING operator and GROUP BY presence rules."""
    if validate_having_operator_is_numeric([hp], "auto_repair"):
        return False
    if validate_having_requires_aggregation([hp], "auto_repair", group_by_cols=group_by_cols or []):
        return False
    return True


def auto_repair_where_having(
    where_params: list[WhereParam], having_param: list[HavingParam], *, group_by_cols: list[Any] | None = None
) -> tuple[list[WhereParam], list[HavingParam]]:
    """Repair misplaced filter and HAVING conditions by moving or flipping them. Filters whose ``left_expr`` contains an aggregation are promoted to HAVING only when the candidate HAVING row uses a numeric comparison operator and GROUP BY is present. HAVING rows whose aggregation is on the right are flipped so the aggregation appears on the left. HAVING rows with no aggregation on either side are demoted to filters."""
    repaired_filters: list[WhereParam] = []
    repaired_having: list[HavingParam] = []
    for fp in where_params or []:
        if fp.left_expr.has_aggregation:
            cand = HavingParam(
                left_expr=fp.left_expr,
                op=fp.op,
                right_expr=fp.right_expr,
                value_type=fp.value_type,
                param_key=fp.param_key,
                raw_value=fp.raw_value,
            )
            if _having_candidate_passes_numeric_and_group_rules(cand, group_by_cols=group_by_cols):
                repaired_having.append(cand)
                debug(f"[intent_repair.auto_repair_where_having] filter->having: {fp.param_key}")
            else:
                repaired_filters.append(fp)
        else:
            repaired_filters.append(fp)
    for hp in having_param or []:
        if hp.left_expr.has_aggregation:
            repaired_having.append(hp)
        elif hp.right_expr and hp.right_expr.has_aggregation:
            repaired_having.append(
                HavingParam(
                    left_expr=hp.right_expr,
                    op=_flip_comparison_op(hp.op),
                    right_expr=hp.left_expr,
                    value_type=hp.value_type,
                    param_key=hp.param_key,
                    raw_value=hp.raw_value,
                )
            )
            debug(f"[intent_repair.auto_repair_where_having] having flip (agg->left): {hp.param_key}")
        else:
            repaired_filters.append(
                WhereParam(
                    left_expr=hp.left_expr,
                    op=hp.op,
                    right_expr=hp.right_expr,
                    value_type=hp.value_type,
                    param_key=hp.param_key,
                    raw_value=hp.raw_value,
                )
            )
            debug(f"[intent_repair.auto_repair_where_having] having->filter: {hp.param_key}")
    return repaired_filters, repaired_having


def _split_qualified_ref(ref: str) -> tuple[str | None, str]:
    """Split ``table.column`` into ``(table, column)``; return ``(None, ref)`` when bare."""
    s = (ref or "").strip()
    if "." in s:
        head, _, tail = s.rpartition(".")
        return (head.strip().lower() or None, tail.strip().lower())
    return (None, s.lower())


def _normalized_expr_term_strings(expr: NormalizedExpr) -> list[str]:
    """Collect leaf column-ref strings across all nested groups."""
    return list(extract_columns_from_expr(expr))


def _term_matches_column(term: str, table: str | None, column: str) -> bool:
    """Return True when *term* is a bare or table-qualified reference to *column*."""
    s = (term or "").strip().lower()
    if not s or s == "*":
        return False
    if "(" in s:
        return False
    if "." in s:
        head, _, tail = s.rpartition(".")
        if tail.strip() != column:
            return False
        if table is None:
            return True
        return head.strip() == table
    if table is not None:
        return False
    return s == column


def _build_column_term_replacer(
    src_table: str | None, src_column: str, dst_table: str | None, dst_column: str
) -> Callable[[str], str]:
    """Return a term-level replacer that swaps matching column references."""

    def repl(term: str) -> str:
        if not _term_matches_column(term, src_table, src_column):
            return term
        if dst_table is not None:
            return f"{dst_table}.{dst_column}"
        return dst_column

    return repl


def _build_table_term_replacer(src_table: str, dst_table: str) -> Callable[[str], str]:
    """Return a term-level replacer that retargets ``src_table.col`` references to ``dst_table.col``."""
    src = src_table.strip().lower()
    dst = dst_table.strip().lower()

    def repl(term: str) -> str:
        s = (term or "").strip()
        if "." not in s or "(" in s:
            return term
        head, _, tail = s.rpartition(".")
        if head.strip().lower() != src:
            return term
        return f"{dst}.{tail.strip().lower()}"

    return repl


def _transform_select_col_expr(sc: SelectCol, transformer: Callable[[NormalizedExpr], NormalizedExpr]) -> SelectCol:
    """Apply *transformer* to a select column's expression."""
    return SelectCol(expr=transformer(sc.expr))


def _transform_order_by_col_expr(oc: OrderByCol, transformer: Callable[[NormalizedExpr], NormalizedExpr]) -> OrderByCol:
    """Apply *transformer* to an order-by column's expression."""
    return OrderByCol(expr=transformer(oc.expr), direction=oc.direction)


def _transform_where_param_expr(fp: WhereParam, transformer: Callable[[NormalizedExpr], NormalizedExpr]) -> WhereParam:
    """Apply *transformer* to both sides of a filter param."""
    return WhereParam(
        left_expr=transformer(fp.left_expr),
        op=fp.op,
        right_expr=transformer(fp.right_expr) if fp.right_expr is not None else None,
        value_type=fp.value_type,
        param_key=fp.param_key,
        param_key_hi=fp.param_key_hi,
        raw_value=fp.raw_value,
    )


def _transform_having_param_expr(
    hp: HavingParam, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> HavingParam:
    """Apply *transformer* to both sides of a having param."""
    return HavingParam(
        left_expr=transformer(hp.left_expr),
        op=hp.op,
        right_expr=transformer(hp.right_expr) if hp.right_expr is not None else None,
        value_type=hp.value_type,
        param_key=hp.param_key,
        raw_value=hp.raw_value,
    )


def _transform_window_spec(ws: WindowSpec, transformer: Callable[[NormalizedExpr], NormalizedExpr]) -> WindowSpec:
    """Apply *transformer* to partition_by, order_by expressions, and optional window argument."""
    new_part = [transformer(p) for p in (ws.partition_by or [])]
    new_orders = [replace(o, expr=transformer(o.expr)) for o in (ws.order_by or [])]
    new_arg = transformer(ws.argument) if ws.argument is not None else None
    return replace(ws, partition_by=new_part, order_by=new_orders, argument=new_arg)


def _transform_window_registry_steps(
    regs: list[WindowRegistryStep] | None, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> list[WindowRegistryStep]:
    """Map *transformer* across every ``WindowRegistryStep.window_spec`` expression subtree."""
    out: list[WindowRegistryStep] = []
    for step in regs or []:
        out.append(replace(step, window_spec=_transform_window_spec(step.window_spec, transformer)))
    return out


def _transform_case_when_expr(
    cw: CaseWhenExpr, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> CaseWhenExpr:
    """Apply *transformer* to branch conditions, branch results, and ``else_result``."""
    new_branches: list[CaseWhenBranch] = []
    for br in cw.branches or []:
        new_branches.append(
            CaseWhenBranch(
                condition=_transform_where_param_expr(br.condition, transformer), result=transformer(br.result)
            )
        )
    new_else = transformer(cw.else_result) if cw.else_result is not None else None
    return replace(cw, branches=new_branches, else_result=new_else)


def _transform_case_registry_steps(
    regs: list[CaseRegistryStep] | None, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> list[CaseRegistryStep]:
    """Map *transformer* across every CASE registry ``case_when`` subtree."""
    out: list[CaseRegistryStep] = []
    for step in regs or []:
        cw = step.case_when
        new_cw = _transform_case_when_expr(cw, transformer)
        out.append(replace(step, case_when=new_cw))
    return out


def _transform_cte_step_exprs(
    step: RuntimeCteStep, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> RuntimeCteStep:
    """Apply *transformer* across a CTE step's select/group/order/filter/having/registry expressions."""
    return replace(
        step,
        select_cols=[_transform_select_col_expr(sc, transformer) for sc in (step.select_cols or [])],
        group_by_cols=[transformer(g) for g in (step.group_by_cols or [])],
        order_by_cols=[_transform_order_by_col_expr(oc, transformer) for oc in (step.order_by_cols or [])],
        where=map_predicate_group(step.where, lambda fp: _transform_where_param_expr(fp, transformer)),
        having=map_predicate_group(step.having, lambda hp: _transform_having_param_expr(hp, transformer)),
        window_registry=_transform_window_registry_steps(step.window_registry, transformer),
        case_registry=_transform_case_registry_steps(step.case_registry, transformer),
    )


def _walk_intent_normalized_exprs(
    intent: RuntimeIntent, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> RuntimeIntent:
    """Map *transformer* across every NormalizedExpr in *intent* (top- level and CTE steps)."""
    return replace(
        intent,
        select_cols=[_transform_select_col_expr(sc, transformer) for sc in (intent.select_cols or [])],
        group_by_cols=[transformer(g) for g in (intent.group_by_cols or [])],
        order_by_cols=[_transform_order_by_col_expr(oc, transformer) for oc in (intent.order_by_cols or [])],
        where=map_predicate_group(intent.where, lambda fp: _transform_where_param_expr(fp, transformer)),
        having=map_predicate_group(intent.having, lambda hp: _transform_having_param_expr(hp, transformer)),
        window_registry=_transform_window_registry_steps(intent.window_registry, transformer),
        case_registry=_transform_case_registry_steps(intent.case_registry, transformer),
        cte_steps=[_transform_cte_step_exprs(c, transformer) for c in (intent.cte_steps or [])],
    )


def _apply_column_replacer_to_intent(intent: RuntimeIntent, replacer: Callable[[str], str]) -> RuntimeIntent:
    """Apply a multiply/divide term-level *replacer* across every NormalizedExpr in *intent*."""

    def transform(expr: NormalizedExpr) -> NormalizedExpr:
        return replace_refs_in_expr(expr, replacer)

    return _walk_intent_normalized_exprs(intent, transform)


def _intent_columns_for_table(intent: RuntimeIntent, table: str) -> list[str]:
    """Collect bare column names already referenced for *table* in *intent*."""
    table_low = table.strip().lower()
    seen: list[str] = []
    for sc in intent.select_cols or []:
        for ref in extract_columns_from_expr(sc.expr):
            t, c = _split_qualified_ref(ref)
            if t == table_low and c not in seen:
                seen.append(c)
    for g in intent.group_by_cols or []:
        for ref in extract_columns_from_expr(g):
            t, c = _split_qualified_ref(ref)
            if t == table_low and c not in seen:
                seen.append(c)
    for fp in where_leaves(intent.where) or []:
        for ref in extract_columns_from_expr(fp.left_expr):
            t, c = _split_qualified_ref(ref)
            if t == table_low and c not in seen:
                seen.append(c)
    return seen


def _table_columns_from_schema(
    schema: SchemaGraph,
    table: str,
    *,
    source_scope: frozenset[str] | None = None,
) -> list[str]:
    """Return lowercase column names for *table* in *schema*; empty when missing."""
    table_low = table.strip().lower()
    meta = schema.tables.get(table_low) if schema and schema.tables else None
    if meta is None:
        meta = schema.tables.get(table) if schema and schema.tables else None
    if meta is None:
        return []
    if not source_scope or not meta.column_member_sources:
        return [c.strip().lower() for c in meta.columns.keys()]
    scoped: list[str] = []
    for col in meta.columns:
        col_low = col.strip().lower()
        holders = meta.column_member_sources.get(col) or meta.column_member_sources.get(col_low) or []
        if not holders or source_scope.intersection(holders):
            scoped.append(col_low)
    return scoped


def _fuzzy_pick(
    target: str,
    candidates: Sequence[str],
    *,
    source_scope: set[str] | frozenset[str] | None = None,
    table_source: Callable[[str], str] | None = None,
) -> str | None:
    """Return the closest match in *candidates* to *target* using ratio cutoff, or None."""
    pool = [c for c in candidates if c]
    if source_scope and table_source is not None:
        scoped = [candidate for candidate in pool if table_source(candidate) in source_scope]
        if scoped:
            pool = scoped
    if not pool:
        return None
    matches = get_close_matches(target.strip().lower(), pool, n=1, cutoff=DIAGNOSTIC_FUZZY_CUTOFF)
    return matches[0] if matches else None


def _repair_unknown_column(intent: RuntimeIntent, schema: SchemaGraph, diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Rewrite an unknown column to its closest schema-known sibling on the same table."""
    raw = (diag.offending_identifier or "").strip()
    if not raw:
        return None
    src_table, src_column = _split_qualified_ref(raw)
    if not src_column:
        return None
    source_scope = _repair_source_scope(schema, intent, diag)
    column_scope = source_scope or None
    if src_table is not None:
        candidates = _table_columns_from_schema(schema, src_table, source_scope=column_scope)
        best = _fuzzy_pick(src_column, candidates)
        if best is None or best == src_column:
            return None
        replacer = _build_column_term_replacer(src_table, src_column, src_table, best)
        debug(f"[intent_repair._repair_unknown_column] {src_table}.{src_column} -> {src_table}.{best}")
        return _apply_column_replacer_to_intent(intent, replacer)
    best_table: str | None = None
    best_column: str | None = None
    for t in intent.tables or []:
        cands = _table_columns_from_schema(schema, t, source_scope=column_scope)
        pick = _fuzzy_pick(src_column, cands)
        if pick is not None and pick != src_column:
            best_table = t.strip().lower()
            best_column = pick
            break
    if best_table is None or best_column is None:
        return None
    replacer = _build_column_term_replacer(None, src_column, best_table, best_column)
    debug(f"[intent_repair._repair_unknown_column] {src_column} -> {best_table}.{best_column}")
    return _apply_column_replacer_to_intent(intent, replacer)


def _repair_ambiguous_column(intent: RuntimeIntent, schema: SchemaGraph, diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Qualify an ambiguous bare column when a single owner table can be resolved."""
    raw = (diag.offending_identifier or "").strip().lower()
    if not raw or "." in raw:
        return None
    owners_csv = (diag.details.get("owners") or "").strip()
    owners = [o.strip().lower() for o in owners_csv.split(",") if o.strip()] if owners_csv else []
    if not owners:
        owners = [t.strip().lower() for t in (intent.tables or []) if raw in _table_columns_from_schema(schema, t)]
    if not owners:
        return None
    detail_source = (diag.details.get("source_id") or "").strip()
    if detail_source:
        source_owners = [owner for owner in owners if _federation_table_source(schema, owner, None) == detail_source]
        if source_owners:
            owners = source_owners
        elif owners:
            return None
    intent_scope = _intent_source_scope(schema, intent) or _single_source_scope_from_schema(schema)
    if intent_scope:
        scoped_owners = [owner for owner in owners if _federation_table_source(schema, owner, None) in intent_scope]
        if scoped_owners:
            owners = scoped_owners
        elif owners:
            return None
    source_ids = {
        _federation_table_source(schema, owner, None)
        for owner in owners
        if _federation_table_source(schema, owner, None)
    }
    if len(source_ids) > 1:
        return None
    intent_tables = [t.strip().lower() for t in (intent.tables or [])]
    chosen: str | None = None
    for t in intent_tables:
        if t in owners:
            chosen = t
            break
    if chosen is None:
        return None
    replacer = _build_column_term_replacer(None, raw, chosen, raw)
    debug(f"[intent_repair._repair_ambiguous_column] {raw} -> {chosen}.{raw}")
    return _apply_column_replacer_to_intent(intent, replacer)


def _repair_unknown_table(intent: RuntimeIntent, schema: SchemaGraph, diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Rewrite an unknown table to its closest schema-known sibling and retarget column refs."""
    raw = (diag.offending_identifier or "").strip().lower()
    if not raw:
        return None
    candidates = list(schema.tables.keys()) if schema and schema.tables else []
    source_scope = _intent_source_scope(schema, intent)
    table_source = (
        (lambda table_name: _federation_table_source(schema, table_name, None) or "") if source_scope else None
    )
    best = _fuzzy_pick(raw, candidates, source_scope=source_scope, table_source=table_source)
    if best is None or best == raw:
        return None
    new_tables = [best if (t.strip().lower() == raw) else t for t in (intent.tables or [])]
    if new_tables == list(intent.tables or []):
        return None
    replacer = _build_table_term_replacer(raw, best)
    rewritten = _apply_column_replacer_to_intent(intent, replacer)
    debug(f"[intent_repair._repair_unknown_table] {raw} -> {best}")
    return replace(rewritten, tables=new_tables)


def _repair_grain_consistency(intent: RuntimeIntent, _schema: SchemaGraph, diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Add the offending non-grouped select column to ``group_by_cols`` when missing."""
    raw = (diag.offending_identifier or "").strip()
    if not raw:
        return None
    new_expr = NormalizedExpr.from_column(raw)
    new_key = expr_canonical_key(new_expr)
    existing = intent.group_by_cols or []
    for g in existing:
        if expr_canonical_key(g) == new_key:
            return None
    debug(f"[intent_repair._repair_grain_consistency] add to GROUP BY: {raw}")
    return replace(intent, group_by_cols=[*existing, new_expr])


def _repair_agg_in_where(intent: RuntimeIntent, _schema: SchemaGraph, _diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Promote any aggregation-bearing WHERE filter into HAVING via :func:`auto_repair_where_having`."""
    new_filters, new_having = auto_repair_where_having(
        where_leaves(intent.where) or [], having_leaves(intent.having) or [], group_by_cols=intent.group_by_cols or []
    )
    if new_filters == list(where_leaves(intent.where) or []) and new_having == list(having_leaves(intent.having) or []):
        return None
    debug("[intent_repair._repair_agg_in_where] filter->having promotion applied")
    return replace(intent, where=predicate_group_from_list(new_filters), having=predicate_group_from_list(new_having))


def _repair_cartesian(intent: RuntimeIntent, _schema: SchemaGraph, _diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Clear the chosen join candidate so the next render re-selects an explicit join path."""
    if not intent.chosen_join_candidate_id and not intent.chosen_join_path_signature:
        return None
    debug("[intent_repair._repair_cartesian] clearing chosen_join_candidate_id and signature")
    return replace(intent, chosen_join_candidate_id="", chosen_join_path_signature=[])


def _repair_where_overlap(intent: RuntimeIntent, _schema: SchemaGraph, _diag: SqlDiagnostic) -> RuntimeIntent | None:
    """De-duplicate contradictory filters; return new intent only when something changed."""
    repaired = dedup_contradictory_where(intent)
    if repaired is intent:
        return None
    return repaired


def _repair_param_binding(intent: RuntimeIntent, _schema: SchemaGraph, diag: SqlDiagnostic) -> RuntimeIntent | None:
    """Drop a filter that references an unbound parameter when it has no literal raw_value."""
    target = (diag.offending_identifier or "").strip()
    if not target:
        return None
    target_low = target.lstrip(":").lower()
    new_filters: list[WhereParam] = []
    dropped = False
    for fp in where_leaves(intent.where) or []:
        keys = {(fp.param_key or "").lower(), (fp.param_key_hi or "").lower()}
        if target_low in keys and fp.raw_value is None:
            dropped = True
            continue
        new_filters.append(fp)
    if not dropped:
        return None
    debug(f"[intent_repair._repair_param_binding] dropped unbound-param filter: {target_low}")
    return replace(intent, where=predicate_group_from_list(new_filters))


DIAGNOSTIC_REPAIR_DISPATCH: dict[
    SqlDiagnosticCode,
    Callable[[RuntimeIntent, SchemaGraph, SqlDiagnostic], RuntimeIntent | None],
] = {
    SqlDiagnosticCode(code_wire): handler
    for code_wire, handler in (
        ("unknown_column", _repair_unknown_column),
        ("ambiguous_column", _repair_ambiguous_column),
        ("unknown_table", _repair_unknown_table),
        ("non_grouped_select_col", _repair_grain_consistency),
        ("agg_in_where", _repair_agg_in_where),
        ("explain_cartesian_join", _repair_cartesian),
        ("explain_zero_estimate", _repair_where_overlap),
        ("param_unbound", _repair_param_binding),
    )
}


def apply_diagnostic_repairs(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    diagnostics: Sequence[SqlDiagnostic],
    *,
    max_attempts_per_code: int = MAX_REPAIR_ATTEMPTS_PER_CODE,
) -> tuple[RuntimeIntent, bool]:
    """Apply structural repairs for each actionable diagnostic; cap attempts per code. Soft diagnostics (``SOFT_DIAGNOSTIC_CODES``) are skipped because they convey EXPLAIN-plan hints rather than structural defects. Returns the rewritten intent and a flag indicating whether any repair primitive returned a non-``None`` result."""
    attempts: dict[SqlDiagnosticCode, int] = {}
    current = intent
    changed = False
    for diag in diagnostics or []:
        if diag.code.value in SOFT_DIAGNOSTIC_CODES:
            continue
        repair = DIAGNOSTIC_REPAIR_DISPATCH.get(diag.code)
        if repair is None:
            continue
        if attempts.get(diag.code, 0) >= max_attempts_per_code:
            continue
        attempts[diag.code] = attempts.get(diag.code, 0) + 1
        result = repair(current, schema, diag)
        if result is not None:
            current = result
            changed = True
    return current, changed


def decompose_in_not_in_where(intent: RuntimeIntent) -> RuntimeIntent:
    """Apply ``_decompose_in_list`` to main and each CTE (filter-only; HAVING is out of scope)."""
    main_where = _decompose_in_list(where_leaves(intent.where) or [])
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        decomposed = _decompose_in_list(where_leaves(cte.where) or [])
        new_ctes.append(replace(cte, where=decomposed))
    out = replace(intent, where=main_where, cte_steps=new_ctes or intent.cte_steps)
    expanded = len((main_where.leaves() if main_where else []) or []) != len(where_leaves(intent.where) or [])
    if not expanded:
        for oc, nc in zip(intent.cte_steps or [], new_ctes, strict=True):
            if len(where_leaves(oc.where) or []) != len(where_leaves(nc.where) or []):
                expanded = True
                break
    if expanded:
        pipeline_trace(
            "intent_after_deterministic_repair.decompose_in_filters",
            stable_json(
                {
                    "main_filters": len(where_leaves(out.where) or []),
                    "cte_steps": len(out.cte_steps or []),
                }
            ),
        )
    return out
