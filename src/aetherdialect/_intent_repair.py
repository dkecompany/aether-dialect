"""Structural intent repairs and deterministic filter/select normalization."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from difflib import get_close_matches
from typing import Any

from ._config import (
    ARRAY_REWRITABLE_OPS,
    BOOLEAN_FALSY_VALUES,
    BOOLEAN_TRUTHY_VALUES,
    CUMULATIVE_PHRASING_RE,
    DATE_COLUMN_VALUE_TYPES,
    DATE_RESULT_SCALARS,
    DESCRIPTIVE_ALLOWED_VALUE_TYPES,
    DESCRIPTIVE_EXCLUDED_VALUE_TYPES,
    DIAGNOSTIC_CODE_PII_GATE_HIT,
    DIAGNOSTIC_FUZZY_CUTOFF,
    IDENTIFIER_RE,
    IMPOSSIBLE_HAVING_RE,
    INTENT_PLACEHOLDER_ANGLE_RE,
    MAX_REPAIR_ATTEMPTS_PER_CODE,
    NON_NUMERIC_AGGS_FOR_DATES,
    NULL_OP_DOUBLE_NEGATED_ALIASES,
    NULL_OP_NEGATED_ALIASES,
    NULL_OP_PLAIN_ALIASES,
    NUMERIC_DATA_TYPES,
    NUMERIC_RESULT_AGGS,
    NUMERIC_RESULT_SCALARS,
    OP_FLIP,
    RANGE_OPS,
    REGISTRY_TOKEN_PATTERN,
    SOFT_DIAGNOSTIC_CODES,
    SQL_KEYWORDS,
    STRING_COLUMN_VALUE_TYPES,
    STRING_OPS,
    UNKNOWN_DATEPART_TO_EXTRACT_UNIT,
    WINDOW_AGG_FUNCTIONS,
)
from ._contracts_base import (
    ColumnMetadata,
    FailureCategory,
    SchemaGraph,
    SqlDiagnostic,
    SqlDiagnosticCode,
    TableMetadata,
)
from ._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
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
from ._core_utils import debug, notify, pipeline_trace_lazy, stable_json
from ._intent_expr import (
    expr_canonical_key,
    extract_columns_from_expr,
    replace_refs_in_expr,
)
from ._sql_gen import render_expr_sql
from ._validation_schema import (
    filter_param_to_having_param,
    selectability_exempt_qualified_refs,
)
from ._validation_semantic import (
    validate_having_operator_is_numeric,
    validate_having_requires_aggregation,
)


def _apply_filters_to_main_and_ctes(
    intent: RuntimeIntent,
    process_fn: Callable[[list[FilterParam]], tuple[list[FilterParam], bool]],
) -> RuntimeIntent:
    """
    Apply a filter processor to the main intent and each CTE, merging results.

    Also extends the processor to every CASE WHEN branch whose ``condition_scope`` is ``"filter"`` so that branch-shaped predicates receive identical repairs as flat ``filters_param`` entries. A processor that returns zero or multiple predicates for a single-element branch input keeps the original branch condition because a CASE branch holds exactly one predicate.
    """
    new_fp, main_changed = process_fn(intent.filters_param or [])
    if not intent.cte_steps:
        result = replace(intent, filters_param=new_fp) if main_changed else intent
        return _apply_filter_processor_to_case_branches(result, process_fn)
    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps:
        cte_fp, c = process_fn(cte.filters_param or [])
        if c:
            cte_changed = True
        new_cte_steps.append(replace(cte, filters_param=cte_fp))
    if not main_changed and not cte_changed:
        return _apply_filter_processor_to_case_branches(intent, process_fn)
    result = replace(intent, filters_param=new_fp)
    if cte_changed:
        result = replace(result, cte_steps=new_cte_steps)
    return _apply_filter_processor_to_case_branches(result, process_fn)


def _apply_having_to_main_and_ctes(
    intent: RuntimeIntent,
    process_fn: Callable[[list[HavingParam]], tuple[list[HavingParam], bool]],
) -> RuntimeIntent:
    """
    Apply a HAVING processor to the main intent and each CTE, merging results.

    Also extends the processor to every CASE WHEN branch whose ``condition_scope`` is ``"having"``. The branch condition is wrapped as a one-element ``HavingParam`` list via :func:`filter_param_to_having_param`, processed, and converted back via :func:`_having_param_to_filter_param`. A processor that returns zero or multiple predicates keeps the original branch because a CASE branch holds exactly one.
    """
    new_hp, main_changed = process_fn(intent.having_param or [])
    if not intent.cte_steps:
        result = replace(intent, having_param=new_hp) if main_changed else intent
        return _apply_having_processor_to_case_branches(result, process_fn)
    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps:
        cte_hp, c = process_fn(cte.having_param or [])
        if c:
            cte_changed = True
        new_cte_steps.append(replace(cte, having_param=cte_hp))
    if not main_changed and not cte_changed:
        return _apply_having_processor_to_case_branches(intent, process_fn)
    result = replace(intent, having_param=new_hp)
    if cte_changed:
        result = replace(result, cte_steps=new_cte_steps)
    return _apply_having_processor_to_case_branches(result, process_fn)


def _having_param_to_filter_param(hp: HavingParam) -> FilterParam:
    """Translate a :class:`HavingParam` back into the matching :class:`FilterParam`."""
    return FilterParam(
        left_expr=hp.left_expr,
        op=hp.op,
        right_expr=hp.right_expr,
        value_type=hp.value_type,
        param_key=hp.param_key,
        raw_value=hp.raw_value,
        bool_op=hp.bool_op,
        filter_group=hp.filter_group,
    )


def _apply_filter_processor_to_case_branches(
    intent: RuntimeIntent,
    process_fn: Callable[[list[FilterParam]], tuple[list[FilterParam], bool]],
) -> RuntimeIntent:
    """Run *process_fn* against every filter-scope CASE branch via :func:`map_case_branch_conditions`."""

    def _branch_transform(conds: list[FilterParam]) -> list[FilterParam]:
        new_list, _ = process_fn(conds)
        return new_list

    return map_case_branch_conditions(intent, _branch_transform, scopes=frozenset({"filter"}))


def _apply_having_processor_to_case_branches(
    intent: RuntimeIntent,
    process_fn: Callable[[list[HavingParam]], tuple[list[HavingParam], bool]],
) -> RuntimeIntent:
    """Run *process_fn* against every having-scope CASE branch via :func:`map_case_branch_conditions`."""

    def _branch_transform(conds: list[FilterParam]) -> list[FilterParam]:
        h_in = [filter_param_to_having_param(c) for c in conds]
        new_h, _ = process_fn(h_in)
        return [_having_param_to_filter_param(h) for h in new_h]

    return map_case_branch_conditions(intent, _branch_transform, scopes=frozenset({"having"}))


CaseBranchTransform = Callable[[list[FilterParam]], list[FilterParam]]


def _walk_case_when_branches(
    case_when: CaseWhenExpr,
    transform: CaseBranchTransform,
    scopes: frozenset[str],
    location: str,
) -> tuple[CaseWhenExpr, bool]:
    """Apply *transform* to each branch condition whose scope is in *scopes*; return updated CASE and changed flag."""
    if not case_when or not case_when.branches:
        return case_when, False
    branch_scope = (case_when.condition_scope or "filter").strip().lower()
    if branch_scope not in scopes:
        return case_when, False
    new_branches: list[CaseWhenBranch] = []
    changed = False
    for bi, branch in enumerate(case_when.branches):
        cond = branch.condition
        if cond is None:
            new_branches.append(branch)
            continue
        produced = transform([cond])
        if not produced:
            debug(
                f"[intent_repair._walk_case_when_branches] {location}.branches[{bi}]: transform returned empty; keeping original",
            )
            new_branches.append(branch)
            continue
        if len(produced) > 1:
            debug(
                f"[intent_repair._walk_case_when_branches] {location}.branches[{bi}]: transform expanded to {len(produced)} predicates; CASE branch only accepts one — keeping first",
            )
        new_cond = produced[0]
        if new_cond is cond:
            new_branches.append(branch)
            continue
        new_branches.append(replace(branch, condition=new_cond))
        changed = True
    if not changed:
        return case_when, False
    return replace(case_when, branches=new_branches), True


def _walk_case_registry(
    registry: Sequence[CaseRegistryStep] | None,
    transform: CaseBranchTransform,
    scopes: frozenset[str],
    location: str,
) -> tuple[list[CaseRegistryStep] | None, bool]:
    """Apply transform to every registered CASE; return new registry list and changed flag."""
    if not registry:
        return list(registry) if registry is not None else None, False
    new_steps: list[CaseRegistryStep] = []
    changed = False
    for step in registry:
        new_cw, c = _walk_case_when_branches(
            step.case_when,
            transform,
            scopes,
            f"{location}.case_registry[{step.registry_id}]",
        )
        if c:
            new_steps.append(replace(step, case_when=new_cw))
            changed = True
        else:
            new_steps.append(step)
    return new_steps, changed


def map_case_branch_conditions(
    intent: RuntimeIntent,
    transform: CaseBranchTransform,
    *,
    scopes: frozenset[str] = frozenset({"filter", "having"}),
) -> RuntimeIntent:
    """
    Apply *transform* to every CASE WHEN branch condition in the intent.

    The transform receives a single-element ``list[FilterParam]`` (the branch condition wrapped) and must return a list of zero or more :class:`FilterParam`. Because a CASE branch holds exactly one condition, only the first returned predicate is used; an empty result keeps the original. Walks ``case_registry[*].case_when`` for both the main intent and every ``cte_steps[*]``. The *scopes* argument restricts processing to branches whose ``CaseWhenExpr.condition_scope`` is in the set (``"filter"`` and/or ``"having"``).
    """
    new_main_registry, mr_changed = _walk_case_registry(
        intent.case_registry,
        transform,
        scopes,
        "main",
    )

    new_cte_steps: list[RuntimeCteStep] = []
    cte_any_changed = False
    for ci, cte in enumerate(intent.cte_steps or []):
        new_cte_registry, cr_changed = _walk_case_registry(
            cte.case_registry,
            transform,
            scopes,
            f"cte_steps[{ci}]",
        )
        if cr_changed:
            cte_any_changed = True
            new_cte_steps.append(replace(cte, case_registry=new_cte_registry))
        else:
            new_cte_steps.append(cte)

    if not mr_changed and not cte_any_changed:
        return intent
    return replace(
        intent,
        case_registry=new_main_registry,
        cte_steps=new_cte_steps,
    )


def _dedup_contradictory_filters_list(
    filters: list[FilterParam],
) -> tuple[list[FilterParam], bool]:
    """
    Drop range filters on a column that also has an equality on that column.

    Args:

        filters: Filter list to deduplicate.

    Returns:

        ``(filters, changed)`` where *changed* is True when any range was removed.
    """
    eq_columns: set[str] = set()
    for fp in filters:
        col = fp.left_expr.primary_column or ""
        if fp.op == "=" and col:
            eq_columns.add(col)

    if not eq_columns:
        return filters, False

    kept: list[FilterParam] = []
    changed = False
    for fp in filters:
        col = fp.left_expr.primary_column or ""
        if col in eq_columns and fp.op in RANGE_OPS:
            debug(f"[intent_repair.dedup_contradictory_filters] dropping {fp.op} on '{col}' that contradicts =")
            changed = True
            continue
        kept.append(fp)
    return kept, changed


def _dedup_contradictory_having_list(
    having: list[HavingParam],
) -> tuple[list[HavingParam], bool]:
    """
    Drop range HAVING predicates on an aggregation expression that also has an equality on that key.

    Keys use :func:`expr_canonical_key` on ``left_expr`` (aggregation expressions lack a single ``primary_column`` like WHERE filters).
    """

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


def dedup_contradictory_filters(intent: RuntimeIntent) -> RuntimeIntent:
    """Remove contradictory range filters and HAVING predicates from main query and CTEs."""
    intent = _apply_filters_to_main_and_ctes(intent, _dedup_contradictory_filters_list)
    return _apply_having_to_main_and_ctes(intent, _dedup_contradictory_having_list)


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


def _filter_right_expr_redundant_with_value(fp: FilterParam) -> bool:
    if fp.right_expr is None or fp.raw_value is None:
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


def _dedup_value_vs_right_expr_filters(
    filters: list[FilterParam],
) -> tuple[list[FilterParam], bool]:
    out: list[FilterParam] = []
    changed = False
    for fp in filters:
        if _filter_right_expr_redundant_with_value(fp):
            out.append(replace(fp, right_expr=None))
            changed = True
        else:
            out.append(fp)
    return out, changed


def _dedup_value_vs_right_expr_havings(
    having: list[HavingParam],
) -> tuple[list[HavingParam], bool]:
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
    intent = _apply_filters_to_main_and_ctes(intent, _dedup_value_vs_right_expr_filters)
    return _apply_having_to_main_and_ctes(intent, _dedup_value_vs_right_expr_havings)


def expand_multi_group_filters(intent: RuntimeIntent) -> RuntimeIntent:
    """
    No-op: ``filter_group`` is a single integer in ``INTENT_SCHEMA``; list expansion was removed.
    """

    return intent


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


def repair_null_equality_filters(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Rewrite ``=`` / ``!=`` / ``<>`` against null into ``is null`` / ``is not null``.

    Args:

        intent: RuntimeIntent whose filters may use equality on null.

    Returns:

        Updated intent for main query and all CTEs, or unchanged when no fix applies.
    """
    intent = _apply_filters_to_main_and_ctes(intent, _repair_null_equality_list)
    return _apply_having_to_main_and_ctes(intent, _repair_null_equality_having_list)


def _repair_null_equality_list(
    filters: list[FilterParam],
) -> tuple[list[FilterParam], bool]:
    repaired: list[FilterParam] = []
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
        if fp.op == "=" and _is_null_value(fp.raw_value):
            repaired.append(
                replace(
                    fp,
                    op="is null",
                    raw_value=None,
                    value_type="null",
                )
            )
            changed = True
        elif fp.op in ("!=", "<>") and _is_null_value(fp.raw_value):
            repaired.append(
                replace(
                    fp,
                    op="is not null",
                    raw_value=None,
                    value_type="null",
                )
            )
            changed = True
        else:
            repaired.append(fp)
    return repaired, changed


def _repair_null_equality_having_list(
    having: list[HavingParam],
) -> tuple[list[HavingParam], bool]:
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
        if hp.op == "=" and _is_null_value(hp.raw_value):
            repaired.append(
                replace(
                    hp,
                    op="is null",
                    raw_value=None,
                    value_type="null",
                )
            )
            changed = True
        elif hp.op in ("!=", "<>") and _is_null_value(hp.raw_value):
            repaired.append(
                replace(
                    hp,
                    op="is not null",
                    raw_value=None,
                    value_type="null",
                )
            )
            changed = True
        else:
            repaired.append(hp)
    return repaired, changed


def infer_cte_output_columns(cte: Any, *, include_agg_prefix: bool = True) -> list[str]:
    """
    Infer CTE output column aliases from ``select_cols`` when ``output_columns`` is empty.

    Args:

        cte: CTE step with ``select_cols`` populated.

        include_agg_prefix: When ``True`` (default), aggregated selects use ``agg_bare`` names
        (for example ``avg_amount``). When ``False``, only the trailing physical column name
        is kept (for example ``rate``), used when fixing scalar CTE bogus ``output_columns``.

    Returns:

        Bare names (with agg prefix when aggregated) for use as output aliases.
    """
    names: list[str] = []
    for sc in cte.select_cols or []:
        col = sc.expr.primary_column if sc.expr else ""
        if not col:
            continue
        bare = col.split(".")[-1].strip().lower()
        agg_fn = (getattr(sc.expr, "agg_func", None) or "").lower()
        if not agg_fn and sc.expr:
            for g in sc.expr.add_groups + sc.expr.sub_groups:
                if g.agg_func:
                    agg_fn = str(g.agg_func).lower()
                    break
        if include_agg_prefix and sc.is_aggregated and agg_fn:
            bare = f"{agg_fn}_{bare}"
        if bare and bare not in names:
            names.append(bare)
    return names


def _is_registry_token(term: str | None) -> bool:
    """
    Return True when *term* is a bare window or case registry token (wNN or cNN).

    Args:

        term: Column reference leaf string.

    Returns:

        True when *term* matches :data:`aetherdialect._config.REGISTRY_TOKEN_PATTERN` with no table qualifier.
    """

    if not term or "." in term:
        return False
    stripped = term.strip().lower()
    return bool(re.fullmatch(REGISTRY_TOKEN_PATTERN, stripped))


def _qualify_term(term: str, output_to_cte: dict[str, str]) -> str:
    """
    Qualify bare CTE output column tokens inside a single column reference string.

    Args:

        term: One column-ref string (or arithmetic fragment).

        output_to_cte: Lowercased bare output column -> owning CTE name.

    Returns:

        Term with matching bare columns rewritten to ``cte.column``.
    """
    if _is_registry_token(term):
        return term
    for col_lower, cte_name in output_to_cte.items():
        pat = re.compile(
            r"(?<!\.)(?<![A-Za-z0-9_])" + re.escape(col_lower) + r"(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        if pat.search(term):
            term = pat.sub(f"{cte_name}.{col_lower}", term)
    return term


def _qualify_expr(expr: NormalizedExpr, output_to_cte: dict[str, str]) -> NormalizedExpr:
    """Apply ``_qualify_term`` to every leaf column_ref reachable from *expr*."""

    return replace_refs_in_expr(expr, lambda ref: _qualify_term(ref, output_to_cte))


def qualify_cte_output_columns(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Prefix references that match a CTE output column with that CTE name.

    Covers the main query and each CTE body's filters, having clauses, window definitions, and CASE registries, not only select/group/order lists.

    Args:

        intent: Intent with ``cte_steps`` populated.

    Returns:

        Intent with qualified expressions, or unchanged when no CTE outputs are declared.
    """

    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return intent

    output_to_cte: dict[str, str] = {}
    for cte in cte_steps:
        explicit_outputs = cte.output_columns or []
        if not explicit_outputs:
            explicit_outputs = infer_cte_output_columns(cte)
        for oc in explicit_outputs:
            bare = oc.split(".")[-1].strip().lower()
            if bare:
                output_to_cte[bare] = cte.cte_name
    if not output_to_cte:
        return intent

    main_tables = {t.strip().lower() for t in (intent.tables or [])}

    def _should_skip(term: str | None) -> bool:
        """Return True when *term* is already qualified with a scope table name."""

        if _is_registry_token(term):
            return True
        if not term or "." not in term:
            return False
        prefix = term.split(".", 1)[0].strip().lower()
        return prefix in main_tables

    def _should_skip_scoped(term: str | None, scope: set[str]) -> bool:
        """Return True when *term* is qualified with a name in *scope*."""

        if _is_registry_token(term):
            return True
        if not term or "." not in term:
            return False
        prefix = term.split(".", 1)[0].strip().lower()
        return prefix in scope

    def _qualify_expr_scoped(
        expr: NormalizedExpr,
        scope: set[str],
    ) -> NormalizedExpr:
        if _should_skip_scoped(expr.primary_column, scope):
            return expr
        return _qualify_expr(expr, output_to_cte)

    def _qualify_filters(
        fps: list[FilterParam],
        scope: set[str],
    ) -> list[FilterParam]:
        out: list[FilterParam] = []
        for fp in fps or []:
            le = _qualify_expr_scoped(fp.left_expr, scope)
            re = _qualify_expr_scoped(fp.right_expr, scope) if fp.right_expr is not None else None
            out.append(replace(fp, left_expr=le, right_expr=re))
        return out

    def _qualify_having(
        hps: list[HavingParam],
        scope: set[str],
    ) -> list[HavingParam]:
        out: list[HavingParam] = []
        for hp in hps or []:
            le = _qualify_expr_scoped(hp.left_expr, scope)
            re = _qualify_expr_scoped(hp.right_expr, scope) if hp.right_expr is not None else None
            out.append(replace(hp, left_expr=le, right_expr=re))
        return out

    def _qualify_wr(
        regs: list[WindowRegistryStep] | None,
        scope: set[str],
    ) -> list[WindowRegistryStep]:
        steps: list[WindowRegistryStep] = []
        for step in regs or []:
            ws = step.window_spec
            np = [_qualify_expr_scoped(e, scope) for e in (ws.partition_by or [])]
            no = [replace(o, expr=_qualify_expr_scoped(o.expr, scope)) for o in (ws.order_by or [])]
            na = _qualify_expr_scoped(ws.argument, scope) if ws.argument is not None else None
            steps.append(
                replace(
                    step,
                    window_spec=replace(ws, partition_by=np, order_by=no, argument=na),
                )
            )
        return steps

    def _qualify_cr(
        regs: list[CaseRegistryStep] | None,
        scope: set[str],
    ) -> list[CaseRegistryStep]:
        out_r: list[CaseRegistryStep] = []
        for step in regs or []:
            cw = step.case_when
            new_branches: list[CaseWhenBranch] = []
            for br in cw.branches or []:
                cond = br.condition
                new_cond = replace(
                    cond,
                    left_expr=_qualify_expr_scoped(cond.left_expr, scope),
                    right_expr=(_qualify_expr_scoped(cond.right_expr, scope) if cond.right_expr is not None else None),
                )
                new_res = _qualify_expr_scoped(br.result, scope)
                new_branches.append(CaseWhenBranch(condition=new_cond, result=new_res))
            new_else = _qualify_expr_scoped(cw.else_result, scope) if cw.else_result is not None else None
            out_r.append(
                replace(
                    step,
                    case_when=replace(cw, branches=new_branches, else_result=new_else),
                )
            )
        return out_r

    new_select_cols = [
        (replace(sc, expr=_qualify_expr(sc.expr, output_to_cte)) if not _should_skip(sc.expr.primary_column) else sc)
        for sc in (intent.select_cols or [])
    ]
    new_group_by = [
        _qualify_expr(g, output_to_cte) if not _should_skip(g.primary_column) else g
        for g in (intent.group_by_cols or [])
    ]
    new_order_by = [
        (
            replace(obc, expr=_qualify_expr(obc.expr, output_to_cte))
            if not _should_skip(obc.expr.primary_column)
            else obc
        )
        for obc in (intent.order_by_cols or [])
    ]
    new_filters = _qualify_filters(intent.filters_param or [], main_tables)
    new_having = _qualify_having(intent.having_param or [], main_tables)
    new_wr = _qualify_wr(intent.window_registry, main_tables)
    new_cr = _qualify_cr(intent.case_registry, main_tables)

    prior_names_lower: list[str] = []
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in cte_steps:
        scope = {t.strip().lower() for t in (cte.tables or [])} | set(prior_names_lower)
        c_sel = [
            (
                replace(sc, expr=_qualify_expr(sc.expr, output_to_cte))
                if not _should_skip_scoped(sc.expr.primary_column, scope)
                else sc
            )
            for sc in (cte.select_cols or [])
        ]
        c_gb = [
            (_qualify_expr(g, output_to_cte) if not _should_skip_scoped(g.primary_column, scope) else g)
            for g in (cte.group_by_cols or [])
        ]
        c_ob = [
            (
                replace(obc, expr=_qualify_expr(obc.expr, output_to_cte))
                if not _should_skip_scoped(obc.expr.primary_column, scope)
                else obc
            )
            for obc in (cte.order_by_cols or [])
        ]
        c_fp = _qualify_filters(cte.filters_param or [], scope)
        c_hp = _qualify_having(cte.having_param or [], scope)
        c_wr = _qualify_wr(cte.window_registry, scope)
        c_cr = _qualify_cr(cte.case_registry, scope)
        new_cte_steps.append(
            replace(
                cte,
                select_cols=c_sel,
                group_by_cols=c_gb,
                order_by_cols=c_ob,
                filters_param=c_fp,
                having_param=c_hp,
                window_registry=c_wr,
                case_registry=c_cr,
            )
        )
        prior_names_lower.append(cte.cte_name.strip().lower())

    if (
        new_select_cols == intent.select_cols
        and new_group_by == intent.group_by_cols
        and new_order_by == intent.order_by_cols
        and new_filters == (intent.filters_param or [])
        and new_having == (intent.having_param or [])
        and new_wr == (intent.window_registry or [])
        and new_cr == (intent.case_registry or [])
        and new_cte_steps == cte_steps
    ):
        return intent

    debug("[qualify_cte_output_columns] qualified unqualified CTE output references")
    return replace(
        intent,
        select_cols=new_select_cols,
        group_by_cols=new_group_by,
        order_by_cols=new_order_by,
        filters_param=new_filters,
        having_param=new_having,
        window_registry=new_wr,
        case_registry=new_cr,
        cte_steps=new_cte_steps,
    )


def _descriptive_column_score(col_name: str, col_meta: ColumnMetadata) -> tuple[int, int, int]:
    """
    Return a sort key (higher is better) for descriptive-column preference.

    Args:

        col_name: Column name.

        col_meta: Column metadata (uniqueness, distinct_count).

    Returns:

        Tuple ``(unique_boost, name_score, distinct_count)``.
    """
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


def best_descriptive_columns(
    table: str,
    schema_graph: SchemaGraph,
    exclude: set[str],
    max_count: int = 2,
) -> list[str]:
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


def _best_composite_name_pair(
    tbl_meta: TableMetadata,
    candidates: list[tuple[str, ColumnMetadata]],
) -> tuple[str, str] | None:
    """
    Return two name-like columns when their composite distinct ratio beats any single.

    Args:

        tbl_meta: Table metadata with ``composite_descriptive_ratios``.

        candidates: Scored (name, meta) pairs already filtered for type and cardinality.

    Returns:

        A pair of column names, or ``None``.
    """
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


def best_descriptive_column(table: str, schema_graph: SchemaGraph, exclude: set[str]) -> str | None:
    """
    Return a single best descriptive column (wrapper around ``max_count=1``).

    Args:

        table: Table name.

        schema_graph: Schema graph.

        exclude: Fully-qualified columns to skip.

    Returns:

        Column name or ``None``.
    """
    cols = best_descriptive_columns(table, schema_graph, exclude, max_count=1)
    return cols[0] if cols else None


def _repair_fk_filters(
    filters: list[FilterParam],
    select_cols: list,
    tables: list[str],
    schema_graph: SchemaGraph,
    label: str = "",
) -> tuple[list[FilterParam], list[str], bool]:
    """
    Scan filters for FK integer + string/enum value pairs (debug only; no rewrite).

    Args:

        filters: Filters to scan.

        select_cols: Current select columns (for descriptive lookup context).

        tables: Intent table list.

        schema_graph: FK and column metadata.

        label: Optional suffix for debug logs.

    Returns:

        ``(filters, tables, changed)``; *changed* indicates a match was logged.
    """
    new_filters: list[FilterParam] = []
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
                f"[intent_resolve.repair_fk_filter_type_mismatch{label}] detected fk filter {col} needing descriptive column"
            )
    return new_filters, tables, changed


def repair_fk_filter_type_mismatch(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Detect string/enum values on numeric FK filters (debug trace; filters not rewritten).

    Filter-only by design; FK-type hints apply to ``filters_param``, not HAVING aggregates.

    Args:

        intent: Main and CTE ``filters_param`` to scan.

        schema_graph: FK targets and descriptive columns.

    Returns:

        Same intent, or ``dataclasses.replace`` with rebound lists if any CTE/main branch matched.
    """
    main_filters, _, main_changed = _repair_fk_filters(
        intent.filters_param or [],
        intent.select_cols or [],
        list(intent.tables or []),
        schema_graph,
    )
    cte_changed = False
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_filters, _, c = _repair_fk_filters(
            cte.filters_param or [],
            cte.select_cols or [],
            list(cte.tables or []),
            schema_graph,
            label=f" CTE '{cte.cte_name}'",
        )
        if c:
            new_cte_steps.append(replace(cte, filters_param=cte_filters))
            cte_changed = True
        else:
            new_cte_steps.append(cte)
    if not main_changed and not cte_changed:
        return intent
    result = intent
    if main_changed:
        result = replace(result, filters_param=main_filters)
    if cte_changed:
        result = replace(result, cte_steps=new_cte_steps)
    return result


def _expand_fk_select_to_descriptive_tables_sel(
    select_cols: list[SelectCol],
    tables: list[str],
    schema_graph: SchemaGraph,
) -> tuple[list[SelectCol], list[str], bool]:
    """
    Expand bare FK integer selects to descriptive columns and extend ``tables``.

    Returns:

        ``(new_select_cols, new_tables, changed)``.
    """

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
        fk_target = col_meta.fk_target
        if not fk_target:
            new_select.append(sc)
            continue
        target_table, _ = fk_target
        descs = best_descriptive_columns(
            target_table,
            schema_graph,
            existing_terms,
            max_count=2,
        )
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
    """
    Rewrite bare non-aggregated FK integer selects on the main query to descriptive columns and add join tables.

    CTE bodies are not expanded so grouped CTE ``GROUP BY`` stays aligned with CTE ``select_cols``.

    Args:

        intent: Intent with ``select_cols`` and ``tables``.

        schema_graph: FK and descriptive metadata.

    Returns:

        Updated intent, or unchanged when no FK select expansion applies on the main query.
    """

    ns, nt, main_ch = _expand_fk_select_to_descriptive_tables_sel(
        list(intent.select_cols or []),
        list(intent.tables or []),
        schema_graph,
    )
    if not main_ch:
        return intent
    return replace(intent, select_cols=ns, tables=nt)


def strip_spurious_group_by(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Clear ``group_by_cols`` when select and having have no aggregation (main and CTEs).

    Args:

        intent: Intent to normalize.

    Returns:

        Intent with spurious grouping removed and grain downgraded from ``grouped`` when applicable.
    """
    main_changed = False
    new_grain = intent.grain
    new_gb = intent.group_by_cols or []
    if intent.group_by_cols:
        has_agg = any(sc.is_aggregated for sc in (intent.select_cols or []))
        has_agg = has_agg or any(hp.left_expr.has_aggregation for hp in (intent.having_param or []))
        if not has_agg:
            debug(
                f"[intent_resolve.strip_spurious_group_by] group_by_cols present without aggregation — stripping {[g.primary_term for g in intent.group_by_cols]}"
            )
            new_grain = "row_level" if intent.grain == "grouped" else intent.grain
            new_gb = []
            main_changed = True

    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps or []:
        if not (cte.group_by_cols or []):
            new_cte_steps.append(cte)
            continue
        cte_has_agg = any(sc.is_aggregated for sc in (cte.select_cols or []))
        cte_has_agg = cte_has_agg or any(hp.left_expr.has_aggregation for hp in (cte.having_param or []))
        if cte_has_agg:
            new_cte_steps.append(cte)
            continue
        debug(
            f"[intent_resolve.strip_spurious_group_by] CTE '{cte.cte_name}' group_by_cols present without aggregation — stripping {[g.primary_term for g in cte.group_by_cols]}"
        )
        cte_grain = "row_level" if cte.grain == "grouped" else cte.grain
        new_cte_steps.append(replace(cte, group_by_cols=[], grain=cte_grain))
        cte_changed = True

    if not main_changed and not cte_changed:
        return intent
    return replace(
        intent,
        group_by_cols=new_gb,
        grain=new_grain,
        cte_steps=new_cte_steps if cte_changed else (intent.cte_steps or []),
    )


def _is_impossible_having(hp: HavingParam) -> bool:
    """
    Return True for impossible COUNT comparisons (e.g. COUNT < 0), not SUM.

    Args:

        hp: Single HAVING clause.

    Returns:

        True when the predicate cannot hold.
    """
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
        numeric_val = float(val) if not isinstance(val, (int, float)) else val
    except (ValueError, TypeError):
        return False
    if op in ("<", "<=") and numeric_val <= 0:
        return True
    if op == "=" and numeric_val < 0:
        return True
    return False


def strip_impossible_having(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Drop HAVING clauses that ``_is_impossible_having`` flags (main and CTEs).

    Args:

        intent: Intent whose ``having_param`` lists are filtered.

    Returns:

        Updated intent, or unchanged when nothing is removed.
    """
    main_having = intent.having_param or []
    kept_main = [hp for hp in main_having if not _is_impossible_having(hp)]
    main_changed = len(kept_main) != len(main_having)
    if main_changed:
        removed = len(main_having) - len(kept_main)
        debug(f"[strip_impossible_having] removed {removed} impossible HAVING condition(s)")

    new_cte_steps = []
    cte_changed = False
    for cte in intent.cte_steps or []:
        cte_having = cte.having_param or []
        kept_cte = [hp for hp in cte_having if not _is_impossible_having(hp)]
        if len(kept_cte) != len(cte_having):
            cte_changed = True
            new_cte_steps.append(replace(cte, having_param=kept_cte))
        else:
            new_cte_steps.append(cte)

    if not main_changed and not cte_changed:
        return intent
    return replace(
        intent,
        having_param=kept_main,
        cte_steps=new_cte_steps if cte_changed else (intent.cte_steps or []),
    )


def _sanitize_table_names_list(
    tables: list[str],
    schema_graph: SchemaGraph,
) -> tuple[list[str], bool]:
    """Return a copy of *tables* with SQL-keyword-prefixed hallucinations corrected when possible."""

    valid_tables = {t.lower(): t for t in schema_graph.tables}
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


def sanitize_table_names(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Remove leading SQL keyword tokens from hallucinated multi-token table names.

    Applies to the main ``tables`` list and each CTE ``tables`` list.

    Args:

        intent: Intent whose ``tables`` may include prefixes like ``FROM orders``.

        schema_graph: Valid table names.

    Returns:

        Sanitized ``tables``, or unchanged intent.
    """

    nt, main_ch = _sanitize_table_names_list(list(intent.tables or []), schema_graph)
    out = replace(intent, tables=nt) if main_ch else intent
    new_cte_steps: list[RuntimeCteStep] = []
    cte_ch = False
    for cte in out.cte_steps or []:
        ctb, c = _sanitize_table_names_list(list(cte.tables or []), schema_graph)
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


def _strip_join_condition_filters(filters: list[FilterParam], schema_graph: SchemaGraph) -> list[FilterParam]:
    """
    Drop ``=`` filters that duplicate a schema FK edge (column-to- column).

    Args:

        filters: Row filters to scan.

        schema_graph: FK edges.

    Returns:

        Filters with join predicates removed.
    """
    fk_pairs: set[tuple[str, str]] = set()
    for tbl in schema_graph.tables.values():
        for fk in tbl.foreign_keys:
            if len(fk.src_cols) == 1 and len(fk.dst_cols) == 1:
                left = f"{fk.src_table}.{fk.src_cols[0]}"
                right = f"{fk.dst_table}.{fk.dst_cols[0]}"
                fk_pairs.add((left, right))
                fk_pairs.add((right, left))
    result: list[FilterParam] = []
    for fp in filters:
        if fp.right_expr is None or fp.op != "=":
            result.append(fp)
            continue
        left_term = fp.left_expr.primary_term
        right_term = fp.right_expr.primary_term
        if (left_term, right_term) in fk_pairs:
            debug(f"[intent_resolve.strip_join_condition_filters] dropping FK join filter: {left_term} = {right_term}")
            continue
        result.append(fp)
    return result


def strip_join_conditions(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Apply ``_strip_join_condition_filters`` to main and each CTE.

    Args:

        intent: Intent to update.

        schema_graph: FK metadata.

    Returns:

        Intent with join-equivalent filters removed from all filter lists.
    """
    new_filters = _strip_join_condition_filters(intent.filters_param or [], schema_graph)
    new_cte_steps = [
        replace(
            cte,
            filters_param=_strip_join_condition_filters(cte.filters_param or [], schema_graph),
        )
        for cte in (intent.cte_steps or [])
    ]
    return replace(intent, filters_param=new_filters, cte_steps=new_cte_steps)


def _is_pk_column(col_ref: str, schema_graph: SchemaGraph) -> bool:
    """
    Return True when *col_ref* is a primary key column.

    Args:

        col_ref: ``table.column`` string.

        schema_graph: Table/column metadata.

    Returns:

        True if the column is a PK.
    """
    if "." not in col_ref:
        return False
    tbl, col = col_ref.split(".", 1)
    tbl_meta = schema_graph.tables.get(tbl)
    if not tbl_meta:
        return False
    col_meta = tbl_meta.columns.get(col)
    return col_meta.is_primary_key if col_meta else False


def _strip_distinct_prefix(term: str) -> str:
    """
    Strip a leading ``DISTINCT `` token from *term*.

    Args:

        term: Raw operand string.

    Returns:

        Term without the prefix, or *term* unchanged.
    """
    if term.upper().startswith("DISTINCT "):
        return term[9:].strip()
    return term


def _normalize_sc_pk_distinct(sc: SelectCol, schema_graph: SchemaGraph) -> SelectCol:
    """For COUNT on a PK, clear the redundant ``DISTINCT`` flag from the MulGroup."""
    e = sc.expr
    g0 = e.add_groups[0] if e.add_groups else None
    agg = (e.agg_func or (g0.agg_func if g0 else "") or "").lower()
    if agg != "count":
        return sc
    if not g0 or not g0.distinct:
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
    """
    Remove ``DISTINCT`` from ``COUNT`` on PK columns (main and CTE selects).

    Args:

        intent: Intent to normalize.

        schema_graph: PK metadata.

    Returns:

        Updated intent.
    """
    new_select = [_normalize_sc_pk_distinct(sc, schema_graph) for sc in (intent.select_cols or [])]
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_select = [_normalize_sc_pk_distinct(sc, schema_graph) for sc in (cte.select_cols or [])]
        new_cte_steps.append(replace(cte, select_cols=cte_select))
    return replace(intent, select_cols=new_select, cte_steps=new_cte_steps)


def _lift_distinct_from_select_col(sc: SelectCol) -> tuple[SelectCol, bool]:
    """
    Strip stray DISTINCT scalar wrappers from non-aggregate groups.

    With the parser-native schema, DISTINCT lives on ``MulGroup.distinct``; this helper only catches legacy-shaped ``DISTINCT(...)`` scalar wrappers that may slip in. Returns ``(sc, changed_flag)``.
    """
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
    expr: NormalizedExpr,
    schema: SchemaGraph,
    cte_steps: Sequence[RuntimeCteStep] | None = None,
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


def _cte_output_value_type(
    qualifier: str,
    output_alias: str,
    cte_steps: Sequence[RuntimeCteStep] | None,
) -> str | None:
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
    expr: NormalizedExpr | None,
    schema: SchemaGraph,
    cte_steps: Sequence[RuntimeCteStep] | None = None,
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
    expr: NormalizedExpr | None,
    schema: SchemaGraph,
    cte_steps: Sequence[RuntimeCteStep] | None = None,
) -> bool:
    if expr is None:
        return False
    vt = _expr_primary_column_value_type(expr, schema, cte_steps)
    if vt and vt in STRING_COLUMN_VALUE_TYPES:
        return True
    return False


def _align_pred_value_type(
    pred: FilterParam | HavingParam,
    schema: SchemaGraph,
    cte_steps: Sequence[RuntimeCteStep] | None = None,
) -> FilterParam | HavingParam:
    left = pred.left_expr
    if left is None:
        return pred
    op = (pred.op or "").lower()
    current = pred.value_type or ""

    if op in ("is null", "is not null"):
        return pred

    if current in ("date_window", "date_diff"):
        return pred

    if op in STRING_OPS:
        if current != "string":
            debug(f"[align_filter_value_type_to_exprs] overriding value_type {current!r} -> 'string' for op {op!r}")
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
                f"[align_filter_value_type_to_exprs] overriding value_type "
                f"{current!r} -> {target!r} on numeric predicate"
            )
            return replace(pred, value_type=target)
        return pred

    left_date = _is_expr_date(left, schema, cte_steps)
    right_date = right is None or _is_expr_date(right, schema, cte_steps)
    if left_date and right_date and right is not None:
        if current != "date":
            debug(f"[align_filter_value_type_to_exprs] overriding value_type {current!r} -> 'date' on date predicate")
            return replace(pred, value_type="date")
        return pred

    if _is_expr_string(left, schema, cte_steps) or _is_expr_string(right, schema, cte_steps):
        if current not in ("string",) and current not in ("date_window", "date_diff"):
            debug(
                f"[align_filter_value_type_to_exprs] overriding value_type "
                f"{current!r} -> 'string' on string column predicate"
            )
            return replace(pred, value_type="string")
    return pred


def _collect_number_typed_predicate_param_keys(
    pred: FilterParam | HavingParam,
) -> set[str]:
    if (pred.value_type or "").lower() != "number":
        return set()
    keys: set[str] = set()
    pk = (pred.param_key or "").strip()
    if pk:
        keys.add(pk)
    if isinstance(pred, FilterParam):
        pkh = (pred.param_key_hi or "").strip()
        if pkh:
            keys.add(pkh)
    return keys


def _maybe_coerce_bool_literal_for_numeric_pred(
    pred: FilterParam | HavingParam,
) -> FilterParam | HavingParam:
    if (pred.value_type or "").lower() != "number":
        return pred
    if isinstance(pred.raw_value, bool):
        return replace(pred, raw_value=1 if pred.raw_value else 0)
    return pred


def _collect_numeric_predicate_param_keys_from_case_registry(
    registry: Sequence[CaseRegistryStep] | None,
) -> set[str]:
    keys: set[str] = set()
    if not registry:
        return keys
    for step in registry:
        cw = step.case_when
        if not cw or not cw.branches:
            continue
        for br in cw.branches:
            cond = br.condition
            if cond is None:
                continue
            keys.update(_collect_number_typed_predicate_param_keys(cond))
    return keys


def _coerce_boolean_bindings_for_number_typed_filters(
    intent: RuntimeIntent,
) -> RuntimeIntent:
    keys: set[str] = set()
    for fp in intent.filters_param or []:
        keys.update(_collect_number_typed_predicate_param_keys(fp))
    for hp in intent.having_param or []:
        keys.update(_collect_number_typed_predicate_param_keys(hp))
    for cte in intent.cte_steps or []:
        for fp in cte.filters_param or []:
            keys.update(_collect_number_typed_predicate_param_keys(fp))
        for hp in cte.having_param or []:
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
    new_filters = [_maybe_coerce_bool_literal_for_numeric_pred(fp) for fp in intent.filters_param or []]
    new_having = [_maybe_coerce_bool_literal_for_numeric_pred(hp) for hp in intent.having_param or []]
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        nf = [_maybe_coerce_bool_literal_for_numeric_pred(x) for x in cte.filters_param or []]
        nh = [_maybe_coerce_bool_literal_for_numeric_pred(x) for x in cte.having_param or []]
        new_ctes.append(
            replace(
                cte,
                filters_param=nf,
                having_param=nh,
                param_values=_patch_param_map(cte.param_values),
            )
        )
    return replace(
        intent,
        filters_param=new_filters,
        having_param=new_having,
        cte_steps=new_ctes,
        param_values=new_pv,
    )


def align_filter_value_type_to_exprs(intent: RuntimeIntent, schema: SchemaGraph) -> RuntimeIntent:
    """
    Align ``FilterParam.value_type`` and ``HavingParam.value_type`` to the actual typing of the predicate sides.

    Decision order: ``is null``/``is not null`` is preserved; ``date_window``/``date_diff`` is preserved; LIKE/ILIKE/contains -> ``string``; both sides numeric (and not string columns) -> ``number``; both sides date -> ``date``; any side string column -> ``string``. Walks main + CTE filters/havings and case-branch conditions, consulting CTE output column metadata when a predicate references a ``cte_name.alias`` reference.
    """
    cte_steps_seq = intent.cte_steps or []
    new_filters = [_align_pred_value_type(fp, schema, cte_steps_seq) for fp in (intent.filters_param or [])]
    new_having = [_align_pred_value_type(hp, schema, cte_steps_seq) for hp in (intent.having_param or [])]
    new_cte_steps = []
    for cte in cte_steps_seq:
        cte_filters = [_align_pred_value_type(fp, schema, cte_steps_seq) for fp in (cte.filters_param or [])]
        cte_having = [_align_pred_value_type(hp, schema, cte_steps_seq) for hp in (cte.having_param or [])]
        new_cte_steps.append(replace(cte, filters_param=cte_filters, having_param=cte_having))
    intent = replace(
        intent,
        filters_param=new_filters,
        having_param=new_having,
        cte_steps=new_cte_steps,
    )

    def _branch_align(conds: list[FilterParam]) -> list[FilterParam]:
        return [_align_pred_value_type(c, schema, cte_steps_seq) for c in conds]

    intent = map_case_branch_conditions(intent, _branch_align)

    def _branch_coerce_bool_num(conds: list[FilterParam]) -> list[FilterParam]:
        if not conds:
            return conds
        return [_maybe_coerce_bool_literal_for_numeric_pred(conds[0])]

    intent = map_case_branch_conditions(
        intent,
        _branch_coerce_bool_num,
        scopes=frozenset({"filter", "having"}),
    )
    return _coerce_boolean_bindings_for_number_typed_filters(intent)


def lift_distinct_modifier_in_multiply(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Strip standalone ``DISTINCT`` prefixes from multiply tokens lacking an aggregate wrapper.

    Bare row-level ``DISTINCT col`` tokens emitted by the LLM cannot render to valid SQL when no
    surrounding aggregate consumes them; this repair removes the prefix so downstream rendering
    succeeds. Multiply tokens inside ``COUNT(...)`` or other aggregates are preserved because the
    deterministic SQL renderer already emits ``COUNT(DISTINCT col)`` for those.

    The first select column from which a bare ``DISTINCT`` is stripped records its index on
    ``intent.distinct_select_index`` (and on each ``RuntimeCteStep.distinct_select_index`` for
    CTE scopes); the renderer reads this to emit ``SELECT DISTINCT``. ``DISTINCT`` is a
    statement-level modifier so only the first stripped index is recorded per scope.

    Args:

        intent: Intent to repair.

    Returns:

        Updated intent with bare DISTINCT prefixes removed from main and CTE select columns
        and the distinct select index recorded.
    """
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
    return replace(
        intent,
        select_cols=new_select,
        cte_steps=new_cte_steps,
        distinct_select_index=main_distinct_index,
    )


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


def _rewrite_expr_unknown_datepart_to_extract(
    expr: NormalizedExpr | None,
) -> NormalizedExpr | None:
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
    if new_expr is sc.expr:
        return sc
    return replace(sc, expr=new_expr)


def replace_unknown_scalar_funcs(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Rewrite ``YEAR``/``MONTH``/``DAY``/``QUARTER``/``DOW`` calls to canonical ``EXTRACT(unit FROM x)``.

    The renderer and validator only accept the ``extract`` scalar; LLMs often produce vendor-specific date-part functions that fail validation or execution. This deterministic step normalizes them in main and CTE select expressions, filters, having, group-by, and order-by exprs.
    """

    def _rewrite_filters(items: list[FilterParam]) -> list[FilterParam]:
        out: list[FilterParam] = []
        for fp in items:
            new_left = _rewrite_expr_unknown_datepart_to_extract(fp.left_expr)
            new_right = _rewrite_expr_unknown_datepart_to_extract(fp.right_expr) if fp.right_expr else fp.right_expr
            out.append(replace(fp, left_expr=new_left, right_expr=new_right))
        return out

    def _rewrite_havings(items: list[HavingParam]) -> list[HavingParam]:
        out: list[HavingParam] = []
        for hp in items:
            new_left = _rewrite_expr_unknown_datepart_to_extract(hp.left_expr)
            new_right = _rewrite_expr_unknown_datepart_to_extract(hp.right_expr) if hp.right_expr else hp.right_expr
            out.append(replace(hp, left_expr=new_left, right_expr=new_right))
        return out

    new_select = [_rewrite_select_col_unknown_datepart(sc) for sc in (intent.select_cols or [])]
    new_group_by = [_rewrite_expr_unknown_datepart_to_extract(g) for g in (intent.group_by_cols or [])]
    new_order_by = [
        replace(obc, expr=_rewrite_expr_unknown_datepart_to_extract(obc.expr)) for obc in (intent.order_by_cols or [])
    ]
    new_filters = _rewrite_filters(intent.filters_param or [])
    new_having = _rewrite_havings(intent.having_param or [])
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_select = [_rewrite_select_col_unknown_datepart(sc) for sc in (cte.select_cols or [])]
        cte_group_by = [_rewrite_expr_unknown_datepart_to_extract(g) for g in (cte.group_by_cols or [])]
        cte_order_by = [
            replace(obc, expr=_rewrite_expr_unknown_datepart_to_extract(obc.expr)) for obc in (cte.order_by_cols or [])
        ]
        cte_filters = _rewrite_filters(cte.filters_param or [])
        cte_having = _rewrite_havings(cte.having_param or [])
        new_cte_steps.append(
            replace(
                cte,
                select_cols=cte_select,
                group_by_cols=cte_group_by,
                order_by_cols=cte_order_by,
                filters_param=cte_filters,
                having_param=cte_having,
            )
        )
    return replace(
        intent,
        select_cols=new_select,
        group_by_cols=new_group_by,
        order_by_cols=new_order_by,
        filters_param=new_filters,
        having_param=new_having,
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
    window_registry: Sequence[WindowRegistryStep] | None,
    case_registry: Sequence[CaseRegistryStep] | None,
) -> list[str]:
    """Column references from window and case registry definitions."""
    buf: list[str] = []
    for step in window_registry or []:
        _append_window_spec_cols(buf, step.window_spec)
    for step in case_registry or []:
        _append_case_when_cols(buf, step.case_when)
    return buf


def _tables_from_columns(cols: list[str]) -> set[str]:
    """
    Return distinct table prefixes from qualified ``table.column`` strings.

    Args:

        cols: Column reference strings.

    Returns:

        Set of table names.
    """
    tables: set[str] = set()
    for col in cols:
        if "." in col:
            head = col.split(".", 1)[0]
            if IDENTIFIER_RE.match(head):
                tables.add(head)
    return tables


def collect_referenced_tables(
    select_cols: list,
    order_by_cols: list,
    group_by_cols: list,
    filters_param: list,
    having_param: list,
    *,
    window_registry: Sequence[WindowRegistryStep] | None = None,
    case_registry: Sequence[CaseRegistryStep] | None = None,
) -> set[str]:
    """
    Union of tables referenced in select, order, group, filters, and having.

    Args:

        select_cols: Select columns.

        order_by_cols: Order-by columns.

        group_by_cols: Group-by expressions.

        filters_param: Row filters.

        having_param: Having clauses.

        window_registry: Optional window registry for ``registry_ref`` and inline window specs.

        case_registry: Optional case registry for ``registry_ref`` and inline CASE payloads.

    Returns:

        Table names appearing in any extracted column reference.
    """
    all_cols: list[str] = []
    for sc in select_cols or []:
        all_cols.extend(cols_from_select_col(sc, window_registry, case_registry))
    all_cols.extend(cols_from_named_registries(window_registry, case_registry))
    for obc in order_by_cols or []:
        all_cols.extend(extract_columns_from_expr(obc.expr))
    for g in group_by_cols or []:
        all_cols.extend(extract_columns_from_expr(g))
    for fp in filters_param or []:
        all_cols.extend(extract_columns_from_expr(fp.left_expr))
        if fp.right_expr:
            all_cols.extend(extract_columns_from_expr(fp.right_expr))
    for hp in having_param or []:
        all_cols.extend(extract_columns_from_expr(hp.left_expr))
        if hp.right_expr:
            all_cols.extend(extract_columns_from_expr(hp.right_expr))
    return _tables_from_columns(all_cols)


def reconcile_tables(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Set ``tables`` at every level to exactly the tables and CTEs referenced at that level.

    For the main scope and each CTE step independently, this function recomputes the
    referenced table set from select, order, group, filter, having, window registry, and
    case registry expressions. The resulting ``tables`` list is the sorted reference set.

    No table is force-added because of CTE chain membership and no prior CTE name is
    inserted into a downstream CTE's tables list. Tables present in the input but not
    referenced are removed; tables referenced but missing from the input are added back.

    Args:

        intent: Intent whose ``tables`` lists need to match what is actually referenced.

    Returns:

        Intent with reconciled ``tables`` lists at the main scope and on each CTE step.
    """
    main_referenced = collect_referenced_tables(
        intent.select_cols,
        intent.order_by_cols,
        intent.group_by_cols,
        intent.filters_param,
        intent.having_param,
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    main_tables = sorted(main_referenced)
    original_main = set(intent.tables or [])
    added_main = main_referenced - original_main
    removed_main = original_main - main_referenced
    if added_main:
        debug(f"[reconcile_tables] main added {sorted(added_main)}")
    if removed_main:
        debug(f"[reconcile_tables] main removed {sorted(removed_main)}")

    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_referenced = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            cte.filters_param,
            cte.having_param,
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
        )
        cte_tables = sorted(cte_referenced)
        cte_original = set(cte.tables or [])
        cte_added = cte_referenced - cte_original
        cte_removed = cte_original - cte_referenced
        if cte_added:
            debug(f"[reconcile_tables] CTE '{cte.cte_name}' added {sorted(cte_added)}")
        if cte_removed:
            debug(f"[reconcile_tables] CTE '{cte.cte_name}' removed {sorted(cte_removed)}")
        new_cte_steps.append(replace(cte, tables=cte_tables))

    return replace(intent, tables=main_tables, cte_steps=new_cte_steps)


def _match_enum_value(raw_value: str, col_meta: ColumnMetadata, schema_graph: SchemaGraph) -> str | None:
    """
    Case-insensitive match of *raw_value* to a DB enum literal for *col_meta*.

    Args:

        raw_value: Filter literal from the question.

        col_meta: Target column metadata (``data_type`` names the enum).

        schema_graph: Holds ``enum_values`` by type name.

    Returns:

        Canonically cased enum member, or ``None``.
    """
    if not schema_graph.enum_values:
        return None
    dtype_lower = (col_meta.data_type or "").lower()
    enum_vals = schema_graph.enum_values.get(dtype_lower)
    if not enum_vals:
        return None
    raw_lower = raw_value.lower()
    for ev in enum_vals:
        if ev.lower() == raw_lower:
            return ev
    return None


def _resolve_filter_list_cascade(
    filters: list[FilterParam],
    schema_graph: SchemaGraph,
    question: str,
) -> tuple[list[FilterParam], bool]:
    """
    Enum-aware casing fix: DB enum literals first, else lowercase for LOWER() SQL.

    Args:

        filters: Filters to adjust.

        schema_graph: Enum definitions and column metadata.

        question: Original question (reserved for future casing hints).

    Returns:

        ``(new_filters, changed)``.
    """
    new_filters: list[FilterParam] = []
    changed = False
    for fp in filters:
        if fp.raw_value is None or fp.value_type not in {"string", "enum"}:
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

        if isinstance(fp.raw_value, list):
            new_vals: list = []
            list_changed = False
            for v in fp.raw_value:
                if not isinstance(v, str):
                    new_vals.append(v)
                    continue
                enum_match = _match_enum_value(v, col_meta, schema_graph)
                if enum_match is not None:
                    if enum_match != v:
                        list_changed = True
                    new_vals.append(enum_match)
                else:
                    lowered = v.lower()
                    if lowered != v:
                        list_changed = True
                    new_vals.append(lowered)
            if list_changed:
                new_filters.append(replace(fp, raw_value=new_vals))
                changed = True
                debug(f"[intent_repair.resolve_filter_list_cascade] resolved list values on {col}")
            else:
                new_filters.append(fp)
            continue

        if not isinstance(fp.raw_value, str):
            new_filters.append(fp)
            continue

        enum_match = _match_enum_value(fp.raw_value, col_meta, schema_graph)
        if enum_match is not None:
            if enum_match != fp.raw_value:
                new_filters.append(replace(fp, raw_value=enum_match))
                changed = True
                debug(f"[intent_repair.resolve_filter_list_cascade] enum {col}: '{fp.raw_value}' -> '{enum_match}'")
            else:
                new_filters.append(fp)
            continue

        lowered = fp.raw_value.lower()
        if lowered != fp.raw_value:
            new_filters.append(replace(fp, raw_value=lowered))
            changed = True
            debug(f"[intent_repair.resolve_filter_list_cascade] lower {col}: '{fp.raw_value}' -> '{lowered}'")
        else:
            new_filters.append(fp)
    return new_filters, changed


def resolve_filter_value_case(intent: RuntimeIntent, schema_graph: SchemaGraph, question: str) -> RuntimeIntent:
    """
    Apply ``_resolve_filter_list_cascade`` to main and CTE filter lists.

    Filter-only by design; HAVING literals are not resolved through this path.

    Args:

        intent: Intent to update.

        schema_graph: Enum and column data.

        question: User question.

    Returns:

        Intent with updated ``raw_value`` casing where applicable.
    """

    def process(filters: list[FilterParam]) -> tuple[list[FilterParam], bool]:
        return _resolve_filter_list_cascade(filters, schema_graph, question)

    return _apply_filters_to_main_and_ctes(intent, process)


def _coerce_element(val: Any, data_type: str) -> Any:
    """
    Coerce one IN-list element toward *data_type* (numeric columns only).

    Args:

        val: List element.

        data_type: Lowercased SQL type string.

    Returns:

        Coerced scalar or *val* unchanged.
    """
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


def _consolidate_in_list(vals: list, data_type: str) -> str:
    """
    Join IN-list values into a comma-separated SQL fragment string.

    Args:

        vals: Coerced list elements.

        data_type: Column type (affects quoting).

    Returns:

        String for direct SQL substitution.
    """
    if all(isinstance(v, (int, float)) for v in vals):
        return ", ".join(str(v) for v in vals)
    parts: list[str] = []
    for v in vals:
        if isinstance(v, str):
            parts.append(f"'{v}'")
        else:
            parts.append(str(v))
    return ", ".join(parts)


def _normalize_in_types_for_list(
    filters: list[FilterParam],
    schema_graph: SchemaGraph,
) -> tuple[list[FilterParam], bool]:
    """
    Coerce IN-list elements to column types, then consolidate to one SQL string.

    Args:

        filters: Filters to process.

        schema_graph: Column types.

    Returns:

        ``(filters, changed)``.
    """
    new_filters: list[FilterParam] = []
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
        consolidated = _consolidate_in_list(coerced, dtype)
        if consolidated != fp.raw_value:
            new_filters.append(replace(fp, raw_value=consolidated))
            changed = True
            debug(f"[intent_resolve_normalize_in_types_for_list] {col}: {fp.raw_value!r} -> {consolidated!r}")
        else:
            new_filters.append(fp)
    return new_filters, changed


def normalize_in_filter_types(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Run IN coercion on main/CTEs, then ``decompose_in_not_in_filters``.

    Filter-only by design; HAVING operates on aggregation results, not ``IN``-list expansion here.

    Args:

        intent: Intent to update.

        schema_graph: Column metadata.

    Returns:

        Intent after consolidation and optional decomposition.
    """

    def process(filters: list[FilterParam]) -> tuple[list[FilterParam], bool]:
        return _normalize_in_types_for_list(filters, schema_graph)

    intent = _apply_filters_to_main_and_ctes(intent, process)
    return decompose_in_not_in_filters(intent)


def _decompose_in_list(
    filters: list[FilterParam],
    max_list_size: int = 10,
) -> list[FilterParam]:
    """
    Split short IN/NOT IN lists into chained ``=``/``!=`` filters with OR/AND.

    Args:

        filters: Input filters.

        max_list_size: Maximum list length to expand.

    Returns:

        Possibly longer filter list.
    """
    new_filters: list[FilterParam] = []
    for fp in filters:
        if fp.filter_group is not None:
            new_filters.append(fp)
            continue
        raw = fp.raw_value
        op_lower = (fp.op or "").lower()
        if op_lower in {"in", "not in"} and isinstance(raw, str):
            parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
            parts = [p for p in parts if p]
            value_type_lower = (fp.value_type or "").lower()
            if value_type_lower in {"integer", "int", "bigint", "smallint"}:
                coerced: list[Any] = []
                for p in parts:
                    try:
                        coerced.append(int(p))
                    except (TypeError, ValueError):
                        coerced.append(p)
                parts = coerced
            elif value_type_lower in {
                "number",
                "numeric",
                "float",
                "double",
                "decimal",
                "real",
            }:
                coerced_f: list[Any] = []
                for p in parts:
                    try:
                        coerced_f.append(float(p))
                    except (TypeError, ValueError):
                        coerced_f.append(p)
                parts = coerced_f
            if parts:
                raw = parts
                fp = replace(fp, raw_value=parts)
        if not isinstance(raw, list) or op_lower not in {"in", "not in"} or len(raw) == 0 or len(raw) > max_list_size:
            new_filters.append(fp)
            continue
        elems = list(raw)
        n = len(elems)
        connector = "OR" if op_lower == "in" else "AND"
        new_group = []
        for idx, val in enumerate(elems):
            link_after = (fp.bool_op or "AND") if idx == n - 1 else connector
            new_fp = replace(
                fp,
                op="=" if op_lower == "in" else "!=",
                raw_value=val,
                bool_op=link_after,
            )
            new_group.append(new_fp)
        new_filters.extend(new_group)
    return new_filters


def decompose_in_not_in_filters(intent: RuntimeIntent) -> RuntimeIntent:
    """Apply ``_decompose_in_list`` to main and each CTE (filter-only; HAVING is out of scope)."""
    main_filters = _decompose_in_list(intent.filters_param or [])
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        decomposed = _decompose_in_list(cte.filters_param or [])
        new_ctes.append(replace(cte, filters_param=decomposed))
    out = replace(intent, filters_param=main_filters, cte_steps=new_ctes or intent.cte_steps)
    expanded = len(main_filters) != len(intent.filters_param or [])
    if not expanded:
        for oc, nc in zip(intent.cte_steps or [], new_ctes, strict=True):
            if len(oc.filters_param or []) != len(nc.filters_param or []):
                expanded = True
                break
    if expanded:
        pipeline_trace_lazy(
            "intent_after_deterministic_repair.decompose_in_filters",
            lambda: stable_json(
                {
                    "main_filters": len(out.filters_param or []),
                    "cte_steps": len(out.cte_steps or []),
                }
            ),
        )
    return out


def _resolve_boolean_value(raw_value: Any, col_meta: ColumnMetadata) -> tuple[Any, str] | None:
    """
    Map *raw_value* to ``True``/``False`` when the column is a native boolean type.

    Args:

        raw_value: Filter literal.

        col_meta: Column metadata.

    Returns:

        ``(bool, "boolean")`` or ``None``.
    """
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


def _normalize_boolean_filter_list(
    filters: list[FilterParam], schema_graph: SchemaGraph
) -> tuple[list[FilterParam], bool]:
    """
    Rewrite boolean-column filters to Python bool and ``value_type`` ``boolean``.

    Args:

        filters: Filters to scan.

        schema_graph: Column types.

    Returns:

        ``(filters, changed)``.
    """
    new_filters: list[FilterParam] = []
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
            f"[intent_resolve_normalize_boolean_filter_list] {col}: "
            f"{fp.raw_value!r} ({fp.value_type}) → {bool_val!r} ({vtype})"
        )
    return new_filters, changed


def normalize_boolean_filter_values(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Apply ``_normalize_boolean_filter_list`` to main and CTE filters.

    Filter-only by design; boolean coercion applies to ``filters_param`` row literals, not HAVING.

    Args:

        intent: Intent to update.

        schema_graph: Column metadata.

    Returns:

        Intent with bool literals for boolean columns where applicable.
    """

    def process(filters: list[FilterParam]) -> tuple[list[FilterParam], bool]:
        return _normalize_boolean_filter_list(filters, schema_graph)

    return _apply_filters_to_main_and_ctes(intent, process)


def _normalize_null_filter_list(
    filters: list[FilterParam],
) -> tuple[list[FilterParam], bool]:
    """
    Force ``value_type="null"`` and ``raw_value=None`` for null operators.

    Args:

        filters: Filters to normalize.

    Returns:

        ``(filters, changed)``.
    """
    result: list[FilterParam] = []
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


def _normalize_null_having_list(
    having: list[HavingParam],
) -> tuple[list[HavingParam], bool]:
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


def normalize_null_filter_values(intent: RuntimeIntent) -> RuntimeIntent:
    """Apply null-operator normalization to main and CTE ``filters_param`` and ``having_param``."""
    intent = _apply_filters_to_main_and_ctes(intent, _normalize_null_filter_list)
    return _apply_having_to_main_and_ctes(intent, _normalize_null_having_list)


def _allocate_window_registry_id(registry: list[WindowRegistryStep]) -> str:
    """Return the next unused ``wNN`` id given existing window registry steps."""

    mx = 0
    for step in registry:
        m = re.fullmatch(r"w(\d{2})", (step.registry_id or "").strip())
        if m:
            mx = max(mx, int(m.group(1)))
    return f"w{mx + 1:02d}"


def _select_cols_have_aggregation(
    select_cols: Sequence[SelectCol],
    window_registry: list[WindowRegistryStep],
) -> bool:
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


def repair_cumulative_phrasing_window_intent(
    intent: RuntimeIntent,
    question_norm: str,
) -> RuntimeIntent:
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
            main_select,
            main_order,
            main_wr,
            list(intent.case_registry or []),
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
                cte_select,
                cte_order,
                cte_wr,
                list(cte.case_registry or []),
            )
            if c:
                cte_changed = True
                new_ctes.append(replace(cte, select_cols=cte_select, window_registry=cte_wr))
                continue
        new_ctes.append(cte)
    if cte_changed:
        intent = replace(intent, cte_steps=new_ctes)
    return intent


def drop_invalid_case_registry_entries(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Remove case-registry rows whose ``case_when`` has no branches and drop select columns that reference those ids.

    Args:

        intent: Intent to normalize.

        schema_graph: Unused; kept for API symmetry with other deterministic repairs.

    Returns:

        Intent without empty CASE registry definitions or dangling ``cNN`` references to them.
    """

    _ = schema_graph

    def prune_scope(
        select_cols: list[SelectCol],
        case_registry: list[CaseRegistryStep],
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

    main_sel, main_cr, main_changed = prune_scope(
        list(intent.select_cols or []),
        list(intent.case_registry or []),
    )
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


def repair_case_when_intent(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Drop ``case_registry`` rows whose ``case_when`` has no branches.

    Args:

        intent: Intent to repair.

        schema_graph: Unused; kept for API symmetry.

    Returns:

        Intent without empty CASE registry definitions.
    """
    _ = schema_graph

    def _strip_registry(regs: list[CaseRegistryStep] | None) -> list[CaseRegistryStep]:
        return [s for s in (regs or []) if s.case_when and s.case_when.branches]

    new_cr = _strip_registry(intent.case_registry)
    new_ctes = [replace(cte, case_registry=_strip_registry(cte.case_registry)) for cte in (intent.cte_steps or [])]
    return replace(intent, case_registry=new_cr, cte_steps=new_ctes)


def _column_meta_for_filter_left(
    fp: FilterParam,
    schema_graph: SchemaGraph,
) -> ColumnMetadata | None:
    """Return metadata when ``left_expr`` references exactly one qualified column."""
    cols = extract_columns_from_expr(fp.left_expr)
    if len(cols) != 1:
        return None
    parts = cols[0].split(".", 1)
    if len(parts) != 2:
        return None
    return schema_graph.get_column(parts[0], parts[1])


def repair_array_filters_intent(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Normalise array-column filters: rewrite ``=``/``like`` on array columns to ``contains`` and remove ``contains`` on non-array columns.

    Filter-only by design; array ``contains`` normalisation targets ``filters_param`` only.

    Args:

        intent: Intent to repair.

        schema_graph: Column array metadata.

    Returns:

        Intent with corrected array filter operators.
    """

    def process(filters: list[FilterParam]) -> tuple[list[FilterParam], bool]:
        out: list[FilterParam] = []
        changed = False
        for fp in filters:
            meta = _column_meta_for_filter_left(fp, schema_graph)
            if fp.op == "contains":
                if meta is not None and meta.element_type:
                    out.append(fp)
                    continue
                vt = (meta.value_type or "").lower() if meta else ""
                if meta is not None and vt in STRING_COLUMN_VALUE_TYPES:
                    rv = fp.raw_value
                    if isinstance(rv, str) and rv and "%" not in rv:
                        rv = f"%{rv}%"
                    new_fp = replace(
                        fp,
                        op="like",
                        raw_value=rv,
                        value_type="string",
                    )
                    out.append(new_fp)
                    changed = True
                    continue
                if meta is None or not meta.element_type:
                    debug(f"[intent_repair.repair_array_filters] dropping contains on non-array column: {fp.param_key}")
                    changed = True
                    continue
            elif fp.op in ARRAY_REWRITABLE_OPS and meta is not None and meta.element_type:
                debug(
                    f"[intent_repair.repair_array_filters] rewriting {fp.op} to contains for array column: {fp.param_key}"
                )
                fp = replace(fp, op="contains", value_type="string")
                changed = True
            out.append(fp)
        return out, changed

    return _apply_filters_to_main_and_ctes(intent, process)


def _norm_expr_blocked_non_selectable_refs(expr: NormalizedExpr, schema: SchemaGraph) -> list[str]:
    """Return qualified ``table.column`` references in *expr* that are not selectable under policy."""

    blocked: list[str] = []
    exempt = selectability_exempt_qualified_refs(expr, schema)
    for ref in extract_columns_from_expr(expr):
        if ref in exempt:
            continue
        parts = ref.split(".", 1)
        if len(parts) != 2:
            continue
        meta = schema.get_column(parts[0], parts[1])
        if meta is not None and not meta.is_selectable:
            blocked.append(ref)
    return blocked


def _select_col_selectable(sc: SelectCol, schema: SchemaGraph) -> bool:
    """Return False when the main expression projects blocked columns without an allowed ``COUNT`` form."""

    return not _norm_expr_blocked_non_selectable_refs(sc.expr, schema)


def _select_col_dropped_blocked_columns(sc: SelectCol, schema: SchemaGraph) -> list[str]:
    """Return the qualified ``table.column`` references in *sc* that fail the selectability gate."""

    return _norm_expr_blocked_non_selectable_refs(sc.expr, schema)


def enforce_sensitivity_policy_intent(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Drop select and ``GROUP BY`` entries that reference hidden-sensitivity columns and notify when any are dropped.

    Args:

        intent: Intent to filter.

        schema_graph: Sensitivity flags per column.

    Returns:

        Intent with restricted columns removed from selects and from ``GROUP BY`` lists.

    Raises:

        ValueError: When every main select column references a hidden-sensitivity column so the
        resulting intent would have no projectable output. The message is tagged with
        :attr:`FailureCategory.SENSITIVITY_ALL_SELECT_DROPPED` so the pipeline boundary can
        classify the rejection.

        ValueError: When ``grain`` is grouped and every ``GROUP BY`` expression is removed for the
        same reason; message prefix :attr:`FailureCategory.SENSITIVITY_ALL_GROUP_BY_DROPPED`.
    """

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
                "Dropping select column(s) referencing hidden-sensitivity fields: " + ", ".join(sorted(set(refs))),
                stage="intent",
                code=DIAGNOSTIC_CODE_PII_GATE_HIT,
            )
    if original_main and not kept_main:
        raise ValueError(
            f"{FailureCategory.SENSITIVITY_ALL_SELECT_DROPPED.value}: every requested select column "
            "references a hidden-sensitivity field; no projectable output remains"
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
            "Dropping GROUP BY expression(s) referencing hidden-sensitivity fields: "
            + ", ".join(sorted(set(dropped_gb_refs))),
            stage="intent",
            code=DIAGNOSTIC_CODE_PII_GATE_HIT,
        )
    if original_gb and intent.grain == "grouped" and not kept_gb:
        raise ValueError(
            f"{FailureCategory.SENSITIVITY_ALL_GROUP_BY_DROPPED.value}: every GROUP BY expression "
            "references a hidden-sensitivity field; no valid grouping keys remain"
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
                    f"Dropping CTE {cte.cte_name!r} select column(s) referencing hidden-sensitivity fields: "
                    + ", ".join(sorted(set(blocked))),
                    stage="intent",
                    code=DIAGNOSTIC_CODE_PII_GATE_HIT,
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
                f"Dropping CTE {cte.cte_name!r} GROUP BY expression(s) referencing hidden-sensitivity fields: "
                + ", ".join(sorted(set(dropped_cte_gb))),
                stage="intent",
                code=DIAGNOSTIC_CODE_PII_GATE_HIT,
            )
        if orig_cte_gb and getattr(cte, "grain", "") == "grouped" and not kept_cte_gb:
            raise ValueError(
                f"{FailureCategory.SENSITIVITY_ALL_GROUP_BY_DROPPED.value}: CTE {cte.cte_name!r}: every GROUP BY "
                "expression references a hidden-sensitivity field; no valid grouping keys remain"
            )
        new_ctes.append(replace(cte, select_cols=kept_cte, group_by_cols=kept_cte_gb))
    return replace(intent, cte_steps=new_ctes)


def intent_text_has_leakable_placeholder(text: str | None) -> bool:
    """
    Return True if *text* still has angle-bracket or numeric placeholder tokens.

    Args:

        text: Raw expression or identifier substring from an intent field.

    Returns:

        True when a known placeholder pattern appears before repair.
    """
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


def _yield_filter_instructional_strings(fp: FilterParam) -> Iterator[str]:
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
        yield from _yield_filter_instructional_strings(branch.condition)
        yield from _yield_normalized_expr_instructional_strings(branch.result)
    if case_when.else_result is not None:
        yield from _yield_normalized_expr_instructional_strings(case_when.else_result)


def _yield_select_col_instructional_strings(col: SelectCol) -> Iterator[str]:
    """Yield strings from a SELECT column expression."""

    yield from _yield_normalized_expr_instructional_strings(col.expr)


def _yield_window_registry_step_instructional_strings(
    step: WindowRegistryStep,
) -> Iterator[str]:
    """Yield strings from a window registry row (id/label and nested expressions)."""

    yield step.registry_id
    yield from _yield_window_spec_instructional_strings(step.window_spec)


def _yield_case_registry_step_instructional_strings(
    step: CaseRegistryStep,
) -> Iterator[str]:
    """Yield strings from a case registry row (id/label and CASE body)."""

    yield step.registry_id
    yield step.label
    yield from _yield_case_when_instructional_strings(step.case_when)


def _yield_runtime_cte_step_instructional_strings(
    step: RuntimeCteStep,
) -> Iterator[str]:
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
    for fp in step.filters_param or []:
        yield from _yield_filter_instructional_strings(fp)
    for hp in step.having_param or []:
        yield from _yield_having_instructional_strings(hp)
    for w in step.window_registry or []:
        yield from _yield_window_registry_step_instructional_strings(w)
    for c in step.case_registry or []:
        yield from _yield_case_registry_step_instructional_strings(c)
    for val in (step.param_values or {}).values():
        yield from _yield_param_value_scan_strings(val)
    yield from step.output_columns or []
    yield from step.chosen_join_path_signature or []


def _yield_runtime_intent_instructional_scan_strings(
    intent: RuntimeIntent,
) -> Iterator[str]:
    """Yield structured intent strings to scan for instructional placeholders."""

    yield from intent.tables or []
    for col in intent.select_cols or []:
        yield from _yield_select_col_instructional_strings(col)
    for gb in intent.group_by_cols or []:
        yield from _yield_normalized_expr_instructional_strings(gb)
    for ob in intent.order_by_cols or []:
        yield from _yield_normalized_expr_instructional_strings(ob.expr)
    for fp in intent.filters_param or []:
        yield from _yield_filter_instructional_strings(fp)
    for hp in intent.having_param or []:
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


def _repair_intent_placeholder_normalized_expr(
    expr: NormalizedExpr,
    alias_map: dict[str, str],
) -> NormalizedExpr:
    """Rewrite placeholder table tokens inside a ``NormalizedExpr``."""

    if not alias_map and not INTENT_PLACEHOLDER_ANGLE_RE.search(_intent_expr_terms_blob(expr)):
        return expr

    def repl(term: str) -> str:
        return _rewrite_intent_placeholder_term(term, alias_map)

    return replace_refs_in_expr(expr, repl)


def _repair_intent_placeholder_filters(
    params: list[FilterParam],
    alias_map: dict[str, str],
) -> list[FilterParam]:
    """Repair filter left/right expressions for placeholder leaks."""

    out: list[FilterParam] = []
    for fp in params:
        le = _repair_intent_placeholder_normalized_expr(fp.left_expr, alias_map)
        rexp = _repair_intent_placeholder_normalized_expr(fp.right_expr, alias_map) if fp.right_expr else None
        out.append(replace(fp, left_expr=le, right_expr=rexp))
    return out


def _repair_intent_placeholder_having(
    params: list[HavingParam],
    alias_map: dict[str, str],
) -> list[HavingParam]:
    """Repair HAVING left/right expressions for placeholder leaks."""

    out: list[HavingParam] = []
    for hp in params:
        le = _repair_intent_placeholder_normalized_expr(hp.left_expr, alias_map)
        rexp = _repair_intent_placeholder_normalized_expr(hp.right_expr, alias_map) if hp.right_expr else None
        out.append(replace(hp, left_expr=le, right_expr=rexp))
    return out


def _repair_intent_placeholder_window_registry_step(
    step: WindowRegistryStep,
    alias_map: dict[str, str],
) -> WindowRegistryStep:
    """Repair window spec inside one window registry row."""

    return replace(
        step,
        window_spec=_repair_intent_placeholder_window_spec(step.window_spec, alias_map),
    )


def _repair_intent_placeholder_case_registry_step(
    step: CaseRegistryStep,
    alias_map: dict[str, str],
) -> CaseRegistryStep:
    """Repair CASE body inside one case registry row."""

    return replace(
        step,
        case_when=_repair_intent_placeholder_case_when(step.case_when, alias_map),
    )


def _repair_intent_placeholder_window_spec(
    ws: WindowSpec,
    alias_map: dict[str, str],
) -> WindowSpec:
    """Repair window partition, order, and argument expressions."""

    pb = [_repair_intent_placeholder_normalized_expr(e, alias_map) for e in ws.partition_by]
    ob = [replace(o, expr=_repair_intent_placeholder_normalized_expr(o.expr, alias_map)) for o in ws.order_by]
    arg = _repair_intent_placeholder_normalized_expr(ws.argument, alias_map) if ws.argument else None
    return replace(ws, partition_by=pb, order_by=ob, argument=arg)


def _repair_intent_placeholder_case_when(
    cw: CaseWhenExpr,
    alias_map: dict[str, str],
) -> CaseWhenExpr:
    """Repair CASE branches and else for placeholder leaks."""

    branches: list[CaseWhenBranch] = []
    for br in cw.branches:
        cond = _repair_intent_placeholder_filters([br.condition], alias_map)[0]
        res = _repair_intent_placeholder_normalized_expr(br.result, alias_map)
        branches.append(CaseWhenBranch(condition=cond, result=res))
    er = _repair_intent_placeholder_normalized_expr(cw.else_result, alias_map) if cw.else_result else None
    return CaseWhenExpr(branches=branches, else_result=er)


def _repair_intent_placeholder_select_cols(
    cols: list[SelectCol],
    alias_map: dict[str, str],
) -> list[SelectCol]:
    """Repair select list expressions."""

    out: list[SelectCol] = []
    for sc in cols:
        ex = _repair_intent_placeholder_normalized_expr(sc.expr, alias_map)
        out.append(replace(sc, expr=ex))
    return out


def _repair_intent_placeholder_order_by_cols(
    cols: list[OrderByCol],
    alias_map: dict[str, str],
) -> list[OrderByCol]:
    """Repair ORDER BY expressions for placeholder leaks."""

    return [replace(obc, expr=_repair_intent_placeholder_normalized_expr(obc.expr, alias_map)) for obc in cols]


def _repair_intent_placeholder_cte_step(
    step: RuntimeCteStep,
    alias_map: dict[str, str],
) -> RuntimeCteStep:
    """Repair one CTE step: selects, group/order, filters, having."""

    return replace(
        step,
        select_cols=_repair_intent_placeholder_select_cols(step.select_cols or [], alias_map),
        group_by_cols=[_repair_intent_placeholder_normalized_expr(g, alias_map) for g in (step.group_by_cols or [])],
        order_by_cols=_repair_intent_placeholder_order_by_cols(
            step.order_by_cols or [],
            alias_map,
        ),
        filters_param=_repair_intent_placeholder_filters(step.filters_param or [], alias_map),
        having_param=_repair_intent_placeholder_having(step.having_param or [], alias_map),
        window_registry=[
            _repair_intent_placeholder_window_registry_step(w, alias_map) for w in (step.window_registry or [])
        ],
        case_registry=[_repair_intent_placeholder_case_registry_step(c, alias_map) for c in (step.case_registry or [])],
    )


def repair_intent_placeholder_tokens(
    intent: RuntimeIntent,
    _schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Rewrite ``table_N``-style leaks using ``intent.tables`` sort order.

    Args:

        intent: Parsed intent that may contain instructional table tokens.

        _schema_graph: Unused; kept so callers pass the active ``SchemaGraph``.

    Returns:

        Intent with qualified prefixes rewritten to real table names when unambiguous from ``intent.tables``.
    """
    tables = list(intent.tables or [])
    if not tables:
        return intent
    alias_map_main = _intent_placeholder_table_alias_map(tables)
    sel = _repair_intent_placeholder_select_cols(intent.select_cols or [], alias_map_main)
    gb = [_repair_intent_placeholder_normalized_expr(g, alias_map_main) for g in (intent.group_by_cols or [])]
    ob = _repair_intent_placeholder_order_by_cols(intent.order_by_cols or [], alias_map_main)
    fp = _repair_intent_placeholder_filters(intent.filters_param or [], alias_map_main)
    hp = _repair_intent_placeholder_having(intent.having_param or [], alias_map_main)
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
        filters_param=fp,
        having_param=hp,
        cte_steps=ctes,
        window_registry=wr,
        case_registry=cr,
    )


def _flip_comparison_op(op: str) -> str:
    """Return the comparison operator to use after swapping left and right operands."""
    return OP_FLIP.get(op, op)


def _having_candidate_passes_numeric_and_group_rules(
    hp: HavingParam,
    *,
    group_by_cols: list[Any] | None,
) -> bool:
    """Return True when *hp* satisfies HAVING operator and GROUP BY presence rules."""

    if validate_having_operator_is_numeric([hp], "auto_repair"):
        return False
    if validate_having_requires_aggregation([hp], "auto_repair", group_by_cols=group_by_cols or []):
        return False
    return True


def auto_repair_filter_having(
    filters_param: list[FilterParam],
    having_param: list[HavingParam],
    *,
    group_by_cols: list[Any] | None = None,
) -> tuple[list[FilterParam], list[HavingParam]]:
    """
    Repair misplaced filter and HAVING conditions by moving or flipping them.

    Filters whose ``left_expr`` contains an aggregation are promoted to HAVING only when the candidate HAVING row uses a numeric comparison operator and GROUP BY is present. HAVING rows whose aggregation is on the right are flipped so the aggregation appears on the left. HAVING rows with no aggregation on either side are demoted to filters.
    """
    repaired_filters: list[FilterParam] = []
    repaired_having: list[HavingParam] = []
    for fp in filters_param or []:
        if fp.left_expr.has_aggregation:
            cand = HavingParam(
                left_expr=fp.left_expr,
                op=fp.op,
                right_expr=fp.right_expr,
                value_type=fp.value_type,
                param_key=fp.param_key,
                raw_value=fp.raw_value,
                bool_op=fp.bool_op,
                filter_group=fp.filter_group,
            )
            if _having_candidate_passes_numeric_and_group_rules(cand, group_by_cols=group_by_cols):
                repaired_having.append(cand)
                debug(f"[intent_repair.auto_repair_filter_having] filter->having: {fp.param_key}")
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
                    bool_op=hp.bool_op,
                    filter_group=hp.filter_group,
                )
            )
            debug(f"[intent_repair.auto_repair_filter_having] having flip (agg->left): {hp.param_key}")
        else:
            repaired_filters.append(
                FilterParam(
                    left_expr=hp.left_expr,
                    op=hp.op,
                    right_expr=hp.right_expr,
                    value_type=hp.value_type,
                    param_key=hp.param_key,
                    raw_value=hp.raw_value,
                    bool_op=hp.bool_op,
                    filter_group=hp.filter_group,
                )
            )
            debug(f"[intent_repair.auto_repair_filter_having] having->filter: {hp.param_key}")
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
    src_table: str | None,
    src_column: str,
    dst_table: str | None,
    dst_column: str,
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


def _transform_filter_param_expr(
    fp: FilterParam, transformer: Callable[[NormalizedExpr], NormalizedExpr]
) -> FilterParam:
    """Apply *transformer* to both sides of a filter param."""

    return FilterParam(
        left_expr=transformer(fp.left_expr),
        op=fp.op,
        right_expr=transformer(fp.right_expr) if fp.right_expr is not None else None,
        value_type=fp.value_type,
        param_key=fp.param_key,
        param_key_hi=fp.param_key_hi,
        raw_value=fp.raw_value,
        bool_op=fp.bool_op,
        filter_group=fp.filter_group,
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
        bool_op=hp.bool_op,
        filter_group=hp.filter_group,
    )


def _transform_window_spec(ws: WindowSpec, transformer: Callable[[NormalizedExpr], NormalizedExpr]) -> WindowSpec:
    """Apply *transformer* to partition_by, order_by expressions, and optional window argument."""

    new_part = [transformer(p) for p in (ws.partition_by or [])]
    new_orders = [replace(o, expr=transformer(o.expr)) for o in (ws.order_by or [])]
    new_arg = transformer(ws.argument) if ws.argument is not None else None
    return replace(ws, partition_by=new_part, order_by=new_orders, argument=new_arg)


def _transform_window_registry_steps(
    regs: list[WindowRegistryStep] | None,
    transformer: Callable[[NormalizedExpr], NormalizedExpr],
) -> list[WindowRegistryStep]:
    """Map *transformer* across every ``WindowRegistryStep.window_spec`` expression subtree."""

    out: list[WindowRegistryStep] = []
    for step in regs or []:
        out.append(replace(step, window_spec=_transform_window_spec(step.window_spec, transformer)))
    return out


def _transform_case_when_expr(
    cw: CaseWhenExpr,
    transformer: Callable[[NormalizedExpr], NormalizedExpr],
) -> CaseWhenExpr:
    """Apply *transformer* to branch conditions, branch results, and ``else_result``."""

    new_branches: list[CaseWhenBranch] = []
    for br in cw.branches or []:
        new_branches.append(
            CaseWhenBranch(
                condition=_transform_filter_param_expr(br.condition, transformer),
                result=transformer(br.result),
            )
        )
    new_else = transformer(cw.else_result) if cw.else_result is not None else None
    return replace(cw, branches=new_branches, else_result=new_else)


def _transform_case_registry_steps(
    regs: list[CaseRegistryStep] | None,
    transformer: Callable[[NormalizedExpr], NormalizedExpr],
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
        filters_param=[_transform_filter_param_expr(fp, transformer) for fp in (step.filters_param or [])],
        having_param=[_transform_having_param_expr(hp, transformer) for hp in (step.having_param or [])],
        window_registry=_transform_window_registry_steps(step.window_registry, transformer),
        case_registry=_transform_case_registry_steps(step.case_registry, transformer),
    )


def _walk_intent_normalized_exprs(
    intent: RuntimeIntent,
    transformer: Callable[[NormalizedExpr], NormalizedExpr],
) -> RuntimeIntent:
    """Map *transformer* across every NormalizedExpr in *intent* (top-level and CTE steps)."""

    return replace(
        intent,
        select_cols=[_transform_select_col_expr(sc, transformer) for sc in (intent.select_cols or [])],
        group_by_cols=[transformer(g) for g in (intent.group_by_cols or [])],
        order_by_cols=[_transform_order_by_col_expr(oc, transformer) for oc in (intent.order_by_cols or [])],
        filters_param=[_transform_filter_param_expr(fp, transformer) for fp in (intent.filters_param or [])],
        having_param=[_transform_having_param_expr(hp, transformer) for hp in (intent.having_param or [])],
        window_registry=_transform_window_registry_steps(intent.window_registry, transformer),
        case_registry=_transform_case_registry_steps(intent.case_registry, transformer),
        cte_steps=[_transform_cte_step_exprs(c, transformer) for c in (intent.cte_steps or [])],
    )


def _apply_column_replacer_to_intent(
    intent: RuntimeIntent,
    replacer: Callable[[str], str],
) -> RuntimeIntent:
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
    for fp in intent.filters_param or []:
        for ref in extract_columns_from_expr(fp.left_expr):
            t, c = _split_qualified_ref(ref)
            if t == table_low and c not in seen:
                seen.append(c)
    return seen


def _table_columns_from_schema(schema: SchemaGraph, table: str) -> list[str]:
    """Return lowercase column names for *table* in *schema*; empty when missing."""

    table_low = table.strip().lower()
    meta = schema.tables.get(table_low) if schema and schema.tables else None
    if meta is None:
        return []
    return [c.strip().lower() for c in meta.columns.keys()]


def _fuzzy_pick(target: str, candidates: Sequence[str]) -> str | None:
    """Return the closest match in *candidates* to *target* using ratio cutoff, or None."""

    pool = [c for c in candidates if c]
    if not pool:
        return None
    matches = get_close_matches(target.strip().lower(), pool, n=1, cutoff=DIAGNOSTIC_FUZZY_CUTOFF)
    return matches[0] if matches else None


def _repair_unknown_column(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """Rewrite an unknown column to its closest schema-known sibling on the same table."""

    raw = (diag.offending_identifier or "").strip()
    if not raw:
        return None
    src_table, src_column = _split_qualified_ref(raw)
    if not src_column:
        return None
    if src_table is not None:
        candidates = _table_columns_from_schema(schema, src_table)
        best = _fuzzy_pick(src_column, candidates)
        if best is None or best == src_column:
            return None
        replacer = _build_column_term_replacer(src_table, src_column, src_table, best)
        debug(f"[intent_repair._repair_unknown_column] {src_table}.{src_column} -> {src_table}.{best}")
        return _apply_column_replacer_to_intent(intent, replacer)
    best_table: str | None = None
    best_column: str | None = None
    for t in intent.tables or []:
        cands = _table_columns_from_schema(schema, t)
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


def _repair_ambiguous_column(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """Qualify an ambiguous bare column with the first owner table that appears in ``intent.tables``."""

    raw = (diag.offending_identifier or "").strip().lower()
    if not raw or "." in raw:
        return None
    owners_csv = (diag.details.get("owners") or "").strip()
    owners = [o.strip().lower() for o in owners_csv.split(",") if o.strip()] if owners_csv else []
    if not owners:
        owners = [t.strip().lower() for t in (intent.tables or []) if raw in _table_columns_from_schema(schema, t)]
    if not owners:
        return None
    intent_tables = [t.strip().lower() for t in (intent.tables or [])]
    chosen: str | None = None
    for t in intent_tables:
        if t in owners:
            chosen = t
            break
    if chosen is None:
        chosen = owners[0]
    replacer = _build_column_term_replacer(None, raw, chosen, raw)
    debug(f"[intent_repair._repair_ambiguous_column] {raw} -> {chosen}.{raw}")
    return _apply_column_replacer_to_intent(intent, replacer)


def _repair_unknown_table(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """Rewrite an unknown table to its closest schema-known sibling and retarget column refs."""

    raw = (diag.offending_identifier or "").strip().lower()
    if not raw:
        return None
    candidates = list(schema.tables.keys()) if schema and schema.tables else []
    best = _fuzzy_pick(raw, candidates)
    if best is None or best == raw:
        return None
    new_tables = [best if (t.strip().lower() == raw) else t for t in (intent.tables or [])]
    if new_tables == list(intent.tables or []):
        return None
    replacer = _build_table_term_replacer(raw, best)
    rewritten = _apply_column_replacer_to_intent(intent, replacer)
    debug(f"[intent_repair._repair_unknown_table] {raw} -> {best}")
    return replace(rewritten, tables=new_tables)


def _repair_grain_consistency(
    intent: RuntimeIntent,
    _schema: SchemaGraph,
    diag: SqlDiagnostic,
) -> RuntimeIntent | None:
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


def _repair_agg_in_where(
    intent: RuntimeIntent,
    _schema: SchemaGraph,
    _diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """Promote any aggregation-bearing WHERE filter into HAVING via :func:`auto_repair_filter_having`."""

    new_filters, new_having = auto_repair_filter_having(
        intent.filters_param or [],
        intent.having_param or [],
        group_by_cols=intent.group_by_cols or [],
    )
    if new_filters == list(intent.filters_param or []) and new_having == list(intent.having_param or []):
        return None
    debug("[intent_repair._repair_agg_in_where] filter->having promotion applied")
    return replace(intent, filters_param=new_filters, having_param=new_having)


def _repair_cartesian(
    intent: RuntimeIntent,
    _schema: SchemaGraph,
    _diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """Clear the chosen join candidate so the next render re-selects an explicit join path."""

    if not intent.chosen_join_candidate_id and not intent.chosen_join_path_signature:
        return None
    debug("[intent_repair._repair_cartesian] clearing chosen_join_candidate_id and signature")
    return replace(intent, chosen_join_candidate_id="", chosen_join_path_signature=[])


def _repair_filter_overlap(
    intent: RuntimeIntent,
    _schema: SchemaGraph,
    _diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """De-duplicate contradictory filters; return new intent only when something changed."""

    repaired = dedup_contradictory_filters(intent)
    if repaired is intent:
        return None
    return repaired


def _repair_param_binding(
    intent: RuntimeIntent,
    _schema: SchemaGraph,
    diag: SqlDiagnostic,
) -> RuntimeIntent | None:
    """Drop a filter that references an unbound parameter when it has no literal raw_value."""

    target = (diag.offending_identifier or "").strip()
    if not target:
        return None
    target_low = target.lstrip(":").lower()
    new_filters: list[FilterParam] = []
    dropped = False
    for fp in intent.filters_param or []:
        keys = {(fp.param_key or "").lower(), (fp.param_key_hi or "").lower()}
        if target_low in keys and fp.raw_value is None:
            dropped = True
            continue
        new_filters.append(fp)
    if not dropped:
        return None
    debug(f"[intent_repair._repair_param_binding] dropped unbound-param filter: {target_low}")
    return replace(intent, filters_param=new_filters)


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
        ("explain_zero_estimate", _repair_filter_overlap),
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
    """
    Apply structural repairs for each actionable diagnostic; cap attempts per code.

    Soft diagnostics (``SOFT_DIAGNOSTIC_CODES``) are skipped because they convey EXPLAIN-plan hints rather than structural defects. Returns the rewritten intent and a flag indicating whether any repair primitive returned a non-``None`` result.
    """

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
