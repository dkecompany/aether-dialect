"""Column resolution, grain rules, schema checks, and expression simplification."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import sqlglot
from sqlglot import exp

from ._config import NULL_CHECK_OPS, REVERSE_OP_MAP, normalize_value_type
from ._contracts_base import (
    CteIntent,
    FailureCategory,
    IntentIssue,
    LITERAL_BEARING_CATEGORIES,
    LogicalIntent,
    SchemaGraph,
    make_intent_issue,
)
from ._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ExprValue,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    WindowRegistryStep,
    effective_select_parts,
    expr_registry_ref,
)
from ._core_utils import debug, normalize_op, pipeline_trace_lazy, stable_json
from ._intent_expr import (
    classify_cte_expr,
    concat_logical_intent_prose,
    extract_columns_from_expr,
    replace_refs_in_expr,
)
from ._intent_repair import (
    best_descriptive_column,
    cols_from_named_registries,
    cols_from_select_col,
    infer_cte_output_columns,
)
from ._sql_gen import render_expr_sql


def normalize_count_star(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Convert COUNT(1) references to COUNT(*) throughout an intent for consistency.

    Args:

        intent: RuntimeIntent to normalize.

    Returns:

        New RuntimeIntent with COUNT(1) rewritten as COUNT(*) in all expressions.
    """

    def _normalize_group(g: MulGroup) -> MulGroup:
        new_multiply = [_normalize_expr(m) for m in g.multiply]
        new_divide = [_normalize_expr(d) for d in g.divide]
        if (g.agg_func or "").lower() == "count" and len(new_multiply) == 1:
            leaf = new_multiply[0]
            is_one = False
            if leaf.column_ref in {"*", "1"}:
                is_one = True
            elif (
                leaf.add_values
                and not leaf.add_groups
                and not leaf.sub_groups
                and not leaf.sub_values
                and len(leaf.add_values) == 1
            ):
                try:
                    is_one = float(leaf.add_values[0].value) == 1.0
                except (TypeError, ValueError):
                    is_one = False
            if is_one:
                new_multiply = [NormalizedExpr(star=True)]
        return replace(g, multiply=new_multiply, divide=new_divide)

    def _normalize_expr(expr: NormalizedExpr) -> NormalizedExpr:
        if not isinstance(expr, NormalizedExpr):
            return expr
        new_add = [_normalize_group(g) for g in expr.add_groups]
        new_sub = [_normalize_group(g) for g in expr.sub_groups]
        return replace(expr, add_groups=new_add, sub_groups=new_sub)

    def _fix_filter_list(params):
        return [
            replace(
                fp,
                left_expr=_normalize_expr(fp.left_expr),
                right_expr=(_normalize_expr(fp.right_expr) if fp.right_expr else None),
            )
            for fp in params
        ]

    new_select_cols = [replace(sc, expr=_normalize_expr(sc.expr)) for sc in (intent.select_cols or [])]
    new_order_by_cols = [replace(obc, expr=_normalize_expr(obc.expr)) for obc in (intent.order_by_cols or [])]
    new_filters = _fix_filter_list(intent.filters_param or [])
    new_having = _fix_filter_list(intent.having_param or [])
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_sc = [replace(sc, expr=_normalize_expr(sc.expr)) for sc in (cte.select_cols or [])]
        cte_obc = [replace(obc, expr=_normalize_expr(obc.expr)) for obc in (cte.order_by_cols or [])]
        cte_fp = _fix_filter_list(cte.filters_param or [])
        cte_hp = _fix_filter_list(cte.having_param or [])
        new_cte_steps.append(
            replace(
                cte,
                select_cols=cte_sc,
                order_by_cols=cte_obc,
                filters_param=cte_fp,
                having_param=cte_hp,
            )
        )
    return replace(
        intent,
        select_cols=new_select_cols,
        order_by_cols=new_order_by_cols,
        filters_param=new_filters,
        having_param=new_having,
        cte_steps=new_cte_steps,
    )


def _is_row_count_count_mulgroup(group: MulGroup) -> bool:
    """Return True when *group* is a COUNT over all rows (``*`` or ``1``)."""

    if (group.agg_func or "").lower() != "count":
        return False
    if not group.multiply:
        return True
    if len(group.multiply) == 1:
        leaf = group.multiply[0]
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


def _rewrite_row_count_count_in_expr(expr: NormalizedExpr, qualified_pk: str) -> NormalizedExpr:
    """Rewrite COUNT(*) / COUNT(1) mul-groups to COUNT(*qualified_pk*)."""

    def map_groups(groups: list[MulGroup]) -> list[MulGroup]:
        mapped: list[MulGroup] = []
        for g in groups:
            if _is_row_count_count_mulgroup(g):
                mapped.append(replace(g, multiply=[NormalizedExpr.from_column(qualified_pk)]))
            else:
                mapped.append(g)
        return mapped

    return replace(
        expr,
        add_groups=map_groups(list(expr.add_groups)),
        sub_groups=map_groups(list(expr.sub_groups)),
    )


def _first_primary_key_fq(schema_graph: SchemaGraph, table: str) -> str | None:
    """Return ``table.column`` for the first declared primary key column, if any."""

    meta = schema_graph.tables.get(table)
    if not meta:
        return None
    for col_name in sorted(meta.columns.keys()):
        if meta.columns[col_name].is_primary_key:
            return f"{table}.{col_name}"
    return None


def qualify_cte_count_star_mulgroups(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Replace CTE SELECT row-count COUNT forms with COUNT(primary_key) for single-table CTEs.

    Args:

        intent: Runtime intent with optional ``cte_steps``.

        schema_graph: Schema metadata for PK lookup.

    Returns:

        Intent with qualified COUNT arguments where applicable.
    """

    steps = intent.cte_steps or []
    if not steps:
        return intent
    new_steps: list[RuntimeCteStep] = []
    changed = False
    for cte in steps:
        base_tables = [t for t in (cte.tables or []) if t in schema_graph.tables]
        if len(base_tables) != 1:
            new_steps.append(cte)
            continue
        pk_fq = _first_primary_key_fq(schema_graph, base_tables[0])
        if not pk_fq:
            new_steps.append(cte)
            continue
        new_cols: list[SelectCol] = []
        for sc in cte.select_cols or []:
            new_expr = _rewrite_row_count_count_in_expr(sc.expr, pk_fq)
            if new_expr != sc.expr:
                changed = True
            new_cols.append(replace(sc, expr=new_expr))
        new_steps.append(replace(cte, select_cols=new_cols))
    if not changed:
        return intent
    return replace(intent, cte_steps=new_steps)


def _build_registry_canonical_rename(
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
) -> dict[str, str]:
    """Return a rename map from declared registry ids to canonical ``w0N`` / ``c0N`` ids."""

    rename: dict[str, str] = {}
    for idx, step in enumerate(window_registry or []):
        old = (step.registry_id or "").strip()
        new = f"w{idx + 1:02d}"
        if old and old != new and old not in rename:
            rename[old] = new
    for idx, step in enumerate(case_registry or []):
        old = (step.registry_id or "").strip()
        new = f"c{idx + 1:02d}"
        if old and old != new and old not in rename:
            rename[old] = new
    return rename


def _rename_filter_param_refs(fp: FilterParam, rename: dict[str, str]) -> FilterParam:
    """Apply *rename* to a `FilterParam`'s left and right exprs."""

    new_left = replace_refs_in_expr(fp.left_expr, lambda r: rename.get(r, r))
    new_right = replace_refs_in_expr(fp.right_expr, lambda r: rename.get(r, r)) if fp.right_expr is not None else None
    return replace(fp, left_expr=new_left, right_expr=new_right)


def _rename_having_param_refs(hp: HavingParam, rename: dict[str, str]) -> HavingParam:
    """Apply *rename* to a `HavingParam`'s left and right exprs."""

    new_left = replace_refs_in_expr(hp.left_expr, lambda r: rename.get(r, r))
    new_right = replace_refs_in_expr(hp.right_expr, lambda r: rename.get(r, r)) if hp.right_expr is not None else None
    return replace(hp, left_expr=new_left, right_expr=new_right)


def _rename_case_when_refs(cw: CaseWhenExpr | None, rename: dict[str, str]) -> CaseWhenExpr | None:
    """Apply *rename* to every condition and result expr inside *cw*."""

    if cw is None:
        return None
    new_branches: list[CaseWhenBranch] = []
    for br in cw.branches or []:
        new_branches.append(
            CaseWhenBranch(
                condition=_rename_filter_param_refs(br.condition, rename),
                result=replace_refs_in_expr(br.result, lambda r: rename.get(r, r)),
            )
        )
    new_else = replace_refs_in_expr(cw.else_result, lambda r: rename.get(r, r)) if cw.else_result is not None else None
    return replace(cw, branches=new_branches, else_result=new_else)


def _rename_select_col_refs(sc: SelectCol, rename: dict[str, str]) -> SelectCol:
    """Apply *rename* across a select column expression."""

    return replace(sc, expr=replace_refs_in_expr(sc.expr, lambda r: rename.get(r, r)))


def _rename_window_registry_steps(
    regs: list[WindowRegistryStep],
    rename: dict[str, str],
) -> list[WindowRegistryStep]:
    """Apply *rename* to ``partition_by``, ``order_by``, and ``argument`` expressions."""

    out: list[WindowRegistryStep] = []
    repl = lambda r: rename.get(r, r)
    for step in regs or []:
        ws = step.window_spec
        np = [replace_refs_in_expr(p, repl) for p in (ws.partition_by or [])]
        no = [replace(o, expr=replace_refs_in_expr(o.expr, repl)) for o in (ws.order_by or [])]
        na = replace_refs_in_expr(ws.argument, repl) if ws.argument is not None else None
        out.append(replace(step, window_spec=replace(ws, partition_by=np, order_by=no, argument=na)))
    return out


def _rename_case_registry_steps(
    regs: list[CaseRegistryStep],
    rename: dict[str, str],
) -> list[CaseRegistryStep]:
    """Apply *rename* to each ``case_when`` subtree."""

    out: list[CaseRegistryStep] = []
    for step in regs or []:
        cw = _rename_case_when_refs(step.case_when, rename)
        fixed = cw if cw is not None else CaseWhenExpr()
        out.append(replace(step, case_when=fixed))
    return out


def _canonicalize_scope(
    select_cols: list[SelectCol],
    group_by_cols: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    filters_param: list[FilterParam],
    having_param: list[HavingParam],
    window_registry: list[WindowRegistryStep],
    case_registry: list[CaseRegistryStep],
) -> tuple[
    list[SelectCol],
    list[NormalizedExpr],
    list[OrderByCol],
    list[FilterParam],
    list[HavingParam],
    list[WindowRegistryStep],
    list[CaseRegistryStep],
]:
    """Renumber registry ids in one query scope and apply the rename to all clause exprs."""

    rename = _build_registry_canonical_rename(window_registry, case_registry)
    new_window_registry = [
        replace(step, registry_id=f"w{idx + 1:02d}") for idx, step in enumerate(window_registry or [])
    ]
    new_case_registry = [replace(step, registry_id=f"c{idx + 1:02d}") for idx, step in enumerate(case_registry or [])]
    if not rename:
        return (
            list(select_cols or []),
            list(group_by_cols or []),
            list(order_by_cols or []),
            list(filters_param or []),
            list(having_param or []),
            new_window_registry,
            new_case_registry,
        )
    repl = lambda r: rename.get(r, r)
    new_select = [_rename_select_col_refs(sc, rename) for sc in select_cols or []]
    new_group_by = [replace_refs_in_expr(g, repl) for g in group_by_cols or []]
    new_order_by = [replace(obc, expr=replace_refs_in_expr(obc.expr, repl)) for obc in order_by_cols or []]
    new_filters = [_rename_filter_param_refs(fp, rename) for fp in filters_param or []]
    new_having = [_rename_having_param_refs(hp, rename) for hp in having_param or []]
    wr_expr = _rename_window_registry_steps(window_registry or [], rename)
    cr_expr = _rename_case_registry_steps(case_registry or [], rename)
    new_window_registry = [replace(step, registry_id=f"w{idx + 1:02d}") for idx, step in enumerate(wr_expr)]
    new_case_registry = [replace(step, registry_id=f"c{idx + 1:02d}") for idx, step in enumerate(cr_expr)]
    return (
        new_select,
        new_group_by,
        new_order_by,
        new_filters,
        new_having,
        new_window_registry,
        new_case_registry,
    )


def canonicalize_registry_ids(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Renumber per-scope ``window_registry`` / ``case_registry`` ids to canonical ``w0N`` / ``c0N``.

    The LLM may emit registry ids in arbitrary shapes (``w1``, ``myrank``, ``c_status``). This step rewrites each scope's registry ids in declaration order and applies the same rename to every reference (a bare ``column_ref`` matching an old id) inside the scope's select, group_by, order_by, filters, and having clauses, including nested expressions inside ``window_registry`` and ``case_registry`` rows. Real schema columns are qualified ``table.column`` tokens and never collide with old registry ids.
    """

    new_select, new_group, new_order, new_filters, new_having, new_wr, new_cr = _canonicalize_scope(
        intent.select_cols or [],
        intent.group_by_cols or [],
        intent.order_by_cols or [],
        intent.filters_param or [],
        intent.having_param or [],
        intent.window_registry or [],
        intent.case_registry or [],
    )
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        c_select, c_group, c_order, c_filters, c_having, c_wr, c_cr = _canonicalize_scope(
            cte.select_cols or [],
            cte.group_by_cols or [],
            cte.order_by_cols or [],
            cte.filters_param or [],
            cte.having_param or [],
            cte.window_registry or [],
            cte.case_registry or [],
        )
        new_cte_steps.append(
            replace(
                cte,
                select_cols=c_select,
                group_by_cols=c_group,
                order_by_cols=c_order,
                filters_param=c_filters,
                having_param=c_having,
                window_registry=c_wr,
                case_registry=c_cr,
            )
        )
    return replace(
        intent,
        select_cols=new_select,
        group_by_cols=new_group,
        order_by_cols=new_order,
        filters_param=new_filters,
        having_param=new_having,
        window_registry=new_wr,
        case_registry=new_cr,
        cte_steps=new_cte_steps,
    )


def sort_select_cols(cols: list[SelectCol]) -> list[SelectCol]:
    """
    Sort select columns so non-aggregated expressions come before aggregated ones and ties are broken by expression signature.

    Args:

        cols: List of SelectCol objects to sort.

    Returns:

        Sorted list of SelectCol objects.
    """

    def key_fn(sc: SelectCol) -> tuple[int, str]:
        return (1 if sc.is_aggregated else 0, sc.signature_key)

    return sorted(cols, key=key_fn)


def _filter_structural_key(fp: FilterParam) -> tuple[str, str, str, str]:
    """
    Return the structural sort key for a single FilterParam.

    Args:

        fp: FilterParam to compute the key for.

    Returns:

        Tuple of (left_sig, op, right_sig, value_type) with all components lowercased.
    """
    left = fp.left_expr.signature_key if fp.left_expr else ""
    right = fp.right_expr.signature_key if fp.right_expr else ""
    return (left, fp.op.lower(), right, fp.value_type.lower())


def _having_structural_key(hp: HavingParam) -> tuple[str, str, str, str]:
    """
    Return the structural sort key for a single HavingParam.

    Args:

        hp: HavingParam to compute the key for.

    Returns:

        Tuple of (left_sig, op, right_sig, value_type) with all components lowercased.
    """
    left = hp.left_expr.signature_key if hp.left_expr else ""
    right = hp.right_expr.signature_key if hp.right_expr else ""
    return (left, hp.op.lower(), right, hp.value_type.lower())


def _forward_links(items: Sequence[FilterParam | HavingParam]) -> list[str]:
    """
    Return the boolean connector after each item toward the next fragment.

    For every index ``i < len(items) - 1`` the value is taken from ``items[i].bool_op`` (normalized to ``AND`` or ``OR``). The final entry is a sentinel ``AND`` for renderers that join fragments.
    """

    n = len(items)
    if n == 0:
        return []
    out: list[str] = []
    for i in range(n):
        if i < n - 1:
            raw = getattr(items[i], "bool_op", None) or "AND"
            op = raw.strip().upper()
            out.append(op if op == "OR" else "AND")
        else:
            out.append("AND")
    return out


def _canonicalize_condition_order(
    items: list,
    structural_key_fn: Callable,
) -> list:
    """
    Reorder AND/OR filter chains canonically and fix ``bool_op`` links.

    Args:

        items: Flat ``FilterParam`` or ``HavingParam`` list.

        structural_key_fn: Sort key for one condition.

    Returns:

        Same items in canonical order with updated ``bool_op`` values.
    """
    if len(items) <= 1:
        return list(items)
    links = _forward_links(items)
    trailing = getattr(items[-1], "bool_op", None) or "AND"
    trailing_norm = trailing.strip().upper()
    trailing_effective = trailing_norm if trailing_norm == "OR" else "AND"
    ops: list[str] = links[:-1]
    chunks: list[list] = []
    current_chunk: list = [items[0]]
    for i, op in enumerate(ops):
        if op == "OR":
            chunks.append(current_chunk)
            current_chunk = [items[i + 1]]
        else:
            current_chunk.append(items[i + 1])
    chunks.append(current_chunk)
    sorted_chunks: list[list] = []
    for chunk in chunks:
        sorted_chunks.append(sorted(chunk, key=structural_key_fn))
    sorted_chunks.sort(key=lambda ch: structural_key_fn(ch[0]))
    result: list = []
    for ci, chunk in enumerate(sorted_chunks):
        for fi, item in enumerate(chunk):
            is_last_in_chunk = fi == len(chunk) - 1
            is_last_chunk = ci == len(sorted_chunks) - 1
            if is_last_chunk and is_last_in_chunk:
                new_bool_op = trailing_effective
            elif is_last_in_chunk:
                new_bool_op = "OR"
            else:
                new_bool_op = "AND"
            result.append(replace(item, bool_op=new_bool_op))
    return result


def _shift_multi_group_representative_filters_forward(
    representatives: list[FilterParam],
) -> list[FilterParam]:
    """
    Convert backward ``bool_op`` on each group's last filter to forward links between representatives.

    Args:

        representatives: One representative per ``filter_group`` (already canonicalized within each group).

    Returns:

        Representatives with ``bool_op`` on item ``i`` meaning the connector after ``i`` toward ``i+1``.
    """

    if len(representatives) <= 1:
        return list(representatives)
    out: list[FilterParam] = []
    for i, rep in enumerate(representatives):
        if i < len(representatives) - 1:
            raw = rep.bool_op or "AND"
            connector = raw.strip().upper()
            if connector != "OR":
                connector = "AND"
            out.append(replace(rep, bool_op=connector))
        else:
            out.append(rep)
    return out


def _shift_multi_group_representative_having_forward(
    representatives: list[HavingParam],
) -> list[HavingParam]:
    """Same as ``_shift_multi_group_representative_filters_forward`` for ``HavingParam`` lists."""

    if len(representatives) <= 1:
        return list(representatives)
    out: list[HavingParam] = []
    for i, rep in enumerate(representatives):
        if i < len(representatives) - 1:
            raw = rep.bool_op or "AND"
            connector = raw.strip().upper()
            if connector != "OR":
                connector = "AND"
            out.append(replace(rep, bool_op=connector))
        else:
            out.append(rep)
    return out


def _clamp_negative_filter_group_filter(fp: FilterParam) -> FilterParam:
    fg = fp.filter_group
    if fg is not None and fg < 0:
        return replace(fp, filter_group=None)
    return fp


def _clamp_negative_filter_group_having(hp: HavingParam) -> HavingParam:
    fg = hp.filter_group
    if fg is not None and fg < 0:
        return replace(hp, filter_group=None)
    return hp


def _coerce_filter_group_list(filters: list[FilterParam]) -> list[FilterParam]:
    if not filters:
        return []
    items = [_clamp_negative_filter_group_filter(fp) for fp in filters]

    any_grouped = any(fp.filter_group is not None for fp in items)
    if not any_grouped:
        if len(items) <= 1:
            return items
        first_b = (items[0].bool_op or "AND").strip().upper()
        rest_all_or = all((items[j].bool_op or "AND").strip().upper() == "OR" for j in range(1, len(items)))
        # Backward-style LLM pattern: first row keeps default AND while every later row
        # carries OR (meaning "OR with the previous row" in prompt terms). Promote to
        # one disjunct per row so SQL matches A OR B OR C. Forward mixed AND/OR chains
        # (OR only on some interior rows) stay flat — see combinatorial matrix §5.
        if first_b == "AND" and rest_all_or:
            return [replace(fp, filter_group=gid, bool_op="AND") for gid, fp in enumerate(items, start=1)]
        return items

    max_gid = max((fp.filter_group for fp in items if fp.filter_group is not None), default=-1)
    next_gid = max_gid + 1
    out: list[FilterParam] = []
    for fp in items:
        if fp.filter_group is None:
            out.append(replace(fp, filter_group=next_gid, bool_op="AND"))
            next_gid += 1
        else:
            out.append(replace(fp, bool_op="AND"))
    return out


def _coerce_having_group_list(having: list[HavingParam]) -> list[HavingParam]:
    if not having:
        return []
    items = [_clamp_negative_filter_group_having(hp) for hp in having]

    any_grouped = any(hp.filter_group is not None for hp in items)
    if not any_grouped:
        if len(items) <= 1:
            return items
        first_b = (items[0].bool_op or "AND").strip().upper()
        rest_all_or = all((items[j].bool_op or "AND").strip().upper() == "OR" for j in range(1, len(items)))
        if first_b == "AND" and rest_all_or:
            return [replace(hp, filter_group=gid, bool_op="AND") for gid, hp in enumerate(items, start=1)]
        return items

    max_gid = max((hp.filter_group for hp in items if hp.filter_group is not None), default=-1)
    next_gid = max_gid + 1
    out: list[HavingParam] = []
    for hp in items:
        if hp.filter_group is None:
            out.append(replace(hp, filter_group=next_gid, bool_op="AND"))
            next_gid += 1
        else:
            out.append(replace(hp, bool_op="AND"))
    return out


def coerce_filter_group_mode(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Normalise ``filter_group`` / ``bool_op`` wiring before ``normalize_filters_havings``.

    Mixed grouped/``None`` rows assign fresh ``filter_group`` ids to ``None`` rows. Flat
    bodies stay flat except for a backward-style pattern where the first row keeps
    default ``AND`` and every following row carries ``OR`` (one disjunct per row).
    Negative ``filter_group`` values clamp to ``None`` before those rules apply.
    """

    new_filters = _coerce_filter_group_list(list(intent.filters_param or []))
    new_having = _coerce_having_group_list(list(intent.having_param or []))
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        nf = _coerce_filter_group_list(list(cte.filters_param or []))
        nh = _coerce_having_group_list(list(cte.having_param or []))
        new_ctes.append(replace(cte, filters_param=nf, having_param=nh))
    return replace(intent, filters_param=new_filters, having_param=new_having, cte_steps=new_ctes)


def sort_filters(filters: list[FilterParam]) -> list[FilterParam]:
    """
    Canonicalize ``filters_param`` ordering.

    Flat mode (no ``filter_group``): preserve existing AND/OR chunk canonicalisation.

    Grouped mode (any row has ``filter_group``): bucket by group id in first-seen order,
    sort within each bucket structurally, and set ``bool_op`` to ``AND`` on every row
    (renderers join disjuncts with OR between buckets).
    """

    if not filters:
        return []
    if any(fp.filter_group is not None for fp in filters):
        filters = _coerce_filter_group_list(list(filters))
    grouped = any(fp.filter_group is not None for fp in filters)
    if not grouped:
        buckets: dict[int | None, list[FilterParam]] = defaultdict(list)
        for fp in filters:
            buckets[fp.filter_group].append(fp)
        canonicalized_groups: list[tuple[int | None, list[FilterParam]]] = []
        for gid, group in buckets.items():
            canonicalized_groups.append((gid, _canonicalize_condition_order(group, _filter_structural_key)))
        if len(canonicalized_groups) == 1:
            return canonicalized_groups[0][1]
        representatives: list[FilterParam] = []
        group_map: dict[int, tuple[int | None, list[FilterParam]]] = {}
        for idx, (gid, group) in enumerate(canonicalized_groups):
            rep = group[-1]
            representatives.append(replace(rep, filter_group=idx))
            group_map[idx] = (gid, group)
        representatives = _shift_multi_group_representative_filters_forward(representatives)
        sorted_reps = _canonicalize_condition_order(representatives, _filter_structural_key)
        result: list[FilterParam] = []
        for _ri, rep in enumerate(sorted_reps):
            proxy_id = rep.filter_group
            assert isinstance(proxy_id, int)
            real_gid, group = group_map[proxy_id]
            inter_connector = rep.bool_op
            for fi, fp in enumerate(group):
                if fi == len(group) - 1:
                    result.append(replace(fp, bool_op=inter_connector, filter_group=real_gid))
                else:
                    result.append(replace(fp, filter_group=real_gid))
        return result

    ordered_ids: list[int] = []
    by_gid: dict[int, list[FilterParam]] = {}
    for fp in filters:
        gid_f = fp.filter_group
        assert gid_f is not None
        if gid_f not in by_gid:
            ordered_ids.append(gid_f)
            by_gid[gid_f] = []
        by_gid[gid_f].append(fp)
    out: list[FilterParam] = []
    for gid_f in ordered_ids:
        bucket = sorted(by_gid[gid_f], key=_filter_structural_key)
        for fp in bucket:
            out.append(replace(fp, bool_op="AND", filter_group=gid_f))
    return out


def sort_having(having: list[HavingParam]) -> list[HavingParam]:
    """
    Same as :func:`sort_filters` but for ``HavingParam`` lists.
    """

    if not having:
        return []
    if any(hp.filter_group is not None for hp in having):
        having = _coerce_having_group_list(list(having))
    grouped = any(hp.filter_group is not None for hp in having)
    if not grouped:
        buckets: dict[int | None, list[HavingParam]] = defaultdict(list)
        for hp in having:
            buckets[hp.filter_group].append(hp)
        canonicalized_groups: list[tuple[int | None, list[HavingParam]]] = []
        for gid, group in buckets.items():
            canonicalized_groups.append((gid, _canonicalize_condition_order(group, _having_structural_key)))
        if len(canonicalized_groups) == 1:
            return canonicalized_groups[0][1]
        representatives: list[HavingParam] = []
        group_map: dict[int, tuple[int | None, list[HavingParam]]] = {}
        for idx, (gid, group) in enumerate(canonicalized_groups):
            rep = group[-1]
            representatives.append(replace(rep, filter_group=idx))
            group_map[idx] = (gid, group)
        representatives = _shift_multi_group_representative_having_forward(representatives)
        sorted_reps = _canonicalize_condition_order(representatives, _having_structural_key)
        result: list[HavingParam] = []
        for _ri, rep in enumerate(sorted_reps):
            proxy_id = rep.filter_group
            assert isinstance(proxy_id, int)
            real_gid, group = group_map[proxy_id]
            inter_connector = rep.bool_op
            for fi, hp in enumerate(group):
                if fi == len(group) - 1:
                    result.append(replace(hp, bool_op=inter_connector, filter_group=real_gid))
                else:
                    result.append(replace(hp, filter_group=real_gid))
        return result

    ordered_ids: list[int] = []
    by_gid: dict[int, list[HavingParam]] = {}
    for hp in having:
        gid_h = hp.filter_group
        assert gid_h is not None
        if gid_h not in by_gid:
            ordered_ids.append(gid_h)
            by_gid[gid_h] = []
        by_gid[gid_h].append(hp)
    out: list[HavingParam] = []
    for gid_h in ordered_ids:
        bucket = sorted(by_gid[gid_h], key=_having_structural_key)
        for hp in bucket:
            out.append(replace(hp, bool_op="AND", filter_group=gid_h))
    return out


def _is_cte_output_groupable(term: str, cte_steps: list[RuntimeCteStep]) -> bool:
    """Return True if term references a CTE output column."""
    if "." not in term:
        return False
    table_part, col_part = term.split(".", 1)
    table_lower = table_part.strip().lower()
    col_lower = col_part.strip().lower()
    for cte in cte_steps or []:
        if cte.cte_name.lower() == table_lower:
            out_cols = cte.output_columns or []
            return any(c.strip().lower() == col_lower for c in out_cols)
    return False


def _select_carries_aggregation(
    sc: SelectCol,
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
) -> bool:
    """Return True when *sc* carries SQL aggregation that participates in GROUP BY mixing rules."""

    parts = effective_select_parts(sc, window_registry, case_registry)
    if parts.window_spec is not None:
        return False
    if expr_registry_ref(sc.expr) is not None:
        return False
    return parts.expr.has_aggregation


def enforce_grain_consistency(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Infer or extend ``group_by_cols`` when mixed agg/non-agg selects imply grouping.

    Args:

        intent: Intent to adjust.

        schema_graph: Column roles and FK metadata.

    Returns:

        Intent with ``grain='grouped'`` and descriptive columns added when needed.
    """
    group_by = list(intent.group_by_cols or [])
    select_cols = list(intent.select_cols or [])
    cte_steps = intent.cte_steps or []
    if not group_by:
        has_agg = any(
            _select_carries_aggregation(sc, intent.window_registry, intent.case_registry) for sc in select_cols
        )
        non_agg = [
            sc
            for sc in select_cols
            if not _select_carries_aggregation(sc, intent.window_registry, intent.case_registry)
        ]
        if not (has_agg and non_agg):
            return intent
        groupable: list[NormalizedExpr] = []
        for sc in non_agg:
            if expr_registry_ref(sc.expr) is not None:
                continue
            term = sc.expr.primary_term
            parts = term.split(".", 1) if "." in term else None
            if not parts:
                groupable.append(sc.expr)
                continue
            if _is_cte_output_groupable(term, cte_steps):
                groupable.append(sc.expr)
                continue
            tbl_meta = schema_graph.tables.get(parts[0])
            col_meta = tbl_meta.columns.get(parts[1]) if tbl_meta else None
            if not col_meta or col_meta.is_groupable:
                groupable.append(sc.expr)
        group_by = sorted(groupable, key=lambda g: g.signature_key)
        debug(
            f"[intent_resolve.enforce_grain_consistency] inferred group_by from groupable non-agg cols: {[g.primary_term for g in group_by]}"
        )
    existing_terms = {sc.expr.primary_term for sc in select_cols}
    gb_terms = {g.primary_term for g in group_by}
    has_agg_check = any(
        _select_carries_aggregation(sc, intent.window_registry, intent.case_registry) for sc in select_cols
    )
    if has_agg_check and gb_terms:
        for sc in select_cols:
            if _select_carries_aggregation(sc, intent.window_registry, intent.case_registry):
                continue
            if expr_registry_ref(sc.expr) is not None:
                continue
            term = sc.expr.primary_term
            if term in gb_terms:
                continue
            parts = term.split(".", 1) if "." in term else None
            if not parts:
                group_by.append(sc.expr)
                gb_terms.add(term)
                debug(f"[intent_resolve.enforce_grain_consistency] auto-added non-agg select col to group_by: {term}")
                continue
            if _is_cte_output_groupable(term, cte_steps):
                group_by.append(sc.expr)
                gb_terms.add(term)
                debug(f"[intent_resolve.enforce_grain_consistency] auto-added CTE output col to group_by: {term}")
                continue
            tbl_meta = schema_graph.tables.get(parts[0])
            col_meta = tbl_meta.columns.get(parts[1]) if tbl_meta else None
            if not col_meta or col_meta.is_groupable:
                group_by.append(sc.expr)
                gb_terms.add(term)
                debug(f"[intent_resolve.enforce_grain_consistency] auto-added non-agg select col to group_by: {term}")
    intent_tables = set(intent.tables or [])
    for gb_expr in list(group_by):
        gb_col = gb_expr.primary_term
        parts = gb_col.split(".", 1) if "." in gb_col else None
        if not parts:
            continue
        tbl_meta = schema_graph.tables.get(parts[0])
        col_meta = tbl_meta.columns.get(parts[1]) if tbl_meta else None
        if not col_meta:
            continue
        if col_meta.is_primary_key:
            desc = best_descriptive_column(parts[0], schema_graph, existing_terms | gb_terms)
            if desc:
                fq = f"{parts[0]}.{desc}"
                group_by.append(NormalizedExpr.from_column(fq))
                select_cols.append(SelectCol(expr=NormalizedExpr.from_column(fq)))
                existing_terms.add(fq)
                gb_terms.add(fq)
                debug(f"[intent_resolve.enforce_grain_consistency] auto-added descriptive column {fq}")
            continue
        if col_meta.is_foreign_key:
            for fk in tbl_meta.foreign_keys or []:
                if parts[1] not in fk.src_cols:
                    continue
                if fk.dst_table not in intent_tables:
                    continue
                desc = best_descriptive_column(fk.dst_table, schema_graph, existing_terms | gb_terms)
                if not desc:
                    continue
                fq = f"{fk.dst_table}.{desc}"
                group_by.append(NormalizedExpr.from_column(fq))
                select_cols.append(SelectCol(expr=NormalizedExpr.from_column(fq)))
                existing_terms.add(fq)
                gb_terms.add(fq)
                debug(
                    f"[intent_resolve.enforce_grain_consistency] auto-added FK descriptive column {fq} via {parts[0]}.{parts[1]}->{fk.dst_table}"
                )
    return replace(
        intent,
        group_by_cols=sorted(group_by, key=lambda g: g.signature_key),
        select_cols=select_cols,
        grain="grouped",
    )


def collect_column_refs_for_post_processing(intent: RuntimeIntent) -> list[str]:
    """
    Gather bare and qualified column tokens from a runtime intent for table resolution.

    Args:

        intent: Parsed intent before ``column_map`` is fully assigned.

    Returns:

        Column reference strings in deterministic walk order.
    """

    all_cols: list[str] = []
    for sc in intent.select_cols or []:
        all_cols.extend(cols_from_select_col(sc, intent.window_registry, intent.case_registry))
    all_cols.extend(cols_from_named_registries(intent.window_registry, intent.case_registry))
    for obc in intent.order_by_cols or []:
        all_cols.extend(extract_columns_from_expr(obc.expr))
    for g in intent.group_by_cols or []:
        all_cols.extend(extract_columns_from_expr(g))
    for fp in intent.filters_param or []:
        all_cols.extend(extract_columns_from_expr(fp.left_expr))
        if fp.right_expr:
            all_cols.extend(extract_columns_from_expr(fp.right_expr))
    for hp in intent.having_param or []:
        all_cols.extend(extract_columns_from_expr(hp.left_expr))
        if hp.right_expr:
            all_cols.extend(extract_columns_from_expr(hp.right_expr))
    return all_cols


def collect_column_refs_for_cte_step(cte: RuntimeCteStep) -> list[str]:
    """
    Gather column tokens from a CTE body for table resolution checks.

    Args:

        cte: CTE step whose expressions are scanned.

    Returns:

        Column reference strings in deterministic walk order.
    """

    all_cols: list[str] = []
    for sc in cte.select_cols or []:
        all_cols.extend(cols_from_select_col(sc, cte.window_registry, cte.case_registry))
    all_cols.extend(cols_from_named_registries(cte.window_registry, cte.case_registry))
    for obc in cte.order_by_cols or []:
        all_cols.extend(extract_columns_from_expr(obc.expr))
    for g in cte.group_by_cols or []:
        all_cols.extend(extract_columns_from_expr(g))
    for fp in cte.filters_param or []:
        all_cols.extend(extract_columns_from_expr(fp.left_expr))
        if fp.right_expr:
            all_cols.extend(extract_columns_from_expr(fp.right_expr))
    for hp in cte.having_param or []:
        all_cols.extend(extract_columns_from_expr(hp.left_expr))
        if hp.right_expr:
            all_cols.extend(extract_columns_from_expr(hp.right_expr))
    return all_cols


def prune_unused_cte_steps(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Drop ``cte_steps`` entries not reachable from the main query via ``tables`` or column refs.

    Preserves original step order for retained steps.

    Args:

        intent: Intent potentially carrying unused CTE definitions.

    Returns:

        Same intent when nothing is removed; otherwise a shallow-updated copy without orphan CTE rows.
    """

    steps = list(intent.cte_steps or [])
    if not steps:
        return intent
    cte_by_name: dict[str, RuntimeCteStep] = {}
    for s in steps:
        if s.cte_name:
            cte_by_name[s.cte_name.lower()] = s
    if not cte_by_name:
        return intent
    cte_names_lower = set(cte_by_name.keys())
    used: set[str] = set()
    for t in intent.tables or []:
        tl = t.lower()
        if tl in cte_names_lower:
            used.add(tl)
    for ref in collect_column_refs_for_post_processing(intent):
        if "." not in ref:
            continue
        pref = ref.split(".", 1)[0].strip().lower()
        if pref in cte_names_lower:
            used.add(pref)
    frontier = set(used)
    while frontier:
        nxt: set[str] = set()
        for name in frontier:
            step = cte_by_name.get(name)
            if step is None:
                continue
            for t in step.tables or []:
                tl = t.lower()
                if tl in cte_names_lower and tl not in used:
                    nxt.add(tl)
            for ref in collect_column_refs_for_cte_step(step):
                if "." not in ref:
                    continue
                pref = ref.split(".", 1)[0].strip().lower()
                if pref in cte_names_lower and pref not in used:
                    nxt.add(pref)
        used |= nxt
        frontier = nxt
    kept = [s for s in steps if s.cte_name and s.cte_name.lower() in used]
    if len(kept) == len(steps):
        return intent
    return replace(intent, cte_steps=kept)


def _accumulate_cte_output_aliases_from_refs(
    refs: Sequence[str],
    cte_names_lower: set[str],
    used_aliases: dict[str, set[str]],
) -> None:
    """
    Record ``cte_name.output_alias`` references for downstream CTE output pruning.

    Args:

        refs: Qualified column reference strings from intent walks.

        cte_names_lower: Lowercased CTE names eligible as qualifiers.

        used_aliases: Mutable map ``cte_name_lower -> set of referenced output alias lower strings``.
    """

    for ref in refs:
        if "." not in ref:
            continue
        qual, alias = ref.split(".", 1)
        ql = qual.strip().lower()
        if ql in cte_names_lower:
            used_aliases[ql].add(alias.strip().lower())


def prune_unused_cte_output_columns(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Drop CTE select_cols and matching output_columns whose alias no downstream consumer references and whose select expression is not a base PK or FK passthrough.

    Each CTE owns an ordered list of select_cols paired index-for-index with output_columns and an
    output_column_metadata map keyed by alias. This pass narrows that triple per CTE.

    A column is preserved when either (a) the main query or any strictly-downstream CTE references
    the alias as cte_name.alias in any column-ref-bearing field, or (b) the select_col is a
    passthrough of a base-table PK or FK column. Rule (b) exists because the JOIN engine derives
    bridge predicates from PK / FK lineage rather than from intent column refs, so a CTE typically
    projects a key column that nothing references explicitly but the engine still needs.

    Internal references inside the producing CTE do not preserve an output entry: those references
    read the underlying expression directly and never go through the alias.

    Args:

        intent: Intent after prune_unused_cte_steps has run.

        schema_graph: Active schema graph used to classify base-table PK and FK columns.

    Returns:

        Intent with each surviving CTE narrowed to its referenced or PK / FK passthrough columns.
        Returns the input intent unchanged when nothing is dropped.
    """

    steps = list(intent.cte_steps or [])
    if not steps:
        return intent
    cte_names_lower = {s.cte_name.lower() for s in steps if s.cte_name}
    if not cte_names_lower:
        return intent
    used_aliases: dict[str, set[str]] = {name: set() for name in cte_names_lower}
    _accumulate_cte_output_aliases_from_refs(
        collect_column_refs_for_post_processing(intent),
        cte_names_lower,
        used_aliases,
    )
    for cte in steps:
        _accumulate_cte_output_aliases_from_refs(
            collect_column_refs_for_cte_step(cte),
            cte_names_lower,
            used_aliases,
        )

    def _passthrough_base_pk_or_fk(sc: SelectCol) -> bool:
        if classify_cte_expr(sc.expr) != "passthrough":
            return False
        base = (sc.expr.primary_column or "").strip()
        if "." not in base or base == "*":
            return False
        tbl, col = base.split(".", 1)
        tbl_meta = schema_graph.tables.get(tbl)
        if tbl_meta is None:
            return False
        col_meta = tbl_meta.columns.get(col) or tbl_meta.columns.get(col.lower())
        if col_meta is None:
            return False
        return bool(col_meta.is_primary_key or col_meta.is_foreign_key)

    new_steps: list[RuntimeCteStep] = []
    changed = False
    for cte in steps:
        ckey = (cte.cte_name or "").lower()
        if not ckey or ckey not in used_aliases:
            new_steps.append(cte)
            continue
        output_cols = list(cte.output_columns or [])
        select_cols = list(cte.select_cols or [])
        n = min(len(output_cols), len(select_cols))
        kept_indices = [
            i
            for i in range(n)
            if output_cols[i].lower() in used_aliases[ckey] or _passthrough_base_pk_or_fk(select_cols[i])
        ]
        if len(kept_indices) == n and n == len(output_cols) == len(select_cols):
            new_steps.append(cte)
            continue
        if not kept_indices:
            new_steps.append(cte)
            continue
        new_select = [select_cols[i] for i in kept_indices]
        new_output = [output_cols[i] for i in kept_indices]
        kept_aliases_lower = {output_cols[i].lower() for i in kept_indices}
        ocm = cte.output_column_metadata or {}
        new_meta = {a: m for a, m in ocm.items() if a.lower() in kept_aliases_lower}
        new_steps.append(
            replace(
                cte,
                select_cols=new_select,
                output_columns=new_output,
                output_column_metadata=new_meta,
            )
        )
        changed = True
    if not changed:
        return intent
    return replace(intent, cte_steps=new_steps)


def enforce_cte_grain_consistency(cte: RuntimeCteStep) -> RuntimeCteStep:
    """
    Derive ``grain`` and sorted ``group_by_cols`` from CTE structure.

    Args:

        cte: CTE step to normalize.

    Returns:

        CTE with ``grouped`` or ``scalar`` grain as appropriate.
    """

    def _cte_select_counts_as_classical_agg(sc: SelectCol) -> bool:
        parts = effective_select_parts(sc, cte.window_registry, cte.case_registry)
        if parts.window_spec is not None:
            return False
        if expr_registry_ref(sc.expr) is not None:
            return False
        return parts.expr.has_aggregation

    has_agg = any(_cte_select_counts_as_classical_agg(sc) for sc in (cte.select_cols or []))
    if not cte.group_by_cols:
        if has_agg and cte.grain != "scalar":
            return replace(cte, grain="scalar")
        return cte
    sorted_gb = sorted(cte.group_by_cols, key=lambda g: g.signature_key)
    return replace(cte, grain="grouped", group_by_cols=sorted_gb)


def resolve_column_map(
    columns: list[str],
    schema_graph: SchemaGraph,
    tables: list[str],
) -> tuple[dict[str, str], list[IntentIssue]]:
    """
    Map bare column names to owning tables (qualified refs checked against *tables*).

    Args:

        columns: Bare or qualified references seen in expressions.

        schema_graph: Known columns per table.

        tables: Allow-list of table names.

    Returns:

        ``(column_map, issues)`` where *column_map* is ``{bare_or_qualified_key: table}`` and *issues*
        contains ``column_ambiguous`` errors when a bare name matches multiple tables.
    """
    column_map: dict[str, str] = {}
    issues: list[IntentIssue] = []
    table_col_index: dict[str, set[str]] = {}
    for tbl in tables:
        if tbl not in schema_graph.tables:
            continue
        table_col_index[tbl] = {c.lower() for c in schema_graph.tables[tbl].columns}
    for col in columns:
        col_stripped = col.strip()
        if "." in col_stripped:
            tbl_ref, col_ref = col_stripped.split(".", 1)
            col_ref_lower = col_ref.strip().lower()
            tbl_ref_lower = tbl_ref.strip().lower()
            for tbl in tables:
                if (
                    tbl.lower() == tbl_ref_lower or tbl.split(".")[-1].lower() == tbl_ref_lower
                ) and col_ref_lower in table_col_index.get(tbl, set()):
                    column_map[col_ref.strip()] = tbl
                    break
            continue
        col_lower = col_stripped.lower()
        candidates = [tbl for tbl in tables if col_lower in table_col_index.get(tbl, set())]
        if len(candidates) == 1:
            column_map[col_stripped] = candidates[0]
        elif len(candidates) > 1:
            cand_sorted = sorted(candidates)
            issues.append(
                make_intent_issue(
                    issue_id=f"column_ambiguous_{col_lower}",
                    category=FailureCategory.COLUMN_AMBIGUOUS,
                    severity="error",
                    message=(
                        f"Column '{col_stripped}' is ambiguous among tables: {', '.join(cand_sorted)}; "
                        "qualify it as table.column."
                    ),
                    context={"column": col_stripped, "candidates": cand_sorted},
                    responsible_stage="logical",
                ),
            )
            debug(f"[intent_resolve.resolve_column_map] ambiguous column '{col_stripped}': {cand_sorted}")
    return column_map, issues


def resolve_cte_column_maps(cte_steps: list[RuntimeCteStep]) -> list[RuntimeCteStep]:
    """
    Fill each CTE's ``column_map`` using prior CTE outputs in order.

    Args:

        cte_steps: Ordered CTE chain.

    Returns:

        New steps with ``column_map`` set.
    """
    cte_output_cols: dict[str, set[str]] = {}
    result = []
    for cte in cte_steps:
        cte_name = cte.cte_name
        out_cols = set(cte.output_columns or [])
        for sc in cte.select_cols or []:
            col = sc.expr.primary_column
            if col:
                out_cols.add(col.split(".")[-1])
        cte_output_cols[cte_name] = out_cols
        available_sources: dict[str, str] = {}
        for prev_cte_name, prev_cols in cte_output_cols.items():
            if prev_cte_name == cte_name:
                continue
            for c in prev_cols:
                available_sources[c.lower()] = prev_cte_name
        cols_to_resolve: list[str] = []
        for sc in cte.select_cols or []:
            cols_to_resolve.extend(cols_from_select_col(sc, cte.window_registry, cte.case_registry))
        cols_to_resolve.extend(cols_from_named_registries(cte.window_registry, cte.case_registry))
        for obc in cte.order_by_cols or []:
            cols_to_resolve.extend(extract_columns_from_expr(obc.expr))
        for fp in cte.filters_param or []:
            cols_to_resolve.extend(extract_columns_from_expr(fp.left_expr))
            if fp.right_expr:
                cols_to_resolve.extend(extract_columns_from_expr(fp.right_expr))
        for hp in cte.having_param or []:
            cols_to_resolve.extend(extract_columns_from_expr(hp.left_expr))
            if hp.right_expr:
                cols_to_resolve.extend(extract_columns_from_expr(hp.right_expr))
        column_map: dict[str, str] = {}
        for col in cols_to_resolve:
            col_stripped = col.strip()
            if "." in col_stripped:
                bare = col_stripped.split(".", 1)[1].strip()
                source = col_stripped.split(".", 1)[0].strip()
                column_map[bare] = source
            elif col_stripped.lower() in available_sources:
                column_map[col_stripped] = available_sources[col_stripped.lower()]
        updated_cte = replace(cte, column_map=column_map)
        result.append(updated_cte)
    return result


def _expr_lineage_signature(
    expr: NormalizedExpr,
    cte_meta_map: Mapping[str, Mapping[str, Any]],
) -> str:
    """
    Build a name-free signature for ``expr`` using physical lineage where available.

    Replaces qualified column references ``alias.col`` with a lineage triple
    ``(phys_table, phys_column, inherits_pk)`` when the alias is a CTE whose
    output metadata is in ``cte_meta_map``; otherwise leaves the qualified
    reference verbatim. CTE-name placeholders are stripped so the resulting
    signature is invariant under CTE renaming.

    Args:

        expr: Expression to fingerprint.

        cte_meta_map: ``cte_name -> {output_alias: CteOutputColumnMeta}``.

    Returns:

        Pipe-joined signature string.
    """

    def _swap(token: str) -> str:
        if "." not in token:
            return token
        head, tail = token.split(".", 1)
        meta_for_cte = cte_meta_map.get(head)
        if meta_for_cte is None:
            return token
        meta = meta_for_cte.get(tail)
        if meta is None:
            return f"CTE_OUT::{tail}"
        return (
            "CTE_LINEAGE("
            f"{meta.lineage_phys_table or ''}.{meta.lineage_phys_column or ''}|"
            f"pk={int(bool(meta.lineage_inherits_pk))}|"
            f"fk={meta.lineage_fk_to_table or ''}.{meta.lineage_fk_to_column or ''}"
            ")"
        )

    return replace_refs_in_expr(expr, _swap).signature_key


def _output_column_lineage_tuples(
    cte: RuntimeCteStep,
) -> tuple[tuple[str, ...], ...]:
    """Return sorted lineage tuples for every output column of *cte*."""

    rows: list[tuple[str, ...]] = []
    for alias in cte.output_columns or []:
        meta = (cte.output_column_metadata or {}).get(alias)
        if meta is None:
            rows.append((alias, "", "", "0", "", "", ""))
            continue
        rows.append(
            (
                alias,
                meta.lineage_phys_table or "",
                meta.lineage_phys_column or "",
                "1" if meta.lineage_inherits_pk else "0",
                meta.lineage_fk_to_table or "",
                meta.lineage_fk_to_column or "",
                meta.role or "",
            )
        )
    return tuple(sorted(rows))


def _compute_cte_structural_hash(
    cte: RuntimeCteStep,
    sibling_hashes: Mapping[str, str],
    sibling_names: set[str],
) -> str:
    """
    Compute a CTE-name- and description- and grain-free structural hash.

    Inter-CTE references in ``cte.tables`` are substituted with the recursively
    computed structural hash of the referenced CTE; base-table names are kept
    verbatim. Output column lineage tuples use the already-standardized output
    column names, so the hash is stable under CTE renaming and free of any
    LLM-chosen identifiers.

    Args:

        cte: CTE step (with output columns already normalized).

        sibling_hashes: ``cte_name -> structural_hash`` for prior topo levels.

        sibling_names: All CTE names in the intent (used to distinguish
        inter-CTE refs from base tables in ``cte.tables``).

    Returns:

        SHA-256 hex digest truncated to 16 characters.
    """

    cte_meta_map: dict[str, dict[str, Any]] = {}
    for name in sibling_names:
        cte_meta_map[name] = {}

    refs_payload: list[tuple[str, str]] = []
    base_payload: list[str] = []
    for t in cte.tables or []:
        if t in sibling_names:
            refs_payload.append(("CTE_REF", sibling_hashes.get(t, "PENDING")))
        else:
            base_payload.append(t)

    select_payload = [
        (sc.is_aggregated, _expr_lineage_signature(sc.expr, cte_meta_map)) for sc in (cte.select_cols or [])
    ]
    group_payload = [_expr_lineage_signature(g, cte_meta_map) for g in (cte.group_by_cols or [])]
    order_payload = [
        (
            (obc.direction or "asc").lower(),
            _expr_lineage_signature(obc.expr, cte_meta_map),
        )
        for obc in (cte.order_by_cols or [])
    ]
    filter_payload = [
        (
            _expr_lineage_signature(fp.left_expr, cte_meta_map),
            (fp.op or "").lower(),
            (fp.value_type or "").lower(),
            (_expr_lineage_signature(fp.right_expr, cte_meta_map) if fp.right_expr else ""),
        )
        for fp in (cte.filters_param or [])
    ]
    having_payload = [
        (
            _expr_lineage_signature(hp.left_expr, cte_meta_map),
            (hp.op or "").lower(),
            (hp.value_type or "").lower(),
            (_expr_lineage_signature(hp.right_expr, cte_meta_map) if hp.right_expr else ""),
        )
        for hp in (cte.having_param or [])
    ]

    payload = {
        "outputs": _output_column_lineage_tuples(cte),
        "select": sorted(select_payload),
        "group_by": sorted(group_payload),
        "order_by": sorted(order_payload),
        "filters": sorted(filter_payload),
        "having": sorted(having_payload),
        "base_tables": sorted(base_payload),
        "cte_refs": sorted(refs_payload),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def reorder_cte_steps_by_dag(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Topologically reorder ``cte_steps`` so dependencies precede consumers.

    A CTE ``B`` depends on CTE ``A`` iff ``A.cte_name`` appears in ``B.tables``.
    Within each topological level CTEs are sorted by their structural hash so
    the order is deterministic and free of LLM-chosen identifiers. Cycles are
    rejected. CTE-to-CTE references in ``tables`` are preserved verbatim and
    rewritten later by :func:`normalize_cte_names`.

    Args:

        intent: Intent whose ``cte_steps`` should be reordered.

    Returns:

        Intent with ``cte_steps`` arranged in dependency order.
    """

    cte_steps = list(intent.cte_steps or [])
    if len(cte_steps) < 2:
        return intent

    name_to_step: dict[str, RuntimeCteStep] = {c.cte_name: c for c in cte_steps}
    sibling_names = set(name_to_step.keys())

    deps: dict[str, set[str]] = {n: set() for n in sibling_names}
    consumers: dict[str, set[str]] = {n: set() for n in sibling_names}
    for c in cte_steps:
        for t in c.tables or []:
            if t in sibling_names and t != c.cte_name:
                deps[c.cte_name].add(t)
                consumers[t].add(c.cte_name)

    hashes: dict[str, str] = {}
    ready = sorted([n for n, ds in deps.items() if not ds])
    ordered: list[str] = []
    pending = {n: set(ds) for n, ds in deps.items()}

    while ready:
        level_nodes = list(ready)
        ready = []
        level_hashed: list[tuple[str, str]] = []
        for n in level_nodes:
            h = _compute_cte_structural_hash(name_to_step[n], hashes, sibling_names)
            hashes[n] = h
            level_hashed.append((h, n))
        level_hashed.sort()
        for _, n in level_hashed:
            ordered.append(n)
            for cons in sorted(consumers[n]):
                pending[cons].discard(n)
                if not pending[cons] and cons not in ordered and cons not in ready:
                    ready.append(cons)
        ready.sort()

    if len(ordered) != len(cte_steps):
        remaining = sorted(n for n in sibling_names if n not in ordered)
        debug(f"[intent_resolve.reorder_cte_steps_by_dag] cycle detected; unprocessed CTEs: {remaining}")
        return intent

    if ordered == [c.cte_name for c in cte_steps]:
        return intent

    debug(f"[intent_resolve.reorder_cte_steps_by_dag] reordered {[c.cte_name for c in cte_steps]} -> {ordered}")
    new_steps = [name_to_step[n] for n in ordered]
    return replace(intent, cte_steps=new_steps)


def normalize_cte_names(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Rename CTEs to ``cte1``, ``cte2``, ... and rewrite all references.

    Args:

        intent: Intent whose ``cte_steps`` should be normalized.

    Returns:

        Intent with consistent CTE names across tables and expressions.
    """
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return intent
    old_to_new: dict[str, str] = {}
    for i, cte in enumerate(cte_steps, start=1):
        new_name = f"cte{i}"
        old_to_new[cte.cte_name] = new_name

    old_to_new_ci: dict[str, str] = {k.lower(): v for k, v in old_to_new.items()}

    def replace_cte_refs(s: str) -> str:

        head, sep, tail = s.partition(".")
        new_head = old_to_new_ci.get(head.lower(), head)
        if not sep:
            return new_head
        return f"{new_head}.{tail}"

    def _update_expr(expr: NormalizedExpr) -> NormalizedExpr:
        return replace_refs_in_expr(expr, replace_cte_refs)

    new_cte_steps = []
    for cte in cte_steps:
        new_name = old_to_new[cte.cte_name]
        new_tables = [replace_cte_refs(t) for t in (cte.tables or [])]
        new_select_cols = [replace(sc, expr=_update_expr(sc.expr)) for sc in (cte.select_cols or [])]
        new_group_by = [_update_expr(g) for g in (cte.group_by_cols or [])]
        new_order_by = [replace(obc, expr=_update_expr(obc.expr)) for obc in (cte.order_by_cols or [])]
        new_filters = []
        for fp in cte.filters_param or []:
            new_fp = replace(
                fp,
                left_expr=_update_expr(fp.left_expr),
                right_expr=_update_expr(fp.right_expr) if fp.right_expr else None,
            )
            new_filters.append(new_fp)
        new_having = []
        for hp in cte.having_param or []:
            new_hp = replace(
                hp,
                left_expr=_update_expr(hp.left_expr),
                right_expr=_update_expr(hp.right_expr) if hp.right_expr else None,
            )
            new_having.append(new_hp)
        new_column_map = {}
        for k, v in (cte.column_map or {}).items():
            new_column_map[replace_cte_refs(k)] = replace_cte_refs(v)
        raw_outputs = list(cte.output_columns or [])
        new_output_columns: list[str] = []
        is_scalar_cte = (cte.grain or "") == "scalar" or getattr(cte, "emission", "") == "scalar_subquery"
        for oc in raw_outputs:
            bare_oc = oc.split(".")[-1].strip().lower()
            if is_scalar_cte and len(raw_outputs) == 1 and bare_oc == (cte.cte_name or "").strip().lower():
                inferred = infer_cte_output_columns(cte, include_agg_prefix=False)
                pick = inferred[0] if inferred else oc
                new_output_columns.append(pick.split(".")[-1] if "." in pick else pick)
                continue
            new_output_columns.append(replace_cte_refs(oc))
        new_ocm = {replace_cte_refs(k): v for k, v in (cte.output_column_metadata or {}).items()}
        new_cte = replace(
            cte,
            cte_name=new_name,
            tables=new_tables,
            select_cols=new_select_cols,
            group_by_cols=new_group_by,
            order_by_cols=new_order_by,
            filters_param=new_filters,
            having_param=new_having,
            column_map=new_column_map,
            output_columns=new_output_columns,
            output_column_metadata=new_ocm,
        )
        new_cte_steps.append(new_cte)

    new_main_tables = [replace_cte_refs(t) for t in (intent.tables or [])]
    new_main_select = [replace(sc, expr=_update_expr(sc.expr)) for sc in (intent.select_cols or [])]
    new_main_group_by = [_update_expr(g) for g in (intent.group_by_cols or [])]
    new_main_order_by = [replace(obc, expr=_update_expr(obc.expr)) for obc in (intent.order_by_cols or [])]
    new_main_filters = []
    for fp in intent.filters_param or []:
        new_fp = replace(
            fp,
            left_expr=_update_expr(fp.left_expr),
            right_expr=_update_expr(fp.right_expr) if fp.right_expr else None,
        )
        new_main_filters.append(new_fp)
    new_main_having = []
    for hp in intent.having_param or []:
        new_hp = replace(
            hp,
            left_expr=_update_expr(hp.left_expr),
            right_expr=_update_expr(hp.right_expr) if hp.right_expr else None,
        )
        new_main_having.append(new_hp)
    new_main_column_map = {}
    for k, v in (intent.column_map or {}).items():
        new_main_column_map[replace_cte_refs(k)] = replace_cte_refs(v)
    return replace(
        intent,
        tables=new_main_tables,
        select_cols=new_main_select,
        group_by_cols=new_main_group_by,
        order_by_cols=new_main_order_by,
        filters_param=new_main_filters,
        having_param=new_main_having,
        column_map=new_main_column_map,
        cte_steps=new_cte_steps,
    )


def rewrite_main_query_refs_to_final_cte_columns(
    intent: RuntimeIntent,
) -> RuntimeIntent:
    """
    Align main-scope ``table.col`` references with each CTE's final ``output_columns`` aliases.

    Bare tokens that match exactly one CTE output column name are prefixed with that CTE name. Covers
    select/group/order, filters, having, window registries, and case registries on the main query.

    Args:

        intent: Runtime intent whose CTE list defines output shapes.

    Returns:

        Intent with ``replace_refs_in_expr`` applied to every main-scope expression carrier.
    """

    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return intent
    by_name: dict[str, RuntimeCteStep] = {c.cte_name.lower(): c for c in cte_steps}

    def _cte_bare_outputs(cte: RuntimeCteStep) -> set[str]:
        explicit = list(cte.output_columns or [])
        if not explicit:
            explicit = infer_cte_output_columns(cte)
        return {o.split(".")[-1].strip().lower() for o in explicit if o}

    def _remap_ref(ref: str) -> str:
        ref_s = (ref or "").strip()
        if "." not in ref_s:
            bare = ref_s.lower()
            if not bare:
                return ref
            owners = [c.cte_name for c in cte_steps if bare in _cte_bare_outputs(c)]
            owners_u = list(dict.fromkeys(owners))
            if len(owners_u) == 1:
                return f"{owners_u[0]}.{bare}"
            return ref
        pref, col = ref_s.rsplit(".", 1)
        cte = by_name.get(pref.lower())
        if cte is None:
            return ref
        explicit = list(cte.output_columns or [])
        bare_outs = {o.split(".")[-1].lower() for o in explicit}
        inferred = infer_cte_output_columns(cte)
        col_l = col.lower()

        if col_l in bare_outs:
            return ref_s

        infer_to_explicit: dict[str, str] = {}
        if explicit and inferred and len(explicit) == len(inferred):
            for inf, exp in zip(inferred, explicit, strict=False):
                inf_b = inf.split(".")[-1].lower()
                exp_b = exp.split(".")[-1]
                if inf_b != exp_b.lower():
                    infer_to_explicit[inf_b] = exp_b
        if col_l in infer_to_explicit:
            return f"{pref}.{infer_to_explicit[col_l]}"

        if len(inferred) == 1 and col_l == (cte.cte_name or "").strip().lower():
            tail = inferred[0].split(".")[-1]
            if explicit:
                exp0 = explicit[0].split(".")[-1]
                if exp0.lower() != tail.lower():
                    return f"{pref}.{exp0}"
            return f"{pref}.{tail}"
        return ref_s

    def _up(expr: NormalizedExpr) -> NormalizedExpr:
        return replace_refs_in_expr(expr, _remap_ref)

    def _map_wr(wr: list[WindowRegistryStep] | None) -> list[WindowRegistryStep]:
        out: list[WindowRegistryStep] = []
        for step in wr or []:
            ws = step.window_spec
            new_part = [_up(p) for p in (ws.partition_by or [])]
            new_orders = [replace(o, expr=_up(o.expr)) for o in (ws.order_by or [])]
            new_arg = _up(ws.argument) if ws.argument is not None else None
            new_ws = replace(ws, partition_by=new_part, order_by=new_orders, argument=new_arg)
            out.append(replace(step, window_spec=new_ws))
        return out

    def _map_cr(regs: list[CaseRegistryStep] | None) -> list[CaseRegistryStep]:
        out: list[CaseRegistryStep] = []
        for step in regs or []:
            cw = step.case_when
            new_branches: list[CaseWhenBranch] = []
            for br in cw.branches or []:
                cond = br.condition
                new_cond = replace(
                    cond,
                    left_expr=_up(cond.left_expr),
                    right_expr=_up(cond.right_expr) if cond.right_expr else None,
                )
                new_branches.append(CaseWhenBranch(condition=new_cond, result=_up(br.result)))
            new_else = _up(cw.else_result) if cw.else_result is not None else None
            new_cw = replace(cw, branches=new_branches, else_result=new_else)
            out.append(replace(step, case_when=new_cw))
        return out

    def _map_sc(sc: SelectCol) -> SelectCol:
        return replace(sc, expr=_up(sc.expr))

    new_select = [_map_sc(sc) for sc in (intent.select_cols or [])]
    new_gb = [_up(g) for g in (intent.group_by_cols or [])]
    new_ob = [replace(obc, expr=_up(obc.expr)) for obc in (intent.order_by_cols or [])]
    new_fp: list[FilterParam] = []
    for fp in intent.filters_param or []:
        new_fp.append(
            replace(
                fp,
                left_expr=_up(fp.left_expr),
                right_expr=_up(fp.right_expr) if fp.right_expr else None,
            )
        )
    new_hp: list[HavingParam] = []
    for hp in intent.having_param or []:
        new_hp.append(
            replace(
                hp,
                left_expr=_up(hp.left_expr),
                right_expr=_up(hp.right_expr) if hp.right_expr else None,
            )
        )
    return replace(
        intent,
        select_cols=new_select,
        group_by_cols=new_gb,
        order_by_cols=new_ob,
        filters_param=new_fp,
        having_param=new_hp,
        window_registry=_map_wr(intent.window_registry),
        case_registry=_map_cr(intent.case_registry),
    )


def _cte_output_alias_map(intent: RuntimeIntent) -> dict[str, str]:
    """
    Map ``cte.expr_form`` -> ``cte.output_alias`` including a normalized key variant.

    Args:

        intent: Intent with ``cte_steps`` and aligned ``output_columns``.

    Returns:

        Replacement map for downstream ref rewriting.
    """
    alias_map: dict[str, str] = {}
    for cte in intent.cte_steps or []:
        output_cols = cte.output_columns or []
        for i, sc in enumerate(cte.select_cols or []):
            if i >= len(output_cols):
                continue
            rendered = render_expr_sql(sc.expr)
            from_ref = f"{cte.cte_name}.{rendered}"
            to_ref = f"{cte.cte_name}.{output_cols[i]}"
            if from_ref != to_ref:
                alias_map[from_ref] = to_ref
    return alias_map


def rewrite_cte_output_refs_to_aliases(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Rewrite main/CTE refs from rendered expr form to declared ``output_columns`` aliases.

    Args:

        intent: Intent after CTE output lists are stable.

    Returns:

        Intent with consistent ``cteN.alias`` references.
    """
    alias_map = _cte_output_alias_map(intent)
    if not alias_map:
        return intent

    def replacer(s: str) -> str:
        return alias_map.get(s, s)

    def _update_expr(expr: NormalizedExpr) -> NormalizedExpr:
        return replace_refs_in_expr(expr, replacer)

    new_cte_steps = []
    for cte in intent.cte_steps or []:
        new_select = [replace(sc, expr=_update_expr(sc.expr)) for sc in (cte.select_cols or [])]
        new_group_by = [_update_expr(g) for g in (cte.group_by_cols or [])]
        new_order_by = [replace(obc, expr=_update_expr(obc.expr)) for obc in (cte.order_by_cols or [])]
        new_filters = [
            replace(
                fp,
                left_expr=_update_expr(fp.left_expr),
                right_expr=_update_expr(fp.right_expr) if fp.right_expr else None,
            )
            for fp in (cte.filters_param or [])
        ]
        new_having = [
            replace(
                hp,
                left_expr=_update_expr(hp.left_expr),
                right_expr=_update_expr(hp.right_expr) if hp.right_expr else None,
            )
            for hp in (cte.having_param or [])
        ]
        new_cte_steps.append(
            replace(
                cte,
                select_cols=new_select,
                group_by_cols=new_group_by,
                order_by_cols=new_order_by,
                filters_param=new_filters,
                having_param=new_having,
            )
        )

    new_main_select = [replace(sc, expr=_update_expr(sc.expr)) for sc in (intent.select_cols or [])]
    new_main_group_by = [_update_expr(g) for g in (intent.group_by_cols or [])]
    new_main_order_by = [replace(obc, expr=_update_expr(obc.expr)) for obc in (intent.order_by_cols or [])]
    new_main_filters = [
        replace(
            fp,
            left_expr=_update_expr(fp.left_expr),
            right_expr=_update_expr(fp.right_expr) if fp.right_expr else None,
        )
        for fp in (intent.filters_param or [])
    ]
    new_main_having = [
        replace(
            hp,
            left_expr=_update_expr(hp.left_expr),
            right_expr=_update_expr(hp.right_expr) if hp.right_expr else None,
        )
        for hp in (intent.having_param or [])
    ]
    return replace(
        intent,
        select_cols=new_main_select,
        group_by_cols=new_main_group_by,
        order_by_cols=new_main_order_by,
        filters_param=new_main_filters,
        having_param=new_main_having,
        cte_steps=new_cte_steps,
    )


def check_qualified_refs_exist(intent: RuntimeIntent, schema_graph: SchemaGraph) -> tuple[RuntimeIntent, list[str]]:
    """
    Check that qualified ``table.col`` references in an intent exist in the schema graph.

    Returns a list of missing-reference messages; the intent is returned unchanged.

    Args:

        intent: RuntimeIntent to validate.

        schema_graph: SchemaGraph providing the authoritative table/column set.

    Returns:

        Tuple of ``(intent, errors)`` where *errors* is a list of human-readable violation strings.
    """
    errors: list[str] = []
    valid_tables = set(schema_graph.tables.keys())
    cte_names = {cte.cte_name for cte in (intent.cte_steps or [])}
    for tbl in intent.tables or []:
        if tbl not in valid_tables and tbl not in cte_names:
            errors.append(f"Unknown table: {tbl}")

    def _check_expr_cols(exprs: list, label: str) -> None:
        for item in exprs:
            expr = item.expr if hasattr(item, "expr") else item
            for col in extract_columns_from_expr(expr):
                if "." in col:
                    tbl_ref, col_ref = col.split(".", 1)
                    if tbl_ref in valid_tables:
                        tbl_meta = schema_graph.tables[tbl_ref]
                        if col_ref not in tbl_meta.columns:
                            errors.append(f"Unknown {label} column: {col}")

    def _check_filter_cols(params: list, label: str) -> None:
        for fp in params:
            for col in extract_columns_from_expr(fp.left_expr):
                if "." in col:
                    tbl_ref, col_ref = col.split(".", 1)
                    if tbl_ref in valid_tables:
                        tbl_meta = schema_graph.tables[tbl_ref]
                        if col_ref not in tbl_meta.columns:
                            errors.append(f"Unknown {label} column: {col}")
            if fp.right_expr:
                for col in extract_columns_from_expr(fp.right_expr):
                    if "." in col:
                        tbl_ref, col_ref = col.split(".", 1)
                        if tbl_ref in valid_tables:
                            tbl_meta = schema_graph.tables[tbl_ref]
                            if col_ref not in tbl_meta.columns:
                                errors.append(f"Unknown {label} column: {col}")

    def _check_bare_cols(cols: list, label: str) -> None:
        for g in cols:
            col = g.primary_term if hasattr(g, "primary_term") else str(g)
            if "." in col:
                tbl_ref, col_ref = col.split(".", 1)
                if tbl_ref in valid_tables:
                    tbl_meta = schema_graph.tables[tbl_ref]
                    if col_ref not in tbl_meta.columns:
                        errors.append(f"Unknown {label} column: {col}")

    _check_expr_cols(intent.select_cols or [], "select")
    _check_expr_cols(intent.order_by_cols or [], "order_by")
    _check_filter_cols(intent.filters_param or [], "filter")
    _check_filter_cols(intent.having_param or [], "having")
    _check_bare_cols(intent.group_by_cols or [], "group_by")
    for cte in intent.cte_steps or []:
        ctx = f"CTE '{cte.cte_name}'"
        for tbl in cte.tables or []:
            if tbl not in valid_tables and tbl not in cte_names:
                errors.append(f"{ctx} unknown table: {tbl}")
        _check_expr_cols(cte.select_cols or [], f"{ctx} select")
        _check_expr_cols(cte.order_by_cols or [], f"{ctx} order_by")
        _check_filter_cols(cte.filters_param or [], f"{ctx} filter")
        _check_filter_cols(cte.having_param or [], f"{ctx} having")
        _check_bare_cols(cte.group_by_cols or [], f"{ctx} group_by")
    if errors:
        debug(f"[intent_resolve.check_qualified_refs_exist] validation errors: {errors}")
    pipeline_trace_lazy(
        "intent_resolve.check_qualified_refs_exist.result",
        lambda: stable_json({"errors": errors, "intent": intent.to_dict()}),
    )
    return intent, errors


def _simplify_expr(expr: NormalizedExpr, param_values: Mapping[str, Any] | None = None) -> NormalizedExpr:
    """
    Fold constants, combine like MulGroups, and normalize coefficients.

    When ``param_values`` is provided, parameterized add/sub values whose bound value resolves
    to ``0`` or ``0.0`` are folded out, and group coefficients whose ``coeff_param_key``
    resolves to ``1`` or ``1.0`` have the key cleared so the renderer omits the multiplier.

    Args:

        expr: Expression tree to simplify.

        param_values: Optional bound parameter map for resolving ``coeff_param_key`` and
        parameterized identity values.

    Returns:

        Equivalent simplified ``NormalizedExpr``.
    """
    pv = param_values or {}

    def _is_zero_value(ev: ExprValue) -> bool:
        if ev.param_key and ev.param_key in pv:
            return pv[ev.param_key] in (0, 0.0)
        return ev.value in (0, 0.0)

    add_groups: list[MulGroup] = []
    sub_groups: list[MulGroup] = []
    add_vals: list[ExprValue] = []
    sub_vals: list[ExprValue] = []
    parameterized_add: list[ExprValue] = []
    parameterized_sub: list[ExprValue] = []
    for v in expr.add_values:
        if _is_zero_value(v):
            continue
        (parameterized_add if v.param_key else add_vals).append(v)
    for v in expr.sub_values:
        if _is_zero_value(v):
            continue
        (parameterized_sub if v.param_key else sub_vals).append(v)
    net_const = sum(v.value for v in add_vals) - sum(v.value for v in sub_vals)
    for g in expr.add_groups:
        if g.coeff_param_key and g.coeff_param_key in pv and pv[g.coeff_param_key] in (1, 1.0):
            g = replace(g, coeff_param_key="")
        if not g.multiply and not g.divide and not g.agg_func and not g.scalar_func and not g.inner_scalar_func:
            net_const += g.coefficient
        else:
            add_groups.append(g)
    for g in expr.sub_groups:
        if g.coeff_param_key and g.coeff_param_key in pv and pv[g.coeff_param_key] in (1, 1.0):
            g = replace(g, coeff_param_key="")
        if not g.multiply and not g.divide and not g.agg_func and not g.scalar_func and not g.inner_scalar_func:
            net_const -= g.coefficient
        else:
            sub_groups.append(g)
    bucket: dict[str, float] = {}
    group_map: dict[str, MulGroup] = {}
    for g in add_groups:
        key = g.structural_key
        bucket[key] = bucket.get(key, 0.0) + g.coefficient
        if key not in group_map:
            group_map[key] = g
    for g in sub_groups:
        key = g.structural_key
        bucket[key] = bucket.get(key, 0.0) - g.coefficient
        if key not in group_map:
            group_map[key] = g
    final_add: list[MulGroup] = []
    final_sub: list[MulGroup] = []
    for key, coeff in bucket.items():
        if coeff == 0.0:
            continue
        ref = group_map[key]
        if coeff > 0:
            final_add.append(
                MulGroup(
                    coefficient=coeff,
                    multiply=list(ref.multiply),
                    divide=list(ref.divide),
                    agg_func=ref.agg_func,
                    scalar_func=ref.scalar_func,
                    inner_scalar_func=ref.inner_scalar_func,
                    scalar_func_args=list(ref.scalar_func_args),
                    inner_scalar_func_args=list(ref.inner_scalar_func_args),
                    distinct=ref.distinct,
                )
            )
        else:
            final_sub.append(
                MulGroup(
                    coefficient=abs(coeff),
                    multiply=list(ref.multiply),
                    divide=list(ref.divide),
                    agg_func=ref.agg_func,
                    scalar_func=ref.scalar_func,
                    inner_scalar_func=ref.inner_scalar_func,
                    scalar_func_args=list(ref.scalar_func_args),
                    inner_scalar_func_args=list(ref.inner_scalar_func_args),
                    distinct=ref.distinct,
                )
            )
    final_add_vals: list[ExprValue] = list(parameterized_add)
    final_sub_vals: list[ExprValue] = list(parameterized_sub)
    if net_const > 0:
        final_add_vals.append(ExprValue(value=net_const))
    elif net_const < 0:
        final_sub_vals.append(ExprValue(value=abs(net_const)))
    return replace(
        expr,
        add_groups=final_add,
        sub_groups=final_sub,
        add_values=final_add_vals,
        sub_values=final_sub_vals,
    )


def _simplify_filter(fp: FilterParam, param_values: Mapping[str, Any] | None = None) -> FilterParam:
    """
    Apply simplify_expr to both sides of a FilterParam.

    Args:

        fp: FilterParam to simplify.

        param_values: Optional bound parameter map forwarded to ``_simplify_expr``.

    Returns:

        New FilterParam with simplified left_expr and right_expr.
    """
    new_left = _simplify_expr(fp.left_expr, param_values)
    new_right = _simplify_expr(fp.right_expr, param_values) if fp.right_expr else None
    return replace(fp, left_expr=new_left, right_expr=new_right)


def _simplify_having(hp: HavingParam, param_values: Mapping[str, Any] | None = None) -> HavingParam:
    """
    Apply simplify_expr to both sides of a HavingParam.

    Args:

        hp: HavingParam to simplify.

        param_values: Optional bound parameter map forwarded to ``_simplify_expr``.

    Returns:

        New HavingParam with simplified left_expr and right_expr.
    """
    new_left = _simplify_expr(hp.left_expr, param_values)
    new_right = _simplify_expr(hp.right_expr, param_values) if hp.right_expr else None
    return replace(hp, left_expr=new_left, right_expr=new_right)


def _raw_sql_opens_with_distinct(raw: str) -> bool:
    return (raw or "").strip().upper().startswith("DISTINCT ")


def _qualified_column_from_distinct_raw_sql(raw: str, schema_graph: SchemaGraph) -> str | None:
    """
    Parse a raw SQL fragment that may start with ``DISTINCT`` and return ``table.column`` when
    the remainder is a single qualified column present in *schema_graph*.
    """

    rs = (raw or "").strip()
    if not rs:
        return None
    inner = rs[9:].strip() if _raw_sql_opens_with_distinct(rs) else rs
    if not inner:
        return None
    try:
        tree = sqlglot.parse_one(inner, dialect=None)
    except Exception:
        return None
    node: exp.Expression | None = tree
    if isinstance(node, exp.Distinct):
        node = node.this
    if not isinstance(node, exp.Column):
        return None
    tbl = (node.table or "").strip()
    name = (node.name or "").strip()
    if not tbl or not name:
        return None
    if schema_graph.get_column(tbl, name) is None:
        return None
    return f"{tbl}.{name}"


def _select_col_is_raw_sql_leaf(sc: SelectCol) -> bool:
    e = sc.expr
    if not (e.raw_sql or "").strip():
        return False
    if e.column_ref or e.star or e.add_groups or e.sub_groups:
        return False
    if e.add_values or e.sub_values or e.agg_func or e.scalar_func or e.inner_scalar_func:
        return False
    if e.scalar_func_args or e.inner_scalar_func_args:
        return False
    if e.cast_type or e.interval or e.keyword:
        return False
    return True


def lift_distinct_select_from_raw_sql(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Promote ``DISTINCT table.column`` fragments stored as ``raw_sql`` into structured column refs.

    The first lifted column index per scope is written to ``distinct_select_index`` when that scope's index is still ``-1`` and the fragment opened with the ``DISTINCT`` keyword.
    """

    def process_scope(
        cols: list[SelectCol],
        distinct_idx: int,
    ) -> tuple[list[SelectCol], int]:
        out_cols: list[SelectCol] = []
        d_idx = distinct_idx
        for i, sc in enumerate(cols or []):
            if not _select_col_is_raw_sql_leaf(sc):
                out_cols.append(sc)
                continue
            raw = sc.expr.raw_sql or ""
            ref = _qualified_column_from_distinct_raw_sql(raw, schema_graph)
            if not ref:
                out_cols.append(sc)
                continue
            out_cols.append(replace(sc, expr=NormalizedExpr.from_column(ref)))
            if d_idx < 0 and _raw_sql_opens_with_distinct(raw):
                d_idx = i
        return out_cols, d_idx

    main_cols, main_d = process_scope(list(intent.select_cols or []), intent.distinct_select_index)
    new_ctes: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        c_cols, c_d = process_scope(list(cte.select_cols or []), cte.distinct_select_index)
        new_ctes.append(replace(cte, select_cols=c_cols, distinct_select_index=c_d))
    return replace(intent, select_cols=main_cols, distinct_select_index=main_d, cte_steps=new_ctes)


def _expr_is_simplified_to_empty(expr: NormalizedExpr) -> bool:
    if expr.column_ref or expr.star or (expr.raw_sql or "").strip():
        return False
    if expr.keyword or expr.interval or expr.cast_type:
        return False
    if expr.add_groups or expr.sub_groups or expr.add_values or expr.sub_values:
        return False
    if expr.agg_func or expr.scalar_func or expr.inner_scalar_func:
        return False
    if expr.scalar_func_args or expr.inner_scalar_func_args:
        return False
    return True


def simplify_exprs(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Apply algebraic simplification to every NormalizedExpr across all intent clauses.

    Uses ``intent.param_values`` to fold parameterized identity values (``0``/``0.0``/``1``/``1.0``)
    when the structural assignment pass has already bound them.

    Args:

        intent: RuntimeIntent whose expressions should be simplified.

    Returns:

        New RuntimeIntent with all expressions in simplified form.
    """
    debug("[intent_resolve.simplify_exprs] simplifying all expressions")
    pv = intent.param_values or {}
    new_select: list[SelectCol] = []
    for sc in intent.select_cols or []:
        simplified = _simplify_expr(sc.expr, pv)
        if _expr_is_simplified_to_empty(simplified) and not _expr_is_simplified_to_empty(sc.expr):
            debug("[intent_resolve.simplify_exprs] preserving select col that would simplify to empty")
            new_select.append(sc)
        else:
            new_select.append(replace(sc, expr=simplified))
    new_order = [replace(obc, expr=_simplify_expr(obc.expr, pv)) for obc in (intent.order_by_cols or [])]
    new_filters = [_simplify_filter(fp, pv) for fp in (intent.filters_param or [])]
    new_having = [_simplify_having(hp, pv) for hp in (intent.having_param or [])]
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_pv = cte.param_values or pv
        cte_select: list[SelectCol] = []
        for sc in cte.select_cols or []:
            simplified_c = _simplify_expr(sc.expr, cte_pv)
            if _expr_is_simplified_to_empty(simplified_c) and not _expr_is_simplified_to_empty(sc.expr):
                debug(
                    f"[intent_resolve.simplify_exprs] preserving CTE '{cte.cte_name}' select col "
                    "that would simplify to empty"
                )
                cte_select.append(sc)
            else:
                cte_select.append(replace(sc, expr=simplified_c))
        cte_order = [replace(obc, expr=_simplify_expr(obc.expr, cte_pv)) for obc in (cte.order_by_cols or [])]
        cte_filters = [_simplify_filter(fp, cte_pv) for fp in (cte.filters_param or [])]
        cte_having = [_simplify_having(hp, cte_pv) for hp in (cte.having_param or [])]
        new_cte_steps.append(
            replace(
                cte,
                select_cols=cte_select,
                order_by_cols=cte_order,
                filters_param=cte_filters,
                having_param=cte_having,
            )
        )
    return replace(
        intent,
        select_cols=new_select,
        order_by_cols=new_order,
        filters_param=new_filters,
        having_param=new_having,
        cte_steps=new_cte_steps,
    )


def _column_is_aggregatable(qualified: str, schema_graph: SchemaGraph) -> bool:
    """Return ``True`` when ``qualified`` (``table.column``) maps to an aggregatable column."""

    if not qualified or "." not in qualified:
        return True
    tbl, col = qualified.split(".", 1)
    table_meta = schema_graph.tables.get(tbl) or schema_graph.tables.get(tbl.lower())
    if table_meta is None:
        return True
    cm = table_meta.columns.get(col) or table_meta.columns.get(col.lower())
    if cm is None:
        return True
    return cm.is_aggregatable


def _gate_expr_aggregatability(expr: NormalizedExpr, schema_graph: SchemaGraph) -> NormalizedExpr:
    """Strip identity offsets and unit coefficients on expressions whose primary column is non-aggregatable."""

    if _column_is_aggregatable(expr.primary_column, schema_graph):
        return expr
    new_groups: list[MulGroup] = []
    for g in expr.add_groups:
        if g.coefficient != 1.0 or g.coeff_param_key:
            g = replace(g, coefficient=1.0, coeff_param_key="")
        new_groups.append(g)
    new_sub_groups: list[MulGroup] = []
    for g in expr.sub_groups:
        if g.coefficient != 1.0 or g.coeff_param_key:
            g = replace(g, coefficient=1.0, coeff_param_key="")
        new_sub_groups.append(g)
    return replace(
        expr,
        add_groups=new_groups,
        sub_groups=new_sub_groups,
        add_values=[],
        sub_values=[],
    )


def apply_aggregatability_gate(intent: RuntimeIntent, schema_graph: SchemaGraph) -> RuntimeIntent:
    """
    Strip nonsensical numeric ornaments from expressions whose primary column is non-aggregatable.

    For PK / FK / IDENTIFIER columns and any other column with ``is_aggregatable == False`` the
    gate clears parameterized add/sub values and resets group coefficients to ``1.0``. Aggregatable
    columns (``NUMERIC_MEASURE`` and overrides) are left untouched.

    Args:

        intent: RuntimeIntent to gate.

        schema_graph: Schema lookup for column metadata.

    Returns:

        Gated RuntimeIntent.
    """

    def _gate_filter(fp: FilterParam) -> FilterParam:
        new_left = _gate_expr_aggregatability(fp.left_expr, schema_graph)
        new_right = _gate_expr_aggregatability(fp.right_expr, schema_graph) if fp.right_expr else None
        return replace(fp, left_expr=new_left, right_expr=new_right)

    def _gate_having(hp: HavingParam) -> HavingParam:
        new_left = _gate_expr_aggregatability(hp.left_expr, schema_graph)
        new_right = _gate_expr_aggregatability(hp.right_expr, schema_graph) if hp.right_expr else None
        return replace(hp, left_expr=new_left, right_expr=new_right)

    new_select = [
        replace(sc, expr=_gate_expr_aggregatability(sc.expr, schema_graph)) for sc in (intent.select_cols or [])
    ]
    new_order = [
        replace(obc, expr=_gate_expr_aggregatability(obc.expr, schema_graph)) for obc in (intent.order_by_cols or [])
    ]
    new_group = [_gate_expr_aggregatability(e, schema_graph) for e in (intent.group_by_cols or [])]
    new_filters = [_gate_filter(fp) for fp in (intent.filters_param or [])]
    new_having = [_gate_having(hp) for hp in (intent.having_param or [])]
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        cte_select = [
            replace(sc, expr=_gate_expr_aggregatability(sc.expr, schema_graph)) for sc in (cte.select_cols or [])
        ]
        cte_order = [
            replace(obc, expr=_gate_expr_aggregatability(obc.expr, schema_graph)) for obc in (cte.order_by_cols or [])
        ]
        cte_group = [_gate_expr_aggregatability(e, schema_graph) for e in (cte.group_by_cols or [])]
        cte_filters = [_gate_filter(fp) for fp in (cte.filters_param or [])]
        cte_having = [_gate_having(hp) for hp in (cte.having_param or [])]
        new_cte_steps.append(
            replace(
                cte,
                select_cols=cte_select,
                order_by_cols=cte_order,
                group_by_cols=cte_group,
                filters_param=cte_filters,
                having_param=cte_having,
            )
        )
    return replace(
        intent,
        select_cols=new_select,
        order_by_cols=new_order,
        group_by_cols=new_group,
        filters_param=new_filters,
        having_param=new_having,
        cte_steps=new_cte_steps,
    )


def _normalize_filter_scalar_on_left(fp: FilterParam) -> FilterParam:
    """
    Swap sides when left is scalar and right is column, flipping the operator.

    Ensures column or table.column expressions are on the left for validation and SQL generation.

    Args:

        fp: FilterParam to normalize.

    Returns:

        FilterParam with column on the left when applicable.
    """
    if not fp.right_expr:
        return fp
    left_cols = [c for c in extract_columns_from_expr(fp.left_expr) if "." in c]
    right_cols = [c for c in extract_columns_from_expr(fp.right_expr) if "." in c]
    if left_cols or not right_cols:
        return fp
    new_op = REVERSE_OP_MAP.get(fp.op, fp.op)
    return FilterParam(
        left_expr=fp.right_expr,
        op=new_op,
        right_expr=fp.left_expr,
        value_type=fp.value_type,
        param_key=fp.param_key,
        raw_value=fp.raw_value,
        bool_op=fp.bool_op,
        filter_group=fp.filter_group,
    )


def _normalized_expr_is_absent(expr: NormalizedExpr) -> bool:
    """
    Return True when *expr* carries no structural content for canonical predicate normalization.

    Treats bare ``column_ref`` and interval leaves as present; pure literals remain absent.
    """

    return (
        not expr.has_column_reference
        and not expr.add_groups
        and not expr.sub_groups
        and not expr.add_values
        and not expr.sub_values
        and not expr.agg_func
        and not expr.scalar_func
        and not expr.inner_scalar_func
    )


def _normalize_filter_canonical(fp: FilterParam) -> FilterParam:
    """
    Normalize a filter to canonical form with a non-empty expression on the left.

    When the left_expr is empty but right_expr is not, swaps the sides and reverses the comparison operator.

    Args:

        fp: FilterParam to normalize.

    Returns:

        FilterParam with the heavier side on the left.
    """
    if _normalized_expr_is_absent(fp.left_expr) and fp.right_expr:
        new_op = REVERSE_OP_MAP.get(fp.op, fp.op)
        return FilterParam(
            left_expr=fp.right_expr,
            op=new_op,
            right_expr=fp.left_expr,
            value_type=fp.value_type,
            param_key=fp.param_key,
            bool_op=fp.bool_op,
            filter_group=fp.filter_group,
        )
    return fp


def _normalize_having_canonical(hp: HavingParam) -> HavingParam:
    """
    Normalize a having condition to canonical form with a non-empty expression on the left.

    Args:

        hp: HavingParam to normalize.

    Returns:

        HavingParam with the heavier side on the left.
    """
    if _normalized_expr_is_absent(hp.left_expr) and hp.right_expr:
        new_op = REVERSE_OP_MAP.get(hp.op, hp.op)
        return HavingParam(
            left_expr=hp.right_expr,
            op=new_op,
            right_expr=hp.left_expr,
            value_type=hp.value_type,
            param_key=hp.param_key,
            bool_op=hp.bool_op,
            filter_group=hp.filter_group,
        )
    return hp


def _normalize_col_to_col_filter(fp: FilterParam) -> FilterParam:
    """
    Normalize an expr-vs-expr filter so the lexicographically smaller signature is on the left.

    Args:

        fp: FilterParam with a right_expr (col-vs-col filter).

    Returns:

        FilterParam with sides swapped and operator reversed if needed.
    """
    if fp.right_expr and not fp.param_key:
        left_sig = fp.left_expr.signature_key
        right_sig = fp.right_expr.signature_key
        if left_sig > right_sig:
            new_op = REVERSE_OP_MAP.get(fp.op, fp.op)
            return FilterParam(
                left_expr=fp.right_expr,
                op=new_op,
                right_expr=fp.left_expr,
                value_type=fp.value_type,
                param_key=fp.param_key,
                bool_op=fp.bool_op,
                filter_group=fp.filter_group,
            )
    return fp


def _normalize_agg_to_agg_having(hp: HavingParam) -> HavingParam:
    """
    Normalize an expr-vs-expr having condition so the lexicographically smaller signature is on the left.

    Args:

        hp: HavingParam with a right_expr (agg-vs-agg having).

    Returns:

        HavingParam with sides swapped and operator reversed if needed.
    """
    if hp.right_expr and not hp.param_key:
        left_sig = hp.left_expr.signature_key
        right_sig = hp.right_expr.signature_key
        if left_sig > right_sig:
            new_op = REVERSE_OP_MAP.get(hp.op, hp.op)
            return HavingParam(
                left_expr=hp.right_expr,
                op=new_op,
                right_expr=hp.left_expr,
                value_type=hp.value_type,
                param_key=hp.param_key,
                bool_op=hp.bool_op,
                filter_group=hp.filter_group,
            )
    return hp


def _normalize_filter(fp: FilterParam) -> FilterParam:
    """
    Apply all normalization steps to a single filter.

    Runs scalar-on-left swap, canonical form, col-vs-col ordering, operator normalization, and value type normalization in sequence.

    Args:

        fp: FilterParam to normalize.

    Returns:

        Fully normalized FilterParam.
    """
    fp = _normalize_filter_scalar_on_left(fp)
    fp = _normalize_filter_canonical(fp)
    fp = _normalize_col_to_col_filter(fp)
    return replace(fp, op=normalize_op(fp.op), value_type=normalize_value_type(fp.value_type))


def _normalize_having(hp: HavingParam) -> HavingParam:
    """
    Apply all normalization steps to a single having condition.

    Runs canonical form, agg-vs-agg ordering, operator normalization, and value type normalization in sequence.

    Args:

        hp: HavingParam to normalize.

    Returns:

        Fully normalized HavingParam.
    """
    hp = _normalize_having_canonical(hp)
    hp = _normalize_agg_to_agg_having(hp)
    return replace(hp, op=normalize_op(hp.op), value_type=normalize_value_type(hp.value_type))


def _dedup_filters(filters: list[FilterParam]) -> list[FilterParam]:
    """
    Remove duplicate filters that share an identical structural signature, bool_op, and filter_group.

    Args:

        filters: List of FilterParam objects to deduplicate.

    Returns:

        List with the first occurrence of each unique (signature_key, bool_op, filter_group) retained.
    """
    seen: set[tuple[str, str, int | None]] = set()
    result: list[FilterParam] = []
    for fp in filters:
        key = (fp.signature_key, fp.bool_op, fp.filter_group)
        if key in seen:
            debug(f"[intent_resolve.dedup_filters] dropping duplicate filter: {key}")
            continue
        seen.add(key)
        result.append(fp)
    return result


def _dedup_having(having: list[HavingParam]) -> list[HavingParam]:
    """
    Remove duplicate having conditions that share an identical structural signature, bool_op, and filter_group.

    Args:

        having: List of HavingParam objects to deduplicate.

    Returns:

        List with the first occurrence of each unique (signature_key, bool_op, filter_group) retained.
    """
    seen: set[tuple[str, str, int | None]] = set()
    result: list[HavingParam] = []
    for hp in having:
        key = (hp.signature_key, hp.bool_op, hp.filter_group)
        if key in seen:
            debug(f"[intent_resolve.dedup_having] dropping duplicate having: {key}")
            continue
        seen.add(key)
        result.append(hp)
    return result


def normalize_filters_havings(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Apply all normalization, deduplication, and sorting rules to filters and having conditions.

    Args:

        intent: RuntimeIntent whose filters and having lists should be normalized.

    Returns:

        New RuntimeIntent with all filters and having conditions normalized, deduplicated, and sorted.
    """
    new_filters = [_normalize_filter(fp) for fp in (intent.filters_param or [])]
    new_having = [_normalize_having(hp) for hp in (intent.having_param or [])]
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_filters = _dedup_filters(sort_filters([_normalize_filter(fp) for fp in (cte.filters_param or [])]))
        cte_having = _dedup_having(sort_having([_normalize_having(hp) for hp in (cte.having_param or [])]))
        new_cte_steps.append(replace(cte, filters_param=cte_filters, having_param=cte_having))
    new_filters = _dedup_filters(sort_filters(new_filters))
    new_having = _dedup_having(sort_having(new_having))
    return replace(
        intent,
        filters_param=new_filters,
        having_param=new_having,
        cte_steps=new_cte_steps,
    )


def _allocate_branch_param_keys_in_case_registry(
    case_registry: list[CaseRegistryStep] | None,
    next_idx: int,
) -> tuple[list[CaseRegistryStep], int]:
    """Ensure CASE registry branch conditions have ``param_key`` or ``right_expr`` when required."""

    if not case_registry:
        return list(case_registry or []), next_idx
    out: list[CaseRegistryStep] = []
    idx = next_idx
    for step in case_registry:
        cw = step.case_when
        if cw is None or not cw.branches:
            out.append(step)
            continue
        new_branches = []
        changed = False
        for branch in cw.branches:
            cond = branch.condition
            if cond is None:
                new_branches.append(branch)
                continue
            if cond.op in NULL_CHECK_OPS or cond.right_expr is not None or cond.param_key:
                new_branches.append(branch)
                continue
            allocated_key = f"cb{idx}"
            idx += 1
            new_branches.append(replace(branch, condition=replace(cond, param_key=allocated_key)))
            changed = True
        if changed:
            out.append(replace(step, case_when=replace(cw, branches=new_branches)))
        else:
            out.append(step)
    return out, idx


def enforce_case_branch_param_keys(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Guarantee every CASE branch condition has a SQL-renderable binding.

    For each branch whose operator is not a null-check, ensures either ``right_expr`` or ``param_key`` is set; allocates a fresh ``cb*`` ``param_key`` otherwise so :func:`_render_case_branch_sql` cannot raise.
    """
    next_idx = 1
    new_main_cr, next_idx = _allocate_branch_param_keys_in_case_registry(intent.case_registry, next_idx)
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in intent.cte_steps or []:
        new_cte_cr, next_idx = _allocate_branch_param_keys_in_case_registry(cte.case_registry, next_idx)
        new_cte_steps.append(replace(cte, case_registry=new_cte_cr))
    return replace(intent, case_registry=new_main_cr, cte_steps=new_cte_steps)


def literal_in_logical_prose(logical: LogicalIntent, token: str) -> bool:
    """
    Return whether a literal token appears as a case-insensitive substring in planner prose.

    Args:

        logical: Planner intent whose prose fields are searched.

        token: Literal text such as a numeric string from a validation issue context.

    Returns:

        False when *token* is empty after stripping; otherwise substring membership in :func:`concat_logical_intent_prose`.
    """

    needle = (token or "").strip()
    if not needle:
        return False
    return needle.lower() in concat_logical_intent_prose(logical).lower()


def attribute_post_stage_b_issue(issue: IntentIssue, logical: LogicalIntent) -> IntentIssue:
    """
    Override ``responsible_stage`` for literal-bearing semantic issues using planner prose.

    Args:

        issue: Post-binding validation issue possibly in :data:`LITERAL_BEARING_CATEGORIES`.

        logical: Planner intent whose prose decides whether the literal was planner-owned.

    Returns:

        The same issue when no override applies; otherwise a copy with ``logical`` or ``format`` stage.
    """

    if issue.category not in LITERAL_BEARING_CATEGORIES:
        return issue
    raw_ctx = issue.context.get("value", "")
    token = str(raw_ctx).strip() if raw_ctx is not None else ""
    if literal_in_logical_prose(logical, token):
        return replace(issue, responsible_stage="format")
    return replace(issue, responsible_stage="logical")


def validate_tables_exist(tables: tuple[str, ...], schema: SchemaGraph) -> list[IntentIssue]:
    """
    Emit logical-stage issues for planner table tokens that are absent from the schema graph.

    Args:

        tables: Planner ``tables`` tuple from the top-level intent.

        schema: Authoritative schema graph.

    Returns:

        One issue per unknown table name.
    """

    known = frozenset(schema.tables.keys())
    out: list[IntentIssue] = []
    for name in tables:
        if name in known:
            continue
        out.append(
            make_intent_issue(
                issue_id=f"unknown_table_{name}",
                category=FailureCategory.UNKNOWN_TABLE,
                severity="error",
                message=f"Unknown table {name!r} in planner tables list.",
                context={"table": name},
                responsible_stage="logical",
            )
        )
    return out


def validate_cte_dependencies(logical: LogicalIntent) -> list[IntentIssue]:
    """
    Validate CTE ``depends_on`` references against base tables and strictly prior CTE names.

    Args:

        logical: Planner intent including ordered ``cte_steps``.

    Returns:

        Logical-stage issues for empty names, duplicates, or illegal forward or unknown references.
    """

    issues: list[IntentIssue] = []
    prior_names: set[str] = set()
    seen_step_names: set[str] = set()
    base_tables = frozenset(logical.tables)
    for idx, step in enumerate(logical.cte_steps):
        name = (step.name or "").strip()
        if not name:
            issues.append(
                make_intent_issue(
                    issue_id=f"cte_empty_name_{idx}",
                    category=FailureCategory.CTE_TABLE_REFERENCE,
                    severity="error",
                    message="CTE step is missing a non-empty name.",
                    context={"index": idx},
                    responsible_stage="logical",
                )
            )
            continue
        if name in seen_step_names:
            issues.append(
                make_intent_issue(
                    issue_id=f"cte_duplicate_name_{name}",
                    category=FailureCategory.CTE_TABLE_REFERENCE,
                    severity="error",
                    message=f"Duplicate CTE name {name!r}.",
                    context={"name": name},
                    responsible_stage="logical",
                )
            )
        seen_step_names.add(name)
        allowed = set(base_tables) | prior_names
        for dep in step.depends_on:
            if dep in allowed:
                continue
            issues.append(
                make_intent_issue(
                    issue_id=f"cte_dep_invalid_{idx}_{dep}",
                    category=FailureCategory.CTE_TABLE_REFERENCE,
                    severity="error",
                    message=f"CTE {name!r} depends_on entry {dep!r} is not a listed base table or a prior CTE.",
                    context={"cte": name, "depends_on": dep},
                    responsible_stage="logical",
                )
            )
        prior_names.add(name)
    return issues
