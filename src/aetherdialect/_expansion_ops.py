"""Deterministic seed-warmup expansion operators and `expand_gold_intents` orchestration. Implements operator registry, FK connectivity helpers, and multi-depth gold expansion."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Any, cast

from ._config import SeedWarmupConfig
from ._constants import CASE_ADD_OPS, CTE_ADD_OPS, HAVING_ADD_OPS, WINDOW_ADD_OPS, ExpansionOperatorId
from ._contracts_base import (
    ColumnRole,
    ExprValue,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RawValue,
    TableRole,
    expr_registry_ref,
)
from ._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SeedWarmupIntent,
    SelectCol,
    classify_seed_warmup_intent_complexity,
)
from ._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ExpansionMetadata,
    SchemaGraph,
    SchemaLimits,
    WindowRegistryStep,
    WindowSpec,
)
from ._core_utils import debug
from ._dialect import extra_filter_ops_for_engine
from ._intent_expr import extract_columns_from_expr, replace_refs_in_expr
from ._intent_process import apply_deterministic_repairs
from ._intent_repair import drop_invalid_case_registry_entries, repair_case_when_intent
from ._intent_resolve import check_qualified_refs_exist
from ._utils import intent_key
from ._validation_execute import curated_warmup_semantic_issues
from ._validation_schema import runtime_scope_registry_error_messages


def _expansion_path_ops(intent: SeedWarmupIntent) -> list[str]:
    em = intent.expansion_metadata
    if em is None:
        return []
    return list(em.expansion_path or [])


def expansion_compatible(parent: SeedWarmupIntent, candidate_op: str) -> bool:
    """Return whether ``candidate_op`` may be applied given the parent expansion path."""
    path = _expansion_path_ops(parent)
    path_set = set(path)
    if candidate_op in WINDOW_ADD_OPS and ExpansionOperatorId.DISTINCT_ADD in path_set:
        return False
    if candidate_op in CASE_ADD_OPS and any(op in WINDOW_ADD_OPS for op in path_set):
        return False
    if candidate_op in CTE_ADD_OPS and ExpansionOperatorId.DISTINCT_ADD in path_set:
        return False
    if candidate_op in WINDOW_ADD_OPS and any(op in WINDOW_ADD_OPS for op in path_set):
        return False
    if candidate_op == ExpansionOperatorId.EMI_MUTATE:
        if ExpansionOperatorId.EMI_MUTATE in path_set or ExpansionOperatorId.FILTER_OR_GROUP in path_set:
            return False
    if candidate_op == ExpansionOperatorId.FILTER_OR_GROUP and ExpansionOperatorId.EMI_MUTATE in path_set:
        return False
    if candidate_op == ExpansionOperatorId.GROUPBY_REMOVE and any(op in HAVING_ADD_OPS for op in path_set):
        return False
    if candidate_op.startswith("JOIN_") and ExpansionOperatorId.CTE_UNNEST_ADD in path_set:
        if len(parent.tables or []) >= 2:
            return False
    if candidate_op in CASE_ADD_OPS and len(parent.case_registry or []) > 0:
        return False
    if candidate_op in WINDOW_ADD_OPS and len(parent.window_registry or []) > 0:
        return False
    if not SeedWarmupConfig.ALLOW_HAVING_EXPR_EXPANSION and candidate_op == ExpansionOperatorId.HAVING_EXPR_ADD:
        return False
    if not SeedWarmupConfig.ALLOW_EMI_MUTATE_EXPANSION and candidate_op == ExpansionOperatorId.EMI_MUTATE:
        return False
    return True


def _accept_expansion_variant(
    var: SeedWarmupIntent,
    schema: SchemaGraph,
    *,
    parent: SeedWarmupIntent,
    candidate_op: str,
) -> SeedWarmupIntent | None:
    """Run repair, reference, compatibility, and semantic gates on one expansion variant."""
    if not expansion_compatible(parent, candidate_op):
        return None
    _, pre_errs = check_qualified_refs_exist(var.to_runtime_intent(), schema)
    if pre_errs:
        return None
    var = _deterministic_repair_warmup_seed(var, schema)
    _, post_errs = check_qualified_refs_exist(var.to_runtime_intent(), schema)
    if post_errs:
        return None
    if var.grain == "grouped" and not var.group_by_cols:
        return None
    if curated_warmup_semantic_issues(var.to_runtime_intent(), schema):
        return None
    return var


_EXPANSION_SUBTREE_POOL: list[SeedWarmupIntent] = []
_EXPANSION_SUBTREE_POOL_MAX: int = 128


def _record_expansion_subtree_pool(intent: SeedWarmupIntent) -> None:
    """Append a validated expansion snapshot for splice reuse when under the pool cap."""
    global _EXPANSION_SUBTREE_POOL
    if len(_EXPANSION_SUBTREE_POOL) >= _EXPANSION_SUBTREE_POOL_MAX:
        return
    _EXPANSION_SUBTREE_POOL.append(copy.deepcopy(intent))


def _join_path_in_multi(schema: SchemaGraph, a: str, b: str) -> bool:
    """Return True when ``join_paths_multi`` lists at least one path between two tables."""
    jpm = getattr(schema, "join_paths_multi", None) or {}
    row = jpm.get(a) or {}
    paths_ab = row.get(b) or []
    if paths_ab:
        return True
    row_b = jpm.get(b) or {}
    return bool(row_b.get(a))


def _tier_expansion_sort_key(intent: SeedWarmupIntent, counts: dict[str, int], denom: int) -> tuple[float, str]:
    """Higher debt against target tier proportions sorts earlier for coverage-guided expansion."""
    tier = classify_seed_warmup_intent_complexity(intent).value
    tgt = SeedWarmupConfig.COMPLEXITY_TARGET_PROPORTIONS.get(tier, 0.2)
    obs = counts.get(tier, 0) / max(denom, 1)
    debt = max(0.0, tgt - obs)
    hb = hashlib.sha256((intent.intent_id or "").encode()).hexdigest()
    return (-float(debt), hb)


def _append_case_registry_column(intent: SeedWarmupIntent, cw: CaseWhenExpr) -> None:
    """Append a ``case_registry`` step and a matching bare-registry select column."""
    registry = list(intent.case_registry or [])
    cid = f"c{len(registry) + 1:02d}"
    registry.append(CaseRegistryStep(registry_id=cid, case_when=cw))
    intent.case_registry = registry
    intent.select_cols = list(intent.select_cols or []) + [SelectCol(expr=NormalizedExpr.from_column(cid))]


def _append_window_registry_column(intent: SeedWarmupIntent, ws: WindowSpec) -> None:
    """Append a ``window_registry`` step and a matching bare-registry select column."""
    registry = list(intent.window_registry or [])
    wid = f"w{len(registry) + 1:02d}"
    registry.append(WindowRegistryStep(registry_id=wid, window_spec=ws))
    intent.window_registry = registry
    intent.select_cols = list(intent.select_cols or []) + [SelectCol(expr=NormalizedExpr.from_column(wid))]


def _finalize_registry_touch_seed(intent: SeedWarmupIntent, schema: SchemaGraph) -> SeedWarmupIntent | None:
    """Apply deterministic registry repairs after operators touch ``case_registry`` or branchy selects."""
    rt = intent.to_runtime_intent()
    rt = drop_invalid_case_registry_entries(rt, schema)
    rt = repair_case_when_intent(rt, schema)
    out = replace(
        intent,
        select_cols=list(rt.select_cols or []),
        case_registry=list(rt.case_registry or []),
        cte_steps=list(rt.cte_steps or []),
        order_by_cols=list(rt.order_by_cols or []),
    )
    if runtime_scope_registry_error_messages(out.to_runtime_intent()):
        return None
    return out


def _column_meta_for_qualified_col(
    full_col: str,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Return column metadata dict for a ``table.column`` reference, or empty."""
    table = _table_from_column_ref(full_col)
    if not table or table not in column_metadata:
        return {}
    bare = full_col.split(".", 1)[1] if "." in full_col else full_col
    return dict(column_metadata[table].get(bare, {}) or {})


def _agg_change_allowed_for_meta(new_agg: str, meta: dict[str, Any]) -> bool:
    """Return whether swapping to *new_agg* is consistent with column role and value type hints."""
    na = (new_agg or "").strip().lower()
    role = (meta.get("role") or "").strip().lower()
    vt = (meta.get("value_type") or "").strip().lower()
    if na == "count":
        return True
    numericish = (
        vt
        in (
            "numeric",
            "integer",
            "float",
            "decimal",
            "double",
            "bigint",
            "smallint",
        )
        or role == ColumnRole.NUMERIC_MEASURE.value.lower()
    )
    orderable = (
        numericish
        or vt
        in (
            "date",
            "datetime",
            "timestamp",
            "time",
            "string",
            "categorical",
            "text",
            "varchar",
        )
        or role
        in (
            ColumnRole.NUMERIC_MEASURE.value.lower(),
            ColumnRole.TEMPORAL.value.lower(),
            ColumnRole.CATEGORICAL.value.lower(),
        )
    )
    if na in ("sum", "avg"):
        return numericish
    if na in ("min", "max"):
        return orderable
    return False


def _strip_order_by_for_distinct_select(intent: SeedWarmupIntent) -> None:
    """Drop ``ORDER BY`` entries that are not keyed by a projected ``table.column`` in ``SELECT``."""
    allowed: set[str] = set()
    for sc in intent.select_cols or []:
        c = sc.expr.primary_column
        if c and "." in c:
            allowed.add(c)
    kept: list[OrderByCol] = []
    for o in intent.order_by_cols or []:
        c = o.expr.primary_column
        if c and c in allowed:
            kept.append(o)
    intent.order_by_cols = kept


def _intent_has_window_select(intent: SeedWarmupIntent) -> bool:
    """Return True when the intent declares window functions via. ``window_registry`` or select refs."""
    if intent.window_registry:
        return True
    return any((expr_registry_ref(sc.expr) or "").startswith("w") for sc in (intent.select_cols or []))


def _get_table_role(schema: SchemaGraph, table: str) -> str | None:
    """Return the table role string for `table` from `schema`."""
    tm = schema.tables.get(table)
    return tm.role if tm else None


def _table_from_column_ref(col_ref: str) -> str:
    """Extract the table name from a `table.column` reference."""
    if not col_ref or "." not in col_ref:
        return ""
    return col_ref.split(".", 1)[0]


def _build_column_metadata(
    schema: SchemaGraph,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build a nested `table` → `column` → metadata dict from `schema`."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for table_name, table_obj in schema.tables.items():
        result[table_name] = {}
        for col_name, col in table_obj.columns.items():
            result[table_name][col_name] = {
                "data_type": col.data_type,
                "role": col.role,
                "nullable": col.null_ratio > 0.0,
                "cardinality": getattr(col, "cardinality", None),
                "value_type": (col.value_type or "").strip().lower(),
                "is_foreign_key": bool(col.is_foreign_key),
                "fk_target": tuple(col.fk_target) if col.fk_target else None,
                "element_type": col.element_type or "",
                "sample_values": list(
                    getattr(col, "value_overlap_sample", None) or getattr(col, "semantic_distinct_values", None) or []
                )[:5],
            }
    return result


def _build_fk_map(schema: SchemaGraph) -> dict[str, list[dict[str, str]]]:
    """Build an FK adjacency map from `schema`."""
    fk_map: dict[str, list[dict[str, str]]] = {}
    for fk in schema.fk_edges:
        source = fk.src_table
        if source not in fk_map:
            fk_map[source] = []
        fk_map[source].append(
            {
                "source_column": fk.src_cols[0] if fk.src_cols else "",
                "target_table": fk.dst_table,
                "target_column": fk.dst_cols[0] if fk.dst_cols else "",
            }
        )
    return fk_map


def _tables_are_connected(
    tables: list[str],
    fk_map: dict[str, list[dict[str, str]]],
) -> bool:
    """Return whether all tables in `tables` form one connected. component via `fk_map`."""
    if len(tables) <= 1:
        return True
    adjacency: dict[str, set[str]] = {t: set() for t in tables}
    for source, fks in fk_map.items():
        if source not in adjacency:
            continue
        for fk in fks:
            target = fk.get("target_table", "")
            if target in adjacency:
                adjacency[source].add(target)
                adjacency[target].add(source)
    visited: set[str] = set()
    stack = [tables[0]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                stack.append(neighbor)
    return len(visited) == len(tables)


def _get_filterable_columns(
    schema: SchemaGraph,
    table_name: str,
) -> list[str]:
    """Return `table.column` references suitable for value filters."""
    if table_name not in schema.tables:
        return []
    table = schema.tables[table_name]
    return [
        f"{table_name}.{c}"
        for c, col in table.columns.items()
        if col.is_visible
        and col.role
        in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.TEMPORAL.value,
            ColumnRole.IDENTIFIER.value,
        )
    ]


def _get_groupable_columns(
    schema: SchemaGraph,
    table_name: str,
) -> list[str]:
    """Return `table.column` references suitable for `GROUP BY`."""
    if table_name not in schema.tables:
        return []
    table = schema.tables[table_name]
    return [
        f"{table_name}.{c}"
        for c, col in table.columns.items()
        if col.is_visible
        and col.role
        in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.TEMPORAL.value,
        )
    ]


def _get_temporal_columns(
    schema: SchemaGraph,
    table_name: str,
) -> list[str]:
    """Return `table.column` references for temporal columns."""
    if table_name not in schema.tables:
        return []
    table = schema.tables[table_name]
    return [
        f"{table_name}.{c}"
        for c, col in table.columns.items()
        if col.is_visible and col.role == ColumnRole.TEMPORAL.value
    ]


def _get_numeric_measure_columns(
    schema: SchemaGraph,
    table_name: str,
) -> list[str]:
    """Return `table.column` references for numeric measure columns."""
    if table_name not in schema.tables:
        return []
    table = schema.tables[table_name]
    return [
        f"{table_name}.{c}"
        for c, col in table.columns.items()
        if col.is_visible and col.role == ColumnRole.NUMERIC_MEASURE.value
    ]


def _get_categorical_identifier_columns(
    schema: SchemaGraph,
    table_name: str,
) -> list[str]:
    """Return `table.column` references suitable as `PARTITION BY` keys. for windows."""
    if table_name not in schema.tables:
        return []
    table = schema.tables[table_name]
    return [
        f"{table_name}.{c}"
        for c, col in table.columns.items()
        if col.is_visible
        and col.role
        in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.IDENTIFIER.value,
        )
    ]


def _filter_value_type_and_op_from_metadata(meta: dict[str, Any]) -> tuple[str | None, str]:
    """Map schema column metadata to a ``FilterParam`` semantic type. and. default comparison op."""
    if meta.get("element_type"):
        return None, "="
    vt = (meta.get("value_type") or "").strip().lower()
    dt = (meta.get("data_type") or "").strip().lower()
    role = (meta.get("role") or "").strip().lower()
    blob = f"{vt} {dt} {role}"
    if "array" in vt or "array" in dt:
        return None, "="
    if role == ColumnRole.BOOLEAN.value or "boolean" in blob:
        return "boolean", "="
    if any(x in blob for x in ("numeric", "integer", "float", "decimal", "number")):
        return "number", "="
    if any(x in blob for x in ("date", "datetime", "timestamp", "temporal")):
        return "date", ">="
    return "string", "="


def _rewrite_table_qualifier(
    intent: SeedWarmupIntent,
    old_table: str,
    new_table: str,
    schema: SchemaGraph,
) -> bool:
    """Rewrite qualified ``old_table`` column references to. ``new_table`` when bare names exist."""
    new_tm = schema.tables.get(new_table)
    if new_tm is None:
        return False
    new_cols = set(new_tm.columns.keys())
    refs: set[str] = set()

    def _note_expr(ex: NormalizedExpr | None) -> None:
        if ex is None:
            return
        for c in extract_columns_from_expr(ex):
            if "." in c:
                refs.add(c)

    def _note_filter(fp: FilterParam) -> None:
        _note_expr(fp.left_expr)
        if fp.right_expr:
            _note_expr(fp.right_expr)

    for sc in intent.select_cols or []:
        _note_expr(sc.expr)
    for g in intent.group_by_cols or []:
        _note_expr(g)
    for ob in intent.order_by_cols or []:
        _note_expr(ob.expr)
    for fp in intent.filters_param or []:
        _note_filter(fp)
    for hp in intent.having_param or []:
        _note_expr(hp.left_expr)
        if hp.right_expr:
            _note_expr(hp.right_expr)
    for wr in intent.window_registry or []:
        ws = wr.window_spec
        for p in ws.partition_by or []:
            _note_expr(p)
        for o in ws.order_by or []:
            _note_expr(o.expr)
        _note_expr(ws.argument)
    for cr in intent.case_registry or []:
        cw = cr.case_when
        for br in cw.branches or []:
            _note_filter(br.condition)
            _note_expr(br.result)
        _note_expr(cw.else_result)

    for ref in refs:
        tbl, bare = ref.split(".", 1)
        if tbl != old_table:
            continue
        if bare not in new_cols:
            return False

    def _repl(col_ref: str) -> str:
        if "." not in col_ref:
            return col_ref
        tbl, bare = col_ref.split(".", 1)
        if tbl == old_table:
            return f"{new_table}.{bare}"
        return col_ref

    def _rex(ex: NormalizedExpr | None) -> NormalizedExpr | None:
        if ex is None:
            return None
        return replace_refs_in_expr(ex, _repl)

    intent.select_cols = [replace(sc, expr=_rex(sc.expr) or sc.expr) for sc in (intent.select_cols or [])]
    intent.group_by_cols = [_rex(g) or g for g in (intent.group_by_cols or [])]
    intent.order_by_cols = [replace(ob, expr=_rex(ob.expr) or ob.expr) for ob in (intent.order_by_cols or [])]
    intent.filters_param = [
        replace(
            fp,
            left_expr=_rex(fp.left_expr) or fp.left_expr,
            right_expr=_rex(fp.right_expr) if fp.right_expr else None,
        )
        for fp in (intent.filters_param or [])
    ]
    intent.having_param = [
        replace(
            hp,
            left_expr=_rex(hp.left_expr) or hp.left_expr,
            right_expr=_rex(hp.right_expr) if hp.right_expr else None,
        )
        for hp in (intent.having_param or [])
    ]
    new_wr: list[WindowRegistryStep] = []
    for wr in intent.window_registry or []:
        ws = wr.window_spec
        new_ws = replace(
            ws,
            partition_by=[_rex(p) or p for p in (ws.partition_by or [])],
            order_by=[replace(o, expr=_rex(o.expr) or o.expr) for o in (ws.order_by or [])],
            argument=_rex(ws.argument) if ws.argument is not None else None,
        )
        new_wr.append(replace(wr, window_spec=new_ws))
    intent.window_registry = new_wr
    new_cr: list[CaseRegistryStep] = []
    for cr in intent.case_registry or []:
        cw = cr.case_when
        new_branches: list[CaseWhenBranch] = []
        for br in cw.branches or []:
            cond = br.condition
            new_cond = replace(
                cond,
                left_expr=_rex(cond.left_expr) or cond.left_expr,
                right_expr=_rex(cond.right_expr) if cond.right_expr else None,
            )
            new_branches.append(
                replace(
                    br,
                    condition=new_cond,
                    result=_rex(br.result) or br.result,
                )
            )
        new_cw = replace(
            cw,
            branches=new_branches,
            else_result=(_rex(cw.else_result) if cw.else_result is not None else cw.else_result),
        )
        new_cr.append(replace(cr, case_when=new_cw))
    intent.case_registry = new_cr
    return True


def _get_dimension_tables(schema: SchemaGraph) -> list[str]:
    """Return every table name whose role is dimension."""
    return [t for t, info in schema.tables.items() if info.role == TableRole.DIMENSION.value]


def _add_expansion_metadata(
    intent: SeedWarmupIntent,
    operator: str,
) -> None:
    """Attach expansion metadata to `intent` in place for `operator`."""
    if intent.expansion_metadata is None:
        intent.expansion_metadata = ExpansionMetadata(
            parent_intent_id="",
            operator=operator,
            depth=1,
            expansion_path=[operator],
        )
    else:
        intent.expansion_metadata = ExpansionMetadata(
            parent_intent_id=(intent.expansion_metadata.parent_intent_id or intent.intent_id),
            operator=operator,
            depth=(intent.expansion_metadata.depth or 0) + 1,
            expansion_path=((intent.expansion_metadata.expansion_path or []) + [operator]),
        )


def _filter_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_ADD: add one value-based filter per filterable column not yet filtered."""
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []

    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_filterable_columns(schema, table):
            if col in current_filter_cols:
                continue
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            vtype, fop = _filter_value_type_and_op_from_metadata(meta)
            if vtype is None:
                continue
            new_intent = copy.deepcopy(intent)
            new_filter = FilterParam(
                left_expr=NormalizedExpr.from_column(col),
                op=fop,
                value_type=vtype,
                param_key=f"f_{col.replace('.', '_')}",
            )
            new_intent.filters_param = list(new_intent.filters_param or []) + [new_filter]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_ADD)
            results.append(new_intent)
    return results


def _filter_expr_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_EXPR_ADD: add column-vs-column comparisons for same-type column pairs."""
    existing = set()
    for f in intent.filters_param or []:
        if f.right_expr:
            existing.add(
                (
                    f.left_expr.primary_column,
                    f.op,
                    f.right_expr.primary_column,
                )
            )
    if len(existing) >= SeedWarmupConfig.MAX_EXPR_COMPARISONS:
        return []

    cols_meta: list[tuple[str, str, str, str, bool, tuple[str, str] | None]] = []
    for table in intent.tables or []:
        if table not in column_metadata:
            continue
        for col_name, col_info in column_metadata[table].items():
            full_col = f"{table}.{col_name}"
            vt = (col_info.get("value_type") or "").strip().lower()
            role = (col_info.get("role") or "").strip().lower()
            cols_meta.append(
                (
                    full_col,
                    vt,
                    role,
                    col_info.get("data_type", "unknown") or "unknown",
                    bool(col_info.get("is_foreign_key")),
                    col_info.get("fk_target"),
                )
            )

    results: list[SeedWarmupIntent] = []
    for i, left_tup in enumerate(cols_meta):
        left_col, left_vt, left_role, left_dt, left_fk, left_tgt = left_tup
        for right_tup in cols_meta[i + 1 :]:
            right_col, right_vt, right_role, right_dt, right_fk, right_tgt = right_tup
            if left_vt != right_vt or left_role != right_role:
                continue
            if left_fk and right_fk and left_tgt != right_tgt:
                continue
            if bool(left_fk) != bool(right_fk) and (
                left_role == ColumnRole.AUDIT.value or right_role == ColumnRole.AUDIT.value
            ):
                continue
            for op in ["=", ">", "<"]:
                if (left_col, op, right_col) in existing:
                    continue
                new_intent = copy.deepcopy(intent)
                new_filter = FilterParam(
                    left_expr=NormalizedExpr.from_column(left_col),
                    op=op,
                    right_expr=NormalizedExpr.from_column(right_col),
                    value_type="column",
                    param_key="",
                )
                new_intent.filters_param = list(new_intent.filters_param or []) + [new_filter]
                _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_EXPR_ADD)
                results.append(new_intent)
    return results


def _swap_agg_func(expr: NormalizedExpr, new_agg: str) -> NormalizedExpr:
    """Return `expr` with its aggregation function switched to. `new_agg`."""
    if expr.agg_func:
        return replace(expr, agg_func=new_agg)
    if expr.add_groups and expr.add_groups[0].agg_func:
        new_group = replace(expr.add_groups[0], agg_func=new_agg)
        return replace(expr, add_groups=[new_group] + list(expr.add_groups[1:]))
    return NormalizedExpr.from_agg(new_agg, expr.primary_column)


def _agg_change(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """AGG_CHANGE: swap aggregation function on each aggregated select column."""
    results: list[SeedWarmupIntent] = []
    alt_aggs = ["count", "sum", "avg", "min", "max"]

    for sc in intent.select_cols or []:
        if not sc.is_aggregated:
            continue
        sc_col = sc.expr.primary_column
        sc_term = sc.expr.primary_term
        meta = _column_meta_for_qualified_col(sc_col, column_metadata)
        for new_agg in alt_aggs:
            if not _agg_change_allowed_for_meta(new_agg, meta):
                continue
            new_term = f"{new_agg}({sc_col})"
            if new_term.lower() == sc_term.lower():
                continue
            new_intent = copy.deepcopy(intent)
            for i, s in enumerate(new_intent.select_cols or []):
                if s.expr.primary_column == sc_col and s.expr.primary_term == sc_term:
                    new_expr = _swap_agg_func(s.expr, new_agg)
                    new_intent.select_cols[i] = SelectCol(expr=new_expr)
                    break
            _add_expansion_metadata(new_intent, ExpansionOperatorId.AGG_CHANGE)
            results.append(new_intent)
    return results


def _groupby_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """GROUPBY_ADD: add one `GROUP BY` column per groupable column not yet grouped."""
    current_gb = {g.primary_column for g in (intent.group_by_cols or [])}
    if len(current_gb) >= SeedWarmupConfig.MAX_GROUPBY:
        return []
    if intent.grain == "scalar":
        return []

    has_agg_sel = any(sc.is_aggregated for sc in intent.select_cols or [])
    has_hav = len(intent.having_param or []) > 0
    if not has_agg_sel and not has_hav:
        return []

    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_groupable_columns(schema, table):
            if col in current_gb:
                continue
            new_intent = copy.deepcopy(intent)
            new_intent.group_by_cols = sorted(
                list(intent.group_by_cols or []) + [NormalizedExpr.from_column(col)],
                key=lambda g: g.signature_key,
            )
            if new_intent.grain == "row_level":
                new_intent.grain = "grouped"
            _add_expansion_metadata(new_intent, ExpansionOperatorId.GROUPBY_ADD)
            results.append(new_intent)
    return results


def _orderby_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """ORDERBY_ADD: add `ORDER BY` for each select or group-by column in ascending and descending order."""
    current_ob = {o.expr.primary_column for o in (intent.order_by_cols or [])}
    if intent.grain == "grouped":
        candidates = [g.primary_column for g in (intent.group_by_cols or [])]
        for sc in intent.select_cols or []:
            if sc.is_aggregated and sc.expr.primary_column not in candidates:
                candidates.append(sc.expr.primary_column)
    else:
        candidates = [g.primary_column for g in (intent.group_by_cols or [])]
        for sc in intent.select_cols or []:
            if sc.expr.primary_column not in candidates:
                candidates.append(sc.expr.primary_column)

    results: list[SeedWarmupIntent] = []
    for col in candidates:
        if col in current_ob:
            continue
        for direction in ["ASC", "DESC"]:
            new_intent = copy.deepcopy(intent)
            new_order = OrderByCol(
                expr=NormalizedExpr.from_column(col),
                direction=direction,
            )
            new_intent.order_by_cols = list(new_intent.order_by_cols or []) + [new_order]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.ORDERBY_ADD)
            results.append(new_intent)
    return results


def _having_value_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """HAVING_VALUE_ADD: add `HAVING` with a value threshold for grouped intents."""
    if intent.grain != "grouped" or not intent.group_by_cols:
        return []
    existing = {(h.left_expr.primary_term, h.op) for h in (intent.having_param or [])}
    numeric_targets: list[str] = []
    for tbl in intent.tables or []:
        numeric_targets.extend(_get_numeric_measure_columns(schema, tbl))
    first_measure = numeric_targets[0] if numeric_targets else None
    results: list[SeedWarmupIntent] = []
    for agg_func in ["count", "sum", "avg", "min", "max"]:
        target = "*" if agg_func == "count" else (first_measure or "")
        if agg_func != "count" and not target:
            continue
        for op in [">", "<", ">=", "<="]:
            left_agg = f"{agg_func}({target})" if target != "*" else f"{agg_func}(*)"
            if (left_agg, op) in existing:
                continue
            new_intent = copy.deepcopy(intent)
            leaf = "*" if agg_func == "count" else target
            new_having = HavingParam(
                left_expr=NormalizedExpr.from_agg(agg_func, leaf),
                op=op,
                value_type="number",
                param_key=f"h_{agg_func}_{op.replace('<', 'lt').replace('>', 'gt').replace('=', 'e')}",
            )
            new_intent.having_param = list(new_intent.having_param or []) + [new_having]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.HAVING_VALUE_ADD)
            results.append(new_intent)
    return results


def _having_expr_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """HAVING_EXPR_ADD: add `HAVING` aggregate-vs-aggregate comparisons for grouped intents."""
    if intent.grain != "grouped" or not intent.group_by_cols:
        return []
    existing = {(h.left_expr.primary_term, h.op) for h in (intent.having_param or [])}
    agg_cols = [sc.expr.primary_column for sc in (intent.select_cols or []) if sc.is_aggregated]
    numeric_targets: list[str] = []
    for tbl in intent.tables or []:
        numeric_targets.extend(_get_numeric_measure_columns(schema, tbl))
    target_col: str | None = None
    for cand in agg_cols:
        if cand and cand != "*":
            target_col = cand
            break
    if target_col is None:
        target_col = numeric_targets[0] if numeric_targets else None
    if target_col is None:
        return []

    agg_pairs = [("count", "avg", ">"), ("sum", "count", "<"), ("avg", "min", ">=")]
    results: list[SeedWarmupIntent] = []
    for left_agg, right_agg, op in agg_pairs:
        if left_agg != "count" and (not target_col or target_col == "*"):
            continue
        left_leaf = "*" if left_agg == "count" else target_col
        right_leaf = "*" if right_agg == "count" else target_col
        left_term = f"{left_agg}({left_leaf})" if left_leaf != "*" else f"{left_agg}(*)"
        if (left_term, op) in existing:
            continue
        new_intent = copy.deepcopy(intent)
        new_having = HavingParam(
            left_expr=NormalizedExpr.from_agg(left_agg, left_leaf),
            op=op,
            right_expr=NormalizedExpr.from_agg(right_agg, right_leaf),
            value_type="expression",
            param_key="",
        )
        new_intent.having_param = list(new_intent.having_param or []) + [new_having]
        _add_expansion_metadata(new_intent, ExpansionOperatorId.HAVING_EXPR_ADD)
        results.append(new_intent)
    return results


def _filter_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_REMOVE: remove each filter one at a time."""
    current = intent.filters_param or []
    if not current:
        return []
    results: list[SeedWarmupIntent] = []
    for i in range(len(current)):
        new_intent = copy.deepcopy(intent)
        fp = list(new_intent.filters_param or [])
        new_intent.filters_param = list(fp[:i] + fp[i + 1 :])
        _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_REMOVE)
        results.append(new_intent)
    return results


def _groupby_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """GROUPBY_REMOVE: remove each `GROUP BY` column one at a time when more than one exists."""
    current = list(intent.group_by_cols or [])
    if len(current) <= 1:
        return []
    results: list[SeedWarmupIntent] = []
    for gb in current:
        new_intent = copy.deepcopy(intent)
        ng = list(new_intent.group_by_cols or [])
        new_intent.group_by_cols = [g for g in ng if g.primary_column != gb.primary_column]
        _add_expansion_metadata(new_intent, ExpansionOperatorId.GROUPBY_REMOVE)
        results.append(new_intent)
    return results


def _having_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """HAVING_REMOVE: remove each `HAVING` condition one at a time."""
    current = intent.having_param or []
    if not current:
        return []
    results: list[SeedWarmupIntent] = []
    for i in range(len(current)):
        new_intent = copy.deepcopy(intent)
        hp = list(new_intent.having_param or [])
        new_intent.having_param = list(hp[:i] + hp[i + 1 :])
        _add_expansion_metadata(new_intent, ExpansionOperatorId.HAVING_REMOVE)
        results.append(new_intent)
    return results


def _join_dimension_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    fk_map: dict[str, list[dict[str, str]]],
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """JOIN_DIMENSION_ADD: add each FK-connected dimension table not already in the intent."""
    current = set(intent.tables or [])
    if len(current) >= SeedWarmupConfig.MAX_TABLES:
        return []
    results: list[SeedWarmupIntent] = []
    for table in list(current):
        for fk in fk_map.get(table, []):
            target = fk.get("target_table")
            if not target or target in current:
                continue
            if (_get_table_role(schema, target) or TableRole.FACT.value) != TableRole.DIMENSION.value:
                continue
            bridge_ok = False
            for src_tbl in current:
                if _join_path_in_multi(schema, src_tbl, target):
                    bridge_ok = True
                    break
            if not bridge_ok:
                continue
            new_tables = list(current | {target})
            if not _tables_are_connected(new_tables, fk_map):
                continue
            new_intent = copy.deepcopy(intent)
            new_intent.tables = sorted(new_tables)
            _add_expansion_metadata(new_intent, ExpansionOperatorId.JOIN_DIMENSION_ADD)
            results.append(new_intent)
    return results


def _join_fact_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    fk_map: dict[str, list[dict[str, str]]],
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """JOIN_FACT_ADD: add each FK-connected fact table not already in the intent."""
    current = set(intent.tables or [])
    if len(current) >= SeedWarmupConfig.MAX_TABLES:
        return []
    results: list[SeedWarmupIntent] = []
    seen_targets: set[str] = set()
    for table in list(current):
        for fk in fk_map.get(table, []):
            target = fk.get("target_table")
            if not target or target in current or target in seen_targets:
                continue
            if (_get_table_role(schema, target) or TableRole.FACT.value) != TableRole.FACT.value:
                continue
            bridge_ok = False
            for src_tbl in current:
                if _join_path_in_multi(schema, src_tbl, target):
                    bridge_ok = True
                    break
            if not bridge_ok:
                continue
            new_tables = list(current | {target})
            if not _tables_are_connected(new_tables, fk_map):
                continue
            seen_targets.add(target)
            new_intent = copy.deepcopy(intent)
            new_intent.tables = sorted(new_tables)
            _add_expansion_metadata(new_intent, ExpansionOperatorId.JOIN_FACT_ADD)
            results.append(new_intent)

        for other_table, other_fks in fk_map.items():
            if other_table in current or other_table in seen_targets:
                continue
            if (_get_table_role(schema, other_table) or TableRole.FACT.value) != TableRole.FACT.value:
                continue
            for ofk in other_fks:
                if ofk.get("target_table") == table:
                    bridge_ok = False
                    for src_tbl in current:
                        if _join_path_in_multi(schema, src_tbl, other_table):
                            bridge_ok = True
                            break
                    if not bridge_ok:
                        continue
                    new_tables = list(current | {other_table})
                    if not _tables_are_connected(new_tables, fk_map):
                        continue
                    seen_targets.add(other_table)
                    new_intent = copy.deepcopy(intent)
                    new_intent.tables = sorted(new_tables)
                    _add_expansion_metadata(
                        new_intent,
                        ExpansionOperatorId.JOIN_FACT_ADD,
                    )
                    results.append(new_intent)
                    break
    return results


def _dimension_swap(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    fk_map: dict[str, list[dict[str, str]]],
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """DIMENSION_SWAP: swap each dimension for an alternative FK- connected dimension."""
    current = list(intent.tables or [])
    results: list[SeedWarmupIntent] = []
    for i, table in enumerate(current):
        if (_get_table_role(schema, table) or TableRole.FACT.value) != TableRole.DIMENSION.value:
            continue
        fact_tables = [
            t for t in current if (_get_table_role(schema, t) or TableRole.FACT.value) == TableRole.FACT.value
        ]
        if not fact_tables:
            continue
        for dim in _get_dimension_tables(schema):
            if dim == table or dim in current:
                continue
            can_join = any(fk.get("target_table") == dim for fact in fact_tables for fk in fk_map.get(fact, []))
            if not can_join:
                continue
            new_tables = current[:i] + [dim] + current[i + 1 :]
            new_intent = copy.deepcopy(intent)
            new_intent.tables = sorted(new_tables)
            if not _rewrite_table_qualifier(new_intent, table, dim, schema):
                continue
            _add_expansion_metadata(new_intent, ExpansionOperatorId.DIMENSION_SWAP)
            results.append(new_intent)
    return results


def _table_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    fk_map: dict[str, list[dict[str, str]]],
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """TABLE_REMOVE: remove each removable dimension table and prune dependent clauses."""
    current = list(intent.tables or [])
    if len(current) <= 1:
        return []
    results: list[SeedWarmupIntent] = []
    for i, table in enumerate(current):
        if (_get_table_role(schema, table) or TableRole.FACT.value) != TableRole.DIMENSION.value:
            continue
        new_tables = current[:i] + current[i + 1 :]
        if not new_tables:
            continue
        if not _tables_are_connected(new_tables, fk_map):
            continue
        new_intent = copy.deepcopy(intent)
        new_intent.tables = sorted(new_tables)
        ts = set(new_tables)

        def _having_refs_removed_table(hp: HavingParam, *, _table: str = table) -> bool:
            for col in extract_columns_from_expr(hp.left_expr):
                if "." in col and _table_from_column_ref(col) == _table:
                    return True
            if hp.right_expr:
                for col in extract_columns_from_expr(hp.right_expr):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
            return False

        def _window_refs_removed_table(ws: WindowSpec, *, _table: str = table) -> bool:
            for p in ws.partition_by or []:
                for col in extract_columns_from_expr(p):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
            for o in ws.order_by or []:
                for col in extract_columns_from_expr(o.expr):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
            if ws.argument:
                for col in extract_columns_from_expr(ws.argument):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
            return False

        def _case_refs_removed_table(cr: CaseRegistryStep, *, _table: str = table) -> bool:
            cw = cr.case_when
            for br in cw.branches or []:
                for col in extract_columns_from_expr(br.result):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
                for col in extract_columns_from_expr(br.condition.left_expr):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
                if br.condition.right_expr:
                    for col in extract_columns_from_expr(br.condition.right_expr):
                        if "." in col and _table_from_column_ref(col) == _table:
                            return True
            if cw.else_result:
                for col in extract_columns_from_expr(cw.else_result):
                    if "." in col and _table_from_column_ref(col) == _table:
                        return True
            return False

        new_filters: list[FilterParam] = []
        for f in new_intent.filters_param or []:
            if _table_from_column_ref(f.left_expr.primary_column) not in ts:
                continue
            if f.right_expr and _table_from_column_ref(f.right_expr.primary_column) == table:
                continue
            new_filters.append(f)
        new_intent.filters_param = new_filters

        new_intent.having_param = [h for h in (new_intent.having_param or []) if not _having_refs_removed_table(h)]

        dropped_win: set[str] = {
            wr.registry_id for wr in (new_intent.window_registry or []) if _window_refs_removed_table(wr.window_spec)
        }
        new_intent.window_registry = [
            wr for wr in (new_intent.window_registry or []) if wr.registry_id not in dropped_win
        ]

        dropped_case: set[str] = {
            cr.registry_id for cr in (new_intent.case_registry or []) if _case_refs_removed_table(cr)
        }
        new_intent.case_registry = [cr for cr in (new_intent.case_registry or []) if cr.registry_id not in dropped_case]

        new_intent.group_by_cols = [
            c for c in (new_intent.group_by_cols or []) if _table_from_column_ref(c.primary_column) in ts
        ]

        def _order_keeps(
            o: OrderByCol,
            *,
            _dropped_win: set[str] = dropped_win,
            _dropped_case: set[str] = dropped_case,
            _ts: set[str] = ts,
        ) -> bool:
            rid = expr_registry_ref(o.expr) or ""
            if rid.startswith("w") and rid in _dropped_win:
                return False
            if rid.startswith("c") and rid in _dropped_case:
                return False
            pr_t = _table_from_column_ref(o.expr.primary_column)
            if pr_t:
                return pr_t in _ts
            return True

        def _select_keeps(
            sc: SelectCol,
            *,
            _dropped_win: set[str] = dropped_win,
            _dropped_case: set[str] = dropped_case,
            _ts: set[str] = ts,
        ) -> bool:
            rid = expr_registry_ref(sc.expr) or ""
            if rid.startswith("w") and rid in _dropped_win:
                return False
            if rid.startswith("c") and rid in _dropped_case:
                return False
            pr_t = _table_from_column_ref(sc.expr.primary_column)
            if pr_t:
                return pr_t in _ts
            return True

        new_intent.order_by_cols = [o for o in (new_intent.order_by_cols or []) if _order_keeps(o)]
        new_intent.select_cols = [sc for sc in (new_intent.select_cols or []) if _select_keeps(sc)]
        if not new_intent.select_cols:
            continue
        _add_expansion_metadata(new_intent, ExpansionOperatorId.TABLE_REMOVE)
        results.append(new_intent)
    return results


def _bridge_intermediate_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    fk_map: dict[str, list[dict[str, str]]],
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """BRIDGE_INTERMEDIATE_ADD: add bridge tables that connect at least two tables already in the intent."""
    current = set(intent.tables or [])
    if len(current) >= SeedWarmupConfig.MAX_TABLES:
        return []
    results: list[SeedWarmupIntent] = []
    for bridge in schema.tables:
        if bridge in current:
            continue
        if (_get_table_role(schema, bridge) or TableRole.FACT.value) != TableRole.BRIDGE.value:
            continue
        connected = {fk.get("target_table") for fk in fk_map.get(bridge, []) if fk.get("target_table") in current}
        if len(connected) < 2:
            continue
        new_tables = list(current | {bridge})
        if not _tables_are_connected(new_tables, fk_map):
            continue
        new_intent = copy.deepcopy(intent)
        new_intent.tables = sorted(new_tables)
        _add_expansion_metadata(
            new_intent,
            ExpansionOperatorId.BRIDGE_INTERMEDIATE_ADD,
        )
        results.append(new_intent)
    return results


def _include_gold(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """INCLUDE_GOLD: include the gold intent unchanged with expansion metadata stamped."""
    gold_copy = copy.deepcopy(intent)
    _add_expansion_metadata(gold_copy, ExpansionOperatorId.INCLUDE_GOLD)
    return [gold_copy]


def _temp_extract_groupby(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """TEMP_EXTRACT_GROUPBY: wrap temporal columns with `extract(unit)` in select and group-by lists."""
    if intent.grain == "scalar":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_temporal_columns(schema, table):
            for unit in SeedWarmupConfig.EXTRACT_EXPANSION_UNITS:
                new_intent = copy.deepcopy(intent)
                extract_expr = NormalizedExpr.from_column(col)
                extract_expr = replace(
                    extract_expr,
                    scalar_func="extract",
                    scalar_func_args=[unit],
                )
                new_intent.select_cols = list(new_intent.select_cols or []) + [SelectCol(expr=extract_expr)]
                new_intent.group_by_cols = sorted(
                    list(new_intent.group_by_cols or []) + [extract_expr],
                    key=lambda g: g.signature_key,
                )
                if new_intent.grain == "row_level":
                    new_intent.grain = "grouped"
                _add_expansion_metadata(
                    new_intent,
                    ExpansionOperatorId.TEMP_EXTRACT_GROUPBY,
                )
                results.append(new_intent)
    return results


def _temp_date_trunc_groupby(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """TEMP_DATE_TRUNC_GROUPBY: wrap temporal columns with `date_trunc(unit)` in group-by and select lists."""
    if intent.grain == "scalar":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_temporal_columns(schema, table):
            for unit in SeedWarmupConfig.DATE_TRUNC_EXPANSION_UNITS:
                new_intent = copy.deepcopy(intent)
                trunc_expr = NormalizedExpr.from_column(col)
                trunc_expr = replace(
                    trunc_expr,
                    scalar_func="date_trunc",
                    scalar_func_args=[unit],
                )
                new_intent.select_cols = list(new_intent.select_cols or []) + [SelectCol(expr=trunc_expr)]
                new_intent.group_by_cols = sorted(
                    list(new_intent.group_by_cols or []) + [trunc_expr],
                    key=lambda g: g.signature_key,
                )
                if new_intent.grain == "row_level":
                    new_intent.grain = "grouped"
                _add_expansion_metadata(
                    new_intent,
                    ExpansionOperatorId.TEMP_DATE_TRUNC_GROUPBY,
                )
                results.append(new_intent)
    return results


def _temp_date_window_filter(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """TEMP_DATE_WINDOW_FILTER: add `date_window` filters on temporal columns using config presets."""
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_temporal_columns(schema, table):
            if col in current_filter_cols:
                continue
            for preset in SeedWarmupConfig.DATE_WINDOW_EXPANSION_PRESETS:
                new_intent = copy.deepcopy(intent)
                new_filter = FilterParam(
                    left_expr=NormalizedExpr.from_column(col),
                    op=">=",
                    value_type="date_window",
                    param_key="",
                    raw_value=dict(preset),
                )
                new_intent.filters_param = list(new_intent.filters_param or []) + [new_filter]
                _add_expansion_metadata(
                    new_intent,
                    ExpansionOperatorId.TEMP_DATE_WINDOW_FILTER,
                )
                results.append(new_intent)
    return results


def _temp_date_diff_filter(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """TEMP_DATE_DIFF_FILTER: add `date_diff` filters on temporal columns using config presets."""
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_temporal_columns(schema, table):
            if col in current_filter_cols:
                continue
            for preset in SeedWarmupConfig.DATE_DIFF_EXPANSION_PRESETS:
                new_intent = copy.deepcopy(intent)
                new_filter = FilterParam(
                    left_expr=NormalizedExpr.from_column(col),
                    op="<=",
                    value_type="date_diff",
                    param_key="",
                    raw_value=dict(preset),
                )
                new_intent.filters_param = list(new_intent.filters_param or []) + [new_filter]
                _add_expansion_metadata(
                    new_intent,
                    ExpansionOperatorId.TEMP_DATE_DIFF_FILTER,
                )
                results.append(new_intent)
    return results


def _num_round_select(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """NUM_ROUND_SELECT: wrap numeric-measure select columns with `round`."""
    numeric_cols: set[str] = set()
    for table in intent.tables or []:
        numeric_cols.update(_get_numeric_measure_columns(schema, table))

    results: list[SeedWarmupIntent] = []
    for idx, sc in enumerate(intent.select_cols or []):
        if sc.expr.primary_column not in numeric_cols:
            continue
        if sc.expr.scalar_func == "round":
            continue
        new_intent = copy.deepcopy(intent)
        new_expr = replace(
            new_intent.select_cols[idx].expr,
            scalar_func="round",
            scalar_func_args=[0],
        )
        new_intent.select_cols[idx] = SelectCol(expr=new_expr)
        _add_expansion_metadata(new_intent, ExpansionOperatorId.NUM_ROUND_SELECT)
        results.append(new_intent)
    return results


def _num_abs_filter(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """NUM_ABS_FILTER: wrap numeric filter left-hand expressions with `abs` for range operators."""
    results: list[SeedWarmupIntent] = []
    for idx, f in enumerate(intent.filters_param or []):
        if f.op not in (">", "<", ">=", "<="):
            continue
        if f.left_expr.scalar_func == "abs":
            continue
        col = f.left_expr.primary_column
        table = _table_from_column_ref(col)
        if not table or table not in column_metadata:
            continue
        bare = col.split(".", 1)[1] if "." in col else col
        col_info = column_metadata.get(table, {}).get(bare, {})
        if col_info.get("role") != ColumnRole.NUMERIC_MEASURE.value:
            continue
        new_intent = copy.deepcopy(intent)
        new_expr = replace(
            new_intent.filters_param[idx].left_expr,
            scalar_func="abs",
        )
        new_intent.filters_param[idx] = replace(
            new_intent.filters_param[idx],
            left_expr=new_expr,
        )
        _add_expansion_metadata(new_intent, ExpansionOperatorId.NUM_ABS_FILTER)
        results.append(new_intent)
    return results


def _distinct_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """DISTINCT_ADD: set ``distinct_select_index`` to the first select column when not already distinct."""
    if intent.distinct_select_index >= 0:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.distinct_select_index = 0
    _strip_order_by_for_distinct_select(new_intent)
    _add_expansion_metadata(new_intent, ExpansionOperatorId.DISTINCT_ADD)
    return [new_intent]


def _limit_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """LIMIT_ADD: add `limit` using representative values from config."""
    if intent.limit is not None:
        return []
    results: list[SeedWarmupIntent] = []
    for val in SeedWarmupConfig.LIMIT_EXPANSION_VALUES:
        new_intent = copy.deepcopy(intent)
        new_intent.limit = val
        _add_expansion_metadata(new_intent, ExpansionOperatorId.LIMIT_ADD)
        results.append(new_intent)
    return results


def _filter_or_group(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_OR_GROUP: convert pairs of existing conjunctive filters into OR groups."""
    filters = intent.filters_param or []
    if len(filters) < 2:
        return []
    results: list[SeedWarmupIntent] = []
    for i in range(len(filters)):
        for j in range(i + 1, len(filters)):
            fi, fj = filters[i], filters[j]
            if fi.right_expr or fj.right_expr:
                continue
            if fi.value_type in ("date_window", "date_diff"):
                continue
            if fj.value_type in ("date_window", "date_diff"):
                continue
            new_intent = copy.deepcopy(intent)
            next_group = max((fp.filter_group or 0) for fp in filters) + 1
            new_fi = replace(
                new_intent.filters_param[i],
                bool_op="OR",
                filter_group=next_group,
            )
            new_fj = replace(
                new_intent.filters_param[j],
                bool_op="OR",
                filter_group=next_group,
            )
            new_intent.filters_param[i] = new_fi
            new_intent.filters_param[j] = new_fj
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_OR_GROUP)
            results.append(new_intent)
    return results


def _window_rank_add(
    intent: SeedWarmupIntent,
    _schema: SchemaGraph,
    _column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """WINDOW_RANK_ADD: add `row_number` over group keys ordered by a select aggregate for grouped grain."""
    if _intent_has_window_select(intent):
        return []
    if intent.grain != "grouped":
        return []
    gbc = intent.group_by_cols or []
    if not gbc:
        return []
    agg_sc = next((sc for sc in (intent.select_cols or []) if sc.is_aggregated), None)
    if agg_sc is None:
        return []
    new_intent = copy.deepcopy(intent)
    ws = WindowSpec(
        function="row_number",
        partition_by=[copy.deepcopy(g) for g in gbc],
        order_by=[OrderByCol(expr=copy.deepcopy(agg_sc.expr), direction="DESC")],
    )
    _append_window_registry_column(new_intent, ws)
    _add_expansion_metadata(new_intent, ExpansionOperatorId.WINDOW_RANK_ADD)
    fin = _finalize_registry_touch_seed(new_intent, _schema)
    return [fin] if fin is not None else []


def _window_sum_partition_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    _column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """WINDOW_SUM_PARTITION_ADD: add `sum(measure) over (partition by dim)` style window columns for row-level grain."""
    if _intent_has_window_select(intent):
        return []
    if intent.grain != "row_level":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for measure in _get_numeric_measure_columns(schema, table):
            for dim in _get_categorical_identifier_columns(schema, table):
                if measure == dim:
                    continue
                new_intent = copy.deepcopy(intent)
                ws = WindowSpec(
                    function="sum",
                    partition_by=[NormalizedExpr.from_column(dim)],
                    order_by=[],
                    argument=NormalizedExpr.from_column(measure),
                )
                _append_window_registry_column(new_intent, ws)
                _add_expansion_metadata(new_intent, ExpansionOperatorId.WINDOW_SUM_PARTITION_ADD)
                fin = _finalize_registry_touch_seed(new_intent, schema)
                if fin is not None:
                    results.append(fin)
    return results


def _select_expr_pair_multiply(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """SELECT_EXPR_PAIR_MULTIPLY: add composed multiply-group expressions from numeric column pairs in select."""
    if intent.grain == "grouped":
        return []
    numeric_cols: list[str] = []
    for table in intent.tables or []:
        numeric_cols.extend(_get_numeric_measure_columns(schema, table))

    if len(numeric_cols) < 2:
        return []

    results: list[SeedWarmupIntent] = []
    for i, column in enumerate(numeric_cols):
        for other_column in numeric_cols[i + 1 :]:
            new_intent = copy.deepcopy(intent)
            composed = NormalizedExpr(
                add_groups=[
                    MulGroup(
                        multiply=[
                            NormalizedExpr.from_column(column),
                            NormalizedExpr.from_column(other_column),
                        ],
                    ),
                ],
            )
            new_intent.select_cols = list(new_intent.select_cols or []) + [SelectCol(expr=composed)]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.SELECT_EXPR_PAIR_MULTIPLY)
            results.append(new_intent)
    return results


def _select_case_label_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Append a numeric-labeled CASE over one measure column (threshold. vs else)."""
    _ = column_metadata
    if intent.grain == "grouped":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_numeric_measure_columns(schema, table):
            new_intent = copy.deepcopy(intent)
            cond = FilterParam(
                left_expr=NormalizedExpr.from_column(col),
                op=">",
                value_type="numeric",
                param_key=f"case_thr_{col.replace('.', '_')}",
            )
            branch = CaseWhenBranch(
                condition=cond,
                result=NormalizedExpr(add_values=[ExprValue(value=1.0)]),
            )
            cw = CaseWhenExpr(
                branches=[branch],
                else_result=NormalizedExpr(add_values=[ExprValue(value=0.0)]),
            )
            _append_case_registry_column(new_intent, cw)
            _add_expansion_metadata(new_intent, ExpansionOperatorId.SELECT_CASE_LABEL_ADD)
            fin = _finalize_registry_touch_seed(new_intent, schema)
            if fin is not None:
                results.append(fin)
    return results


def _window_lag_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    _column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Add a `lag` window column over row-level intents (partition + temporal order)."""
    if _intent_has_window_select(intent):
        return []
    if intent.grain != "row_level":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        temporal = _get_temporal_columns(schema, table)
        if not temporal:
            continue
        order_col = temporal[0]
        for measure in _get_numeric_measure_columns(schema, table):
            for dim in _get_categorical_identifier_columns(schema, table):
                if measure == dim:
                    continue
                new_intent = copy.deepcopy(intent)
                ws = WindowSpec(
                    function="lag",
                    partition_by=[NormalizedExpr.from_column(dim)],
                    order_by=[OrderByCol(expr=NormalizedExpr.from_column(order_col), direction="ASC")],
                    argument=NormalizedExpr.from_column(measure),
                )
                _append_window_registry_column(new_intent, ws)
                _add_expansion_metadata(new_intent, ExpansionOperatorId.WINDOW_LAG_ADD)
                fin = _finalize_registry_touch_seed(new_intent, schema)
                if fin is not None:
                    results.append(fin)
    return results


def _window_lead_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Add a `lead` window column (same shape as `_window_lag_add`)."""
    lag_variants = _window_lag_add(intent, schema, column_metadata)
    results: list[SeedWarmupIntent] = []
    for v in lag_variants:
        vc = copy.deepcopy(v)
        new_wr: list[WindowRegistryStep] = []
        for step in vc.window_registry or []:
            if step.window_spec.function == "lag":
                new_wr.append(replace(step, window_spec=replace(step.window_spec, function="lead")))
            else:
                new_wr.append(step)
        vc.window_registry = new_wr
        vc.expansion_metadata = None
        _add_expansion_metadata(vc, ExpansionOperatorId.WINDOW_LEAD_ADD)
        fin = _finalize_registry_touch_seed(vc, schema)
        if fin is not None:
            results.append(fin)
    return results


def _filter_ilike_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Add case-insensitive `ilike` filters on categorical string. columns (PostgreSQL only)."""
    _ = column_metadata
    if "ilike" not in extra_filter_ops_for_engine():
        return []
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        if table not in schema.tables:
            continue
        for col_name, col in schema.tables[table].columns.items():
            full = f"{table}.{col_name}"
            if full in current_filter_cols:
                continue
            if col.role != ColumnRole.CATEGORICAL.value:
                continue
            if col.value_type not in ("string", "categorical", ""):
                continue
            new_intent = copy.deepcopy(intent)
            new_filter = FilterParam(
                left_expr=NormalizedExpr.from_column(full),
                op="ilike",
                value_type="string",
                param_key=f"ilk_{col_name}",
            )
            new_intent.filters_param = list(new_intent.filters_param or []) + [new_filter]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_ILIKE_ADD)
            results.append(new_intent)
    return results


def _filter_array_contains_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Add `contains` filters for columns that declare an array. `element_type`."""
    _ = column_metadata
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        if table not in schema.tables:
            continue
        for col_name, col in schema.tables[table].columns.items():
            if not col.element_type:
                continue
            full = f"{table}.{col_name}"
            if full in current_filter_cols:
                continue
            new_intent = copy.deepcopy(intent)
            new_filter = FilterParam(
                left_expr=NormalizedExpr.from_column(full),
                op="contains",
                value_type="array",
                param_key=f"arr_{col_name}",
            )
            new_intent.filters_param = list(new_intent.filters_param or []) + [new_filter]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_ARRAY_CONTAINS_ADD)
            results.append(new_intent)
    return results


def _orderby_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Remove the last `ORDER BY` column when present."""
    _ = schema, column_metadata
    ob = list(intent.order_by_cols or [])
    if not ob:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.order_by_cols = ob[:-1]
    _add_expansion_metadata(new_intent, ExpansionOperatorId.ORDERBY_REMOVE)
    return [new_intent]


def _limit_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Clear a set `limit` when present."""
    _ = schema, column_metadata
    if intent.limit is None:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.limit = None
    _add_expansion_metadata(new_intent, ExpansionOperatorId.LIMIT_REMOVE)
    return [new_intent]


def _select_col_trim(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Drop one non-aggregated select column when multiple exist."""
    _ = schema, column_metadata
    scs = list(intent.select_cols or [])
    non_agg_idx = [
        i
        for i, sc in enumerate(scs)
        if not sc.is_aggregated and not (expr_registry_ref(sc.expr) or "").startswith(("w", "c"))
    ]
    if len(non_agg_idx) < 2:
        return []
    results: list[SeedWarmupIntent] = []
    for idx in non_agg_idx:
        new_intent = copy.deepcopy(intent)
        new_intent.select_cols = [c for j, c in enumerate(scs) if j != idx]
        _add_expansion_metadata(new_intent, ExpansionOperatorId.SELECT_COL_TRIM)
        results.append(new_intent)
    return results


def _window_strip(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Remove select columns that only carry a window specification."""
    _ = schema, column_metadata
    scs = list(intent.select_cols or [])
    reg_by = {s.registry_id: s for s in (intent.window_registry or [])}

    def _pure_rank_registry_col(sc: SelectCol) -> bool:
        rid = expr_registry_ref(sc.expr) or ""
        if not rid.startswith("w"):
            return False
        step = reg_by.get(rid)
        if step is None:
            return False
        ws = step.window_spec
        if ws.function not in ("row_number", "rank", "dense_rank"):
            return False
        arg = ws.argument
        if arg is None:
            return True
        return not arg.signature_key

    if not any(_pure_rank_registry_col(sc) for sc in scs):
        return []
    new_intent = copy.deepcopy(intent)
    dropped_ids: set[str] = set()
    kept: list[SelectCol] = []
    for sc in scs:
        if _pure_rank_registry_col(sc):
            rid = expr_registry_ref(sc.expr) or ""
            if rid:
                dropped_ids.add(rid)
            continue
        kept.append(sc)
    new_intent.select_cols = kept
    new_intent.window_registry = [s for s in (new_intent.window_registry or []) if s.registry_id not in dropped_ids]
    new_intent.order_by_cols = [
        o for o in (new_intent.order_by_cols or []) if (expr_registry_ref(o.expr) or "") not in dropped_ids
    ]
    if len(new_intent.select_cols) == len(scs):
        return []
    _add_expansion_metadata(new_intent, ExpansionOperatorId.WINDOW_STRIP)
    fin = _finalize_registry_touch_seed(new_intent, schema)
    return [fin] if fin is not None else []


def _distinct_remove(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """Turn off ``distinct_select_index`` when it is set on the intent."""
    _ = schema, column_metadata
    if intent.distinct_select_index < 0:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.distinct_select_index = -1
    _add_expansion_metadata(new_intent, ExpansionOperatorId.DISTINCT_REMOVE)
    return [new_intent]


def _splice_subtree(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
    fk_map: dict[str, list[dict[str, str]]],
) -> list[SeedWarmupIntent]:
    """Borrow one filter predicate from the subtree pool when table. overlap exists."""
    _ = schema, fk_map, column_metadata
    if not _EXPANSION_SUBTREE_POOL:
        return []
    idx = hash(intent.intent_id or "") % len(_EXPANSION_SUBTREE_POOL)
    donor = _EXPANSION_SUBTREE_POOL[idx]
    fps = donor.filters_param or []
    if not fps:
        return []
    overlap = set(intent.tables or []) & set(donor.tables or [])
    if not overlap:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.filters_param = list(new_intent.filters_param or []) + [copy.deepcopy(fps[0])]
    _add_expansion_metadata(new_intent, ExpansionOperatorId.SPLICE_SUBTREE)
    return [new_intent]


def _filter_null_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_NULL_ADD: add IS NULL filters on nullable columns."""
    _ = schema
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_filterable_columns(schema, table):
            if col in current_filter_cols:
                continue
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            if not meta.get("nullable"):
                continue
            new_intent = copy.deepcopy(intent)
            new_intent.filters_param = list(new_intent.filters_param or []) + [
                FilterParam(
                    left_expr=NormalizedExpr.from_column(col),
                    op="is null",
                    value_type="null",
                    param_key=f"null_{col.replace('.', '_')}",
                ),
            ]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_NULL_ADD)
            results.append(new_intent)
    return results


def _filter_not_null_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_NOT_NULL_ADD: add IS NOT NULL filters on nullable columns."""
    _ = schema
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_filterable_columns(schema, table):
            if col in current_filter_cols:
                continue
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            if not meta.get("nullable"):
                continue
            new_intent = copy.deepcopy(intent)
            new_intent.filters_param = list(new_intent.filters_param or []) + [
                FilterParam(
                    left_expr=NormalizedExpr.from_column(col),
                    op="is not null",
                    value_type="null",
                    param_key=f"notnull_{col.replace('.', '_')}",
                ),
            ]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_NOT_NULL_ADD)
            results.append(new_intent)
    return results


def _filter_in_list_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_IN_LIST_ADD: add IN-list filters using profiled categorical samples."""
    _ = schema
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_filterable_columns(schema, table):
            if col in current_filter_cols:
                continue
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            role = (meta.get("role") or "").strip().lower()
            if role not in (ColumnRole.CATEGORICAL.value.lower(), ColumnRole.IDENTIFIER.value.lower()):
                vt = (meta.get("value_type") or "").strip().lower()
                if vt not in ("string", "categorical"):
                    continue
            samples = meta.get("sample_values") or []
            if len(samples) < 2:
                continue
            pick = [str(v) for v in samples[:3]]
            new_intent = copy.deepcopy(intent)
            new_intent.filters_param = list(new_intent.filters_param or []) + [
                FilterParam(
                    left_expr=NormalizedExpr.from_column(col),
                    op="in",
                    value_type="string_list",
                    param_key=f"in_{col.replace('.', '_')}",
                    raw_value=cast(RawValue, pick),
                ),
            ]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_IN_LIST_ADD)
            results.append(new_intent)
    return results


def _filter_like_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """FILTER_LIKE_ADD: add LIKE pattern filters on categorical string columns."""
    _ = schema
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_filterable_columns(schema, table):
            if col in current_filter_cols:
                continue
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            samples = meta.get("sample_values") or []
            if not samples:
                continue
            sample = str(samples[0])
            if len(sample) < 3:
                continue
            pattern = f"%{sample[:3]}%"
            new_intent = copy.deepcopy(intent)
            new_intent.filters_param = list(new_intent.filters_param or []) + [
                FilterParam(
                    left_expr=NormalizedExpr.from_column(col),
                    op="like",
                    value_type="string",
                    param_key=f"like_{col.replace('.', '_')}",
                    raw_value=pattern,
                ),
            ]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.FILTER_LIKE_ADD)
            results.append(new_intent)
    return results


def _having_match_select_agg(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """HAVING_MATCH_SELECT_AGG: add HAVING thresholds on aggregates already present in SELECT."""
    _ = schema, column_metadata
    if intent.grain != "grouped" or not intent.group_by_cols:
        return []
    agg_cols = [sc for sc in (intent.select_cols or []) if sc.is_aggregated]
    if not agg_cols:
        return []
    existing = {(h.left_expr.primary_term, h.op) for h in (intent.having_param or [])}
    results: list[SeedWarmupIntent] = []
    for sc in agg_cols:
        left_term = sc.expr.primary_term
        if not left_term:
            continue
        for op in (">", "<", ">=", "<="):
            if (left_term, op) in existing:
                continue
            new_intent = copy.deepcopy(intent)
            new_having = HavingParam(
                left_expr=copy.deepcopy(sc.expr),
                op=op,
                value_type="number",
                param_key=f"hmatch_{op.replace('<', 'lt').replace('>', 'gt').replace('=', 'e')}",
            )
            new_intent.having_param = list(new_intent.having_param or []) + [new_having]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.HAVING_MATCH_SELECT_AGG)
            results.append(new_intent)
    return results


def _count_distinct_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """COUNT_DISTINCT_ADD: add count(distinct identifier) select columns."""
    _ = column_metadata
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_categorical_identifier_columns(schema, table):
            new_intent = copy.deepcopy(intent)
            expr = NormalizedExpr(
                add_groups=[
                    MulGroup(
                        multiply=[NormalizedExpr.from_column(col)],
                        agg_func="count",
                        distinct=True,
                    ),
                ],
            )
            new_intent.select_cols = list(new_intent.select_cols or []) + [SelectCol(expr=expr)]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.COUNT_DISTINCT_ADD)
            results.append(new_intent)
    return results


def _case_categorical_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """CASE_CATEGORICAL_ADD: append string-labeled CASE branches from profiled categorical values."""
    _ = column_metadata
    if intent.grain == "grouped":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_categorical_identifier_columns(schema, table):
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            samples = meta.get("sample_values") or []
            if len(samples) < 1:
                continue
            label_val = str(samples[0])
            new_intent = copy.deepcopy(intent)
            cond = FilterParam(
                left_expr=NormalizedExpr.from_column(col),
                op="=",
                value_type="string",
                param_key=f"case_cat_{col.replace('.', '_')}",
                raw_value=label_val,
            )
            branch = CaseWhenBranch(
                condition=cond,
                result=NormalizedExpr(raw_sql=f"'label_{label_val[:12]}'"),
            )
            cw = CaseWhenExpr(
                branches=[branch],
                else_result=NormalizedExpr(raw_sql="'other'"),
            )
            _append_case_registry_column(new_intent, cw)
            _add_expansion_metadata(new_intent, ExpansionOperatorId.CASE_CATEGORICAL_ADD)
            fin = _finalize_registry_touch_seed(new_intent, schema)
            if fin is not None:
                results.append(fin)
    return results


def _cte_wrap_grouped(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """CTE_WRAP_GROUPED: wrap the grouped body in ``cte1`` and query it from the outer scope."""
    _ = schema, column_metadata
    if intent.grain != "grouped" or not intent.group_by_cols:
        return []
    if intent.cte_steps:
        return []
    if len(intent.tables or []) > SeedWarmupConfig.MAX_TABLES:
        return []
    if not any(sc.is_aggregated for sc in (intent.select_cols or [])):
        return []
    new_intent = copy.deepcopy(intent)
    cte_step = RuntimeCteStep(
        cte_name="cte1",
        tables=list(intent.tables or []),
        select_cols=copy.deepcopy(list(intent.select_cols or [])),
        group_by_cols=copy.deepcopy(list(intent.group_by_cols or [])),
        order_by_cols=copy.deepcopy(list(intent.order_by_cols or [])),
        filters_param=copy.deepcopy(list(intent.filters_param or [])),
        having_param=copy.deepcopy(list(intent.having_param or [])),
        grain="grouped",
        emission="join_table",
    )
    new_intent.cte_steps = [cte_step]
    new_intent.tables = ["cte1"]
    new_intent.filters_param = []
    new_intent.having_param = []
    _add_expansion_metadata(new_intent, ExpansionOperatorId.CTE_WRAP_GROUPED)
    return [new_intent]


def _cte_scalar_threshold(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """CTE_SCALAR_THRESHOLD: scalar params CTE with aggregate threshold and a filter referencing it."""
    _ = column_metadata
    if intent.cte_steps:
        return []
    if intent.grain == "scalar":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        if table not in schema.tables:
            continue
        for measure in _get_numeric_measure_columns(schema, table):
            cte_name = "params"
            agg_expr = NormalizedExpr(
                add_groups=[
                    MulGroup(
                        multiply=[NormalizedExpr.from_column(measure)],
                        agg_func="avg",
                    ),
                ],
            )
            cte_step = RuntimeCteStep(
                cte_name=cte_name,
                tables=[table],
                select_cols=[SelectCol(expr=agg_expr)],
                grain="scalar",
                emission="scalar_subquery",
                output_columns=["threshold"],
            )
            new_intent = copy.deepcopy(intent)
            new_intent.cte_steps = [cte_step]
            new_intent.filters_param = list(new_intent.filters_param or []) + [
                FilterParam(
                    left_expr=NormalizedExpr.from_column(measure),
                    op=">",
                    right_expr=NormalizedExpr.from_column(f"{cte_name}.threshold"),
                    value_type="expression",
                    param_key="",
                ),
            ]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.CTE_SCALAR_THRESHOLD)
            results.append(new_intent)
    return results


def _window_rank_variant_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    function: str,
    operator_id: str,
) -> list[SeedWarmupIntent]:
    """Shared grouped-window rank/dense_rank/rank helper ordered by a select aggregate."""
    if _intent_has_window_select(intent):
        return []
    if intent.grain != "grouped":
        return []
    gbc = intent.group_by_cols or []
    if not gbc:
        return []
    agg_sc = next((sc for sc in (intent.select_cols or []) if sc.is_aggregated), None)
    if agg_sc is None:
        return []
    new_intent = copy.deepcopy(intent)
    ws = WindowSpec(
        function=function,
        partition_by=[copy.deepcopy(g) for g in gbc],
        order_by=[OrderByCol(expr=copy.deepcopy(agg_sc.expr), direction="DESC")],
    )
    _append_window_registry_column(new_intent, ws)
    _add_expansion_metadata(new_intent, operator_id)
    fin = _finalize_registry_touch_seed(new_intent, schema)
    return [fin] if fin is not None else []


def _window_dense_rank_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """WINDOW_DENSE_RANK_ADD: add ``dense_rank`` over group keys ordered by a select aggregate."""
    _ = column_metadata
    return _window_rank_variant_add(intent, schema, "dense_rank", ExpansionOperatorId.WINDOW_DENSE_RANK_ADD)


def _window_rank_func_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """WINDOW_RANK_FUNC_ADD: add ``rank`` over group keys ordered by a select aggregate."""
    _ = column_metadata
    return _window_rank_variant_add(intent, schema, "rank", ExpansionOperatorId.WINDOW_RANK_FUNC_ADD)


def _window_avg_partition_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """WINDOW_AVG_PARTITION_ADD: add ``avg(measure) over (partition by dim)`` for row-level grain."""
    _ = column_metadata
    if _intent_has_window_select(intent):
        return []
    if intent.grain != "row_level":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for measure in _get_numeric_measure_columns(schema, table):
            for dim in _get_categorical_identifier_columns(schema, table):
                if measure == dim:
                    continue
                new_intent = copy.deepcopy(intent)
                ws = WindowSpec(
                    function="avg",
                    partition_by=[NormalizedExpr.from_column(dim)],
                    order_by=[],
                    argument=NormalizedExpr.from_column(measure),
                )
                _append_window_registry_column(new_intent, ws)
                _add_expansion_metadata(new_intent, ExpansionOperatorId.WINDOW_AVG_PARTITION_ADD)
                fin = _finalize_registry_touch_seed(new_intent, schema)
                if fin is not None:
                    results.append(fin)
    return results


def _orderby_window_columndd(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """ORDERBY_WINDOW_COL_ADD: order by the first window registry column when present."""
    _ = schema, column_metadata
    registry = intent.window_registry or []
    if not registry:
        return []
    wid = registry[0].registry_id
    existing = {o.expr.primary_term for o in (intent.order_by_cols or [])}
    if wid in existing:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.order_by_cols = list(new_intent.order_by_cols or []) + [
        OrderByCol(expr=NormalizedExpr.from_column(wid), direction="DESC"),
    ]
    _add_expansion_metadata(new_intent, ExpansionOperatorId.ORDERBY_WINDOW_COL_ADD)
    return [new_intent]


def _select_coalesce_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """SELECT_COALESCE_ADD: coalesce nullable numeric measures with zero in the select list."""
    if intent.grain == "grouped":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_numeric_measure_columns(schema, table):
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            if not meta.get("nullable"):
                continue
            new_intent = copy.deepcopy(intent)
            coalesce_expr = NormalizedExpr(
                add_groups=[MulGroup(multiply=[NormalizedExpr.from_column(col)])],
                scalar_func="coalesce",
                scalar_func_args=[0.0],
                sarg_param_keys=[f"coalesce_zero_{col.replace('.', '_')}"],
            )
            new_intent.select_cols = list(new_intent.select_cols or []) + [SelectCol(expr=coalesce_expr)]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.SELECT_COALESCE_ADD)
            results.append(new_intent)
    return results


def _select_string_scalar_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """SELECT_STRING_SCALAR_ADD: wrap categorical display columns with ``upper`` in select."""
    _ = column_metadata
    if intent.grain == "grouped":
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_categorical_identifier_columns(schema, table):
            new_intent = copy.deepcopy(intent)
            upper_expr = replace(
                NormalizedExpr.from_column(col),
                scalar_func="upper",
            )
            new_intent.select_cols = list(new_intent.select_cols or []) + [SelectCol(expr=upper_expr)]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.SELECT_STRING_SCALAR_ADD)
            results.append(new_intent)
    return results


def _temp_extract_filter(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """TEMP_EXTRACT_FILTER: filter on ``extract(year)`` of a temporal column."""
    _ = column_metadata
    current_filter_cols = {f.left_expr.primary_column for f in (intent.filters_param or [])}
    if len(current_filter_cols) >= SeedWarmupConfig.MAX_FILTERS:
        return []
    results: list[SeedWarmupIntent] = []
    for table in intent.tables or []:
        for col in _get_temporal_columns(schema, table):
            bare = col.split(".", 1)[1] if "." in col else col
            meta = column_metadata.get(table, {}).get(bare, {})
            samples = meta.get("sample_values") or []
            year_val = "2020"
            if samples:
                sample = str(samples[0])
                if len(sample) >= 4 and sample[:4].isdigit():
                    year_val = sample[:4]
            extract_expr = replace(
                NormalizedExpr.from_column(col),
                scalar_func="extract",
                scalar_func_args=["year"],
            )
            new_intent = copy.deepcopy(intent)
            new_intent.filters_param = list(new_intent.filters_param or []) + [
                FilterParam(
                    left_expr=extract_expr,
                    op="=",
                    value_type="number",
                    param_key=f"extract_yr_{col.replace('.', '_')}",
                    raw_value=year_val,
                ),
            ]
            _add_expansion_metadata(new_intent, ExpansionOperatorId.TEMP_EXTRACT_FILTER)
            results.append(new_intent)
    return results


def _multi_cte_chain_add(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[SeedWarmupIntent]:
    """MULTI_CTE_CHAIN_ADD: append a second CTE that reads from an existing wrapped grouped CTE."""
    _ = schema, column_metadata
    if len(intent.cte_steps or []) != 1:
        return []
    first = intent.cte_steps[0]
    if first.cte_name != "cte1" or first.grain != "grouped":
        return []
    if intent.tables != ["cte1"]:
        return []
    new_intent = copy.deepcopy(intent)
    second = RuntimeCteStep(
        cte_name="cte2",
        tables=["cte1"],
        select_cols=copy.deepcopy(list(first.select_cols or [])),
        group_by_cols=copy.deepcopy(list(first.group_by_cols or [])),
        order_by_cols=copy.deepcopy(list(first.order_by_cols or [])),
        filters_param=[],
        having_param=[],
        grain="grouped",
        emission="join_table",
    )
    new_intent.cte_steps = list(new_intent.cte_steps or []) + [second]
    new_intent.tables = ["cte2"]
    new_intent.filters_param = []
    new_intent.having_param = []
    _add_expansion_metadata(new_intent, ExpansionOperatorId.MULTI_CTE_CHAIN_ADD)
    return [new_intent]


def _emi_equivalence_augment(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]],
    fk_map: dict[str, list[dict[str, str]]],
) -> list[SeedWarmupIntent]:
    """Duplicate an existing AND filter to preserve row sets under. conjunctive semantics."""
    _ = schema, fk_map, column_metadata
    fps = intent.filters_param or []
    if not fps:
        return []
    new_intent = copy.deepcopy(intent)
    new_intent.filters_param = list(new_intent.filters_param or []) + [copy.deepcopy(fps[0])]
    _add_expansion_metadata(new_intent, ExpansionOperatorId.EMI_MUTATE)
    return [new_intent]


def _expansion_noop(
    intent: SeedWarmupIntent,
    schema: SchemaGraph,
    column_metadata: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[SeedWarmupIntent]:
    """Placeholder expansion operator reserved for a future deterministic transform."""
    _ = intent, schema, column_metadata
    return []


def _build_operator_registry(
    column_metadata: dict[str, dict[str, dict[str, Any]]],
    fk_map: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Build the registry mapping operator ids to callables."""
    return {
        ExpansionOperatorId.FILTER_ADD: lambda i, s: _filter_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_EXPR_ADD: lambda i, s: _filter_expr_add(i, s, column_metadata),
        ExpansionOperatorId.AGG_CHANGE: lambda i, s: _agg_change(i, s, column_metadata),
        ExpansionOperatorId.GROUPBY_ADD: lambda i, s: _groupby_add(i, s, column_metadata),
        ExpansionOperatorId.ORDERBY_ADD: lambda i, s: _orderby_add(i, s, column_metadata),
        ExpansionOperatorId.HAVING_VALUE_ADD: lambda i, s: _having_value_add(i, s, column_metadata),
        ExpansionOperatorId.HAVING_EXPR_ADD: lambda i, s: _having_expr_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_REMOVE: lambda i, s: _filter_remove(i, s, column_metadata),
        ExpansionOperatorId.GROUPBY_REMOVE: lambda i, s: _groupby_remove(i, s, column_metadata),
        ExpansionOperatorId.HAVING_REMOVE: lambda i, s: _having_remove(i, s, column_metadata),
        ExpansionOperatorId.JOIN_DIMENSION_ADD: lambda i, s: _join_dimension_add(i, s, fk_map, column_metadata),
        ExpansionOperatorId.JOIN_FACT_ADD: lambda i, s: _join_fact_add(i, s, fk_map, column_metadata),
        ExpansionOperatorId.DIMENSION_SWAP: lambda i, s: _dimension_swap(i, s, fk_map, column_metadata),
        ExpansionOperatorId.TABLE_REMOVE: lambda i, s: _table_remove(i, s, fk_map, column_metadata),
        ExpansionOperatorId.BRIDGE_INTERMEDIATE_ADD: lambda i, s: _bridge_intermediate_add(
            i, s, fk_map, column_metadata
        ),
        ExpansionOperatorId.INCLUDE_GOLD: lambda i, s: _include_gold(i, s, column_metadata),
        ExpansionOperatorId.TEMP_EXTRACT_GROUPBY: lambda i, s: _temp_extract_groupby(i, s, column_metadata),
        ExpansionOperatorId.TEMP_DATE_TRUNC_GROUPBY: lambda i, s: _temp_date_trunc_groupby(i, s, column_metadata),
        ExpansionOperatorId.TEMP_DATE_WINDOW_FILTER: lambda i, s: _temp_date_window_filter(i, s, column_metadata),
        ExpansionOperatorId.TEMP_DATE_DIFF_FILTER: lambda i, s: _temp_date_diff_filter(i, s, column_metadata),
        ExpansionOperatorId.NUM_ROUND_SELECT: lambda i, s: _num_round_select(i, s, column_metadata),
        ExpansionOperatorId.NUM_ABS_FILTER: lambda i, s: _num_abs_filter(i, s, column_metadata),
        ExpansionOperatorId.DISTINCT_ADD: lambda i, s: _distinct_add(i, s, column_metadata),
        ExpansionOperatorId.LIMIT_ADD: lambda i, s: _limit_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_OR_GROUP: lambda i, s: _filter_or_group(i, s, column_metadata),
        ExpansionOperatorId.SELECT_EXPR_PAIR_MULTIPLY: lambda i, s: _select_expr_pair_multiply(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_RANK_ADD: lambda i, s: _window_rank_add(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_SUM_PARTITION_ADD: lambda i, s: _window_sum_partition_add(i, s, column_metadata),
        ExpansionOperatorId.SELECT_CASE_LABEL_ADD: lambda i, s: _select_case_label_add(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_LAG_ADD: lambda i, s: _window_lag_add(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_LEAD_ADD: lambda i, s: _window_lead_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_ILIKE_ADD: lambda i, s: _filter_ilike_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_ARRAY_CONTAINS_ADD: lambda i, s: _filter_array_contains_add(i, s, column_metadata),
        ExpansionOperatorId.ORDERBY_REMOVE: lambda i, s: _orderby_remove(i, s, column_metadata),
        ExpansionOperatorId.LIMIT_REMOVE: lambda i, s: _limit_remove(i, s, column_metadata),
        ExpansionOperatorId.SELECT_COL_TRIM: lambda i, s: _select_col_trim(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_STRIP: lambda i, s: _window_strip(i, s, column_metadata),
        ExpansionOperatorId.DISTINCT_REMOVE: lambda i, s: _distinct_remove(i, s, column_metadata),
        ExpansionOperatorId.SPLICE_SUBTREE: lambda i, s: _splice_subtree(i, s, column_metadata, fk_map),
        ExpansionOperatorId.EMI_MUTATE: lambda i, s: _emi_equivalence_augment(i, s, column_metadata, fk_map),
        ExpansionOperatorId.FILTER_NULL_ADD: lambda i, s: _filter_null_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_NOT_NULL_ADD: lambda i, s: _filter_not_null_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_IN_LIST_ADD: lambda i, s: _filter_in_list_add(i, s, column_metadata),
        ExpansionOperatorId.FILTER_LIKE_ADD: lambda i, s: _filter_like_add(i, s, column_metadata),
        ExpansionOperatorId.HAVING_MATCH_SELECT_AGG: lambda i, s: _having_match_select_agg(i, s, column_metadata),
        ExpansionOperatorId.COUNT_DISTINCT_ADD: lambda i, s: _count_distinct_add(i, s, column_metadata),
        ExpansionOperatorId.CASE_CATEGORICAL_ADD: lambda i, s: _case_categorical_add(i, s, column_metadata),
        ExpansionOperatorId.CTE_WRAP_GROUPED: lambda i, s: _cte_wrap_grouped(i, s, column_metadata),
        ExpansionOperatorId.CTE_SCALAR_THRESHOLD: lambda i, s: _cte_scalar_threshold(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_DENSE_RANK_ADD: lambda i, s: _window_dense_rank_add(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_RANK_FUNC_ADD: lambda i, s: _window_rank_func_add(i, s, column_metadata),
        ExpansionOperatorId.WINDOW_AVG_PARTITION_ADD: lambda i, s: _window_avg_partition_add(i, s, column_metadata),
        ExpansionOperatorId.ORDERBY_WINDOW_COL_ADD: lambda i, s: _orderby_window_columndd(i, s, column_metadata),
        ExpansionOperatorId.SELECT_COALESCE_ADD: lambda i, s: _select_coalesce_add(i, s, column_metadata),
        ExpansionOperatorId.SELECT_STRING_SCALAR_ADD: lambda i, s: _select_string_scalar_add(i, s, column_metadata),
        ExpansionOperatorId.TEMP_EXTRACT_FILTER: lambda i, s: _temp_extract_filter(i, s, column_metadata),
        ExpansionOperatorId.CTE_UNNEST_ADD: lambda i, s: _expansion_noop(i, s, column_metadata),
        ExpansionOperatorId.SELF_JOIN_CTE_ADD: lambda i, s: _expansion_noop(i, s, column_metadata),
        ExpansionOperatorId.MULTI_CTE_CHAIN_ADD: lambda i, s: _multi_cte_chain_add(i, s, column_metadata),
        ExpansionOperatorId.SPLICE_HAVING_SUBTREE: lambda i, s: _expansion_noop(i, s, column_metadata),
        ExpansionOperatorId.SPLICE_WINDOW_SUBTREE: lambda i, s: _expansion_noop(i, s, column_metadata),
    }


def _deterministic_repair_warmup_seed(
    seed: SeedWarmupIntent,
    schema: SchemaGraph,
) -> SeedWarmupIntent:
    """Run the same deterministic repair chain as interactive parsing. on. a seed intent."""
    rt: RuntimeIntent = seed.to_runtime_intent()
    nl = seed.natural_language or seed.intent_id or ""
    repaired = apply_deterministic_repairs(rt, schema, nl)
    return replace(
        seed,
        case_registry=list(repaired.case_registry),
        cte_steps=repaired.cte_steps,
        distinct_select_index=repaired.distinct_select_index,
        filters_param=repaired.filters_param,
        grain=repaired.grain,
        group_by_cols=repaired.group_by_cols,
        having_param=repaired.having_param,
        limit=repaired.limit,
        natural_language=repaired.natural_language or seed.natural_language,
        order_by_cols=repaired.order_by_cols,
        param_values=repaired.param_values or seed.param_values,
        select_cols=repaired.select_cols,
        tables=repaired.tables,
        window_registry=list(repaired.window_registry),
    )


def _expand_single_depth(
    intents: list[SeedWarmupIntent],
    schema: SchemaGraph,
    operators: dict[str, Any],
    seen_keys: set[str],
    source_tag: str,
) -> list[SeedWarmupIntent]:
    """Run every registered operator on each intent and collect unique. accepted variants."""
    results: list[SeedWarmupIntent] = []
    for intent in intents:
        for op_name, op_func in operators.items():
            if not expansion_compatible(intent, op_name):
                continue
            variants = op_func(intent, schema)
            for var in variants:
                accepted = _accept_expansion_variant(
                    var,
                    schema,
                    parent=intent,
                    candidate_op=op_name,
                )
                if accepted is None:
                    continue
                var = accepted
                var_key = intent_key(var.to_runtime_intent())
                if var_key in seen_keys:
                    continue
                seen_keys.add(var_key)
                if var.expansion_metadata and var.expansion_metadata.operator == ExpansionOperatorId.INCLUDE_GOLD:
                    var.source = intent.source
                else:
                    var.source = source_tag
                results.append(var)
    return results


def expand_gold_intents(
    gold_intents: list[SeedWarmupIntent],
    schema: SchemaGraph,
    limits: SchemaLimits | None = None,
    max_depth: int | None = None,
) -> list[SeedWarmupIntent]:
    """Expand gold intents into synthetic intents via multi-depth. deterministic expansion."""
    if limits is not None:
        SeedWarmupConfig.MAX_FILTERS = limits.max_filters
        SeedWarmupConfig.MAX_GROUPBY = limits.max_groupby
        SeedWarmupConfig.MAX_TABLES = limits.max_tables
        debug(
            f"expand_gold_intents: using SchemaLimits "
            f"max_filters={limits.max_filters}, "
            f"max_groupby={limits.max_groupby}, "
            f"max_tables={limits.max_tables}"
        )

    if max_depth is None:
        max_depth = SeedWarmupConfig.MAX_EXPANSION_DEPTH

    debug(f"expand_gold_intents: expanding {len(gold_intents)} gold intents with max_depth={max_depth}")

    column_metadata = _build_column_metadata(schema)
    fk_map = _build_fk_map(schema)
    operators = _build_operator_registry(column_metadata, fk_map)

    seen_keys: set[str] = set()
    for gold in gold_intents:
        seen_keys.add(intent_key(gold.to_runtime_intent()))

    tier_counts: defaultdict[str, int] = defaultdict(int)
    for gold in gold_intents:
        tier_counts[classify_seed_warmup_intent_complexity(gold).value] += 1

    current_layer = list(gold_intents)
    all_synthetic: list[SeedWarmupIntent] = []

    for depth in range(1, max_depth + 1):
        layer_tag = "depth1" if depth == 1 else "depth2"
        denom = len(seen_keys)
        cc = dict(tier_counts)
        current_layer.sort(key=lambda it: _tier_expansion_sort_key(it, cc, denom))
        new_variants = _expand_single_depth(
            current_layer,
            schema,
            operators,
            seen_keys,
            layer_tag,
        )
        debug(f"expand_gold_intents: depth={depth} produced {len(new_variants)} new variants")
        if not new_variants:
            break
        for var in new_variants:
            tier_counts[classify_seed_warmup_intent_complexity(var).value] += 1
            _record_expansion_subtree_pool(var)
        all_synthetic.extend(new_variants)
        current_layer = new_variants

    debug(f"expand_gold_intents: generated {len(all_synthetic)} unique synthetic intents across {max_depth} depth(s)")
    return all_synthetic
