"""Deterministic SQL building, FK join enumeration, repair prompts, and canonical join/predicate normalisation. Each registered engine uses its own AST path via the :class:`~aetherdialect._dialect.Dialect` adapter (pglast for PostgreSQL; sqlglot for all other engines). The adapter exposes ``parse_select``, ``ordered_join_carrier_froms``, ``attach_joins``, and ``emit_sql`` so this module never names a parser library directly."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Literal, cast

from sqlglot import exp, parse_one

from ._config import PolicyConfig
from ._constants import (
    CANONICAL_FEEDBACK_DIALECT,
    DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP,
    DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH,
    DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED,
    DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE,
    DETERMINISTIC_PROBE_EDGE_KINDS,
    DISTINCT_ON_CTE_NAME_PREFIX,
    DISTINCT_ON_RANK_COLUMN,
    JOIN_CHOICE_PROMPT_KEY_ORDER,
    JOIN_CHOICE_SCOPE_MAIN,
    JOIN_EDGE_KIND_RANK,
    JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR,
    JOIN_PATH_EDGE_KIND_WHERE_BUCKET,
    JOIN_PATH_EDGE_KINDS,
    JOIN_PRIOR_FEEDBACK_HEADING,
    SCALAR_FUNCTIONS_LEADING_ARG,
    SQL_WINDOW_FUNCTION_UPPER,
    anti_join_presence_column,
)
from ._contracts_base import (
    PROBE_CTE_EMISSION_KINDS,
    CteEmissionKind,
    HavingParam,
    JoinColumnCountMismatchError,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    JoinPathTieCapExceededError,
    ClauseWidenedRowsetError,
    LlmJsonExhausted,
    MulGroup,
    NoJoinPathError,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    ScalarArg,
    WhereParam,
    expr_registry_ref,
    merge_predicate_groups,
    predicate_group_from_list,
    register_render_expr_sql,
)
from ._contracts_core import RuntimeCteStep, RuntimeIntent, ScopeClass, SelectCol, effective_select_parts
from ._contracts_schema import (
    CaseWhenExpr,
    SchemaGraph,
    VirtualTableSpec,
    WindowSpec,
    current_case_registry_steps,
    current_window_registry_steps,
    registry_render_scope,
)
from ._core_utils import (
    debug,
    join_signature_tables,
    notify,
    pipeline_trace,
    prompt_cache_schema_scope,
    prompt_json,
    schema_prompt_cache_id,
    stable_json,
)
from ._dialect import Dialect, JoinEdge, get_dialect
from ._dialect_postgres import append_pglast_select_targets
from ._intent_expr import extract_columns_from_expr
from ._llm_provider import llm_json
from ._schema_catalog import value_overlap_stats_for_columns
from ._schema_graph import join_path_pair_tie_count, stored_join_paths_for_pair
from ._validation_schema import (
    collect_fan_out_sensitive_aggregates,
    get_col_meta,
    get_col_type,
    multiplying_edges_for_table,
    qualifies_as_semi_join_probe,
    validate_clause_widened_rowset,
)


class JoinCandidateCapExceededError(Exception):
    """Raised when join path cross-product enumeration exceeds the refusal cap."""

    def __init__(
        self,
        enumerated: int,
        cap: int,
        *,
        tables: list[str] | None = None,
        root: str | None = None,
    ) -> None:
        self.enumerated = enumerated
        self.cap = cap
        self.tables = list(tables) if tables is not None else None
        self.root = root
        tables_text = ",".join(self.tables) if self.tables else "?"
        root_text = f" root={self.root!r}" if self.root else ""
        super().__init__(
            f"join candidate cross-product cap exceeded: {enumerated} paths (limit {cap}) tables={tables_text}{root_text}"
        )


class JoinProbeEdgeKindMismatchError(Exception):
    """Raised when join path signature and edge-kind lists are not aligned."""

    def __init__(self, signature_len: int, kinds_len: int) -> None:
        self.signature_len = signature_len
        self.kinds_len = kinds_len
        super().__init__(f"join path edge_kinds length mismatch: {kinds_len} kinds for {signature_len} segments")


def _refuse_join_candidate_cap(
    enumerated: int,
    cap: int,
    *,
    tables: list[str] | None = None,
    root: str | None = None,
) -> None:
    """Emit a session diagnostic and refuse when enumeration exceeds the cross-product cap."""
    detail_parts: list[tuple[str, str]] = [("cap", str(cap)), ("enumerated", str(enumerated))]
    if tables:
        detail_parts.append(("tables", ",".join(tables)))
    if root:
        detail_parts.append(("root", root))
    notify(
        (f"Join path enumeration produced {enumerated} candidates exceeding the cap of {cap}; refusing to truncate."),
        stage="join",
        code=DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP,
        level="error",
        details=tuple(detail_parts),
    )
    raise JoinCandidateCapExceededError(enumerated, cap, tables=tables, root=root)


def _effective_join_path_tie_cap(tie_cap: int | None) -> int:
    return max(1, int(tie_cap if tie_cap is not None else PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP))


def _refuse_join_path_tie_ceiling(
    source_table: str,
    target_table: str,
    path_count: int,
    ceiling: int,
) -> None:
    """Emit a session diagnostic and refuse when equal-length paths exceed the tie ceiling."""
    notify(
        (
            f"Equal-length join paths between {source_table!r} and {target_table!r} exceed the tie "
            f"ceiling ({path_count} paths, limit {ceiling}); refusing."
        ),
        stage="join",
        code=DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED,
        level="error",
        details=(
            ("source", source_table),
            ("target", target_table),
            ("path_count", str(path_count)),
            ("ceiling", str(ceiling)),
        ),
    )
    raise JoinPathTieCapExceededError(source_table, target_table, path_count, ceiling)


def _enforce_join_path_pair_tie_ceiling(
    schema: SchemaGraph,
    source_table: str,
    target_table: str,
    *,
    tie_cap: int | None,
) -> None:
    ceiling = _effective_join_path_tie_cap(tie_cap)
    path_count = join_path_pair_tie_count(schema.join_paths_multi, source_table, target_table)
    if path_count > ceiling:
        _refuse_join_path_tie_ceiling(source_table, target_table, path_count, ceiling)


def fan_out_penalty_for_path_edges(
    edges: list[dict[str, Any]],
    intent: RuntimeIntent | None,
    schema: SchemaGraph,
    *,
    from_anchor: str | None = None,
) -> int:
    """Count how many fan-out-sensitive aggregates would be multiplied on *edges*."""
    if intent is None or not edges:
        return 0
    signature = []
    for edge in edges:
        src = str(edge.get("src_table", ""))
        dst = str(edge.get("dst_table", ""))
        sc = edge.get("src_cols") or []
        dc = edge.get("dst_cols") or []
        if src and dst and sc and dc:
            signature.append(f"{src}.{','.join(sc)}->{dst}.{','.join(dc)}")
    anchor = from_anchor or (intent.tables[0] if intent.tables else None)
    penalty = 0
    for _func, tbl, _col, _distinct in collect_fan_out_sensitive_aggregates(intent):
        if multiplying_edges_for_table(signature, tbl, schema, from_anchor=anchor):
            penalty += 1
    return penalty


_SQL_GEN_SCHEMA: ContextVar[SchemaGraph | None] = ContextVar("_SQL_GEN_SCHEMA", default=None)
_SQL_GEN_CTE_OUTPUTS: ContextVar[dict[str, Any] | None] = ContextVar("_SQL_GEN_CTE_OUTPUTS", default=None)


@contextmanager
def sql_gen_type_scope(schema: SchemaGraph | None, cte_outputs: dict[str, Any] | None = None) -> Iterator[None]:
    """Bind schema and CTE output metadata for operand type checks during expression render."""
    token_schema = _SQL_GEN_SCHEMA.set(schema)
    token_cte = _SQL_GEN_CTE_OUTPUTS.set(cte_outputs or {})
    try:
        yield
    finally:
        _SQL_GEN_SCHEMA.reset(token_schema)
        _SQL_GEN_CTE_OUTPUTS.reset(token_cte)


def _mulgroup_value_kind(g: MulGroup) -> str | None:
    """Return ``date``, ``integer``, ``number``, or ``None`` from column metadata when schema is bound."""
    schema = _SQL_GEN_SCHEMA.get()
    if schema is None:
        return None
    cte_outputs = _SQL_GEN_CTE_OUTPUTS.get() or {}
    kinds: set[str] = set()
    for term in g.multiply + g.divide:
        for col in extract_columns_from_expr(term):
            vt = get_col_type(col, schema, cte_outputs)
            if vt == "date":
                kinds.add("date")
            elif vt == "integer":
                kinds.add("integer")
            elif vt == "number":
                kinds.add("number")
            elif vt:
                kinds.add("other")
    if "date" in kinds and len(kinds) == 1:
        return "date"
    if kinds <= {"integer"}:
        return "integer"
    if kinds <= {"number", "integer"}:
        return "number"
    return None


def _literal_param_is_integer_day(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _operands_are_date_minus_date(base: MulGroup, offset: MulGroup) -> bool:
    return _mulgroup_value_kind(base) == "date" and _mulgroup_value_kind(offset) == "date"


def _operands_are_numeric_minus_numeric(base: MulGroup, offset: MulGroup) -> bool:
    base_kind = _mulgroup_value_kind(base)
    offset_kind = _mulgroup_value_kind(offset)
    return base_kind in ("number", "integer") and offset_kind in ("number", "integer")


def _operands_allow_date_integer_days(
    base: MulGroup, offset: MulGroup | None, *, offset_literal: Any | None = None
) -> bool:
    if _mulgroup_value_kind(base) != "date":
        return False
    if offset_literal is not None:
        return _literal_param_is_integer_day(offset_literal)
    if offset is not None:
        return _mulgroup_value_kind(offset) in ("integer", "number")
    return False


def databricks_unqualify_agg_arg_sql(sql: str) -> str:
    """For Spark/Databricks output, drop table qualifiers on the first. argument of ``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` within *sql*."""
    if not (sql and sql.strip()):
        return sql
    try:
        tree = parse_one(sql, dialect="databricks")
    except Exception:
        return sql
    if isinstance(tree, exp.Column) and tree.table:
        tree.set("table", None)
        return tree.sql(dialect="databricks")
    for cls in (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max):
        for node in tree.find_all(cls):
            first = getattr(node, "this", None)
            if isinstance(first, exp.Column) and first.table:
                first.set("table", None)
    return tree.sql(dialect="databricks")


def _maybe_databricks_unqualify_window_sql_frag(sql: str, dialect: Dialect | None) -> str:
    """Apply dialect window-aggregate normalization when supported."""
    if dialect is None:
        return sql
    return dialect.normalize_window_agg_sql_frag(sql)


def cte_to_intent_for_ranking(cte: RuntimeCteStep) -> RuntimeIntent:
    """Build a synthetic `RuntimeIntent` from `RuntimeCteStep` for CTE- scope join enumeration."""
    return RuntimeIntent(
        tables=cte.tables,
        grain=cte.grain,
        select_cols=cte.select_cols,
        group_by_cols=cte.group_by_cols,
        order_by_cols=cte.order_by_cols,
        where=cte.where,
        having=cte.having,
        param_values=cte.param_values,
        column_map=cte.column_map,
        limit=cte.limit,
        cte_steps=[],
        window_registry=list(cte.window_registry or []),
        case_registry=list(cte.case_registry or []),
    )


def join_candidate_spans_tables(candidate: dict[str, Any], scope_tables: list[str]) -> bool:
    """Return True when a candidate's join path touches every table in *scope_tables*."""
    sig = candidate.get("join_path_signature") or []
    if not isinstance(sig, list):
        return False
    covered = join_signature_tables([str(x) for x in sig])
    return set(scope_tables) <= covered


def _compose_hybrid_fk_semantic_paths(
    base_paths: list[list[dict[str, Any]]],
    sem_paths: list[list[dict[str, Any]]],
    target: frozenset[str],
    schema: SchemaGraph,
) -> list[list[dict[str, Any]]]:
    """Attach semantic bridge edge(s) onto FK-spanning backbones for partial connectivity."""
    if len(target) < 2 or not sem_paths:
        return []
    out: list[list[dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    bases = [p for p in base_paths if p] or [[]]
    for base in bases:
        extended = _extend_join_paths_with_bridges([base], sem_paths, target, schema)
        for path in extended:
            if not _path_has_semantic_edge(path):
                continue
            sig = _join_edge_sig_tuple(path, schema)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(path)
    return out


def join_candidate_map(join_hints: dict[str, Any]) -> dict[str, list[str]]:
    """Build map from candidate ID to join path signature."""
    out: dict[str, list[str]] = {}
    for c in join_hints.get("candidates", []):
        cid = c.get("candidate_id")
        sig = c.get("join_path_signature")
        if isinstance(cid, str) and isinstance(sig, list):
            out[cid] = [str(x) for x in sig]
    return out


def edge_kinds_for_join_candidate(join_hints: dict[str, Any], candidate_id: str) -> list[str]:
    """Return ``edge_kinds`` for *candidate_id* from a join-hints payload."""
    for cand in join_hints.get("candidates") or []:
        if cand.get("candidate_id") == candidate_id:
            return [str(kind) for kind in (cand.get("edge_kinds") or [])]
    return []


def _analyze_join_topology(sig: list[str]) -> tuple[str, str, list[str]]:
    """Analyze join signature to determine topology type, hub table, and. leaf tables."""
    if not sig:
        return ("none", "", [])
    table_counts: dict[str, int] = {}
    for item in sig:
        if "->" not in item:
            continue
        left, right = item.split("->", 1)
        left_table = left.split(".")[0].strip()
        right_table = right.split(".")[0].strip()
        table_counts[left_table] = table_counts.get(left_table, 0) + 1
        table_counts[right_table] = table_counts.get(right_table, 0) + 1
    if not table_counts:
        return ("none", "", [])
    leaves = sorted([t for t, c in table_counts.items() if c == 1])
    hubs = sorted([t for t, c in table_counts.items() if c > 1], key=lambda t: (-table_counts[t], t))
    if len(leaves) == 2 and len(hubs) == len(table_counts) - 2:
        left_roots: set[str] = set()
        right_roots: set[str] = set()
        for item in sig:
            if "->" not in item:
                continue
            left_part, right_part = item.split("->", 1)
            left_roots.add(left_part.split(".")[0].strip().lower())
            right_roots.add(right_part.split(".")[0].strip().lower())
        if len(hubs) == 1 and len(left_roots) == 1 and hubs[0].lower() in left_roots:
            return ("star", hubs[0], leaves)
        if len(hubs) == 1 and len(right_roots) == 1 and hubs[0].lower() in right_roots:
            return ("star", hubs[0], leaves)
        return ("linear", min(leaves), leaves)
    if len(hubs) == 1:
        return ("star", hubs[0], leaves)
    if hubs:
        return ("tree", hubs[0], leaves)
    return ("linear", min(table_counts.keys(), key=str.lower), sorted(table_counts.keys(), key=str.lower))


def _wrap_for_case_insensitive(expr: str, dialect: Dialect) -> str:
    """Wrap expression for case-insensitive string comparison."""
    return dialect.render_case_insensitive_wrap(expr)


def _phys_table_key(tbl: str) -> str:
    """Return the unqualified lowercase table name for join bookkeeping."""
    return tbl.split(".")[-1].strip().strip('"').strip("`").lower()


def _table_sql_token(tbl: str) -> str:
    """Unqualified table token suitable for SQL references (preserves casing of last segment)."""
    return tbl.split(".")[-1].strip().strip('"').strip("`")


def cte_emission_map(cte_steps: list[RuntimeCteStep] | None) -> dict[str, CteEmissionKind]:
    """Map CTE names to their declared emission kind."""
    out: dict[str, CteEmissionKind] = {}
    for cte in cte_steps or []:
        name = (cte.cte_name or "").strip()
        if name:
            out[name] = getattr(cte, "emission", "join_table") or "join_table"
    return out


def probe_cte_names(cte_steps: list[RuntimeCteStep] | None) -> frozenset[str]:
    """Return CTE names whose emission is ``semi_join`` or ``anti_join``."""
    names: set[str] = set()
    for cte in cte_steps or []:
        em = getattr(cte, "emission", "join_table") or "join_table"
        if em in PROBE_CTE_EMISSION_KINDS:
            cn = (cte.cte_name or "").strip()
            if cn:
                names.add(cn)
    return frozenset(names)


def _normalized_table_name_set(names: list[str] | None) -> set[str]:
    return {_phys_table_key(t) for t in (names or []) if t}


def _table_in_set(tbl: str, candidates: set[str]) -> bool:
    return _phys_table_key(tbl) in candidates


def _edge_fk_points_to_paired(
    join_tbl: str, paired_tbl: str, cols_on_join: list[str], schema: SchemaGraph | None
) -> tuple[bool, bool]:
    """Return whether *join_tbl* carries an FK to *paired_tbl* and whether it is nullable."""
    if schema is None:
        return False, False
    tmeta = schema.tables.get(join_tbl)
    if not tmeta:
        return False, False
    fk_any = False
    fk_nullable = False
    for jc in cols_on_join:
        cm = tmeta.columns.get(jc)
        if cm and cm.is_foreign_key and cm.fk_target and cm.fk_target[0] == paired_tbl:
            fk_any = True
            if cm.is_nullable:
                fk_nullable = True
    return fk_any, fk_nullable


def _edge_is_one_to_many_ambiguous(
    join_tbl: str, paired_tbl: str, cols_on_join: list[str], schema: SchemaGraph | None
) -> bool:
    """True when the left side is the parent of a one-to-many edge the schema cannot settle."""
    fk_any, fk_nullable = _edge_fk_points_to_paired(join_tbl, paired_tbl, cols_on_join, schema)
    return fk_any and not fk_nullable


def _profiled_orphan_rate_for_edge(
    left_tbl: str, join_tbl: str, cols_on_join: list[str], schema: SchemaGraph | None
) -> float | None:
    """Estimate the fraction of left-side rows with no child match on a one-to-many edge."""
    if schema is None:
        return None
    if not _edge_is_one_to_many_ambiguous(join_tbl, left_tbl, cols_on_join, schema):
        return None
    left_meta = schema.tables.get(left_tbl)
    if left_meta is None:
        return None
    left_rows = int(getattr(left_meta, "row_count", 0) or 0)
    if left_rows <= 0:
        return None
    referenced_parents = 0
    join_meta = schema.tables.get(join_tbl)
    if join_meta is not None:
        for jc in cols_on_join:
            cm = join_meta.columns.get(jc)
            if cm and cm.is_foreign_key and cm.fk_target and cm.fk_target[0] == left_tbl:
                referenced_parents = int(getattr(cm, "distinct_count", 0) or 0)
                break
    referenced_parents = min(referenced_parents, left_rows)
    return max(0.0, min(1.0, 1.0 - (referenced_parents / left_rows)))


def _join_kind_for_edge(
    join_tbl: str,
    paired_tbl: str,
    cols_on_join: list[str],
    schema: SchemaGraph | None,
    *,
    right_emission: CteEmissionKind | None = None,
    left_is_preserved: bool = False,
) -> str:
    """Return ``LEFT`` or ``INNER`` join modifier (leading space) for one edge."""
    if right_emission == "anti_join":
        return " LEFT"
    if right_emission == "semi_join":
        return " INNER"
    if left_is_preserved:
        return " LEFT"
    if schema is None:
        return " INNER"
    fk_any, fk_nullable = _edge_fk_points_to_paired(join_tbl, paired_tbl, cols_on_join, schema)
    if fk_any and fk_nullable:
        return " LEFT"
    return " INNER"


def _partition_path_join_vs_where(
    signature: list[str], edge_kinds: list[str]
) -> tuple[
    list[tuple[int, str, str, list[str], list[str]]],
    list[tuple[int, str, str, list[str], list[str]]],
]:
    """Split parsed path segments into JOIN-bucket and WHERE-bucket edges by ``edge_kinds``. Each returned tuple is ``(orig_index, left_tbl, right_tbl, left_cols, right_cols)`` parsed from the signature. WHERE-bucket edges use kinds in :data:`~aetherdialect._constants.JOIN_PATH_EDGE_KIND_WHERE_BUCKET`; every other declared kind routes to the JOIN bucket. An unknown or missing kind raises :class:`~aetherdialect._contracts_base.JoinInjectionFailedError` naming the kind and segment."""
    join_bucket: list[tuple[int, str, str, list[str], list[str]]] = []
    where_bucket: list[tuple[int, str, str, list[str], list[str]]] = []
    for idx, seg in enumerate(signature):
        seg = seg.strip()
        if "->" not in seg:
            continue
        left_part, right_part = seg.split("->", 1)
        if "." not in left_part or "." not in right_part:
            continue
        left_tbl, left_cols = left_part.strip().split(".", 1)
        right_tbl, right_cols = right_part.strip().split(".", 1)
        lcols = [c.strip() for c in left_cols.split(",")]
        rcols = [c.strip() for c in right_cols.split(",")]
        if not lcols or not rcols:
            continue
        kind = edge_kinds[idx] if idx < len(edge_kinds) else ""
        if not kind or kind not in JOIN_PATH_EDGE_KINDS:
            detail = "missing" if not kind else f"unknown join edge kind {kind!r}"
            raise JoinInjectionFailedError(
                f"{detail} for path segment {seg!r}",
                det_sql="",
                join_sigs_ordered=[list(signature)],
                edge_kinds_ordered=[list(edge_kinds)],
            )
        record = (idx, left_tbl, right_tbl, lcols, rcols)
        if kind in JOIN_PATH_EDGE_KIND_WHERE_BUCKET:
            where_bucket.append(record)
        else:
            join_bucket.append(record)
    return join_bucket, where_bucket


def _extra_from_tables_for_where_edges(
    where_edges: list[tuple[int, str, str, list[str], list[str]]], tables_already_in_from: set[str]
) -> list[str]:
    """Return deduplicated extra-FROM table names introduced only by WHERE-bucket edges. Tables already covered by the anchor or JOIN tree are skipped. Order is the first appearance across the where edges."""
    seen: set[str] = set(tables_already_in_from)
    out: list[str] = []
    for _idx, left_tbl, right_tbl, _lc, _rc in where_edges:
        for tbl in (left_tbl, right_tbl):
            key = _phys_table_key(tbl)
            if key in seen:
                continue
            seen.add(key)
            out.append(_table_sql_token(tbl))
    return out


def _pair_join_columns(
    left_cols: list[str],
    right_cols: list[str],
    *,
    segment: str,
) -> Iterator[tuple[str, str]]:
    if len(left_cols) != len(right_cols):
        raise JoinColumnCountMismatchError(segment, len(left_cols), len(right_cols))
    yield from zip(left_cols, right_cols, strict=True)


def _assert_join_segment_schema_refs(
    schema: SchemaGraph,
    left_tbl: str,
    right_tbl: str,
    lcols: list[str],
    rcols: list[str],
    segment: str,
    *,
    virtual_tables: frozenset[str] | None = None,
) -> None:
    """Refuse join-path segments whose physical table or column names are absent from *schema*."""
    virtual_keys = {_phys_table_key(t) for t in (virtual_tables or ())}
    for tbl, cols in ((left_tbl, lcols), (right_tbl, rcols)):
        if _phys_table_key(tbl) in virtual_keys:
            continue
        tmeta = schema.tables.get(tbl)
        if tmeta is None:
            raise JoinInjectionFailedError(
                f"join path segment {segment!r} references missing table {tbl!r}",
                det_sql="",
                join_sigs_ordered=[],
                edge_kinds_ordered=[],
            )
        for col in cols:
            if col not in tmeta.columns:
                raise JoinInjectionFailedError(
                    f"join path segment {segment!r} references missing column {tbl}.{col!r}",
                    det_sql="",
                    join_sigs_ordered=[],
                    edge_kinds_ordered=[],
                )


def _join_path_tables_from_signature(signature: list[str], from_table: str) -> list[str]:
    """Collect physical table names from a join signature plus the FROM anchor."""
    tables = set(join_signature_tables([str(x) for x in signature]))
    if from_table:
        tables.add(from_table.strip())
    return sorted(tables)


def _raise_unresolved_join_path(signature: list[str], from_table: str) -> None:
    raise NoJoinPathError(f"join path from {from_table!r}", _join_path_tables_from_signature(signature, from_table))


def _join_edges_from_signature(
    signature: list[str],
    edge_kinds: list[str],
    from_table: str,
    schema: SchemaGraph | None = None,
    cte_emissions: dict[str, CteEmissionKind] | None = None,
    *,
    preserve_tables: list[str] | None = None,
    probe_cte_names: frozenset[str] | None = None,
    dialect: Dialect | None = None,
) -> tuple[list[JoinEdge], list[JoinEdge], list[str], list[str]] | None:
    """Resolve a join-path signature into JOIN edges, WHERE-bucket edges, extra FROM tables, and anti-join predicates."""
    if not from_table or not signature:
        return None
    join_segments, where_segments = _partition_path_join_vs_where(signature, edge_kinds)
    if not join_segments and not where_segments:
        return None
    anchor = from_table.strip()
    anchor_key = _phys_table_key(anchor)
    if probe_cte_names and _table_in_set(anchor, set(probe_cte_names)):
        raise NoJoinPathError(f"probe CTE '{anchor}' cannot be the join anchor", [anchor])
    preserved_roots = _normalized_table_name_set(preserve_tables)
    preserved_frontier = set(preserved_roots)
    phys_instances: dict[str, list[str]] = defaultdict(list)
    phys_instances[anchor_key].append(_table_sql_token(anchor))
    join_edges: list[JoinEdge] = []
    anti_join_predicates: list[str] = []
    unused: list[tuple[int, str, str, list[str], list[str]]] = list(join_segments)
    if schema is not None:
        virtual_tables: set[str] = set(cte_emissions or {})
        if probe_cte_names:
            virtual_tables.update(probe_cte_names)
        virtual_frozen = frozenset(virtual_tables)
        for _idx, left_tbl, right_tbl, lcols, rcols in join_segments:
            segment_repr = f"{left_tbl}.{','.join(lcols)}->{right_tbl}.{','.join(rcols)}"
            _assert_join_segment_schema_refs(
                schema, left_tbl, right_tbl, lcols, rcols, segment_repr, virtual_tables=virtual_frozen
            )
        for _idx, left_tbl, right_tbl, lcols, rcols in where_segments:
            segment_repr = f"{left_tbl}.{','.join(lcols)}->{right_tbl}.{','.join(rcols)}"
            _assert_join_segment_schema_refs(
                schema, left_tbl, right_tbl, lcols, rcols, segment_repr, virtual_tables=virtual_frozen
            )
    while unused:
        frontier: list[
            tuple[
                int,
                int,
                str,
                str,
                tuple[tuple[str, str, str, str], ...],
                list[str],
                str,
            ]
        ] = []
        for u_idx, (sig_i, left_tbl, right_tbl, lcols, rcols) in enumerate(unused):
            segment_repr = f"{left_tbl}.{','.join(lcols)}->{right_tbl}.{','.join(rcols)}"
            lk = _phys_table_key(left_tbl)
            rk = _phys_table_key(right_tbl)
            if lk == rk:
                raise NoJoinPathError(
                    f"self-join in path segment '{left_tbl}.{','.join(lcols)}->{right_tbl}.{','.join(rcols)}'",
                    [left_tbl],
                )
            li = lk in phys_instances and bool(phys_instances[lk])
            ri = rk in phys_instances and bool(phys_instances[rk])
            if li and ri:
                continue
            if li:
                join_tbl, paired_tbl, cols_on_join = right_tbl, left_tbl, rcols
                existing = phys_instances[lk][-1]
                left_token = existing
            elif ri:
                join_tbl, paired_tbl, cols_on_join = left_tbl, right_tbl, lcols
                existing = phys_instances[rk][-1]
                left_token = existing
            else:
                continue
            if probe_cte_names and _table_in_set(paired_tbl, set(probe_cte_names)):
                raise NoJoinPathError(
                    f"probe CTE '{paired_tbl}' cannot appear on the left of a join edge", [paired_tbl]
                )
            join_k = _phys_table_key(join_tbl)
            if join_k in phys_instances:
                continue
            new_tok = _table_sql_token(join_tbl)
            if join_tbl == right_tbl:
                on_terms = tuple(
                    (existing, lc, new_tok, rc) for lc, rc in _pair_join_columns(lcols, rcols, segment=segment_repr)
                )
            else:
                on_terms = tuple(
                    (new_tok, lc, existing, rc) for lc, rc in _pair_join_columns(lcols, rcols, segment=segment_repr)
                )
            frontier.append((sig_i, u_idx, join_tbl, paired_tbl, on_terms, cols_on_join, left_token))
        if not frontier:
            _raise_unresolved_join_path(signature, anchor)
        sig_i, u_idx, join_tbl, paired_tbl, on_terms_chosen, cols_on_join, left_token = min(
            frontier, key=lambda t: (t[0], _phys_table_key(t[2]), _phys_table_key(t[3]))
        )
        unused.pop(u_idx)
        join_emission = (cte_emissions or {}).get(join_tbl) or (cte_emissions or {}).get(_table_sql_token(join_tbl))
        left_is_preserved = _table_in_set(left_token, preserved_frontier) or _table_in_set(
            paired_tbl, preserved_frontier
        )
        kind_modifier = (
            _join_kind_for_edge(
                join_tbl,
                paired_tbl,
                cols_on_join,
                schema,
                right_emission=join_emission,
                left_is_preserved=left_is_preserved,
            )
            .strip()
            .upper()
            or "INNER"
        )
        join_edges.append(
            JoinEdge(
                table=join_tbl,
                alias=None,
                kind="LEFT" if kind_modifier == "LEFT" else "INNER",
                on_terms=on_terms_chosen,
            )
        )
        if join_emission == "anti_join":
            tbl_tok = _table_sql_token(join_tbl)
            marker = anti_join_presence_column(tbl_tok)
            if dialect is not None:
                anti_join_predicates.append(f"{dialect.quote_table_column(tbl_tok, marker)} IS NULL")
            else:
                anti_join_predicates.append(f"{tbl_tok}.{marker} IS NULL")
        if left_is_preserved or kind_modifier == "LEFT":
            preserved_frontier.add(_phys_table_key(join_tbl))
            preserved_frontier.add(_phys_table_key(paired_tbl))
            preserved_frontier.add(_phys_table_key(left_token))
        phys_instances[_phys_table_key(join_tbl)].append(_table_sql_token(join_tbl))
    if len(join_edges) != len(join_segments):
        _raise_unresolved_join_path(signature, anchor)
    where_edges: list[JoinEdge] = []
    for _idx, left_tbl, right_tbl, lcols, rcols in where_segments:
        left_tok = _table_sql_token(left_tbl)
        right_tok = _table_sql_token(right_tbl)
        segment_repr = f"{left_tbl}.{','.join(lcols)}->{right_tbl}.{','.join(rcols)}"
        on_terms = tuple(
            (left_tok, lc, right_tok, rc) for lc, rc in _pair_join_columns(lcols, rcols, segment=segment_repr)
        )
        where_edges.append(JoinEdge(table=right_tok, alias=None, kind="INNER", on_terms=on_terms))
    tables_in_from: set[str] = set(phys_instances.keys())
    extra_from_tables = _extra_from_tables_for_where_edges(where_segments, tables_in_from)
    emit_semantic_profile_where_diagnostics(
        schema,
        where_segments=where_segments,
        edge_kinds=edge_kinds,
    )
    return join_edges, where_edges, extra_from_tables, anti_join_predicates


def _orient_join_sig_for_from(sig: list[str], from_table: str) -> list[str]:
    """Reorient join segments so that no target duplicates the FROM. table."""
    if not from_table:
        return sig
    oriented: list[str] = []
    for seg in sig:
        if "->" not in seg:
            oriented.append(seg)
            continue
        left, right = seg.split("->", 1)
        right_tbl = right.split(".")[0].strip().lower()
        if right_tbl == from_table:
            oriented.append(f"{right.strip()}->{left.strip()}")
        else:
            oriented.append(seg)
    return oriented


def _canonicalize_join_sig_segments(oriented: list[str]) -> list[str]:
    """Sort join signature segments for star/tree topologies before SQL. emission."""
    if len(oriented) <= 1:
        return oriented
    topology_type, _, _ = _analyze_join_topology(oriented)
    if topology_type in ("star", "tree"):
        return sorted(oriented, key=lambda s: s.strip().lower())
    return oriented


def canonicalize_stored_join_path_signature(
    signature: list[str],
    *,
    from_anchor: str | None = None,
) -> list[str]:
    """Return the emission-order join signature stored on intents and templates."""
    if not signature:
        return []
    oriented = _orient_join_sig_for_from(list(signature), from_anchor or "")
    return _canonicalize_join_sig_segments(oriented)


def _try_ast_inject_joins(
    det_sql: str,
    join_sigs_ordered: list[list[str]],
    edge_kinds_ordered: list[list[str]],
    schema: SchemaGraph | None,
    dialect: Dialect,
    cte_emissions: dict[str, CteEmissionKind] | None = None,
    *,
    preserve_tables: list[str] | None = None,
    probe_cte_names: frozenset[str] | None = None,
) -> str | None:
    """Parse deterministic SQL via the dialect adapter, attach joins on ordered carriers, and re-render. Returns ``None`` when parsing fails, carrier count mismatches, any slot fails to resolve structured edges, or any attach call fails."""
    parsed = dialect.parse_select(det_sql)
    if parsed is None:
        return None
    carriers = dialect.ordered_join_carrier_froms(parsed)
    if carriers is None:
        return None
    sigs_padded: list[list[str]] = list(join_sigs_ordered)
    kinds_padded: list[list[str]] = list(edge_kinds_ordered)
    if len(sigs_padded) != len(carriers):
        raise JoinInjectionAlignmentError(
            f"join_sigs_ordered length {len(sigs_padded)} does not match dialect join carrier count {len(carriers)}"
        )
    if len(kinds_padded) < len(sigs_padded):
        kinds_padded = kinds_padded + [[] for _ in range(len(sigs_padded) - len(kinds_padded))]
    per_slot_join: list[list[JoinEdge]] = []
    per_slot_where: list[list[JoinEdge]] = []
    per_slot_extra_from: list[list[str]] = []
    per_slot_anti_join: list[list[str]] = []
    for carrier, sig, kinds in zip(carriers, sigs_padded, kinds_padded, strict=True):
        if not sig:
            per_slot_join.append([])
            per_slot_where.append([])
            per_slot_extra_from.append([])
            per_slot_anti_join.append([])
            continue
        from_anchor = dialect.from_anchor_of(carrier)
        if not from_anchor:
            return None
        oriented = _orient_join_sig_for_from(sig, from_anchor)
        kinds_aligned = list(kinds)
        canon = _canonicalize_join_sig_segments(list(oriented))
        if canon != oriented:
            order = sorted(range(len(oriented)), key=lambda i: oriented[i].strip().lower())
            kinds_aligned = [kinds_aligned[i] for i in order]
        oriented = canon
        resolved = _join_edges_from_signature(
            oriented,
            kinds_aligned,
            from_anchor,
            schema,
            cte_emissions,
            preserve_tables=preserve_tables,
            probe_cte_names=probe_cte_names,
            dialect=dialect,
        )
        if resolved is None:
            return None
        join_edges, where_edges, extra_from_tables, anti_join_predicates = resolved
        if not join_edges and not where_edges:
            return None
        per_slot_join.append(join_edges)
        per_slot_where.append(where_edges)
        per_slot_extra_from.append(extra_from_tables)
        per_slot_anti_join.append(anti_join_predicates)
    for carrier, edges in zip(carriers, per_slot_join, strict=True):
        if not edges:
            continue
        if not dialect.attach_joins(parsed, carrier, edges):
            return None
    for carrier, where_edges, extra_from, anti_preds in zip(
        carriers, per_slot_where, per_slot_extra_from, per_slot_anti_join, strict=True
    ):
        if where_edges or extra_from:
            if not dialect.attach_extra_from_and_where(parsed, carrier, extra_from, where_edges):
                return None
        if anti_preds and not dialect.attach_where_sql_fragments(carrier, anti_preds):
            return None
    try:
        return dialect.emit_sql(parsed)
    except Exception:
        return None


def inject_join_into_deterministic_sql(
    det_sql: str,
    join_sigs_ordered: list[list[str]],
    schema: SchemaGraph | None = None,
    *,
    edge_kinds_ordered: list[list[str]] | None = None,
    dialect: Dialect | None = None,
    cte_emissions: dict[str, CteEmissionKind] | None = None,
    preserve_tables: list[str] | None = None,
    probe_cte_names: frozenset[str] | None = None,
) -> str:
    """Attach JOIN clauses for each ordered carrier (CTE inner SELECTs. left-to-right then outer SELECT) via the dialect's AST adapter and re-emit. Returns *det_sql* unchanged when *join_sigs_ordered* is empty or *dialect* is ``None``."""
    pipeline_trace(
        "sql_gen.inject_join_into_deterministic_sql.input",
        lambda: stable_json(
            {
                "det_sql": det_sql,
                "join_sigs_ordered": join_sigs_ordered,
                "edge_kinds_ordered": edge_kinds_ordered,
            }
        ),
    )
    if not join_sigs_ordered or dialect is None:
        return det_sql
    kinds_in: list[list[str]] = list(edge_kinds_ordered or [])
    try:
        ast_out = _try_ast_inject_joins(
            det_sql,
            join_sigs_ordered,
            kinds_in,
            schema,
            dialect,
            cte_emissions,
            preserve_tables=preserve_tables,
            probe_cte_names=probe_cte_names,
        )
    except JoinInjectionAlignmentError as exc:
        pipeline_trace(
            "sql_gen.inject_join.alignment_failed",
            stable_json(
                {
                    "det_sql": det_sql,
                    "join_sigs_ordered": join_sigs_ordered,
                    "edge_kinds_ordered": kinds_in,
                    "error": str(exc),
                }
            ),
        )
        raise
    if ast_out is None:
        pipeline_trace(
            "sql_gen.inject_join_into_deterministic_sql.ast_failed",
            lambda: stable_json(
                {
                    "det_sql": det_sql,
                    "join_sigs_ordered": join_sigs_ordered,
                    "edge_kinds_ordered": kinds_in,
                }
            ),
        )
        raise JoinInjectionFailedError(
            "AST join injection could not attach structured edges to deterministic SQL.",
            det_sql=det_sql,
            join_sigs_ordered=list(join_sigs_ordered),
            edge_kinds_ordered=list(kinds_in),
        )
    pipeline_trace("sql_gen.inject_join_into_deterministic_sql.output", lambda: stable_json({"sql": ast_out}))
    return ast_out


def _canonical_join_edge_string(schema: SchemaGraph | None, e: dict[str, Any]) -> str:
    """Build one undirected join edge string in a stable orientation. When *schema* is provided, a physical edge matching a declared catalog ``FKEdge`` is oriented as ``src->dst`` from that edge. Otherwise, lexicographic ``src_table`` / ``dst_table`` order breaks ties for inferred or virtual edges."""
    st = str(e["src_table"])
    dt = str(e["dst_table"])
    sc = [str(c) for c in e["src_cols"]]
    dc = [str(c) for c in e["dst_cols"]]
    sc_l = [c.lower() for c in sc]
    dc_l = [c.lower() for c in dc]
    if schema is not None:
        for fk in schema.fk_edges:
            if (
                fk.src_table == st
                and fk.dst_table == dt
                and [c.lower() for c in fk.src_cols] == sc_l
                and [c.lower() for c in fk.dst_cols] == dc_l
            ):
                return f"{fk.src_table}.{','.join(fk.src_cols)}->{fk.dst_table}.{','.join(fk.dst_cols)}"
            if (
                fk.src_table == dt
                and fk.dst_table == st
                and [c.lower() for c in fk.src_cols] == dc_l
                and [c.lower() for c in fk.dst_cols] == sc_l
            ):
                return f"{fk.src_table}.{','.join(fk.src_cols)}->{fk.dst_table}.{','.join(fk.dst_cols)}"
    if st == dt:
        return f"{st}.{','.join(sc)}->{dt}.{','.join(dc)}"
    if st < dt:
        return f"{st}.{','.join(sc)}->{dt}.{','.join(dc)}"
    return f"{dt}.{','.join(dc)}->{st}.{','.join(sc)}"


def _join_path_signature_for_path(path: list[dict[str, Any]], schema: SchemaGraph | None = None) -> list[str]:
    """Generate signature strings for each edge on a join path."""
    return [_canonical_join_edge_string(schema, e) for e in path]


def _candidate_join_paths_for_tables(
    schema: SchemaGraph, tables: list[str], *, cross_product_cap: int | None = None, tie_cap: int | None = None
) -> list[list[dict[str, Any]]]:
    """Compute all candidate join paths for a set of tables by trying. every table as root."""
    tables = sorted(set(tables))
    if len(tables) < 2:
        return [[]]

    def uniq_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Deduplicate edge dicts by canonical unordered endpoint pair.

        Args:

            edges: List of join edge dicts.

        Returns:

            List of edges with duplicate logical pairs removed.
        """
        seen: set[tuple[Any, ...]] = set()
        out: list[dict[str, Any]] = []
        for e in edges:
            pair = ((e["src_table"], tuple(e["src_cols"])), (e["dst_table"], tuple(e["dst_cols"])))
            canonical = tuple(sorted(pair))
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(e)
        return out

    table_set = set(tables)

    def _edges_cover_tables(edges: list[dict[str, Any]], root: str) -> set[str]:
        """Collect all table names reachable from edges plus the given. root. Args: edges: List of join edge dicts. root: Root table name always included in the covered set. Returns: Set of table names appearing in `edges` union `{root}`."""
        covered = {root}
        for e in edges:
            covered.add(e["src_table"])
            covered.add(e["dst_table"])
        return covered

    def _merge_paths_cartesian(root: str, others: list[str], allow_bridges: bool) -> list[list[dict[str, Any]]]:
        """Merge shortest-path ties from root to every other table via. a. capped cross-product. Args: root: Root table for path lookup in ``schema.join_paths_multi``. others: Tables that must be connected via merged edges. allow_bridges: When false, path endpoints must stay within the intent table set. Returns: Distinct merged edge lists, each covering every required table when possible."""
        max_out = max(1, int(PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP))
        others_sorted = sorted(others)
        if not others_sorted:
            return [[]]
        option_lists: list[list[list[dict[str, Any]]]] = []
        for target in others_sorted:
            _enforce_join_path_pair_tie_ceiling(schema, root, target, tie_cap=tie_cap)
            raw_paths, _overflow = stored_join_paths_for_pair(schema.join_paths_multi, root, target)
            valid: list[list[dict[str, Any]]] = []
            for p in raw_paths:
                if not p:
                    continue
                path_tables = _edges_cover_tables(p, root)
                if target not in path_tables:
                    continue
                if not allow_bridges and not path_tables <= table_set:
                    continue
                valid.append(p)
            if not valid:
                return []
            option_lists.append(valid)

        out_merges: list[list[dict[str, Any]]] = []
        seen_sig: set[tuple[str, ...]] = set()
        for combo in itertools.product(*option_lists):
            merged: list[dict[str, Any]] = []
            for p in combo:
                merged.extend(p)
            deduped = uniq_edges(merged)
            covered = _edges_cover_tables(deduped, root)
            if not table_set <= covered:
                continue
            if not allow_bridges and not covered <= table_set:
                continue
            sig = tuple(sorted(_join_path_signature_for_path(deduped, schema)))
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            out_merges.append(deduped)
        out_merges.sort(key=lambda m: (len(m), tuple(_join_path_signature_for_path(m, schema))))
        if len(out_merges) > max_out:
            _refuse_join_candidate_cap(len(out_merges), max_out, tables=list(tables), root=root)
        return out_merges

    def _collect(allow_bridges: bool) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        """Enumerate deduped join path candidates for all roots under. bridge policy. Args: allow_bridges: Description. Returns: unique path shape."""
        candidates: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for root in tables:
            others = [t for t in tables if t != root]
            for merged in _merge_paths_cartesian(root, others, allow_bridges):
                edge_tables = {root} | {e["src_table"] for e in merged} | {e["dst_table"] for e in merged}
                if not table_set <= edge_tables:
                    continue
                if not allow_bridges and not edge_tables <= table_set:
                    continue
                sig = tuple(sorted(_join_path_signature_for_path(merged, schema)))
                if sig not in candidates:
                    candidates[sig] = merged
        return candidates

    all_candidates = _collect(allow_bridges=False)
    if not all_candidates:
        all_candidates = _collect(allow_bridges=True)
        if all_candidates:
            debug(
                f"[sql_gen.candidate_join_paths_for_tables] no direct paths, found {len(all_candidates)} bridge paths"
            )

    cap = max(1, int(cross_product_cap or PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP))
    res = list(all_candidates.values())
    res.sort(key=lambda m: (len(m), tuple(_join_path_signature_for_path(m, schema))))
    if len(res) > cap:
        _refuse_join_candidate_cap(len(res), cap, tables=list(tables))
    return res


def _nodes_in_join_path(path: list[dict[str, Any]]) -> set[str]:
    """Return table names touched by *path* edges."""
    s: set[str] = set()
    for e in path:
        s.add(e["src_table"])
        s.add(e["dst_table"])
    return s


def _join_edge_sig_tuple(path: list[dict[str, Any]], schema: SchemaGraph | None = None) -> tuple[str, ...]:
    """Stable tuple key for deduplicating join paths."""
    return tuple(sorted(_join_path_signature_for_path(path, schema)))


def _dedupe_edges_stable(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate logical edges while preserving first-seen order."""
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for e in edges:
        pair = ((e["src_table"], tuple(e["src_cols"])), (e["dst_table"], tuple(e["dst_cols"])))
        canon = tuple(sorted(pair))
        if canon in seen:
            continue
        seen.add(canon)
        out.append(e)
    return out


def _tag_physical_join_path(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy physical FK path edges and attach ``edge_kind`` for join payloads."""
    out: list[dict[str, Any]] = []
    for e in path:
        d: dict[str, Any] = {
            "src_table": e["src_table"],
            "src_cols": list(e["src_cols"]),
            "dst_table": e["dst_table"],
            "dst_cols": list(e["dst_cols"]),
        }
        tag = e.get("inference_tag")
        _semantic_edge_msg = "user_override_semantic must be routed through semantic_join_neighbors, not as an FKEdge"
        assert tag != "user_override_semantic", _semantic_edge_msg
        if tag is None or tag == "" or tag == "catalog" or tag == "user_override_structural":
            d["edge_kind"] = "catalog_fk"
        elif tag == "self":
            d["edge_kind"] = "self_fk"
        elif tag == "suffix":
            d["edge_kind"] = "inferred_suffix_fk"
        elif tag == "composite":
            d["edge_kind"] = "composite_fk"
        elif tag in ("semantic", "semantic_promoted"):
            d["edge_kind"] = "inferred_semantic_fk"
        else:
            raise ValueError(f"unknown FK inference_tag: {tag!r}")
        out.append(d)
    return out


def _path_has_semantic_edge(path: list[dict[str, Any]]) -> bool:
    """Return True when any edge uses semantic overlap kinds."""
    for e in path:
        k = e.get("edge_kind", "")
        if k in ("semantic_profile", "semantic_profile_virtual"):
            return True
    return False


def _join_path_edge_kind_rank_key(path: list[dict[str, Any]]) -> tuple[int, ...]:
    """Return sorted per-edge kind ranks for stable preference ordering (lower is stronger)."""
    ranks = [JOIN_EDGE_KIND_RANK.get(str(e.get("edge_kind", "")), 99) for e in path]
    return tuple(sorted(ranks))


def _collect_virtual_bridge_single_edges(
    scope_tables: list[str], schema: SchemaGraph, virtual_specs: dict[str, VirtualTableSpec]
) -> list[list[dict[str, Any]]]:
    """Each item is a one-edge path for virtual PK/FK/shared-lineage bridges."""
    scope_set = set(scope_tables)
    out_single: list[list[dict[str, Any]]] = []
    seen_sig: set[tuple[str, ...]] = set()

    def add_edge(edge: dict[str, Any]) -> None:
        sig = tuple(sorted(_join_path_signature_for_path([edge], schema)))
        if sig in seen_sig:
            return
        seen_sig.add(sig)
        out_single.append([edge])
        src_t = edge.get("src_table")
        dst_t = edge.get("dst_table")
        ek = edge.get("edge_kind", "")
        if (
            isinstance(src_t, str)
            and isinstance(dst_t, str)
            and ek.startswith("virtual_shared")
            and src_t in virt_names
            and dst_t in virt_names
        ):
            pipeline_trace(
                "pipeline.join_resolve.skip_bridge_for_same_lineage_ctes",
                lambda: stable_json({"edge_kind": ek, "src_table": src_t, "dst_table": dst_t}),
            )

    virt_names = [t for t in scope_tables if t in virtual_specs]
    phys_names = [t for t in scope_tables if t in schema.tables]

    for V in virt_names:
        spec = virtual_specs[V]
        for P in phys_names:
            pk = list(schema.tables[P].primary_key or [])
            if not pk:
                continue
            dst_cols: list[str] = []
            ok = True
            for pkc in pk:
                hit = None
                for alias, vcol in spec.columns.items():
                    if vcol.lineage_phys_table == P and vcol.lineage_phys_column == pkc and vcol.inherits_pk:
                        hit = alias
                        break
                if not hit:
                    ok = False
                    break
                dst_cols.append(hit)
            if ok:
                add_edge(
                    {
                        "src_table": P,
                        "src_cols": list(pk),
                        "dst_table": V,
                        "dst_cols": dst_cols,
                        "edge_kind": "virtual_pk_bridge",
                    }
                )

        for alias, vcol in spec.columns.items():
            if not vcol.fk_to:
                continue
            D, dcol = vcol.fk_to
            if D in scope_set:
                add_edge(
                    {
                        "src_table": V,
                        "src_cols": [alias],
                        "dst_table": D,
                        "dst_cols": [dcol],
                        "edge_kind": "virtual_fk_bridge",
                    }
                )
            elif D in schema.tables:
                add_edge(
                    {
                        "src_table": V,
                        "src_cols": [alias],
                        "dst_table": D,
                        "dst_cols": [dcol],
                        "edge_kind": "virtual_fk_shadow_path",
                    }
                )

    virt_sorted = sorted(virt_names)
    for i, v1 in enumerate(virt_sorted):
        s1 = virtual_specs[v1]
        for v2 in virt_sorted[i + 1 :]:
            s2 = virtual_specs[v2]
            for a1, c1 in s1.columns.items():
                if not c1.lineage_phys_table or not c1.lineage_phys_column:
                    continue
                role1 = c1.inherits_pk or bool(c1.fk_to)
                lk1 = (c1.lineage_phys_table, c1.lineage_phys_column)
                for a2, c2 in s2.columns.items():
                    if not c2.lineage_phys_table or not c2.lineage_phys_column:
                        continue
                    lk2 = (c2.lineage_phys_table, c2.lineage_phys_column)
                    if lk1 != lk2:
                        continue
                    role2 = c2.inherits_pk or bool(c2.fk_to)
                    if not role1 and not role2:
                        continue
                    if (v1, a1) <= (v2, a2):
                        t_left, c_left, t_right, c_right = v1, a1, v2, a2
                    else:
                        t_left, c_left, t_right, c_right = v2, a2, v1, a1
                    base_tbl = c1.lineage_phys_table
                    phys_col = c1.lineage_phys_column
                    pk_cols: set[str] = set()
                    if base_tbl and base_tbl in schema.tables:
                        pk_cols = set(schema.tables[base_tbl].primary_key or [])
                    if c1.inherits_pk and c2.inherits_pk and phys_col in pk_cols:
                        edge_kind = "virtual_shared_pk"
                    elif role1 and role2:
                        edge_kind = "virtual_shared_base"
                    else:
                        edge_kind = "virtual_shared_lineage"
                    add_edge(
                        {
                            "src_table": t_left,
                            "src_cols": [c_left],
                            "dst_table": t_right,
                            "dst_cols": [c_right],
                            "edge_kind": edge_kind,
                        }
                    )

    for i, v1 in enumerate(virt_sorted):
        s1 = virtual_specs[v1]
        for v2 in virt_sorted[i + 1 :]:
            s2 = virtual_specs[v2]
            for a1, c1 in s1.columns.items():
                fk1 = c1.fk_to
                if not fk1:
                    continue
                for a2, c2 in s2.columns.items():
                    fk2 = c2.fk_to
                    if not fk2 or fk1 != fk2:
                        continue
                    if (v1, a1) <= (v2, a2):
                        t_left, c_left, t_right, c_right = v1, a1, v2, a2
                    else:
                        t_left, c_left, t_right, c_right = v2, a2, v1, a1
                    add_edge(
                        {
                            "src_table": t_left,
                            "src_cols": [c_left],
                            "dst_table": t_right,
                            "dst_cols": [c_right],
                            "edge_kind": "virtual_shared_fk_target",
                        }
                    )

    return out_single


def _extend_join_paths_with_bridges(
    base_paths: list[list[dict[str, Any]]],
    bridges: list[list[dict[str, Any]]],
    target: frozenset[str],
    schema: SchemaGraph,
) -> list[list[dict[str, Any]]]:
    """Attach virtual bridge edges to physical paths until all *target* tables appear."""
    if len(target) < 2:
        return [[]]
    bridges_sorted = sorted(bridges, key=lambda p: _join_path_signature_for_path(p, schema))
    out_map: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    bases = base_paths if base_paths else [[]]
    max_steps = max(len(bridges_sorted) * 4, 8)

    for base in bases:
        acc = _dedupe_edges_stable(base)
        steps = 0
        while not (target <= _nodes_in_join_path(acc)) and steps < max_steps:
            steps += 1
            before = len(_nodes_in_join_path(acc))
            progressed = False
            for br in bridges_sorted:
                acc2 = _dedupe_edges_stable(acc + br)
                if len(_nodes_in_join_path(acc2)) > before:
                    acc = acc2
                    progressed = True
                    break
            if not progressed:
                break
        if target <= _nodes_in_join_path(acc):
            out_map.setdefault(_join_edge_sig_tuple(acc, schema), acc)

    if not out_map:
        bridge_acc: list[dict[str, Any]] = []
        steps = 0
        while not (target <= _nodes_in_join_path(bridge_acc)) and steps < max_steps:
            steps += 1
            before = len(_nodes_in_join_path(bridge_acc))
            progressed = False
            for br in bridges_sorted:
                acc2 = _dedupe_edges_stable(bridge_acc + br)
                if len(_nodes_in_join_path(acc2)) > before:
                    bridge_acc = acc2
                    progressed = True
                    break
            if not progressed:
                break
        if target <= _nodes_in_join_path(bridge_acc):
            out_map.setdefault(_join_edge_sig_tuple(bridge_acc, schema), bridge_acc)

    return list(out_map.values())


def _path_uses_only_visible_columns(path: list[dict[str, Any]], schema: SchemaGraph) -> bool:
    """Return True iff every physical-table column referenced by *path* edges is visible. Virtual (CTE) columns are not gated here; only physical schema columns are checked against :attr:`ColumnMetadata.is_visible`. Used to filter LLM-facing join candidates so denied or hidden-sensitivity columns never appear in any rendered signature."""
    for edge in path:
        for tbl_key, cols_key in (("src_table", "src_cols"), ("dst_table", "dst_cols")):
            tbl = edge.get(tbl_key)
            if not tbl or tbl not in schema.tables:
                continue
            tmeta = schema.tables[tbl]
            for col in edge.get(cols_key, []) or []:
                cm = tmeta.columns.get(col)
                if cm is not None and (not cm.is_visible or not cm.is_selectable):
                    return False
    return True


def enumerate_join_paths_base(
    scope_tables: list[str],
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
    *,
    cross_product_cap: int | None = None,
    tie_cap: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Enumerate FK-derived and virtual-bridge join paths covering all. scope tables."""
    stable_scope = list(dict.fromkeys(scope_tables))
    if len(stable_scope) < 2:
        return [[]]
    target = frozenset(stable_scope)
    phys = sorted([t for t in stable_scope if t in schema.tables])
    implicit_extra: set[str] = set()
    for _vname, spec in virtual_specs.items():
        for _alias, vcol in spec.columns.items():
            fk = vcol.fk_to
            if not fk:
                continue
            dtab, _dcol = fk
            if dtab in schema.tables and dtab not in stable_scope:
                implicit_extra.add(dtab)
    phys_implicit = sorted(set(phys) | implicit_extra)
    virt_edges = _collect_virtual_bridge_single_edges(stable_scope, schema, virtual_specs)
    phys_paths_raw: list[list[dict[str, Any]]] = []
    if len(phys_implicit) >= 2:
        phys_paths_raw = [
            _tag_physical_join_path(p)
            for p in _candidate_join_paths_for_tables(
                schema, phys_implicit, cross_product_cap=cross_product_cap, tie_cap=tie_cap
            )
        ]
    elif len(phys) >= 2:
        phys_paths_raw = [
            _tag_physical_join_path(p)
            for p in _candidate_join_paths_for_tables(
                schema, phys, cross_product_cap=cross_product_cap, tie_cap=tie_cap
            )
        ]
    else:
        phys_paths_raw = [[]]
    merged = _extend_join_paths_with_bridges(phys_paths_raw, virt_edges, target, schema)
    if not any(target <= _nodes_in_join_path(p) for p in merged):
        merged = _extend_join_paths_with_bridges([[]], virt_edges, target, schema)
    filtered = [p for p in merged if target <= _nodes_in_join_path(p)]
    visible = [p for p in filtered if _path_uses_only_visible_columns(p, schema)]
    return visible if visible else [[]]


def enumerate_semantic_paths(
    scope_tables: list[str], schema: SchemaGraph, virtual_specs: dict[str, VirtualTableSpec]
) -> list[list[dict[str, Any]]]:
    """Build single-edge semantic paths from profiled. ``semantic_join_neighbors`` on physical columns (and the same lists lifted onto virtual CTE columns), plus an overlap pass for virtual–virtual pairs that is not represented on the physical graph."""
    scope_set = set(scope_tables)
    out: list[list[dict[str, Any]]] = []
    seen: set[tuple[tuple[str, str], tuple[str, str]]] = set()

    def edge_key(a: tuple[str, str], b: tuple[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
        return (a, b) if a <= b else (b, a)

    def emit(left_t: str, left_c: str, right_t: str, right_c: str, virt1: bool, virt2: bool) -> None:
        ek = edge_key((left_t, left_c), (right_t, right_c))
        if ek in seen:
            return
        seen.add(ek)
        kind = "semantic_profile_virtual" if virt1 and virt2 else "semantic_profile"
        if (left_t, left_c) <= (right_t, right_c):
            lt, lc, rt, rc = left_t, left_c, right_t, right_c
        else:
            lt, lc, rt, rc = right_t, right_c, left_t, left_c
        out.append(
            [
                {
                    "src_table": lt,
                    "src_cols": [lc],
                    "dst_table": rt,
                    "dst_cols": [rc],
                    "edge_kind": kind,
                }
            ]
        )

    for t in scope_tables:
        if t not in schema.tables:
            continue
        for cn, cmeta in schema.tables[t].columns.items():
            for nt, nc in cmeta.semantic_join_neighbors:
                if nt not in scope_set:
                    continue
                if nt not in schema.tables or nc not in schema.tables[nt].columns:
                    continue
                left_src = schema.tables[t].source_id
                right_src = schema.tables[nt].source_id
                if left_src and right_src and left_src != right_src:
                    continue
                emit(t, cn, nt, nc, False, False)

    for vt in scope_tables:
        if vt not in virtual_specs:
            continue
        for an, vc in virtual_specs[vt].columns.items():
            for nt, nc in vc.semantic_join_neighbors:
                if nt not in scope_set:
                    continue
                if nt in schema.tables and nc in schema.tables[nt].columns:
                    emit(vt, an, nt, nc, True, False)

    min_ratio = PolicyConfig.SEMANTIC_JOIN_MIN_OVERLAP_RATIO
    vcols: list[tuple[str, str, list[str]]] = []
    for t in scope_tables:
        if t not in virtual_specs:
            continue
        for an, vc in virtual_specs[t].columns.items():
            vals = list(vc.semantic_distinct_values)
            if vals:
                vcols.append((t, an, vals))
    for i, (t1, c1, v1) in enumerate(vcols):
        s1 = set(v1)
        if not s1:
            continue
        for t2, c2, v2 in vcols[i + 1 :]:
            if t1 == t2:
                continue
            s2 = set(v2)
            if not s2:
                continue
            inter = len(s1 & s2)
            if inter / min(len(s1), len(s2)) < min_ratio:
                continue
            emit(t1, c1, t2, c2, True, True)

    return [p for p in out if _path_uses_only_visible_columns(p, schema)]


def _dedupe_paths_by_sig(paths: list[list[dict[str, Any]]], schema: SchemaGraph) -> list[list[dict[str, Any]]]:
    """Deduplicate path lists by sorted join signature tuple."""
    out_map: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for p in paths:
        out_map.setdefault(_join_edge_sig_tuple(p, schema), p)
    return list(out_map.values())


def _order_join_candidates_stable(
    paths: list[list[dict[str, Any]]],
    schema: SchemaGraph,
    intent: RuntimeIntent | None = None,
) -> list[list[dict[str, Any]]]:
    """Deterministic ordering: fan-out penalty, base tier, length, edge kinds, signatures, path dump."""
    anchor = sorted(set(intent.tables))[0] if intent and intent.tables else None

    def sort_key(p: list[dict[str, Any]]) -> tuple[Any, ...]:
        penalty = fan_out_penalty_for_path_edges(p, intent, schema, from_anchor=anchor) if intent is not None else 0
        ext = 1 if _path_has_semantic_edge(p) else 0
        kind_rank = _join_path_edge_kind_rank_key(p)
        kinds = tuple(sorted(e.get("edge_kind", "") for e in p))
        sigs = tuple(sorted(_join_path_signature_for_path(p, schema)))
        edge_dump = stable_json([{"s": e["src_table"], "d": e["dst_table"], "k": e.get("edge_kind", "")} for e in p])
        return (penalty, ext, kind_rank, len(p), kinds, sigs, edge_dump)

    return sorted(paths, key=sort_key)


def tables_in_join_scope(
    tables: list[str] | None, schema: SchemaGraph, virtual_specs: dict[str, VirtualTableSpec]
) -> list[str]:
    """Return declared names that resolve to physical tables or non- scalar virtual CTE specs. Scalar-subquery CTEs (``emission == "scalar_subquery"``) are excluded from the join scope: they are CROSS JOIN'd into the FROM list at render time and do not participate in physical / virtual FK enumeration."""
    out: list[str] = []
    seen: set[str] = set()
    for t in tables or []:
        if t in schema.tables:
            if t not in seen:
                out.append(t)
                seen.add(t)
            continue
        spec = virtual_specs.get(t)
        if spec is None:
            continue
        if (spec.emission or "join_table") == "scalar_subquery":
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def physical_tables_for_join_hints(tables: list[str] | None, schema: SchemaGraph) -> list[str]:
    """Return physical table names from `tables` that exist in `schema`."""
    if not tables:
        return []
    by_lower: dict[str, str] = {k.lower(): k for k in schema.tables}
    out: list[str] = []
    for raw in tables:
        if not raw:
            continue
        key = by_lower.get(raw.lower())
        if key is None:
            continue
        out.append(key)
    return out


def join_hints_multi(
    schema: SchemaGraph,
    tables: list[str],
    intent: RuntimeIntent | None = None,
    prepend_paths: list[list[dict[str, Any]]] | None = None,
    virtual_specs: dict[str, VirtualTableSpec] | None = None,
    include_semantic: bool = False,
    *,
    cross_product_cap: int | None = None,
    tie_cap: int | None = None,
) -> dict[str, Any]:
    """Build ordered join candidates with ``edge_kinds`` and. ``candidate_tier`` labels."""
    _ = prepend_paths
    virtual_specs = virtual_specs or {}
    if len(tables) <= 1:
        debug("[sql_gen.join_hints_multi] single table, returning J00")
        return {
            "candidates": [
                {
                    "candidate_id": "J00",
                    "join_path_signature": [],
                    "edge_kinds": [],
                    "candidate_tier": "base",
                    "edge_count": 0,
                }
            ]
        }

    base_paths = enumerate_join_paths_base(
        tables, schema, virtual_specs, cross_product_cap=cross_product_cap, tie_cap=tie_cap
    )
    sem_paths = enumerate_semantic_paths(tables, schema, virtual_specs) if include_semantic else []
    hybrid_paths: list[list[dict[str, Any]]] = []
    if include_semantic and len(tables) >= 2:
        hybrid_paths = _compose_hybrid_fk_semantic_paths(base_paths, sem_paths, frozenset(tables), schema)
    merged_paths = _dedupe_paths_by_sig(base_paths + sem_paths + hybrid_paths, schema)
    if not include_semantic:
        merged_paths = [p for p in merged_paths if not _path_has_semantic_edge(p)]
    ordered = _order_join_candidates_stable(merged_paths, schema, intent)
    ordered_eff = [p for p in ordered if p]
    if not ordered_eff:
        merged_nonempty = [p for p in merged_paths if p]
        if merged_nonempty:
            ordered_eff = _order_join_candidates_stable(merged_nonempty, schema, intent)
            ordered_eff = [p for p in ordered_eff if p]
    if not ordered_eff:
        debug("[sql_gen.join_hints_multi] no candidates, returning J00")
        return {
            "candidates": [
                {
                    "candidate_id": "J00",
                    "join_path_signature": [],
                    "edge_kinds": [],
                    "candidate_tier": "base",
                    "edge_count": 0,
                }
            ]
        }

    probe_names = probe_cte_names(intent.cte_steps if intent is not None else None)
    resolved_probe_segments, resolved_probe_kinds = resolve_probe_edge_segments_from_lineage(
        intent, probe_names, schema, virtual_specs
    )

    out: list[dict[str, Any]] = []
    for idx, edges in enumerate(ordered_eff):
        sigs = _join_path_signature_for_path(edges, schema)
        kinds = [str(e.get("edge_kind", "catalog_fk")) for e in edges]
        tier = "extended" if _path_has_semantic_edge(edges) else "base"
        out.append(
            {
                "candidate_id": f"J{idx + 1:02d}",
                "join_path_signature": sigs,
                "edge_kinds": kinds,
                "candidate_tier": tier,
                "edge_count": len(edges),
            }
        )
    if resolved_probe_segments:
        out = apply_probe_edge_lineage_resolution(out, probe_names, resolved_probe_segments, resolved_probe_kinds)
    out = collapse_probe_edge_candidate_variation(out, probe_names)
    debug(f"[sql_gen.join_hints_multi] generated {len(out)} candidates")
    return {"candidates": out}


def _join_path_segment_touches_probe(seg: str, probe_names: frozenset[str]) -> bool:
    """Return whether a join-path signature segment references a probe CTE."""
    seg = seg.strip()
    if "->" not in seg:
        return False
    left_part, right_part = seg.split("->", 1)
    if "." not in left_part or "." not in right_part:
        return False
    left_tbl = left_part.strip().split(".", 1)[0].strip()
    right_tbl = right_part.strip().split(".", 1)[0].strip()
    return left_tbl in probe_names or right_tbl in probe_names


def resolve_probe_edge_segments_from_lineage(
    intent: RuntimeIntent | None,
    probe_names: frozenset[str],
    schema: SchemaGraph | None,
    virtual_specs: dict[str, VirtualTableSpec] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve probe join edges from declared output-column lineage before candidate classification."""
    if not probe_names or intent is None:
        return [], []
    virtual_specs = virtual_specs or {}
    segments: list[str] = []
    kinds: list[str] = []
    seen: set[str] = set()
    for probe in sorted(probe_names):
        spec = virtual_specs.get(probe)
        if spec is None:
            continue
        for alias, vcol in spec.columns.items():
            if not vcol.fk_to:
                continue
            dst_table, dst_col = vcol.fk_to
            edge = {
                "src_table": probe,
                "src_cols": [alias],
                "dst_table": dst_table,
                "dst_cols": [dst_col],
                "edge_kind": "virtual_fk_bridge",
            }
            seg = _canonical_join_edge_string(schema, edge)
            if seg not in seen:
                seen.add(seg)
                segments.append(seg)
                kinds.append("virtual_fk_bridge")
        if any(vcol.fk_to for vcol in spec.columns.values()):
            continue
        if schema is None:
            continue
        phys_tables: set[str] = set()
        for vcol in spec.columns.values():
            if vcol.inherits_pk and vcol.lineage_phys_table:
                phys_tables.add(vcol.lineage_phys_table)
        for phys_table in sorted(phys_tables):
            tmeta = schema.tables.get(phys_table)
            if tmeta is None:
                continue
            pk = list(tmeta.primary_key or [])
            if not pk:
                continue
            dst_cols: list[str] = []
            ok = True
            for pk_col in pk:
                hit: str | None = None
                for alias, vcol in spec.columns.items():
                    if (
                        vcol.lineage_phys_table == phys_table
                        and vcol.lineage_phys_column == pk_col
                        and vcol.inherits_pk
                    ):
                        hit = alias
                        break
                if not hit:
                    ok = False
                    break
                dst_cols.append(hit)
            if not ok:
                continue
            edge = {
                "src_table": phys_table,
                "src_cols": list(pk),
                "dst_table": probe,
                "dst_cols": dst_cols,
                "edge_kind": "virtual_pk_bridge",
            }
            seg = _canonical_join_edge_string(schema, edge)
            if seg not in seen:
                seen.add(seg)
                segments.append(seg)
                kinds.append("virtual_pk_bridge")
    return segments, kinds


def normalize_probe_edges_in_join_path_signature(
    signature: list[str],
    edge_kinds: list[str],
    probe_names: frozenset[str],
    resolved_segments: list[str],
    resolved_kinds: list[str],
) -> tuple[list[str], list[str]]:
    """Replace probe-touching segments with lineage-resolved probe edges."""
    if not probe_names or not resolved_segments:
        return list(signature), list(edge_kinds)
    sig = list(signature or [])
    kinds = list(edge_kinds or [])
    if len(kinds) != len(sig):
        raise JoinProbeEdgeKindMismatchError(len(sig), len(kinds))
    kept_sig: list[str] = []
    kept_kinds: list[str] = []
    for seg, kind in zip(sig, kinds, strict=True):
        if _join_path_segment_touches_probe(seg, probe_names):
            continue
        kept_sig.append(seg)
        kept_kinds.append(kind)
    for seg, kind in zip(resolved_segments, resolved_kinds, strict=True):
        if seg not in kept_sig:
            kept_sig.append(seg)
            kept_kinds.append(kind)
    return kept_sig, kept_kinds


def apply_probe_edge_lineage_resolution(
    candidates: list[dict[str, Any]],
    probe_names: frozenset[str],
    resolved_segments: list[str],
    resolved_kinds: list[str],
) -> list[dict[str, Any]]:
    """Normalize each candidate's join path to use lineage-resolved probe edges."""
    if not probe_names or not resolved_segments or not candidates:
        return candidates
    out: list[dict[str, Any]] = []
    for cand in candidates:
        sig, kinds = normalize_probe_edges_in_join_path_signature(
            list(cand.get("join_path_signature") or []),
            list(cand.get("edge_kinds") or []),
            probe_names,
            resolved_segments,
            resolved_kinds,
        )
        updated = dict(cand)
        updated["join_path_signature"] = sig
        updated["edge_kinds"] = kinds
        updated["edge_count"] = len(sig)
        out.append(updated)
    return out


def _join_path_signature_without_probe_edges(signature: list[str], probe_names: frozenset[str]) -> tuple[str, ...]:
    """Project a join path signature onto edges that do not touch a probe CTE."""
    if not probe_names:
        return tuple(sorted(signature))
    kept: list[str] = []
    for seg in signature:
        seg = seg.strip()
        if _join_path_segment_touches_probe(seg, probe_names):
            continue
        kept.append(seg)
    return tuple(sorted(kept))


def _collapse_key_excluding_deterministic_probe_edges(
    signature: list[str],
    edge_kinds: list[str],
    probe_names: frozenset[str],
) -> tuple[str, ...]:
    """Project a join signature for probe collapse, keeping semantic probe attachments."""
    kept: list[str] = []
    for seg, kind in zip(signature, edge_kinds, strict=True):
        if _join_path_segment_touches_probe(seg, probe_names) and kind in DETERMINISTIC_PROBE_EDGE_KINDS:
            continue
        kept.append(seg)
    return tuple(sorted(kept))


def collapse_probe_edge_candidate_variation(
    candidates: list[dict[str, Any]], probe_names: frozenset[str]
) -> list[dict[str, Any]]:
    """Collapse join candidates that differ only on deterministic probe attachment edges."""
    if not probe_names or not candidates:
        return candidates
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for cand in candidates:
        sig = list(cand.get("join_path_signature") or [])
        kinds = list(cand.get("edge_kinds") or [])
        if len(kinds) < len(sig):
            kinds = kinds + ["catalog_fk"] * (len(sig) - len(kinds))
        key = _collapse_key_excluding_deterministic_probe_edges(sig, kinds, probe_names)
        if key in seen:
            continue
        seen[key] = cand
        out.append(cand)
    return out if out else candidates


def join_choice_scope_key_cte(cte_name: str) -> str:
    """Return the canonical join-choice scope key for a CTE name."""
    return f"cte:{cte_name}"


def _join_candidate_bucket(c: dict[str, Any]) -> Literal["fk", "semantic"]:
    """Map stored tiers to FK versus semantic buckets for join policy."""
    tier = c.get("candidate_tier")
    if tier in ("extended", "semantic"):
        return "semantic"
    return "fk"


def split_fk_vs_semantic_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition non-``J00`` candidates into FK-style paths versus semantic tiers."""
    fk: list[dict[str, Any]] = []
    sem: list[dict[str, Any]] = []
    for c in candidates:
        cid = c.get("candidate_id")
        if not isinstance(cid, str) or cid == "J00":
            continue
        if _join_candidate_bucket(c) == "semantic":
            sem.append(c)
        else:
            fk.append(c)
    return fk, sem


def classify_scope_candidates(candidates: list[dict[str, Any]], *, needs_join: bool = True) -> ScopeClass:
    """Classify a scope's candidate list for deterministic resolution versus LLM routing. When *needs_join* is false, an all-``J00`` payload is treated as ``single_table`` instead of ``empty``."""
    if not candidates:
        return ScopeClass.empty if needs_join else ScopeClass.single_table
    non_j00 = [c for c in candidates if c.get("candidate_id") != "J00"]
    if not non_j00:
        return ScopeClass.empty if needs_join else ScopeClass.single_table
    fk, sem = split_fk_vs_semantic_candidates(candidates)
    n_fk = len(fk)
    n_sem = len(sem)
    if n_fk == 0 and n_sem == 0:
        return ScopeClass.empty
    if n_fk == 0:
        return ScopeClass.semantic_only
    if n_sem == 0:
        if n_fk == 1:
            return ScopeClass.single_fk
        return ScopeClass.multi_fk
    if n_fk == 1:
        return ScopeClass.single_fk_with_semantic
    return ScopeClass.multi_fk_with_semantic


def _candidates_slice_fk_only(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sliced = [c for c in candidates if c.get("candidate_id") == "J00" or _join_candidate_bucket(c) == "fk"]
    if sliced:
        return sliced
    return [c for c in candidates if c.get("candidate_id") == "J00"]


def join_scope_pass1_plan(
    *,
    main_multi_table: bool,
    main_tables: list[str],
    main_candidates: list[dict[str, Any]],
    cte_scopes: list[tuple[str, list[str], list[dict[str, Any]]]],
    forbid_na: bool,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, bool], dict[str, ScopeClass]]:
    """Build preset join ids and the first-pass join-choice LLM scope. list."""
    preset: dict[str, str] = {}
    llm_scopes: list[dict[str, Any]] = []
    accept_na_by_scope: dict[str, bool] = {}
    scope_class: dict[str, ScopeClass] = {}

    if not main_multi_table:
        preset[JOIN_CHOICE_SCOPE_MAIN] = "J00"
    else:
        sc_main = classify_scope_candidates(main_candidates, needs_join=True)
        scope_class[JOIN_CHOICE_SCOPE_MAIN] = sc_main
        if sc_main == ScopeClass.empty:
            raise NoJoinPathError("main query", list(main_tables))
        if sc_main == ScopeClass.single_table:
            preset[JOIN_CHOICE_SCOPE_MAIN] = "J00"
        elif sc_main == ScopeClass.single_fk:
            fk_main, _ = split_fk_vs_semantic_candidates(main_candidates)
            preset[JOIN_CHOICE_SCOPE_MAIN] = str(fk_main[0]["candidate_id"])
        elif sc_main == ScopeClass.semantic_only:
            _, sem_main = split_fk_vs_semantic_candidates(main_candidates)
            if len(sem_main) == 1:
                preset[JOIN_CHOICE_SCOPE_MAIN] = str(sem_main[0]["candidate_id"])
            else:
                llm_scopes.append(
                    {
                        "scope": JOIN_CHOICE_SCOPE_MAIN,
                        "tables": list(main_tables),
                        "candidates": list(main_candidates),
                    }
                )
                accept_na_by_scope[JOIN_CHOICE_SCOPE_MAIN] = False
        elif sc_main == ScopeClass.multi_fk:
            llm_scopes.append(
                {
                    "scope": JOIN_CHOICE_SCOPE_MAIN,
                    "tables": list(main_tables),
                    "candidates": _candidates_slice_fk_only(main_candidates),
                }
            )
            accept_na_by_scope[JOIN_CHOICE_SCOPE_MAIN] = False
        elif sc_main == ScopeClass.single_fk_with_semantic:
            fk_slice = _candidates_slice_fk_only(main_candidates)
            llm_scopes.append(
                {
                    "scope": JOIN_CHOICE_SCOPE_MAIN,
                    "tables": list(main_tables),
                    "candidates": fk_slice,
                }
            )
            spans = any(join_candidate_spans_tables(c, main_tables) for c in fk_slice)
            accept_na_by_scope[JOIN_CHOICE_SCOPE_MAIN] = False if (forbid_na or spans) else True
        else:
            fk_slice = _candidates_slice_fk_only(main_candidates)
            llm_scopes.append(
                {
                    "scope": JOIN_CHOICE_SCOPE_MAIN,
                    "tables": list(main_tables),
                    "candidates": fk_slice,
                }
            )
            spans = any(join_candidate_spans_tables(c, main_tables) for c in fk_slice)
            accept_na_by_scope[JOIN_CHOICE_SCOPE_MAIN] = False if (forbid_na or spans) else True

    for cte_name, tbls, cands in cte_scopes:
        sk = join_choice_scope_key_cte(cte_name)
        sc_cte = classify_scope_candidates(cands, needs_join=True)
        scope_class[sk] = sc_cte
        if sc_cte == ScopeClass.empty:
            raise NoJoinPathError(f"CTE '{cte_name}'", list(tbls))
        if sc_cte == ScopeClass.single_table:
            preset[sk] = "J00"
        elif sc_cte == ScopeClass.single_fk:
            fk_c, _ = split_fk_vs_semantic_candidates(cands)
            preset[sk] = str(fk_c[0]["candidate_id"])
        elif sc_cte == ScopeClass.semantic_only:
            _, sem_cte = split_fk_vs_semantic_candidates(cands)
            if len(sem_cte) == 1:
                preset[sk] = str(sem_cte[0]["candidate_id"])
            else:
                llm_scopes.append({"scope": sk, "tables": list(tbls), "candidates": list(cands)})
                accept_na_by_scope[sk] = False
        elif sc_cte == ScopeClass.multi_fk:
            llm_scopes.append(
                {
                    "scope": sk,
                    "tables": list(tbls),
                    "candidates": _candidates_slice_fk_only(cands),
                }
            )
            accept_na_by_scope[sk] = False
        elif sc_cte == ScopeClass.single_fk_with_semantic:
            fk_slice = _candidates_slice_fk_only(cands)
            llm_scopes.append(
                {
                    "scope": sk,
                    "tables": list(tbls),
                    "candidates": fk_slice,
                }
            )
            spans = any(join_candidate_spans_tables(c, tbls) for c in fk_slice)
            accept_na_by_scope[sk] = False if (forbid_na or spans) else True
        else:
            fk_slice = _candidates_slice_fk_only(cands)
            llm_scopes.append(
                {
                    "scope": sk,
                    "tables": list(tbls),
                    "candidates": fk_slice,
                }
            )
            spans = any(join_candidate_spans_tables(c, tbls) for c in fk_slice)
            accept_na_by_scope[sk] = False if (forbid_na or spans) else True

    return preset, llm_scopes, accept_na_by_scope, scope_class


def join_scope_pass2_llm_scopes(
    na_scope_keys: frozenset[str],
    join_main: dict[str, Any],
    join_cte: dict[str, dict[str, Any]],
    intent: RuntimeIntent,
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
) -> list[dict[str, Any]]:
    """Build second-pass join-choice payloads with FK and semantic candidates for NA scopes."""
    out: list[dict[str, Any]] = []
    for sk in na_scope_keys:
        if sk == JOIN_CHOICE_SCOPE_MAIN:
            tables = list(tables_in_join_scope(intent.tables, schema, virtual_specs))
            cands = list(join_main.get("candidates") or [])
        elif sk.startswith("cte:"):
            cte_name = sk.split(":", 1)[1]
            step = next((s for s in (intent.cte_steps or []) if s.cte_name == cte_name), None)
            hints = join_cte.get(cte_name) or {}
            cands = list(hints.get("candidates") or [])
            tables = list(tables_in_join_scope(step.tables or [], schema, virtual_specs)) if step is not None else []
        else:
            continue
        out.append({"scope": sk, "tables": tables, "candidates": cands})
    return out


def merge_join_hints_for_na_scopes(
    pass1_main: dict[str, Any],
    pass1_cte: dict[str, dict[str, Any]],
    intent: RuntimeIntent,
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
    na_scopes: frozenset[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Rebuild join hint payloads for scopes that returned NA, keeping. others on pass one."""
    if JOIN_CHOICE_SCOPE_MAIN in na_scopes:
        scope = tables_in_join_scope(intent.tables, schema, virtual_specs)
        main = join_hints_multi(schema, scope, intent, virtual_specs=virtual_specs, include_semantic=True)
    else:
        main = deepcopy(pass1_main)

    cte_out: dict[str, dict[str, Any]] = {}
    cte_by_name = {s.cte_name: s for s in (intent.cte_steps or [])}
    for cte_name, hints in pass1_cte.items():
        if cte_name in na_scopes:
            step = cte_by_name.get(cte_name)
            if step and len(step.tables or []) >= 2:
                scope = tables_in_join_scope(step.tables, schema, virtual_specs)
                cte_slice = cte_to_intent_for_ranking(step)
                cte_out[cte_name] = join_hints_multi(
                    schema, scope, cte_slice, virtual_specs=virtual_specs, include_semantic=True
                )
            else:
                cte_out[cte_name] = deepcopy(hints)
        else:
            cte_out[cte_name] = deepcopy(hints)
    return main, cte_out


def _quote_simple_qualified_mul_token(term: str, dialect: Dialect | None) -> str:
    """Quote ``table.column`` tokens in a MulGroup multiply/divide term when *dialect* is set."""
    if dialect is None or not term:
        return term
    raw = term
    t = term.strip()
    dis = ""
    if len(t) >= 9 and t[:9].upper() == "DISTINCT ":
        dis = "DISTINCT "
        t = t[9:].lstrip()
    if "(" in t or ")" in t or t.count(".") != 1:
        return raw
    tbl, col = t.split(".", 1)
    if not tbl or not col or any(c.isspace() for c in tbl) or any(c.isspace() for c in col):
        return raw
    return f"{dis}{dialect.quote_table_column(tbl, col)}"


def _render_scalar_func_args(func_name: str, args: list[ScalarArg] | None, param_keys: list[str] | None) -> list[str]:
    """Render scalar-function argument tokens, inlining literals when a. param key is absent."""
    out: list[str] = []
    fn = (func_name or "").lower()
    keys = list(param_keys or [])
    vals = list(args or [])
    for idx, val in enumerate(vals):
        key = keys[idx] if idx < len(keys) else ""
        if key:
            out.append(f":{key}")
            continue
        if isinstance(val, bool):
            out.append("TRUE" if val else "FALSE")
            continue
        if isinstance(val, (int, float)):
            out.append(str(val))
            continue
        text = str(val)
        if fn == "extract" and idx == 0:
            out.append(text.upper())
            continue
        out.append("'" + text.replace("'", "''") + "'")
    return out


def _render_mul_term(node: NormalizedExpr, dialect: Dialect | None) -> str:
    """Render a multiply/divide leaf NormalizedExpr child as a SQL token."""
    if node.raw_sql:
        return node.raw_sql
    if node.star:
        return "*"
    if node.keyword:
        return node.keyword.upper()
    if node.interval is not None:
        val, unit = node.interval
        n = int(val) if float(val).is_integer() else val
        return f"INTERVAL '{n} {unit}'"
    if node.cast_type:
        if node.add_groups and node.add_groups[0].multiply:
            inner_sql = _render_mul_term(node.add_groups[0].multiply[0], dialect)
        else:
            inner_sql = ""
        return f"CAST({inner_sql} AS {node.cast_type})"
    if node.column_ref:
        return _quote_simple_qualified_mul_token(node.column_ref, dialect)
    return render_expr_sql(node, dialect)


def _render_group_sql(g: MulGroup, dialect: Dialect | None = None) -> str:
    """Render a MulGroup as a SQL fragment for expression guide."""
    if not g.multiply:
        return "1"
    if (g.scalar_func or "").lower() == "concat":
        base = ", ".join(_render_mul_term(x, dialect) for x in g.multiply)
    else:
        base = " * ".join(_render_mul_term(x, dialect) for x in g.multiply)
    if g.divide:
        base = f"({base}) / ({' * '.join(_render_mul_term(x, dialect) for x in g.divide)})"
    if g.coeff_param_key:
        base = f":{g.coeff_param_key} * {base}"
    elif g.coefficient != 1.0:
        base = f"{g.coefficient} * {base}"
    if g.inner_scalar_func:
        iargs = _render_scalar_func_args(g.inner_scalar_func, g.inner_scalar_func_args, g.isarg_param_keys)
        args_str = ", ".join(iargs)
        if g.inner_scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and args_str:
            inner = f"{g.inner_scalar_func.upper()}({args_str}, {base})"
        else:
            inner = f"{g.inner_scalar_func.upper()}({base}{', ' + args_str if args_str else ''})"
    else:
        inner = base
    if g.agg_func:
        if g.agg_func == "string_agg":
            sep_sql = f":{g.agg_sep_param_key}" if g.agg_sep_param_key else "','"
            order_sql = _render_order_by_sql(list(g.agg_order_by), dialect) if g.agg_order_by else ""
            if dialect is not None:
                mid = dialect.render_string_agg(inner, sep_sql, order_sql)
            elif order_sql:
                mid = f"STRING_AGG({inner}, {sep_sql} ORDER BY {order_sql})"
            else:
                mid = f"STRING_AGG({inner}, {sep_sql})"
        elif g.agg_func == "stddev":
            mid = f"STDDEV_SAMP({inner})"
        elif g.agg_func == "variance":
            mid = f"VAR_SAMP({inner})"
        elif g.agg_func == "median":
            mid = (
                dialect.render_median(inner)
                if dialect is not None
                else f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {inner})"
            )
        elif g.distinct:
            mid = f"{g.agg_func.upper()}(DISTINCT {inner})"
        else:
            mid = f"{g.agg_func.upper()}({inner})"
    else:
        mid = inner
    if g.scalar_func:
        sargs = _render_scalar_func_args(g.scalar_func, g.scalar_func_args, g.sarg_param_keys)
        args_str = ", ".join(sargs)
        if g.scalar_func.lower() == "extract" and args_str:
            return f"EXTRACT({args_str} FROM {mid})"
        if g.scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and args_str:
            return f"{g.scalar_func.upper()}({args_str}, {mid})"
        return f"{g.scalar_func.upper()}({mid}{', ' + args_str if args_str else ''})"
    return mid


def _try_render_date_integer_day_expr(expr: NormalizedExpr, dialect: Dialect | None) -> str | None:
    """Render date-column +/- integer days when operand types match; skip date-date and numeric-numeric."""
    if dialect is None or expr.agg_func or expr.scalar_func or expr.inner_scalar_func:
        return None
    if expr.add_values and len(expr.add_groups) == 1 and not expr.sub_groups and not expr.sub_values:
        if not _operands_allow_date_integer_days(expr.add_groups[0], None, offset_literal=expr.add_values[0].value):
            return None
        base_sql = _render_group_sql(expr.add_groups[0], dialect)
        return dialect.render_date_integer_days(base_sql, "+", str(int(expr.add_values[0].value)))
    if expr.sub_values and len(expr.add_groups) == 1 and not expr.sub_groups and not expr.add_values:
        if not _operands_allow_date_integer_days(expr.add_groups[0], None, offset_literal=expr.sub_values[0].value):
            return None
        base_sql = _render_group_sql(expr.add_groups[0], dialect)
        return dialect.render_date_integer_days(base_sql, "-", str(int(expr.sub_values[0].value)))
    if len(expr.add_groups) == 2 and not expr.sub_groups and not expr.add_values and not expr.sub_values:
        base_group = expr.add_groups[1]
        offset_group = expr.add_groups[0]
        if not _operands_allow_date_integer_days(base_group, offset_group):
            return None
        offset_sql = _render_group_sql(offset_group, dialect)
        base_sql = _render_group_sql(base_group, dialect)
        return dialect.render_date_integer_days(base_sql, "+", offset_sql)
    if len(expr.add_groups) == 1 and len(expr.sub_groups) == 1 and not expr.add_values and not expr.sub_values:
        base_group = expr.add_groups[0]
        offset_group = expr.sub_groups[0]
        if _operands_are_date_minus_date(base_group, offset_group) or _operands_are_numeric_minus_numeric(
            base_group, offset_group
        ):
            return None
        if not _operands_allow_date_integer_days(base_group, offset_group):
            return None
        base_sql = _render_group_sql(base_group, dialect)
        offset_sql = _render_group_sql(offset_group, dialect)
        return dialect.render_date_integer_days(base_sql, "-", offset_sql)
    return None


def render_expr_sql(expr: NormalizedExpr, dialect: Dialect | None = None) -> str:
    """Render a NormalizedExpr as a SQL fragment for expression guide."""
    ref = expr_registry_ref(expr) or ""
    if ref.startswith("w"):
        win_by = {s.registry_id: s for s in current_window_registry_steps()}
        win_step = win_by.get(ref)
        if win_step is not None:
            return _render_window_over_sql(win_step.window_spec, dialect)
        return "0"
    if ref.startswith("c"):
        case_by = {s.registry_id: s for s in current_case_registry_steps()}
        case_step = case_by.get(ref)
        if case_step is not None:
            return _render_case_when_sql(case_step.case_when, dialect)
        return "0"
    sl = (expr.string_literal or "").strip()
    if sl:
        if dialect is not None:
            return dialect.quote_string_literal(sl)
        esc = sl.replace("'", "''")
        return f"'{esc}'"
    if (
        not expr.add_groups
        and not expr.sub_groups
        and not expr.add_values
        and not expr.sub_values
        and not expr.agg_func
        and not expr.scalar_func
        and not expr.inner_scalar_func
    ):
        if expr.column_ref:
            return _render_mul_term(expr, dialect)
        if expr.star:
            return "*"
        if expr.keyword:
            return expr.keyword.upper()
        if expr.raw_sql:
            return expr.raw_sql
        if expr.cast_type or expr.interval is not None:
            return _render_mul_term(expr, dialect)
    date_int = _try_render_date_integer_day_expr(expr, dialect)
    if date_int is not None:
        return date_int
    parts: list[str] = []
    for g in expr.add_groups:
        parts.append(_render_group_sql(g, dialect))
    for v in expr.add_values:
        parts.append(f":{v.param_key}" if v.param_key else str(v.value))
    sub_parts: list[str] = []
    for g in expr.sub_groups:
        sub_parts.append(_render_group_sql(g, dialect))
    for v in expr.sub_values:
        sub_parts.append(f":{v.param_key}" if v.param_key else str(v.value))
    result = " + ".join(parts) if parts else "0"
    if sub_parts:
        result = f"{result} - {' - '.join(sub_parts)}"
    if expr.inner_scalar_func and not any(g.inner_scalar_func for g in expr.add_groups):
        iargs = _render_scalar_func_args(expr.inner_scalar_func, expr.inner_scalar_func_args, expr.isarg_param_keys)
        args_str = ", ".join(iargs)
        if expr.inner_scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and args_str:
            result = f"{expr.inner_scalar_func.upper()}({args_str}, {result})"
        else:
            result = f"{expr.inner_scalar_func.upper()}({result}{', ' + args_str if args_str else ''})"
    if expr.agg_func and not any(g.agg_func for g in expr.add_groups):
        result = f"{expr.agg_func.upper()}({result})"
    if expr.scalar_func and not any(g.scalar_func for g in expr.add_groups):
        sargs = _render_scalar_func_args(expr.scalar_func, expr.scalar_func_args, expr.sarg_param_keys)
        args_str = ", ".join(sargs)
        if expr.scalar_func.lower() == "extract" and args_str:
            result = f"EXTRACT({args_str} FROM {result})"
        elif expr.scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and args_str:
            result = f"{expr.scalar_func.upper()}({args_str}, {result})"
        else:
            result = f"{expr.scalar_func.upper()}({result}{', ' + args_str if args_str else ''})"
    return result


def classify_cte_emission(cte: RuntimeCteStep, intent: RuntimeIntent, schema: SchemaGraph | None) -> CteEmissionKind:
    """Decide whether a CTE renders as a regular join table or a single- row CROSS JOIN scalar."""
    explicit = getattr(cte, "emission", "join_table") or "join_table"
    if explicit == "semi_join":
        if schema is not None and qualifies_as_semi_join_probe(cte, intent, schema):
            return "semi_join"
    elif explicit == "anti_join":
        return "anti_join"
    if (cte.grain or "") != "scalar":
        return "join_table"
    if cte.limit is not None and cte.limit != 1:
        return "join_table"
    outs = list(cte.output_columns or [])
    if len(outs) != 1:
        return "join_table"
    cn = (cte.cte_name or "").strip()
    if not cn:
        return "join_table"
    base_tables = [t for t in (cte.tables or []) if t]
    if len(base_tables) != 1:
        return "join_table"
    sel_cols = list(cte.select_cols or [])
    if len(sel_cols) != 1:
        return "join_table"
    if not sel_cols[0].is_aggregated:
        return "join_table"
    if cte.group_by_cols:
        return "join_table"
    return "scalar_subquery"


def _render_case_branch_sql(fp: WhereParam, dialect: Dialect | None = None) -> str:
    """Render a single filter as a SQL predicate for CASE WHEN."""
    left = render_expr_sql(fp.left_expr, dialect)
    op = fp.op or "="
    if fp.op in ("is null", "is not null"):
        return f"{left} {op.upper()}"
    if fp.op == "between" and fp.param_key and fp.param_key_hi:
        return f"{left} BETWEEN :{fp.param_key} AND :{fp.param_key_hi}"
    if fp.op in ("in", "not in") and fp.param_key:
        return f"{left} {op.upper()} (:{fp.param_key})"
    if fp.right_expr:
        return f"{left} {op} {render_expr_sql(fp.right_expr, dialect)}"
    if fp.param_key:
        return f"{left} {op} :{fp.param_key}"
    raise ValueError(
        f"CASE branch missing right operand for {left!r} {op!r}: neither right_expr nor param_key set; "
        "upstream parameter allocation must bind a value for branch literals before rendering"
    )


def _render_case_when_sql(cw: CaseWhenExpr, dialect: Dialect | None = None) -> str:
    """Render a CASE expression for SELECT."""
    parts: list[str] = ["CASE"]
    for br in cw.branches:
        parts.append(
            f"WHEN {_render_case_branch_sql(br.condition, dialect)} THEN {render_expr_sql(br.result, dialect)}"
        )
    if cw.else_result is not None:
        parts.append(f"ELSE {render_expr_sql(cw.else_result, dialect)}")
    parts.append("END")
    return " ".join(parts)


def _window_bound_sql(kind: str, offset: int | None) -> str:
    """Map a window frame bound token to SQL text."""
    k = (kind or "current_row").strip().lower()
    if k == "unbounded_preceding":
        return "UNBOUNDED PRECEDING"
    if k == "unbounded_following":
        return "UNBOUNDED FOLLOWING"
    if k == "current_row":
        return "CURRENT ROW"
    if k == "n_preceding":
        return f"{int(offset or 0)} PRECEDING"
    if k == "n_following":
        return f"{int(offset or 0)} FOLLOWING"
    return "CURRENT ROW"


def _render_window_over_sql(ws: WindowSpec, dialect: Dialect | None = None) -> str:
    """Render a window aggregate or function with ``OVER (PARTITION BY. ... ORDER BY ...)``."""
    fn = SQL_WINDOW_FUNCTION_UPPER.get(ws.function, ws.function.upper())
    if ws.function in ("sum", "avg"):
        arg_sql = render_expr_sql(ws.argument, dialect) if ws.argument else "*"
        arg_sql = _maybe_databricks_unqualify_window_sql_frag(arg_sql, dialect)
        core = f"{fn}({arg_sql})"
    elif ws.function in ("lag", "lead", "first_value", "last_value", "nth_value"):
        if ws.argument:
            arg_sql = render_expr_sql(ws.argument, dialect)
        elif ws.function == "nth_value" and ws.partition_by:
            arg_sql = render_expr_sql(ws.partition_by[0], dialect)
        else:
            arg_sql = "*" if ws.function != "nth_value" else "NULL"
        arg_sql = _maybe_databricks_unqualify_window_sql_frag(arg_sql, dialect)
        if ws.function == "nth_value":
            core = f"{fn}({arg_sql}, {int(ws.numeric_argument or 0)})"
        else:
            core = f"{fn}({arg_sql})"
    elif ws.function == "ntile":
        core = f"{fn}({int(ws.numeric_argument or 0)})"
    elif ws.function in ("percent_rank", "cume_dist"):
        core = f"{fn}()"
    else:
        core = f"{fn}()"
    over_parts: list[str] = []
    if ws.partition_by:
        pe = ", ".join(render_expr_sql(e, dialect) for e in ws.partition_by)
        over_parts.append(f"PARTITION BY {pe}")
    if ws.order_by:
        over_parts.append(f"ORDER BY {_render_order_by_sql(list(ws.order_by), dialect)}")
    inner = " ".join(over_parts)
    if ws.frame_kind in ("rows", "range") and ws.order_by:
        fk = "ROWS" if ws.frame_kind == "rows" else "RANGE"
        fs = ws.frame_start or "unbounded_preceding"
        fe = ws.frame_end or "current_row"
        inner = (
            f"{inner} {fk} BETWEEN {_window_bound_sql(fs, ws.frame_start_offset)} "
            f"AND {_window_bound_sql(fe, ws.frame_end_offset)}"
        )
    return f"{core} OVER ({inner})"


def render_select_col_sql(sc: SelectCol, dialect: Dialect | None = None) -> str:
    """Render a select column including optional CASE or window. function."""
    parts = effective_select_parts(sc, None, None)
    if parts.case_when is not None:
        return _render_case_when_sql(parts.case_when, dialect)
    if parts.window_spec is not None:
        return _render_window_over_sql(parts.window_spec, dialect)
    return render_expr_sql(parts.expr, dialect)


def _driver_table_from_join_path_signature(signature: list[str]) -> str | None:
    """Return the left-hand table name from the first valid join path segment."""
    for seg in signature:
        seg_stripped = seg.strip()
        if "->" not in seg_stripped:
            continue
        left_part = seg_stripped.split("->", 1)[0].strip()
        if "." not in left_part:
            continue
        left_tbl = left_part.split(".", 1)[0].strip()
        if left_tbl:
            return left_tbl
    return None


def _intent_table_spelling(name: str, tables: list[str]) -> str | None:
    """Resolve *name* to the spelling used in *tables* when case- insensitively equal."""
    for t in tables:
        if t.lower() == name.lower():
            return t
    return None


def _first_anchor_table_from_group_by_cols(group_by_cols: list[NormalizedExpr] | None, tables: list[str]) -> str | None:
    """Return the first group-by column table that appears in *tables*, or ``None``."""
    for expr in group_by_cols or []:
        col = getattr(expr, "primary_column", "") or ""
        if "." not in col:
            continue
        raw_tbl = col.split(".", 1)[0].strip()
        resolved = _intent_table_spelling(raw_tbl, tables)
        if resolved:
            return resolved
    return None


def _first_anchor_table_from_order_by_cols(order_by_cols: list[Any], tables: list[str]) -> str | None:
    """Return the first order-by column table that appears in *tables*, or ``None``."""
    for obc in order_by_cols or []:
        expr = getattr(obc, "expr", None)
        col = getattr(expr, "primary_column", "") or "" if expr is not None else ""
        if "." not in col:
            continue
        raw_tbl = col.split(".", 1)[0].strip()
        resolved = _intent_table_spelling(raw_tbl, tables)
        if resolved:
            return resolved
    return None


def _from_anchor_for_multi_table_block(
    tables: list[str],
    order_by_cols: list[Any],
    schema: SchemaGraph | None,
    join_signature: list[str] | None,
    *,
    grain: str | None = None,
    group_by_cols: list[NormalizedExpr] | None = None,
    probe_cte_names: frozenset[str] | None = None,
) -> str | None:
    """Pick ``FROM`` for a multi-table block. Precedence: grouped grain uses the first group-by table in scope; else the first order-by table in scope; else the join-path signature driver; else row-count heuristic. Semi-join and anti-join probe CTEs are never chosen as anchor."""
    if not tables or len(tables) <= 1:
        return None
    probe_names = probe_cte_names or frozenset()
    anchor_pool = [t for t in tables if t not in probe_names]
    work_tables = anchor_pool if anchor_pool else list(tables)
    sig_list = [s for s in (join_signature or []) if s]
    if grain == "grouped":
        gb_anchor = _first_anchor_table_from_group_by_cols(group_by_cols, work_tables)
        if gb_anchor:
            return gb_anchor
    ob_anchor = _first_anchor_table_from_order_by_cols(list(order_by_cols) if order_by_cols else [], work_tables)
    if ob_anchor:
        return ob_anchor
    driver_raw = _driver_table_from_join_path_signature(sig_list) if sig_list else None
    resolved = _intent_table_spelling(driver_raw, work_tables) if driver_raw else None
    if resolved:
        return resolved
    return _deterministic_from_anchor_table(work_tables, order_by_cols, schema)


def _deterministic_from_anchor_table(
    tables: list[str], order_by_cols: list[OrderByCol], schema: SchemaGraph | None
) -> str | None:
    """Pick a ``FROM`` anchor for multi-table row-level blocks without reordering *tables* list."""
    if not tables or len(tables) <= 1:
        return None
    for obc in order_by_cols or []:
        col = getattr(obc.expr, "primary_column", "") or ""
        if "." in col:
            prefix = col.split(".", 1)[0].strip()
            if prefix in tables:
                return prefix
            for t in tables:
                if t.lower() == prefix.lower():
                    return t
    if schema is None or not getattr(schema, "tables", None):
        return None
    known_lower = {k.lower() for k in schema.tables}
    physical = [t for t in tables if t.lower() in known_lower]
    if not physical:
        return None
    ranked: list[tuple[int, str]] = []
    for raw in physical:
        row_count = 0
        for key, tinfo in schema.tables.items():
            if key.lower() == raw.lower():
                row_count = getattr(tinfo, "row_count", 0) or 0
                break
        ranked.append((row_count, raw))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1] if ranked else None


def render_feedback_sql(intent: RuntimeIntent, schema: SchemaGraph | None) -> str | None:
    """Render dialect-neutral SQL for failure-summary LLM payloads only; never persisted."""
    if schema is None:
        return None
    try:
        return build_deterministic_sql(intent, schema=schema, dialect=get_dialect(CANONICAL_FEEDBACK_DIALECT))
    except Exception:
        return None


def _anti_join_presence_where(tables: list[str], cte_emissions: dict[str, CteEmissionKind]) -> list[WhereParam]:
    """Build renderer-owned ``IS NULL`` predicates for anti-join probe CTEs in *tables*."""
    filters: list[WhereParam] = []
    for tbl in tables or []:
        if cte_emissions.get(tbl) != "anti_join":
            continue
        presence = anti_join_presence_column(tbl)
        filters.append(WhereParam(left_expr=NormalizedExpr.from_column(f"{tbl}.{presence}"), op="is null"))
    return filters


def _aggregate_needs_zero_fill(expr: NormalizedExpr, preserved_tables: set[str]) -> bool:
    """Return True when a count/sum aggregate references a non-preserved table while preservation is active."""
    if not preserved_tables:
        return False
    agg = (expr.agg_func or "").lower()
    if agg not in ("count", "sum"):
        return False
    for col in extract_columns_from_expr(expr):
        if "." not in col:
            continue
        tbl = col.split(".", 1)[0]
        if _phys_table_key(tbl) not in preserved_tables:
            return True
    return False


def _foreign_key_relates_columns(
    schema: SchemaGraph,
    src_table: str,
    src_col: str,
    dst_table: str,
    dst_col: str,
) -> bool:
    """Return True when a foreign-key edge already joins the two column endpoints."""
    pairs = {
        (src_table, src_col, dst_table, dst_col),
        (dst_table, dst_col, src_table, src_col),
    }
    for tbl_name, tbl in schema.tables.items():
        for fk in tbl.foreign_keys:
            for sc, dc in zip(fk.src_cols, fk.dst_cols, strict=False):
                if (tbl_name, sc, fk.dst_table, dc) in pairs:
                    return True
    return False


def emit_semantic_profile_where_diagnostics(
    schema: SchemaGraph | None,
    *,
    where_segments: list[tuple[int, str, str, list[str], list[str]]],
    edge_kinds: list[str],
) -> None:
    """Surface SEMANTIC_PROFILE_WHERE_EDGE when profile overlap renders as a WHERE equality."""
    if schema is None or not where_segments:
        return
    for sig_i, left_tbl, right_tbl, lcols, rcols in where_segments:
        kind = edge_kinds[sig_i] if sig_i < len(edge_kinds) else ""
        if kind not in JOIN_PATH_EDGE_KIND_WHERE_BUCKET:
            continue
        left_meta = schema.tables.get(left_tbl)
        right_meta = schema.tables.get(right_tbl)
        if left_meta is None or right_meta is None:
            continue
        for lc, rc in zip(lcols, rcols, strict=False):
            left_col = left_meta.columns.get(lc)
            right_col = right_meta.columns.get(rc)
            if left_col is None or right_col is None:
                continue
            if _foreign_key_relates_columns(schema, left_tbl, lc, right_tbl, rc):
                continue
            stats = value_overlap_stats_for_columns(left_col, right_col)
            if stats is None:
                inter, ratio = 0, 0.0
            else:
                inter, ratio = stats
            notify(
                (
                    f"Inferred relationship rendered as a WHERE filter: {left_tbl}.{lc} = "
                    f"{right_tbl}.{rc}. No foreign key joins these columns "
                    f"(overlap intersection {inter}, ratio {ratio:.0%}). "
                    "Declare the relationship with foreign_keys_add when it is structural, "
                    "or with a semantic override when it is intentional but not a foreign key."
                ),
                stage="join",
                code=DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE,
                level="info",
                details=(
                    ("left", f"{left_tbl}.{lc}"),
                    ("right", f"{right_tbl}.{rc}"),
                    ("intersection", str(inter)),
                    ("overlap_ratio", f"{ratio:.4f}"),
                ),
            )


def inner_equality_pairs_from_resolved_join_path(
    signature: list[str],
    edge_kinds: list[str],
    from_anchor: str,
    schema: SchemaGraph | None,
    *,
    preserve_tables: list[str] | None = None,
    cte_emissions: dict[str, CteEmissionKind] | None = None,
    probe_cte_names: frozenset[str] | None = None,
) -> set[frozenset[str]]:
    """Return unordered ``table.column`` pairs rendered as INNER ``ON`` equalities for one resolved scope."""
    if not signature or not from_anchor or not edge_kinds:
        return set()
    if len(edge_kinds) < len(signature):
        return set()
    join_segments, _where_segments = _partition_path_join_vs_where(signature, edge_kinds)
    if not join_segments:
        return set()
    preserved_roots = _normalized_table_name_set(preserve_tables)
    pairs: set[frozenset[str]] = set()
    for _idx, left_tbl, right_tbl, lcols, rcols in join_segments:
        if len(lcols) != 1 or len(rcols) != 1:
            continue
        if _phys_table_key(left_tbl) in preserved_roots or _phys_table_key(right_tbl) in preserved_roots:
            continue
        left_col = f"{left_tbl}.{lcols[0]}"
        right_col = f"{right_tbl}.{rcols[0]}"
        join_emission = (cte_emissions or {}).get(left_tbl) or (cte_emissions or {}).get(right_tbl)
        if probe_cte_names and (
            _table_in_set(left_tbl, set(probe_cte_names)) or _table_in_set(right_tbl, set(probe_cte_names))
        ):
            continue
        if join_emission in ("anti_join", "semi_join"):
            continue
        fk_left, fk_null_left = _edge_fk_points_to_paired(left_tbl, right_tbl, lcols, schema)
        if fk_left:
            if fk_null_left:
                continue
            pairs.add(frozenset({left_col, right_col}))
            continue
        fk_right, fk_null_right = _edge_fk_points_to_paired(right_tbl, left_tbl, rcols, schema)
        if fk_right:
            if fk_null_right:
                continue
            pairs.add(frozenset({left_col, right_col}))
            continue
        pairs.add(frozenset({left_col, right_col}))
    return pairs


def emit_join_orphan_rate_diagnostics(
    intent: RuntimeIntent,
    schema: SchemaGraph | None,
    *,
    join_signature: list[str] | None = None,
    edge_kinds: list[str] | None = None,
    from_anchor: str | None = None,
    preserve_tables: list[str] | None = None,
) -> None:
    """Surface JOIN_ORPHAN_RATE_HIGH when an ambiguous edge uses INNER and orphan rate exceeds the floor."""
    if schema is None or preserve_tables:
        return
    sig = list(join_signature or intent.chosen_join_path_signature or [])
    if not sig or not from_anchor:
        return
    sig = canonicalize_stored_join_path_signature(sig, from_anchor=from_anchor)
    kinds = list(edge_kinds or [])
    join_segments, _where_segments = _partition_path_join_vs_where(sig, kinds)
    anchor_key = _phys_table_key(from_anchor)
    phys_instances: set[str] = {anchor_key}
    for _idx, left_tbl, right_tbl, lcols, rcols in join_segments:
        lk = _phys_table_key(left_tbl)
        rk = _phys_table_key(right_tbl)
        li = lk in phys_instances
        ri = rk in phys_instances
        if li and ri:
            continue
        if li:
            join_tbl, paired_tbl, cols_on_join = right_tbl, left_tbl, rcols
            left_token = left_tbl
        elif ri:
            join_tbl, paired_tbl, cols_on_join = left_tbl, right_tbl, lcols
            left_token = left_tbl
        else:
            continue
        if not _edge_is_one_to_many_ambiguous(join_tbl, paired_tbl, cols_on_join, schema):
            phys_instances.add(_phys_table_key(join_tbl))
            continue
        kind = _join_kind_for_edge(join_tbl, paired_tbl, cols_on_join, schema).strip().upper()
        if kind != "INNER":
            phys_instances.add(_phys_table_key(join_tbl))
            continue
        orphan_rate = _profiled_orphan_rate_for_edge(left_token, join_tbl, cols_on_join, schema)
        if orphan_rate is None or orphan_rate <= JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR:
            phys_instances.add(_phys_table_key(join_tbl))
            continue
        notify(
            (
                f"Entities with no match on edge {left_token}->{join_tbl} were excluded "
                f"(profiled orphan rate {orphan_rate:.0%} exceeds "
                f"{JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR:.0%} floor). "
                "Consider preserve_tables if empty entities should appear."
            ),
            stage="join",
            code=DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH,
            level="warning",
            details=(("left_table", left_token), ("right_table", join_tbl), ("orphan_rate", f"{orphan_rate:.4f}")),
        )
        phys_instances.add(_phys_table_key(join_tbl))


def _raise_clause_widened_rowset_errors(
    intent: RuntimeIntent,
    schema: SchemaGraph | None,
    *,
    join_signature: list[str] | None = None,
    from_anchor: str | None = None,
    context: str = "main query",
) -> None:
    """Refuse LIMIT / DISTINCT ON when the resolved join path widens the anchor row set."""
    if schema is None:
        return
    issues = validate_clause_widened_rowset(
        intent,
        schema,
        context,
        join_signature=join_signature,
        from_anchor=from_anchor,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise ClauseWidenedRowsetError(context, errors[0].message)


def build_deterministic_sql(
    intent: RuntimeIntent,
    cte_join_hints: dict[str, dict[str, Any]] | None = None,
    schema: SchemaGraph | None = None,
    dialect: Dialect | None = None,
    join_signature_for_from_anchor: list[str] | None = None,
    cte_join_signatures_for_from_anchor: dict[str, list[str]] | None = None,
) -> str:
    """Build a rough deterministic SQL from a RuntimeIntent."""
    keep_cte = {s.cte_name.lower() for s in (intent.cte_steps or []) if s.cte_name}
    if cte_join_hints:
        cte_join_hints = {k: v for k, v in cte_join_hints.items() if str(k).strip().lower() in keep_cte}
    if cte_join_signatures_for_from_anchor:
        cte_join_signatures_for_from_anchor = {
            k: v for k, v in cte_join_signatures_for_from_anchor.items() if str(k).strip().lower() in keep_cte
        }
    if dialect is None:
        dialect = get_dialect()
    effective_main_sig = join_signature_for_from_anchor
    if not effective_main_sig and intent.chosen_join_path_signature:
        effective_main_sig = intent.chosen_join_path_signature
    main_anchor = _from_anchor_for_multi_table_block(
        intent.tables or [],
        list(intent.order_by_cols or []),
        schema,
        list(effective_main_sig) if effective_main_sig else None,
        grain=intent.grain,
        group_by_cols=list(intent.group_by_cols or []),
        probe_cte_names=probe_cte_names(intent.cte_steps),
    )
    _raise_clause_widened_rowset_errors(
        intent,
        schema,
        join_signature=list(effective_main_sig) if effective_main_sig else None,
        from_anchor=main_anchor,
        context="main query",
    )
    parts: list[str] = []

    cte_steps = intent.cte_steps or []
    cte_emissions = cte_emission_map(cte_steps)
    probe_names = probe_cte_names(cte_steps)
    scalar_cte_names: set[str] = {
        (cte.cte_name or "")
        for cte in cte_steps
        if getattr(cte, "emission", "join_table") == "scalar_subquery" and cte.cte_name
    }

    def _scalar_extras_for_scope(scope_tables: list[str] | None, anchor: str | None) -> list[str]:
        if not scope_tables or not scalar_cte_names:
            return []
        return [t for t in scope_tables if t in scalar_cte_names and t != anchor]

    cte_clauses: list[str] = []
    for cte in cte_steps:
        emission = cte_emissions.get(cte.cte_name or "", "join_table")
        cte_sig = None
        if cte_join_signatures_for_from_anchor and cte.cte_name:
            cte_sig = cte_join_signatures_for_from_anchor.get(cte.cte_name)
        cte_anchor = _from_anchor_for_multi_table_block(
            cte.tables or [],
            list(cte.order_by_cols or []),
            schema,
            cte_sig,
            grain=cte.grain,
            group_by_cols=list(cte.group_by_cols or []),
            probe_cte_names=probe_names,
        )
        cte_extras = _scalar_extras_for_scope(cte.tables or [], cte_anchor)
        distinct_idx = cte.distinct_select_index
        append_select: list[str] = []
        if emission in PROBE_CTE_EMISSION_KINDS and (cte.select_cols or []):
            distinct_idx = 0
        if emission == "anti_join" and cte.cte_name:
            presence = anti_join_presence_column(cte.cte_name)
            append_select.append(f"1 AS {presence}")
        with registry_render_scope(cte.window_registry, cte.case_registry):
            cte_where = cte.where
            cte_having = cte.having
            cte_distinct_on = list(cte.distinct_on or [])
            cte_order = list(cte.order_by_cols or [])
            cte_limit = cte.limit
            cte_select_exprs = _render_select_column_exprs(
                cte.select_cols or [],
                dialect,
                cte.output_columns or [],
                schema=schema,
                for_cte=True,
                append_select_sql=append_select or None,
                preserve_tables=list(cte.preserve_tables or []),
            )
            cte_sql = _build_deterministic_select_block(
                cte.select_cols or [],
                cte.tables or [],
                cte.group_by_cols or [],
                [] if cte_distinct_on else cte_order,
                cte_where,
                cte_having,
                None if cte_distinct_on else cte_limit,
                cte.grain or "row_level",
                dialect,
                cte.output_columns or [],
                schema=schema,
                for_cte=True,
                from_table_override=cte_anchor,
                extra_from_tables=cte_extras or None,
                distinct_select_index=distinct_idx,
                param_values=cte.param_values,
                append_select_sql=append_select or None,
                preserve_tables=list(cte.preserve_tables or []),
            )
            if cte_distinct_on:
                cte_sql = wrap_core_sql_with_distinct_on(
                    cte_sql,
                    select_exprs=cte_select_exprs,
                    distinct_on=cte_distinct_on,
                    order_by_cols=cte_order,
                    limit=cte_limit,
                    dialect=dialect,
                )
        cte_clauses.append(f"{cte.cte_name} AS (\n{cte_sql}\n)")
    if cte_clauses:
        parts.append("WITH " + ",\n".join(cte_clauses))

    effective_main_sig = join_signature_for_from_anchor
    if not effective_main_sig and intent.chosen_join_path_signature:
        effective_main_sig = intent.chosen_join_path_signature
    main_anchor = _from_anchor_for_multi_table_block(
        intent.tables or [],
        list(intent.order_by_cols or []),
        schema,
        list(effective_main_sig) if effective_main_sig else None,
        grain=intent.grain,
        group_by_cols=list(intent.group_by_cols or []),
        probe_cte_names=probe_names,
    )
    main_extras = _scalar_extras_for_scope(intent.tables or [], main_anchor)
    main_where = intent.where
    extra_filters = _anti_join_presence_where(intent.tables or [], cte_emissions)
    if extra_filters:
        main_where = merge_predicate_groups("and", [main_where, predicate_group_from_list(extra_filters)])
    main_distinct_idx = intent.distinct_select_index
    if any(cte_emissions.get(t) == "anti_join" for t in (intent.tables or [])):
        if (intent.select_cols or []) and main_distinct_idx < 0:
            main_distinct_idx = 0
    main_distinct_on = list(intent.distinct_on or [])
    main_order = list(intent.order_by_cols or [])
    main_limit = intent.limit
    with registry_render_scope(intent.window_registry, intent.case_registry):
        main_select_exprs = _render_select_column_exprs(
            intent.select_cols or [],
            dialect,
            None,
            schema=schema,
            for_cte=False,
            preserve_tables=list(intent.preserve_tables or []),
        )
        main_sql = _build_deterministic_select_block(
            intent.select_cols or [],
            intent.tables or [],
            intent.group_by_cols or [],
            [] if main_distinct_on else main_order,
            main_where,
            intent.having,
            None if main_distinct_on else main_limit,
            intent.grain or "row_level",
            dialect,
            schema=schema,
            for_cte=False,
            from_table_override=main_anchor,
            extra_from_tables=main_extras or None,
            distinct_select_index=main_distinct_idx,
            param_values=intent.param_values,
            preserve_tables=list(intent.preserve_tables or []),
        )
        if main_distinct_on:
            reserved = {c.cte_name for c in cte_steps if c.cte_name}
            reserved.update(intent.planner_cte_names or [])
            don_name = _allocate_distinct_on_cte_name(reserved)
            inner_sql = main_sql
            outer_lines = [
                f"SELECT {', '.join(main_select_exprs)}",
                f"FROM {dialect.quote_schema_qualified(don_name)}",
                f"WHERE {DISTINCT_ON_RANK_COLUMN} = 1",
            ]
            if main_order:
                outer_lines.append(f"ORDER BY {_render_order_by_sql(main_order, dialect)}")
            if main_limit:
                outer_lines.append(f"LIMIT {main_limit}")
            main_sql = "\n".join(outer_lines)
            rank_partition = ", ".join(render_expr_sql(expr, dialect) for expr in main_distinct_on)
            rank_order = _render_order_by_sql(main_order, dialect)
            rank_expr = (
                f"ROW_NUMBER() OVER (PARTITION BY {rank_partition} ORDER BY {rank_order}) AS {DISTINCT_ON_RANK_COLUMN}"
            )
            inner_lines = inner_sql.split("\n")
            inner_lines[0] = inner_lines[0] + ", " + rank_expr
            don_body = "\n".join(inner_lines)
            if cte_clauses:
                parts[-1] = parts[-1] + f",\n{don_name} AS (\n{don_body}\n)"
            else:
                parts.insert(0, f"WITH {don_name} AS (\n{don_body}\n)")
    parts.append(main_sql)

    return "\n".join(parts)


def _parse_select_list_expression(expr_sql: str, sqlglot_dialect: str) -> exp.Expression | None:
    """Parse one SELECT-list expression via sqlglot."""
    try:
        tree = parse_one(f"SELECT {expr_sql}", dialect=sqlglot_dialect or None)
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    exprs = list(tree.args.get("expressions") or [])
    if len(exprs) != 1:
        return None
    return exprs[0]


def _parse_predicate_fragment(frag: str, *, sqlglot_dialect: str = "") -> exp.Expression | None:
    """Parse a bare predicate fragment into a sqlglot boolean expression."""
    try:
        tree = parse_one(f"SELECT 1 WHERE {frag}", dialect=sqlglot_dialect or None)
    except Exception:
        return None
    where = tree.args.get("where")
    if where is None:
        return None
    return where.this


def _emit_bool_ast_preserving_leaves(node: exp.Expression, leaves: dict[int, str]) -> str:
    """Render an ``exp.And`` / ``exp.Or`` tree using original leaf SQL from *leaves*."""
    if isinstance(node, exp.And):
        left = _emit_bool_ast_preserving_leaves(node.this, leaves)
        right = _emit_bool_ast_preserving_leaves(node.expression, leaves)
        return f"({left} AND {right})"
    if isinstance(node, exp.Or):
        left = _emit_bool_ast_preserving_leaves(node.this, leaves)
        right = _emit_bool_ast_preserving_leaves(node.expression, leaves)
        return f"({left} OR {right})"
    return leaves[id(node)]


def _chain_bool_sql_fragments(fragments: Sequence[str], connectors: Sequence[str], *, sqlglot_dialect: str = "") -> str:
    """Left-associate *fragments* with per-link *connectors* using ``exp.And`` / ``exp.Or``."""
    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]
    normalized_connectors: list[str] = []
    for i in range(1, len(fragments)):
        raw = connectors[i - 1] if i - 1 < len(connectors) else "AND"
        connector = (raw or "AND").strip().upper()
        if connector not in ("AND", "OR"):
            connector = "AND"
        normalized_connectors.append(connector)
    for frag in fragments:
        if _parse_predicate_fragment(frag, sqlglot_dialect=sqlglot_dialect) is None:
            raise ValueError(f"invalid predicate fragment: {frag!r}")
    if len(set(normalized_connectors)) == 1:
        return f" {normalized_connectors[0]} ".join(fragments)
    leaves: dict[int, str] = {}
    head = _parse_predicate_fragment(fragments[0], sqlglot_dialect=sqlglot_dialect)
    assert head is not None
    leaves[id(head)] = fragments[0]
    node: exp.Expression = head
    for i in range(1, len(fragments)):
        nxt = _parse_predicate_fragment(fragments[i], sqlglot_dialect=sqlglot_dialect)
        assert nxt is not None
        leaves[id(nxt)] = fragments[i]
        cls = exp.And if normalized_connectors[i - 1] == "AND" else exp.Or
        node = cls(this=node, expression=nxt)
    return _emit_bool_ast_preserving_leaves(node, leaves)


def _append_select_expressions(parsed: Any, expr_sqls: Sequence[str], dialect: Dialect) -> bool:
    """Append parsed SELECT-list expressions onto a dialect-parsed SELECT handle."""
    sqlglot_dialect = getattr(dialect, "sqlglot_dialect", "") or ""
    if isinstance(parsed, exp.Select):
        appended: list[exp.Expression] = []
        for expr_sql in expr_sqls:
            node = _parse_select_list_expression(expr_sql, sqlglot_dialect)
            if node is None:
                return False
            appended.append(node)
        existing = list(parsed.args.get("expressions") or [])
        parsed.set("expressions", existing + appended)
        return True
    root = getattr(getattr(parsed, "root", None), "stmt", None)
    if root is not None and type(root).__name__ == "SelectStmt":
        return append_pglast_select_targets(root, expr_sqls)
    return False


def _render_clause_chain_inner(parts: Sequence[tuple[str, str]], *, sqlglot_dialect: str = "") -> str:
    """Join ``(fragment, bool_op)`` tuples using each tuple's ``bool_op`` as the connector after that fragment."""
    if not parts:
        return ""
    fragments = [frag for frag, _ in parts]
    connectors = [op for _, op in parts[:-1]]
    return _chain_bool_sql_fragments(fragments, connectors, sqlglot_dialect=sqlglot_dialect)


def _render_clause_chain(parts: list[tuple[str, str]], *, sqlglot_dialect: str = "") -> str:
    """Join clause fragments and parenthesize the full chain when any. forward link is ``OR``."""
    if not parts:
        return ""
    inner = _render_clause_chain_inner(parts, sqlglot_dialect=sqlglot_dialect)
    has_or = any((op or "AND").strip().upper() == "OR" for _, op in parts[:-1])
    if has_or and len(parts) > 1:
        return f"({inner})"
    return inner


def _render_predicate_group_sql(group: PredicateGroup | None, render_leaf: Any, *, sqlglot_dialect: str = "") -> str:
    """Render a :class:`PredicateGroup` tree into SQL boolean text."""
    if group is None or group.is_empty():
        return ""
    pieces: list[str] = []
    for pred in group.predicates:
        frag = render_leaf(pred)
        if frag:
            pieces.append(frag)
    for child in group.groups:
        child_sql = _render_predicate_group_sql(child, render_leaf, sqlglot_dialect=sqlglot_dialect)
        if child_sql:
            pieces.append(f"({child_sql})")
    if not pieces:
        return ""
    if len(pieces) == 1:
        return pieces[0]
    connector = "AND" if group.op == "and" else "OR"
    return _chain_bool_sql_fragments(pieces, [connector] * (len(pieces) - 1), sqlglot_dialect=sqlglot_dialect)


def _legacy_render_predicate_group_sql(parts: list[tuple[str, str, int | None]]) -> str:
    """Deprecated tuple renderer kept for transitional callers."""
    if not parts:
        return ""
    flat = [(frag, bop) for frag, bop, _ in parts]
    return _render_clause_chain(flat)


def _join_clause_parts_with_bool_op(parts: list[tuple[str, str]]) -> str:
    """Chain SQL clause fragments using forward ``bool_op`` links; delegates to :func:`_render_clause_chain`."""
    return _render_clause_chain(parts)


def _maybe_render_array_unnest_select(
    sc: SelectCol,
    schema: SchemaGraph | None,
    cte_outputs: dict[str, Any],
    dialect: Dialect,
    output_aliases: list[str] | None,
    idx: int,
    *,
    for_cte: bool,
) -> str | None:
    """When building a CTE, expand bare array columns via. UNNEST/EXPLODE."""
    if not for_cte or schema is None:
        return None
    if not dialect.supports_unnest_select_item:
        return None
    parts = effective_select_parts(sc, None, None)
    if parts.window_spec or parts.case_when or sc.is_aggregated:
        return None
    col = parts.expr.primary_column
    if not col or "." not in col:
        return None
    meta = get_col_meta(col, schema, cte_outputs)
    if meta is None or not getattr(meta, "element_type", None):
        return None
    col_sql = render_expr_sql(parts.expr, dialect)
    alias = output_aliases[idx] if output_aliases and idx < len(output_aliases) else f"unnest_{idx}"
    return dialect.render_array_unnest(col_sql, alias)


def _column_meta_for_predicate(
    pred: WhereParam | HavingParam, schema: SchemaGraph | None, cte_outputs: dict[str, Any] | None = None
) -> Any | None:
    """Resolve column metadata when the predicate left side is a single qualified column."""
    if schema is None:
        return None
    cols = extract_columns_from_expr(pred.left_expr)
    if len(cols) != 1:
        return None
    return get_col_meta(cols[0], schema, cte_outputs or {})


def _render_predicate_clause(
    pred: WhereParam | HavingParam,
    dialect: Dialect,
    *,
    is_having: bool = False,
    param_values: Mapping[str, Any] | None = None,
    schema: SchemaGraph | None = None,
    cte_outputs: dict[str, Any] | None = None,
) -> str:
    """Render one WHERE or HAVING predicate into SQL text."""
    left = render_expr_sql(pred.left_expr, dialect)
    op = pred.op or (">" if is_having else "=")
    op_cmp = (pred.op or "").strip().lower()
    relational_cmp = op_cmp in (">", "<", ">=", "<=", "=", "!=", "<>")
    case_insensitive = pred.value_type == "string" and op_cmp not in (
        "is null",
        "is not null",
        "ilike",
        "not ilike",
        "contains",
    )
    if pred.right_expr is not None and relational_cmp:
        case_insensitive = False
    if case_insensitive:
        left = _wrap_for_case_insensitive(left, dialect)
    if op.lower() in ("is null", "is not null"):
        return f"{left} {op.upper()}"
    resolved = pred.resolved_value(param_values)
    if pred.value_type == "date_window" and isinstance(resolved, dict):
        parts = _render_date_window_predicate(pred, left, dialect, param_values=param_values)
        return " AND ".join(parts) if parts else ""
    if pred.value_type == "date_diff" and isinstance(resolved, dict):
        rv = resolved
        unit = rv.get("unit", "day")
        amount = int(rv.get("amount", 0)) if rv.get("amount") is not None else 0
        hop = pred.op or ">"
        add_parts = [_render_group_sql(g, dialect) for g in pred.left_expr.add_groups]
        sub_parts = [_render_group_sql(g, dialect) for g in pred.left_expr.sub_groups]
        if pred.right_expr is not None and not add_parts and not sub_parts:
            left_sql = render_expr_sql(pred.left_expr, dialect)
            right_sql = render_expr_sql(pred.right_expr, dialect)
            minuend_sql = right_sql
            subtrahend_sql = left_sql
            if hop in (">", ">="):
                minuend_sql, subtrahend_sql = right_sql, left_sql
            elif hop in ("<", "<="):
                minuend_sql, subtrahend_sql = left_sql, right_sql
            frag = dialect.render_date_diff(
                left_sql, hop, unit, amount, minuend_sql=minuend_sql, subtrahend_sql=subtrahend_sql
            )
        else:
            frag = dialect.render_date_diff(
                left,
                hop,
                unit,
                amount,
                minuend_sql=add_parts[0] if add_parts else "",
                subtrahend_sql=sub_parts[0] if sub_parts else "",
            )
        return frag
    if (pred.op or "").strip().lower() == "contains":
        if not pred.param_key:
            return ""
        pk = pred.param_key
        column_meta = _column_meta_for_predicate(pred, schema, cte_outputs)
        return dialect.render_array_contains(left, pk, column_meta=column_meta)
    if op_cmp in ("ilike", "not ilike") and not dialect.supports_ilike:
        like_op = "LIKE" if op_cmp == "ilike" else "NOT LIKE"
        left_wrapped = _wrap_for_case_insensitive(left, dialect)
        if pred.param_key:
            return f"{left_wrapped} {like_op} LOWER(:{pred.param_key})"
        if pred.raw_value is not None:
            escaped = str(pred.raw_value).replace("'", "''")
            return f"{left_wrapped} {like_op} LOWER('{escaped}')"
    if pred.right_expr:
        right = render_expr_sql(pred.right_expr, dialect)
        cmp_op = op
        op_lower = (pred.op or "").strip().lower()
        if op_lower == "in":
            cmp_op = "="
        elif op_lower == "not in":
            cmp_op = "<>"
        if case_insensitive:
            right = _wrap_for_case_insensitive(right, dialect)
        return f"{left} {cmp_op} {right}"
    if (pred.op or "").lower() in ("in", "not in"):
        if not pred.param_key:
            return ""
        return f"{left} {op.upper()} (:{pred.param_key})"
    if pred.param_key:
        val_needs_lower = case_insensitive and op.lower() in ("like", "not like")
        val_ref = f"LOWER(:{pred.param_key})" if val_needs_lower else f":{pred.param_key}"
        return f"{left} {op} {val_ref}"
    if pred.raw_value is not None:
        return ""
    if is_having:
        return f"{left} {op} ?"
    return ""


def _render_select_column_exprs(
    select_cols: list[SelectCol],
    dialect: Dialect,
    output_aliases: list[str] | None,
    *,
    schema: SchemaGraph | None = None,
    for_cte: bool = False,
    append_select_sql: list[str] | None = None,
    preserve_tables: list[str] | None = None,
) -> list[str]:
    """Render SELECT list expressions for deterministic SQL blocks."""
    select_exprs: list[str] = []
    cte_outputs: dict[str, Any] = {}
    preserved_set = _normalized_table_name_set(preserve_tables)
    preservation_active = bool(preserved_set)
    with sql_gen_type_scope(schema, cte_outputs):
        for idx, sc in enumerate(select_cols):
            unnest_sql = _maybe_render_array_unnest_select(
                sc, schema, cte_outputs, dialect, output_aliases, idx, for_cte=for_cte
            )
            if unnest_sql is not None:
                rendered = unnest_sql
            else:
                rendered = render_select_col_sql(sc, dialect)
                parts = effective_select_parts(sc, None, None)
                if preservation_active and _aggregate_needs_zero_fill(parts.expr, preserved_set):
                    rendered = f"COALESCE({rendered}, 0)"
                if output_aliases and idx < len(output_aliases):
                    rendered = f"{rendered} AS {output_aliases[idx]}"
            select_exprs.append(rendered)
        if append_select_sql:
            select_exprs.extend(append_select_sql)
    return select_exprs


def _render_order_by_sql(order_by_cols: list[OrderByCol], dialect: Dialect | None) -> str:
    ob_exprs = []
    for obc in order_by_cols or []:
        rendered = render_expr_sql(obc.expr, dialect)
        direction = obc.direction.upper() if obc.direction else "ASC"
        if dialect is not None:
            ob_exprs.append(dialect.render_order_by_col(rendered, direction, obc.nulls))
        elif obc.nulls in ("first", "last"):
            placement = "NULLS FIRST" if obc.nulls == "first" else "NULLS LAST"
            ob_exprs.append(f"{rendered} {direction} {placement}")
        else:
            ob_exprs.append(f"{rendered} {direction}")
    return ", ".join(ob_exprs)


def _allocate_distinct_on_cte_name(reserved: set[str]) -> str:
    reserved_ci = {name.strip().lower() for name in reserved if name}
    idx = 1
    while True:
        candidate = f"{DISTINCT_ON_CTE_NAME_PREFIX}{idx}"
        if candidate.lower() not in reserved_ci:
            return candidate
        idx += 1


def wrap_core_sql_with_distinct_on(
    core_sql: str,
    *,
    select_exprs: list[str],
    distinct_on: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    limit: int | None,
    dialect: Dialect,
) -> str:
    """Wrap a core SELECT block with a ROW_NUMBER partition filter."""
    partition = ", ".join(render_expr_sql(expr, dialect) for expr in distinct_on)
    order_sql = _render_order_by_sql(order_by_cols, dialect)
    rank_expr = f"ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {order_sql}) AS {DISTINCT_ON_RANK_COLUMN}"
    parsed = dialect.parse_select(core_sql)
    if parsed is None:
        raise ValueError("wrap_core_sql_with_distinct_on: core_sql is not parseable as SELECT")
    if not _append_select_expressions(parsed, [rank_expr], dialect):
        raise ValueError("wrap_core_sql_with_distinct_on: could not append rank expression")
    inner_sql = dialect.emit_sql(parsed)
    outer_select = ", ".join(select_exprs)
    outer_parts = [
        f"SELECT {outer_select}",
        f"FROM (\n{inner_sql}\n) AS _don_src",
        f"WHERE {DISTINCT_ON_RANK_COLUMN} = 1",
    ]
    if order_by_cols:
        outer_parts.append(f"ORDER BY {order_sql}")
    if limit:
        outer_parts.append(f"LIMIT {limit}")
    return "\n".join(outer_parts)


def _build_deterministic_select_block(
    select_cols: list[SelectCol],
    tables: list[str],
    group_by_cols: list[NormalizedExpr],
    order_by_cols: list[OrderByCol],
    where: PredicateGroup | None,
    having: PredicateGroup | None,
    limit: int | None,
    grain: str,
    dialect: Dialect,
    output_aliases: list[str] | None = None,
    *,
    schema: SchemaGraph | None = None,
    for_cte: bool = False,
    from_table_override: str | None = None,
    extra_from_tables: list[str] | None = None,
    distinct_select_index: int = -1,
    param_values: Mapping[str, Any] | None = None,
    append_select_sql: list[str] | None = None,
    preserve_tables: list[str] | None = None,
) -> str:
    """Build a single SELECT block from structured intent clauses."""
    lines: list[str] = []

    select_exprs: list[str] = []
    cte_outputs: dict[str, Any] = {}
    preserved_set = _normalized_table_name_set(preserve_tables)
    preservation_active = bool(preserved_set)
    with sql_gen_type_scope(schema, cte_outputs):
        for idx, sc in enumerate(select_cols):
            unnest_sql = _maybe_render_array_unnest_select(
                sc, schema, cte_outputs, dialect, output_aliases, idx, for_cte=for_cte
            )
            if unnest_sql is not None:
                rendered = unnest_sql
            else:
                rendered = render_select_col_sql(sc, dialect)
                parts = effective_select_parts(sc, None, None)
                if preservation_active and _aggregate_needs_zero_fill(parts.expr, preserved_set):
                    rendered = f"COALESCE({rendered}, 0)"
                if output_aliases and idx < len(output_aliases):
                    rendered = f"{rendered} AS {output_aliases[idx]}"
            select_exprs.append(rendered)

        if append_select_sql:
            select_exprs.extend(append_select_sql)

        select_keyword = "SELECT DISTINCT" if distinct_select_index >= 0 else "SELECT"
        lines.append(select_keyword + " " + ", ".join(select_exprs))

        if tables:
            from_tbl = from_table_override if from_table_override else tables[0]
            from_sql = dialect.quote_schema_qualified(from_tbl)
            if extra_from_tables:
                pipeline_trace(
                    "pipeline.generate_and_validate_sql.scalar_cte_cross_join",
                    lambda: stable_json({"anchor": from_tbl, "scalar_ctes": list(extra_from_tables)}),
                )
                for extra in extra_from_tables:
                    from_sql = f"{from_sql} CROSS JOIN {dialect.quote_schema_qualified(extra)}"
            lines.append(f"FROM {from_sql}")

        def _render_where_leaf(pred: WhereParam | HavingParam) -> str:
            return _render_predicate_clause(
                pred, dialect, is_having=False, param_values=param_values, schema=schema, cte_outputs=cte_outputs
            )

        where_sql = _render_predicate_group_sql(where, _render_where_leaf, sqlglot_dialect=dialect.sqlglot_dialect)
        if where_sql:
            lines.append("WHERE " + where_sql)

        if group_by_cols:
            gb_exprs = [render_expr_sql(g, dialect) for g in group_by_cols]
            lines.append("GROUP BY " + ", ".join(gb_exprs))

        def _render_having_leaf(pred: WhereParam | HavingParam) -> str:
            return _render_predicate_clause(
                pred, dialect, is_having=True, param_values=param_values, schema=schema, cte_outputs=cte_outputs
            )

        having_sql = _render_predicate_group_sql(having, _render_having_leaf, sqlglot_dialect=dialect.sqlglot_dialect)
        if having_sql:
            lines.append("HAVING " + having_sql)

        if order_by_cols:
            lines.append("ORDER BY " + _render_order_by_sql(order_by_cols, dialect))

        if limit:
            lines.append(f"LIMIT {limit}")

    return "\n".join(lines)


def _effective_select_col_for_sql(sc: SelectCol) -> SelectCol:
    """Return a select column with registry references reduced to the. resolved base expression."""
    parts = effective_select_parts(sc, None, None)
    return SelectCol(expr=parts.expr)


def generate_col_alias(sc: SelectCol) -> str:
    """Build a deterministic display alias from a SelectCol's. expression. metadata."""
    sc = _effective_select_col_for_sql(sc)
    expr = sc.expr
    col = expr.primary_column
    if col:
        col_clean = col.rsplit(".", 1)[-1].lower()
    else:
        col_clean = ""

    def _alias_token(group: MulGroup) -> str:
        if not group.multiply:
            return "x"
        leaf = group.multiply[0]
        if leaf.column_ref:
            return leaf.column_ref.rsplit(".", 1)[-1].lower()
        if leaf.star:
            return "all"
        if leaf.keyword:
            return leaf.keyword.lower()
        return "x"

    groups = expr.add_groups or []
    if len(groups) >= 2 and not expr.agg_func and not expr.scalar_func:
        parts = [_alias_token(g) for g in groups]
        alias = "_times_".join(parts)
    elif expr.sub_groups and groups:
        plus_part = _alias_token(groups[0])
        minus_part = _alias_token(expr.sub_groups[0])
        alias = f"{plus_part}_minus_{minus_part}"
    elif col_clean:
        alias = col_clean
    else:
        return ""

    distinct_prefix = ""
    if groups and (
        groups[0].distinct
        or any(m.column_ref and m.column_ref.upper().startswith("DISTINCT ") for m in groups[0].multiply)
    ):
        distinct_prefix = "distinct_"

    if expr.agg_func:
        alias = f"{expr.agg_func}_{distinct_prefix}{alias}"
    elif groups and groups[0].agg_func:
        alias = f"{groups[0].agg_func}_{distinct_prefix}{alias}"
    elif distinct_prefix:
        alias = f"{distinct_prefix}{alias}"
    if expr.inner_scalar_func:
        alias = f"{expr.inner_scalar_func}_{alias}"
    if expr.scalar_func:
        alias = f"{expr.scalar_func}_{alias}"

    return alias.lower()


def select_col_prefers_llm_display_alias(sc: SelectCol) -> bool:
    """Whether. :func:`aetherdialect._pipeline.enriched_display_alias_map` should ask the LLM for a display header."""
    parts = effective_select_parts(sc, None, None)
    if parts.case_when is not None:
        return True
    if parts.window_spec is not None:
        return True
    if not generate_col_alias(sc):
        return True
    sc_eff = _effective_select_col_for_sql(sc)
    expr = sc_eff.expr
    agg_count = sum(1 for g in (expr.add_groups or []) if g.agg_func)
    if expr.agg_func:
        agg_count += 1
    if sc_eff.is_aggregated and agg_count == 0:
        agg_count = 1
    if agg_count > 1:
        return False
    scalars: list[str] = []
    if expr.scalar_func:
        scalars.append(expr.scalar_func.lower())
    if expr.inner_scalar_func:
        scalars.append(expr.inner_scalar_func.lower())
    if len(scalars) > 1:
        return False
    groups = expr.add_groups or []
    subs = expr.sub_groups or []
    if (len(groups) >= 2 or subs) and agg_count == 0 and "concat" not in scalars:
        return False
    return True


def build_display_sql(
    sql_param: str, intent: RuntimeIntent, display_alias_map: dict[str, str] | None, dialect: Dialect
) -> str:
    """Build display SQL with deterministic aliases via the dialect's. AST projection-replace path. The projection list flows through :meth:`aetherdialect._dialect.Dialect.replace_projection` and re- emitted via :meth:`aetherdialect._dialect.Dialect.emit_sql`. Returns *sql_param* unchanged when ``select_cols`` is empty or the AST replace cannot be performed."""
    cols = intent.select_cols or []
    if not cols:
        return sql_param
    overrides = display_alias_map or {}
    items: list[tuple[str, str]] = []
    seen_aliases: set[str] = set()
    for idx, sc in enumerate(cols):
        expr_str = render_select_col_sql(sc)
        alias = overrides.get(sc.signature_key) or generate_col_alias(sc)
        if not alias:
            alias = f"col_{idx + 1}"
        base = alias
        counter = 2
        while alias in seen_aliases:
            alias = f"{base}_{counter}"
            counter += 1
        seen_aliases.add(alias)
        items.append((expr_str, alias))
    try:
        parsed = dialect.parse_select(sql_param)
    except Exception:
        return sql_param
    if parsed is None:
        return sql_param
    try:
        attached = dialect.replace_projection(parsed, [(e, a) for e, a in items])
    except Exception:
        return sql_param
    if not attached:
        return sql_param
    try:
        rendered = dialect.emit_sql(parsed)
    except Exception:
        return sql_param
    return rendered if isinstance(rendered, str) and rendered else sql_param


def _date_window_inclusive_upper_sql(left_rendered: str, unit: str, dialect: Dialect) -> str:
    """Upper inclusive anchor for relative ``date_window`` ranges (through the clock anchor for the unit class)."""
    return dialect.render_date_window_inclusive_upper(left_rendered, unit)


def _render_date_window_predicate(
    pred: WhereParam | HavingParam,
    left_rendered: str,
    dialect: Dialect,
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[str]:
    """Render WHERE/HAVING clause part(s) for a date_window filter or. having condition."""
    resolved = pred.resolved_value(param_values)
    rv = resolved if isinstance(resolved, dict) else {}
    if "start" in rv and "end" in rv:
        start_val = rv["start"]
        end_val = rv["end"]
        if isinstance(start_val, str) and isinstance(end_val, str):
            return [
                f"{left_rendered} >= '{start_val}'",
                f"{left_rendered} <= '{end_val}'",
            ]
    unit = str(rv.get("unit", "day") or "day")
    amt_raw = rv.get("amount")
    amount = int(amt_raw) if amt_raw is not None else 0
    lower_sql = dialect.render_date_window(left_rendered, ">=", unit, amount)
    upper_sql = _date_window_inclusive_upper_sql(left_rendered, unit, dialect)
    return [lower_sql, upper_sql]


def _serialize_join_candidate_row(c: dict[str, Any], *, schema: SchemaGraph | None = None) -> dict[str, Any]:
    """Return a join-candidate row suitable for join-choice JSON payloads. When *schema* is supplied, asserts that no column in the rendered ``join_path_signature`` is hidden by visibility rules. This is a defensive guard against missed filter paths in :func:`enumerate_join_paths_base` / :func:`enumerate_semantic_paths`."""
    if schema is not None:
        sigs = c.get("join_path_signature") or []
        for seg in sigs:
            seg_str = str(seg).strip()
            if "->" not in seg_str:
                continue
            left_part, right_part = seg_str.split("->", 1)
            for side in (left_part.strip(), right_part.strip()):
                if "." not in side:
                    continue
                tbl, cols = side.split(".", 1)
                if tbl not in schema.tables:
                    continue
                tmeta = schema.tables[tbl]
                for col in cols.split(","):
                    col = col.strip()
                    cm = tmeta.columns.get(col)
                    if cm is not None and not cm.is_visible:
                        raise AssertionError(
                            f"join candidate signature '{seg_str}' references non-visible column {tbl}.{col}"
                        )
    return {
        "candidate_id": c.get("candidate_id"),
        "join_path_signature": c.get("join_path_signature"),
    }


def _serialize_join_choice_candidate(c: dict[str, Any], *, schema: SchemaGraph | None = None) -> dict[str, Any]:
    """Serialize one join-choice candidate for intra-source or cross- source scopes."""
    if c.get("join_path_signature") is not None:
        return _serialize_join_candidate_row(c, schema=schema)
    return {
        "candidate_id": c.get("candidate_id"),
        "logical_key": c.get("logical_key"),
        "left": c.get("left"),
        "right": c.get("right"),
        "kind": c.get("kind"),
    }


def build_join_choice_prompt(
    q_norm: str,
    deterministic_sql: str,
    llm_scopes: list[dict[str, Any]],
    *,
    schema: SchemaGraph | None = None,
    prior_join_feedback: list[str] | None = None,
) -> tuple[str, str]:
    """Build minimal prompt for LLM to return per-scope join candidate. IDs."""
    system = (
        "You are a join selector for text-to-SQL. Output ONLY valid JSON. "
        "Each scope lists its own join candidates; candidate ids are scoped and may repeat across "
        "scopes. Return a single object ``choices`` mapping each scope string to a candidate id "
        "(``J00``, ``J01``, …) or the string ``NA`` when none of the listed candidates fit that scope."
    )
    if prior_join_feedback:
        lines = "\n".join(f"- {x}" for x in prior_join_feedback if str(x).strip())
        if lines:
            system += "\n\n" + JOIN_PRIOR_FEEDBACK_HEADING + "\n" + lines
    scopes_payload = []
    for block in llm_scopes:
        cands = block.get("candidates") or []
        scopes_payload.append(
            {
                "scope": block.get("scope"),
                "tables": block.get("tables") or [],
                "candidates": [
                    _serialize_join_choice_candidate(c, schema=schema) for c in cands if isinstance(c, dict)
                ],
            }
        )
    user = prompt_json(
        {
            "task": (
                "Given the question and the deterministic SQL template, choose one join candidate id "
                "per scope. Return only the ids; do not modify the SQL."
            ),
            "question": q_norm,
            "deterministic_sql": deterministic_sql,
            "scopes": scopes_payload,
            "output_format": {
                "choices": ('Dict keyed by scope: {"main": "J01" | "NA", "cte:<cte_name>": "J01" | "NA", ...}'),
            },
        },
        JOIN_CHOICE_PROMPT_KEY_ORDER,
    )
    return system, user


def _valid_join_choice_ids_from_candidates(candidates: list[dict[str, Any]]) -> frozenset[str]:
    """Collect stripped non-empty ``candidate_id`` strings from a candidate list."""
    out: set[str] = set()
    for c in candidates or []:
        cid = c.get("candidate_id")
        if isinstance(cid, str) and cid.strip():
            out.add(cid.strip())
    return frozenset(out)


def _parse_join_choice_payload(parsed: dict[str, Any]) -> dict[str, str]:
    """Extract per-scope join choices from an LLM JSON object."""
    raw = parsed.get("choices")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _scope_choice_valid_ids(candidates: list[dict[str, Any]], *, allow_na: bool) -> frozenset[str]:
    """Allowed ids for one scope, optionally including the ``NA`` sentinel."""
    base = _valid_join_choice_ids_from_candidates(candidates)
    return base | frozenset({"NA"}) if allow_na else base


def _join_choice_payload_acceptable_pass1(
    merged: dict[str, str],
    required_scopes: frozenset[str],
    llm_scopes: list[dict[str, Any]],
    accept_na_by_scope: dict[str, bool],
) -> bool:
    """Return whether merged choices satisfy first-pass validation for every required scope."""
    scope_candidates = {str(s["scope"]): list(s.get("candidates") or []) for s in llm_scopes if "scope" in s}
    for sk in required_scopes:
        if sk not in merged:
            return False
        val = merged[sk]
        allow_na = accept_na_by_scope.get(sk, False)
        allowed = _scope_choice_valid_ids(scope_candidates.get(sk, []), allow_na=allow_na)
        if val not in allowed:
            return False
    return True


def _join_choice_payload_valid_final(
    merged: dict[str, str], required_scopes: frozenset[str], llm_scopes: list[dict[str, Any]]
) -> bool:
    """Return whether every required scope has a non-NA candidate id present in that scope's list."""
    scope_candidates = {str(s["scope"]): list(s.get("candidates") or []) for s in llm_scopes if "scope" in s}
    for sk in required_scopes:
        if sk not in merged:
            return False
        val = merged[sk]
        allowed = _scope_choice_valid_ids(scope_candidates.get(sk, []), allow_na=False)
        if val not in allowed:
            return False
    return True


def first_base_non_j00_candidate_id(hints: dict[str, Any]) -> str | None:
    """Return the first non-J00 candidate id whose tier is base, if any."""
    for c in hints.get("candidates", []) or []:
        cid = c.get("candidate_id")
        tier = c.get("candidate_tier", "base")
        if isinstance(cid, str) and cid != "J00" and tier == "base":
            return cid
    return None


def get_join_choice_from_llm(
    q_norm: str,
    deterministic_sql: str,
    *,
    llm_scopes: list[dict[str, Any]],
    preset_choices: dict[str, str] | None = None,
    accept_na_by_scope: dict[str, bool] | None = None,
    require_final: bool = False,
    schema: SchemaGraph | None = None,
    prior_join_feedback: list[str] | None = None,
) -> dict[str, str]:
    """Call LLM to get per-scope join candidate ids for the listed. scopes."""
    preset = dict(preset_choices or {})
    accept_na = dict(accept_na_by_scope or {})
    if not llm_scopes:
        return preset
    required = frozenset(str(s["scope"]) for s in llm_scopes if s.get("scope") is not None)
    with prompt_cache_schema_scope(schema_prompt_cache_id(schema)):
        for _attempt in range(2):
            try:
                system, user = build_join_choice_prompt(
                    q_norm, deterministic_sql, llm_scopes, schema=schema, prior_join_feedback=prior_join_feedback
                )
                parsed = llm_json(system, user, retries=1, task="join")
            except LlmJsonExhausted as exc:
                debug(f"[sql_gen.get_join_choice_from_llm] exhausted attempt {_attempt + 1}: {exc}")
                continue
            raw = _parse_join_choice_payload(parsed)
            merged = dict(preset)
            for sk in required:
                if sk not in raw:
                    continue
                val = raw[sk]
                allow_na_here = accept_na.get(sk, False)
                cands = next((x.get("candidates") or [] for x in llm_scopes if str(x.get("scope")) == sk), [])
                allowed = _scope_choice_valid_ids(cands, allow_na=allow_na_here)
                if val in allowed:
                    merged[sk] = val
                else:
                    debug(f"[sql_gen.get_join_choice_from_llm] dropped invalid join choice id for {sk}: {val!r}")
            if require_final:
                if _join_choice_payload_valid_final(merged, required, llm_scopes):
                    return merged
            elif _join_choice_payload_acceptable_pass1(merged, required, llm_scopes, accept_na):
                return merged
    out = dict(preset)
    scope_rows = {str(s["scope"]): s for s in llm_scopes if s.get("scope") is not None}
    for sk in required:
        cands = next((x.get("candidates") or [] for x in llm_scopes if str(x.get("scope")) == sk), [])
        valid = _scope_choice_valid_ids(cands, allow_na=False)
        cur = out.get(sk)
        if cur in valid and cur != "J00":
            continue
        if cur in valid and cur == "J00" and len(scope_rows.get(sk, {}).get("tables") or []) < 2:
            continue
        pick: str | None = None
        for c in sorted(cands, key=lambda x: str(x.get("candidate_id", ""))):
            cid = c.get("candidate_id")
            if isinstance(cid, str) and cid != "J00" and cid in valid:
                pick = cid
                break
        if pick:
            out[sk] = pick
            continue
        row = scope_rows.get(sk, {})
        tables = list(row.get("tables") or [])
        if len(tables) >= 2:
            raise NoJoinPathError(f"join_choice:{sk}", tables)
        out[sk] = "J00"
    return out


register_render_expr_sql(render_expr_sql)

render_predicate_group_sql = _render_predicate_group_sql
render_predicate_clause = _render_predicate_clause
