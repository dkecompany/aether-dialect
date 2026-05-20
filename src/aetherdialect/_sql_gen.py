"""
Deterministic SQL building, FK join enumeration, repair prompts, and canonical join/predicate normalisation.

Each backend uses its own AST library exclusively: pglast for PostgreSQL, sqlglot (Spark dialect) for Databricks.

The:class:`~aetherdialect._dialect.Dialect` adapter exposes ``parse_select``, ``ordered_join_carrier_froms``, ``attach_joins``, and ``emit_sql`` so this module never names a parser library directly.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from typing import Any, Literal

from sqlglot import exp, parse_one

from ._config import (
    JOIN_CHOICE_SCOPE_MAIN,
    JOIN_EDGE_KIND_RANK,
    SCALAR_FUNCTIONS_LEADING_ARG,
    SQL_WINDOW_FUNCTION_UPPER,
    PolicyConfig,
)

JOIN_PRIOR_FEEDBACK_HEADING: str = "Previously rejected joins for this question (avoid these table sets / FK paths):"

from ._contracts_base import (
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    LlmJsonExhausted,
    NoJoinPathError,
    SchemaGraph,
    VirtualTableSpec,
)
from ._contracts_core import (
    CaseWhenExpr,
    CteEmissionKind,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    RuntimeCteStep,
    RuntimeIntent,
    ScalarArg,
    SelectCol,
    WindowSpec,
    current_case_registry_steps,
    current_window_registry_steps,
    effective_select_parts,
    expr_registry_ref,
    register_render_expr_sql,
    registry_render_scope,
)
from ._core_utils import (
    debug,
    llm_json,
    pipeline_trace_lazy,
    stable_json,
)
from ._dialect import Dialect, JoinEdge, get_dialect
from ._validation_schema import get_col_meta


def _databricks_unqualify_agg_arg_sql(sql: str) -> str:
    """
    For Spark/Databricks output, drop table qualifiers on the first argument of
    ``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` within *sql*.

    Args:

        sql: Rendered fragment (argument or ``ORDER BY`` sub-expression).

    Returns:

        Possibly rewritten SQL text.
    """

    if not (sql and sql.strip()):
        return sql
    try:
        tree = parse_one(sql, dialect="spark")
    except Exception:
        return sql
    if isinstance(tree, exp.Column) and tree.table:
        tree.set("table", None)
        return tree.sql(dialect="spark")
    for cls in (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max):
        for node in tree.find_all(cls):
            first = getattr(node, "this", None)
            if isinstance(first, exp.Column) and first.table:
                first.set("table", None)
    return tree.sql(dialect="spark")


def _maybe_databricks_unqualify_window_sql_frag(sql: str, dialect: Dialect | None) -> str:
    """Apply :func:`_databricks_unqualify_agg_arg_sql` only on Databricks dialect."""

    if dialect is None or getattr(dialect, "name", "") != "databricks":
        return sql
    return _databricks_unqualify_agg_arg_sql(sql)


def cte_to_intent_for_ranking(cte: RuntimeCteStep) -> RuntimeIntent:
    """
    Build a synthetic `RuntimeIntent` from `RuntimeCteStep` for CTE-scope join enumeration.

    Args:

        cte: CTE step whose body defines the local join scope.

    Returns:

        A minimal intent mirroring the CTE body lists and limits.
    """
    return RuntimeIntent(
        tables=cte.tables,
        grain=cte.grain,
        select_cols=cte.select_cols,
        group_by_cols=cte.group_by_cols,
        order_by_cols=cte.order_by_cols,
        filters_param=cte.filters_param,
        having_param=cte.having_param,
        param_values=cte.param_values,
        column_map=cte.column_map,
        limit=cte.limit,
        cte_steps=[],
        window_registry=list(cte.window_registry or []),
        case_registry=list(cte.case_registry or []),
    )


def join_candidate_map(join_hints: dict[str, Any]) -> dict[str, list[str]]:
    """
    Build map from candidate ID to join path signature.

    Args:

        join_hints: The join hints dict produced by `join_hints_multi`.

    Returns:

        Dict mapping `candidate_id` to list of join path signature strings.
    """
    out: dict[str, list[str]] = {}
    for c in join_hints.get("candidates", []):
        cid = c.get("candidate_id")
        sig = c.get("join_path_signature")
        if isinstance(cid, str) and isinstance(sig, list):
            out[cid] = [str(x) for x in sig]
    return out


def _analyze_join_topology(sig: list[str]) -> tuple[str, str, list[str]]:
    """
    Analyze join signature to determine topology type, hub table, and leaf tables.

    Args:

        sig: List of join path signature strings.

    Returns:

        the list of endpoint tables.
    """
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
    hubs = sorted(
        [t for t, c in table_counts.items() if c > 1],
        key=lambda t: (-table_counts[t], t),
    )
    if len(leaves) == 2 and len(hubs) == len(table_counts) - 2:
        left_roots: set[str] = set()
        for item in sig:
            if "->" not in item:
                continue
            left_part, _right = item.split("->", 1)
            left_roots.add(left_part.split(".")[0].strip().lower())
        if len(hubs) == 1 and len(left_roots) == 1 and hubs[0].lower() in left_roots:
            return ("star", hubs[0], leaves)
        return ("linear", min(leaves), leaves)
    if len(hubs) == 1:
        return ("star", hubs[0], leaves)
    if hubs:
        return ("tree", hubs[0], leaves)
    return ("linear", min(table_counts.keys()), list(table_counts.keys()))


def _wrap_for_case_insensitive(expr: str, dialect: Dialect) -> str:
    """
    Wrap expression for case-insensitive string comparison.

    Args:

        expr: SQL expression fragment to wrap.

        dialect: Active database dialect.

    Returns:

        SQL fragment for case-insensitive comparison of `expr`.
    """
    return dialect.render_case_insensitive_wrap(expr)


def _phys_table_key(tbl: str) -> str:
    """Return the unqualified lowercase table name for join bookkeeping."""

    return tbl.split(".")[-1].strip().strip('"').strip("`").lower()


def _table_sql_token(tbl: str) -> str:
    """Unqualified table token suitable for SQL references (preserves casing of last segment)."""

    return tbl.split(".")[-1].strip().strip('"').strip("`")


def _join_kind_for_edge(
    join_tbl: str,
    paired_tbl: str,
    cols_on_join: list[str],
    schema: SchemaGraph | None,
) -> str:
    """
    Return ``LEFT`` or ``INNER`` join modifier (leading space) for one edge.

    ``LEFT`` is emitted only when the joining FK column on the source side is nullable; otherwise ``INNER``. Dimension-role bias is intentionally not used here.
    """

    if schema is None:
        return " INNER"
    tmeta = schema.tables.get(join_tbl)
    if not tmeta:
        return " INNER"
    fk_any = False
    fk_nullable = False
    for jc in cols_on_join:
        cm = tmeta.columns.get(jc)
        if cm and cm.is_foreign_key and cm.fk_target and cm.fk_target[0] == paired_tbl:
            fk_any = True
            if cm.is_nullable:
                fk_nullable = True
    if fk_any and fk_nullable:
        return " LEFT"
    return " INNER"


def _join_on_equality_sql(
    left_tbl: str,
    lc: str,
    right_tbl: str,
    rc: str,
    dialect: Dialect | None,
) -> str:
    """Build one ``table.col = table.col`` ON conjunct, using *dialect* quoting when set."""

    if dialect is not None:
        return f"{dialect.quote_table_column(left_tbl, lc)} = {dialect.quote_table_column(right_tbl, rc)}"
    return f"{left_tbl}.{lc} = {right_tbl}.{rc}"


def _join_clause_from_signature(
    signature: list[str],
    from_table: str = "",
    schema: SchemaGraph | None = None,
    dialect: Dialect | None = None,
) -> str:
    """
    Build JOIN clause text from a join path signature.

    Args:

        signature: Description.

        from_table: Optional FROM table name; tables already in the chain are tracked to avoid duplicate JOINs.

        schema: Optional schema graph used to choose `LEFT` vs `INNER` join from table role.

    Returns:

        `signature` is empty.
    """
    if not signature:
        return ""
    edges: list[tuple[int, str, str, list[str], list[str], str, str]] = []
    for idx, seg in enumerate(signature):
        seg = seg.strip()
        if "->" not in seg:
            continue
        left_part, right_part = seg.split("->", 1)
        left_part = left_part.strip()
        right_part = right_part.strip()
        if "." not in left_part or "." not in right_part:
            continue
        left_tbl, left_cols = left_part.split(".", 1)
        right_tbl, right_cols = right_part.split(".", 1)
        left_col_list = [c.strip() for c in left_cols.split(",")]
        right_col_list = [c.strip() for c in right_cols.split(",")]
        on_terms_join = [
            _join_on_equality_sql(left_tbl, lc, right_tbl, rc, dialect)
            for lc, rc in zip(left_col_list, right_col_list, strict=False)
        ]
        if not on_terms_join:
            continue
        on_sql = " AND ".join(on_terms_join)
        edges.append(
            (
                idx,
                left_tbl,
                right_tbl,
                left_col_list,
                right_col_list,
                on_sql,
                left_tbl + right_tbl,
            ),
        )
    if not edges:
        return ""
    if not from_table:
        chain: set[str] = set()
        parts_legacy: list[str] = []
        for (
            _idx,
            left_tbl,
            right_tbl,
            left_col_list,
            right_col_list,
            on_sql,
            _key,
        ) in edges:
            right_tbl_lower = right_tbl.lower()
            if right_tbl_lower in chain:
                join_tbl = left_tbl
            else:
                join_tbl = right_tbl
            join_key = join_tbl.strip().lower()
            if join_key in chain:
                continue
            chain.add(join_key)
            if join_tbl == left_tbl:
                cols_on_join = left_col_list
                paired_tbl = right_tbl
            else:
                cols_on_join = right_col_list
                paired_tbl = left_tbl
            join_kind = _join_kind_for_edge(join_tbl, paired_tbl, cols_on_join, schema)
            parts_legacy.append(f"{join_kind} JOIN {join_tbl} ON {on_sql}")
        return "".join(parts_legacy)
    anchor = from_table.strip()
    anchor_key = _phys_table_key(anchor)
    phys_instances: dict[str, list[str]] = defaultdict(list)
    phys_instances[anchor_key].append(_table_sql_token(anchor))
    parts: list[str] = []
    unused: list[tuple[int, str, str, list[str], list[str], str, str]] = list(edges)
    while unused:
        frontier: list[tuple[int, int, str, str, str, list[str], str]] = []
        for u_idx, (
            sig_i,
            left_tbl,
            right_tbl,
            lcols,
            rcols,
            _on_sql,
            _ek,
        ) in enumerate(unused):
            lk = _phys_table_key(left_tbl)
            rk = _phys_table_key(right_tbl)
            if lk == rk:
                if len(phys_instances[lk]) < 1:
                    continue
                existing = phys_instances[lk][-1]
                inst_n = len(phys_instances[lk]) + 1
                new_alias = f"{_table_sql_token(left_tbl)}__sj{inst_n}"
                on_new = " AND ".join(
                    _join_on_equality_sql(existing, lc, new_alias, rc, dialect)
                    for lc, rc in zip(lcols, rcols, strict=False)
                )
                join_kind = _join_kind_for_edge(
                    _table_sql_token(left_tbl),
                    _table_sql_token(right_tbl),
                    lcols,
                    schema,
                )
                frontier.append(
                    (
                        sig_i,
                        u_idx,
                        left_tbl,
                        new_alias,
                        on_new,
                        lcols,
                        f"SELF::{new_alias}",
                    ),
                )
                continue
            li = lk in phys_instances and bool(phys_instances[lk])
            ri = rk in phys_instances and bool(phys_instances[rk])
            if li and ri:
                continue
            if li:
                join_tbl, paired_tbl, cols_on_join = right_tbl, left_tbl, rcols
                existing = phys_instances[lk][-1]
            elif ri:
                join_tbl, paired_tbl, cols_on_join = left_tbl, right_tbl, lcols
                existing = phys_instances[rk][-1]
            else:
                continue
            join_k = _phys_table_key(join_tbl)
            if join_k in phys_instances:
                continue
            new_tok = _table_sql_token(join_tbl)
            if join_tbl == right_tbl:
                on_new = " AND ".join(
                    _join_on_equality_sql(existing, lc, new_tok, rc, dialect)
                    for lc, rc in zip(lcols, rcols, strict=False)
                )
            else:
                on_new = " AND ".join(
                    _join_on_equality_sql(new_tok, lc, existing, rc, dialect)
                    for lc, rc in zip(lcols, rcols, strict=False)
                )
            frontier.append((sig_i, u_idx, join_tbl, paired_tbl, on_new, cols_on_join, new_tok))
        if not frontier:
            break
        sig_i, u_idx, join_tbl, paired_tbl, on_new, cols_on_join, extra = min(
            frontier,
            key=lambda t: (t[0], _phys_table_key(t[2]), _phys_table_key(t[3])),
        )
        unused.pop(u_idx)
        join_kind = _join_kind_for_edge(join_tbl, paired_tbl, cols_on_join, schema)
        if isinstance(extra, str) and extra.startswith("SELF::"):
            new_alias = extra.split("SELF::", 1)[1]
            parts.append(f"{join_kind} JOIN {join_tbl} AS {new_alias} ON {on_new}")
            phys_instances[_phys_table_key(join_tbl)].append(new_alias)
        else:
            new_tok = extra
            parts.append(f"{join_kind} JOIN {join_tbl} ON {on_new}")
            phys_instances[_phys_table_key(join_tbl)].append(new_tok)
    return "".join(parts)


_WHERE_BUCKET_EDGE_KINDS: frozenset[str] = frozenset({"semantic_profile", "semantic_profile_virtual"})


def _partition_path_join_vs_where(
    signature: list[str],
    edge_kinds: list[str],
) -> tuple[
    list[tuple[int, str, str, list[str], list[str]]],
    list[tuple[int, str, str, list[str], list[str]]],
]:
    """
    Split parsed path segments into JOIN-bucket and WHERE-bucket edges by ``edge_kinds``.

    Each returned tuple is ``(orig_index, left_tbl, right_tbl, left_cols, right_cols)`` parsed from the signature. WHERE-bucket edges are those whose corresponding kind is in :data:`_WHERE_BUCKET_EDGE_KINDS` (Tier-B semantic edges); all other kinds (including missing or unknown) flow into the JOIN bucket.
    """
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
        record = (idx, left_tbl, right_tbl, lcols, rcols)
        if kind in _WHERE_BUCKET_EDGE_KINDS:
            where_bucket.append(record)
        else:
            join_bucket.append(record)
    return join_bucket, where_bucket


def _extra_from_tables_for_where_edges(
    where_edges: list[tuple[int, str, str, list[str], list[str]]],
    tables_already_in_from: set[str],
) -> list[str]:
    """
    Return deduplicated extra-FROM table names introduced only by WHERE-bucket edges.

    Tables already covered by the anchor or JOIN tree are skipped. Order is the first appearance across the where edges.
    """
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


def _join_edges_from_signature(
    signature: list[str],
    edge_kinds: list[str],
    from_table: str,
    schema: SchemaGraph | None = None,
) -> tuple[list[JoinEdge], list[JoinEdge], list[str]] | None:
    """
    Resolve a join-path signature into JOIN edges, WHERE-bucket edges, and extra FROM tables.

    The signature is partitioned by ``edge_kinds`` into two buckets via :func:`_partition_path_join_vs_where`:

    * **JOIN bucket** — Tier-A FK / virtual-bridge segments are walked outward from ``from_table`` and rendered as :class:`JoinEdge` objects suitable for :meth:`aetherdialect._dialect.Dialect.attach_joins`. * **WHERE bucket** — Tier-B semantic segments become :class:`JoinEdge` objects whose ``on_terms`` carry equality conjuncts; the dialect's :meth:`attach_extra_from_and_where` AND-injects them into ``WHERE`` and adds any missing endpoints to ``FROM``.

    Self-joins are not handled by the planner — they must come from a CTE wrap. A self-join segment in the JOIN bucket therefore raises :class:`NoJoinPathError`.
    """

    if not from_table or not signature:
        return None
    join_segments, where_segments = _partition_path_join_vs_where(signature, edge_kinds)
    if not join_segments and not where_segments:
        return None
    anchor = from_table.strip()
    anchor_key = _phys_table_key(anchor)
    phys_instances: dict[str, list[str]] = defaultdict(list)
    phys_instances[anchor_key].append(_table_sql_token(anchor))
    join_edges: list[JoinEdge] = []
    unused: list[tuple[int, str, str, list[str], list[str]]] = list(join_segments)
    while unused:
        frontier: list[
            tuple[
                int,
                int,
                str,
                str,
                tuple[tuple[str, str, str, str], ...],
                list[str],
            ]
        ] = []
        for u_idx, (sig_i, left_tbl, right_tbl, lcols, rcols) in enumerate(unused):
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
            elif ri:
                join_tbl, paired_tbl, cols_on_join = left_tbl, right_tbl, lcols
                existing = phys_instances[rk][-1]
            else:
                continue
            join_k = _phys_table_key(join_tbl)
            if join_k in phys_instances:
                continue
            new_tok = _table_sql_token(join_tbl)
            if join_tbl == right_tbl:
                on_terms = tuple((existing, lc, new_tok, rc) for lc, rc in zip(lcols, rcols, strict=False))
            else:
                on_terms = tuple((new_tok, lc, existing, rc) for lc, rc in zip(lcols, rcols, strict=False))
            frontier.append((sig_i, u_idx, join_tbl, paired_tbl, on_terms, cols_on_join))
        if not frontier:
            return None
        sig_i, u_idx, join_tbl, paired_tbl, on_terms_chosen, cols_on_join = min(
            frontier,
            key=lambda t: (t[0], _phys_table_key(t[2]), _phys_table_key(t[3])),
        )
        unused.pop(u_idx)
        kind_modifier = _join_kind_for_edge(join_tbl, paired_tbl, cols_on_join, schema).strip().upper() or "INNER"
        join_edges.append(
            JoinEdge(
                table=join_tbl,
                alias=None,
                kind="LEFT" if kind_modifier == "LEFT" else "INNER",
                on_terms=on_terms_chosen,
            ),
        )
        phys_instances[_phys_table_key(join_tbl)].append(_table_sql_token(join_tbl))
    if len(join_edges) != len(join_segments):
        return None
    where_edges: list[JoinEdge] = []
    for _idx, left_tbl, right_tbl, lcols, rcols in where_segments:
        left_tok = _table_sql_token(left_tbl)
        right_tok = _table_sql_token(right_tbl)
        on_terms = tuple((left_tok, lc, right_tok, rc) for lc, rc in zip(lcols, rcols, strict=False))
        where_edges.append(
            JoinEdge(
                table=right_tok,
                alias=None,
                kind="INNER",
                on_terms=on_terms,
            ),
        )
    tables_in_from: set[str] = set(phys_instances.keys())
    extra_from_tables = _extra_from_tables_for_where_edges(where_segments, tables_in_from)
    return join_edges, where_edges, extra_from_tables


def _orient_join_sig_for_from(
    sig: list[str],
    from_table: str,
) -> list[str]:
    """
    Reorient join segments so that no target duplicates the FROM table.

    Args:

        sig: Description.

        from_table: Current FROM table name used to flip segments whose right side matches it.

    Returns:

        JOIN t` artefacts when `tables[0]` sits on the target side.
    """
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
    """
    Sort join signature segments for star/tree topologies before SQL emission.

    Args:

        oriented: Join segments already oriented to the active ``FROM`` table.

    Returns:

        Lexicographically ordered segments when topology is star or tree; otherwise unchanged.
    """
    if len(oriented) <= 1:
        return oriented
    topology_type, _, _ = _analyze_join_topology(oriented)
    if topology_type in ("star", "tree"):
        return sorted(oriented, key=lambda s: s.strip().lower())
    return oriented


def _try_ast_inject_joins(
    det_sql: str,
    join_sigs_ordered: list[list[str]],
    edge_kinds_ordered: list[list[str]],
    schema: SchemaGraph | None,
    dialect: Dialect,
) -> str | None:
    """
    Parse deterministic SQL via the dialect adapter, attach joins on ordered carriers, and re-render.

    Returns ``None`` when parsing fails, carrier count mismatches, any slot fails to resolve structured edges, or any attach call fails.
    """

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
    for carrier, sig, kinds in zip(carriers, sigs_padded, kinds_padded, strict=True):
        if not sig:
            per_slot_join.append([])
            per_slot_where.append([])
            per_slot_extra_from.append([])
            continue
        from_anchor = dialect.from_anchor_of(carrier)
        if not from_anchor:
            return None
        oriented = _orient_join_sig_for_from(sig, from_anchor)
        kinds_aligned = list(kinds)
        if len(kinds_aligned) < len(oriented):
            kinds_aligned = kinds_aligned + [""] * (len(oriented) - len(kinds_aligned))
        canon = _canonicalize_join_sig_segments(list(oriented))
        if canon != oriented:
            order = sorted(range(len(oriented)), key=lambda i: oriented[i].strip().lower())
            kinds_aligned = [kinds_aligned[i] for i in order]
        oriented = canon
        resolved = _join_edges_from_signature(oriented, kinds_aligned, from_anchor, schema)
        if resolved is None:
            return None
        join_edges, where_edges, extra_from_tables = resolved
        if not join_edges and not where_edges:
            return None
        per_slot_join.append(join_edges)
        per_slot_where.append(where_edges)
        per_slot_extra_from.append(extra_from_tables)
    for carrier, edges in zip(carriers, per_slot_join, strict=True):
        if not edges:
            continue
        if not dialect.attach_joins(parsed, carrier, edges):
            return None
    for carrier, where_edges, extra_from in zip(
        carriers,
        per_slot_where,
        per_slot_extra_from,
        strict=True,
    ):
        if not where_edges and not extra_from:
            continue
        if not dialect.attach_extra_from_and_where(parsed, carrier, extra_from, where_edges):
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
) -> str:
    """
    Attach JOIN clauses for each ordered carrier (CTE inner SELECTs left-to-right then outer SELECT) via the dialect's AST adapter and re-emit.

    Returns *det_sql* unchanged when *join_sigs_ordered* is empty or *dialect* is ``None``.

    Raises:class:`JoinInjectionAlignmentError` when the carrier count from the dialect does not match ``join_sigs_ordered``.

    Raises:class:`JoinInjectionFailedError` when parsing fails, no join anchor can be read, edges cannot be resolved, or AST attach/emit fails.

    The AST path preserves ``:pN`` / ``:sN`` placeholders verbatim.

        ``edge_kinds_ordered`` carries one parallel per-segment kind list per carrier; segments with a kind in :data:`_WHERE_BUCKET_EDGE_KINDS` (Tier-B semantic edges) are routed through :meth:`aetherdialect._dialect.Dialect.attach_extra_from_and_where` instead of ``JOIN ... ON``.
    """
    pipeline_trace_lazy(
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
        ast_out = _try_ast_inject_joins(det_sql, join_sigs_ordered, kinds_in, schema, dialect)
    except JoinInjectionAlignmentError as exc:
        pipeline_trace_lazy(
            "sql_gen.inject_join.alignment_failed",
            lambda: stable_json(
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
        pipeline_trace_lazy(
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
    pipeline_trace_lazy(
        "sql_gen.inject_join_into_deterministic_sql.output",
        lambda: stable_json({"sql": ast_out}),
    )
    return ast_out


def _canonical_join_edge_string(schema: SchemaGraph | None, e: dict[str, Any]) -> str:
    """
    Build one undirected join edge string in a stable orientation.

    When *schema* is provided, a physical edge matching a declared catalog ``FKEdge`` is oriented as ``src->dst`` from that edge. Otherwise, lexicographic ``src_table`` / ``dst_table`` order breaks ties for inferred or virtual edges.

    Args:

        schema: Optional schema graph for catalog FK lookup.

        e: Edge dict with ``src_table``, ``src_cols``, ``dst_table``, ``dst_cols``.

    Returns:

        A single ``table.col,...->table.col,...`` fragment.
    """

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


def _join_path_signature_for_path(
    path: list[dict[str, Any]],
    schema: SchemaGraph | None = None,
) -> list[str]:
    """
    Generate signature strings for each edge on a join path.

    Args:

        path: Ordered edge dicts for one candidate path.

        schema: Optional graph used to orient catalog FK edges consistently.

    Returns:

        One string per edge, each ``src.cols->dst.cols`` in canonical orientation.
    """

    return [_canonical_join_edge_string(schema, e) for e in path]


def _candidate_join_paths_for_tables(schema: SchemaGraph, tables: list[str]) -> list[list[dict[str, Any]]]:
    """
    Compute all candidate join paths for a set of tables by trying every table as root.

    Args:

        schema: The schema graph containing pre-computed join paths.

        tables: List of table names that must all be reachable in each candidate.

    Returns:

        when no direct paths exist.
    """
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
        seen: set = set()
        out: list[dict[str, Any]] = []
        for e in edges:
            pair = (
                (e["src_table"], tuple(e["src_cols"])),
                (e["dst_table"], tuple(e["dst_cols"])),
            )
            canonical = tuple(sorted(pair))
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(e)
        return out

    table_set = set(tables)

    def _edges_cover_tables(edges: list[dict[str, Any]], root: str) -> set[str]:
        """
        Collect all table names reachable from edges plus the given root.

        Args:

            edges: List of join edge dicts.

            root: Root table name always included in the covered set.

        Returns:

            Set of table names appearing in `edges` union `{root}`.
        """
        covered = {root}
        for e in edges:
            covered.add(e["src_table"])
            covered.add(e["dst_table"])
        return covered

    def _merge_paths_cartesian(root: str, others: list[str], allow_bridges: bool) -> list[list[dict[str, Any]]]:
        """
        Merge shortest-path ties from root to every other table via a capped cross-product.

        Args:

            root: Root table for path lookup in ``schema.join_paths_multi``.

            others: Tables that must be connected via merged edges.

            allow_bridges: When false, path endpoints must stay within the intent table set.

        Returns:

            Distinct merged edge lists, each covering every required table when possible.
        """

        max_out = max(1, int(PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP))
        others_sorted = sorted(others)
        if not others_sorted:
            return [[]]
        option_lists: list[list[list[dict[str, Any]]]] = []
        for target in others_sorted:
            raw_paths = schema.join_paths_multi.get(root, {}).get(target, [])
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
            if len(out_merges) >= max_out:
                break
        return out_merges

    def _collect(allow_bridges: bool) -> dict[tuple, list[dict[str, Any]]]:
        """
        Enumerate deduped join path candidates for all roots under bridge policy.

        Args:

            allow_bridges: Description.

        Returns:

            unique path shape.
        """
        candidates: dict[tuple, list[dict[str, Any]]] = {}
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

    cap = max(1, int(PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP))
    res = list(all_candidates.values())
    res.sort(key=lambda m: (len(m), tuple(_join_path_signature_for_path(m, schema))))
    return res[:cap]


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
        pair = (
            (e["src_table"], tuple(e["src_cols"])),
            (e["dst_table"], tuple(e["dst_cols"])),
        )
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
        assert tag != "user_override_semantic", (
            "user_override_semantic must be routed through semantic_join_neighbors, not as an FKEdge"
        )
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
    scope_tables: list[str],
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
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
            pipeline_trace_lazy(
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
        acc: list[dict[str, Any]] = []
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

    return list(out_map.values())


def _path_uses_only_visible_columns(
    path: list[dict[str, Any]],
    schema: SchemaGraph,
) -> bool:
    """
    Return True iff every physical-table column referenced by *path* edges is visible.

    Virtual (CTE) columns are not gated here; only physical schema columns are checked against :attr:`ColumnMetadata.is_visible`. Used to filter LLM-facing join candidates so denied or hidden-sensitivity columns never appear in any rendered signature.
    """
    for edge in path:
        for tbl_key, cols_key in (("src_table", "src_cols"), ("dst_table", "dst_cols")):
            tbl = edge.get(tbl_key)
            if not tbl or tbl not in schema.tables:
                continue
            tmeta = schema.tables[tbl]
            for col in edge.get(cols_key, []) or []:
                cm = tmeta.columns.get(col)
                if cm is not None and not cm.is_visible:
                    return False
    return True


def enumerate_join_paths_base(
    scope_tables: list[str],
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
) -> list[list[dict[str, Any]]]:
    """
    Enumerate FK-derived and virtual-bridge join paths covering all scope tables.

    Args:

        scope_tables: Physical and virtual (CTE) names in this join scope.

        schema: Physical schema graph.

        virtual_specs: Virtual column specs keyed by CTE name.

    Returns:

        Edge lists with ``edge_kind`` on each edge; empty inner list when impossible.
    """
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
        phys_paths_raw = [_tag_physical_join_path(p) for p in _candidate_join_paths_for_tables(schema, phys_implicit)]
    elif len(phys) >= 2:
        phys_paths_raw = [_tag_physical_join_path(p) for p in _candidate_join_paths_for_tables(schema, phys)]
    else:
        phys_paths_raw = [[]]
    merged = _extend_join_paths_with_bridges(phys_paths_raw, virt_edges, target, schema)
    if not any(target <= _nodes_in_join_path(p) for p in merged):
        merged = _extend_join_paths_with_bridges([[]], virt_edges, target, schema)
    filtered = [p for p in merged if target <= _nodes_in_join_path(p)]
    visible = [p for p in filtered if _path_uses_only_visible_columns(p, schema)]
    return visible if visible else [[]]


def enumerate_semantic_paths(
    scope_tables: list[str],
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
) -> list[list[dict[str, Any]]]:
    """
    Build single-edge semantic paths from profiled ``semantic_join_neighbors`` on physical
    columns (and the same lists lifted onto virtual CTE columns), plus an overlap pass for
    virtual–virtual pairs that is not represented on the physical graph.

    Args:

        scope_tables: Tables and CTE names in the join scope.

        schema: Physical schema.

        virtual_specs: CTE virtual column specs.

    Returns:

        One-edge paths using ``semantic_profile`` / ``semantic_profile_virtual`` kinds.
    """
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


def _dedupe_paths_by_sig(
    paths: list[list[dict[str, Any]]],
    schema: SchemaGraph,
) -> list[list[dict[str, Any]]]:
    """Deduplicate path lists by sorted join signature tuple."""
    out_map: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for p in paths:
        out_map.setdefault(_join_edge_sig_tuple(p, schema), p)
    return list(out_map.values())


def _order_join_candidates_stable(
    paths: list[list[dict[str, Any]]],
    schema: SchemaGraph,
) -> list[list[dict[str, Any]]]:
    """Deterministic ordering: base tier first, then length, edge kinds, signatures, path dump."""

    def sort_key(p: list[dict[str, Any]]) -> tuple[Any, ...]:
        ext = 1 if _path_has_semantic_edge(p) else 0
        kind_rank = _join_path_edge_kind_rank_key(p)
        kinds = tuple(sorted(e.get("edge_kind", "") for e in p))
        sigs = tuple(sorted(_join_path_signature_for_path(p, schema)))
        edge_dump = stable_json(
            [{"s": e["src_table"], "d": e["dst_table"], "k": e.get("edge_kind", "")} for e in p],
        )
        return (ext, kind_rank, len(p), kinds, sigs, edge_dump)

    return sorted(paths, key=sort_key)


def tables_in_join_scope(
    tables: list[str] | None,
    schema: SchemaGraph,
    virtual_specs: dict[str, VirtualTableSpec],
) -> list[str]:
    """
    Return declared names that resolve to physical tables or non-scalar virtual CTE specs.

    Scalar-subquery CTEs (``emission == "scalar_subquery"``) are excluded from the join scope:
    they are CROSS JOIN'd into the FROM list at render time and do not participate in
    physical / virtual FK enumeration.

    Args:

        tables: Intent or CTE ``tables`` list.

        schema: Physical schema graph.

        virtual_specs: Map of CTE name to virtual table spec.

    Returns:

        Stable-unique names in first-seen order.
    """
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


def physical_tables_for_join_hints(
    tables: list[str] | None,
    schema: SchemaGraph,
) -> list[str]:
    """
    Return physical table names from `tables` that exist in `schema`.

    Args:

        tables: Declared table list, possibly mixing CTE names and bases.

        schema: Loaded schema graph.

    Returns:

        Schema table keys in input order. The same physical table may appear more than once
        when *tables* lists it repeatedly (self-join). CTE aliases and unknown names are dropped.
    """
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
) -> dict[str, Any]:
    """
    Build ordered join candidates with ``edge_kinds`` and ``candidate_tier`` labels.

    Args:

        schema: The schema graph.

        tables: Physical and virtual table names for this scope.

        intent: Optional intent slice (unused for ordering after scoring removal).

        prepend_paths: Deprecated; ignored.

        virtual_specs: CTE virtual specs for lineage-aware bridges.

        include_semantic: When true, append semantic overlap edges (extended tier).

    Returns:

        Dict with ``candidates`` carrying ids, signatures, ``edge_kinds``, and ``candidate_tier``.
    """
    _ = intent
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

    base_paths = enumerate_join_paths_base(tables, schema, virtual_specs)
    sem_paths = enumerate_semantic_paths(tables, schema, virtual_specs) if include_semantic else []
    merged_paths = _dedupe_paths_by_sig(base_paths + sem_paths, schema)
    if not include_semantic:
        merged_paths = [p for p in merged_paths if not _path_has_semantic_edge(p)]
    ordered = _order_join_candidates_stable(merged_paths, schema)
    ordered_eff = [p for p in ordered if p]
    if not ordered_eff:
        merged_nonempty = [p for p in merged_paths if p]
        if merged_nonempty:
            ordered_eff = _order_join_candidates_stable(merged_nonempty, schema)
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
    debug(f"[sql_gen.join_hints_multi] generated {len(out)} candidates")
    return {"candidates": out}


def join_choice_scope_key_cte(cte_name: str) -> str:
    """Return the canonical join-choice scope key for a CTE name."""

    return f"cte:{cte_name}"


class ScopeClass(str, Enum):
    """Join candidate mixture for per-scope disambiguation policy."""

    single_table = "single_table"
    single_fk = "single_fk"
    multi_fk = "multi_fk"
    single_fk_with_semantic = "single_fk_with_semantic"
    multi_fk_with_semantic = "multi_fk_with_semantic"
    semantic_only = "semantic_only"
    empty = "empty"


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


def classify_scope_candidates(
    candidates: list[dict[str, Any]],
    *,
    needs_join: bool = True,
) -> ScopeClass:
    """
    Classify a scope's candidate list for deterministic resolution versus LLM routing.

    When *needs_join* is false, an all-``J00`` payload is treated as ``single_table`` instead of ``empty``.
    """

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


def join_llm_needed_from_candidate_hints(
    candidates: list[dict[str, Any]],
    *,
    needs_join: bool = True,
) -> bool:
    """Return True when join disambiguation still requires a join-choice LLM call."""

    sc = classify_scope_candidates(candidates, needs_join=needs_join)
    return sc in (
        ScopeClass.multi_fk,
        ScopeClass.single_fk_with_semantic,
        ScopeClass.multi_fk_with_semantic,
        ScopeClass.semantic_only,
    )


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
    """
    Build preset join ids and the first-pass join-choice LLM scope list.

    Raises:
        NoJoinPathError: When a multi-table scope has no substantive candidates.
    """

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
            llm_scopes.append(
                {
                    "scope": JOIN_CHOICE_SCOPE_MAIN,
                    "tables": list(main_tables),
                    "candidates": _candidates_slice_fk_only(main_candidates),
                }
            )
            accept_na_by_scope[JOIN_CHOICE_SCOPE_MAIN] = False if forbid_na else True
        else:
            llm_scopes.append(
                {
                    "scope": JOIN_CHOICE_SCOPE_MAIN,
                    "tables": list(main_tables),
                    "candidates": _candidates_slice_fk_only(main_candidates),
                }
            )
            accept_na_by_scope[JOIN_CHOICE_SCOPE_MAIN] = False if forbid_na else True

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
            llm_scopes.append(
                {
                    "scope": sk,
                    "tables": list(tbls),
                    "candidates": _candidates_slice_fk_only(cands),
                }
            )
            accept_na_by_scope[sk] = False if forbid_na else True
        else:
            llm_scopes.append(
                {
                    "scope": sk,
                    "tables": list(tbls),
                    "candidates": _candidates_slice_fk_only(cands),
                }
            )
            accept_na_by_scope[sk] = False if forbid_na else True

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
    """
    Rebuild join hint payloads for scopes that returned NA, keeping others on pass one.

    Args:

        pass1_main: First-pass main join hints.

        pass1_cte: First-pass per-CTE join hints.

        intent: Full runtime intent.

        schema: Physical schema graph.

        virtual_specs: Virtual CTE column specs.

        na_scopes: ``"main"`` and/or CTE names that chose NA.

    Returns:

        ``(main_hints, cte_hints)`` for the second join-selector call.
    """
    if JOIN_CHOICE_SCOPE_MAIN in na_scopes:
        scope = tables_in_join_scope(intent.tables, schema, virtual_specs)
        main = join_hints_multi(
            schema,
            scope,
            intent,
            virtual_specs=virtual_specs,
            include_semantic=True,
        )
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
                    schema,
                    scope,
                    cte_slice,
                    virtual_specs=virtual_specs,
                    include_semantic=True,
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


def _render_scalar_func_args(
    func_name: str,
    args: list[ScalarArg] | None,
    param_keys: list[str] | None,
) -> list[str]:
    """
    Render scalar-function argument tokens, inlining literals when a param key is absent.

    Args:

        func_name: Scalar function name driving unit-literal formatting (``EXTRACT`` keeps
        bare identifiers; all others quote string literals).

    args: Ordered argument values from the normalized expression.

        param_keys: Bind-parameter keys aligned by index with *args*; empty strings denote
        literals that must be emitted inline rather than as ``:placeholder``.

    Returns:

        List of SQL token fragments ready to be joined with ``", "``.
    """
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
    """
    Render a MulGroup as a SQL fragment for expression guide.

    Args:

        g: Description.

        dialect: When set, quotes bare ``table.column`` multiply/divide tokens.

    Returns:

        `"ROUND(SUM(:coeff * table.col), 2)"`.
    """
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
        if g.distinct:
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


def render_expr_sql(expr: NormalizedExpr, dialect: Dialect | None = None) -> str:
    """
    Render a NormalizedExpr as a SQL fragment for expression guide.

    Args:

        expr: Description.

        dialect: When set, quotes bare ``table.column`` tokens inside groups.

    Returns:

        SQL fragment string that the LLM should produce for this expression.
    """

    ref = expr_registry_ref(expr) or ""
    if ref.startswith("w"):
        win_by = {s.registry_id: s for s in current_window_registry_steps()}
        step = win_by.get(ref)
        if step is not None:
            return _render_window_over_sql(step.window_spec, dialect)
        return "0"
    if ref.startswith("c"):
        case_by = {s.registry_id: s for s in current_case_registry_steps()}
        step = case_by.get(ref)
        if step is not None:
            return _render_case_when_sql(step.case_when, dialect)
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
        iargs = _render_scalar_func_args(
            expr.inner_scalar_func,
            expr.inner_scalar_func_args,
            expr.isarg_param_keys,
        )
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


def classify_cte_emission(
    cte: RuntimeCteStep,
    intent: RuntimeIntent,
    schema: SchemaGraph | None,
) -> CteEmissionKind:
    """
    Decide whether a CTE renders as a regular join table or a single-row CROSS JOIN scalar.

    Args:

        cte: CTE step after structural normalization.

        intent: Full intent (unused; kept for API stability).

        schema: Physical schema; unused.

    Returns:

        ``scalar_subquery`` only when the CTE produces exactly one row at planning time:
        ``grain == "scalar"``, exactly one output column, exactly one base table, exactly
        one select column, that select column is aggregated, and there is no ``GROUP BY``.
        Any deviation forces ``join_table`` so the CTE participates as a normal joinable
        relation. Inlining as ``(SELECT ...)`` is never produced; CROSS JOIN against a real
        ``WITH`` clause is the only scalar path.
    """
    _ = intent
    _ = schema
    if (cte.grain or "") != "scalar":
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


def _render_case_branch_sql(fp: FilterParam, dialect: Dialect | None = None) -> str:
    """
    Render a single filter as a SQL predicate for CASE WHEN.

    Args:

        fp: Description.

        dialect: Optional dialect for qualified column quoting.

    Returns:

        SQL predicate string for one `WHEN` branch.

    Raises:

        ValueError: If the filter has neither a literal *right_expr* nor a *param_key*
        and the operator is not a unary null-check, indicating the upstream pipeline
        failed to allocate a parameter binding for the branch's literal value.
    """
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
    """
    Render a CASE expression for SELECT.

    Args:

        cw: Case/when branches and optional else result.

        dialect: Optional dialect for qualified column quoting.

    Returns:

        SQL `CASE ... END` fragment.
    """
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


def _render_window_over_sql(
    ws: WindowSpec,
    dialect: Dialect | None = None,
) -> str:
    """
    Render a window aggregate or function with ``OVER (PARTITION BY ... ORDER BY ...)``.

    Args:

        expr: Inner expression for window arguments when applicable.

        ws: Window specification.

        dialect: Optional dialect for qualified column quoting.

    Returns:

        SQL fragment including ``OVER`` clause.
    """
    fn = SQL_WINDOW_FUNCTION_UPPER.get(ws.function, ws.function.upper())
    if ws.function in ("sum", "avg"):
        arg_sql = render_expr_sql(ws.argument, dialect) if ws.argument else "*"
        arg_sql = _maybe_databricks_unqualify_window_sql_frag(arg_sql, dialect)
        core = f"{fn}({arg_sql})"
    elif ws.function in ("lag", "lead", "first_value", "last_value"):
        arg_sql = render_expr_sql(ws.argument, dialect) if ws.argument else "*"
        arg_sql = _maybe_databricks_unqualify_window_sql_frag(arg_sql, dialect)
        core = f"{fn}({arg_sql})"
    else:
        core = f"{fn}()"
    over_parts: list[str] = []
    if ws.partition_by:
        pe = ", ".join(render_expr_sql(e, dialect) for e in ws.partition_by)
        over_parts.append(f"PARTITION BY {pe}")
    if ws.order_by:
        ob = ", ".join(f"{render_expr_sql(o.expr, dialect)} {o.direction.upper()}" for o in ws.order_by)
        over_parts.append(f"ORDER BY {ob}")
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
    """
    Render a select column including optional CASE or window function.

    Args:

        sc: Select column metadata (case/window/aggregate/plain expression).

        dialect: Optional dialect for qualified column quoting.

    Returns:

        SQL fragment for the SELECT list entry.
    """

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
    """Resolve *name* to the spelling used in *tables* when case-insensitively equal."""

    for t in tables:
        if t.lower() == name.lower():
            return t
    return None


def _first_anchor_table_from_group_by_cols(
    group_by_cols: list[NormalizedExpr] | None,
    tables: list[str],
) -> str | None:
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


def _first_anchor_table_from_order_by_cols(
    order_by_cols: list[Any],
    tables: list[str],
) -> str | None:
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
) -> str | None:
    """
    Pick ``FROM`` for a multi-table block.

    Precedence: grouped grain uses the first group-by table in scope; else the first order-by table in scope; else the join-path signature driver; else row-count heuristic.
    """

    if not tables or len(tables) <= 1:
        return None
    sig_list = [s for s in (join_signature or []) if s]
    if grain == "grouped":
        gb_anchor = _first_anchor_table_from_group_by_cols(group_by_cols, tables)
        if gb_anchor:
            return gb_anchor
    ob_anchor = _first_anchor_table_from_order_by_cols(
        list(order_by_cols) if order_by_cols else [],
        tables,
    )
    if ob_anchor:
        return ob_anchor
    driver_raw = _driver_table_from_join_path_signature(sig_list) if sig_list else None
    resolved = _intent_table_spelling(driver_raw, tables) if driver_raw else None
    if resolved:
        return resolved
    return _deterministic_from_anchor_table(tables, order_by_cols, schema)


def _deterministic_from_anchor_table(
    tables: list[str],
    order_by_cols: list,
    schema: SchemaGraph | None,
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


def build_deterministic_sql(
    intent: RuntimeIntent,
    cte_join_hints: dict[str, dict[str, Any]] | None = None,
    schema: SchemaGraph | None = None,
    dialect: Dialect | None = None,
    join_signature_for_from_anchor: list[str] | None = None,
    cte_join_signatures_for_from_anchor: dict[str, list[str]] | None = None,
) -> str:
    """
    Build a rough deterministic SQL from a RuntimeIntent.

    Args:

        intent: Description.

        cte_join_hints: Optional per-CTE join hint payloads for downstream use.

        schema: Optional schema graph for deterministic block rendering.

        dialect: Dialect for render helpers; defaults to ``get_dialect()``.

        join_signature_for_from_anchor: Optional main-query join path to align ``FROM`` with path driver.

        cte_join_signatures_for_from_anchor: Optional CTE name to join signature for per-CTE ``FROM`` alignment.

    Returns:

        SELECT expressions.
    """
    keep_cte = {s.cte_name.lower() for s in (intent.cte_steps or []) if s.cte_name}
    if cte_join_hints:
        cte_join_hints = {k: v for k, v in cte_join_hints.items() if str(k).strip().lower() in keep_cte}
    if cte_join_signatures_for_from_anchor:
        cte_join_signatures_for_from_anchor = {
            k: v for k, v in cte_join_signatures_for_from_anchor.items() if str(k).strip().lower() in keep_cte
        }
    if dialect is None:
        dialect = get_dialect()
    parts: list[str] = []

    cte_steps = intent.cte_steps or []
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
        )
        cte_extras = _scalar_extras_for_scope(cte.tables or [], cte_anchor)
        with registry_render_scope(cte.window_registry, cte.case_registry):
            cte_sql = _build_deterministic_select_block(
                cte.select_cols or [],
                cte.tables or [],
                cte.group_by_cols or [],
                cte.order_by_cols or [],
                cte.filters_param or [],
                cte.having_param or [],
                cte.limit,
                cte.grain or "row_level",
                dialect,
                cte.output_columns or [],
                schema=schema,
                for_cte=True,
                from_table_override=cte_anchor,
                extra_from_tables=cte_extras or None,
                distinct_select_index=cte.distinct_select_index,
                param_values=cte.param_values,
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
    )
    main_extras = _scalar_extras_for_scope(intent.tables or [], main_anchor)
    with registry_render_scope(intent.window_registry, intent.case_registry):
        main_sql = _build_deterministic_select_block(
            intent.select_cols or [],
            intent.tables or [],
            intent.group_by_cols or [],
            intent.order_by_cols or [],
            intent.filters_param or [],
            intent.having_param or [],
            intent.limit,
            intent.grain or "row_level",
            dialect,
            schema=schema,
            for_cte=False,
            from_table_override=main_anchor,
            extra_from_tables=main_extras or None,
            distinct_select_index=intent.distinct_select_index,
            param_values=intent.param_values,
        )
    parts.append(main_sql)

    return "\n".join(parts)


def _render_clause_chain_inner(parts: Sequence[tuple[str, str]]) -> str:
    """
    Join ``(fragment, bool_op)`` tuples using each tuple's ``bool_op`` as the connector after that fragment.

    Args:

        parts: Non-empty sequence of SQL fragments with forward ``bool_op`` links.

    Returns:

        Single SQL clause without outer parentheses.
    """

    result = parts[0][0]
    for i in range(1, len(parts)):
        raw = parts[i - 1][1] or "AND"
        connector = raw.strip().upper()
        if connector not in ("AND", "OR"):
            connector = "AND"
        result = f"{result} {connector} {parts[i][0]}"
    return result


def _render_clause_chain(parts: list[tuple[str, str]]) -> str:
    """
    Join clause fragments and parenthesize the full chain when any forward link is ``OR``.

    Args:

        parts: SQL fragments with ``bool_op`` after each fragment except the last (ignored).

    Returns:

        Combined SQL string.
    """

    if not parts:
        return ""
    inner = _render_clause_chain_inner(parts)
    has_or = any((op or "AND").strip().upper() == "OR" for _, op in parts[:-1])
    if has_or and len(parts) > 1:
        return f"({inner})"
    return inner


def _join_flat_predicate_parts_with_filter_groups(
    parts: list[tuple[str, str, int | None]],
) -> str:
    """
    Join rendered predicate fragments.

    Flat mode (every ``filter_group`` is ``None``): forward ``bool_op`` between adjacent
    fragments via :func:`_render_clause_chain`.

    Grouped mode (any row has a non-``None`` ``filter_group``): bucket by ``filter_group``
    in first-seen order; within each bucket join with **AND** (ignoring stored ``bool_op``);
    join distinct buckets with **OR**. Single-row buckets are not parenthesized;
    multi-row buckets are wrapped in ``(...)``. A single bucket with multiple rows is
    joined with AND only (no outer parentheses).
    """

    if not parts:
        return ""
    flat_mode = all(gid is None for _, _, gid in parts)
    if flat_mode:
        flat = [(frag, bop) for frag, bop, _ in parts]
        return _render_clause_chain(flat)

    ordered_gids: list[int] = []
    buckets: dict[int, list[str]] = {}
    for frag, _bop, gid in parts:
        g = gid if gid is not None else -1
        if g not in buckets:
            ordered_gids.append(g)
            buckets[g] = []
        buckets[g].append(frag)

    multi_bucket = len(ordered_gids) > 1
    pieces: list[str] = []
    for g in ordered_gids:
        frags = buckets[g]
        if len(frags) == 1:
            pieces.append(frags[0])
        elif multi_bucket:
            inner_parts = [(f, "AND") for f in frags]
            inner = _render_clause_chain_inner(inner_parts)
            pieces.append(f"({inner})")
        else:
            inner_parts = [(f, "AND") for f in frags]
            pieces.append(_render_clause_chain_inner(inner_parts))

    if len(pieces) == 1:
        return pieces[0]
    return " OR ".join(pieces)


def _join_clause_parts_with_bool_op(
    parts: list[tuple[str, str]],
) -> str:
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
    """
    When building a CTE, expand bare array columns via UNNEST/EXPLODE.

    Args:

        sc: Select column for the CTE output.

        schema: Schema graph for column element type lookup.

        cte_outputs: Map of CTE names to prior outputs for metadata resolution.

        dialect: Dialect for UNNEST/EXPLODE rendering.

        output_aliases: Optional deterministic output aliases for the CTE.

        idx: Zero-based index into `select_cols` and `output_aliases`.

        for_cte: When false, array unnest is skipped.

    Returns:

        UNNEST/EXPLODE select expression SQL, or `None` when not applicable.
    """
    if not for_cte or schema is None:
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


def _render_predicate_clause(
    pred: FilterParam | HavingParam,
    dialect: Dialect,
    *,
    is_having: bool = False,
    param_values: Mapping[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """
    Render one WHERE or HAVING predicate into one or more ``(fragment, bool_op)`` tuples.

    Handles case-insensitive string compares, ``date_window``, ``date_diff``, ``contains``, expression-vs-expression, bound parameters, and inline ``raw_value`` binds.

    Date-arithmetic value dicts are looked up via :meth:`FilterParam.resolved_value` so harvested params remain accessible.

    When ``is_having`` is true and no branch matches, emits a placeholder ``?`` bind (legacy HAVING shape).
    """

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
    bool_op = getattr(pred, "bool_op", "AND") or "AND"
    out: list[tuple[str, str]] = []
    if op.lower() in ("is null", "is not null"):
        out.append((f"{left} {op.upper()}", bool_op))
        return out
    resolved = pred.resolved_value(param_values)
    if pred.value_type == "date_window" and isinstance(resolved, dict):
        for dw_frag in _render_date_window_predicate(pred, left, dialect, param_values=param_values):
            out.append((dw_frag, "AND"))
        return out
    if pred.value_type == "date_diff" and isinstance(resolved, dict):
        rv = resolved
        unit = rv.get("unit", "day")
        amount = int(rv.get("amount", 0)) if rv.get("amount") is not None else 0
        hop = pred.op or ">"
        add_parts = [_render_group_sql(g, dialect) for g in pred.left_expr.add_groups]
        sub_parts = [_render_group_sql(g, dialect) for g in pred.left_expr.sub_groups]
        frag = dialect.render_date_diff(
            left,
            hop,
            unit,
            amount,
            minuend_sql=add_parts[0] if add_parts else "",
            subtrahend_sql=sub_parts[0] if sub_parts else "",
        )
        out.append((frag, bool_op))
        return out
    if (pred.op or "").strip().lower() == "contains":
        pk = pred.param_key or "p?"
        frag = dialect.render_array_contains(left, pk)
        out.append((frag, bool_op))
        return out
    if pred.right_expr:
        right = render_expr_sql(pred.right_expr, dialect)
        if case_insensitive:
            right = _wrap_for_case_insensitive(right, dialect)
        out.append((f"{left} {op} {right}", bool_op))
        return out
    if (pred.op or "").lower() in ("in", "not in"):
        pkey = pred.param_key or "p?"
        out.append((f"{left} {op.upper()} (:{pkey})", bool_op))
        return out
    if pred.param_key:
        val_needs_lower = case_insensitive and op.lower() in ("like", "not like")
        val_ref = f"LOWER(:{pred.param_key})" if val_needs_lower else f":{pred.param_key}"
        out.append((f"{left} {op} {val_ref}", bool_op))
        return out
    if pred.raw_value is not None:
        pkey = pred.param_key or "p?"
        val_needs_lower = case_insensitive and op.lower() in ("like", "not like")
        val_ref = f"LOWER(:{pkey})" if val_needs_lower else f":{pkey}"
        out.append((f"{left} {op} {val_ref}", bool_op))
        return out
    if is_having:
        return [(f"{left} {op} ?", bool_op)]
    return []


def _build_deterministic_select_block(
    select_cols: list[SelectCol],
    tables: list[str],
    group_by_cols: list[NormalizedExpr],
    order_by_cols: list,
    filters_param: list,
    having_param: list,
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
) -> str:
    """
    Build a single SELECT block from structured intent clauses.

    Args:

        select_cols: Columns for the SELECT list.

        tables: FROM tables; only ``tables[0]`` (or ``from_table_override``) appears in the
        emitted ``FROM``. Any additional tables are attached as JOIN nodes downstream
        via :func:`inject_join_into_deterministic_sql`.

        group_by_cols: Expressions for GROUP BY.

        order_by_cols: Order-by column specs.

        filters_param: WHERE filter parameters.

        having_param: HAVING filter parameters.

        limit: Optional LIMIT value.

        grain: Grain string (for example row-level vs aggregate).

        output_aliases: Optional AS aliases for CTE output columns.

        dialect: Dialect for rendering helpers.

        schema: Optional schema for array unnest and column metadata.

        for_cte: When true, enables CTE-specific rendering such as array unnest.

        from_table_override: When set, used as the ``FROM`` table instead of ``tables[0]``.

    Returns:

        multiple tables are present.
    """
    lines: list[str] = []

    select_exprs: list[str] = []
    cte_outputs: dict[str, Any] = {}
    for idx, sc in enumerate(select_cols):
        unnest_sql = _maybe_render_array_unnest_select(
            sc,
            schema,
            cte_outputs,
            dialect,
            output_aliases,
            idx,
            for_cte=for_cte,
        )
        if unnest_sql is not None:
            rendered = unnest_sql
        else:
            rendered = render_select_col_sql(sc, dialect)
            if output_aliases and idx < len(output_aliases):
                rendered = f"{rendered} AS {output_aliases[idx]}"
        select_exprs.append(rendered)

    select_keyword = "SELECT DISTINCT" if distinct_select_index >= 0 else "SELECT"
    lines.append(select_keyword + " " + ", ".join(select_exprs))

    if tables:
        from_tbl = from_table_override if from_table_override else tables[0]
        from_sql = dialect.quote_schema_qualified(from_tbl)
        if extra_from_tables:
            pipeline_trace_lazy(
                "pipeline.generate_and_validate_sql.scalar_cte_cross_join",
                lambda: stable_json({"anchor": from_tbl, "scalar_ctes": list(extra_from_tables)}),
            )
            for extra in extra_from_tables:
                from_sql = f"{from_sql} CROSS JOIN {dialect.quote_schema_qualified(extra)}"
        lines.append(f"FROM {from_sql}")

    where_rows: list[tuple[str, str, int | None]] = []
    for fp in filters_param:
        for frag, bop in _render_predicate_clause(fp, dialect, is_having=False, param_values=param_values):
            where_rows.append((frag, bop, fp.filter_group))
    if where_rows:
        lines.append("WHERE " + _join_flat_predicate_parts_with_filter_groups(where_rows))

    if group_by_cols:
        gb_exprs = [render_expr_sql(g, dialect) for g in group_by_cols]
        lines.append("GROUP BY " + ", ".join(gb_exprs))

    having_rows: list[tuple[str, str, int | None]] = []
    for hp in having_param:
        for frag, bop in _render_predicate_clause(hp, dialect, is_having=True, param_values=param_values):
            having_rows.append((frag, bop, hp.filter_group))
    if having_rows:
        lines.append("HAVING " + _join_flat_predicate_parts_with_filter_groups(having_rows))

    if order_by_cols:
        ob_exprs = []
        for obc in order_by_cols:
            rendered = render_expr_sql(obc.expr, dialect)
            direction = obc.direction.upper() if obc.direction else "ASC"
            ob_exprs.append(f"{rendered} {direction}")
        lines.append("ORDER BY " + ", ".join(ob_exprs))

    if limit:
        lines.append(f"LIMIT {limit}")

    return "\n".join(lines)


def _effective_select_col_for_sql(sc: SelectCol) -> SelectCol:
    """
    Return a select column with registry references reduced to the resolved base expression.

    Args:

        sc: Original select column.

    Returns:

        Select column without registry indirection in ``expr``.
    """

    parts = effective_select_parts(sc, None, None)
    return SelectCol(expr=parts.expr)


def generate_col_alias(sc: SelectCol) -> str:
    """
    Build a deterministic display alias from a SelectCol's expression metadata.

    Args:

        sc: Description.

    Returns:

        the caller must assign `col_<idx>`.
    """
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
    """
    Whether :func:`aetherdialect._pipeline.enriched_display_alias_map` should ask the LLM for a display header.

    Args:

        sc: Select column contract.

    Returns:

        True for CASE/window/aggregations/scalar wrappers or when deterministic alias is empty.
    """

    parts = effective_select_parts(sc, None, None)
    if parts.case_when is not None:
        return True
    if parts.window_spec is not None:
        return True
    sc = _effective_select_col_for_sql(sc)
    if sc.is_aggregated:
        return True
    expr = sc.expr
    if expr.scalar_func or expr.inner_scalar_func:
        return True
    if not generate_col_alias(sc):
        return True
    return False


def build_display_sql(
    sql_param: str,
    intent: RuntimeIntent,
    display_alias_map: dict[str, str] | None,
    dialect: Dialect,
) -> str:
    """
    Build display SQL with deterministic aliases via the dialect's AST projection-replace path.

    The projection list flows through :meth:`aetherdialect._dialect.Dialect.replace_projection`
    and re-emitted via :meth:`aetherdialect._dialect.Dialect.emit_sql`. Returns *sql_param* unchanged
    when ``select_cols`` is empty or the AST replace cannot be performed.

    Args:

        sql_param: Parameterized SQL whose ``FROM`` clause and tail are preserved.

        intent: Runtime intent whose ``select_cols`` define the projection.

        display_alias_map: Optional ``signature_key`` to display header overrides.

        dialect: Dialect adapter for AST-driven rewriting; required.

    Returns:

        SQL with the rewritten projection list, or *sql_param* unchanged on no-op or AST failure.
    """

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

    u = str(unit or "day").lower()
    if dialect.name == "postgresql":
        anchor = "CURRENT_TIMESTAMP" if u in {"hour", "minute", "second"} else "CURRENT_DATE"
        return f"{left_rendered} <= {anchor}"
    if dialect.name == "databricks":
        anchor = "current_timestamp()" if u in {"hour", "minute", "second"} else "current_date()"
        return f"{left_rendered} <= {anchor}"
    anchor = "CURRENT_DATE"
    return f"{left_rendered} <= {anchor}"


def _render_date_window_predicate(
    pred: FilterParam | HavingParam,
    left_rendered: str,
    dialect: Dialect,
    *,
    param_values: Mapping[str, Any] | None = None,
) -> list[str]:
    """
    Render WHERE/HAVING clause part(s) for a date_window filter or having condition.

    Args:

        pred: Filter or HAVING row with ``value_type`` ``date_window`` and a dict value (inline ``raw_value`` or harvested into ``param_values``).

        left_rendered: Left-hand SQL for the filtered column.

        dialect: Dialect for date window rendering.

        param_values: Bound parameter map used as a fallback when ``raw_value`` was harvested.

    Returns:

        List of SQL fragments: explicit ``start``/``end`` windows emit two literals; relative
        ``unit``/``amount`` windows emit a lower bound from :meth:`Dialect.render_date_window`
        plus a dialect-specific inclusive upper anchor (for example ``CURRENT_DATE`` / ``current_date()``).
    """

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


def _serialize_join_candidate_row(
    c: dict[str, Any],
    *,
    schema: SchemaGraph | None = None,
) -> dict[str, Any]:
    """
    Return a join-candidate row suitable for join-choice JSON payloads.

    When *schema* is supplied, asserts that no column in the rendered ``join_path_signature`` is hidden by visibility rules. This is a defensive guard against missed filter paths in :func:`enumerate_join_paths_base` / :func:`enumerate_semantic_paths`.
    """

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
                            f"join candidate signature '{seg_str}' references non-visible column {tbl}.{col}",
                        )
    return {
        "candidate_id": c.get("candidate_id"),
        "join_path_signature": c.get("join_path_signature"),
        "edge_kinds": c.get("edge_kinds"),
        "candidate_tier": c.get("candidate_tier"),
    }


def build_join_choice_prompt(
    q_norm: str,
    deterministic_sql: str,
    llm_scopes: list[dict[str, Any]],
    *,
    schema: SchemaGraph | None = None,
    prior_join_feedback: list[str] | None = None,
) -> tuple[str, str]:
    """
    Build minimal prompt for LLM to return per-scope join candidate IDs.

    Args:

        q_norm: Normalised natural-language question.

        deterministic_sql: Template SQL whose joins must be chosen.

        llm_scopes: Each entry has ``scope``, ``tables``, and ``candidates`` for one join scope.

        schema: Optional schema graph forwarded to ``_serialize_join_candidate_row`` so that
        non-visible columns surfacing in any candidate signature raise ``AssertionError``.

        prior_join_feedback: Optional user rejection summaries for this question.

    Returns:

        System and user strings for ``llm_json``.
    """
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
                "candidates": [_serialize_join_candidate_row(c, schema=schema) for c in cands if isinstance(c, dict)],
            }
        )
    user = stable_json(
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
        }
    )
    return system, user


def _valid_join_choice_ids_from_candidates(
    candidates: list[dict[str, Any]],
) -> frozenset[str]:
    """Collect stripped non-empty ``candidate_id`` strings from a candidate list."""

    out: set[str] = set()
    for c in candidates or []:
        cid = c.get("candidate_id")
        if isinstance(cid, str) and cid.strip():
            out.add(cid.strip())
    return frozenset(out)


def _valid_main_join_candidate_ids(join_candidates: dict[str, Any]) -> frozenset[str]:
    """
    Collect non-empty ``candidate_id`` strings from the main join hint payload.

    Args:

        join_candidates: Dict with a ``candidates`` list from ``join_hints_multi``.

    Returns:

        Frozenset of allowed main-query join candidate ids.
    """

    return _valid_join_choice_ids_from_candidates(join_candidates.get("candidates") or [])


def _valid_cte_join_candidate_ids(
    cte_join_hints: dict[str, dict[str, Any]] | None,
) -> dict[str, frozenset[str]]:
    """
    Map each CTE name to the set of allowed ``candidate_id`` strings for that CTE.

    Args:

        cte_join_hints: Per-CTE join hint payloads or ``None``.

    Returns:

        CTE name to allowed ids; empty when *cte_join_hints* is ``None``.
    """

    if not cte_join_hints:
        return {}
    result: dict[str, frozenset[str]] = {}
    for cte, h in cte_join_hints.items():
        result[cte] = _valid_join_choice_ids_from_candidates(h.get("candidates") or [])
    return result


def _parse_join_choice_payload(parsed: dict[str, Any]) -> dict[str, str]:
    """
    Extract per-scope join choices from an LLM JSON object.

    Args:

        parsed: Raw dict from the join-selector model.

    Returns:

        Scope key to stripped candidate id or ``NA``.
    """

    raw = parsed.get("choices")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _scope_choice_valid_ids(
    candidates: list[dict[str, Any]],
    *,
    allow_na: bool,
) -> frozenset[str]:
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
    merged: dict[str, str],
    required_scopes: frozenset[str],
    llm_scopes: list[dict[str, Any]],
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
    """
    Call LLM to get per-scope join candidate ids for the listed scopes.

    Args:

        q_norm: Normalised natural-language question.

        deterministic_sql: Template SQL sent to the join selector.

        llm_scopes: Scopes sent to the model; each dict has ``scope``, ``tables``, ``candidates``.

        preset_choices: Deterministic preset ids merged before validation.

        accept_na_by_scope: Whether ``NA`` is accepted for each scope key on this pass.

        require_final: When true, only non-``NA`` ids that appear in each scope's candidate list pass.

        schema: Schema graph for serialization guards.

        prior_join_feedback: Summaries of earlier wrong-table/join rejections for this question.

    Returns:

        Merged scope-to-id map including presets. After retries, each unresolved scope picks the
        first non-``J00`` candidate in stable id order, or raises :class:`NoJoinPathError` when the
        scope spans multiple tables and no join path exists.
    """

    preset = dict(preset_choices or {})
    accept_na = dict(accept_na_by_scope or {})
    if not llm_scopes:
        return preset
    required = frozenset(str(s["scope"]) for s in llm_scopes if s.get("scope") is not None)
    for _attempt in range(2):
        try:
            system, user = build_join_choice_prompt(
                q_norm,
                deterministic_sql,
                llm_scopes,
                schema=schema,
                prior_join_feedback=prior_join_feedback,
            )
            parsed = llm_json(system, user, retries=1, task="join")
        except LlmJsonExhausted as exc:
            debug(f"[sql_gen.get_join_choice_from_llm] exhausted attempt {_attempt + 1}: {exc}")
            continue
        if not isinstance(parsed, dict):
            continue
        raw = _parse_join_choice_payload(parsed)
        merged = dict(preset)
        for sk in required:
            if sk not in raw:
                continue
            val = raw[sk]
            allow_na_here = accept_na.get(sk, False)
            cands = next(
                (x.get("candidates") or [] for x in llm_scopes if str(x.get("scope")) == sk),
                [],
            )
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
        cands = next(
            (x.get("candidates") or [] for x in llm_scopes if str(x.get("scope")) == sk),
            [],
        )
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
