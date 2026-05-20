"""SQL to RuntimeIntent conversion plus query-log fetching for SQL-history warmup."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import sqlglot
from sqlglot import exp as sqlglot_exp

try:
    import pglast
except ImportError:
    pglast = None

from ._config import (
    PG_LAST_WINDOW_FRAME_OPTIONS_INLINE_DEFAULT,
    PG_LAST_WINDOW_FRAME_OPTIONS_RANGE_UNBOUNDED_CURRENT,
    PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_OFFSET_CURRENT,
    PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_UNBOUNDED_PAIR,
    SQLGLOT_DIALECT_BY_ENGINE,
    SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY,
    SQL_TO_INTENT_LITERAL_PLACEHOLDER_DATE,
    SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM,
    SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR,
    SQL_TO_INTENT_PARAM_KEY_PREFIX,
    SELF_JOIN_CTE_NAME_PREFIX,
    WARMUP_PARAPHRASE_COUNT_FROM_SQL,
    WARMUP_ROUND_TRIP_CARDINALITY_TOLERANCE,
    WARMUP_ROUND_TRIP_LIMIT,
    WINDOW_DEFAULT_FRAME_END_WITHOUT_ORDER,
    WINDOW_DEFAULT_FRAME_END_WITH_ORDER,
    WINDOW_DEFAULT_FRAME_KIND_WITHOUT_ORDER,
    WINDOW_DEFAULT_FRAME_KIND_WITH_ORDER,
    WINDOW_DEFAULT_FRAME_START_WITHOUT_ORDER,
    WINDOW_DEFAULT_FRAME_START_WITH_ORDER,
    SeedWarmupConfig,
)
from ._contracts_base import SchemaGraph
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
    SeedWarmupIntent,
    SelectCol,
    WindowRegistryStep,
    WindowSpec,
    runtime_intent_to_concrete,
)
from ._core_utils import sha256, stable_json
from ._dialect import (
    DatabricksDialect,
    PostgresDialect,
    _pg_columnref_to_pair,
    _pg_funcname,
)
from ._intent_process import (
    cte_structural_signature,
    union_runtime_concrete_compatibility,
)
from ._sql_gen import build_deterministic_sql
from ._utils import body_similarity_key, sql_shape
from sqlalchemy import text


@dataclass
class PgExtra:
    """
    Per-select conversion scratch space for qualifier swaps and registry emission.

    Args:

        qual_swap: Maps an original range alias to the qualifier introduced when a self-join branch is lifted into a CTE.

        case_registry: Collected CASE registry rows emitted while converting projections.

        window_registry: Collected window registry rows emitted while converting projections.

        case_counter: Monotonic counter used for deterministic ``cNN`` identifiers.

        window_counter: Monotonic counter used for deterministic ``wNN`` identifiers.

        self_join_steps: ``RuntimeCteStep`` rows emitted when a repeated physical table is lifted before parsing the body.
    """

    qual_swap: dict[str, str] = field(default_factory=dict)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    self_join_steps: list[RuntimeCteStep] = field(default_factory=list)
    case_counter: int = 0
    window_counter: int = 0

    def next_case_id(self) -> str:
        """Reserve the next sequential CASE registry identifier."""

        self.case_counter += 1
        return f"c{self.case_counter:02d}"

    def next_window_id(self) -> str:
        """Reserve the next sequential window registry identifier."""

        self.window_counter += 1
        return f"w{self.window_counter:02d}"


def _normalize_window_spec_ansi_defaults(ws: WindowSpec) -> WindowSpec:
    """
    Fill ANSI-equivalent explicit ROWS bounds when the frame was omitted.

    Args:

        ws: Parsed window specification.

    Returns:

        Updated spec when ``frame_kind`` was ``none``; otherwise *ws* unchanged.
    """

    if ws.frame_kind != "none":
        return ws
    if ws.order_by:
        return replace(
            ws,
            frame_kind=WINDOW_DEFAULT_FRAME_KIND_WITH_ORDER,
            frame_start=WINDOW_DEFAULT_FRAME_START_WITH_ORDER,
            frame_end=WINDOW_DEFAULT_FRAME_END_WITH_ORDER,
        )
    return replace(
        ws,
        frame_kind=WINDOW_DEFAULT_FRAME_KIND_WITHOUT_ORDER,
        frame_start=WINDOW_DEFAULT_FRAME_START_WITHOUT_ORDER,
        frame_end=WINDOW_DEFAULT_FRAME_END_WITHOUT_ORDER,
    )


def _normalize_window_registry_step(w: WindowRegistryStep) -> WindowRegistryStep:
    """Apply ANSI default frames to one window registry row."""

    return replace(w, window_spec=_normalize_window_spec_ansi_defaults(w.window_spec))


def _normalize_runtime_cte_step_windows(step: RuntimeCteStep) -> RuntimeCteStep:
    """Normalise window frames on a single CTE branch."""

    wr = [_normalize_window_registry_step(x) for x in (step.window_registry or [])]
    return replace(step, window_registry=wr)


def _cte_body_shell_key(step: RuntimeCteStep) -> str:
    """Structural CTE body fingerprint excluding SELECT projections (matches union clustering)."""

    parts: list[str] = [
        step.grain or "row_level",
        ",".join(sorted(step.tables or [])),
        ",".join(sorted(f.signature_key for f in (step.filters_param or []))),
        ",".join(sorted(g.signature_key for g in (step.group_by_cols or []))),
        ",".join(sorted(o.signature_key for o in (step.order_by_cols or []))),
        ",".join(sorted(h.signature_key for h in (step.having_param or []))),
        ",".join(sorted(s.signature_key for s in (step.window_registry or []))),
        ",".join(sorted(s.signature_key for s in (step.case_registry or []))),
    ]
    return "|".join(parts)


def _runtime_intent_from_cte_step(step: RuntimeCteStep) -> RuntimeIntent:
    """Project a CTE body onto a runtime intent shell for union eligibility checks."""

    return RuntimeIntent(
        tables=list(step.tables or []),
        grain=step.grain or "row_level",
        select_cols=list(step.select_cols or []),
        group_by_cols=list(step.group_by_cols or []),
        order_by_cols=list(step.order_by_cols or []),
        filters_param=list(step.filters_param or []),
        having_param=list(step.having_param or []),
        param_values={},
        cte_steps=[],
        natural_language="",
        limit=step.limit,
        column_map=dict(step.column_map or {}),
        chosen_join_candidate_id=step.chosen_join_candidate_id or "",
        chosen_join_path_signature=list(step.chosen_join_path_signature or []),
        window_registry=list(step.window_registry or []),
        case_registry=list(step.case_registry or []),
        distinct_select_index=int(step.distinct_select_index),
    )


def _merge_union_compatible_cte_cluster(
    cluster: list[RuntimeCteStep],
) -> list[RuntimeCteStep]:
    """Greedy union-merge of CTE steps sharing :func:`_cte_body_shell_key`."""

    if len(cluster) <= 1:
        return list(cluster)
    pending = list(cluster)
    changed = True
    while changed and len(pending) > 1:
        changed = False
        for i in range(len(pending)):
            for j in range(i + 1, len(pending)):
                a, b = pending[i], pending[j]
                ra = _runtime_intent_from_cte_step(a)
                rb = _runtime_intent_from_cte_step(b)
                conc = runtime_intent_to_concrete(ra, "cte_dedup")
                row = union_runtime_concrete_compatibility(rb, conc)
                if row is None:
                    continue
                union_cols = row[0]
                merged_name = a.cte_name if a.cte_name <= b.cte_name else b.cte_name
                merged_step = replace(
                    a,
                    cte_name=merged_name,
                    select_cols=list(union_cols),
                )
                pending = [merged_step] + [pending[k] for k in range(len(pending)) if k not in (i, j)]
                changed = True
                break
            if changed:
                break
    return pending


def _dedup_cte_steps(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Group CTE steps by structural family and union-merge where the bare-only diff rule passes.

    Args:

        intent: Intent possibly carrying duplicate CTE families.

    Returns:

        Intent with merged CTE steps when partitions qualify under union policy.
    """

    steps_in = list(intent.cte_steps or [])
    if not steps_in:
        return intent
    steps = [_normalize_runtime_cte_step_windows(s) for s in steps_in]
    if len(steps) < 2:
        return replace(intent, cte_steps=steps)
    buckets: dict[str, list[RuntimeCteStep]] = {}
    bucket_order: list[str] = []
    for st in steps:
        bk = _cte_body_shell_key(st)
        if bk not in buckets:
            bucket_order.append(bk)
            buckets[bk] = []
        buckets[bk].append(st)
    merged_steps: list[RuntimeCteStep] = []
    for bk in bucket_order:
        merged_steps.extend(_merge_union_compatible_cte_cluster(buckets[bk]))
    merged_steps.sort(key=lambda s: s.cte_name)
    return replace(intent, cte_steps=merged_steps)


def _dedup_window_registry(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Normalise omitted window frames to ANSI defaults before deduplicating registry rows.

    Args:

        intent: Intent carrying window registry entries.

    Returns:

        Intent with normalised frames and de-duplicated ``window_registry``.
    """

    wr_in = list(intent.window_registry or [])
    wr = [_normalize_window_registry_step(w) for w in wr_in]
    if not wr:
        return intent
    seen: set[str] = set()
    kept: list[WindowRegistryStep] = []
    for w in wr:
        k = w.window_spec.signature_key
        if k in seen:
            continue
        seen.add(k)
        kept.append(w)
    return replace(intent, window_registry=kept)


def _dedup_case_registry(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Strict-equality deduplication of CASE registry entries by canonical signature keys.

    Args:

        intent: Intent carrying case registry entries.

    Returns:

        Intent with duplicate CASE registry rows removed.
    """

    cr = list(intent.case_registry or [])
    if len(cr) < 2:
        return intent
    seen: set[str] = set()
    kept: list[Any] = []
    for c in cr:
        k = c.signature_key
        if k in seen:
            continue
        seen.add(k)
        kept.append(c)
    if len(kept) == len(cr):
        return intent
    return replace(intent, case_registry=kept)


@dataclass(frozen=True)
class ConverterResult:
    """One SQL string converted to a RuntimeIntent or a structured failure."""

    sql_hash: str
    intent: RuntimeIntent | None
    failure_code: str | None
    failure_detail: str


def _hash_sql(sql: str) -> str:
    """Return a stable hexadecimal digest for *sql*."""

    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _infer_sqlglot_read(dialect: Any) -> str:
    """
    Map a live :class:`~aetherdialect._dialect.Dialect` instance to a sqlglot ``read`` dialect name.

    Args:

        dialect: Active dialect wrapper.

    Returns:

        Lowercase sqlglot reader key.
    """

    if isinstance(dialect, PostgresDialect):
        return str(SQLGLOT_DIALECT_BY_ENGINE.get("postgresql", "postgres"))
    if isinstance(dialect, DatabricksDialect):
        return str(SQLGLOT_DIALECT_BY_ENGINE.get("databricks", "spark"))
    return "postgres"


def _expr_from_sqlglot_projection(
    node: sqlglot_exp.Expression,
) -> NormalizedExpr | None:
    """
    Build a minimal :class:`NormalizedExpr` leaf from a sqlglot SELECT projection.

    Args:

        node: Projection expression (possibly wrapped in Alias).

    Returns:

        Normalised expression when a plain column or star-free literal leaf is recognised.
    """

    if isinstance(node, sqlglot_exp.Alias):
        node = node.this
    if isinstance(node, sqlglot_exp.Column):
        qual = f"{node.table}.{node.name}" if node.table else str(node.name)
        return NormalizedExpr.from_column(qual)
    if isinstance(node, sqlglot_exp.Literal):
        return NormalizedExpr(string_literal=str(node.this))
    return None


def _collect_tables_from_select(sel: sqlglot_exp.Select) -> list[str]:
    """
    Collect base table names referenced by a sqlglot SELECT (best-effort, unqualified names).

    Args:

        sel: Parsed SELECT root.

    Returns:

        Sorted unique physical table identifiers.
    """

    names: set[str] = set()
    for t in sel.find_all(sqlglot_exp.Table):
        if t.name:
            names.add(str(t.name))
    return sorted(names)


def _runtime_from_sqlglot_select(sel: sqlglot_exp.Select, schema: SchemaGraph) -> RuntimeIntent | None:
    """
    Materialise a :class:`RuntimeIntent` from a sqlglot SELECT when projections are simple columns or literals.

    Args:

        sel: Parsed SELECT.

        schema: Schema graph for reference validation.

    Returns:

        Intent or ``None`` when constructs are not supported in this conversion tier.
    """

    if sel.find(sqlglot_exp.Union):
        return None
    if sel.find(sqlglot_exp.With):
        return None
    for sq in sel.find_all(sqlglot_exp.Subquery):
        if sq.parent and isinstance(sq.parent, (sqlglot_exp.From, sqlglot_exp.Join)):
            return None
    tables = _collect_tables_from_select(sel)
    for t in tables:
        if t not in schema.tables:
            return None
    select_cols: list[SelectCol] = []
    for proj in sel.expressions:
        ex = _expr_from_sqlglot_projection(proj)
        if ex is None:
            return None
        select_cols.append(SelectCol(expr=ex))
    grain = "row_level"
    limit_val = None
    if sel.args.get("limit"):
        lim = sel.args["limit"]
        if isinstance(lim, sqlglot_exp.Limit) and isinstance(lim.expression, sqlglot_exp.Literal):
            try:
                limit_val = int(str(lim.expression.this))
            except (TypeError, ValueError):
                limit_val = None
    return RuntimeIntent(
        tables=tables,
        grain=grain,
        select_cols=select_cols,
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        param_values={},
        cte_steps=[],
        natural_language="",
        limit=limit_val,
    )


def _runtime_from_pglast_sql(sql: str, schema: SchemaGraph) -> RuntimeIntent | None:
    """
    Best-effort :mod:`pglast` SELECT → :class:`RuntimeIntent` extraction.

    Unsupported constructs return ``None`` so callers can fall back to sqlglot.
    """

    if pglast is None:
        return None
    from pglast.ast import RawStmt

    try:
        stmts = pglast.parse_sql(sql)
    except Exception:
        return None
    if len(stmts) != 1:
        return None
    raw = stmts[0]
    stmt = raw.stmt if isinstance(raw, RawStmt) else raw
    param_store: dict[str, Any] = {}
    counter_holder = [0]

    def next_lit_key() -> str:
        counter_holder[0] += 1
        return f"{SQL_TO_INTENT_PARAM_KEY_PREFIX}{counter_holder[0]}"

    out = _pg_convert_pg_statement(stmt, schema, param_store, next_lit_key)
    if out is None:
        return None
    intent, cte_steps = out
    tables_agg: set[str] = set(intent.tables or [])
    for st in cte_steps:
        tables_agg.update(st.tables or [])
    intent = replace(
        intent,
        tables=sorted(tables_agg),
        param_values=dict(param_store),
        cte_steps=cte_steps,
    )
    return intent


def _pg_convert_pg_statement(
    stmt: Any,
    schema: SchemaGraph,
    param_store: dict[str, Any],
    next_lit_key: Any,
) -> tuple[RuntimeIntent, list[RuntimeCteStep]] | None:
    """Dispatch top-level ``SELECT`` (optionally wrapped in ``WITH``)."""

    from pglast.ast import SelectStmt

    if not isinstance(stmt, SelectStmt):
        return None
    return _pg_convert_select_stmt(stmt, schema, param_store, next_lit_key)


def _pg_convert_select_stmt(
    sel: Any,
    schema: SchemaGraph,
    param_store: dict[str, Any],
    next_lit_key: Any,
    *,
    outer_allowed_cte: frozenset[str] | None = None,
) -> tuple[RuntimeIntent, list[RuntimeCteStep]] | None:
    """Convert one ``SelectStmt``, recursively stripping ``WITH`` into ``RuntimeCteStep`` rows."""

    from pglast.ast import CommonTableExpr, SelectStmt

    allowed_cte = set(outer_allowed_cte or ())
    cte_steps: list[RuntimeCteStep] = []
    if getattr(sel, "withClause", None):
        wc = sel.withClause
        if getattr(wc, "recursive", False):
            return None
        prior: set[str] = set()
        for cte in wc.ctes or ():
            if not isinstance(cte, CommonTableExpr):
                return None
            name = str(cte.ctename or "").strip()
            inner = cte.ctequery
            if not isinstance(inner, SelectStmt):
                return None
            inner_allowed = frozenset(prior | allowed_cte)
            chunk = _pg_materialize_cte_body(inner, schema, name, param_store, next_lit_key, inner_allowed)
            if chunk is None:
                return None
            cte_steps.extend(chunk)
            prior.add(name)
        allowed_cte = allowed_cte | prior

    body_pair = _pg_runtime_intent_body_from_select(
        sel, schema, param_store, next_lit_key, frozenset(allowed_cte), PgExtra()
    )
    body, sj_body = body_pair
    if body is None:
        return None
    return body, sj_body + cte_steps


def _pg_materialize_cte_body(
    sel: Any,
    schema: SchemaGraph,
    cte_name: str,
    param_store: dict[str, Any],
    next_lit_key: Any,
    allowed_cte: frozenset[str],
) -> list[RuntimeCteStep] | None:
    """
    Flatten a CTE body ``SelectStmt`` into ordered ``RuntimeCteStep`` rows.

    Nested non-recursive ``WITH`` inside the body is expanded into preceding steps, then the named CTE shell.
    """

    from pglast.ast import CommonTableExpr, SelectStmt

    if not isinstance(sel, SelectStmt):
        return None

    nested_steps: list[RuntimeCteStep] = []
    allowed_here: set[str] = set(allowed_cte)

    if getattr(sel, "withClause", None):
        wc = sel.withClause
        if getattr(wc, "recursive", False):
            return None
        prior: set[str] = set()
        for cte in wc.ctes or ():
            if not isinstance(cte, CommonTableExpr):
                return None
            nm = str(cte.ctename or "").strip()
            inner_allowed = frozenset(prior | allowed_here)
            sub = _pg_materialize_cte_body(cte.ctequery, schema, nm, param_store, next_lit_key, inner_allowed)
            if sub is None:
                return None
            nested_steps.extend(sub)
            prior.add(nm)
        allowed_here |= prior

    body_pair = _pg_runtime_intent_body_from_select(
        sel, schema, param_store, next_lit_key, frozenset(allowed_here), PgExtra()
    )
    body, sj_local = body_pair
    if body is None:
        return None
    step = _pg_runtime_cte_step_from_body(sel, body, cte_name)
    return nested_steps + sj_local + [step]


def _pg_runtime_cte_step_from_body(sel: Any, body: RuntimeIntent, cte_name: str) -> RuntimeCteStep:
    """Wrap a converted CTE ``RuntimeIntent`` fragment as ``RuntimeCteStep``."""

    output_columns = _pg_infer_output_columns(sel)
    return RuntimeCteStep(
        cte_name=cte_name,
        tables=list(body.tables or []),
        select_cols=list(body.select_cols or []),
        group_by_cols=list(body.group_by_cols or []),
        order_by_cols=list(body.order_by_cols or []),
        filters_param=list(body.filters_param or []),
        having_param=list(body.having_param or []),
        param_values={},
        output_columns=output_columns,
        grain=body.grain or "row_level",
        limit=body.limit,
        limit_param_key=body.limit_param_key or "",
        distinct_select_index=int(body.distinct_select_index),
    )


def _pg_infer_output_columns(sel: Any) -> list[str]:
    """Derive snake_case-ish output names from ``ResTarget`` aliases."""

    names: list[str] = []
    for i, rt in enumerate(sel.targetList or ()):
        nm = getattr(rt, "name", None)
        if nm:
            names.append(str(nm).strip())
        else:
            names.append(f"col_{i}")
    return names


def _pg_collect_range_bindings(
    from_clause: Any,
    schema_tables: set[str],
    allowed_cte: frozenset[str],
) -> list[tuple[str, str]] | None:
    """
    Collect ordered ``(alias, relation_target)`` pairs from a ``FROM`` clause.

    Args:

        from_clause: ``SelectStmt.fromClause`` sequence.

        schema_tables: Known physical relation identifiers.

        allowed_cte: CTE names visible in this scope.

    Returns:

        Binding list or ``None`` when an unsupported join construct appears.
    """

    from pglast.ast import JoinExpr, RangeSubselect, RangeVar
    from pglast.enums import JoinType

    allowed_joins = frozenset(
        {
            JoinType.JOIN_INNER,
            JoinType.JOIN_LEFT,
            JoinType.JOIN_RIGHT,
            JoinType.JOIN_FULL,
        }
    )
    bindings: list[tuple[str, str]] = []

    def walk(node: Any) -> bool:
        if isinstance(node, JoinExpr):
            if node.jointype not in allowed_joins:
                return False
            return walk(node.larg) and walk(node.rarg)
        if isinstance(node, RangeSubselect):
            return False
        if isinstance(node, RangeVar):
            rel = str(node.relname or "").strip()
            if not rel:
                return False
            if rel not in schema_tables and rel not in allowed_cte:
                return False
            alias = rel
            if node.alias and getattr(node.alias, "aliasname", None):
                alias = str(node.alias.aliasname).strip()
            bindings.append((alias, rel))
            return True
        return False

    for item in from_clause or ():
        if not walk(item):
            return None
    return bindings


def _pg_bindings_to_alias_map(bindings: list[tuple[str, str]]) -> dict[str, str]:
    """Materialise a plain alias→target map from ordered bindings."""

    out: dict[str, str] = {}
    for alias, target in bindings:
        out[alias] = target
    return out


def _pg_count_alias_refs(sel: Any, alias: str) -> int:
    """
    Count ``ColumnRef`` qualifiers equal to *alias* anywhere under *sel*.

    Args:

        sel: ``SelectStmt`` subtree.

        alias: Table or range alias to score.

    Returns:

        Non-negative hit count.
    """

    from pglast.ast import ColumnRef

    n = 0

    def visit(node: Any) -> None:
        nonlocal n
        if isinstance(node, ColumnRef):
            pr = _pg_columnref_to_pair(node)
            if pr and pr[0] == alias:
                n += 1

    def walk(x: Any) -> None:
        if x is None:
            return
        visit(x)
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
            return
        if isinstance(x, (int, float, str, bool)):
            return
        for name in getattr(x, "__slots__", ()):
            if name in ("location", "location2"):
                continue
            walk(getattr(x, name, None))

    walk(sel)
    return n


def _pg_rewrite_rangevar_to_cte(
    from_clause: Any,
    lift_alias: str,
    physical: str,
    cte_name: str,
) -> bool:
    """
    Rewrite the lifted ``RangeVar`` so it references the synthetic CTE name.

    Args:

        from_clause: ``SelectStmt.fromClause`` sequence.

        lift_alias: Alias chosen for the lifted branch.

        physical: Physical relation being duplicated.

        cte_name: Synthetic ``WITH`` identifier.

    Returns:

        True when a matching range was rewritten.
    """

    from pglast.ast import Alias, JoinExpr, RangeVar
    from pglast.enums import JoinType

    allowed_joins = frozenset(
        {
            JoinType.JOIN_INNER,
            JoinType.JOIN_LEFT,
            JoinType.JOIN_RIGHT,
            JoinType.JOIN_FULL,
        }
    )
    changed = False

    def walk(node: Any) -> None:
        nonlocal changed
        if isinstance(node, JoinExpr):
            if node.jointype not in allowed_joins:
                return
            walk(node.larg)
            walk(node.rarg)
            return
        if isinstance(node, RangeVar):
            rel = str(node.relname or "").strip()
            al = rel
            if node.alias and getattr(node.alias, "aliasname", None):
                al = str(node.alias.aliasname).strip()
            if al == lift_alias and rel == physical:
                node.relname = cte_name
                node.schemaname = None
                node.alias = Alias(aliasname=cte_name)
                changed = True

    for item in from_clause or ():
        walk(item)
    return changed


def _pg_try_lift_self_join(sel: Any, schema_tables: set[str], allowed_cte: frozenset[str], pgx: PgExtra) -> bool:
    """
    Detect a two-alias self-join on one physical table and lift the lighter branch into a CTE step.

    Args:

        sel: ``SelectStmt`` being converted.

        schema_tables: Known physical tables.

        allowed_cte: Visible CTE identifiers.

        pgx: Mutable conversion scratch space.

    Returns:

        False when the statement must be rejected at the pglast tier.
    """

    from pglast.ast import SelectStmt

    if not isinstance(sel, SelectStmt) or not getattr(sel, "fromClause", None):
        return True
    bindings = _pg_collect_range_bindings(sel.fromClause, schema_tables, allowed_cte)
    if bindings is None:
        return False
    physical_groups: dict[str, list[str]] = {}
    for alias, tgt in bindings:
        if tgt in schema_tables:
            physical_groups.setdefault(tgt, []).append(alias)
    dup_targets = [t for t, als in physical_groups.items() if len(als) > 1]
    if not dup_targets:
        return True
    if len(dup_targets) != 1:
        return False
    phys = dup_targets[0]
    aliases = physical_groups[phys]
    if len(aliases) != 2:
        return False
    a0, a1 = aliases[0], aliases[1]
    n0, n1 = _pg_count_alias_refs(sel, a0), _pg_count_alias_refs(sel, a1)
    lift = a0 if n0 < n1 else a1
    cte_name = f"{SELF_JOIN_CTE_NAME_PREFIX}{lift}"
    if cte_name in allowed_cte or cte_name in schema_tables:
        return False
    if not _pg_rewrite_rangevar_to_cte(sel.fromClause, lift, phys, cte_name):
        return False
    pgx.qual_swap[lift] = cte_name
    pgx.self_join_steps.append(
        RuntimeCteStep(
            cte_name=cte_name,
            tables=[phys],
            select_cols=[SelectCol(expr=NormalizedExpr(star=True))],
            output_columns=["col_0"],
            grain="row_level",
            distinct_select_index=-1,
        )
    )
    return True


def _pg_runtime_intent_body_from_select(
    sel: Any,
    schema: SchemaGraph,
    param_store: dict[str, Any],
    next_lit_key: Any,
    allowed_cte: frozenset[str],
    pg_extra: PgExtra | None = None,
) -> tuple[RuntimeIntent | None, list[RuntimeCteStep]]:
    """Convert ``SelectStmt`` core fields when ``WITH`` was already lifted."""

    from pglast.ast import A_Const, ResTarget, SelectStmt
    from pglast.enums import SetOperation

    if not isinstance(sel, SelectStmt):
        return None, []
    if sel.op != SetOperation.SETOP_NONE:
        return None, []
    if getattr(sel, "windowClause", None):
        return None, []
    if getattr(sel, "intoClause", None):
        return None, []
    schema_tables = set(schema.tables.keys())
    pgx = pg_extra if pg_extra is not None else PgExtra()
    if not _pg_try_lift_self_join(sel, schema_tables, allowed_cte, pgx):
        return None, []
    extra_allowed = frozenset(set(allowed_cte) | {s.cte_name for s in pgx.self_join_steps})
    bindings = _pg_collect_range_bindings(sel.fromClause, schema_tables, extra_allowed)
    if bindings is None:
        return None, []
    alias_map = _pg_bindings_to_alias_map(bindings)

    scope_aliases = sorted(alias_map.keys())
    single_alias: str | None = scope_aliases[0] if len(scope_aliases) == 1 else None

    physical_tables = sorted({alias_map[a] for a in alias_map if alias_map[a] in schema_tables})
    for rel in physical_tables:
        if rel not in schema_tables:
            return None, []

    select_cols: list[SelectCol] = []
    for rt in sel.targetList or ():
        if not isinstance(rt, ResTarget):
            return None, []
        ex = _pg_res_target_to_expr(rt, alias_map, single_alias, param_store, next_lit_key, pgx, select_cols)
        if ex is None:
            return None, []
        select_cols.append(SelectCol(expr=ex))

    distinct_idx = _pg_distinct_select_index(sel, alias_map, single_alias, select_cols, pgx)

    filters: list[FilterParam] = []
    if getattr(sel, "whereClause", None):
        fp = _pg_where_to_filters(sel.whereClause, alias_map, single_alias, param_store, next_lit_key, pgx)
        if fp is None:
            return None, []
        filters = fp

    group_by_cols: list[NormalizedExpr] = []
    if getattr(sel, "groupClause", None):
        gb = _pg_group_clause(
            sel.groupClause,
            alias_map,
            single_alias,
            select_cols,
            param_store,
            next_lit_key,
            pgx,
        )
        if gb is None:
            return None, []
        group_by_cols = gb

    order_by_cols: list[OrderByCol] = []
    if getattr(sel, "sortClause", None):
        ob = _pg_sort_clause(
            sel.sortClause,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            select_cols,
            pgx,
        )
        if ob is None:
            return None, []
        order_by_cols = ob

    having_param: list[HavingParam] = []
    if getattr(sel, "havingClause", None):
        hp = _pg_having_to_params(sel.havingClause, alias_map, single_alias, param_store, next_lit_key, pgx)
        if hp is None:
            return None, []
        having_param = hp

    limit_val: int | None = None
    off_node = getattr(sel, "limitOffset", None)
    if off_node is not None:
        if not isinstance(off_node, A_Const):
            return None, []
        off_i = _pg_const_int_only(off_node)
        if off_i is None:
            return None, []
        param_store[SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY] = off_i

    lc = getattr(sel, "limitCount", None)
    if lc is not None:
        if not isinstance(lc, A_Const):
            return None, []
        lim_i = _pg_const_int_only(lc)
        if lim_i is None:
            return None, []
        limit_val = lim_i

    grain = "row_level"
    intent = RuntimeIntent(
        tables=physical_tables,
        grain=grain,
        select_cols=select_cols,
        group_by_cols=group_by_cols,
        order_by_cols=order_by_cols,
        filters_param=filters,
        having_param=having_param,
        param_values={},
        cte_steps=[],
        natural_language="",
        limit=limit_val,
        distinct_select_index=distinct_idx,
        window_registry=list(pgx.window_registry),
        case_registry=list(pgx.case_registry),
    )
    return intent, list(pgx.self_join_steps)


def _pg_build_alias_map(
    from_clause: Any,
    schema_tables: set[str],
    allowed_cte: frozenset[str],
) -> dict[str, str] | None:
    """Map ``RangeVar`` qualifier → underlying relation name (physical table or CTE id)."""

    bindings = _pg_collect_range_bindings(from_clause, schema_tables, allowed_cte)
    if bindings is None:
        return None
    return _pg_bindings_to_alias_map(bindings)


def _pg_qual_column_name(
    colref: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    pgx: PgExtra | None = None,
) -> str | None:
    """Turn ``ColumnRef`` into ``qual.col`` using FROM aliases."""

    pair = _pg_columnref_to_pair(colref)
    if pair is None:
        return None
    pre, col = pair
    if pre is not None and pgx and pre in pgx.qual_swap:
        pre = pgx.qual_swap[pre]
    if pre is None:
        if single_alias:
            return f"{single_alias}.{col}"
        return None
    if pre in alias_map:
        return f"{pre}.{col}"
    return None


def _pg_map_where_scalar_op(op_raw: str | None) -> str | None:
    """Map pg ``A_Expr`` operator token strings to normalized ``FilterParam`` / ``HavingParam`` ops."""

    if op_raw is None:
        return None
    key = str(op_raw).strip().lower()
    return {
        "=": "=",
        "==": "=",
        "<>": "<>",
        "!=": "<>",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
        "~~": "like",
        "!~~": "not like",
        "~~*": "ilike",
        "!~~*": "not ilike",
    }.get(key)


def _pg_distinct_select_index(
    sel: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    select_cols: list[SelectCol],
    pgx: PgExtra | None = None,
) -> int:
    """Derive ``distinct_select_index`` from ``distinctClause`` when parsable."""

    from pglast.ast import ColumnRef

    dc = getattr(sel, "distinctClause", None)
    if not dc:
        return -1
    if len(dc) == 1 and dc[0] is None:
        return 0
    for dexpr in dc:
        if isinstance(dexpr, ColumnRef):
            qn = _pg_qual_column_name(dexpr, alias_map, single_alias, pgx)
            if qn:
                for j, sc in enumerate(select_cols):
                    cr = (sc.expr.column_ref or "").strip()
                    if cr == qn:
                        return j
    return 0


_PG_SIMPLE_AGG_NAMES: frozenset[str] = frozenset({"count", "sum", "avg", "min", "max"})


def _pg_having_to_params(
    having: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> list[HavingParam] | None:
    """Extract ``HavingParam`` rows from ``HAVING`` (``AND`` chains; aggregate left, literal/column right)."""

    from pglast.ast import A_Expr, BoolExpr, ColumnRef, FuncCall, NullTest, SubLink
    from pglast.enums import A_Expr_Kind, BoolExprType, NullTestType

    if isinstance(having, BoolExpr) and having.boolop != BoolExprType.AND_EXPR:
        return None
    parts = _pg_flatten_bool_and(having)
    out: list[HavingParam] = []
    for p in parts:
        if isinstance(p, SubLink):
            return None
        if isinstance(p, NullTest):
            arg_node = p.arg
            left_e = _pg_expr_full(
                arg_node,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=None,
            )
            if left_e is None:
                return None
            if p.nulltesttype == NullTestType.IS_NULL:
                op_n = "is null"
            elif p.nulltesttype == NullTestType.IS_NOT_NULL:
                op_n = "is not null"
            else:
                return None
            out.append(
                HavingParam(
                    left_expr=left_e,
                    op=op_n,
                    value_type="null",
                )
            )
            continue
        if not isinstance(p, A_Expr):
            return None
        if p.kind not in (A_Expr_Kind.AEXPR_OP, A_Expr_Kind.AEXPR_LIKE):
            return None
        op_raw = _pg_a_expr_op_name(p)
        mapped_op = _pg_map_where_scalar_op(op_raw)
        if mapped_op is None:
            return None
        lexpr, rexpr = p.lexpr, p.rexpr
        if lexpr is None or rexpr is None:
            return None
        if not isinstance(lexpr, FuncCall):
            return None
        left_n = _pg_aggregate_funcall_to_expr(lexpr, alias_map, single_alias, param_store, next_lit_key, pgx)
        if left_n is None:
            return None
        if isinstance(rexpr, ColumnRef):
            right_q = _pg_qual_column_name(rexpr, alias_map, single_alias, pgx)
            if right_q is None:
                return None
            out.append(
                HavingParam(
                    left_expr=left_n,
                    op=mapped_op,
                    right_expr=NormalizedExpr.from_column(right_q),
                    value_type="column",
                )
            )
            continue
        lit = _pg_where_literal_payload(rexpr)
        if lit is not None:
            raw_v, vt = lit
            pk = next_lit_key()
            param_store[pk] = raw_v
            out.append(
                HavingParam(
                    left_expr=left_n,
                    op=mapped_op,
                    value_type=vt,
                    param_key=pk,
                )
            )
            continue
        right_e = _pg_expr_full(
            rexpr,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if right_e is None:
            return None
        out.append(
            HavingParam(
                left_expr=left_n,
                op=mapped_op,
                right_expr=right_e,
                value_type="column",
            )
        )
    return out


def _pg_const_int_only(node: Any) -> int | None:
    """Parse ``A_Const`` integer literal."""

    from pglast.ast import A_Const, Integer

    if not isinstance(node, A_Const) or getattr(node, "isnull", False):
        return None
    v = node.val
    if isinstance(v, Integer):
        try:
            return int(v.ival)
        except (TypeError, ValueError, AttributeError):
            return None
    return None


def _pg_const_payload(node: Any) -> tuple[Any, str, str] | None:
    """
    Return ``(raw_value, value_type, placeholder_expr)`` for literal masking.

    ``placeholder_expr`` is one of the SQL_TO_INTENT_LITERAL_PLACEHOLDER_* tokens stored in ``NormalizedExpr.string_literal``.
    """

    from pglast.ast import A_Const

    if not isinstance(node, A_Const) or getattr(node, "isnull", False):
        return None
    v = node.val
    vk = type(v).__name__
    if vk == "Integer":
        return int(v.ival), "integer", SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM
    if vk == "Float":
        return float(v.fval), "number", SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM
    if vk == "String":
        return str(v.sval), "string", SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR
    if vk == "Boolean":
        return bool(v.boolval), "boolean", SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR
    return None


def _pg_mask_literal(param_store: dict[str, Any], next_lit_key: Any, payload: tuple[Any, str, str]) -> NormalizedExpr:
    raw_v, _vt, ph = payload
    key = next_lit_key()
    param_store[key] = raw_v
    return NormalizedExpr(string_literal=ph)


def _pg_expr_leaf(
    node: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> NormalizedExpr | None:
    """Recognise column refs or literals for projections / predicates."""

    from pglast.ast import A_Const, ColumnRef, TypeCast

    if isinstance(node, ColumnRef):
        qn = _pg_qual_column_name(node, alias_map, single_alias, pgx)
        if qn is None:
            return None
        return NormalizedExpr.from_column(qn)
    if isinstance(node, A_Const):
        pay = _pg_const_payload(node)
        if pay is None:
            return None
        return _pg_mask_literal(param_store, next_lit_key, pay)
    if isinstance(node, TypeCast) and isinstance(node.arg, A_Const):
        pay = _pg_const_payload(node.arg)
        if pay is None:
            return None
        return _pg_mask_literal(param_store, next_lit_key, pay)
    return None


def _pg_typename_cast_str(type_name: Any) -> str | None:
    """Render a ``TypeName`` chain as a SQL type string."""

    if type_name is None:
        return None
    names = getattr(type_name, "names", None) or ()
    if not names:
        return None
    parts: list[str] = []
    for s in names:
        sv = getattr(s, "sval", None)
        if not isinstance(sv, str) or not sv.strip():
            return None
        parts.append(sv.strip())
    return ".".join(parts)


def _pg_interval_magnitude_unit_from_string(raw: str) -> tuple[float, str] | None:
    """Parse ``INTERVAL`` text payload into magnitude and unit."""

    t = (raw or "").strip()
    if not t:
        return None
    parts = t.split()
    if len(parts) >= 2:
        try:
            return (float(parts[0]), " ".join(parts[1:]).lower())
        except (TypeError, ValueError):
            return None
    if len(parts) == 1:
        try:
            return (float(parts[0]), "day")
        except (TypeError, ValueError):
            return None
    return None


def _pg_sql_value_function_keyword(node: Any) -> str | None:
    """Map ``SQLValueFunction`` to ``NormalizedExpr.keyword``."""

    from pglast.ast import SQLValueFunction
    from pglast.enums import SQLValueFunctionOp

    if not isinstance(node, SQLValueFunction):
        return None
    op = getattr(node, "op", None)
    mapping = {
        SQLValueFunctionOp.SVFOP_CURRENT_DATE: "current_date",
        SQLValueFunctionOp.SVFOP_CURRENT_TIME: "current_time",
        SQLValueFunctionOp.SVFOP_CURRENT_TIMESTAMP: "current_timestamp",
        SQLValueFunctionOp.SVFOP_LOCALTIME: "localtime",
        SQLValueFunctionOp.SVFOP_LOCALTIMESTAMP: "localtimestamp",
    }
    return mapping.get(op)


def _pg_collect_product_factors(expr: NormalizedExpr) -> list[NormalizedExpr]:
    """Flatten a pure multiplicative chain into factor leaves."""

    if (
        expr.add_groups
        and len(expr.add_groups) == 1
        and not expr.sub_groups
        and not expr.add_values
        and not expr.sub_values
        and not expr.string_literal
        and not expr.scalar_func
    ):
        g = expr.add_groups[0]
        if (
            not g.divide
            and not g.agg_func
            and not g.scalar_func
            and float(g.coefficient) == 1.0
            and not g.coeff_param_key
            and len(g.multiply) > 1
        ):
            return list(g.multiply)
    return [expr]


def _pg_merge_product_exprs(left: NormalizedExpr, right: NormalizedExpr) -> NormalizedExpr:
    """Build ``NormalizedExpr`` for ``left * right``."""

    lf = _pg_collect_product_factors(left)
    rf = _pg_collect_product_factors(right)
    return NormalizedExpr(add_groups=[MulGroup(multiply=lf + rf)])


def _pg_merge_ratio_exprs(left: NormalizedExpr, right: NormalizedExpr) -> NormalizedExpr:
    """Build ``NormalizedExpr`` for ``left / right``."""

    lf = _pg_collect_product_factors(left)
    rf = _pg_collect_product_factors(right)
    return NormalizedExpr(add_groups=[MulGroup(multiply=lf, divide=rf)])


def _pg_wrap_mul_term(expr: NormalizedExpr) -> MulGroup:
    """Promote *expr* to a single multiplicative term."""

    return MulGroup(multiply=[expr])


def _pg_expr_sql_token(expr: NormalizedExpr) -> str:
    """Serialize a simple expression as a trailing scalar-function token."""

    if expr.column_ref:
        return str(expr.column_ref)
    if expr.string_literal:
        return str(expr.string_literal)
    if expr.keyword:
        return str(expr.keyword)
    return ""


def _pg_window_partition_exprs(
    nodes: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None,
) -> list[NormalizedExpr] | None:
    """Convert ``PARTITION BY`` expressions."""

    out: list[NormalizedExpr] = []
    for n in nodes or ():
        ex = _pg_expr_full(
            n,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if ex is None:
            return None
        out.append(ex)
    return out


def _pg_window_sort_clause(
    nodes: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    select_cols: list[SelectCol],
    pgx: PgExtra | None,
) -> list[OrderByCol] | None:
    """Convert ``ORDER BY`` inside ``OVER``."""

    from pglast.ast import A_Const, SortBy
    from pglast.enums import SortByDir

    out: list[OrderByCol] = []
    for sb in nodes or ():
        if not isinstance(sb, SortBy):
            return None
        node = sb.node
        if isinstance(node, A_Const):
            ord_i = _pg_const_int_only(node)
            if ord_i is None or ord_i < 1 or ord_i > len(select_cols):
                return None
            ex = select_cols[ord_i - 1].expr
        else:
            ex = _pg_expr_full(
                node,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if ex is None:
                return None
        direnum = getattr(sb, "sortby_dir", SortByDir.SORTBY_DEFAULT)
        direction = "DESC" if direnum == SortByDir.SORTBY_DESC else "ASC"
        out.append(OrderByCol(expr=ex, direction=direction))
    return out


def _pg_window_def_to_spec(
    wdef: Any,
    fn: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    select_cols: list[SelectCol],
    pgx: PgExtra | None,
) -> WindowSpec | None:
    """Build ``WindowSpec`` from an inline ``WindowDef``."""

    from pglast.ast import FuncCall

    if wdef is None:
        return None
    if getattr(wdef, "refname", None):
        return None
    if not isinstance(fn, FuncCall):
        return None
    raw = (_pg_funcname(fn) or "").strip().lower()
    fn_name = raw.split(".")[-1] if raw else ""
    if not fn_name:
        return None
    part = _pg_window_partition_exprs(wdef.partitionClause, alias_map, single_alias, param_store, next_lit_key, pgx)
    if part is None:
        return None
    order = _pg_window_sort_clause(
        wdef.orderClause,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        select_cols,
        pgx,
    )
    if order is None:
        return None
    arg_expr: NormalizedExpr | None = None
    args = fn.args or ()
    if args:
        arg_expr = _pg_expr_full(
            args[0],
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=select_cols,
        )
        if arg_expr is None:
            return None
    fo = int(getattr(wdef, "frameOptions", 0) or 0)
    frame_kind: str = "none"
    frame_start: str | None = None
    frame_end: str | None = None
    frame_start_offset: int | None = None
    frame_end_offset: int | None = None
    if fo == PG_LAST_WINDOW_FRAME_OPTIONS_INLINE_DEFAULT:
        frame_kind = "none"
    elif fo == PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_UNBOUNDED_PAIR:
        frame_kind = "rows"
        frame_start = "UNBOUNDED PRECEDING"
        frame_end = "UNBOUNDED FOLLOWING"
    elif fo == PG_LAST_WINDOW_FRAME_OPTIONS_RANGE_UNBOUNDED_CURRENT:
        frame_kind = "range"
        frame_start = "UNBOUNDED PRECEDING"
        frame_end = "CURRENT ROW"
    elif fo == PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_OFFSET_CURRENT:
        frame_kind = "rows"
        off = _pg_const_int_only(getattr(wdef, "startOffset", None))
        if off is None:
            return None
        frame_start = f"{off} PRECEDING"
        frame_end = "CURRENT ROW"
        frame_start_offset = off
    else:
        return None
    ws = WindowSpec(
        function=fn_name,
        partition_by=part,
        order_by=order,
        argument=arg_expr,
        frame_kind=frame_kind,  # type: ignore[arg-type]
        frame_start=frame_start,
        frame_end=frame_end,
        frame_start_offset=frame_start_offset,
        frame_end_offset=frame_end_offset,
    )
    return _normalize_window_spec_ansi_defaults(ws)


def _pg_funcall_with_over_to_registry_step(
    fn: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    select_cols: list[SelectCol],
    pgx: PgExtra,
) -> NormalizedExpr | None:
    """Emit ``WindowRegistryStep`` for ``FuncCall`` with ``OVER``."""

    from pglast.ast import FuncCall

    if not isinstance(fn, FuncCall) or not getattr(fn, "over", None):
        return None
    ws = _pg_window_def_to_spec(
        fn.over,
        fn,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        select_cols,
        pgx,
    )
    if ws is None:
        return None
    rid = pgx.next_window_id()
    pgx.window_registry.append(WindowRegistryStep(registry_id=rid, window_spec=ws))
    return NormalizedExpr.from_column(rid)


def _pg_caseexpr_to_registry_step(
    node: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra,
) -> NormalizedExpr | None:
    """Emit ``CaseRegistryStep`` for ``CaseExpr``."""

    from pglast.ast import CaseExpr, CaseWhen

    if not isinstance(node, CaseExpr):
        return None
    branches: list[CaseWhenBranch] = []
    for cw in node.args or ():
        if not isinstance(cw, CaseWhen):
            return None
        cond = _pg_single_predicate_to_filter(cw.expr, alias_map, single_alias, param_store, next_lit_key, pgx)
        if cond is None:
            return None
        cond = replace(cond, bool_op="AND", filter_group=None)
        res = _pg_expr_full(
            cw.result,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if res is None:
            return None
        branches.append(CaseWhenBranch(condition=cond, result=res))
    else_result: NormalizedExpr | None = None
    if getattr(node, "defresult", None) is not None:
        else_result = _pg_expr_full(
            node.defresult,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if else_result is None:
            return None
    rid = pgx.next_case_id()
    pgx.case_registry.append(
        CaseRegistryStep(
            registry_id=rid,
            case_when=CaseWhenExpr(branches=branches, else_result=else_result),
        )
    )
    return NormalizedExpr.from_column(rid)


def _pg_coalesce_to_expr(
    node: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None,
) -> NormalizedExpr | None:
    """Map ``CoalesceExpr`` to ``NormalizedExpr``."""

    from pglast.ast import A_Const, CoalesceExpr

    if not isinstance(node, CoalesceExpr):
        return None
    args = list(node.args or ())
    if len(args) < 2:
        return None
    first = _pg_expr_full(
        args[0],
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        pgx,
        allow_aggregate=False,
        allow_window=False,
        select_cols=None,
    )
    if first is None:
        return None
    trail_args: list[Any] = []
    trail_keys: list[str] = []
    for a in args[1:]:
        if isinstance(a, A_Const):
            pay = _pg_const_payload(a)
            if pay is None:
                return None
            pk = next_lit_key()
            param_store[pk] = pay[0]
            trail_keys.append(pk)
            if isinstance(pay[0], (int, float)):
                trail_args.append(float(pay[0]))
            else:
                trail_args.append(str(pay[0]))
            continue
        ex = _pg_expr_full(
            a,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if ex is None or not ex.column_ref:
            return None
        trail_keys.append("")
        trail_args.append(str(ex.column_ref))
    mg = MulGroup(multiply=[first])
    return NormalizedExpr(
        add_groups=[mg],
        scalar_func="coalesce",
        scalar_func_args=trail_args,
        sarg_param_keys=trail_keys,
    )


def _pg_expr_full(
    node: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
    *,
    allow_aggregate: bool = True,
    allow_window: bool = True,
    select_cols: list[SelectCol] | None = None,
) -> NormalizedExpr | None:
    """Parse PostgreSQL expression trees into ``NormalizedExpr`` or registry references."""

    from pglast.ast import A_Const, A_Expr, CaseExpr, CoalesceExpr, FuncCall, TypeCast
    from pglast.enums import A_Expr_Kind

    leaf = _pg_expr_leaf(node, alias_map, single_alias, param_store, next_lit_key, pgx)
    if leaf is not None:
        return leaf
    kw = _pg_sql_value_function_keyword(node)
    if kw is not None:
        return NormalizedExpr(keyword=kw)
    if isinstance(node, CaseExpr):
        if pgx is None:
            return None
        return _pg_caseexpr_to_registry_step(node, alias_map, single_alias, param_store, next_lit_key, pgx)
    if isinstance(node, CoalesceExpr):
        return _pg_coalesce_to_expr(node, alias_map, single_alias, param_store, next_lit_key, pgx)
    if isinstance(node, FuncCall):
        if getattr(node, "over", None):
            if not (allow_window and pgx is not None and select_cols is not None):
                return None
            win = _pg_funcall_with_over_to_registry_step(
                node,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                select_cols,
                pgx,
            )
            return win
        if allow_aggregate:
            agg = _pg_aggregate_funcall_to_expr(node, alias_map, single_alias, param_store, next_lit_key, pgx)
            if agg is not None:
                return agg
        raw = (_pg_funcname(node) or "").strip().lower()
        fn_name = raw.split(".")[-1] if raw else ""
        if fn_name == "extract":
            parts = list(node.args or ())
            if len(parts) != 2:
                return None
            fld = _pg_const_payload(parts[0])
            if fld is None or fld[1] != "string":
                return None
            inner = _pg_expr_full(
                parts[1],
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if inner is None:
                return None
            mg = MulGroup(
                multiply=[inner],
                scalar_func="extract",
                scalar_func_args=[str(fld[0]).strip()],
            )
            return NormalizedExpr(add_groups=[mg])
        if fn_name in {"upper", "lower", "trim"}:
            parts = list(node.args or ())
            if len(parts) != 1:
                return None
            inner = _pg_expr_full(
                parts[0],
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if inner is None:
                return None
            mg = MulGroup(multiply=[inner], scalar_func=fn_name)
            return NormalizedExpr(add_groups=[mg])
        return None
    if isinstance(node, TypeCast):
        tn = _pg_typename_cast_str(getattr(node, "typeName", None))
        if tn and "interval" in tn.lower():
            if not isinstance(node.arg, A_Const):
                return None
            sp = _pg_const_payload(node.arg)
            if sp is None or sp[1] != "string":
                return None
            iv = _pg_interval_magnitude_unit_from_string(str(sp[0]))
            if iv is None:
                return None
            return NormalizedExpr(interval=iv)
        inner = _pg_expr_full(
            node.arg,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        if inner is None or not tn:
            return None
        return NormalizedExpr(add_groups=[MulGroup(multiply=[inner])], cast_type=tn)
    if isinstance(node, A_Expr):
        if node.kind == A_Expr_Kind.AEXPR_NULLIF:
            lx = node.lexpr
            rx = node.rexpr
            if lx is None or rx is None:
                return None
            l_e = _pg_expr_full(
                lx,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            r_e = _pg_expr_full(
                rx,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if l_e is None or r_e is None:
                return None
            mg = MulGroup(
                multiply=[l_e],
                scalar_func="nullif",
                scalar_func_args=[_pg_expr_sql_token(r_e)],
            )
            return NormalizedExpr(add_groups=[mg])
        op_raw = (_pg_a_expr_op_name(node) or "").strip()
        if node.kind == A_Expr_Kind.AEXPR_OP and op_raw in {"+", "-", "*", "/"}:
            lx = node.lexpr
            rx = node.rexpr
            if op_raw == "-" and lx is None and rx is not None:
                inner = _pg_expr_full(
                    rx,
                    alias_map,
                    single_alias,
                    param_store,
                    next_lit_key,
                    pgx,
                    allow_aggregate=allow_aggregate,
                    allow_window=allow_window,
                    select_cols=select_cols,
                )
                if inner is None:
                    return None
                return NormalizedExpr(sub_groups=[_pg_wrap_mul_term(inner)])
            if lx is None or rx is None:
                return None
            left = _pg_expr_full(
                lx,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=allow_aggregate,
                allow_window=allow_window,
                select_cols=select_cols,
            )
            right = _pg_expr_full(
                rx,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=allow_aggregate,
                allow_window=allow_window,
                select_cols=select_cols,
            )
            if left is None or right is None:
                return None
            if op_raw == "*":
                return _pg_merge_product_exprs(left, right)
            if op_raw == "/":
                return _pg_merge_ratio_exprs(left, right)
            lg = _pg_wrap_mul_term(left)
            rg = _pg_wrap_mul_term(right)
            if op_raw == "+":
                return NormalizedExpr(add_groups=[lg, rg])
            return NormalizedExpr(add_groups=[lg], sub_groups=[rg])
    return None


def _pg_aggregate_funcall_to_expr(
    fn: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> NormalizedExpr | None:
    """Map aggregate ``FuncCall`` nodes to ``NormalizedExpr``."""

    from pglast.ast import FuncCall

    if not isinstance(fn, FuncCall):
        return None
    raw = (_pg_funcname(fn) or "").strip().lower()
    fn_name = raw.split(".")[-1] if raw else ""
    if fn_name not in _PG_SIMPLE_AGG_NAMES:
        return None
    if getattr(fn, "agg_star", False):
        return NormalizedExpr.from_agg(fn_name, "*")
    args = fn.args or ()
    if len(args) != 1:
        return None
    inner = _pg_expr_full(
        args[0],
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        pgx,
        allow_aggregate=False,
        allow_window=False,
        select_cols=None,
    )
    if inner is None:
        return None
    mg = MulGroup(
        multiply=[inner],
        agg_func=fn_name,
        distinct=bool(getattr(fn, "agg_distinct", False)),
    )
    return NormalizedExpr(add_groups=[mg])


def _pg_res_target_to_expr(
    rt: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
    select_cols: list[SelectCol] | None = None,
) -> NormalizedExpr | None:
    """Convert ``ResTarget.val`` including arithmetic, CASE, windows, and aggregates."""

    return _pg_expr_full(
        rt.val,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        pgx,
        allow_aggregate=True,
        allow_window=True,
        select_cols=select_cols,
    )


def _pg_flatten_bool_and(node: Any) -> list[Any]:
    """Flatten nested ``AND`` ``BoolExpr`` chains."""

    from pglast.ast import BoolExpr
    from pglast.enums import BoolExprType

    if isinstance(node, BoolExpr) and node.boolop == BoolExprType.AND_EXPR:
        out: list[Any] = []
        for a in node.args or ():
            out.extend(_pg_flatten_bool_and(a))
        return out
    return [node]


def _pg_where_literal_payload(node: Any) -> tuple[Any, str] | None:
    """Extract typed literal for WHERE RHS without masking SELECT placeholders."""

    from pglast.ast import A_Const, TypeCast

    if isinstance(node, A_Const):
        p = _pg_const_payload(node)
        return (p[0], p[1]) if p else None
    if isinstance(node, TypeCast) and isinstance(node.arg, A_Const):
        p = _pg_const_payload(node.arg)
        return (p[0], p[1]) if p else None
    return None


def _pg_collect_in_literal_values(rexpr: Any) -> tuple[list[Any], str] | None:
    """Homogeneous literal list for ``IN`` / ``NOT IN`` right-hand sides."""

    seq = rexpr if isinstance(rexpr, (list, tuple)) else (rexpr,)
    out: list[Any] = []
    vt: str | None = None
    for item in seq:
        lit = _pg_where_literal_payload(item)
        if lit is None:
            return None
        val, t = lit
        if vt is None:
            vt = t
        elif vt != t:
            return None
        out.append(val)
    if not out:
        return None
    return out, vt


def _pg_single_predicate_to_filter(
    p: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> FilterParam | None:
    """Convert one ``WHERE`` predicate leaf (no nested ``BoolExpr``)."""

    from pglast.ast import A_Expr, ColumnRef, NullTest, SubLink
    from pglast.enums import A_Expr_Kind, NullTestType

    if isinstance(p, SubLink):
        return None
    if isinstance(p, NullTest):
        left_e = _pg_expr_full(
            p.arg,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        if p.nulltesttype == NullTestType.IS_NULL:
            op_n = "is null"
        elif p.nulltesttype == NullTestType.IS_NOT_NULL:
            op_n = "is not null"
        else:
            return None
        return FilterParam(
            left_expr=left_e,
            op=op_n,
            value_type="null",
        )
    if isinstance(p, A_Expr) and p.kind == A_Expr_Kind.AEXPR_BETWEEN:
        lexpr, rexpr = p.lexpr, p.rexpr
        if lexpr is None or rexpr is None:
            return None
        left_e = _pg_expr_full(
            lexpr,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        seq = rexpr if isinstance(rexpr, (list, tuple)) else (rexpr,)
        if len(seq) != 2:
            return None
        lo_lit = _pg_where_literal_payload(seq[0])
        hi_lit = _pg_where_literal_payload(seq[1])
        if lo_lit is None or hi_lit is None or lo_lit[1] != hi_lit[1]:
            return None
        vt = lo_lit[1]
        pk_lo = next_lit_key()
        pk_hi = next_lit_key()
        param_store[pk_lo] = lo_lit[0]
        param_store[pk_hi] = hi_lit[0]
        return FilterParam(
            left_expr=left_e,
            op="between",
            value_type=vt,
            param_key=pk_lo,
            param_key_hi=pk_hi,
        )
    if not isinstance(p, A_Expr):
        return None
    if p.kind == A_Expr_Kind.AEXPR_IN:
        lexpr, rexpr = p.lexpr, p.rexpr
        if lexpr is None or rexpr is None:
            return None
        left_e = _pg_expr_full(
            lexpr,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            pgx,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        names = getattr(p, "name", None) or ()
        not_in = len(names) == 1 and getattr(names[0], "sval", None) == "<>"
        op_in = "not in" if not_in else "in"
        collected = _pg_collect_in_literal_values(rexpr)
        if collected is None:
            return None
        vals, vt = collected
        pk = next_lit_key()
        param_store[pk] = vals
        return FilterParam(
            left_expr=left_e,
            op=op_in,
            value_type=vt,
            param_key=pk,
        )
    if p.kind not in (A_Expr_Kind.AEXPR_OP, A_Expr_Kind.AEXPR_LIKE):
        return None
    op_raw = _pg_a_expr_op_name(p)
    mapped_op = _pg_map_where_scalar_op(op_raw)
    if mapped_op is None:
        return None
    lexpr, rexpr = p.lexpr, p.rexpr
    if lexpr is None or rexpr is None:
        return None
    left_e = _pg_expr_full(
        lexpr,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        pgx,
        allow_aggregate=False,
        allow_window=False,
        select_cols=None,
    )
    if left_e is None:
        return None
    if isinstance(rexpr, ColumnRef):
        right_q = _pg_qual_column_name(rexpr, alias_map, single_alias, pgx)
        if right_q is None:
            return None
        return FilterParam(
            left_expr=left_e,
            op=mapped_op,
            right_expr=NormalizedExpr.from_column(right_q),
            value_type="column",
        )
    lit = _pg_where_literal_payload(rexpr)
    if lit is not None:
        raw_v, vt = lit
        pk = next_lit_key()
        param_store[pk] = raw_v
        return FilterParam(
            left_expr=left_e,
            op=mapped_op,
            value_type=vt,
            param_key=pk,
        )
    right_e = _pg_expr_full(
        rexpr,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        pgx,
        allow_aggregate=False,
        allow_window=False,
        select_cols=None,
    )
    if right_e is None:
        return None
    return FilterParam(
        left_expr=left_e,
        op=mapped_op,
        right_expr=right_e,
        value_type="column",
    )


def _pg_bool_nesting_too_deep(node: Any, depth: int) -> bool:
    """
    Reject boolean trees deeper than two levels of ``AND``/``OR``.

    Args:

        node: Predicate subtree.

        depth: Current depth counter.

    Returns:

        True when the subtree violates the depth bound.
    """

    from pglast.ast import BoolExpr
    from pglast.enums import BoolExprType

    if depth > 2:
        return True
    if isinstance(node, BoolExpr):
        if node.boolop == BoolExprType.NOT_EXPR:
            return True
        for arg in node.args or ():
            if _pg_bool_nesting_too_deep(arg, depth + 1):
                return True
    return False


def _pg_walk_bool_to_filter_groups(
    where: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> list[FilterParam] | None:
    """
    Map ``WHERE`` boolean trees to ``FilterParam`` rows with optional ``filter_group`` ids.

    Args:

        where: ``WHERE`` clause root.

        alias_map: Range alias map.

        single_alias: Sole alias shortcut when exactly one range exists.

        param_store: Literal binding targets.

        next_lit_key: Generator for bind keys.

        pgx: Optional qualifier swap context.

    Returns:

        Filters or ``None`` when the shape is outside the supported OR-of-AND tier.
    """

    from pglast.ast import BoolExpr
    from pglast.enums import BoolExprType

    if _pg_bool_nesting_too_deep(where, 0):
        return None
    if isinstance(where, BoolExpr) and where.boolop == BoolExprType.NOT_EXPR:
        return None

    if isinstance(where, BoolExpr) and where.boolop == BoolExprType.OR_EXPR:
        out_or: list[FilterParam] = []
        gid = 1
        for arg in where.args or ():
            if isinstance(arg, BoolExpr) and arg.boolop == BoolExprType.AND_EXPR:
                sub = _pg_flatten_bool_and(arg)
                first_arm = True
                for subp in sub:
                    fp = _pg_single_predicate_to_filter(subp, alias_map, single_alias, param_store, next_lit_key, pgx)
                    if fp is None:
                        return None
                    bo = "AND"
                    if first_arm:
                        first_arm = False
                    else:
                        bo = "AND"
                    fp = replace(fp, filter_group=gid, bool_op=bo)
                    out_or.append(fp)
                gid += 1
                continue
            fp = _pg_single_predicate_to_filter(arg, alias_map, single_alias, param_store, next_lit_key, pgx)
            if fp is None:
                return None
            fp = replace(fp, filter_group=gid, bool_op="AND")
            out_or.append(fp)
            gid += 1
        return out_or

    if isinstance(where, BoolExpr) and where.boolop != BoolExprType.AND_EXPR:
        return None
    parts = _pg_flatten_bool_and(where)
    out: list[FilterParam] = []
    for p in parts:
        fp = _pg_single_predicate_to_filter(p, alias_map, single_alias, param_store, next_lit_key, pgx)
        if fp is None:
            return None
        out.append(fp)
    return out


def _pg_where_to_filters(
    where: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> list[FilterParam] | None:
    """Extract ``FilterParam`` rows from ``WHERE`` using grouped OR-of-AND semantics."""

    return _pg_walk_bool_to_filter_groups(where, alias_map, single_alias, param_store, next_lit_key, pgx)


def _pg_a_expr_op_name(expr: Any) -> str | None:
    """Lowercase operator name from ``A_Expr.name``."""

    names = getattr(expr, "name", None) or ()
    if len(names) != 1:
        return None
    sval = getattr(names[0], "sval", None)
    return str(sval).lower() if isinstance(sval, str) else None


def _pg_group_clause(
    nodes: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    select_cols: list[SelectCol],
    param_store: dict[str, Any],
    next_lit_key: Any,
    pgx: PgExtra | None = None,
) -> list[NormalizedExpr] | None:
    """Convert ``GROUP BY`` list: columns, ordinals, or leaf expressions."""

    from pglast.ast import A_Const, ColumnRef

    out: list[NormalizedExpr] = []
    for n in nodes or ():
        if isinstance(n, ColumnRef):
            qn = _pg_qual_column_name(n, alias_map, single_alias, pgx)
            if qn is None:
                return None
            out.append(NormalizedExpr.from_column(qn))
        elif isinstance(n, A_Const):
            idx = _pg_const_int_only(n)
            if idx is None or idx < 1 or idx > len(select_cols):
                return None
            out.append(select_cols[idx - 1].expr)
        else:
            ex = _pg_expr_full(
                n,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if ex is None:
                return None
            out.append(ex)
    return out


def _pg_sort_clause(
    nodes: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Any,
    select_cols: list[SelectCol],
    pgx: PgExtra | None = None,
) -> list[OrderByCol] | None:
    """Convert ``ORDER BY`` using column refs, ordinals, or leaf expressions."""

    from pglast.ast import A_Const, SortBy
    from pglast.enums import SortByDir

    out: list[OrderByCol] = []
    for sb in nodes or ():
        if not isinstance(sb, SortBy):
            return None
        node = sb.node
        if isinstance(node, A_Const):
            ord_i = _pg_const_int_only(node)
            if ord_i is None or ord_i < 1 or ord_i > len(select_cols):
                return None
            ex = select_cols[ord_i - 1].expr
        else:
            ex = _pg_expr_full(
                node,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                pgx,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if ex is None:
                return None
        direnum = getattr(sb, "sortby_dir", SortByDir.SORTBY_DEFAULT)
        direction = "DESC" if direnum == SortByDir.SORTBY_DESC else "ASC"
        out.append(OrderByCol(expr=ex, direction=direction))
    return out


def _convert_sqlglot(sql: str, schema: SchemaGraph, sqlglot_read: str) -> RuntimeIntent:
    """
    sqlglot-based SQL to RuntimeIntent for dialects using the sqlglot reader *sqlglot_read*.

    Prefers transpiling warehouse-specific syntax to PostgreSQL-shaped SQL and reusing the
    pglast extraction path when available so Spark/Databricks SELECT shapes match the Postgres
    converter coverage. Falls back to the minimal sqlglot tier when transpilation or pglast
    extraction yields nothing.

    Args:

        sql: Raw SQL text.

        schema: Active schema graph.

        sqlglot_read: sqlglot ``read=`` dialect name.

    Returns:

        Parsed runtime intent.

    Raises:

        ValueError: When parsing or structural extraction fails.
    """

    transpiled: list[str] = []
    try:
        transpiled = sqlglot.transpile(sql, read=sqlglot_read, write="postgres")
    except Exception:
        transpiled = []
    pglast_available = False
    if pglast is not None:
        try:
            pglast.parse_sql("SELECT 1")
            pglast_available = True
        except Exception:
            pglast_available = False
    if pglast_available and transpiled:
        for tx in transpiled:
            rt_try = _runtime_from_pglast_sql(tx, schema)
            if rt_try is not None:
                rt_try = _dedup_cte_steps(rt_try)
                rt_try = _dedup_window_registry(rt_try)
                rt_try = _dedup_case_registry(rt_try)
                return rt_try
    try:
        tree = sqlglot.parse_one(sql, read=sqlglot_read)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(tree, sqlglot_exp.Select):
        sel = tree.find(sqlglot_exp.Select)
        if sel is None:
            raise ValueError("expected SELECT")
    else:
        sel = tree
    rt = _runtime_from_sqlglot_select(sel, schema)
    if rt is None:
        raise ValueError("unsupported SELECT shape for sqlglot conversion tier")
    rt = _dedup_cte_steps(rt)
    rt = _dedup_window_registry(rt)
    rt = _dedup_case_registry(rt)
    return rt


def _convert_postgres(sql: str, schema: SchemaGraph) -> RuntimeIntent:
    """
    PostgreSQL SQL → :class:`RuntimeIntent`.

    When :mod:`pglast` is installed, validates parseability then prefers AST extraction; falls back to sqlglot when AST extraction yields ``None``.
    """

    sqlglot_read = str(SQLGLOT_DIALECT_BY_ENGINE.get("postgresql", "postgres"))
    pg_ok = False
    if pglast is not None:
        try:
            pglast.parse_sql(sql)
            pg_ok = True
        except Exception as exc:
            raise ValueError(f"postgres parse rejected SQL: {exc}") from exc

    if pg_ok:
        rt = _runtime_from_pglast_sql(sql, schema)
        if rt is not None:
            rt = _dedup_cte_steps(rt)
            rt = _dedup_window_registry(rt)
            rt = _dedup_case_registry(rt)
            return rt
    return _convert_sqlglot(sql, schema, sqlglot_read)


def _check_round_trip_shape(
    original_sql: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    sqlglot_read: str,
) -> bool:
    """
    Layer 1: structural SQL shape parity between the original text and the inferred intent.

    Args:

        original_sql: User SQL.

        intent: Converted intent.

        schema: Schema graph.

        dialect: Active dialect instance.

        sqlglot_read: sqlglot reader key for :func:`sql_shape`.

    Returns:

        True when shapes align sufficiently for this tier.
    """

    try:
        a = sql_shape(original_sql, intent, sqlglot_dialect=sqlglot_read)
        round_sql = sqlglot.transpile(original_sql, read=sqlglot_read, write=sqlglot_read)[0]
        b = sql_shape(round_sql, intent, sqlglot_dialect=sqlglot_read)
    except Exception:
        return False
    return a.num_joins == b.num_joins and a.has_group_by == b.has_group_by and a.has_agg == b.has_agg


def _pg_plan_rows_from_explain_payload(payload: Any) -> float | None:
    """
    Extract the top-level ``Plan Rows`` estimate from a PostgreSQL JSON ``EXPLAIN`` payload.

    Args:

        payload: Raw ``EXPLAIN`` cell value or JSON text.

    Returns:

        Floating estimate when present.
    """

    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    if isinstance(payload, list) and payload:
        root = payload[0]
        if isinstance(root, dict):
            plan = root.get("Plan")
            if isinstance(plan, dict):
                pr = plan.get("Plan Rows")
                if isinstance(pr, (int, float)):
                    return float(pr)
    return None


def _databricks_plan_rows_from_explain_text(payload: str) -> float | None:
    """
    Extract a coarse row-count estimate from Spark/Databricks ``EXPLAIN COST`` text.

    Args:

        payload: Concatenated plan lines.

    Returns:

        First plausible numeric rows estimate, or ``None`` when none match.
    """

    if not payload:
        return None
    for pat in (
        r"(?i)Statistics\s*\([^)]*rowCount\s*=\s*(\d+)",
        r"(?i)rowCount[=:\s]+(\d+)",
        r"(?i)numRows[=:\s]+(\d+)",
        r"(?i)rows[=:\s]+(\d+)",
    ):
        m = re.search(pat, payload)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _check_round_trip_intent_parity(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    sqlglot_read: str,
) -> bool:
    """
    Layer 2: structural presence gate before deeper regeneration probes run elsewhere.

    Args:

        intent: Converted intent.

        schema: Schema graph (unused at this tier).

        dialect: Active dialect (unused at this tier).

        sqlglot_read: Reader token (unused at this tier).

    Returns:

        True when tables and projections exist on the converted intent.
    """

    del schema, dialect, sqlglot_read
    return bool(intent.tables) and bool(intent.select_cols)


def _check_round_trip_execute(
    original_sql: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    *,
    limit: int = WARMUP_ROUND_TRIP_LIMIT,
) -> bool:
    """
    Layer 3: compare planner cardinality estimates between original and regenerated SQL.

    Args:

        original_sql: User SQL.

        intent: Converted intent.

        schema: Schema graph.

        dialect: Active dialect.

        limit: Unused placeholder retained for API symmetry with execution caps.

    Returns:

        True when estimates agree within tolerance, when EXPLAIN is unavailable, or when binding fails softly.
    """

    del limit
    finalize = getattr(dialect, "finalize_render", None)
    if not callable(finalize):
        return True
    tol = float(WARMUP_ROUND_TRIP_CARDINALITY_TOLERANCE)

    def explain_rows_postgres(sql_text: str) -> float | None:
        eng = getattr(dialect, "engine", None)
        if eng is None:
            return None
        try:
            finalized = finalize(sql_text, {}, schema=schema, intent=intent)
            explain_sql = f"EXPLAIN (FORMAT JSON, COSTS true) {finalized}"
            with eng.connect() as conn:
                rows = conn.execute(text(explain_sql), {}).fetchall()
            payload = rows[0][0] if rows else None
            return _pg_plan_rows_from_explain_payload(payload)
        except Exception:
            return None

    def explain_rows_databricks(sql_text: str) -> float | None:
        if not isinstance(dialect, DatabricksDialect):
            return None
        try:
            finalized = finalize(sql_text, {}, schema=schema, intent=intent)
            explain_sql = f"EXPLAIN COST {finalized}"
            text_payload = ""
            eng = getattr(dialect, "engine", None)
            if eng is not None:
                with eng.connect() as conn:
                    rows = conn.execute(text(explain_sql)).fetchall()
                text_payload = "\n".join(str(r[0]) for r in rows if r and r[0] is not None)
            elif getattr(dialect, "connection", None) is not None:
                cursor = dialect.connection.cursor()
                try:
                    cursor.execute(explain_sql)
                    rows = cursor.fetchall() or []
                finally:
                    cursor.close()
                text_payload = "\n".join(str(r[0]) for r in rows if r and r[0] is not None)
            elif getattr(dialect, "spark", None) is not None:
                explain_df = dialect.spark.sql(explain_sql)
                rows = explain_df.collect()
                text_payload = "\n".join(str(r[0]) for r in rows if r and len(r) > 0)
            return _databricks_plan_rows_from_explain_text(text_payload)
        except Exception:
            return None

    def explain_rows(sql_text: str) -> float | None:
        if isinstance(dialect, PostgresDialect):
            return explain_rows_postgres(sql_text)
        if isinstance(dialect, DatabricksDialect):
            return explain_rows_databricks(sql_text)
        return None

    try:
        det = build_deterministic_sql(intent, None, schema, dialect)
    except Exception:
        return False
    r0 = explain_rows(original_sql)
    r1 = explain_rows(det)
    if r0 is None or r1 is None:
        return True
    denom = max(abs(r0), abs(r1), 1.0)
    return abs(r0 - r1) / denom <= tol


def convert_sql_to_intent(
    sql: str,
    schema: SchemaGraph,
    dialect: Any,
    *,
    verify_via_execute: bool = True,
) -> ConverterResult:
    """
    Convert one SQL statement to a RuntimeIntent and validate the round trip.

    Args:

        sql: Single SQL statement text.

        schema: Active schema graph.

        dialect: Live dialect instance (PostgreSQL or Databricks).

        verify_via_execute: When True, attempt layer-3 execution validation when wired.

    Returns:

        :class:`ConverterResult` with intent or a failure code drawn from ``SQL_TO_INTENT_FAILURE_CODES``.
    """

    h = _hash_sql(sql)
    sqlglot_read = _infer_sqlglot_read(dialect)
    try:
        if isinstance(dialect, PostgresDialect):
            intent = _convert_postgres(sql, schema)
        else:
            intent = _convert_sqlglot(sql, schema, sqlglot_read)
    except Exception as exc:
        return ConverterResult(h, None, "SQL_PARSE_FAILED", str(exc))
    if not _check_round_trip_shape(sql, intent, schema, dialect, sqlglot_read):
        return ConverterResult(h, None, "ROUND_TRIP_SHAPE_MISMATCH", "shape gate failed")
    if not _check_round_trip_intent_parity(intent, schema, dialect, sqlglot_read):
        return ConverterResult(h, None, "ROUND_TRIP_INTENT_DRIFT", "intent parity gate failed")
    if verify_via_execute and not _check_round_trip_execute(
        sql, intent, schema, dialect, limit=WARMUP_ROUND_TRIP_LIMIT
    ):
        return ConverterResult(h, None, "ROUND_TRIP_EXECUTE_MISMATCH", "execute oracle gate failed")
    return ConverterResult(h, intent, None, "")


def _union_merge_bucket_key(intent: RuntimeIntent) -> str:
    """
    Structural shell excluding SELECT projections so union-compatible intents cluster.

    Args:

        intent: Runtime intent whose non-projection body defines the bucket.

    Returns:

        Hex digest stable under deterministic ordering of nested signature keys.
    """

    fp = {
        "tables": sorted(intent.tables or []),
        "grain": intent.grain or "row_level",
        "distinct_select_index": int(intent.distinct_select_index),
        "filters": sorted(f.signature_key for f in (intent.filters_param or [])),
        "group_by_cols": sorted(g.signature_key for g in (intent.group_by_cols or [])),
        "order_by_cols": sorted(o.signature_key for o in (intent.order_by_cols or [])),
        "having_param": sorted(h.signature_key for h in (intent.having_param or [])),
        "cte_steps": cte_structural_signature(intent.cte_steps or []),
        "window_registry": sorted(w.signature_key for w in (intent.window_registry or [])),
        "case_registry": sorted(c.signature_key for c in (intent.case_registry or [])),
    }
    return sha256(stable_json(fp))


def _merge_union_compatible_intents(
    cluster: list[RuntimeIntent],
) -> list[RuntimeIntent]:
    """
    Pairwise-merge compatible intents using the same union rules as template matching.

    Args:

        cluster: Intents sharing :func:`_union_merge_bucket_key`.

    Returns:

        Surviving intents after greedy merges; incompatible peers remain distinct.
    """

    if len(cluster) <= 1:
        return list(cluster)
    pending = list(cluster)
    changed = True
    while changed and len(pending) > 1:
        changed = False
        for i in range(len(pending)):
            for j in range(i + 1, len(pending)):
                a, b = pending[i], pending[j]
                conc = runtime_intent_to_concrete(a, "sql_hist_dedup")
                row = union_runtime_concrete_compatibility(b, conc)
                if row is None:
                    continue
                union_cols = row[0]
                combined = replace(a, select_cols=list(union_cols))
                pending = [combined] + [pending[k] for k in range(len(pending)) if k not in (i, j)]
                changed = True
                break
            if changed:
                break
    return pending


def dedup_runtime_intents(intents: list[RuntimeIntent]) -> list[RuntimeIntent]:
    """
    Cluster intents by structural shell (excluding projections), then union-merge selects.

    Pairwise merges use :func:`~aetherdialect._intent_process.union_runtime_concrete_compatibility`
    so aggregate symmetry and bare-column difference caps match template union policy.

    Args:

        intents: Candidate intents from SQL history.

    Returns:

        Deduped list stable under deterministic ordering by :func:`~aetherdialect._utils.body_similarity_key`.
    """

    if not intents:
        return []
    buckets: dict[str, list[RuntimeIntent]] = {}
    order: list[str] = []
    for ri in intents:
        bk = _union_merge_bucket_key(ri)
        if bk not in buckets:
            order.append(bk)
            buckets[bk] = []
        buckets[bk].append(ri)
    merged_flat: list[RuntimeIntent] = []
    for bk in order:
        merged_flat.extend(_merge_union_compatible_intents(buckets[bk]))
    return sorted(merged_flat, key=lambda r: body_similarity_key(r))


def compute_sql_history_content_hash(statements: list[str]) -> str:
    """Return SHA-256 hex over sorted unique non-empty SQL statement texts."""

    dedup = sorted({s.strip() for s in statements if s.strip()})
    payload = "\n".join(dedup).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_sql_history_statements(filepath: str) -> list[str]:
    """Load SQL statements from a text file using the same line conventions as seed question files."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SQL history file not found: {filepath}")

    statements: list[str] = []
    auto_number = 1

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            sql_text = None

            match = re.match(r"^(\d+)\.\s*(.+)$", line)
            if match:
                sql_text = match.group(2).strip()

            if not sql_text:
                match = re.match(r"^(\d+)\)\s*(.+)$", line)
                if match:
                    sql_text = match.group(2).strip()

            if not sql_text:
                match = re.match(r'^["\'](.+)["\']$', line)
                if match:
                    sql_text = match.group(1).strip()

            if not sql_text:
                sql_text = line
                auto_number += 1

            sql_text = sql_text.strip("\"'")
            if sql_text:
                statements.append(sql_text)

    mx = SeedWarmupConfig.MAX_SEED_QUESTIONS
    if len(statements) > mx:
        statements = statements[:mx]
    return statements


def seed_warmup_intent_from_runtime_intent(
    rt: RuntimeIntent,
    *,
    intent_id: str,
    seed_index: int,
) -> SeedWarmupIntent:
    """Materialize a seed-warmup intent row from a converted runtime intent for SQL-history warmup."""

    return SeedWarmupIntent(
        intent_id=intent_id,
        tables=list(rt.tables),
        grain=rt.grain,
        select_cols=list(rt.select_cols),
        group_by_cols=list(rt.group_by_cols),
        order_by_cols=list(rt.order_by_cols),
        filters_param=list(rt.filters_param),
        having_param=list(rt.having_param),
        param_values=dict(rt.param_values),
        cte_steps=list(rt.cte_steps),
        natural_language=str(rt.natural_language or ""),
        limit=rt.limit,
        distinct_select_index=int(rt.distinct_select_index),
        source="sql_history",
        seed_index=seed_index,
        window_registry=list(rt.window_registry),
        case_registry=list(rt.case_registry),
    )


class QueryLogSource(Protocol):
    """Read-only fetcher of historical SQL statements."""

    def is_available(self, conn: Any) -> bool:
        """Return True when the source can run against *conn*."""

        ...

    def fetch(
        self,
        conn: Any,
        *,
        lookback_days: int,
        max_queries: int,
        min_runs: int,
        user_filter: str | None,
    ) -> list[str]:
        """Return distinct SQL texts newest-first within policy caps."""

        ...


def _stable_sql_text_for_history(sql_text: str) -> str:
    """
    Replace inline numeric and single-quoted literals so hashed snapshots stay stable.

    Args:

        sql_text: Raw SQL string.

    Returns:

        Masked text suitable for cross-run digesting.
    """

    s = re.sub(r"\b\d+\.\d+\b", "<num>", sql_text)
    s = re.sub(r"\b\d+\b", "<num>", s)
    s = re.sub(r"'(?:[^']|'')*'", "<str>", s)
    return s


@dataclass(frozen=True)
class PostgresQueryLogSource:
    """pg_stat_statements-backed query log."""

    def is_available(self, conn: Any) -> bool:
        """Return True when *conn* exposes PostgreSQL and ``pg_stat_statements`` is installed."""

        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements' LIMIT 1",
            )
            row = cur.fetchone()
            try:
                cur.close()
            except Exception:
                pass
            return row is not None
        except Exception:
            return False

    def fetch(
        self,
        conn: Any,
        *,
        lookback_days: int,
        max_queries: int,
        min_runs: int,
        user_filter: str | None,
    ) -> list[str]:
        """Fetch recent statements when pg_stat_statements is installed."""

        del lookback_days
        try:
            cur = conn.cursor()
        except Exception:
            return []
        parts = [
            "SELECT query",
            "FROM pg_stat_statements",
            "WHERE calls >= %s",
        ]
        bind: list[Any] = [int(min_runs)]
        if user_filter:
            parts.append("AND userid = (SELECT oid FROM pg_roles WHERE rolname = %s LIMIT 1)")
            bind.append(str(user_filter))
        parts.append("ORDER BY calls DESC NULLS LAST")
        parts.append("LIMIT %s")
        bind.append(int(max_queries))
        stmt = " ".join(parts)
        try:
            cur.execute(stmt, tuple(bind))
            rows = cur.fetchall() or []
        except Exception:
            try:
                cur.close()
            except Exception:
                pass
            return []
        try:
            cur.close()
        except Exception:
            pass
        out: list[str] = []
        for row in rows:
            if not row:
                continue
            raw_q = row[0]
            if raw_q is None:
                continue
            out.append(_stable_sql_text_for_history(str(raw_q)))
        return out


@dataclass(frozen=True)
class DatabricksQueryLogSource:
    """Databricks query history fetcher."""

    def is_available(self, conn: Any) -> bool:
        """Return True when a Databricks session handle is present."""

        del conn
        return True

    def fetch(
        self,
        conn: Any,
        *,
        lookback_days: int,
        max_queries: int,
        min_runs: int,
        user_filter: str | None,
    ) -> list[str]:
        """Fetch SQL texts from system tables when permitted."""

        del min_runs
        try:
            cur = conn.cursor()
        except Exception:
            return []
        parts = [
            "SELECT statement_text AS q",
            "FROM system.query.history",
            "WHERE start_time >= date_sub(current_timestamp(), CAST(%s AS INT))",
        ]
        bind: list[Any] = [int(lookback_days)]
        if user_filter:
            parts.append("AND user_name = %s")
            bind.append(str(user_filter))
        parts.append("ORDER BY start_time DESC NULLS LAST")
        parts.append("LIMIT %s")
        bind.append(int(max_queries))
        stmt = " ".join(parts)
        try:
            cur.execute(stmt, tuple(bind))
            rows = cur.fetchall() or []
        except Exception:
            try:
                cur.close()
            except Exception:
                pass
            return []
        try:
            cur.close()
        except Exception:
            pass
        out: list[str] = []
        for row in rows:
            if not row:
                continue
            raw_q = row[0]
            if raw_q is None:
                continue
            out.append(_stable_sql_text_for_history(str(raw_q)))
        return out


def fetch_query_log(
    dialect_name: str,
    conn: Any,
    *,
    lookback_days: int,
    max_queries: int,
    min_runs: int,
    user_filter: str | None,
) -> list[str]:
    """
    Dispatch to the dialect-appropriate :class:`QueryLogSource` implementation.

    Args:

        dialect_name: ``postgresql`` or ``databricks``.

        conn: Live driver session.

        lookback_days: Rolling window in days.

        max_queries: Upper bound on rows returned.

        min_runs: Minimum execution count filter when supported.

        user_filter: Optional subject filter.

    Returns:

        SQL strings sorted for deterministic hashing by callers.
    """

    dn = str(dialect_name).strip().lower()
    if dn == "postgresql":
        src: QueryLogSource = PostgresQueryLogSource()
    elif dn == "databricks":
        src = DatabricksQueryLogSource()
    else:
        return []
    if not src.is_available(conn):
        return []
    return src.fetch(
        conn,
        lookback_days=lookback_days,
        max_queries=max_queries,
        min_runs=min_runs,
        user_filter=user_filter,
    )


def sql_history_paraphrase_budget() -> int:
    """Return configured paraphrase count for SQL-history warmup."""

    return int(WARMUP_PARAPHRASE_COUNT_FROM_SQL)
