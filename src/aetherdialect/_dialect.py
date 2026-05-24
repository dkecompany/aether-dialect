"""
Database dialect abstraction: AST/EXPLAIN validation, CTE extraction, and execution helpers for PostgreSQL and Databricks.

``sqlglot`` (Spark dialect plus permissive cross-dialect inspection) is a required core dependency. ``pglast`` (Postgres AST) is an optional dependency installed via the ``postgresql`` extra: it is not imported when this module loads; the first use of :class:`PostgresDialect` (or its PostgreSQL-only helpers) loads ``pglast`` once and raises ``ImportError`` if it is missing—similar to how Databricks-only drivers stay off the import path until a Databricks dialect or session is built. PostgreSQL parsing, structural validation, and join injection re-emission use ``pglast`` only on those paths; Databricks uses ``sqlglot`` (Spark dialect) for the same operations.

The:class:`Dialect` adapter exposes ``parse_select``, ``ordered_join_carrier_froms``, ``attach_joins``, and ``emit_sql`` so callers in :mod:`aetherdialect._sql_gen` and :mod:`aetherdialect._validation_execute` never name a parser library directly. Optional ``databricks.sql`` and ``pyspark`` are imported only when a Databricks connection or Spark session is constructed so installations without those drivers can still import this module.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import sqlglot
from sqlalchemy import create_engine, text

from . import _core_utils
from ._config import (
    AGGREGATE_FUNCTION_NAMES,
    DATABRICKS_TABLE_QUALIFY_SKIP_IDENTIFIERS,
    DBR_CARTESIAN_TOKENS,
    DBR_ZERO_ROW_RE,
    DOLLAR_PLACEHOLDER_RE,
    EXPLAIN_PERMISSION_DENIED_PATTERNS,
    NAMED_PLACEHOLDER_RE,
    PG_AGG_FUNCNAMES,
    PG_INNER_CONDITION_KEYS,
    PG_JOIN_CONDITION_KEYS,
    PG_JOIN_NODE_TYPES,
    PG_NAMED_PLACEHOLDER_RE,
    SQLGLOT_DIALECT_BY_ENGINE,
    UNIT_TO_DAYS,
    DatabricksRuntimeConfig,
    EngineConfig,
    PolicyConfig,
    PostgresRuntimeConfig,
    cost_cap_active,
    diagnostic_debug_enabled,
    effective_explain_timeout_ms,
)
from ._contracts_base import (
    AccessError,
    CatalogStructuralConstraintsIndex,
    CatalogTableStructuralConstraints,
    ConfigError,
    DatabasePingFailed,
    FKEdge,
    SchemaContext,
    SchemaGraph,
    SchemaInclude,
    SqlDiagnostic,
    SqlDiagnosticCode,
    StatementTimeoutError,
)
from ._contracts_core import FilterParam, NormalizedExpr, RuntimeIntent
from ._core_utils import (
    canonicalize_sql,
    debug,
    engine_connect_likely_transient,
    normalize_array_contains_param_value,
    pipeline_trace_lazy,
    reduce_structural_sql_placeholders,
    sha256,
    stable_json,
    substitute_params,
)
from ._schema import load_or_create_schema_databricks, load_or_create_schema_postgresql
from ._schema_profiling import (
    _cursor_rows_as_dicts,
    profile_schema,
    profile_schema_spark,
    profile_schema_sql_connector,
)


class _PgLastRuntime:
    """Lazy-loaded pglast bundle for PostgreSQL-only code paths."""

    __slots__ = (
        "parse_sql",
        "ast",
        "join_type",
        "a_expr_kind",
        "bool_expr_type",
        "raw_stream_cls",
    )

    def __init__(self) -> None:
        import pglast
        from pglast.enums import A_Expr_Kind, BoolExprType, JoinType
        from pglast.stream import RawStream

        self.parse_sql = pglast.parse_sql
        self.ast = pglast.ast
        self.join_type = JoinType
        self.a_expr_kind = A_Expr_Kind
        self.bool_expr_type = BoolExprType
        self.raw_stream_cls = RawStream


_pgl_runtime: _PgLastRuntime | None = None


def _require_pglast() -> _PgLastRuntime:
    global _pgl_runtime
    if _pgl_runtime is None:
        try:
            _pgl_runtime = _PgLastRuntime()
        except ImportError as exc:
            raise ImportError(
                "PostgresDialect requires the 'pglast' package. Install with: pip install aetherdialect[postgresql]"
            ) from exc
    return _pgl_runtime


class _PgParsedSelect:
    """Container for a pglast-parsed ``SELECT`` plus its named-placeholder round-trip map."""

    __slots__ = ("root", "name_to_index", "index_to_name")

    def __init__(
        self,
        root: Any,
        name_to_index: dict[str, int],
        index_to_name: dict[int, str],
    ) -> None:
        self.root = root
        self.name_to_index = name_to_index
        self.index_to_name = index_to_name


@dataclass(frozen=True, slots=True)
class JoinEdge:
    """
    One JOIN to attach to a carrier SELECT.

    ``table`` is the bare physical table name being joined in. ``alias`` is the AS-alias used when the same physical table appears multiple times (self-join); ``None`` for a single-instance join. ``kind`` is ``"INNER"`` or ``"LEFT"``. Each ``on_terms`` tuple is ``(left_token, left_col, right_token, right_col)`` where the tokens are the table name or alias to qualify the column with in the ``ON`` clause.
    """

    table: str
    alias: str | None
    kind: Literal["INNER", "LEFT"]
    on_terms: tuple[tuple[str, str, str, str], ...] = field(default_factory=tuple)


def _pg_encode_named_placeholders(
    sql: str,
) -> tuple[str, dict[str, int], dict[int, str]]:
    """Replace ``:name`` placeholders with ``$N`` so pglast can parse the SQL."""

    name_to_index: dict[str, int] = {}
    index_to_name: dict[int, str] = {}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in name_to_index:
            idx = len(name_to_index) + 1
            name_to_index[name] = idx
            index_to_name[idx] = name
        return f"${name_to_index[name]}"

    encoded = NAMED_PLACEHOLDER_RE.sub(repl, sql)
    return encoded, name_to_index, index_to_name


def _pg_decode_dollar_placeholders(sql: str, index_to_name: dict[int, str]) -> str:
    """Restore original ``:name`` placeholders from pglast-emitted ``$N`` markers."""

    if not index_to_name:
        return sql

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        name = index_to_name.get(idx)
        return f":{name}" if name is not None else match.group(0)

    return DOLLAR_PLACEHOLDER_RE.sub(repl, sql)


def _pg_relation_indexed_columns(schema: SchemaGraph | None, relation_name: str) -> set[str]:
    """Return the set of column names on *relation_name* that the schema marks as primary or foreign keys."""

    if schema is None or not relation_name:
        return set()
    table = schema.tables.get(relation_name)
    if table is None:
        return set()
    out: set[str] = set()
    for col in table.columns.values():
        if col.is_primary_key or col.is_foreign_key:
            out.add(col.name)
    return out


def _pg_walk_explain_plan(node: dict[str, Any], schema: SchemaGraph | None) -> list[SqlDiagnostic]:
    """Recursively walk a PostgreSQL ``EXPLAIN (FORMAT JSON)`` plan node and emit soft diagnostics."""

    diags: list[SqlDiagnostic] = []
    node_type = str(node.get("Node Type", ""))
    if node_type in PG_JOIN_NODE_TYPES:
        has_join_cond = any(k in node for k in PG_JOIN_CONDITION_KEYS)
        inner_plans = node.get("Plans", []) or []
        inner_has_cond = any(any(k in p for k in PG_INNER_CONDITION_KEYS) for p in inner_plans)
        if not has_join_cond and not inner_has_cond:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN,
                    message=f"{node_type} without join condition",
                    node_kind=node_type,
                )
            )
    plan_rows = node.get("Plan Rows")
    if isinstance(plan_rows, (int, float)) and plan_rows == 0:
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE,
                message="planner estimates zero rows",
                node_kind=node_type or None,
            )
        )
    if node_type == "Seq Scan":
        relation_name = str(node.get("Relation Name", ""))
        filter_text = str(node.get("Filter", ""))
        if filter_text and relation_name:
            indexed = _pg_relation_indexed_columns(schema, relation_name)
            if indexed and any(col in filter_text for col in indexed):
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED,
                        message=f"sequential scan on {relation_name} filters indexed column",
                        node_kind=node_type,
                        offending_identifier=relation_name,
                    )
                )
    for child in node.get("Plans", []) or []:
        diags.extend(_pg_walk_explain_plan(child, schema))
    return diags


def _pg_diagnostics_from_explain_json(raw: Any, schema: SchemaGraph | None) -> list[SqlDiagnostic]:
    """Parse a PostgreSQL ``EXPLAIN (FORMAT JSON)`` row payload into soft diagnostics."""

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return []
    else:
        payload = raw
    if not isinstance(payload, list) or not payload:
        return []
    head = payload[0]
    if not isinstance(head, dict):
        return []
    plan = head.get("Plan")
    if not isinstance(plan, dict):
        return []
    return _pg_walk_explain_plan(plan, schema)


def _databricks_diagnostics_from_explain_text(
    text_payload: str,
) -> list[SqlDiagnostic]:
    """Scan a Spark/Databricks ``EXPLAIN`` text payload for soft plan-shape findings."""

    if not text_payload:
        return []
    diags: list[SqlDiagnostic] = []
    if any(tok in text_payload for tok in DBR_CARTESIAN_TOKENS):
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN,
                message="Spark plan contains an unconditioned join",
            )
        )
    if DBR_ZERO_ROW_RE.search(text_payload):
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE,
                message="Spark plan estimates zero rows",
            )
        )
    return diags


def _pg_root_plan_estimates(plan: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return coarse ``(plan_rows, estimated_bytes)`` from a PostgreSQL JSON plan root."""

    rows_v = plan.get("Plan Rows")
    width_v = plan.get("Plan Width")
    pr = float(rows_v) if isinstance(rows_v, (int, float)) else None
    pw = float(width_v) if isinstance(width_v, (int, float)) else None
    if pr is None:
        return None, None
    est_bytes = pr * pw if pw is not None else None
    return pr, est_bytes


def _databricks_plan_stats_from_explain_text(
    text_payload: str,
) -> tuple[float | None, float | None]:
    """Extract coarse row and byte estimates from Spark/Databricks ``EXPLAIN COST`` text."""

    if not text_payload:
        return None, None
    row_est: float | None = None
    for pat in (
        r"(?i)Statistics\s*\([^)]*rowCount\s*=\s*(\d+)",
        r"(?i)rowCount[=:\s]+(\d+)",
        r"(?i)numRows[=:\s]+(\d+)",
    ):
        m = re.search(pat, text_payload)
        if m:
            try:
                row_est = float(m.group(1))
                break
            except (TypeError, ValueError):
                continue
    byte_est: float | None = None
    m_sz = re.search(r"(?i)sizeInBytes\s*=\s*([\d.]+)\s*([KMGT]?iB|[KMGT]?B|bytes?)", text_payload)
    if m_sz:
        try:
            val = float(m_sz.group(1))
            unit = (m_sz.group(2) or "b").lower().replace("bytes", "b").replace("byte", "b")
            mult = {
                "b": 1.0,
                "ib": 1.0,
                "kib": 1024.0,
                "mib": 1024.0**2,
                "gib": 1024.0**3,
                "tib": 1024.0**4,
                "kb": 1000.0,
                "mb": 1_000_000.0,
                "gb": 1e9,
                "tb": 1e12,
            }
            byte_est = val * mult.get(unit, 1.0)
        except (TypeError, ValueError, IndexError):
            byte_est = None
    return row_est, byte_est


def _explain_cost_gate_violation(est_rows: float | None, est_bytes: float | None) -> tuple[bool, str]:
    """Return ``(True, message)`` when planner estimates exceed configured caps."""

    caps_r = PolicyConfig.MAX_QUERY_COST_ROWS
    caps_b = PolicyConfig.MAX_QUERY_COST_BYTES
    over_r = cost_cap_active(caps_r) and est_rows is not None and est_rows > float(caps_r)
    over_b = cost_cap_active(caps_b) and est_bytes is not None and est_bytes > float(caps_b)
    if not (over_r or over_b):
        return False, ""
    msg = (
        f"EXPLAIN cost gate exceeded: estimated_rows={est_rows} estimated_bytes={est_bytes} "
        f"(limits rows<={caps_r} bytes<={caps_b})"
    )
    return True, msg


def _trace_finalize_render_stage(stage: str, sql_in: str, sql_out: str) -> None:
    """
    Log one ``finalize_render`` sub-step for debugging and ``PIPELINE_TRACE`` capture.

    Args:

        stage: Sub-step name (e.g. ``prepare_for_execution``).

        sql_in: SQL string entering the sub-step.

        sql_out: SQL string leaving the sub-step.
    """

    debug(f"[dialect.finalize_render.{stage}] in_sql_len={len(sql_in)} out_sql_len={len(sql_out)}")
    pipeline_trace_lazy(
        f"dialect.finalize_render.{stage}",
        lambda: stable_json({"in": sql_in, "out": sql_out}),
    )


def _qualify_tables_ast(
    sql: str,
    *,
    sqlglot_dialect: str,
    catalog: str | None,
    schema: str,
    cte_names: set[str],
    backtick: bool,
) -> str:
    """
    Qualify bare table references with ``schema`` (and optional ``catalog``) using sqlglot AST.

    Walks every :class:`sqlglot.exp.Table` node, skipping CTE references, identifiers in
    :data:`DATABRICKS_TABLE_QUALIFY_SKIP_IDENTIFIERS`, and tables that already carry a
    ``db`` or ``catalog`` qualifier. On any parse failure the original SQL is returned
    unchanged so callers never see a broken transformation.

    Args:

        sql: Source SQL.

        sqlglot_dialect: sqlglot read/write dialect, e.g. ``"spark"`` or ``"postgres"``.

        catalog: Catalog/database to set when the dialect supports three-part names.

        schema: Schema name to set as ``db`` on each qualifying table node.

        cte_names: Names declared by CTEs in the same statement; matches are skipped.

        backtick: When True identifiers are emitted with backticks (Spark); when False
        with the dialect's default quoting.
    """
    if not sql or not sql.strip():
        return sql
    if not schema:
        return sql
    cte_names_lower = {n.lower() for n in cte_names if n}
    skip_lower = {s.lower() for s in DATABRICKS_TABLE_QUALIFY_SKIP_IDENTIFIERS}
    try:
        parsed = sqlglot.parse_one(sql, read=sqlglot_dialect)
    except Exception:
        debug(f"[_qualify_tables_ast] sqlglot parse failed; preserving input SQL (len={len(sql)})")
        return sql
    if parsed is None:
        debug(f"[_qualify_tables_ast] sqlglot parse_one returned None; preserving input SQL (len={len(sql)})")
        return sql
    for cte in parsed.find_all(sqlglot.exp.CTE):
        alias = cte.alias_or_name
        if alias:
            cte_names_lower.add(alias.lower())
    for table in parsed.find_all(sqlglot.exp.Table):
        name = (table.name or "").lower()
        if not name:
            continue
        if name in cte_names_lower:
            continue
        if name in skip_lower:
            continue
        if table.args.get("db") or table.args.get("catalog"):
            continue
        table.set("db", sqlglot.exp.to_identifier(schema, quoted=backtick))
        if catalog:
            table.set("catalog", sqlglot.exp.to_identifier(catalog, quoted=backtick))
    try:
        out = parsed.sql(dialect=sqlglot_dialect, identify=backtick)
        if sql.strip() and not out.strip():
            debug(f"[_qualify_tables_ast] sqlglot emission empty; preserving input SQL (len={len(sql)})")
            return sql
        return out
    except Exception:
        debug(f"[_qualify_tables_ast] sqlglot serialize failed; preserving input SQL (len={len(sql)})")
        return sql


def finalize_executable_sql(
    sql_param: str,
    params: dict[str, Any],
    structural_defaults: dict[str, Any] | None = None,
    *,
    sqlglot_dialect: str,
) -> str:
    """Reduce structural placeholders, substitute parameters, then AST-simplify the literal SQL."""

    reduced, remaining = reduce_structural_sql_placeholders(
        sql_param,
        dict(params),
        structural_defaults,
    )
    substituted = substitute_params(reduced, remaining)
    return sql_simplify_executable(substituted, sqlglot_dialect=sqlglot_dialect)


def _dbr_format_partition_literal(val: Any) -> str:
    """Format a Python value as a Spark SQL literal."""

    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    return f"'{str(val)}'"


def _dbr_format_partition_predicate(
    table: str,
    col: str,
    fp: FilterParam,
    params: dict[str, Any],
) -> str | None:
    """Format a single partition predicate for Spark SQL."""

    qual = f"`{table}`.`{col}`"
    val = fp.param_key and params.get(fp.param_key)
    if val is None and fp.raw_value is not None:
        val = fp.raw_value

    if fp.op == "=":
        if val is None:
            return None
        lit = _dbr_format_partition_literal(val)
        return f"{qual} = {lit}"
    if fp.op in (">=", "<=", ">", "<"):
        if val is None:
            return None
        lit = _dbr_format_partition_literal(val)
        return f"{qual} {fp.op} {lit}"
    if fp.op == "in":
        if val is None:
            return None
        if isinstance(val, list):
            parts = [_dbr_format_partition_literal(v) for v in val]
            return f"{qual} IN ({', '.join(parts)})"
        lit = _dbr_format_partition_literal(val)
        return f"{qual} IN ({lit})"
    return None


def _dbr_get_column_ref(expr: NormalizedExpr) -> tuple[str | None, str | None]:
    """Extract table and column names from a normalized expression primary term."""

    term = (expr.primary_term or "").strip()
    if not term:
        return None, None
    if "." in term:
        parts = term.rsplit(".", 1)
        return parts[0].strip() or None, parts[1].strip() or None
    return None, term


def _dbr_format_grouped_predicate(
    table: str,
    col: str,
    fps: list[FilterParam],
    params: dict[str, Any],
) -> str | None:
    """Format a grouped partition predicate as IN or BETWEEN-style SQL."""

    qual = f"`{table}`.`{col}`"
    ops = {fp.op for fp in fps}
    if ops <= {"="} and len(fps) > 1:
        parts = []
        for fp in fps:
            val = fp.param_key and params.get(fp.param_key) or fp.raw_value
            if val is not None:
                parts.append(_dbr_format_partition_literal(val))
        if parts:
            return f"{qual} IN ({', '.join(parts)})"
        return None
    if ops <= {">=", "<="} and len(fps) == 2:
        ge = next((f for f in fps if f.op == ">="), None)
        le = next((f for f in fps if f.op == "<="), None)
        if ge and le:
            v1 = ge.param_key and params.get(ge.param_key) or ge.raw_value
            v2 = le.param_key and params.get(le.param_key) or le.raw_value
            if v1 is not None and v2 is not None:
                return (
                    f"({qual} >= {_dbr_format_partition_literal(v1)} AND {qual} <= {_dbr_format_partition_literal(v2)})"
                )
        return None
    if len(fps) == 1:
        return _dbr_format_partition_predicate(table, col, fps[0], params)
    return None


def _dbr_build_partition_predicates(
    schema: SchemaGraph,
    intent: RuntimeIntent,
    params: dict[str, Any],
) -> list[str]:
    """Build Spark-formatted partition predicates from intent filters and params."""

    tables = intent.tables or []
    filters = intent.filters_param or []
    if not tables or not filters:
        return []

    grouped: dict[tuple[str, str], list[FilterParam]] = {}

    for table_name in tables:
        table_meta = schema.tables.get(table_name)
        if not table_meta or not table_meta.partition_columns:
            continue
        part_cols_lower = {c.lower(): c for c in table_meta.partition_columns}

        for fp in filters:
            col_ref = _dbr_get_column_ref(fp.left_expr)
            if not col_ref:
                continue
            table_part, col_part = col_ref
            col_lower = col_part.lower() if col_part else ""
            if col_lower not in part_cols_lower:
                continue
            actual_col = part_cols_lower[col_lower]
            table_for_pred = table_part or (tables[0] if tables else "")
            if table_part and table_part.lower() not in {t.lower() for t in tables}:
                continue
            key = (table_for_pred.lower(), actual_col.lower())
            grouped.setdefault(key, []).append(fp)

    result: list[str] = []
    for (table_key, col_key), fps in grouped.items():
        table_name = next((t for t in tables if t.lower() == table_key), tables[0] if tables else "")
        table_meta = schema.tables.get(table_name)
        col_name = (
            next(
                (c for c in table_meta.partition_columns if c.lower() == col_key),
                col_key,
            )
            if table_meta
            else col_key
        )
        pred = _dbr_format_grouped_predicate(table_name, col_name, fps, params)
        if pred:
            result.append(pred)

    return result


def _dbr_contains_filter_param_keys(intent: RuntimeIntent) -> set[str]:
    """Collect ``param_key`` values from ``contains`` filters in main and CTE intents."""

    keys: set[str] = set()
    for cte in intent.cte_steps or []:
        for fp in cte.filters_param or []:
            if fp.op == "contains" and fp.param_key:
                keys.add(fp.param_key)
    for fp in intent.filters_param or []:
        if fp.op == "contains" and fp.param_key:
            keys.add(fp.param_key)
    return keys


def _dbr_flatten_param_values(intent: RuntimeIntent) -> dict[str, Any]:
    """Merge CTE and main params and normalize values used by ``contains`` filters."""

    merged: dict[str, Any] = {}
    for cte in intent.cte_steps or []:
        merged.update(cte.param_values or {})
    merged.update(intent.param_values or {})
    contains_keys = _dbr_contains_filter_param_keys(intent)
    if not contains_keys:
        return merged
    out = dict(merged)
    for key in contains_keys:
        if key in out:
            out[key] = normalize_array_contains_param_value(out[key])
    return out


def _dbr_predicate_already_in_sql(
    sql: str,
    combined: str,
    predicates: list[str],
) -> bool:
    """Return True when every partition predicate already appears in the SQL text."""

    sql_norm = sql.replace(" ", "").replace("\n", " ").lower()
    for pred in predicates:
        pred_norm = pred.replace(" ", "").lower()
        if pred_norm not in sql_norm:
            return False
    return True


def _dbr_append_where_via_ast(sql: str, predicate: str) -> str | None:
    """Append *predicate* to the WHERE clause using a sqlglot Spark AST round-trip; return ``None`` on parse failure."""

    try:
        tree = sqlglot.parse_one(sql, read="spark")
    except Exception:
        return None
    if not isinstance(tree, sqlglot.exp.Select):
        return None
    try:
        updated = tree.where(predicate, append=True, dialect="spark")
    except Exception:
        return None
    return updated.sql(dialect="spark")


def _dbr_append_to_where(sql: str, predicate: str) -> str:
    """Append *predicate* to the SQL's WHERE clause via the Spark AST; raise on failure."""

    out = _dbr_append_where_via_ast(sql, predicate)
    if out is None:
        raise ValueError("Spark AST refused to append WHERE predicate; SQL is unparseable")
    return out


def _dbr_inject_partition_filters(
    sql: str,
    schema: SchemaGraph,
    intent: RuntimeIntent,
) -> str:
    """Append missing predicates on Delta table partition columns for pruning."""

    params = _dbr_flatten_param_values(intent)
    predicates = _dbr_build_partition_predicates(schema, intent, params)
    if not predicates:
        return sql
    combined = " AND ".join(predicates)
    if _dbr_predicate_already_in_sql(sql, combined, predicates):
        return sql
    return _dbr_append_to_where(sql, combined)


def _is_permission_denied_error(message: str) -> bool:
    """Return True when *message* indicates the database refused EXPLAIN due to credentials."""

    lower = (message or "").lower()
    return any(pat in lower for pat in EXPLAIN_PERMISSION_DENIED_PATTERNS)


def active_sqlglot_dialect() -> str:
    """Return the sqlglot dialect token matching the configured ``EngineConfig.TYPE``."""

    engine_type = (EngineConfig.TYPE or "").strip().lower()
    token = SQLGLOT_DIALECT_BY_ENGINE.get(engine_type)
    if token is None:
        raise ValueError(
            f"No sqlglot dialect mapping for engine type {engine_type!r}; expected one of "
            f"{sorted(SQLGLOT_DIALECT_BY_ENGINE)}"
        )
    return token


def _inspect_parse(sql: str, *, sqlglot_dialect: str) -> sqlglot.exp.Expression | None:
    """Parse *sql* with the given sqlglot *sqlglot_dialect*; returns ``None`` on parser failure."""

    if not sql or not isinstance(sql, str):
        return None
    if not sqlglot_dialect:
        raise ValueError("_inspect_parse requires a non-empty sqlglot_dialect")
    try:
        return sqlglot.parse_one(sql, read=sqlglot_dialect)
    except Exception:
        return None


def _normalize_named_placeholders(sql: str) -> str:
    """
    Convert dialect-specific named placeholders back to ``:name`` form.

    sqlglot's Postgres generator emits ``%(name)s`` when serialising
    :class:`sqlglot.expressions.Placeholder` nodes, but the rest of the pipeline
    (``substitute_params``, SQLAlchemy ``text(...)`` binds) expects the canonical
    ``:name`` form. This helper rewrites ``%(name)s`` → ``:name`` so the placeholder
    template stays dialect-agnostic regardless of the round-trip dialect used during
    AST simplification or parameter abstraction.

    Args:

        sql: SQL text potentially containing ``%(name)s`` placeholders.

    Returns:

        The same SQL with named placeholders normalised to ``:name``.
    """

    return PG_NAMED_PLACEHOLDER_RE.sub(lambda m: f":{m.group(1)}", sql)


def _format_interval_unit(unit: str, amount: int) -> tuple[int, str]:
    """
    Return ``(amount, unit)`` rewritten to a SQL-compatible ANSI interval unit.

    SQL ``INTERVAL`` literals do not understand ``quarter`` or ``half_year``; both PostgreSQL
    and Spark expect base units such as ``month``. This helper converts those composite units
    to ``month`` (``quarter`` -> 3, ``half_year`` -> 6) and pluralises the unit when *amount*
    is not 1 so the rendered fragment reads naturally (``2 days``, ``1 month``, etc.).

    Args:

        unit: Canonical relative-date unit name (``day``, ``week``, ``month``, ``quarter``,
        ``half_year``, ``year``, ``hour``, ``minute``, or ``second``).

        amount: Interval magnitude in *unit*.

    Returns:

        Tuple ``(scaled_amount, plural_unit)`` ready to interpolate into an
        ``INTERVAL '<amount> <unit>'`` literal.
    """

    canonical = (unit or "").strip().lower()
    if canonical == "quarter":
        scaled = amount * 3
        base = "month"
    elif canonical == "half_year":
        scaled = amount * 6
        base = "month"
    else:
        scaled = amount
        base = canonical or "day"
    plural = f"{base}s" if scaled != 1 else base
    return scaled, plural


def _emit_via_ast(sql: str, dialect_name: str) -> str:
    """
    Round-trip a SQL fragment through the sqlglot AST and re-emit via the dialect generator.

    Used by dialect render helpers so the final fragment passes through a parser/generator
    pair (rather than being built only by f-string concatenation). The post-processor
    restores ``:name`` placeholders that the Postgres generator otherwise rewrites to
    ``%(name)s``.

    Args:

        sql: Pre-composed SQL fragment.

        dialect_name: sqlglot dialect identifier (``"postgres"`` or ``"spark"``).

    Returns:

        Re-emitted SQL with canonical ``:name`` placeholders.
    """
    tree = sqlglot.parse_one(sql, dialect=dialect_name)
    return _normalize_named_placeholders(tree.sql(dialect=dialect_name))


def _outer_select(parsed: sqlglot.exp.Expression) -> sqlglot.exp.Select | None:
    """Return the outer ``Select`` from a parsed expression, ignoring CTE inner selects."""

    if isinstance(parsed, sqlglot.exp.Select):
        return parsed
    inner = parsed.find(sqlglot.exp.Select)
    return inner if isinstance(inner, sqlglot.exp.Select) else None


def sql_outer_select_aliases(sql: str, *, sqlglot_dialect: str) -> list[str]:
    """Return the column display names of the outermost ``SELECT`` projection list."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return []
    select = _outer_select(parsed)
    if select is None:
        return []
    headers: list[str] = []
    for proj in select.expressions or []:
        alias = proj.alias_or_name if hasattr(proj, "alias_or_name") else ""
        if alias:
            headers.append(alias)
            continue
        if isinstance(proj, sqlglot.exp.Column):
            headers.append(proj.name)
            continue
        headers.append(proj.sql().replace(" ", "_"))
    return headers


def sql_outer_has_join_or_comma_from(sql: str, *, sqlglot_dialect: str) -> bool:
    """Return True when the outer ``SELECT`` uses an explicit JOIN or a comma-separated multi-relation FROM."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return False
    select = _outer_select(parsed)
    if select is None:
        return False
    if select.args.get("joins"):
        return True
    from_clause = select.args.get("from")
    if from_clause is None:
        return False
    tables: list[sqlglot.exp.Expression] = []
    if isinstance(from_clause, sqlglot.exp.From):
        first = from_clause.this
        if first is not None:
            tables.append(first)
        extras = from_clause.args.get("expressions") or []
        tables.extend(extras)
    return sum(1 for t in tables if isinstance(t, (sqlglot.exp.Table, sqlglot.exp.Subquery))) >= 2


def sql_count_outer_joins(sql: str, *, sqlglot_dialect: str) -> int:
    """Return the number of explicit JOIN clauses across all SELECT scopes."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return 0
    return sum(len(node.args.get("joins") or []) for node in parsed.find_all(sqlglot.exp.Select))


def sql_has_group_by(sql: str, *, sqlglot_dialect: str) -> bool:
    """Return True when any ``SELECT`` in *sql* has a ``GROUP BY`` clause."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return False
    return any(node.args.get("group") for node in parsed.find_all(sqlglot.exp.Select))


def sql_has_distinct(sql: str, *, sqlglot_dialect: str) -> bool:
    """Return True when any ``SELECT`` in *sql* uses ``SELECT DISTINCT``."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return False
    return any(node.args.get("distinct") for node in parsed.find_all(sqlglot.exp.Select))


def sql_has_aggregate(sql: str, *, sqlglot_dialect: str) -> bool:
    """Return True when *sql* contains an aggregate function call."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return False
    for node in parsed.walk():
        candidate = node[0] if isinstance(node, tuple) else node
        if isinstance(
            candidate,
            (
                sqlglot.exp.Sum,
                sqlglot.exp.Count,
                sqlglot.exp.Avg,
                sqlglot.exp.Min,
                sqlglot.exp.Max,
                sqlglot.exp.Stddev,
                sqlglot.exp.Variance,
            ),
        ):
            return True
        if isinstance(candidate, sqlglot.exp.Anonymous):
            name = (candidate.name or "").lower()
            if name in AGGREGATE_FUNCTION_NAMES:
                return True
    return False


def sql_cte_names(sql: str, *, sqlglot_dialect: str) -> set[str]:
    """Return lowercase names of all CTE definitions in *sql*."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return set()
    names: set[str] = set()
    with_clause = parsed.args.get("with") if isinstance(parsed, sqlglot.exp.Select) else None
    if with_clause is None:
        with_clause = parsed.find(sqlglot.exp.With)
    if with_clause is None:
        return names
    for cte in with_clause.expressions or []:
        if isinstance(cte, sqlglot.exp.CTE):
            alias_name = cte.alias_or_name
            if alias_name:
                names.add(alias_name.lower())
    return names


def sql_tables_referenced(sql: str, *, sqlglot_dialect: str) -> set[str]:
    """Return lowercase physical-table names referenced in *sql*, excluding CTE definitions."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return set()
    cte_names = sql_cte_names(sql, sqlglot_dialect=sqlglot_dialect)
    tables: set[str] = set()
    for tbl in parsed.find_all(sqlglot.exp.Table):
        name = (tbl.name or "").lower()
        if name and name not in cte_names:
            tables.add(name)
    return tables


def _simplify_arithmetic_identities_in_tree(
    tree: sqlglot.exp.Expression,
) -> sqlglot.exp.Expression:
    """In-place simplify ``1*x``, ``x*1``, ``x+0``, ``x-0``, drop ``LIMIT NULL`` and rewrite ``NOT (X IS NULL)`` to ``X IS NOT NULL``."""

    for node in list(tree.walk()):
        candidate = node[0] if isinstance(node, tuple) else node
        if isinstance(candidate, sqlglot.exp.Mul):
            left = candidate.left
            right = candidate.right
            if isinstance(left, sqlglot.exp.Literal) and not left.is_string and left.this in ("1", "1.0"):
                candidate.replace(right.copy())
                continue
            if isinstance(right, sqlglot.exp.Literal) and not right.is_string and right.this in ("1", "1.0"):
                candidate.replace(left.copy())
                continue
        if isinstance(candidate, (sqlglot.exp.Add, sqlglot.exp.Sub)):
            right = candidate.right
            if isinstance(right, sqlglot.exp.Literal) and not right.is_string and right.this in ("0", "0.0"):
                candidate.replace(candidate.left.copy())
                continue
        if isinstance(candidate, sqlglot.exp.Not):
            inner = candidate.this
            if (
                isinstance(inner, sqlglot.exp.Is)
                and isinstance(inner.args.get("expression"), sqlglot.exp.Null)
                and inner.this is not None
            ):
                replacement = sqlglot.exp.Is(
                    this=inner.this.copy(),
                    expression=sqlglot.exp.Not(this=sqlglot.exp.Null()),
                )
                candidate.replace(replacement)
                continue
    for select in tree.find_all(sqlglot.exp.Select):
        limit_node = select.args.get("limit")
        if limit_node is None:
            continue
        expr = limit_node.expression if hasattr(limit_node, "expression") else None
        if isinstance(expr, sqlglot.exp.Null) or (
            isinstance(expr, sqlglot.exp.Column) and (expr.name or "").lower() == "none"
        ):
            select.set("limit", None)
    return tree


def sql_simplify_executable(sql: str, *, sqlglot_dialect: str) -> str:
    """Drop trivial arithmetic identities (``1*x``, ``x*1``, ``x+0``, ``x-0``) and ``LIMIT NULL/None`` via AST."""

    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        debug(f"[sql_simplify_executable] parse returned None; preserving input SQL (len={len(sql)})")
        return sql
    simplified = _simplify_arithmetic_identities_in_tree(parsed)
    try:
        out = _normalize_named_placeholders(simplified.sql(dialect=sqlglot_dialect))
        if sql.strip() and not out.strip():
            debug(f"[sql_simplify_executable] sqlglot emission empty; preserving input SQL (len={len(sql)})")
            return sql
        return out
    except Exception:
        debug(f"[sql_simplify_executable] sqlglot round-trip refused; preserving input SQL (len={len(sql)})")
        return sql


def parse_extract_arguments(inner: str) -> tuple[str, str] | None:
    """
    Parse the body of a SQL ``EXTRACT(<unit> FROM <expr>)`` call.

    Splits *inner* on the first top-level ``FROM`` (whitespace-bounded, depth zero) and
    returns ``(unit_lowercase, source_expr_sql)``. Returns ``None`` when no top-level
    ``FROM`` separator is found or either side is empty. The source expression is
    returned verbatim so the caller's downstream rendering preserves any dialect-specific
    quoting present in *inner*.

    Args:

        inner: The argument body between the parentheses of an ``EXTRACT`` call.

    Returns:

        ``(unit_lowercase, source_expr_sql)`` on success, or ``None``.
    """

    if not isinstance(inner, str) or not inner.strip():
        return None
    text_body = inner
    depth = 0
    in_single = False
    in_double = False
    length = len(text_body)
    i = 0
    while i < length:
        ch = text_body[i]
        if in_single:
            if ch == "'" and i + 1 < length and text_body[i + 1] == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == '"' and i + 1 < length and text_body[i + 1] == '"':
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if (
            depth == 0
            and (ch == "F" or ch == "f")
            and i + 4 <= length
            and text_body[i : i + 4].lower() == "from"
            and (i == 0 or text_body[i - 1].isspace())
            and (i + 4 == length or text_body[i + 4].isspace())
        ):
            unit_part = text_body[:i].strip()
            source_part = text_body[i + 4 :].strip()
            if not unit_part or not source_part:
                return None
            return unit_part.strip("'\"").lower(), source_part
        i += 1
    return None


def parameter_abstract(sql: str, *, sqlglot_dialect: str) -> tuple[str, dict[str, Any]]:
    """
    Replace literal nodes with ``:p1``, ``:p2``, … via sqlglot AST traversal.

    Numeric literals are recorded as their parsed value (``int`` or ``float``); string
    literals are recorded with surrounding single quotes preserved. Returns ``(sql, {})``
    unchanged when the SQL is unparseable.

    Args:

        sql: SQL with inline literals.

        sqlglot_dialect: sqlglot dialect token (``"postgres"`` or ``"spark"``).

    Returns:

        ``(sql_with_placeholders, {pN: original_literal})``.
    """

    if not isinstance(sql, str) or not sql:
        return sql, {}
    parsed = _inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
    if parsed is None:
        return sql, {}
    params: dict[str, Any] = {}
    counter = 0
    for literal in list(parsed.find_all(sqlglot.exp.Literal)):
        counter += 1
        key = f"p{counter}"
        if literal.is_string:
            params[key] = f"'{literal.name}'"
        else:
            raw_value = literal.this
            try:
                num = float(raw_value)
                params[key] = int(num) if num == int(num) else num
            except (TypeError, ValueError):
                params[key] = raw_value
        literal.replace(sqlglot.exp.Placeholder(this=key))
    try:
        rendered = _normalize_named_placeholders(parsed.sql(dialect=sqlglot_dialect))
    except Exception:
        return sql, {}
    return " ".join(rendered.split()).strip(), params


def compute_sql_fp(sql: str, *, sqlglot_dialect: str) -> str:
    """Return the canonical-abstracted-lowercased SHA-256 fingerprint for identity keys."""

    if not sql:
        return sha256("")
    canon = canonicalize_sql(sql)
    abstracted, _ = parameter_abstract(canon, sqlglot_dialect=sqlglot_dialect)
    return sha256(abstracted.lower())


def _check_schema_references_shared(
    refs: list[tuple[str | None, str]],
    alias_to_table: dict[str, str],
    cte_names: set[str],
    schema: SchemaGraph,
) -> list[SqlDiagnostic]:
    """
    Validate ``(table_or_alias, column)`` pairs against *schema*.

    Resolves each prefix through *alias_to_table*. References whose resolved table is a CTE name in *cte_names* are skipped (CTE projection columns are not in the schema graph). Unqualified references are checked for ambiguity across all FROM-side tables; qualified references are checked for table existence and column membership using lowercase normalisation.
    """
    diags: list[SqlDiagnostic] = []
    seen: set[tuple[str | None, str]] = set()
    from_tables: list[str] = []
    for alias_key, real in alias_to_table.items():
        real_low = (real or "").lower()
        if real_low and real_low not in cte_names and real_low in schema.tables:
            if real_low not in from_tables:
                from_tables.append(real_low)
        _ = alias_key
    for prefix, column in refs:
        column_low = (column or "").lower()
        if not column_low or column_low == "*":
            continue
        key = (prefix.lower() if prefix else None, column_low)
        if key in seen:
            continue
        seen.add(key)
        if prefix is None:
            owners = [t for t in from_tables if column_low in schema.tables[t].columns]
            if len(owners) == 0:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.UNKNOWN_COLUMN,
                        message=f"unknown column {column_low!r}",
                        node_kind="ColumnRef",
                        offending_identifier=column_low,
                    )
                )
            elif len(owners) > 1:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.AMBIGUOUS_COLUMN,
                        message=f"ambiguous column {column_low!r} in {owners}",
                        node_kind="ColumnRef",
                        offending_identifier=column_low,
                        details={"owners": ",".join(owners)},
                    )
                )
            continue
        prefix_low = prefix.lower()
        resolved = (alias_to_table.get(prefix_low) or prefix_low).lower()
        if resolved in cte_names:
            continue
        if resolved not in schema.tables:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.UNKNOWN_TABLE,
                    message=f"unknown table {resolved!r}",
                    node_kind="Table",
                    offending_identifier=resolved,
                )
            )
            continue
        if column_low not in schema.tables[resolved].columns:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.UNKNOWN_COLUMN,
                    message=f"unknown column {resolved}.{column_low}",
                    node_kind="ColumnRef",
                    offending_identifier=f"{resolved}.{column_low}",
                )
            )
    return diags


def _reflect_include_for_schema_build(ctx: SchemaContext) -> SchemaInclude:
    """Mirror :func:`aetherdialect._schema._effective_reflect_include` so partial and full rebuilds agree."""

    if ctx.allow_objects:
        return "both"
    return ctx.include


class Dialect:
    """Base interface for dialect-specific SQL validation and introspection."""

    name: str = "base"
    sqlglot_dialect: ClassVar[str] = ""

    def __init__(self, config):
        """
        Attach runtime configuration used by dialect operations.

        Args:

            config: `PostgresRuntimeConfig`, `DatabricksRuntimeConfig`, or compatible runtime config.

        Returns:

            None.
        """
        self.config = config
        self._explain_disabled: bool = False

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        """
        Backwards-shaped wrapper over :meth:`ast_validate_full`.

        Returns ``(True, "")`` when no diagnostics are emitted, otherwise ``(False, first_code)`` where ``first_code`` is the string value of the first diagnostic's code.
        """
        diags = self.ast_validate_full(sql)
        if not diags:
            return True, ""
        return False, str(diags[0].code.value)

    def ast_validate_full(
        self,
        sql: str,
        *,
        schema: SchemaGraph | None = None,
        declared_params: set[str] | None = None,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> list[SqlDiagnostic]:
        """
        Validate SQL structurally and (when *schema* is provided) semantically without a live connection.

        Args:

            sql: SQL text to validate.

            schema: Optional schema graph; enables column/table existence and
            ambiguity checks.

            declared_params: Optional set of named placeholder labels declared
            by the caller; enables param coverage checks.

            scalar_cte_names: Optional lowercased CTE names allowed as INNER JOIN
            without ``ON`` / scalar ``CROSS JOIN`` targets.

        Returns:

            List of structured :class:`SqlDiagnostic` findings; empty list means valid.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def parse_select(self, sql: str) -> Any | None:
        """
        Parse *sql* with the dialect's native AST library and return an opaque handle.

        The handle is consumed only by :meth:`ordered_join_carrier_froms`,
        :meth:`attach_joins`, and :meth:`emit_sql` on the same dialect instance.
        Returns ``None`` when the SQL cannot be parsed or is not a single SELECT.

        Args:

            sql: SQL text containing ``:pN`` / ``:sN`` named placeholders.

        Returns:

            Opaque parsed handle or ``None``.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def ordered_join_carrier_froms(self, parsed: Any) -> list[Any] | None:
        """
        Return per-FROM handles in JOIN-placeholder injection order.

        Order is each CTE inner SELECT's FROM left-to-right followed by the outer SELECT's FROM.
        Returns ``None`` for unsupported shapes (e.g. top-level ``UNION``).

        Args:

            parsed: Handle returned by :meth:`parse_select`.

        Returns:

            Ordered list of opaque FROM handles or ``None``.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def attach_joins(
        self,
        parsed: Any,
        from_handle: Any,
        edges: list[JoinEdge],
    ) -> bool:
        """
        Attach the given structured *edges* as JOIN nodes onto *from_handle*.

        Implementations construct dialect-native JOIN AST nodes directly from *edges* and
        graft them into *from_handle* without re-parsing any SQL fragment.

        Args:

            parsed: Handle returned by :meth:`parse_select`.

            from_handle: One element returned by :meth:`ordered_join_carrier_froms`.

            edges: Structured join edges in the order they should appear.

        Returns:

            ``True`` on success, ``False`` when grafting fails.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def attach_extra_from_and_where(
        self,
        parsed: Any,
        from_handle: Any,
        extra_from_tables: list[str],
        where_edges: list[JoinEdge],
    ) -> bool:
        """
        AND-inject *where_edges*' equality predicates into *from_handle*'s ``WHERE`` and append
        any *extra_from_tables* to its ``FROM`` clause.

        Used to render Tier-B semantic edges (``edge_kind`` ``semantic_profile`` /
        ``semantic_profile_virtual``) as comma-FROM + ``WHERE`` equality predicates rather than
        ``JOIN ... ON``. ``where_edges[i].on_terms`` is a tuple of
        ``(left_token, left_col, right_token, right_col)`` equality conjuncts that get AND-ed
        into the existing ``WHERE``.

        Returns ``True`` on success (including the no-op case when both lists are empty),
        ``False`` when grafting fails.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def from_anchor_of(self, carrier: Any) -> str | None:
        """
        Return the bare anchor table name of *carrier*'s ``FROM`` clause.

        Used by :mod:`aetherdialect._sql_gen` to orient join signatures around the carrier's
        ``FROM`` table without resorting to text regex over the rendered SQL prefix.

        Args:

            carrier: One element returned by :meth:`ordered_join_carrier_froms`.

        Returns:

            Bare lowercase table name, or ``None`` when the FROM cannot be resolved
            to a single table.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def replace_projection(
        self,
        parsed: Any,
        items: list[tuple[str, str | None]],
    ) -> bool:
        """
        Replace the outer ``SELECT`` projection list with *items*.

        Each ``(expr_sql, alias)`` pair is parsed as a single SELECT-list expression in the dialect's native parser and grafted as a ``ResTarget``/``sqlglot.exp.Alias`` node so the surrounding SQL is reconstructed without text splicing.

        Returns ``True`` on success and ``False`` when any expression or the host statement cannot be parsed.
        """
        raise NotImplementedError

    def emit_sql(self, parsed: Any) -> str:
        """
        Re-emit SQL from *parsed* preserving ``:pN`` / ``:sN`` placeholders verbatim.

        Args:

            parsed: Handle returned by :meth:`parse_select`.

        Returns:

            Rendered SQL string.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def explain_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, str]:
        """
        Backwards-shaped wrapper over :meth:`explain_diagnose`.

        Returns ``(ok, raw_message)`` discarding structured diagnostics.
        """
        ok, _diags, raw = self.explain_diagnose(
            sql,
            params,
            schema=schema,
            intent=intent,
        )
        return ok, raw

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """
        Run ``EXPLAIN`` against the live engine and return structured findings.

        Args:

            sql: SQL text to explain.

            params: Optional bind parameters by name.

            schema: Optional schema graph used for plan-shape diagnostics
            (e.g. seq scan over indexed predicate).

            intent: Optional runtime intent paired with *schema*; used by
            Databricks for partition-pruning checks.

        Returns:

            ``(ok, diagnostics, raw_explain_output)`` where ``ok`` is False only for
            hard EXPLAIN failures (parse error, unknown column, timeout); soft
            findings (zero estimate, missed pruning, seq scan) are reported as
            diagnostics with ``ok=True``.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def can_explain(self) -> bool:
        """
        Return True when ``explain_sql`` can run against a live or embedded engine.

        ``EXPLAIN`` is always attempted when a backend exists; it is permanently disabled for this dialect instance once a permission-denied error has been observed (see :meth:`_disable_explain_on_permission_denied`).
        """
        if self._explain_disabled:
            return False
        return getattr(self, "engine", None) is not None

    def quote_table_column(self, table: str, column: str) -> str:
        """
        Return a dialect-safe ``table.column`` reference for SQL emission.

        Args:

            table: Logical or physical table name (may include a catalog/schema prefix).

            column: Column name.

        Returns:

            Quoted identifier pair joined by a dot.
        """

        return f"{table}.{column}"

    def _disable_explain_on_permission_denied(self, error_message: str) -> bool:
        """
        Flip ``_explain_disabled`` when *error_message* indicates a credentials issue.

        Returns True when the error was classified as permission denied (and EXPLAIN has been disabled for this dialect instance), otherwise False.
        """

        if _is_permission_denied_error(error_message):
            if not self._explain_disabled:
                debug(
                    f"[dialect.explain_sql] permission denied ({error_message!r}); "
                    f"disabling EXPLAIN for this dialect instance"
                )
            self._explain_disabled = True
            return True
        return False

    def prepare_for_execution(self, sql: str) -> str:
        """
        Return SQL in the form required for execution (identity by default).

        Args:

            sql: SQL text before any engine-specific rewriting.

        Returns:

            SQL string unchanged for generic dialects.
        """
        return sql

    def finalize_render(
        self,
        sql_param: str,
        params: dict[str, Any],
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
        execution_sql_override: str | None = None,
        structural_defaults: dict[str, Any] | None = None,
    ) -> str:
        """
        Produce executable SQL: dialect qualification, structural reduction, parameter substitution, and AST simplification.

        Args:

            sql_param: Parameterized SQL with ``:pN`` / ``:sN`` placeholders.

            params: Resolved parameter values.

            schema: Optional schema graph (used by dialects that inject execution hints).

            intent: Optional runtime intent (required for dialect-specific hints such as partition pruning).

            execution_sql_override: Optional pre-transformed SQL instead of ``prepare_for_execution(sql_param)``.

            structural_defaults: Template structural defaults for ``:sN`` inlining.

        Returns:

            Executable SQL string.
        """
        sql_in_raw = execution_sql_override or sql_param
        prepared = execution_sql_override or self.prepare_for_execution(sql_param)
        _trace_finalize_render_stage("prepare_for_execution", sql_in_raw, prepared)
        result = finalize_executable_sql(
            prepared,
            params,
            structural_defaults,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        _trace_finalize_render_stage("finalize_executable_sql", prepared, result)
        non_empty_in = (execution_sql_override or sql_param or "").strip()
        if non_empty_in and not result.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=finalize_executable_sql"
            )
        return result

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple]:
        """
        Execute SQL and return rows as tuples.

        Args:

            sql: SQL text to run.

            params: Optional bind parameters for engines that support them.

        Returns:

            Result rows.

        Raises:

            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def quote_identifier(self, ident: str) -> str:
        """
        Quote a single SQL identifier using ANSI double quotes.

        Args:

            ident: Bare identifier without surrounding quotes.

        Returns:

            Quoted identifier safe for reserved words and mixed case.
        """

        s = str(ident).strip()
        esc = s.replace('"', '""')
        return f'"{esc}"'

    def quote_schema_qualified(self, name: str) -> str:
        """
        Quote a dotted identifier path as one quoted fragment per segment.

        Args:

            name: Table or relation name, optionally ``schema.relation`` or deeper.

        Returns:

            Dotted sequence of ``quote_identifier`` results.
        """

        parts = [p for p in str(name).strip().split(".") if p]
        if not parts:
            return self.quote_identifier(name)
        return ".".join(self.quote_identifier(p) for p in parts)

    def quote_string_literal(self, text: str) -> str:
        """
        Render a string value as a single-quoted SQL string literal.

        Args:

            text: Unquoted string content.

        Returns:

            SQL literal with standard single-quote escaping.
        """

        s = str(text)
        esc = s.replace("'", "''")
        return f"'{esc}'"

    def render_date_diff(
        self,
        left_expr: str,
        op: str,
        unit: str,
        amount: int,
        *,
        minuend_sql: str = "",
        subtrahend_sql: str = "",
    ) -> str:
        """
        Render a date-difference comparison predicate.

        Args:

            left_expr: Left-hand date or interval expression SQL.

            op: Comparison operator.

            unit: Calendar unit name.

            amount: Interval magnitude.

            minuend_sql: First date column SQL (the one being subtracted from).

            subtrahend_sql: Second date column SQL (the one subtracted).

        Returns:

            SQL predicate string.
        """
        scaled, plural_unit = _format_interval_unit(unit, amount)
        return f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"

    def render_array_contains(self, column_sql: str, param_key: str) -> str:
        """
        Render array membership (contains) for WHERE/HAVING.

        Args:

            column_sql: Array-typed column or expression SQL.

            param_key: Bind placeholder name without a leading colon.

        Returns:

            Boolean SQL fragment.
        """
        return f":{param_key} = ANY({column_sql})"

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """
        Render UNNEST or equivalent for a SELECT list item.

        Args:

            column_sql: Array column SQL.

            alias: Output alias.

        Returns:

            Select-list fragment with alias.
        """
        return f"UNNEST({column_sql}) AS {alias}"

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """
        Render a relative date-window boundary expression.

        Args:

            column: Column SQL to compare.

            op: Comparison operator.

            unit: Calendar or time unit.

            amount: Units relative to current date; zero truncates to period start.

        Returns:

            SQL suitable for WHERE.
        """
        if amount == 0:
            return f"{column} {op} DATE_TRUNC('{unit}', CURRENT_DATE)"
        scaled, plural_unit = _format_interval_unit(unit, amount)
        return f"{column} {op} CURRENT_DATE - INTERVAL '{scaled} {plural_unit}'"

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """
        Wrap an expression for case-insensitive string comparison.

        Args:

            expr: SQL expression to wrap.

        Returns:

            Wrapped expression SQL.
        """
        return f"LOWER({expr})"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
    ) -> SchemaGraph:
        """
        Build a schema graph from catalog or DDL fallback.

        Args:

            include: Reflect base tables, views, or both.

            allow_objects: When set, restrict catalog reflection to these relation names (case-insensitive).

        Returns:

            Populated ``SchemaGraph`` without LLM role hints.

        Raises:

            NotImplementedError: Unless implemented by a concrete dialect.
        """
        raise NotImplementedError

    def compute_ddl_probe(self, schema_context: SchemaContext) -> str:
        """
        Return a cheap deterministic fingerprint of the live DDL the cache should be valid against.

        Concrete dialects should run a single ``information_schema.columns`` query (or equivalent) scoped to the active schema/catalog and return a SHA-256 hex digest over the sorted ``(table, column, ordinal_position, data_type, is_nullable)`` rows. This probe is consulted by :func:`aetherdialect._schema.build_schema_graph` to short-circuit cache loads without re-reflecting or re-profiling the schema.

        The base implementation returns an empty string, which disables the fast path and forces the existing fingerprint-based cache validation. Returning ``""`` is also the contract for "probe not available at collection time" (e.g., transient DB error): callers must never propagate exceptions from this method.

        Args:

            schema_context: Active scope; concrete dialects may use ``include`` / ``allow_objects`` to narrow the query.

        Returns:

            Hex digest string, or ``""`` to disable the probe-based fast path.
        """
        _ = schema_context
        return ""

    def reflect_only(self, schema_context: SchemaContext) -> SchemaGraph:
        """
        Reflect a structural-only ``SchemaGraph`` honouring ``schema_context.include``.

        Used by the partial-rebuild diff path: only structural shape (tables, columns, FKs) is needed in order to compute a :class:`SchemaDiff`; profiling is run later, on the affected subset only.

        The default implementation delegates to :meth:`reflect_schema_graph` with the effective include kind. Dialects may override to skip work that is unnecessary for the diff (e.g., enum value enrichment).
        """
        include = _reflect_include_for_schema_build(schema_context)
        allow_obj = schema_context.allow_objects if schema_context.allow_objects else None
        return self.reflect_schema_graph(include=include, allow_objects=allow_obj)

    def profile_schema(self, sg: SchemaGraph) -> None:
        """
        Populate column statistics and physical metadata on *sg* in place.

        Args:

            sg: Schema graph to enrich.

        Raises:

            NotImplementedError: Unless implemented by a concrete dialect.
        """
        raise NotImplementedError

    def refresh_full_table_distinct_for_pk_inference(
        self,
        table_name: str,
        col_name: str,
        *,
        table_kind: Literal["table", "view"] = "table",
    ) -> tuple[int, int, float] | None:
        """
        When profiling used sampling, run a full-table ``COUNT(*)``, ``COUNT(DISTINCT col)``, and null ratio.

        Returns ``(distinct_count, row_count, null_ratio)`` or ``None`` when unsupported or on failure.
        """
        return None

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """
        Build a ``FROM``-clause sampling suffix for row-count statistics queries.

        Args:

            use_sample: Whether sampling is active for this column.

            row_count: Total rows in the table.

            sample_size: Target sample row count.

            random_seed: Repeatable seed when the engine supports it.

            table_kind: Physical table or non-materialized view.

        Returns:

            Suffix SQL (may be empty).
        """
        if not use_sample:
            return ""
        return f"LIMIT {sample_size}"

    def profiling_stats_use_subquery_when_sampling(
        self,
        table_kind: Literal["table", "view"] = "table",
    ) -> bool:
        """
        Return True when distinct/null stats must scan a sampled subquery.

        Args:

            table_kind: Physical table or non-materialized view.

        Returns:

            Whether to wrap ``SELECT col FROM table …`` for statistics.
        """
        return True


_PG_STRUCTURAL_CODE_TO_DIAG: dict[str, SqlDiagnosticCode] = {
    "ast_parse_failed": SqlDiagnosticCode.AST_PARSE_FAILED,
    "multiple_statements": SqlDiagnosticCode.MULTIPLE_STATEMENTS,
    "no_root": SqlDiagnosticCode.NO_ROOT,
    "not_select": SqlDiagnosticCode.NOT_SELECT,
    "subquery_not_allowed": SqlDiagnosticCode.SUBQUERY_NOT_ALLOWED,
    "using_not_allowed": SqlDiagnosticCode.USING_NOT_ALLOWED,
    "cross_join_not_allowed": SqlDiagnosticCode.CROSS_JOIN_NOT_ALLOWED,
    "self_join_not_allowed": SqlDiagnosticCode.SELF_JOIN_NOT_ALLOWED,
    "exists_not_allowed": SqlDiagnosticCode.EXISTS_NOT_ALLOWED,
    "lateral_not_allowed": SqlDiagnosticCode.LATERAL_NOT_ALLOWED,
    "forbidden_structure": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
    "cte_recursive": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
    "cte_malformed": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
    "cte_contains_subquery": SqlDiagnosticCode.SUBQUERY_NOT_ALLOWED,
    "cte_contains_exists": SqlDiagnosticCode.EXISTS_NOT_ALLOWED,
    "cte_contains_set_op": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
}


def _pg_walk_nodes(root: Any) -> Any:
    """Yield every pglast AST node reachable from *root*."""
    try:
        from pglast.ast import Node
    except ImportError:
        Node = ()

    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        yield node
        if isinstance(node, Node):
            for attr in node:
                if attr == "ancestors":
                    continue
                val = getattr(node, attr, None)
                if val is None:
                    continue
                if isinstance(val, list | tuple):
                    for item in val:
                        if isinstance(item, Node):
                            stack.append(item)
                elif isinstance(val, Node):
                    stack.append(val)
            continue
        try:
            attrs = vars(node)
        except TypeError:
            continue
        for value in attrs.values():
            if isinstance(value, list | tuple):
                for item in value:
                    if hasattr(item, "__class__") and not isinstance(item, str | int | float | bool | bytes):
                        stack.append(item)
            elif hasattr(value, "__class__") and not isinstance(value, str | int | float | bool | bytes):
                stack.append(value)


def _pg_node_kind(node: Any) -> str:
    """Return the pglast AST node class name (e.g. ``ColumnRef``, ``RangeVar``)."""
    return getattr(node, "__class__", type("x", (), {})).__name__


def _pg_columnref_to_pair(node: Any) -> tuple[str | None, str] | None:
    """Convert a pglast ``ColumnRef`` to ``(prefix, column)`` or ``None`` for ``*`` / unsupported shapes."""
    fields = getattr(node, "fields", None) or ()
    parts: list[str] = []
    for fld in fields:
        kind = _pg_node_kind(fld)
        if kind == "String":
            sval = getattr(fld, "sval", None) or getattr(fld, "str", None)
            if isinstance(sval, str):
                parts.append(sval)
                continue
            return None
        if kind == "A_Star":
            return None
        return None
    if not parts:
        return None
    if len(parts) == 1:
        return None, parts[0]
    return parts[-2], parts[-1]


def _pg_funcname(node: Any) -> str:
    """Return the lowercased function name of a pglast ``FuncCall``."""
    fn = getattr(node, "funcname", None) or ()
    parts: list[str] = []
    for f in fn:
        sval = getattr(f, "sval", None) or getattr(f, "str", None)
        if isinstance(sval, str):
            parts.append(sval)
    return ".".join(parts).lower() if parts else ""


class PostgresDialect(Dialect):
    """PostgreSQL implementation using pglast and SQLAlchemy."""

    name: str = "postgresql"
    sqlglot_dialect: ClassVar[str] = "postgres"

    def quote_table_column(self, table: str, column: str) -> str:
        """Double-quote PostgreSQL identifiers for ``table.column`` emission."""

        def q(x: str) -> str:
            return '"' + str(x).replace('"', '""') + '"'

        return f"{q(table)}.{q(column)}"

    def __init__(self, config, sqlalchemy_engine: Any | None = None):
        """
        Create a SQLAlchemy engine from `PostgresRuntimeConfig`.

        Args:

            config: PostgreSQL runtime configuration.
            sqlalchemy_engine: Optional SQLAlchemy engine supplied by the integrator.

        Returns:

            None.
        """
        try:
            _require_pglast()
        except ImportError as e:
            raise ImportError(
                "PostgresDialect requires the 'pglast' package. Install with: pip install aetherdialect[postgresql]"
            ) from e
        super().__init__(config)
        if sqlalchemy_engine is not None:
            self.engine = sqlalchemy_engine
        else:
            self.engine = create_engine(config.db_url(), future=True)

    def _strip_schema(self, ident: str) -> str:
        """
        Strip schema prefix from an identifier and return a lowercase table name.

        Args:

            ident: Possibly qualified identifier string.

        Returns:

            Final dot segment, lowercased.
        """
        s = (ident or "").strip().lower()
        if "." in s:
            s = s.split(".")[-1]
        return s

    def _collect_from_items(
        self,
        fr: Any,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> tuple[bool, dict[str, str], bool, bool, bool, bool, bool]:
        """
        Collect FROM-clause aliases and flags for unsupported join shapes.

        Args:

            fr: FROM clause node or list of FROM items from the AST.

        Returns:

            `(ok, alias_to_table, has_subquery, has_using, has_cross_join, has_self_join, has_items)`; `ok` is False on unsupported nodes.
        """
        alias_to_table: dict[str, str] = {}
        has_subquery = False
        has_using = False
        has_cross_join = False
        has_self_join = False
        seen_tables: set[str] = set()
        ok = True

        def add_alias(relname: str, alias: Any) -> None:
            nonlocal alias_to_table, has_self_join, seen_tables
            t = self._strip_schema(relname)
            if t in seen_tables:
                has_self_join = True
            seen_tables.add(t)
            if alias is None:
                alias_to_table[t] = t
                return
            an = getattr(alias, "aliasname", None)
            if isinstance(an, str) and an:
                alias_to_table[self._strip_schema(an)] = t
            alias_to_table[t] = t

        def walk(item: Any) -> bool:
            nonlocal has_subquery, has_using, has_cross_join, ok
            if item is None:
                ok = False
                return False
            tag = getattr(item, "__class__", type("x", (), {})).__name__
            if tag == "RangeVar":
                add_alias(getattr(item, "relname", "") or "", getattr(item, "alias", None))
                return True
            if tag == "JoinExpr":
                if getattr(item, "usingClause", None) is not None or getattr(item, "isNatural", False):
                    has_using = True
                join_type = getattr(item, "jointype", None)
                if join_type is not None and str(join_type) == "JoinType.JOIN_INNER":
                    quals = getattr(item, "quals", None)
                    if quals is None:
                        allow = False
                        if scalar_cte_names:
                            rarg = getattr(item, "rarg", None)
                            rtag = getattr(rarg, "__class__", type("x", (), {})).__name__ if rarg is not None else ""
                            if rtag == "RangeVar" and rarg is not None:
                                reln = (getattr(rarg, "relname", "") or "").lower()
                                if reln and reln in scalar_cte_names:
                                    allow = True
                        if not allow:
                            has_cross_join = True
                if not walk(getattr(item, "larg", None)):
                    return False
                if not walk(getattr(item, "rarg", None)):
                    return False
                return True
            if tag in {
                "RangeSubselect",
                "RangeFunction",
                "RangeTableFunc",
                "RangeTableSample",
            }:
                has_subquery = True
                ok = False
                return False
            ok = False
            return False

        if fr is None:
            return False, {}, False, False, False, False, False
        for it in fr if isinstance(fr, list | tuple) else [fr]:
            if not walk(it):
                ok = False
                break
            return (
                ok,
                alias_to_table,
                has_subquery,
                has_using,
                has_cross_join,
                has_self_join,
                True,
            )
        return (
            ok,
            alias_to_table,
            has_subquery,
            has_using,
            has_cross_join,
            has_self_join,
            True,
        )

    def _validate_cte_bodies(self, with_clause: Any) -> tuple[bool, str]:
        """
        Validate CTE bodies against structural restrictions.

        Forbids recursive CTEs, subqueries, EXISTS sublinks, and set operations inside any CTE body. Window functions and ``CASE`` expressions are allowed.

        Args:

            with_clause: AST `WithClause` node or None.

        Returns:

            `(ok, error_code)`; `error_code` is empty on success.
        """
        if with_clause is None:
            return True, ""

        if getattr(with_clause, "recursive", False):
            return False, "cte_recursive"

        ctes = getattr(with_clause, "ctes", [])
        if not ctes:
            return True, ""

        for cte in ctes:
            cte_query = getattr(cte, "ctequery", None)
            if cte_query is None:
                return False, "cte_malformed"

            def walk_cte(n: Any) -> str | None:
                tag = getattr(n, "__class__", type("x", (), {})).__name__
                if tag in {"RangeSubselect", "SubLink"}:
                    if tag == "SubLink":
                        sublink_type = getattr(n, "subLinkType", None)
                        if sublink_type is not None and sublink_type == 0:
                            return "cte_contains_exists"
                    return "cte_contains_subquery"
                if tag == "SetOperationStmt":
                    return "cte_contains_set_op"
                try:
                    attrs = vars(n)
                except TypeError:
                    return None
                for attr in attrs.values():
                    if isinstance(attr, list):
                        for x in attr:
                            if hasattr(x, "__class__"):
                                err = walk_cte(x)
                                if err:
                                    return err
                    elif hasattr(attr, "__class__"):
                        err = walk_cte(attr)
                        if err:
                            return err
                return None

            err = walk_cte(cte_query)
            if err:
                return False, err

        return True, ""

    def _ast_structural_valid(
        self,
        sql: str,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Validate SQL structure using the pglast AST.

        Checks that the SQL is a single SELECT statement free of subqueries in ``FROM``, CROSS JOINs, self-joins, USING clauses, EXISTS sublinks, LATERAL, and set operations. Window functions and ``CASE`` expressions are allowed. Also validates any CTE bodies with the same rules.

        Args:

            sql: SQL text to validate.

        Returns:

            `(ok, error_code)`; empty code on success.
        """
        try:
            p = _require_pglast()
            stmts = p.parse_sql(canonicalize_sql(sql))
        except Exception:
            return False, "ast_parse_failed"

        if not stmts or len(stmts) != 1:
            return False, "multiple_statements"

        root = getattr(stmts[0], "stmt", None)
        if root is None:
            return False, "no_root"

        if getattr(root, "__class__", type("x", (), {})).__name__ != "SelectStmt":
            return False, "not_select"

        with_clause = getattr(root, "withClause", None)
        has_cte = with_clause is not None

        if has_cte:
            ok, err = self._validate_cte_bodies(with_clause)
            if not ok:
                return False, err

        fr = getattr(root, "fromClause", None)
        if fr is not None:
            _, _, has_subq, has_using, has_cross, has_self, _ = self._collect_from_items(
                fr,
                scalar_cte_names,
            )
            if has_subq:
                return False, "subquery_not_allowed"
            if has_using:
                return False, "using_not_allowed"
            if has_cross:
                return False, "cross_join_not_allowed"
            if has_self:
                return False, "self_join_not_allowed"

        has_exists = False
        has_lateral = False

        def walk(n: Any) -> bool:
            nonlocal has_exists, has_lateral
            tag = getattr(n, "__class__", type("x", (), {})).__name__
            if tag in {
                "RangeSubselect",
                "SubLink",
                "SetOperationStmt",
            }:
                if tag == "SubLink":
                    sublink_type = getattr(n, "subLinkType", None)
                    if sublink_type is not None and sublink_type == 0:
                        has_exists = True
                return False
            if tag == "RangeFunction":
                is_lateral = getattr(n, "lateral", False)
                if is_lateral:
                    has_lateral = True
                    return False

            try:
                attrs = vars(n)
            except TypeError:
                return True

            for attr in attrs.values():
                if isinstance(attr, list):
                    for x in attr:
                        if hasattr(x, "__class__") and not walk(x):
                            return False
                elif hasattr(attr, "__class__"):
                    if not walk(attr):
                        return False
            return True

        if not walk(root):
            if has_exists:
                return False, "exists_not_allowed"
            if has_lateral:
                return False, "lateral_not_allowed"
            return False, "forbidden_structure"

        return True, ""

    def ast_validate_full(
        self,
        sql: str,
        *,
        schema: SchemaGraph | None = None,
        declared_params: set[str] | None = None,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> list[SqlDiagnostic]:
        """
        Validate SQL via pglast structurally and (when *schema* is given) semantically.

        Args:

            sql: SQL text.

            schema: Optional schema graph; enables column/table existence and
            ambiguity checks via :func:`_check_schema_references_shared`.

            declared_params: Optional declared placeholder names; missing
            placeholders emit :attr:`SqlDiagnosticCode.PARAM_UNBOUND`.

            scalar_cte_names: Lowercased scalar-emission CTE names allowed for
            comma-style joins without ``ON``.

        Returns:

            List of :class:`SqlDiagnostic` findings; empty list means valid.
        """
        ok, code = self._ast_structural_valid(sql, scalar_cte_names=scalar_cte_names)
        if not ok:
            mapped = _PG_STRUCTURAL_CODE_TO_DIAG.get(code, SqlDiagnosticCode.FORBIDDEN_STRUCTURE)
            return [SqlDiagnostic(code=mapped, message=code, node_kind=None)]
        diags: list[SqlDiagnostic] = []
        try:
            p = _require_pglast()
            stmts = p.parse_sql(canonicalize_sql(sql))
        except Exception:
            return [SqlDiagnostic(code=SqlDiagnosticCode.AST_PARSE_FAILED, message="parse failed")]
        if not stmts:
            return diags
        root = getattr(stmts[0], "stmt", None)
        if root is None:
            return diags
        cte_names = self._pg_collect_cte_names(root)
        alias_to_table = self._pg_collect_table_aliases(root)
        if schema is not None:
            refs = self._pg_collect_column_refs(root)
            diags += _check_schema_references_shared(refs, alias_to_table, cte_names, schema)
        diags += self._pg_check_grouping(root)
        diags += self._pg_check_cte_closure(root, cte_names)
        if declared_params is not None:
            diags += self._pg_check_param_coverage(sql, declared_params)
        return diags

    def _pg_collect_cte_names(self, root: Any) -> set[str]:
        """Return the set of lowercased CTE names defined on *root*'s ``WITH`` clause."""
        names: set[str] = set()
        with_clause = getattr(root, "withClause", None)
        if with_clause is None:
            return names
        for cte in getattr(with_clause, "ctes", None) or ():
            ctename = getattr(cte, "ctename", None)
            if isinstance(ctename, str) and ctename:
                names.add(ctename.lower())
        return names

    def _pg_collect_table_aliases(self, root: Any) -> dict[str, str]:
        """Return ``{alias_or_table_lc: real_table_lc}`` for every ``RangeVar`` reachable from *root*."""
        out: dict[str, str] = {}
        for node in _pg_walk_nodes(root):
            if _pg_node_kind(node) != "RangeVar":
                continue
            relname = getattr(node, "relname", None) or ""
            if not relname:
                continue
            real = relname.lower()
            out[real] = real
            alias = getattr(node, "alias", None)
            if alias is not None:
                aliasname = getattr(alias, "aliasname", None)
                if isinstance(aliasname, str) and aliasname:
                    out[aliasname.lower()] = real
        return out

    def _pg_collect_column_refs(self, root: Any) -> list[tuple[str | None, str]]:
        """Return ``(prefix, column)`` pairs for every ``ColumnRef`` reachable from *root*."""
        out: list[tuple[str | None, str]] = []
        for node in _pg_walk_nodes(root):
            if _pg_node_kind(node) != "ColumnRef":
                continue
            pair = _pg_columnref_to_pair(node)
            if pair is not None:
                out.append(pair)
        return out

    def _pg_check_grouping(self, root: Any) -> list[SqlDiagnostic]:
        """
        Emit grain diagnostics for *root*: aggregates in WHERE and HAVING-without-GROUP-BY.

        The non-grouped-select-col check is intentionally omitted because the renderer's own grain enforcement is more accurate than a string-level reconstruction here.
        """
        diags: list[SqlDiagnostic] = []
        where = getattr(root, "whereClause", None)
        if where is not None:
            for node in _pg_walk_nodes(where):
                if _pg_node_kind(node) == "FuncCall":
                    name = _pg_funcname(node)
                    if name in PG_AGG_FUNCNAMES:
                        diags.append(
                            SqlDiagnostic(
                                code=SqlDiagnosticCode.AGG_IN_WHERE,
                                message=f"aggregate {name!r} in WHERE",
                                node_kind="FuncCall",
                                offending_identifier=name,
                            )
                        )
                        break
        having = getattr(root, "havingClause", None)
        group = getattr(root, "groupClause", None) or ()
        if having is not None and not group:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.HAVING_WITHOUT_GROUP,
                    message="HAVING without GROUP BY",
                    node_kind="SelectStmt",
                )
            )
        return diags

    def _pg_check_cte_closure(self, root: Any, cte_names: set[str]) -> list[SqlDiagnostic]:
        """Flag CTE names that are defined but never referenced by a ``RangeVar`` outside their own definition."""
        if not cte_names:
            return []
        referenced: set[str] = set()
        with_clause = getattr(root, "withClause", None)
        defining_queries: set[int] = set()
        if with_clause is not None:
            for cte in getattr(with_clause, "ctes", None) or ():
                inner = getattr(cte, "ctequery", None)
                if inner is not None:
                    defining_queries.add(id(inner))
        for node in _pg_walk_nodes(root):
            if _pg_node_kind(node) != "RangeVar":
                continue
            relname = getattr(node, "relname", None)
            if isinstance(relname, str) and relname.lower() in cte_names:
                referenced.add(relname.lower())
        unreferenced = sorted(cte_names - referenced)
        return [
            SqlDiagnostic(
                code=SqlDiagnosticCode.CTE_UNREFERENCED,
                message=f"CTE {n!r} is defined but never referenced",
                node_kind="CommonTableExpr",
                offending_identifier=n,
            )
            for n in unreferenced
        ]

    def _pg_check_param_coverage(self, sql: str, declared: set[str]) -> list[SqlDiagnostic]:
        """Emit a diagnostic for each ``:name`` placeholder in *sql* not present in *declared*."""
        diags: list[SqlDiagnostic] = []
        seen: set[str] = set()
        for match in NAMED_PLACEHOLDER_RE.finditer(sql):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            if name not in declared:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.PARAM_UNBOUND,
                        message=f"unbound placeholder :{name}",
                        node_kind="ParamRef",
                        offending_identifier=name,
                    )
                )
        return diags

    def parse_select(self, sql: str) -> _PgParsedSelect | None:
        """
        Parse *sql* with pglast after encoding ``:name`` placeholders as ``$N``.

        Single-line ``--`` comments are stripped before whitespace canonicalization so they do not absorb subsequent clauses when newlines collapse.

        Returns ``None`` for non-``SELECT`` roots, multi-statement input, or parse failure.
        """

        decommented = re.sub(r"--[^\n]*", "", sql)
        encoded, name_to_index, index_to_name = _pg_encode_named_placeholders(canonicalize_sql(decommented))
        try:
            p = _require_pglast()
            stmts = p.parse_sql(encoded)
        except Exception:
            return None
        if not stmts or len(stmts) != 1:
            return None
        root = getattr(stmts[0], "stmt", None)
        if root is None or type(root).__name__ != "SelectStmt":
            return None
        return _PgParsedSelect(stmts[0], name_to_index, index_to_name)

    def ordered_join_carrier_froms(self, parsed: _PgParsedSelect) -> list[Any] | None:
        """
        Return the inner-CTE ``SelectStmt`` nodes (left-to-right) followed by the outer ``SelectStmt``.

        Each handle is a ``SelectStmt`` whose ``fromClause`` is rewritten by :meth:`attach_joins`.
        """

        root = getattr(parsed.root, "stmt", None)
        if root is None or type(root).__name__ != "SelectStmt":
            return None
        carriers: list[Any] = []
        with_clause = getattr(root, "withClause", None)
        if with_clause is not None:
            ctes = getattr(with_clause, "ctes", None) or ()
            for cte in ctes:
                inner = getattr(cte, "ctequery", None)
                if inner is not None and type(inner).__name__ == "SelectStmt":
                    if getattr(inner, "fromClause", None) is not None:
                        carriers.append(inner)
        if getattr(root, "fromClause", None) is not None:
            carriers.append(root)
        return carriers

    def from_anchor_of(self, carrier: Any) -> str | None:
        """
        Read the bare table name of *carrier*'s leftmost ``FROM`` leaf.

        When the first ``FROM`` element is a ``JoinExpr`` tree (for example after ``CROSS JOIN`` attachment),
        this walks ``larg`` until a ``RangeVar`` is reached. Returns ``None`` for subqueries or empty ``FROM``.
        """

        from_clause = getattr(carrier, "fromClause", None) or ()
        if len(from_clause) != 1:
            return None
        first = from_clause[0]
        while type(first).__name__ == "JoinExpr":
            first = getattr(first, "larg", None)
            if first is None:
                return None
        if type(first).__name__ != "RangeVar":
            return None
        relname = getattr(first, "relname", None)
        if not relname:
            return None
        return str(relname).lower()

    def attach_joins(
        self,
        parsed: _PgParsedSelect,
        from_handle: Any,
        edges: list[JoinEdge],
    ) -> bool:
        """
        Build a left-deep tree of pglast ``JoinExpr`` nodes from *edges* and replace
        *from_handle*'s ``fromClause`` with the resulting single-element list.
        """

        if not edges:
            return False
        from_clause = getattr(from_handle, "fromClause", None) or ()
        if len(from_clause) != 1:
            return False
        p = _require_pglast()
        current: Any = from_clause[0]
        for edge in edges:
            quals = self._pg_build_on_quals(edge.on_terms)
            if quals is None:
                return False
            rarg = p.ast.RangeVar(
                relname=edge.table,
                inh=True,
                relpersistence="p",
            )
            if edge.alias:
                rarg.alias = p.ast.Alias(aliasname=edge.alias)
            jt = p.join_type.JOIN_INNER if edge.kind == "INNER" else p.join_type.JOIN_LEFT
            current = p.ast.JoinExpr(
                jointype=jt,
                isNatural=False,
                larg=current,
                rarg=rarg,
                quals=quals,
            )
        try:
            from_handle.fromClause = (current,)
        except Exception:
            return False
        return True

    @staticmethod
    def _pg_build_on_quals(
        on_terms: tuple[tuple[str, str, str, str], ...],
    ) -> Any | None:
        """Return a single ``A_Expr`` or an ``AND``-joined ``BoolExpr`` over equality conjuncts."""

        if not on_terms:
            return None
        p = _require_pglast()
        eqs: list[Any] = []
        for left_token, left_col, right_token, right_col in on_terms:
            lhs = p.ast.ColumnRef(
                fields=(p.ast.String(sval=left_token), p.ast.String(sval=left_col)),
            )
            rhs = p.ast.ColumnRef(
                fields=(p.ast.String(sval=right_token), p.ast.String(sval=right_col)),
            )
            eqs.append(
                p.ast.A_Expr(
                    kind=p.a_expr_kind.AEXPR_OP,
                    name=(p.ast.String(sval="="),),
                    lexpr=lhs,
                    rexpr=rhs,
                ),
            )
        if len(eqs) == 1:
            return eqs[0]
        return p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=tuple(eqs))

    def attach_extra_from_and_where(
        self,
        parsed: _PgParsedSelect,
        from_handle: Any,
        extra_from_tables: list[str],
        where_edges: list[JoinEdge],
    ) -> bool:
        """Append RangeVar entries to ``fromClause`` and AND equality predicates into ``whereClause``."""

        if not extra_from_tables and not where_edges:
            return True
        p = _require_pglast()
        existing_from = list(getattr(from_handle, "fromClause", None) or ())
        for tbl in extra_from_tables:
            existing_from.append(
                p.ast.RangeVar(relname=tbl, inh=True, relpersistence="p"),
            )
        try:
            from_handle.fromClause = tuple(existing_from)
        except Exception:
            return False
        if not where_edges:
            return True
        new_eqs: list[Any] = []
        for edge in where_edges:
            for left_token, left_col, right_token, right_col in edge.on_terms:
                lhs = p.ast.ColumnRef(
                    fields=(
                        p.ast.String(sval=left_token),
                        p.ast.String(sval=left_col),
                    ),
                )
                rhs = p.ast.ColumnRef(
                    fields=(
                        p.ast.String(sval=right_token),
                        p.ast.String(sval=right_col),
                    ),
                )
                new_eqs.append(
                    p.ast.A_Expr(
                        kind=p.a_expr_kind.AEXPR_OP,
                        name=(p.ast.String(sval="="),),
                        lexpr=lhs,
                        rexpr=rhs,
                    ),
                )
        if not new_eqs:
            return True
        if len(new_eqs) == 1:
            new_pred: Any = new_eqs[0]
        else:
            new_pred = p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=tuple(new_eqs))
        existing_where = getattr(from_handle, "whereClause", None)
        if existing_where is None:
            merged: Any = new_pred
        elif (
            type(existing_where).__name__ == "BoolExpr"
            and getattr(existing_where, "boolop", None) == p.bool_expr_type.AND_EXPR
        ):
            merged_args = tuple(getattr(existing_where, "args", ()) or ()) + tuple(new_eqs)
            merged = p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=merged_args)
        else:
            merged = p.ast.BoolExpr(
                boolop=p.bool_expr_type.AND_EXPR,
                args=((existing_where, new_pred) if len(new_eqs) == 1 else (existing_where, *new_eqs)),
            )
        try:
            from_handle.whereClause = merged
        except Exception:
            return False
        return True

    def replace_projection(
        self,
        parsed: _PgParsedSelect,
        items: list[tuple[str, str | None]],
    ) -> bool:
        """Replace the outer ``SelectStmt``'s ``targetList`` with ``ResTarget`` nodes parsed from *items*."""

        root = getattr(parsed.root, "stmt", None)
        if root is None or type(root).__name__ != "SelectStmt":
            return False
        p = _require_pglast()
        new_targets: list[Any] = []
        for expr_sql, alias in items:
            encoded, _, _ = _pg_encode_named_placeholders(expr_sql)
            try:
                probe = p.parse_sql(f"SELECT {encoded}")
            except Exception:
                return False
            if not probe:
                return False
            probe_select = getattr(probe[0], "stmt", None)
            if probe_select is None or type(probe_select).__name__ != "SelectStmt":
                return False
            tlist = getattr(probe_select, "targetList", None) or ()
            if len(tlist) != 1:
                return False
            value_node = tlist[0].val
            new_targets.append(p.ast.ResTarget(name=alias or None, val=value_node))
        try:
            root.targetList = tuple(new_targets)
        except Exception:
            return False
        return True

    def emit_sql(self, parsed: _PgParsedSelect) -> str:
        """Render *parsed* via pglast ``RawStream`` and decode ``$N`` back to ``:name``."""

        p = _require_pglast()
        rendered = p.raw_stream_cls()(parsed.root)
        return _pg_decode_dollar_placeholders(rendered, parsed.index_to_name)

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """
        Run PostgreSQL ``EXPLAIN (FORMAT JSON, COSTS true)`` and return ``(ok, diagnostics, raw_message)``.

        ``ok`` is False only on hard validation failures (parse errors, unknown
        identifiers, timeouts). Permission-denied disables EXPLAIN for the
        remainder of this dialect instance and is reported as ``ok=True`` with
        no diagnostics so the caller can proceed without treating missing
        privileges as invalid SQL. Soft plan-shape findings (suspected
        cartesian joins, zero-row estimates, sequential scans on indexed
        columns) are emitted as :class:`SqlDiagnostic` entries with codes from
        ``SOFT_DIAGNOSTIC_CODES`` in ``_config`` so callers may apply confidence
        penalties without rejecting the SQL.

        Args:

            sql: SQL text.

            params: Optional bind parameters.

            schema: When provided, enables :data:`SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED` detection by checking primary/foreign key columns of the scanned relation.

            intent: Unused on PostgreSQL; accepted for a uniform signature.

        Returns:

            ``(ok, diagnostics, raw_message)``.
        """
        finalized = self.finalize_render(
            sql,
            params or {},
            schema=schema,
            intent=intent,
        )
        explain_sql = f"EXPLAIN (FORMAT JSON, COSTS true) {finalized}"
        try:
            tm = effective_explain_timeout_ms()
            if tm is not None:
                ms = int(tm)
                with self.engine.begin() as conn:
                    conn.execute(text(f"SET LOCAL statement_timeout = {ms}"))
                    rows = conn.execute(text(explain_sql), params or {}).fetchall()
            else:
                with self.engine.connect() as conn:
                    rows = conn.execute(text(explain_sql), params or {}).fetchall()
            payload: Any = None
            if rows:
                first_row = rows[0]
                payload = first_row[0] if len(first_row) > 0 else None
            pay = payload
            if isinstance(pay, str):
                try:
                    pay = json.loads(pay)
                except (ValueError, TypeError):
                    pay = None
            est_rows: float | None = None
            est_bytes: float | None = None
            if isinstance(pay, list) and pay and isinstance(pay[0], dict):
                rp = pay[0].get("Plan")
                if isinstance(rp, dict):
                    est_rows, est_bytes = _pg_root_plan_estimates(rp)
            failed, why = _explain_cost_gate_violation(est_rows, est_bytes)
            if failed:
                return (
                    False,
                    [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                    why,
                )
            soft_diags = _pg_diagnostics_from_explain_json(payload, schema)
            return True, soft_diags, ""
        except Exception as e:
            err = str(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (
                False,
                [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)],
                err,
            )

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple]:
        """
        Execute SQL via SQLAlchemy and return row tuples.

        Args:

            sql: SQL text.

            params: Optional bind parameters.

        Returns:

            Result rows as tuples.
        """
        try:
            tm = PolicyConfig.STATEMENT_TIMEOUT_MS
            if cost_cap_active(tm):
                ms = int(tm)
                with self.engine.begin() as conn:
                    conn.execute(text(f"SET LOCAL statement_timeout = {ms}"))
                    rows = conn.execute(text(sql), params or {}).fetchall()
            else:
                with self.engine.connect() as conn:
                    rows = conn.execute(text(sql), params or {}).fetchall()
            return [tuple(r) for r in rows]
        except Exception as e:
            err = str(e)
            if _is_permission_denied_error(err):
                raise AccessError("execute", err) from e
            el = err.lower()
            if "statement timeout" in el or "timeout expired" in el or "query canceled" in el:
                raise StatementTimeoutError(err) from e
            raise

    def render_date_diff(
        self,
        left_expr: str,
        op: str,
        unit: str,
        amount: int,
        *,
        minuend_sql: str = "",
        subtrahend_sql: str = "",
    ) -> str:
        """
        Render PostgreSQL interval date-difference comparison.

        Args:

            left_expr: Left-hand expression SQL.

            op: Comparison operator.

            unit: ``day``, ``week``, ``month``, or ``year``.

            amount: Interval magnitude.

            minuend_sql: Unused on PostgreSQL (interval subtraction works natively).

            subtrahend_sql: Unused on PostgreSQL.

        Returns:

            Predicate SQL with ``INTERVAL``.
        """
        scaled, plural_unit = _format_interval_unit(unit, amount)
        sql = f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"
        return _emit_via_ast(sql, "postgres")

    def render_array_contains(self, column_sql: str, param_key: str) -> str:
        """
        Render PostgreSQL array membership as a single ``ANY``-comparison predicate.

        Avoids ``EXISTS`` / subquery / ``ARRAY[`` constructs so the fragment passes
        ``_enforce_select_only`` and ``_ast_structural_valid``. Lowercases both sides
        and trims surrounding whitespace and quote characters from the bound value
        for case-insensitive, quote-tolerant matching against ``text[]`` columns.

        Args:

            column_sql: Array column SQL.

            param_key: Bind parameter name without colon.

        Returns:

            Boolean SQL predicate of the form ``<param> = ANY(<lowered-array>)``.
        """
        delimiter = "CHR(31)"
        lowered_elements = f"string_to_array(LOWER(array_to_string({column_sql}, {delimiter})), {delimiter})"
        norm_param = f"LOWER(BTRIM(CAST(:{param_key} AS TEXT), ' ' || CHR(34) || CHR(39)))"
        sql = f"{norm_param} = ANY({lowered_elements})"
        return _emit_via_ast(sql, "postgres")

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """
        Render PostgreSQL ``UNNEST`` for SELECT list.

        Args:

            column_sql: Array column SQL.

            alias: Output alias.

        Returns:

            UNNEST fragment with alias.
        """
        sql = f"UNNEST({column_sql}) AS {alias}"
        return _emit_via_ast(sql, "postgres")

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """
        Render PostgreSQL date window boundaries.

        Args:

            column: Column SQL.

            op: Comparison operator.

            unit: Calendar unit.

            amount: Distance from current date.

        Returns:

            WHERE fragment SQL.
        """
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', CURRENT_DATE)"
        else:
            scaled, plural_unit = _format_interval_unit(unit, amount)
            sql = f"{column} {op} CURRENT_DATE - INTERVAL '{scaled} {plural_unit}'"
        return _emit_via_ast(sql, "postgres")

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
    ) -> SchemaGraph:
        """
        Reflect PostgreSQL metadata or parse ``SQL_FILE_PATH`` DDL.

        Args:

            include: Reflect base tables, views, or both.

            allow_objects: When set, restrict reflection to these relation names (case-insensitive).

        Returns:

            ``SchemaGraph`` from the database or file fallback.
        """
        return load_or_create_schema_postgresql(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
        )

    def compute_ddl_probe(self, schema_context: SchemaContext) -> str:
        """
        Return SHA-256 over ``information_schema.columns`` rows for the configured PostgreSQL schema.

        Always returns ``""`` instead of raising on connection / permission / query errors so the caller falls back to the existing fingerprint validation path.
        """
        _ = schema_context
        try:
            schema_name = str(self.config.SCHEMA or "public")
            cols_sql = (
                "SELECT table_schema, table_name, column_name, ordinal_position, data_type, is_nullable "
                "FROM information_schema.columns WHERE table_schema = :s "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            unique_sql = (
                "SELECT kcu.table_schema, kcu.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_schema = kcu.constraint_schema "
                " AND tc.constraint_name = kcu.constraint_name "
                "WHERE tc.table_schema = :s AND tc.constraint_type = 'UNIQUE' "
                "ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name"
            )
            with self.engine.connect() as conn:
                rows = conn.execute(text(cols_sql), {"s": schema_name}).fetchall()
                uniq_rows = conn.execute(text(unique_sql), {"s": schema_name}).fetchall()
            payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
            payload_uniq = "\n".join("|".join("" if c is None else str(c) for c in r) for r in uniq_rows)
            return sha256(payload_cols + "\n##UNIQUE##\n" + payload_uniq)
        except Exception as exc:
            debug(f"[dialect.PostgresDialect.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""

    def profile_schema(self, sg: SchemaGraph) -> None:
        """
        Run SQLAlchemy-backed column profiling for PostgreSQL.

        Args:

            sg: Schema graph to update in place.
        """
        profile_schema(self.engine, sg, dialect=self)

    def refresh_full_table_distinct_for_pk_inference(
        self,
        table_name: str,
        col_name: str,
        *,
        table_kind: Literal["table", "view"] = "table",
    ) -> tuple[int, int, float] | None:
        """
        Run full-table statistics for PK inference after sampled profiling.

        Args:

            table_name: Reflected table name.

            col_name: Column name.

            table_kind: Physical table or view (reserved for dialect-specific sampling rules).

        Returns:

            ``(distinct_count, row_count, null_ratio)`` or ``None`` on failure.
        """
        try:
            _ = table_kind
            safe_tbl = str(table_name).replace('"', '""')
            safe_col = str(col_name).replace('"', '""')
            sql = text(
                f'SELECT COUNT(*) AS cnt, COUNT(DISTINCT "{safe_col}") AS dist, '
                f'COUNT(*) - COUNT("{safe_col}") AS nulls FROM "{safe_tbl}"',
            )
            with self.engine.connect() as conn:
                row = conn.execute(sql).fetchone()
            if not row:
                return None
            cnt = int(row[0] or 0)
            dist = int(row[1] or 0)
            nulls = int(row[2] or 0)
            nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
            return (dist, cnt, nr)
        except Exception as exc:
            debug(f"[dialect.PostgresDialect.refresh_full_table_distinct_for_pk_inference] failed: {exc!r}")
            return None

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """
        Return a ``TABLESAMPLE BERNOULLI`` suffix for PostgreSQL statistics.

        Args:

            use_sample: Whether sampling applies.

            row_count: Table row count for percentage calculation.

            sample_size: Target sample size.

            random_seed: ``REPEATABLE`` seed for stable samples.

            table_kind: Physical table or non-materialized view.

        Returns:

            Sampling suffix or empty string.
        """
        if table_kind == "view":
            return ""
        if not use_sample:
            return ""
        pct = 100 * sample_size / row_count if row_count else 0.0
        return f"TABLESAMPLE BERNOULLI ({pct:.2f}) REPEATABLE ({random_seed})"

    def profiling_stats_use_subquery_when_sampling(
        self,
        table_kind: Literal["table", "view"] = "table",
    ) -> bool:
        """
        PostgreSQL samples the base table directly with ``TABLESAMPLE``.

        Args:

            table_kind: Physical table or non-materialized view.

        Returns:

            False so statistics use a single-table ``FROM`` for physical tables; views use a subquery with ``LIMIT``.
        """
        return table_kind == "view"


def _unit_to_approx_days(unit: str, amount: int) -> int:
    """Convert a calendar unit and magnitude to an approximate day count."""
    return amount * UNIT_TO_DAYS.get(unit, 1)


def _spark_from_clause_root(sel: sqlglot.exp.Select) -> sqlglot.exp.Expression | None:
    from_ = sel.args.get("from_")
    if from_ is None:
        return None
    if isinstance(from_, sqlglot.exp.From):
        return from_.this
    return None


def _spark_walk_from_branches(expr: sqlglot.exp.Expression | None):
    if expr is None:
        return
    if isinstance(expr, sqlglot.exp.Join):
        yield from _spark_walk_from_branches(expr.this)
        yield from _spark_walk_from_branches(expr.expression)
        return
    yield expr


def _spark_tables_from_from_root(root: sqlglot.exp.Expression | None) -> list[str]:
    names: list[str] = []
    for node in _spark_walk_from_branches(root):
        if isinstance(node, sqlglot.exp.Table) and node.name:
            names.append(node.name.strip().lower())
    return names


def _spark_join_rhs_unwrapped(join: sqlglot.exp.Join) -> sqlglot.exp.Expression | None:
    raw = join.args.get("expression") or join.args.get("this")
    node = raw
    while isinstance(node, sqlglot.exp.Alias):
        node = node.this
    return node


def _spark_validate_select_structural_inner(
    select: sqlglot.exp.Select,
    scalar_cte_names: frozenset[str] | None = None,
) -> tuple[bool, str]:
    if list(select.find_all(sqlglot.exp.Exists)):
        return False, "exists_not_allowed"
    if list(select.find_all(sqlglot.exp.Lateral)):
        return False, "lateral_not_allowed"
    for join in select.find_all(sqlglot.exp.Join):
        if join.args.get("using"):
            return False, "using_not_allowed"
        kind = join.args.get("kind")
        if kind is not None and str(kind).upper() == "CROSS":
            allowed = False
            right = _spark_join_rhs_unwrapped(join)
            if scalar_cte_names and isinstance(right, sqlglot.exp.Table):
                rn = (right.name or "").strip().lower()
                if rn and rn in scalar_cte_names:
                    allowed = True
            if not allowed:
                return False, "cross_join_not_allowed"
    if list(select.find_all(sqlglot.exp.Subquery)):
        return False, "subquery_not_allowed"
    if list(select.find_all(sqlglot.exp.Union)):
        return False, "forbidden_structure"
    names = _spark_tables_from_from_root(_spark_from_clause_root(select))
    if len(names) >= 2 and len(names) != len(set(names)):
        return False, "self_join_not_allowed"
    return True, ""


def _spark_validate_with_ctes(with_clause: sqlglot.exp.With) -> tuple[bool, str]:
    if with_clause.args.get("recursive"):
        return False, "cte_recursive"
    for cte in with_clause.expressions:
        body = cte.this
        if isinstance(body, sqlglot.exp.Union):
            return False, "cte_contains_set_op"
        if isinstance(body, sqlglot.exp.Select):
            nested = body.args.get("with_")
            if nested is not None:
                okn, errn = _spark_validate_with_ctes(nested)
                if not okn:
                    return False, errn
            inner = body.copy()
            inner.set("with_", None)
            ok, err = _spark_validate_select_structural_inner(inner)
            if not ok:
                return False, err
    return True, ""


def _ast_spark_structural_valid_sqlglot(
    sql: str,
    scalar_cte_names: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """
    Structural policy for Spark SQL via sqlglot, aligned with Postgres ``_ast_structural_valid`` intent.

    Parity (same intent as ``PostgresDialect._ast_structural_valid``): single top-level ``SELECT``; no top-level ``UNION``; non-recursive ``WITH``; CTE bodies must not start with a set operation and must satisfy the same structural rules as a main body; main query must not use ``EXISTS``, ``LATERAL``, ``JOIN ... USING``, ``CROSS JOIN`` (including comma joins sqlglot normalizes to cross), derived tables in ``FROM``, ``UNION`` anywhere under the main ``SELECT``, or self-join on the same bare table name in ``FROM``.

    Gaps vs Postgres: no full ``SubLink`` walk; no ``SetOperationStmt`` outside sqlglot ``Union``; sqlglot may fold duplicate self-joins so ``self_join_not_allowed`` triggers less often than pglast; engine-specific syntax is deferred to ``explain_sql``.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="spark")
    except Exception:
        return False, "ast_parse_failed"
    if isinstance(tree, sqlglot.exp.Union):
        return False, "multiple_statements"
    if not isinstance(tree, sqlglot.exp.Select):
        return False, "not_select"
    select = tree
    wc = select.args.get("with_")
    if wc is not None:
        okw, errw = _spark_validate_with_ctes(wc)
        if not okw:
            return False, errw
    main = select.copy()
    main.set("with_", None)
    return _spark_validate_select_structural_inner(main, scalar_cte_names)


def _databricks_normalize_datetrunc_sql(sql: str) -> str:
    """
    Rewrite parsed ``Anonymous`` ``DATETRUNC`` call sites so emission matches Spark ``DATE_TRUNC`` ordering.

    Args:

        sql: Spark SQL text after placeholder finalisation.

    Returns:

        SQL with ``Anonymous`` ``DATETRUNC`` nodes rewritten as ``TimestampTrunc`` AST (renders as ``DATE_TRUNC``).
    """

    try:
        tree = sqlglot.parse_one(sql, dialect="spark")
    except Exception:
        debug(f"[_databricks_normalize_datetrunc_sql] sqlglot parse failed; preserving input SQL (len={len(sql)})")
        return sql
    exp = sqlglot.expressions
    for anon in list(tree.find_all(exp.Anonymous)):
        if str(anon.this).upper() != "DATETRUNC":
            continue
        parts = anon.expressions
        if len(parts) != 2:
            continue
        try:
            e0, e1 = parts[0], parts[1]
            lit0 = isinstance(e0, exp.Literal)
            lit1 = isinstance(e1, exp.Literal)
            if lit0 and not lit1:
                unit_sql = e0.sql(dialect="spark")
                expr_sql = e1.sql(dialect="spark")
            elif lit1 and not lit0:
                unit_sql = e1.sql(dialect="spark")
                expr_sql = e0.sql(dialect="spark")
            else:
                expr_sql = e0.sql(dialect="spark")
                unit_sql = e1.sql(dialect="spark")
            frag = f"SELECT DATE_TRUNC({unit_sql}, {expr_sql})"
            wrapped = sqlglot.parse_one(frag, dialect="spark")
            dtn = list(wrapped.find_all(exp.DateTrunc))
            if dtn:
                anon.replace(dtn[0])
                continue
            dtn = list(wrapped.find_all(exp.TimestampTrunc))
            if dtn:
                anon.replace(dtn[0])
        except Exception:
            continue
    try:
        out = tree.sql(dialect="spark")
        if sql.strip() and not out.strip():
            debug(
                f"[_databricks_normalize_datetrunc_sql] sqlglot emission empty; preserving input SQL (len={len(sql)})"
            )
            return sql
        return out
    except Exception:
        debug(f"[_databricks_normalize_datetrunc_sql] sqlglot serialize failed; preserving input SQL (len={len(sql)})")
        return sql


UNITY_INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL: str = (
    "SELECT constraint_catalog, constraint_schema, constraint_name, table_catalog, "
    "table_schema, table_name, constraint_type "
    "FROM `{catalog_esc}`.information_schema.table_constraints "
    "WHERE lower(table_schema) = lower('{schema_lit}') "
    "AND upper(constraint_type) IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE') "
    "ORDER BY table_name, constraint_name"
)

UNITY_INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL: str = (
    "SELECT constraint_catalog, constraint_schema, constraint_name, table_catalog, "
    "table_schema, table_name, column_name, ordinal_position "
    "FROM `{catalog_esc}`.information_schema.key_column_usage "
    "WHERE lower(table_schema) = lower('{schema_lit}') "
    "ORDER BY constraint_name, ordinal_position"
)

UNITY_INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL: str = (
    "SELECT constraint_catalog, constraint_schema, constraint_name, "
    "unique_constraint_catalog, unique_constraint_schema, unique_constraint_name "
    "FROM `{catalog_esc}`.information_schema.referential_constraints "
    "WHERE lower(constraint_schema) = lower('{schema_lit}')"
)

UNITY_INFORMATION_SCHEMA_TABLES_TABLE_TYPE_SQL: str = (
    "SELECT lower(table_name) AS t, table_type "
    "FROM `{catalog_esc}`.information_schema.tables "
    "WHERE lower(table_schema) = lower('{schema_lit}')"
)

UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL: str = (
    "SELECT table_schema, table_name, column_name, ordinal_position, data_type, is_nullable "
    "FROM `{catalog_esc}`.information_schema.columns "
    "WHERE lower(table_schema) = lower('{schema_lit}') "
    "ORDER BY table_schema, table_name, ordinal_position"
)

UNITY_INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL: str = (
    "SELECT kcu.table_schema, kcu.table_name, kcu.column_name "
    "FROM `{catalog_esc}`.information_schema.table_constraints tc "
    "JOIN `{catalog_esc}`.information_schema.key_column_usage kcu "
    "  ON tc.constraint_schema = kcu.constraint_schema "
    " AND tc.constraint_name = kcu.constraint_name "
    "WHERE lower(tc.table_schema) = lower('{schema_lit}') "
    "  AND upper(tc.constraint_type) = 'UNIQUE' "
    "ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name"
)


def _unity_information_schema_normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return *row* with lowercased string keys for stable Unity Catalog driver column naming."""

    return {str(k).lower(): v for k, v in row.items()}


def _unity_connector_fetchall_dict_rows(cursor: Any, sql: str) -> list[dict[str, Any]]:
    """Execute *sql* on *cursor* and return lower-keyed row dicts."""

    cursor.execute(sql)
    if not cursor.description:
        return []
    col_names = [d[0] for d in cursor.description]
    return [
        _unity_information_schema_normalize_row(dict(zip(col_names, row, strict=True)))
        for row in (cursor.fetchall() or [])
    ]


def _unity_spark_collect_normalized_dicts(spark: Any, sql: str) -> list[dict[str, Any]]:
    """Execute *sql* on *spark* and return lower-keyed row dicts."""

    rows: list[dict[str, Any]] = []
    for r in spark.sql(sql).collect():
        d = r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)
        rows.append(_unity_information_schema_normalize_row(d))
    return rows


def _unity_information_schema_key_column_lists(
    kcu_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    """Group ``key_column_usage`` rows into ordered column-name lists keyed by constraint identity."""

    buckets: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for r in kcu_rows:
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        pos = int(r.get("ordinal_position") or 0)
        cname = str(r.get("column_name") or "")
        buckets.setdefault((cs, cn), []).append((pos, cname))
    out: dict[tuple[str, str], list[str]] = {}
    for key, pairs in buckets.items():
        pairs.sort(key=lambda x: x[0])
        out[key] = [p[1] for p in pairs if p[1]]
    return out


def _unity_trailing_relation_name(ref: str) -> str:
    """Return the trailing SQL identifier segment from a possibly qualified ``catalog.schema.table`` reference."""

    s = str(ref or "").strip()
    if not s:
        return ""
    parts = re.split(r"\s*\.\s*", s)
    tokens: list[str] = []
    for part in parts:
        t = part.strip().strip("`").strip('"').strip()
        if t:
            tokens.append(t)
    return tokens[-1] if tokens else ""


def unity_structural_constraints_index_from_information_schema_rows(
    tc_rows: list[dict[str, Any]],
    kcu_rows: list[dict[str, Any]],
    rc_rows: list[dict[str, Any]],
) -> CatalogStructuralConstraintsIndex:
    """Join normalized Unity ``information_schema`` constraint rows into a :class:`CatalogStructuralConstraintsIndex`."""

    tc_norm = [_unity_information_schema_normalize_row(dict(r)) for r in tc_rows]
    kcu_norm = [_unity_information_schema_normalize_row(dict(r)) for r in kcu_rows]
    rc_norm = [_unity_information_schema_normalize_row(dict(r)) for r in rc_rows]
    kcu_cols = _unity_information_schema_key_column_lists(kcu_norm)
    tc_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in tc_norm:
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        tc_by_key[(cs, cn)] = r
    rc_by_fk: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rc_norm:
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        rc_by_fk[(cs, cn)] = r
    tables_out: dict[str, CatalogTableStructuralConstraints] = {}

    def bundle_for(tk: str) -> CatalogTableStructuralConstraints:
        if tk not in tables_out:
            tables_out[tk] = CatalogTableStructuralConstraints()
        return tables_out[tk]

    def append_pk(tk: str, cols: list[str]) -> None:
        b = bundle_for(tk)
        seen = set(b.primary_keys)
        for c in cols:
            if c and c not in seen:
                seen.add(c)
                b.primary_keys.append(c)

    for r in tc_norm:
        ctype = str(r.get("constraint_type") or "").strip().upper()
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        tname = str(r.get("table_name") or "")
        tk = tname.lower()
        cols = kcu_cols.get((cs, cn), [])
        if ctype == "PRIMARY KEY":
            append_pk(tk, cols)
        elif ctype == "UNIQUE" and len(cols) == 1:
            b = bundle_for(tk)
            uq = cols[0]
            if uq and uq not in b.unique_columns:
                b.unique_columns.append(uq)
        elif ctype == "FOREIGN KEY":
            rc = rc_by_fk.get((cs, cn))
            if not rc:
                continue
            ucs = str(rc.get("unique_constraint_schema") or "")
            ucn = str(rc.get("unique_constraint_name") or "")
            parent_tc = tc_by_key.get((ucs, ucn))
            if not parent_tc:
                continue
            parent_table = str(parent_tc.get("table_name") or "")
            parent_cols = kcu_cols.get((ucs, ucn), [])
            child_table = str(r.get("table_name") or "")
            child_cols = kcu_cols.get((cs, cn), [])
            if not child_cols or not parent_cols or len(child_cols) != len(parent_cols):
                continue
            dst_simple = _unity_trailing_relation_name(parent_table)
            ctk = child_table.lower()
            edge = FKEdge(
                src_table=child_table,
                src_cols=list(child_cols),
                dst_table=dst_simple,
                dst_cols=list(parent_cols),
            )
            bundle_for(ctk).foreign_keys.append(edge)

    return CatalogStructuralConstraintsIndex(tables=tables_out)


class DatabricksDialect(Dialect):
    """Databricks / Spark SQL dialect using EXPLAIN and optional native SQL connector."""

    name: str = "databricks"
    sqlglot_dialect: ClassVar[str] = "spark"

    def quote_table_column(self, table: str, column: str) -> str:
        """Backtick-quote Spark identifiers for ``table.column`` emission."""

        def q(x: str) -> str:
            return "`" + str(x).replace("`", "``") + "`"

        return f"{q(table)}.{q(column)}"

    def __init__(self, config, sqlalchemy_engine: Any | None = None):
        """
        Open a native Databricks SQL connection or fall back to a PySpark session.

        When warehouse credentials are configured (``server_hostname``, ``http_path``, ``access_token``), the ``databricks-sql-connector`` is preferred.  If the connector import or connection attempt fails, the dialect falls back to a cluster-local ``SparkSession``.  A ``RuntimeError`` is raised only when **neither** backend can be established.

        Args:

            config: `DatabricksRuntimeConfig` (catalog, schema, credentials).
            sqlalchemy_engine: Optional SQLAlchemy engine owned by the integrator (skips default warehouse setup when set).

        Raises:

            RuntimeError: If neither connector nor ``SparkSession`` can be created.
        """
        super().__init__(config)

        self.connection = None
        self.spark = None
        self.engine = None

        if sqlalchemy_engine is not None:
            self.engine = sqlalchemy_engine
            debug("[DatabricksDialect.__init__] using caller-provided SQLAlchemy engine")
            return

        connector_error: str | None = None
        last_connector_exc: BaseException | None = None

        if config.has_native_connection():
            try:
                import databricks.sql

                _core_utils.progress(
                    "  Connecting to Databricks SQL warehouse (cold start can take several minutes)...",
                )
                connect_started = time.monotonic()
                self.connection = databricks.sql.connect(
                    server_hostname=config.SERVER_HOSTNAME,
                    http_path=config.HTTP_PATH,
                    access_token=config.ACCESS_TOKEN,
                    _retry_stop_after_attempts_count=30,
                    _retry_delay_max=30,
                    _retry_delay_min=1,
                )
            except Exception as exc:
                last_connector_exc = exc
                connector_error = str(exc)
                debug(f"[DatabricksDialect.__init__] databricks-sql-connector failed: {exc}")

            if self.connection is not None:
                try:
                    cursor = self.connection.cursor()
                    try:
                        cursor.execute("SELECT 1")
                        cursor.fetchall()
                    finally:
                        cursor.close()
                except Exception as exc:
                    debug(f"[DatabricksDialect.__init__] warehouse warmup probe failed: {exc}")
                    if engine_connect_likely_transient(exc):
                        raise DatabasePingFailed(
                            "Databricks warehouse warmup probe failed after connect.",
                        ) from exc
                    raise
                _core_utils.progress(
                    f"  Warehouse ready in {time.monotonic() - connect_started:.1f}s.",
                )
                url = config.sqlalchemy_url()
                if url:
                    try:
                        self.engine = create_engine(url, future=True)
                    except Exception as exc:
                        debug(f"[DatabricksDialect.__init__] SQLAlchemy engine not created: {exc}")
                debug("[DatabricksDialect.__init__] using databricks-sql-connector (warehouse)")
                return

            msg = (
                "databricks-sql-connector failed to open a warehouse session "
                f"({connector_error}). Verify the warehouse is reachable and the "
                "access token is valid; warehouses can take several minutes to "
                "cold-start."
            )
            if last_connector_exc is not None and engine_connect_likely_transient(last_connector_exc):
                raise DatabasePingFailed(msg) from last_connector_exc
            if last_connector_exc is not None:
                raise RuntimeError(msg) from last_connector_exc
            raise RuntimeError(msg)

        self._init_spark_fallback(connector_error)

    def _init_spark_fallback(self, connector_error: str | None) -> None:
        """
        Attempt to initialise a Spark session as the execution backend.

        Tries ``databricks.connect.DatabricksSession`` first to honour the installed ``databricks-connect`` build of ``pyspark``, which hard-rejects ``SparkSession.builder.getOrCreate()``. Falls back to ``pyspark.sql.SparkSession`` only when ``databricks.connect`` is not importable.

        Raises:class:`ConfigError` with the canonical missing-credential hint when neither path yields a session.

        Args:

            connector_error: Diagnostic from a prior connector failure, or ``None``.

        Raises:

            ConfigError: Neither ``DatabricksSession`` nor ``SparkSession`` could be created.
        """

        connect_error: str | None = None
        try:
            from databricks.connect import DatabricksSession

            self.spark = DatabricksSession.builder.getOrCreate()
        except ImportError:
            connect_error = "databricks.connect not installed"
        except Exception as exc:
            connect_error = str(exc)
        else:
            if connector_error is not None:
                debug(
                    f"[DatabricksDialect.__init__] fell back to DatabricksSession after "
                    f"databricks-sql-connector error: {connector_error}"
                )
            else:
                debug("[DatabricksDialect.__init__] using DatabricksSession (databricks-connect)")
            return

        try:
            from pyspark.sql import SparkSession

            self.spark = SparkSession.builder.getOrCreate()
        except Exception as exc:
            spark_error = str(exc)
            hint = "Databricks requires either all SQL warehouse connection variables or an active PySpark session."
            details: list[str] = []
            if connector_error is not None:
                details.append(f"databricks-sql-connector failed ({connector_error})")
            if connect_error is not None:
                details.append(f"DatabricksSession unavailable ({connect_error})")
            details.append(f"SparkSession unavailable ({spark_error})")
            raise ConfigError(f"{hint} " + "; ".join(details)) from exc

        if connector_error is not None:
            debug(
                f"[DatabricksDialect.__init__] fell back to PySpark after "
                f"databricks-sql-connector error: {connector_error}; "
                f"databricks-connect unavailable: {connect_error}"
            )
        else:
            debug("[DatabricksDialect.__init__] using PySpark SparkSession (cluster)")

    def _ast_spark_structural_valid(
        self,
        sql: str,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Sqlglot structural checks mirroring Postgres ``_ast_structural_valid`` intent.

        Args:

            sql: Raw SQL text.

            scalar_cte_names: Scalar-emission CTE names allowed as explicit ``CROSS JOIN`` targets.

        Returns:

            ``(True, "")`` or ``(False, error_code)``.
        """

        return _ast_spark_structural_valid_sqlglot(sql, scalar_cte_names=scalar_cte_names)

    def ast_validate_full(
        self,
        sql: str,
        *,
        schema: SchemaGraph | None = None,
        declared_params: set[str] | None = None,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> list[SqlDiagnostic]:
        """
        Validate Spark SQL structurally and (when *schema* is given) semantically.

        Args:

            sql: SQL text.

            schema: Optional schema graph; enables column/table existence and
            ambiguity checks via :func:`_check_schema_references_shared`.

            declared_params: Optional declared placeholder names; missing ``:name``
            placeholders emit :attr:`SqlDiagnosticCode.PARAM_UNBOUND`.

            scalar_cte_names: Scalar-emission CTE names allowed as ``CROSS JOIN`` targets.

        Returns:

            List of :class:`SqlDiagnostic` findings; empty list means valid.
        """
        ok, code = self._ast_spark_structural_valid(sql, scalar_cte_names=scalar_cte_names)
        if not ok:
            mapped = _PG_STRUCTURAL_CODE_TO_DIAG.get(code, SqlDiagnosticCode.FORBIDDEN_STRUCTURE)
            return [SqlDiagnostic(code=mapped, message=code, node_kind=None)]
        diags: list[SqlDiagnostic] = []
        try:
            tree = sqlglot.parse_one(sql, dialect="spark")
        except Exception:
            return [SqlDiagnostic(code=SqlDiagnosticCode.AST_PARSE_FAILED, message="parse failed")]
        if not isinstance(tree, sqlglot.exp.Select):
            return diags
        cte_names = self._spark_collect_cte_names(tree)
        alias_to_table = self._spark_collect_table_aliases(tree)
        if schema is not None:
            refs = self._spark_collect_column_refs(tree)
            diags += _check_schema_references_shared(refs, alias_to_table, cte_names, schema)
        diags += self._spark_check_grouping(tree)
        diags += self._spark_check_cte_closure(tree, cte_names)
        if declared_params is not None:
            diags += self._spark_check_param_coverage(sql, declared_params)
        return diags

    def _spark_collect_cte_names(self, tree: sqlglot.exp.Select) -> set[str]:
        """Return the set of lowercased CTE names defined on *tree*'s ``WITH`` clause."""
        names: set[str] = set()
        wc = tree.args.get("with_")
        if wc is None:
            return names
        for cte in wc.expressions or ():
            alias = cte.alias_or_name
            if isinstance(alias, str) and alias:
                names.add(alias.lower())
        return names

    def _spark_collect_table_aliases(self, tree: sqlglot.exp.Select) -> dict[str, str]:
        """Return ``{alias_or_table_lc: real_table_lc}`` for every ``sqlglot.exp.Table`` in *tree*."""
        out: dict[str, str] = {}
        for t in tree.find_all(sqlglot.exp.Table):
            real = (t.name or "").lower()
            if not real:
                continue
            out[real] = real
            alias_node = t.args.get("alias")
            if alias_node is not None:
                a = alias_node.name if hasattr(alias_node, "name") else None
                if isinstance(a, str) and a:
                    out[a.lower()] = real
        return out

    def _spark_collect_column_refs(self, tree: sqlglot.exp.Select) -> list[tuple[str | None, str]]:
        """Return ``(prefix, column)`` pairs for every ``sqlglot.exp.Column`` in *tree*."""
        out: list[tuple[str | None, str]] = []
        for c in tree.find_all(sqlglot.exp.Column):
            col = c.name or ""
            if not col or col == "*":
                continue
            tbl = c.table or None
            out.append((tbl or None, col))
        return out

    def _spark_check_grouping(self, tree: sqlglot.exp.Select) -> list[SqlDiagnostic]:
        """Emit AGG_IN_WHERE and HAVING_WITHOUT_GROUP diagnostics for *tree*."""
        diags: list[SqlDiagnostic] = []
        where = tree.args.get("where")
        if where is not None:
            for agg in where.find_all(sqlglot.exp.AggFunc):
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.AGG_IN_WHERE,
                        message=f"aggregate {type(agg).__name__.lower()!r} in WHERE",
                        node_kind="AggFunc",
                        offending_identifier=type(agg).__name__.lower(),
                    )
                )
                break
        having = tree.args.get("having")
        group = tree.args.get("group")
        if having is not None and group is None:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.HAVING_WITHOUT_GROUP,
                    message="HAVING without GROUP BY",
                    node_kind="Select",
                )
            )
        return diags

    def _spark_check_cte_closure(self, tree: sqlglot.exp.Select, cte_names: set[str]) -> list[SqlDiagnostic]:
        """Flag CTE names defined but never referenced as a table elsewhere."""
        if not cte_names:
            return []
        wc = tree.args.get("with_")
        defining_ids: set[int] = set()
        if wc is not None:
            for cte in wc.expressions or ():
                inner = cte.this
                if inner is not None:
                    defining_ids.add(id(inner))
        referenced: set[str] = set()
        for t in tree.find_all(sqlglot.exp.Table):
            name = (t.name or "").lower()
            if name in cte_names:
                referenced.add(name)
        for col in tree.find_all(sqlglot.exp.Column):
            tbl = col.table
            if tbl is None:
                continue
            name = tbl if isinstance(tbl, str) else getattr(tbl, "name", "") or ""
            if isinstance(name, str) and name.lower() in cte_names:
                referenced.add(name.lower())
        unreferenced = sorted(cte_names - referenced)
        return [
            SqlDiagnostic(
                code=SqlDiagnosticCode.CTE_UNREFERENCED,
                message=f"CTE {n!r} is defined but never referenced",
                node_kind="CTE",
                offending_identifier=n,
            )
            for n in unreferenced
        ]

    def _spark_check_param_coverage(self, sql: str, declared: set[str]) -> list[SqlDiagnostic]:
        """Emit a diagnostic for each ``:name`` placeholder in *sql* not present in *declared*."""
        diags: list[SqlDiagnostic] = []
        seen: set[str] = set()
        for match in NAMED_PLACEHOLDER_RE.finditer(sql):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            if name not in declared:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.PARAM_UNBOUND,
                        message=f"unbound placeholder :{name}",
                        node_kind="Parameter",
                        offending_identifier=name,
                    )
                )
        return diags

    def parse_select(self, sql: str) -> sqlglot.exp.Select | None:
        """Parse *sql* via sqlglot's ``spark`` dialect; return ``None`` for non-``SELECT`` roots or parse failure."""

        try:
            tree = sqlglot.parse_one(sql, dialect="spark")
        except Exception:
            return None
        if not isinstance(tree, sqlglot.exp.Select):
            return None
        return tree

    def ordered_join_carrier_froms(self, parsed: sqlglot.exp.Select) -> list[sqlglot.exp.Select] | None:
        """
        Return inner-CTE ``Select`` nodes (left-to-right) followed by the outer ``Select``.

        Joins are appended to each ``Select`` (sqlglot stores ``joins`` on the ``Select`` itself). ``None`` is returned for unsupported shapes (e.g. top-level ``UNION``).
        """

        if isinstance(parsed, sqlglot.exp.Union):
            return None
        if not isinstance(parsed, sqlglot.exp.Select):
            return None
        out: list[sqlglot.exp.Select] = []
        wc = parsed.args.get("with_")
        if wc:
            for cte in wc.expressions:
                inner = cte.this
                if isinstance(inner, sqlglot.exp.Select) and isinstance(inner.args.get("from_"), sqlglot.exp.From):
                    out.append(inner)
        if isinstance(parsed.args.get("from_"), sqlglot.exp.From):
            out.append(parsed)
        return out

    def from_anchor_of(self, carrier: sqlglot.exp.Select) -> str | None:
        """
        Read the bare table name of *carrier*'s leftmost ``FROM`` leaf via sqlglot.

        Descends nested ``Join`` nodes until a ``Table`` is reached. Returns ``None`` for subqueries or unsupported shapes.
        """

        from_node = carrier.args.get("from_") if isinstance(carrier, sqlglot.exp.Select) else None
        if not isinstance(from_node, sqlglot.exp.From):
            return None
        target = from_node.this
        while isinstance(target, sqlglot.exp.Join):
            inner = target.this
            if inner is None:
                return None
            target = inner
        if not isinstance(target, sqlglot.exp.Table):
            return None
        name = target.name or ""
        if not name:
            return None
        return name.lower()

    def attach_joins(
        self,
        parsed: sqlglot.exp.Select,
        from_handle: sqlglot.exp.Select,
        edges: list[JoinEdge],
    ) -> bool:
        """Build sqlglot ``sqlglot.exp.Join`` nodes from *edges* and append them to *from_handle*."""

        if not edges:
            return False
        if not isinstance(from_handle, sqlglot.exp.Select):
            return False
        new_joins: list[sqlglot.exp.Join] = []
        for edge in edges:
            on_expr = self._dbr_build_on_expr(edge.on_terms)
            if on_expr is None:
                return False
            table_node = sqlglot.exp.Table(this=sqlglot.exp.to_identifier(edge.table))
            if edge.alias:
                table_node.set(
                    "alias",
                    sqlglot.exp.TableAlias(this=sqlglot.exp.to_identifier(edge.alias)),
                )
            join_kwargs: dict[str, Any] = {
                "this": table_node,
                "on": on_expr,
                "kind": "INNER",
            }
            if edge.kind == "LEFT":
                join_kwargs["side"] = "LEFT"
                join_kwargs["kind"] = None
            new_joins.append(sqlglot.exp.Join(**{k: v for k, v in join_kwargs.items() if v is not None}))
        existing = list(from_handle.args.get("joins") or [])
        from_handle.set("joins", existing + new_joins)
        return True

    def _dbr_build_on_expr(
        self,
        on_terms: tuple[tuple[str, str, str, str], ...],
    ) -> sqlglot.exp.Expression | None:
        """Return a single ``sqlglot.exp.EQ`` or an ``AND``-tree over dialect-quoted equality conjuncts."""

        if not on_terms:
            return None
        eqs: list[sqlglot.exp.Expression] = []
        for left_token, left_col, right_token, right_col in on_terms:
            lhs_sql = self.quote_table_column(left_token, left_col)
            rhs_sql = self.quote_table_column(right_token, right_col)
            try:
                pred_tree = sqlglot.parse_one(
                    f"SELECT 1 FROM t WHERE {lhs_sql} = {rhs_sql}",
                    dialect="spark",
                )
            except Exception:
                return None
            where_node = pred_tree.args.get("where")
            if where_node is None:
                return None
            eqs.append(where_node.this)
        node: sqlglot.exp.Expression = eqs[0]
        for nxt in eqs[1:]:
            node = sqlglot.exp.And(this=node, expression=nxt)
        _core_utils.pipeline_trace_lazy(
            "pipeline.join_resolve.dialect_quote_join_clause",
            lambda: _core_utils.stable_json({"conjuncts": len(on_terms)}),
        )
        return node

    def attach_extra_from_and_where(
        self,
        parsed: sqlglot.exp.Select,
        from_handle: sqlglot.exp.Select,
        extra_from_tables: list[str],
        where_edges: list[JoinEdge],
    ) -> bool:
        """Append comma-FROM tables and AND equality predicates into the carrier ``WHERE``."""

        if not extra_from_tables and not where_edges:
            return True
        if not isinstance(from_handle, sqlglot.exp.Select):
            return False
        if extra_from_tables:
            existing_joins = list(from_handle.args.get("joins") or [])
            for tbl in extra_from_tables:
                existing_joins.append(
                    sqlglot.exp.Join(
                        this=sqlglot.exp.Table(this=sqlglot.exp.to_identifier(tbl)),
                        kind="CROSS",
                    ),
                )
            from_handle.set("joins", existing_joins)
        if not where_edges:
            return True
        new_eqs: list[sqlglot.exp.Expression] = []
        for edge in where_edges:
            pred = self._dbr_build_on_expr(edge.on_terms)
            if pred is None:
                return False
            new_eqs.append(pred)
        if not new_eqs:
            return True
        new_pred: sqlglot.exp.Expression = new_eqs[0]
        for nxt in new_eqs[1:]:
            new_pred = sqlglot.exp.And(this=new_pred, expression=nxt)
        existing_where = from_handle.args.get("where")
        if existing_where is None:
            from_handle.set("where", sqlglot.exp.Where(this=new_pred))
        else:
            existing_pred = existing_where.this
            merged_pred: sqlglot.exp.Expression = sqlglot.exp.And(this=existing_pred, expression=new_pred)
            existing_where.set("this", merged_pred)
        return True

    def replace_projection(
        self,
        parsed: sqlglot.exp.Select,
        items: list[tuple[str, str | None]],
    ) -> bool:
        """Replace the outer ``Select``'s projection list by parsing each *expr_sql* via sqlglot."""

        if not isinstance(parsed, sqlglot.exp.Select):
            return False
        new_exprs: list[sqlglot.exp.Expression] = []
        for expr_sql, alias in items:
            try:
                tree = sqlglot.parse_one(f"SELECT {expr_sql}", dialect="spark")
            except Exception:
                return False
            if not isinstance(tree, sqlglot.exp.Select):
                return False
            tlist = tree.args.get("expressions") or []
            if len(tlist) != 1:
                return False
            value_node = tlist[0]
            if isinstance(value_node, sqlglot.exp.Alias) and alias:
                value_node = value_node.this
            if alias:
                value_node = sqlglot.exp.alias_(value_node, alias)
            new_exprs.append(value_node)
        parsed.set("expressions", new_exprs)
        return True

    def emit_sql(self, parsed: sqlglot.exp.Select) -> str:
        """Render *parsed* via sqlglot's ``spark`` dialect, preserving ``:pN`` / ``:sN`` placeholders."""

        return parsed.sql(dialect="spark")

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """
        Run Spark/Databricks ``EXPLAIN`` and return ``(ok, diagnostics, raw_message)``.

        ``ok`` is False only on hard validation failures. A permission-denied
        error disables EXPLAIN for the remainder of this dialect instance and
        is reported as ``ok=True`` with no diagnostics so the caller can
        proceed without treating missing privileges as invalid SQL. Soft
        plan-shape findings (suspected cartesian joins, zero-row estimates)
        are emitted as :class:`SqlDiagnostic` entries with codes from
        ``SOFT_DIAGNOSTIC_CODES`` in ``_config`` so callers may apply confidence
        penalties without rejecting the SQL.

        Args:

            sql: SQL text.

            params: Ignored for Databricks warehouse and Spark paths.

            schema: When set with *intent*, partition filters are injected so
            ``EXPLAIN`` matches execution.

            intent: Paired with *schema* for partition filter injection.

        Returns:

            ``(ok, diagnostics, raw_message)``.
        """
        finalized = self.finalize_render(
            sql,
            params or {},
            schema=schema,
            intent=intent,
        )
        explain_sql = f"EXPLAIN COST {finalized}"
        if self.engine is not None:
            try:
                with self.engine.connect() as conn:
                    rows = conn.execute(text(explain_sql)).fetchall()
                text_payload = "\n".join(str(r[0]) for r in rows if r and r[0] is not None)
                er, eb = _databricks_plan_stats_from_explain_text(text_payload)
                failed, why = _explain_cost_gate_violation(er, eb)
                if failed:
                    return (
                        False,
                        [
                            SqlDiagnostic(
                                code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED,
                                message=why,
                            )
                        ],
                        why,
                    )
                return True, _databricks_diagnostics_from_explain_text(text_payload), ""
            except Exception as e:
                err = str(e)
                if self._disable_explain_on_permission_denied(err):
                    return True, [], ""
                return (
                    False,
                    [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)],
                    err,
                )
        if self.connection is not None:
            try:
                cursor = self.connection.cursor()
                cursor.execute(explain_sql)
                rows = cursor.fetchall()
                cursor.close()
                text_payload = "\n".join(str(r[0]) for r in rows if r and r[0] is not None)
                er, eb = _databricks_plan_stats_from_explain_text(text_payload)
                failed, why = _explain_cost_gate_violation(er, eb)
                if failed:
                    return (
                        False,
                        [
                            SqlDiagnostic(
                                code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED,
                                message=why,
                            )
                        ],
                        why,
                    )
                return True, _databricks_diagnostics_from_explain_text(text_payload), ""
            except Exception as e:
                err = str(e)
                if self._disable_explain_on_permission_denied(err):
                    return True, [], ""
                return (
                    False,
                    [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)],
                    err,
                )
        if self.spark is not None:
            try:
                tm_ex = effective_explain_timeout_ms()
                if tm_ex is not None:
                    self.spark.conf.set(
                        "spark.databricks.sql.statementTimeout",
                        f"{int(tm_ex)}ms",
                    )
                explain_df = self.spark.sql(explain_sql)
                rows = explain_df.collect()
                text_payload = "\n".join(str(r[0]) for r in rows if r and r[0] is not None)
                er, eb = _databricks_plan_stats_from_explain_text(text_payload)
                failed, why = _explain_cost_gate_violation(er, eb)
                if failed:
                    return (
                        False,
                        [
                            SqlDiagnostic(
                                code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED,
                                message=why,
                            )
                        ],
                        why,
                    )
                return True, _databricks_diagnostics_from_explain_text(text_payload), ""
            except Exception as e:
                err = str(e)
                if self._disable_explain_on_permission_denied(err):
                    return True, [], ""
                return (
                    False,
                    [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)],
                    err,
                )
        return True, [], ""

    def prepare_for_execution(self, sql: str) -> str:
        """
        Return SQL with `FROM`/`JOIN` tables qualified for Spark.

        Args:

            sql: SQL text before qualification.

        Returns:

            Qualified SQL string.
        """
        return _qualify_tables_ast(
            sql,
            sqlglot_dialect="spark",
            catalog=str(self.config.CATALOG),
            schema=str(self.config.SCHEMA),
            cte_names=set(),
            backtick=True,
        )

    def _qualify_table_references(self, sql: str) -> str:
        """Deprecated regex-based qualifier kept as a thin wrapper around the AST helper."""
        return self.prepare_for_execution(sql)

    def can_explain(self) -> bool:
        """
        Return True when SQLAlchemy, the native connector, or Spark can run EXPLAIN.

        Returns ``False`` once a permission-denied error has disabled EXPLAIN for this dialect instance (see :meth:`Dialect._disable_explain_on_permission_denied`).
        """
        if self._explain_disabled:
            return False
        if self.engine is not None:
            return True
        if self.connection is not None or self.spark is not None:
            return True
        return False

    def inject_partition_filters(self, sql: str, schema: SchemaGraph, intent: RuntimeIntent) -> str:
        """
        Append partition predicates for Delta table pruning when missing from the query.

        Args:

            sql: Executable Spark SQL.

            schema: Schema graph with partition column metadata.

            intent: Runtime intent with filters and parameters.

        Returns:

            SQL with predicates merged into WHERE when needed.
        """
        return _dbr_inject_partition_filters(sql, schema, intent)

    def finalize_render(
        self,
        sql_param: str,
        params: dict[str, Any],
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
        execution_sql_override: str | None = None,
        structural_defaults: dict[str, Any] | None = None,
    ) -> str:
        """
        Qualify tables, finalize parameters, AST-simplify, and inject partition filters for Spark execution.

        Args:

            sql_param: Parameterized SQL.

            params: Resolved bind values.

            schema: Optional schema graph (required for partition pruning).

            intent: Optional runtime intent (required for partition pruning).

            execution_sql_override: Optional pre-qualified SQL body.

            structural_defaults: Structural defaults for ``:sN`` placeholders.

        Returns:

            Executable Spark SQL.
        """
        sql_in_raw = execution_sql_override or sql_param
        prepared = execution_sql_override or self.prepare_for_execution(sql_param)
        _trace_finalize_render_stage("prepare_for_execution", sql_in_raw, prepared)
        substituted = finalize_executable_sql(
            prepared,
            params,
            structural_defaults,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        _trace_finalize_render_stage("finalize_executable_sql", prepared, substituted)
        non_empty_in = (execution_sql_override or sql_param or "").strip()
        if non_empty_in and not substituted.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=finalize_executable_sql"
            )
        after_fin = substituted
        substituted = _databricks_normalize_datetrunc_sql(substituted)
        _trace_finalize_render_stage(
            "_databricks_normalize_datetrunc_sql",
            after_fin,
            substituted,
        )
        if non_empty_in and not substituted.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=_databricks_normalize_datetrunc_sql"
            )
        if schema is not None and intent is not None:
            before_inj = substituted
            substituted = self.inject_partition_filters(substituted, schema, intent)
            _trace_finalize_render_stage(
                "inject_partition_filters",
                before_inj,
                substituted,
            )
            if non_empty_in and not substituted.strip():
                raise RuntimeError(
                    "dialect.finalize_render produced empty SQL from non-empty input; "
                    "last_non_empty_stage=inject_partition_filters"
                )
        return substituted

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple]:
        """
        Execute finalized Spark SQL via the warehouse connector or Spark session.

        Args:

            sql: SQL already qualified and finalized for execution.

            params: Ignored; literals are inlined before execution.

        Returns:

            Result rows as tuples.
        """
        _ = params
        if diagnostic_debug_enabled():
            debug(f"[DatabricksDialect.execute] sql=\n{sql}")

        try:
            if self.connection is not None:
                cursor = self.connection.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                cursor.close()
                return [tuple(row) for row in rows]

            tm = PolicyConfig.STATEMENT_TIMEOUT_MS
            if cost_cap_active(tm):
                self.spark.conf.set("spark.databricks.sql.statementTimeout", f"{int(tm)}ms")
            df = self.spark.sql(sql)
            return [tuple(row) for row in df.collect()]
        except Exception as e:
            err = str(e)
            if _is_permission_denied_error(err):
                raise AccessError("execute", err) from e
            el = err.lower()
            if "timeout" in el and ("statement" in el or "cancel" in el or "deadline" in el):
                raise StatementTimeoutError(err) from e
            raise

    def quote_identifier(self, ident: str) -> str:
        """
        Quote a Spark identifier with backticks.

        Args:

            ident: Bare identifier without surrounding quotes.

        Returns:

            Spark-style escaped identifier wrapped in grave accents.
        """

        s = str(ident).strip()
        esc = s.replace("`", "``")
        return f"`{esc}`"

    def unity_table_types_map(self) -> dict[str, str]:
        """Return lowercased Unity relation name to ``information_schema.tables.table_type`` string."""

        catalog = str(self.config.CATALOG or "")
        schema_name = str(self.config.SCHEMA or "")
        types: dict[str, str] = {}
        if not catalog or not schema_name:
            return types
        esc_cat = catalog.replace("`", "``")
        lit = str(schema_name).replace("'", "''")
        q = UNITY_INFORMATION_SCHEMA_TABLES_TABLE_TYPE_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        try:
            if self.connection is not None:
                with self.connection.cursor() as cur:
                    cur.execute(q)
                    for row in cur.fetchall() or []:
                        types[str(row[0]).lower()] = str(row[1])
            elif self.spark is not None:
                for row in self.spark.sql(q).collect():
                    d = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)
                    key_t = d.get("t")
                    if key_t is None:
                        key_t = d.get("table_name")
                    types[str(key_t).lower()] = str(d.get("table_type") or d.get("TABLE_TYPE") or "")
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.unity_table_types_map] failed: {exc!r}")
        return types

    def unity_structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and single-column UNIQUE metadata from Unity ``information_schema``."""

        catalog = str(self.config.CATALOG or "")
        schema_name = str(self.config.SCHEMA or "")
        if not catalog or not schema_name:
            return CatalogStructuralConstraintsIndex.empty()
        esc_cat = catalog.replace("`", "``")
        lit = str(schema_name).replace("'", "''")
        tc_sql = UNITY_INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        kcu_sql = UNITY_INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        rc_sql = UNITY_INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        try:
            if self.connection is not None:
                with self.connection.cursor() as cur:
                    t_rows = _unity_connector_fetchall_dict_rows(cur, tc_sql)
                    k_rows = _unity_connector_fetchall_dict_rows(cur, kcu_sql)
                    r_rows = _unity_connector_fetchall_dict_rows(cur, rc_sql)
                return unity_structural_constraints_index_from_information_schema_rows(t_rows, k_rows, r_rows)
            if self.spark is not None:
                t_rows = _unity_spark_collect_normalized_dicts(self.spark, tc_sql)
                k_rows = _unity_spark_collect_normalized_dicts(self.spark, kcu_sql)
                r_rows = _unity_spark_collect_normalized_dicts(self.spark, rc_sql)
                return unity_structural_constraints_index_from_information_schema_rows(t_rows, k_rows, r_rows)
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.unity_structural_constraints_index] failed: {exc!r}")
            return CatalogStructuralConstraintsIndex.empty()

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
    ) -> SchemaGraph:
        """
        Build a graph from Unity Catalog or DDL fallback.

        Args:

            include: Reflect stored relations, views, or both.

            allow_objects: When set, restrict catalog extraction to these relation names (case-insensitive).

        Returns:

            ``SchemaGraph`` for the configured catalog and schema.
        """
        unity_types = self.unity_table_types_map()
        structural_index = self.unity_structural_constraints_index()
        return load_or_create_schema_databricks(
            spark_session=self.spark,
            connection=self.connection,
            include=include,
            allow_objects=allow_objects,
            unity_table_types=unity_types,
            structural_constraints_index=structural_index,
        )

    def compute_ddl_probe(self, schema_context: SchemaContext) -> str:
        """
        Return SHA-256 over ``information_schema.columns`` rows for the configured catalog and schema on Databricks.

        Tries the SQL connector first, then falls back to a Spark session. Always returns ``""`` rather than raising so build_schema_graph degrades to the legacy fingerprint validation when the probe cannot run.
        """
        _ = schema_context
        try:
            catalog = str(self.config.CATALOG or "")
            schema_name = str(self.config.SCHEMA or "")
            if not catalog or not schema_name:
                return ""
            esc_cat = catalog.replace("`", "``")
            esc_sch = schema_name.replace("'", "''")
            cols_sql = UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL.format(catalog_esc=esc_cat, schema_lit=esc_sch)
            unique_sql = UNITY_INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL.format(
                catalog_esc=esc_cat,
                schema_lit=esc_sch,
            )
            rows: list[tuple] = []
            uniq_rows: list[tuple] = []
            if self.connection is not None:
                with self.connection.cursor() as cur:
                    cur.execute(cols_sql)
                    rows = list(cur.fetchall() or [])
                    try:
                        cur.execute(unique_sql)
                        uniq_rows = list(cur.fetchall() or [])
                    except Exception as uexc:
                        debug(f"[dialect.DatabricksDialect.compute_ddl_probe] unique probe failed: {uexc!r}")
                        uniq_rows = []
            elif self.spark is not None:
                for r in self.spark.sql(cols_sql).collect():
                    d = r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)
                    rows.append(
                        (
                            d.get("table_schema"),
                            d.get("table_name"),
                            d.get("column_name"),
                            d.get("ordinal_position"),
                            d.get("data_type"),
                            d.get("is_nullable"),
                        )
                    )
                try:
                    for r in self.spark.sql(unique_sql).collect():
                        d = r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)
                        uniq_rows.append(
                            (
                                d.get("table_schema"),
                                d.get("table_name"),
                                d.get("column_name"),
                            )
                        )
                except Exception as uexc:
                    debug(f"[dialect.DatabricksDialect.compute_ddl_probe] unique probe failed: {uexc!r}")
                    uniq_rows = []
            else:
                return ""
            payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
            payload_uniq = "\n".join("|".join("" if c is None else str(c) for c in r) for r in uniq_rows)
            return sha256(payload_cols + "\n##UNIQUE##\n" + payload_uniq)
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""

    def profile_schema(self, sg: SchemaGraph) -> None:
        """
        Profile Databricks tables via the SQL connector or Spark.

        Args:

            sg: Schema graph to update in place.
        """
        catalog = self.config.CATALOG
        schema_name = self.config.SCHEMA
        if self.connection is not None:
            profile_schema_sql_connector(self.connection, catalog, schema_name, sg)
            return
        spark = self.spark
        if spark is None:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
        profile_schema_spark(spark, catalog, schema_name, sg)

    def refresh_full_table_distinct_for_pk_inference(
        self,
        table_name: str,
        col_name: str,
        *,
        table_kind: Literal["table", "view"] = "table",
    ) -> tuple[int, int, float] | None:
        """
        Run full-table statistics for PK inference after sampled profiling.

        Args:

            table_name: Table name within the configured catalog/schema.

            col_name: Column name.

            table_kind: Physical table or view (reserved).

        Returns:

            ``(distinct_count, row_count, null_ratio)`` or ``None`` on failure.
        """
        try:
            _ = table_kind
            catalog = self.config.CATALOG
            schema_name = self.config.SCHEMA
            full_table = f"`{catalog}`.`{schema_name}`.`{table_name}`"
            sql = (
                f"SELECT COUNT(*) AS cnt, COUNT(DISTINCT `{col_name}`) AS dist, "
                f"COUNT(*) - COUNT(`{col_name}`) AS nulls FROM {full_table}"
            )
            if self.connection is not None:
                with self.connection.cursor() as cursor:
                    cursor.execute(sql)
                    rows = _cursor_rows_as_dicts(cursor)
                if not rows:
                    return None
                r = rows[0]
                cnt = int(r.get("cnt") or 0)
                dist = int(r.get("dist") or 0)
                nulls = int(r.get("nulls") or 0)
                nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
                return (dist, cnt, nr)
            spark = self.spark
            if spark is None:
                from pyspark.sql import SparkSession

                spark = SparkSession.builder.getOrCreate()
            row = spark.sql(sql).collect()[0]
            cnt = int(row["cnt"] or 0)
            dist = int(row["dist"] or 0)
            nulls = int(row["nulls"] or 0)
            nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
            return (dist, cnt, nr)
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.refresh_full_table_distinct_for_pk_inference] failed: {exc!r}")
            return None

    def render_date_diff(
        self,
        left_expr: str,
        op: str,
        unit: str,
        amount: int,
        *,
        minuend_sql: str = "",
        subtrahend_sql: str = "",
    ) -> str:
        """
        Render Spark ``DATEDIFF``-based date-difference comparison.

        When *minuend_sql* and *subtrahend_sql* are available the method emits ``DATEDIFF(minuend, subtrahend)`` which returns an integer, avoiding the INTERVAL-vs-INT type mismatch that raw date subtraction causes on Databricks/Spark.

        Args:

            left_expr: Pre-rendered subtraction expression (fallback).

            op: Comparison operator.

            unit: Calendar unit.

            amount: Magnitude in that unit.

            minuend_sql: First date column SQL.

            subtrahend_sql: Second date column SQL.

        Returns:

            Predicate SQL.
        """
        days = _unit_to_approx_days(unit, amount)
        if minuend_sql and subtrahend_sql:
            sql = f"DATEDIFF({minuend_sql}, {subtrahend_sql}) {op} {days}"
        else:
            sql = f"({left_expr}) {op} {days}"
        return _emit_via_ast(sql, "spark")

    def render_array_contains(self, column_sql: str, param_key: str) -> str:
        """
        Render Databricks array membership with trimmed element comparison.

        Args:

            column_sql: Array column SQL.

            param_key: Bind parameter name without colon.

        Returns:

            ``ARRAY_CONTAINS`` predicate SQL.
        """
        trim_set = "CONCAT(' ', chr(34), chr(39))"
        norm_bind = f"LOWER(TRIM(CAST(:{param_key} AS STRING), {trim_set}))"
        xform = f"TRANSFORM({column_sql}, _ac_x -> LOWER(TRIM(CAST(_ac_x AS STRING), {trim_set})))"
        sql = f"({column_sql} IS NOT NULL AND ARRAY_CONTAINS({xform}, {norm_bind}))"
        return _emit_via_ast(sql, "spark")

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """
        Render Spark ``EXPLODE`` for SELECT list.

        Args:

            column_sql: Array column SQL.

            alias: Output alias.

        Returns:

            EXPLODE fragment with alias.
        """
        sql = f"EXPLODE({column_sql}) AS {alias}"
        return _emit_via_ast(sql, "spark")

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """
        Render Spark date window boundaries.

        Args:

            column: Column SQL.

            op: Comparison operator.

            unit: Calendar or time unit.

            amount: Distance from current date.

        Returns:

            WHERE fragment SQL.
        """
        if amount == 0:
            sql = f"{column} {op} date_trunc('{unit}', current_date())"
        elif unit == "day":
            sql = f"{column} {op} date_sub(current_date(), {amount})"
        elif unit == "week":
            sql = f"{column} {op} date_sub(current_date(), {amount * 7})"
        elif unit == "month":
            sql = f"{column} {op} add_months(current_date(), -{amount})"
        elif unit == "quarter":
            sql = f"{column} {op} add_months(current_date(), -{amount * 3})"
        elif unit == "half_year":
            sql = f"{column} {op} add_months(current_date(), -{amount * 6})"
        elif unit == "year":
            sql = f"{column} {op} add_months(current_date(), -{amount * 12})"
        elif unit in {"hour", "minute", "second"}:
            scaled, plural_unit = _format_interval_unit(unit, amount)
            sql = f"{column} {op} (current_timestamp() - INTERVAL '{scaled} {plural_unit}')"
        else:
            sql = f"{column} {op} date_sub(current_date(), {amount})"
        return _emit_via_ast(sql, "spark")

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """
        Wrap an expression for case-insensitive comparison on Spark.

        Args:

            expr: SQL expression.

        Returns:

            ``LOWER(TRIM(...))`` SQL.
        """
        return f"LOWER(TRIM({expr}))"


_DIALECT_REGISTRY: dict[str, type[Dialect]] = {}


def extra_filter_ops_for_engine(engine_type: str | None = None) -> list[str]:
    """
    Return dialect-specific filter operators for intent-parse prompts without constructing a dialect.

    Avoids ``PostgresDialect.__init__`` (and ``create_engine``) when only ``extra_filter_ops`` text is needed.
    """

    et = (engine_type or EngineConfig.TYPE).strip().lower()
    if et == "postgresql":
        return ["ilike", "not ilike"]
    return []


def register_dialect(name: str, cls: type[Dialect]) -> None:
    """
    Register a dialect implementation under an engine name.

    Args:

        name: Engine string such as ``postgresql``.

        cls: Concrete ``Dialect`` subclass.
    """
    _DIALECT_REGISTRY[name] = cls


def resolve_dialect(name_or_api: str | Dialect) -> Dialect:
    """
    Return a dialect instance for helpers that accept either an engine name or a live dialect.

    Args:

        name_or_api: Engine string such as ``"postgresql"`` or an existing ``Dialect`` instance.

    Returns:

        A ``Dialect`` suitable for ``finalize_render`` and ``execute``.

    Raises:

        TypeError: When ``name_or_api`` is neither ``str`` nor ``Dialect``.
    """

    if isinstance(name_or_api, Dialect):
        return name_or_api
    if isinstance(name_or_api, str):
        return get_dialect(name_or_api)
    raise TypeError(f"Expected str or Dialect, got {type(name_or_api).__name__}")


def get_dialect(
    engine_type: str | None = None,
    config: Any | None = None,
    sqlalchemy_engine: Any | None = None,
) -> Dialect:
    """
    Construct the dialect implementation for an engine type.

    Args:

        engine_type: Engine name; defaults to ``EngineConfig.TYPE``.

        config: Runtime config class or instance; defaults to ``EngineConfig.RUNTIME``.

        sqlalchemy_engine: Optional SQLAlchemy :class:`sqlalchemy.engine.Engine` owned by the
            caller (read-replica routing or external pool management).

    Returns:

        Registered dialect instance.

    Raises:

        ValueError: If ``engine_type`` is not registered.
    """
    if engine_type is None:
        engine_type = EngineConfig.TYPE
    if config is None:
        default_runtime_by_engine: dict[str, Any] = {
            "postgresql": PostgresRuntimeConfig,
            "databricks": DatabricksRuntimeConfig,
        }
        config = default_runtime_by_engine.get(engine_type, EngineConfig.RUNTIME)
    if engine_type not in _DIALECT_REGISTRY:
        raise ValueError(f"Unsupported dialect: {engine_type}")
    ctor = _DIALECT_REGISTRY[engine_type]
    if sqlalchemy_engine is not None:
        return ctor(config, sqlalchemy_engine=sqlalchemy_engine)
    return ctor(config)


register_dialect("postgresql", PostgresDialect)
register_dialect("databricks", DatabricksDialect)
