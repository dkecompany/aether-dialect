"""Column profiling, LLM classification, DDL parsing, and catalog extraction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import sqlglot
from sqlalchemy import text

from ._config import EngineConfig, PolicyConfig
from ._constants import (
    BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES,
    BOOLEAN_ANTONYM_MIN_STEM_LEN,
    BOOLEAN_NEGATION_PREFIXES,
    BOOLEAN_NEGATION_SUFFIXES,
    BOOLEAN_TRUTH_PATTERN_MAP,
    BUSINESS_KNOWLEDGE_NOTES_EXTRACT_SYSTEM,
    COLUMN_DEFINITION_STOP_WORDS,
    DATE_COLUMN_NAME_TOKENS,
    DEFAULT_RANDOM_SEED,
    DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED,
    DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED,
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED,
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT,
    DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED,
    DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE,
    DURATION_COLUMN_NAME_TOKENS,
    JSON_COLUMN_TYPE_TOKENS,
    NAME_COLUMN_PATTERN,
    ROLE_VALUE_TYPE_COMPAT,
    SCHEMA_CLASSIFY_ERROR_DETAIL_CAP,
    SCHEMA_CLASSIFY_SYSTEM,
    SCHEMA_CONSISTENCY_REFINE_SYSTEM,
    SCHEMA_NOTES_REFINE_SYSTEM,
    UNKNOWN_VALUE_TYPE,
    VALID_SENSITIVITY_LEVELS,
    YEAR_LIKE_COLUMN_NAME_TOKENS,
)
from ._contracts_base import (
    BusinessKnowledgeEntry,
    BusinessKnowledgeKind,
    ColumnRole,
    ConfigError,
    DescriptionOwner,
    RoleOwner,
    SensitivityClassification,
    TableKind,
)
from ._contracts_schema import CatalogStructuralConstraintsIndex, ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from ._core_utils import (
    artifact_lock,
    column_has_unknown_value_type,
    cost_cap_active,
    debug,
    diagnostic_debug_enabled,
    normalize_column_type,
    normalized_value_overlap_sets,
    notify,
    safe_json_loads,
    stable_json,
)
from ._dialect import Dialect, DialectRegistry
from ._llm_provider import LLMProvider

_pglast_module: Any | None = None
AlterTableStmt: type[Any] | None = None
CreateStmt: type[Any] | None = None
AlterTableType: type[Any] | None = None
ConstrType: type[Any] | None = None
RawStream: type[Any] | None = None

try:
    import pglast as _pglast_imported
    from pglast.ast import AlterTableStmt as _AlterTableStmtCls
    from pglast.ast import CreateStmt as _CreateStmtCls
    from pglast.enums import AlterTableType as _AlterTableTypeCls
    from pglast.enums import ConstrType as _ConstrTypeCls
    from pglast.stream import RawStream as _RawStreamCls

    _pglast_module = _pglast_imported
    AlterTableStmt = _AlterTableStmtCls
    CreateStmt = _CreateStmtCls
    AlterTableType = _AlterTableTypeCls
    ConstrType = _ConstrTypeCls
    RawStream = _RawStreamCls
except ImportError:
    pass

_PG_LAST_SQL_AVAILABLE: bool = _pglast_module is not None


def _pglast_raw_stream_render(node: Any) -> str:
    if RawStream is None:
        raise RuntimeError("pglast is not installed")
    return str(RawStream()(node)).strip()


def resolve_profile_timeout_ms(dialect_or_engine: Any) -> int | None:
    """Return the profiling statement timeout for *dialect_or_engine*, preferring per-dialect overrides."""
    for candidate in (dialect_or_engine, getattr(dialect_or_engine, "dialect", None)):
        if candidate is None:
            continue
        override = getattr(candidate, "profile_timeout_ms", None)
        if override is not None:
            return int(override)
    return PolicyConfig.PROFILE_TIMEOUT_MS


def apply_profile_timeout_to_dialect(dialect: Any, profile_timeout_ms: int | None) -> None:
    """Stamp per-member ``profile_timeout_ms`` onto *dialect* for schema profiling."""
    if profile_timeout_ms is None:
        return
    dialect.profile_timeout_ms = int(profile_timeout_ms)


def _stamp_profile_timeout_from_engine(dialect: Dialect, engine: Any) -> None:
    """Prefer engine- or dialect-level ``profile_timeout_ms`` before profiling a schema graph."""
    for candidate in (engine, getattr(engine, "_dialect", None), dialect):
        if candidate is None:
            continue
        override = getattr(candidate, "profile_timeout_ms", None)
        if override is not None:
            apply_profile_timeout_to_dialect(dialect, int(override))
            return


def _maybe_set_profile_statement_timeout(conn: Any, dialect_or_engine: Any) -> None:
    """Apply the resolved profiling statement timeout via the active dialect when supported."""
    dialect = dialect_or_engine
    if not hasattr(dialect, "profile_statement_timeout_sql"):
        eng = getattr(dialect_or_engine, "dialect", None)
        if eng is not None and hasattr(eng, "profile_statement_timeout_sql"):
            dialect = eng
        else:
            return
    tm = resolve_profile_timeout_ms(dialect_or_engine)
    if tm is None or not cost_cap_active(tm):
        return
    sql = dialect.profile_statement_timeout_sql(int(tm))
    if isinstance(sql, str) and sql.strip():
        conn.execute(text(sql))


def collect_profiling_frequent_values(raw: list[Any] | None) -> list[str]:
    """Return distinct profiling values in first-seen order, capped at ``CATEGORICAL_SAMPLE_SIZE``."""
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    cap = PolicyConfig.CATEGORICAL_SAMPLE_SIZE
    for v in raw:
        if v is None:
            continue
        s = str(v).strip()
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _has_boolean_like_values(col: ColumnMetadata) -> tuple[bool, str | None]:
    """Check if a column's top-K values match a known boolean-like. pattern."""
    if col.distinct_count != 2:
        return False, None
    if not col.frequent_values or len(col.frequent_values) != 2:
        return False, None
    values_lower = frozenset(str(v).lower().strip() for v in col.frequent_values)
    truth_norm = BOOLEAN_TRUTH_PATTERN_MAP.get(values_lower)
    if truth_norm is None:
        return False, None
    for raw in col.frequent_values:
        if str(raw).lower().strip() == truth_norm:
            return True, str(raw).strip()
    return True, truth_norm


def _is_boolean_like_column(col: ColumnMetadata) -> bool:
    """Return True when the column behaves like a two-valued boolean flag."""
    if col.role == ColumnRole.BOOLEAN.value:
        return True
    if "bool" in (col.data_type or "").lower():
        return True
    if col.is_primary_key or col.is_foreign_key:
        return False
    matched, _truth = _has_boolean_like_values(col)
    if matched:
        return True
    return False


def _normalize_profiling_sample_clause(qcol: str, sample_clause: str) -> str:
    """Rewrite base-dialect ``ORDER BY 1`` sampling and ``{col}`` hash predicates."""
    if "{col}" in sample_clause:
        sample_clause = sample_clause.replace("{col}", qcol)
    prefix = "ORDER BY 1 LIMIT "
    if sample_clause.startswith(prefix):
        return f"ORDER BY {qcol} {sample_clause[len('ORDER BY 1 ') :]}"
    return sample_clause


def _new_profiling_deep_query_budget(limit: int | None) -> Any:
    """Return a schema-wide cap on expensive per-column profiling queries."""

    class _ProfilingDeepQueryBudget:
        __slots__ = ("_remaining",)

        def __init__(self, budget_limit: int | None) -> None:
            self._remaining = budget_limit

        def allow(self, cost: int = 1) -> bool:
            if self._remaining is None:
                return True
            if self._remaining < cost:
                return False
            self._remaining -= cost
            return True

    return _ProfilingDeepQueryBudget(limit)


def _resolve_profiling_sample_params(
    dialect: Dialect,
    *,
    use_sample: bool,
    row_count: int,
    sample_size: int,
    table_kind: TableKind,
) -> tuple[str, bool]:
    """Return sampling suffix and subquery flag, with ordered-limit fallback when sampling is required but unsupported."""
    if not use_sample:
        return "", False
    sample_clause = dialect.profiling_stats_sample_suffix(
        use_sample=True,
        row_count=row_count,
        sample_size=sample_size,
        random_seed=DEFAULT_RANDOM_SEED,
        table_kind=table_kind,
    )
    use_subquery = dialect.profiling_stats_use_subquery_when_sampling(table_kind)
    if not sample_clause.strip():
        sample_clause = dialect.profiling_ordered_limit_sample_suffix(sample_size)
        use_subquery = True
    return sample_clause, use_subquery


def _build_profile_stats_sql(qcol: str, qtbl: str, *, use_sample: bool, sample_clause: str, use_subquery: bool) -> str:
    """Build the column statistics query (row count, distinct count, null count)."""
    if use_sample and use_subquery:
        sample_clause = _normalize_profiling_sample_clause(qcol, sample_clause)
    if use_sample and not use_subquery:
        return (
            f"SELECT COUNT(*) as cnt, COUNT(DISTINCT {qcol}) as dist, "
            f"COUNT(*) - COUNT({qcol}) as nulls FROM {qtbl} {sample_clause}"
        )
    if use_sample:
        return (
            f"SELECT COUNT(*) as cnt, COUNT(DISTINCT {qcol}) as dist, "
            f"COUNT(*) - COUNT({qcol}) as nulls "
            f"FROM (SELECT {qcol} FROM {qtbl} {sample_clause}) t"
        )
    return f"SELECT COUNT(*) as cnt, COUNT(DISTINCT {qcol}) as dist, COUNT(*) - COUNT({qcol}) as nulls FROM {qtbl}"


def _build_frequent_values_sql(
    qcol: str, qtbl: str, limit: int, *, sample_clause: str = "", use_subquery: bool = False
) -> str:
    """Build a frequency-ordered value sample query capped at ``limit``."""
    if sample_clause and use_subquery:
        sample_clause = _normalize_profiling_sample_clause(qcol, sample_clause)
        return (
            f"SELECT t.c AS v FROM (SELECT {qcol} AS c FROM {qtbl} {sample_clause}) t "
            f"WHERE t.c IS NOT NULL GROUP BY t.c ORDER BY COUNT(*) DESC LIMIT {limit}"
        )
    suffix = f" {sample_clause}" if sample_clause else ""
    return (
        f"SELECT {qcol} AS v FROM {qtbl}{suffix} WHERE {qcol} IS NOT NULL "
        f"GROUP BY {qcol} ORDER BY COUNT(*) DESC LIMIT {limit}"
    )


def _build_minmax_sql(qcol: str, qtbl: str, *, sample_clause: str = "", use_subquery: bool = False) -> str:
    """Build a min/max aggregate query for numeric or date columns."""
    if sample_clause and use_subquery:
        sample_clause = _normalize_profiling_sample_clause(qcol, sample_clause)
        return f"SELECT MIN(t.c), MAX(t.c) FROM (SELECT {qcol} AS c FROM {qtbl} {sample_clause}) t"
    suffix = f" {sample_clause}" if sample_clause else ""
    return f"SELECT MIN({qcol}), MAX({qcol}) FROM {qtbl}{suffix}"


def _build_mode_sql(qcol: str, qtbl: str, *, sample_clause: str = "", use_subquery: bool = False) -> str:
    """Build a query returning the maximum per-value frequency for a column."""
    if sample_clause and use_subquery:
        sample_clause = _normalize_profiling_sample_clause(qcol, sample_clause)
        return (
            f"SELECT MAX(c) FROM ("
            f"SELECT COUNT(*) AS c FROM (SELECT {qcol} AS c FROM {qtbl} {sample_clause}) t "
            f"WHERE t.c IS NOT NULL GROUP BY t.c) s"
        )
    suffix = f" {sample_clause}" if sample_clause else ""
    return f"SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM {qtbl}{suffix} WHERE {qcol} IS NOT NULL GROUP BY {qcol}) s"


def _build_composite_descriptive_sql(
    dialect: Dialect,
    q1: str,
    q2: str,
    qtbl: str,
    *,
    row_count: int,
    table_kind: TableKind,
    sample_threshold: int | None = None,
    sample_size: int | None = None,
) -> str:
    """Build a bounded composite-descriptive distinct-count query."""
    if sample_threshold is None:
        sample_threshold = PolicyConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = PolicyConfig.PROFILING_SAMPLE_SIZE
    use_sample = row_count > sample_threshold
    sample_clause, use_subquery = _resolve_profiling_sample_params(
        dialect,
        use_sample=use_sample,
        row_count=row_count,
        sample_size=sample_size,
        table_kind=table_kind,
    )
    if use_sample and use_subquery:
        return (
            f"SELECT COUNT(DISTINCT CONCAT(t.c1, ' ', t.c2)) FROM "
            f"(SELECT {q1} AS c1, {q2} AS c2 FROM {qtbl} {sample_clause}) t"
        )
    suffix = f" {sample_clause}" if sample_clause else ""
    return f"SELECT COUNT(DISTINCT CONCAT({q1}, ' ', {q2})) FROM {qtbl}{suffix}"


def _build_value_overlap_sample_sql(
    dialect: Dialect,
    qcol: str,
    qtbl: str,
    limit: int,
    *,
    sample_clause: str = "",
    use_subquery: bool = False,
    fixed_width: bool = False,
) -> str:
    """Build an ascending distinct overlap sample with optional table sampling."""

    def _overlap_cast(expr: str) -> str:
        cast_expr = dialect.profiling_text_cast_sql(expr)
        if fixed_width:
            cast_expr = dialect.render_fixed_width_text_wrap(cast_expr)
        return cast_expr

    cast_expr = _overlap_cast("t.c")
    if sample_clause and use_subquery:
        sample_clause = _normalize_profiling_sample_clause(qcol, sample_clause)
        return (
            f"SELECT DISTINCT {cast_expr} AS v FROM "
            f"(SELECT {qcol} AS c FROM {qtbl} {sample_clause}) t "
            f"WHERE t.c IS NOT NULL ORDER BY v ASC LIMIT {limit}"
        )
    overlap_cast = _overlap_cast(qcol)
    return (
        f"SELECT DISTINCT {overlap_cast} AS v FROM {qtbl} {sample_clause} "
        f"WHERE {qcol} IS NOT NULL ORDER BY v ASC LIMIT {limit}"
    )


def _column_low_cardinality_catalog_eligible(col: ColumnMetadata, row_count: int) -> bool:
    """Return True when a column's distinct values fit the full-value catalog limits."""
    if col.distinct_count <= 0 or row_count <= 0:
        return False
    if col.distinct_count <= PolicyConfig.LOW_CARDINALITY_FULL_VALUES_LIMIT:
        return True
    ratio = col.distinct_count / row_count
    return bool(ratio <= PolicyConfig.LOW_CARDINALITY_DISTINCT_RATIO)


def _build_array_element_distinct_sql(
    dialect: Dialect, qcol: str, qtbl: str, limit: int, *, sample_clause: str = "", use_subquery: bool = False
) -> str | None:
    """Build a one-time DISTINCT query over unnested array elements when supported."""
    engine_type = str(getattr(dialect, "engine_type", "") or "").lower()
    if engine_type not in ("postgresql", "duckdb", "redshift"):
        return None
    elem_alias = "elem"
    cast_expr = dialect.profiling_text_cast_sql(f"u.{elem_alias}")
    if sample_clause and use_subquery:
        sample_clause = _normalize_profiling_sample_clause(qcol, sample_clause)
        from_clause = f"(SELECT {qcol} AS c FROM {qtbl} {sample_clause}) t"
        unnest_ref = "t.c"
    else:
        from_clause = f"{qtbl} t"
        unnest_ref = f"t.{qcol}"
    return (
        f"SELECT DISTINCT {cast_expr} AS v FROM {from_clause}, "
        f"LATERAL unnest({unnest_ref}) AS u({elem_alias}) "
        f"WHERE u.{elem_alias} IS NOT NULL ORDER BY v ASC LIMIT {limit}"
    )


def _column_is_binary_value_type(col: ColumnMetadata) -> bool:
    """Return True when profiling must not read stored byte payloads."""
    return (col.value_type or "").strip().lower() == "binary"


def _column_is_unknown_value_type(col: ColumnMetadata) -> bool:
    """Return True when the column type is not mapped to a known value- type bucket."""
    return column_has_unknown_value_type(col)


def _mark_unknown_column_profile_skipped(col: ColumnMetadata, row_count: int) -> None:
    """Record an unknown-typed column as profiled without value sampling."""
    col.profile_skipped_reason = "unknown"
    col.row_count = row_count
    col.distinct_count = 0
    col.distinct_ratio = None
    col.null_ratio = None
    col.distinct_from_sample = False
    col.min_val = None
    col.max_val = None
    col.frequent_values = []
    col.value_overlap_sample = []
    col.mode_frequency_ratio = 0.0


def _mark_binary_column_profile_skipped(col: ColumnMetadata, row_count: int) -> None:
    """Record a binary column as profiled without scanning stored bytes."""
    col.profile_skipped_reason = "binary"
    col.row_count = row_count
    col.distinct_count = 0
    col.distinct_ratio = None
    col.null_ratio = None
    col.distinct_from_sample = False
    col.min_val = None
    col.max_val = None
    col.frequent_values = []
    col.value_overlap_sample = []
    col.mode_frequency_ratio = 0.0
    col.profile_failed = False


def _column_profiling_excluded(col: ColumnMetadata) -> bool:
    """Return True when profiling must not read column values from the database."""
    return col.sensitivity != SensitivityClassification.NONE


def _apply_collation_overlap_semantics(dialect: Dialect, col: ColumnMetadata) -> None:
    """Resolve per-column collation semantics used by overlap comparison."""
    col.is_case_insensitive_collation = dialect.column_is_case_insensitive_collation(col)
    col.overlap_comparison = "case_folded" if col.is_case_insensitive_collation else "exact"


def _profile_column(
    dialect: Dialect,
    engine: Any,
    col: ColumnMetadata,
    table_name: str,
    row_count: int,
    sample_threshold: int | None = None,
    sample_size: int | None = None,
    *,
    table_kind: TableKind = TableKind.TABLE,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile a single column and update its metadata in-place."""
    if _column_profiling_excluded(col):
        debug(f"[schema_profiling.profile_column] skipping sensitive column {table_name}.{col.name}")
        return
    if _column_is_binary_value_type(col):
        _mark_binary_column_profile_skipped(col, row_count)
        debug(f"[schema_profiling.profile_column] skipping binary column {table_name}.{col.name}")
        return
    if _column_is_unknown_value_type(col):
        _mark_unknown_column_profile_skipped(col, row_count)
        debug(f"[schema_profiling.profile_column] skipping unknown-type column {table_name}.{col.name}")
        return
    debug(f"[schema_profiling.profile_column] profiling {table_name}.{col.name}")
    if sample_threshold is None:
        sample_threshold = PolicyConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = PolicyConfig.PROFILING_SAMPLE_SIZE
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(None)

    col.row_count = row_count
    _apply_collation_overlap_semantics(dialect, col)
    use_sample = row_count > sample_threshold
    qcol = dialect.quote_identifier(col.name)
    qtbl = dialect.qualified_table_ref(table_name, kind=table_kind)

    try:
        with engine.connect() as conn:
            _maybe_set_profile_statement_timeout(conn, dialect)
            sample_clause, use_subquery = _resolve_profiling_sample_params(
                dialect,
                use_sample=use_sample,
                row_count=row_count,
                sample_size=sample_size,
                table_kind=table_kind,
            )

            stats_sql = _build_profile_stats_sql(
                qcol, qtbl, use_sample=use_sample, sample_clause=sample_clause, use_subquery=use_subquery
            )

            result = conn.execute(text(stats_sql)).fetchone()
            cnt = int(result[0] or 0)
            dist = int(result[1] or 0)
            nulls = int(result[2] or 0)

            col.distinct_count = dist
            col.distinct_ratio = dist / cnt if cnt > 0 else None
            col.null_ratio = nulls / cnt if cnt > 0 else None
            col.distinct_from_sample = bool(use_sample)

            if (col.value_type in ("integer", "number") or col.value_type == "date") and deep_query_budget.allow():
                minmax_sql = _build_minmax_sql(qcol, qtbl, sample_clause=sample_clause, use_subquery=use_subquery)
                minmax_result = conn.execute(text(minmax_sql)).fetchone()
                if minmax_result:
                    col.min_val = str(minmax_result[0]) if minmax_result[0] is not None else None
                    col.max_val = str(minmax_result[1]) if minmax_result[1] is not None else None

            if _column_frequent_values_eligible(col) and deep_query_budget.allow():
                freq_sql = _build_frequent_values_sql(
                    qcol,
                    qtbl,
                    PolicyConfig.CATEGORICAL_SAMPLE_SIZE,
                    sample_clause=sample_clause,
                    use_subquery=use_subquery,
                )
                freq_result = conn.execute(text(freq_sql)).fetchall()
                col.frequent_values = collect_profiling_frequent_values(
                    [row[0] for row in freq_result if row[0] is not None]
                )
            if deep_query_budget.allow():
                mode_sql = _build_mode_sql(qcol, qtbl, sample_clause=sample_clause, use_subquery=use_subquery)
                mode_row = conn.execute(text(mode_sql)).fetchone()
                non_null = max(0, cnt - nulls)
                if mode_row and mode_row[0] is not None and non_null > 0:
                    top_freq = mode_row[0] or 0
                    col.mode_frequency_ratio = float(top_freq) / float(non_null) if top_freq else 0.0
                else:
                    col.mode_frequency_ratio = 0.0

            if _column_value_overlap_eligible(col) and deep_query_budget.allow():
                cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
                if _column_low_cardinality_catalog_eligible(col, cnt):
                    cap = PolicyConfig.LOW_CARDINALITY_FULL_VALUES_LIMIT
                overlap_sql = _build_value_overlap_sample_sql(
                    dialect,
                    qcol,
                    qtbl,
                    cap,
                    sample_clause=sample_clause,
                    use_subquery=use_subquery,
                    fixed_width=col.is_fixed_width_text,
                )
                overlap_rows = conn.execute(text(overlap_sql)).fetchall()
                col.value_overlap_sample = [str(r[0]) for r in overlap_rows if r[0] is not None]

            is_arr, _elt = array_element_type_from_data_type(col.data_type or "")
            if is_arr:
                arr_cap = PolicyConfig.LOW_CARDINALITY_FULL_VALUES_LIMIT
                arr_sql = _build_array_element_distinct_sql(
                    dialect, qcol, qtbl, arr_cap, sample_clause=sample_clause, use_subquery=use_subquery
                )
                if arr_sql:
                    arr_rows = conn.execute(text(arr_sql)).fetchall()
                    elements = [str(r[0]) for r in arr_rows if r[0] is not None]
                    if elements:
                        col.frequent_values = collect_profiling_frequent_values(elements)
                        col.value_overlap_sample = sorted({v for v in elements})
    except Exception as exc:
        _record_column_profile_failure(table_name, col, exc)


def _record_column_profile_failure(table_name: str, col: ColumnMetadata, exc: Exception) -> None:
    col.profile_failed = True
    col.distinct_count = 0
    col.distinct_ratio = 0.0
    col.distinct_from_sample = False
    col.null_ratio = 0.0
    col.frequent_values = []
    col.value_overlap_sample = []
    col.min_val = None
    col.max_val = None
    col.is_unique = False
    col.mode_frequency_ratio = 0.0
    notify(
        f"column profile failed for {table_name}.{col.name}: {exc}",
        stage="schema",
        code=DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED,
        level="warning",
        details=(("table", table_name), ("column", col.name)),
    )


def _column_value_overlap_eligible(col: ColumnMetadata) -> bool:
    """Return True when distinct-value sampling supports overlap checks."""
    if _column_is_binary_value_type(col):
        return False
    if _column_is_unknown_value_type(col):
        return False
    if col.distinct_count <= 0:
        return False
    vt = (col.value_type or "").lower()
    if col.is_primary_key or col.is_foreign_key or vt == "identifier":
        return True
    if col.is_fixed_width_text:
        return False
    if col.distinct_count > PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT * 20:
        return False
    if vt in ("string", "categorical", "free_text", "date"):
        return True
    if vt in ("integer", "number") and col.distinct_count <= PolicyConfig.CATEGORICAL_MAX_CARDINALITY:
        return True
    return False


def _column_frequent_values_eligible(col: ColumnMetadata) -> bool:
    """Return True when frequency-ordered value sampling is useful."""
    if _column_is_binary_value_type(col):
        return False
    if _column_is_unknown_value_type(col):
        return False
    return col.distinct_count > 0


def _composite_descriptive_name_columns(table: TableMetadata) -> list[str]:
    return [
        col_name
        for col_name, col_meta in table.columns.items()
        if (col_meta.value_type or "").lower() == "string"
        and NAME_COLUMN_PATTERN.search(col_name)
        and not col_meta.is_primary_key
        and not col_meta.is_foreign_key
        and not _column_profiling_excluded(col_meta)
    ]


def _mark_composite_descriptive_columns_unprofiled(table: TableMetadata, name_cols: list[str]) -> None:
    table.composite_descriptive_ratios.clear()
    for col_name in name_cols:
        col = table.columns.get(col_name)
        if col is None:
            continue
        col.distinct_count = 0
        col.distinct_ratio = 0.0
        col.distinct_from_sample = False
        col.null_ratio = 0.0
        col.frequent_values = []
        col.value_overlap_sample = []
        col.min_val = None
        col.max_val = None
        col.is_unique = False
        col.mode_frequency_ratio = 0.0
        col.profile_failed = True


def _record_composite_descriptive_profile_failure(
    table: TableMetadata,
    name_cols: list[str],
    exc: Exception,
) -> None:
    _mark_composite_descriptive_columns_unprofiled(table, name_cols)
    notify(
        f"composite descriptive profile failed for table {table.name!r}: {exc}",
        stage="schema",
        code=DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED,
        level="warning",
        details=(("table", table.name),),
    )


def _profile_composite_descriptive(dialect: Dialect, engine: Any, table: TableMetadata) -> None:
    """Compute composite distinct ratios for name-like column pairs."""
    name_cols = _composite_descriptive_name_columns(table)
    if len(name_cols) < 2:
        return
    row_count = table.row_count or 0
    if row_count == 0:
        return

    try:
        full_table = dialect.qualified_table_ref(table.name, kind=table.kind)
        with engine.connect() as conn:
            _maybe_set_profile_statement_timeout(conn, dialect)
            for i in range(len(name_cols)):
                for j in range(i + 1, len(name_cols)):
                    c1, c2 = name_cols[i], name_cols[j]
                    q1 = dialect.quote_identifier(c1)
                    q2 = dialect.quote_identifier(c2)
                    sql = _build_composite_descriptive_sql(
                        dialect,
                        q1,
                        q2,
                        full_table,
                        row_count=row_count,
                        table_kind=table.kind,
                    )
                    composite_distinct = int(conn.execute(text(sql)).scalar() or 0)
                    ratio = composite_distinct / row_count if row_count > 0 else None
                    if ratio is not None:
                        table.composite_descriptive_ratios[(c1, c2)] = ratio
                        debug(
                            f"[schema_profiling._profile_composite_descriptive] "
                            f"{table.name}.({c1}, {c2}) composite_ratio={ratio:.4f}"
                        )
    except Exception as exc:
        _record_composite_descriptive_profile_failure(table, name_cols, exc)


def _profile_table(
    dialect: Dialect,
    engine: Any,
    table: TableMetadata,
    *,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile all columns in a table and update metadata in-place."""
    debug(f"[schema_profiling.profile_table] profiling {table.name} ({len(table.columns)} columns)")
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    with engine.connect() as conn:
        _maybe_set_profile_statement_timeout(conn, dialect)
        full_table = dialect.qualified_table_ref(table.name, kind=table.kind)
        count_sql = f"SELECT COUNT(*) FROM {full_table}"
        row_count = int(conn.execute(text(count_sql)).scalar() or 0)
        table.row_count = row_count

    for col in table.columns.values():
        _profile_column(
            dialect,
            engine,
            col,
            table.name,
            row_count,
            table_kind=table.kind,
            deep_query_budget=deep_query_budget,
        )

    _profile_composite_descriptive(dialect, engine, table)

    debug(f"[schema_profiling.profile_table] completed: {table.name}")


def profile_schema(engine: Any, schema: SchemaGraph, dialect: Dialect) -> None:
    """Profile all tables in a schema and update metadata in-place."""
    _stamp_profile_timeout_from_engine(dialect, engine)
    debug(f"[schema_profiling.profile_schema] profiling {len(schema.tables)} tables")
    deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    total = len(schema.tables)
    for idx, table_name in enumerate(sorted(schema.tables.keys(), key=str.lower), start=1):
        table = schema.tables[table_name]
        notify(
            f"  Profiling [{idx}/{total}] {table.name}",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("table", table.name), ("index", str(idx)), ("total", str(total))),
        )
        _profile_table(dialect, engine, table, deep_query_budget=deep_query_budget)
    debug("[schema_profiling.profile_schema] completed")


def _profile_column_spark(
    spark: Any,
    catalog: str,
    schema_name: str,
    col: ColumnMetadata,
    table_name: str,
    row_count: int,
    sample_threshold: int | None = None,
    sample_size: int | None = None,
    *,
    dialect: Dialect,
    table_kind: TableKind = TableKind.TABLE,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile a single column from a Databricks table via Spark SQL. and. update metadata in-place."""
    if _column_is_binary_value_type(col):
        _mark_binary_column_profile_skipped(col, row_count)
        debug(f"[schema_profiling.profile_column_spark] skipping binary column {table_name}.{col.name}")
        return
    if _column_is_unknown_value_type(col):
        _mark_unknown_column_profile_skipped(col, row_count)
        debug(f"[schema_profiling.profile_column_spark] skipping unknown-type column {table_name}.{col.name}")
        return
    debug(f"[schema_profiling.profile_column_spark] profiling {table_name}.{col.name}")
    if sample_threshold is None:
        sample_threshold = PolicyConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = PolicyConfig.PROFILING_SAMPLE_SIZE
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(None)

    col.row_count = row_count
    _apply_collation_overlap_semantics(dialect, col)
    use_sample = row_count > sample_threshold
    qcol = dialect.quote_identifier(col.name)

    try:
        full_table = dialect.qualified_table_ref(table_name, kind=table_kind)
        sample_clause, use_subquery = _resolve_profiling_sample_params(
            dialect,
            use_sample=use_sample,
            row_count=row_count,
            sample_size=sample_size,
            table_kind=table_kind,
        )

        if use_sample:
            stats_sql = f"""
                SELECT
                    COUNT(*) as cnt,
                    COUNT(DISTINCT {qcol}) as dist,
                    COUNT(*) - COUNT({qcol}) as nulls
                FROM {full_table} {sample_clause}
            """
        else:
            stats_sql = f"""
                SELECT
                    COUNT(*) as cnt,
                    COUNT(DISTINCT {qcol}) as dist,
                    COUNT(*) - COUNT({qcol}) as nulls
                FROM {full_table}
            """

        result = spark.sql(stats_sql).collect()[0]
        cnt = int(result["cnt"] or 0)
        dist = int(result["dist"] or 0)
        nulls = int(result["nulls"] or 0)

        col.distinct_count = dist
        col.distinct_ratio = dist / cnt if cnt > 0 else None
        col.null_ratio = nulls / cnt if cnt > 0 else None
        col.distinct_from_sample = bool(use_sample)

        if (col.value_type in ("integer", "number") or col.value_type == "date") and deep_query_budget.allow():
            minmax_sql = _build_minmax_sql(qcol, full_table, sample_clause=sample_clause, use_subquery=use_subquery)
            minmax_result = spark.sql(minmax_sql).collect()[0]
            if minmax_result:
                col.min_val = str(minmax_result[0]) if minmax_result[0] is not None else None
                col.max_val = str(minmax_result[1]) if minmax_result[1] is not None else None

        if _column_frequent_values_eligible(col) and deep_query_budget.allow():
            freq_sql = _build_frequent_values_sql(
                qcol,
                full_table,
                PolicyConfig.CATEGORICAL_SAMPLE_SIZE,
                sample_clause=sample_clause,
                use_subquery=use_subquery,
            )
            freq_result = spark.sql(freq_sql).collect()
            col.frequent_values = collect_profiling_frequent_values(
                [row["v"] for row in freq_result if row["v"] is not None]
            )
        if deep_query_budget.allow():
            mode_sql = _build_mode_sql(qcol, full_table, sample_clause=sample_clause, use_subquery=use_subquery)
            mode_rows = spark.sql(mode_sql).collect()
            non_null = max(0, cnt - nulls)
            if mode_rows and mode_rows[0][0] is not None and non_null > 0:
                top_freq = mode_rows[0][0] or 0
                col.mode_frequency_ratio = float(top_freq) / float(non_null) if top_freq else 0.0
            else:
                col.mode_frequency_ratio = 0.0

        if _column_value_overlap_eligible(col) and deep_query_budget.allow():
            cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
            overlap_sample = f" {sample_clause}" if sample_clause else ""
            cast_expr = dialect.profiling_text_cast_sql(qcol)
            if col.is_fixed_width_text:
                cast_expr = dialect.render_fixed_width_text_wrap(cast_expr)
            sem_sql = (
                f"SELECT DISTINCT {cast_expr} AS v FROM {full_table}{overlap_sample} "
                f"WHERE {qcol} IS NOT NULL ORDER BY v ASC LIMIT {cap}"
            )
            sem_rows = spark.sql(sem_sql).collect()
            col.value_overlap_sample = [str(r["v"]) for r in sem_rows if r.get("v") is not None]
    except Exception as exc:
        _record_column_profile_failure(table_name, col, exc)


def _profile_composite_descriptive_spark(
    spark: Any, catalog: str, schema_name: str, table: TableMetadata, *, dialect: Dialect
) -> None:
    """Compute composite distinct ratios for name-like column pairs via. Spark."""
    name_cols = _composite_descriptive_name_columns(table)
    if len(name_cols) < 2:
        return
    row_count = table.row_count or 0
    if row_count == 0:
        return
    full_table = dialect.qualified_table_ref(table.name)
    try:
        for i in range(len(name_cols)):
            for j in range(i + 1, len(name_cols)):
                c1, c2 = name_cols[i], name_cols[j]
                q1 = dialect.quote_identifier(c1)
                q2 = dialect.quote_identifier(c2)
                sql = _build_composite_descriptive_sql(
                    dialect,
                    q1,
                    q2,
                    full_table,
                    row_count=row_count,
                    table_kind=table.kind,
                )
                composite_distinct = int(spark.sql(sql).collect()[0][0] or 0)
                ratio = composite_distinct / row_count if row_count > 0 else None
                if ratio is not None:
                    table.composite_descriptive_ratios[(c1, c2)] = ratio
                    debug(
                        f"[schema_profiling._profile_composite_descriptive_spark] "
                        f"{table.name}.({c1}, {c2}) composite_ratio={ratio:.4f}"
                    )
    except Exception as exc:
        _record_composite_descriptive_profile_failure(table, name_cols, exc)


def _profile_table_spark(
    spark: Any,
    catalog: str,
    schema_name: str,
    table: TableMetadata,
    *,
    dialect: Dialect,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile all columns in a Databricks table via Spark queries."""
    debug(f"[schema_profiling.profile_table_spark] profiling {table.name} ({len(table.columns)} columns)")
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    try:
        full_table = dialect.qualified_table_ref(table.name)
        count_sql = f"SELECT COUNT(*) FROM {full_table}"
        row_count = int(spark.sql(count_sql).collect()[0][0] or 0)
        table.row_count = row_count
    except Exception as e:
        raise ConfigError(f"schema profiling failed for {table.name}: {e}") from e

    for col in table.columns.values():
        _profile_column_spark(
            spark,
            catalog,
            schema_name,
            col,
            table.name,
            row_count,
            dialect=dialect,
            table_kind=table.kind,
            deep_query_budget=deep_query_budget,
        )

    _profile_composite_descriptive_spark(spark, catalog, schema_name, table, dialect=dialect)

    debug(f"[schema_profiling.profile_table_spark] completed: {table.name}")


def profile_schema_spark(spark: Any, catalog: str, schema_name: str, schema: SchemaGraph, *, dialect: Dialect) -> None:
    """Profile all tables in a Databricks schema via Spark queries."""
    debug(f"[schema_profiling.profile_schema_spark] profiling {len(schema.tables)} tables")
    deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    total = len(schema.tables)
    for idx, table_name in enumerate(sorted(schema.tables.keys(), key=str.lower), start=1):
        table = schema.tables[table_name]
        notify(
            f"  Profiling [{idx}/{total}] {table.name}",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("table", table.name), ("index", str(idx)), ("total", str(total))),
        )
        _profile_table_spark(spark, catalog, schema_name, table, dialect=dialect, deep_query_budget=deep_query_budget)
    debug("[schema_profiling.profile_schema_spark] completed")


def _cursor_rows_as_dicts(cursor: Any) -> list[dict[str, Any]]:
    """Convert cursor result rows to a list of dicts keyed by column. name."""
    if not cursor.description:
        return []
    col_names = [d[0] for d in cursor.description]
    return [dict(zip(col_names, row, strict=True)) for row in cursor.fetchall()]


def _trailing_table_identifier(ref: str) -> str:
    """Return the trailing SQL identifier segment from a possibly qualified relation reference."""
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


def _tables_meta_foreign_key_dicts_from_edges(edges: list[FKEdge]) -> list[dict[str, Any]]:
    """Convert :class:`FKEdge` instances into ``tables_meta`` foreign- key dict entries."""
    out: list[dict[str, Any]] = []
    for e in edges:
        out.append(
            {
                "src_cols": list(e.src_cols),
                "dst_table": str(e.dst_table),
                "dst_cols": list(e.dst_cols),
            }
        )
    return out


def _profile_column_sql_connector(
    connection: Any,
    catalog: str,
    schema_name: str,
    col: ColumnMetadata,
    table_name: str,
    row_count: int,
    sample_threshold: int | None = None,
    sample_size: int | None = None,
    *,
    dialect: Dialect,
    table_kind: TableKind = TableKind.TABLE,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile a single column via databricks-sql-connector and update. metadata in-place."""
    if _column_is_binary_value_type(col):
        _mark_binary_column_profile_skipped(col, row_count)
        debug(f"[schema_profiling.profile_column_sql_connector] skipping binary column {table_name}.{col.name}")
        return
    if _column_is_unknown_value_type(col):
        _mark_unknown_column_profile_skipped(col, row_count)
        debug(f"[schema_profiling.profile_column_sql_connector] skipping unknown-type column {table_name}.{col.name}")
        return
    if sample_threshold is None:
        sample_threshold = PolicyConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = PolicyConfig.PROFILING_SAMPLE_SIZE
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(None)
    col.row_count = row_count
    _apply_collation_overlap_semantics(dialect, col)
    use_sample = row_count > sample_threshold
    full_table = dialect.qualified_table_ref(table_name, kind=table_kind)
    sample_clause, use_subquery = _resolve_profiling_sample_params(
        dialect,
        use_sample=use_sample,
        row_count=row_count,
        sample_size=sample_size,
        table_kind=table_kind,
    )
    qcol = dialect.quote_identifier(col.name)
    try:
        with connection.cursor() as cursor:
            if use_sample:
                stats_sql = f"""
                    SELECT
                        COUNT(*) as cnt,
                        COUNT(DISTINCT {qcol}) as dist,
                        COUNT(*) - COUNT({qcol}) as nulls
                    FROM {full_table} {sample_clause}
                """
            else:
                stats_sql = f"""
                    SELECT
                        COUNT(*) as cnt,
                        COUNT(DISTINCT {qcol}) as dist,
                        COUNT(*) - COUNT({qcol}) as nulls
                    FROM {full_table}
                """
            cursor.execute(stats_sql)
            rows = _cursor_rows_as_dicts(cursor)
            stats_cnt = 0
            stats_nulls = 0
            if rows:
                r = rows[0]
                cnt = int(r.get("cnt") or 0)
                dist = int(r.get("dist") or 0)
                nulls = int(r.get("nulls") or 0)
                col.distinct_count = dist
                col.distinct_ratio = dist / cnt if cnt > 0 else None
                col.null_ratio = nulls / cnt if cnt > 0 else None
                col.distinct_from_sample = bool(use_sample)
                stats_cnt = cnt
                stats_nulls = nulls
            if (col.value_type in ("integer", "number") or col.value_type == "date") and deep_query_budget.allow():
                minmax_sql = _build_minmax_sql(qcol, full_table, sample_clause=sample_clause, use_subquery=use_subquery)
                cursor.execute(minmax_sql)
                minmax_rows = _cursor_rows_as_dicts(cursor)
                if minmax_rows:
                    vals = list(minmax_rows[0].values())
                    col.min_val = str(vals[0]) if vals and vals[0] is not None else None
                    col.max_val = str(vals[1]) if len(vals) > 1 and vals[1] is not None else None
            if _column_frequent_values_eligible(col) and deep_query_budget.allow():
                freq_sql = _build_frequent_values_sql(
                    qcol,
                    full_table,
                    PolicyConfig.CATEGORICAL_SAMPLE_SIZE,
                    sample_clause=sample_clause,
                    use_subquery=use_subquery,
                )
                cursor.execute(freq_sql)
                freq_rows = _cursor_rows_as_dicts(cursor)
                col.frequent_values = collect_profiling_frequent_values(
                    [r["v"] for r in freq_rows if r and r.get("v") is not None]
                )
            if deep_query_budget.allow():
                mode_sql = _build_mode_sql(qcol, full_table, sample_clause=sample_clause, use_subquery=use_subquery)
                cursor.execute(mode_sql)
                mode_rows = _cursor_rows_as_dicts(cursor)
                non_null = max(0, stats_cnt - stats_nulls)
                if mode_rows and list(mode_rows[0].values())[0] is not None and non_null > 0:
                    top_freq = list(mode_rows[0].values())[0] or 0
                    col.mode_frequency_ratio = float(top_freq) / float(non_null) if top_freq else 0.0
                else:
                    col.mode_frequency_ratio = 0.0
            if _column_value_overlap_eligible(col) and deep_query_budget.allow():
                cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
                overlap_sample = f" {sample_clause}" if sample_clause else ""
                cast_expr = dialect.profiling_text_cast_sql(qcol)
                if col.is_fixed_width_text:
                    cast_expr = dialect.render_fixed_width_text_wrap(cast_expr)
                sem_sql = (
                    f"SELECT DISTINCT {cast_expr} AS v FROM {full_table}{overlap_sample} "
                    f"WHERE {qcol} IS NOT NULL ORDER BY v ASC LIMIT {cap}"
                )
                cursor.execute(sem_sql)
                sem_rows = _cursor_rows_as_dicts(cursor)
                col.value_overlap_sample = [str(r["v"]) for r in sem_rows if r.get("v") is not None]
    except Exception as exc:
        _record_column_profile_failure(table_name, col, exc)


def _profile_composite_descriptive_sql_connector(
    connection: Any, catalog: str, schema_name: str, table: TableMetadata, *, dialect: Dialect
) -> None:
    """Compute composite distinct ratios for name-like column pairs via. SQL connector."""
    name_cols = _composite_descriptive_name_columns(table)
    if len(name_cols) < 2:
        return
    row_count = table.row_count or 0
    if row_count == 0:
        return
    full_table = dialect.qualified_table_ref(table.name)
    try:
        with connection.cursor() as cursor:
            for i in range(len(name_cols)):
                for j in range(i + 1, len(name_cols)):
                    c1, c2 = name_cols[i], name_cols[j]
                    q1 = dialect.quote_identifier(c1)
                    q2 = dialect.quote_identifier(c2)
                    sql = _build_composite_descriptive_sql(
                        dialect,
                        q1,
                        q2,
                        full_table,
                        row_count=row_count,
                        table_kind=table.kind,
                    )
                    cursor.execute(sql)
                    rows = _cursor_rows_as_dicts(cursor)
                    composite_distinct = int(list(rows[0].values())[0] or 0) if rows and rows[0] else 0
                    ratio = composite_distinct / row_count if row_count > 0 else None
                    if ratio is not None:
                        table.composite_descriptive_ratios[(c1, c2)] = ratio
                        debug(
                            f"[schema_profiling._profile_composite_descriptive_sql_connector] "
                            f"{table.name}.({c1}, {c2}) composite_ratio={ratio:.4f}"
                        )
    except Exception as exc:
        _record_composite_descriptive_profile_failure(table, name_cols, exc)


def _profile_table_sql_connector(
    connection: Any,
    catalog: str,
    schema_name: str,
    table: TableMetadata,
    *,
    dialect: Dialect,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile all columns in a Databricks table via databricks-sql- connector."""
    debug(f"[schema_profiling._profile_table_sql_connector] profiling {table.name}")
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    full_table = dialect.qualified_table_ref(table.name)
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {full_table}")
        rows = _cursor_rows_as_dicts(cursor)
        row_count = int(rows[0].get(list(rows[0].keys())[0], 0) or 0) if rows else 0
    table.row_count = row_count
    for col in table.columns.values():
        _profile_column_sql_connector(
            connection,
            catalog,
            schema_name,
            col,
            table.name,
            row_count,
            dialect=dialect,
            table_kind=table.kind,
            deep_query_budget=deep_query_budget,
        )
    _profile_composite_descriptive_sql_connector(connection, catalog, schema_name, table, dialect=dialect)
    debug(f"[schema_profiling._profile_table_sql_connector] completed: {table.name}")


def profile_schema_sql_connector(
    connection: Any, catalog: str, schema_name: str, schema: SchemaGraph, *, dialect: Dialect
) -> None:
    """Profile all tables in a Databricks schema via databricks-sql- connector."""
    debug(f"[schema_profiling.profile_schema_sql_connector] profiling {len(schema.tables)} tables")
    deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    total = len(schema.tables)
    for idx, table_name in enumerate(sorted(schema.tables.keys(), key=str.lower), start=1):
        table = schema.tables[table_name]
        notify(
            f"  Profiling [{idx}/{total}] {table.name}",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("table", table.name), ("index", str(idx)), ("total", str(total))),
        )
        _profile_table_sql_connector(
            connection, catalog, schema_name, table, dialect=dialect, deep_query_budget=deep_query_budget
        )
    debug("[schema_profiling.profile_schema_sql_connector] completed")


def extract_tables_from_catalog_sql_connector(
    connection: Any,
    catalog: str,
    schema: str,
    *,
    allow_objects: frozenset[str] | None = None,
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> dict[str, dict[str, Any]]:
    """Extract full table metadata from a Databricks Unity Catalog. schema via SQL connector."""
    tables = {}
    allow_lower = frozenset(str(x).lower() for x in allow_objects) if allow_objects else None
    with connection.cursor() as cursor:
        cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
        table_rows = _cursor_rows_as_dicts(cursor)
    table_col = "tableName"
    if table_rows and table_rows[0]:
        row0 = table_rows[0]
        if "tableName" in row0:
            table_col = "tableName"
        elif "tablename" in row0:
            table_col = "tablename"
        else:
            table_col = list(row0.keys())[0]
    for row in table_rows or []:
        table_name = row.get(table_col) if row else None
        if not table_name:
            continue
        if allow_lower is not None and str(table_name).lower() not in allow_lower:
            continue
        full_table = f"{catalog}.{schema}.{table_name}"
        debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] extracting: {full_table}")
        with connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE TABLE {full_table}")
            cols = _cursor_rows_as_dicts(cursor)
        column_names = []
        column_types = []
        column_is_nullable: list[bool] = []
        null_map = structural_constraints_index.column_nullability.get(str(table_name).lower(), {})
        for col in cols:
            cname = col.get("col_name") or col.get("colname")
            if not cname or str(cname).startswith("#"):
                break
            column_names.append(cname)
            column_types.append(col.get("data_type") or "STRING")
            if str(cname) in null_map:
                column_is_nullable.append(bool(null_map[str(cname)]))
            else:
                column_is_nullable.append(True)
        bundle = structural_constraints_index.tables.get(str(table_name).lower())
        use_info_schema = bundle is not None and bool(
            bundle.primary_keys or bundle.foreign_keys or bundle.unique_columns
        )
        if use_info_schema and bundle is not None:
            primary_keys = list(bundle.primary_keys)
            foreign_keys = _tables_meta_foreign_key_dicts_from_edges(bundle.foreign_keys)
            unique_columns = list(bundle.unique_columns)
        else:
            primary_keys = []
            foreign_keys = []
            unique_columns = []
        partition_columns: list[str] = []
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SHOW CREATE TABLE {full_table}")
                create_rows = _cursor_rows_as_dicts(cursor)
            if create_rows:
                row0 = create_rows[0] or {}
                stmt = row0.get("createtab_stmt") or (list(row0.values())[0] if row0 else None)
                if stmt:
                    partition_columns = _parse_partition_columns_from_create_stmt(stmt)
                    if not use_info_schema:
                        pk_dd, fk_dd, uq_dd = _parse_catalog_constraints_from_ddl(stmt)
                        primary_keys = pk_dd
                        foreign_keys = fk_dd
                        unique_columns = uq_dd
                    debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] ddl_found: {full_table}")
        except Exception as e:
            debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] ddl_error: {full_table} {e}")

        if not partition_columns:
            partition_columns = _extract_partition_columns_from_describe_detail_sql_connector(connection, full_table)

        properties = {}
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SHOW TBLPROPERTIES {full_table}")
                prop_rows = _cursor_rows_as_dicts(cursor)
            for r in prop_rows or []:
                k = r.get("key")
                v = r.get("value")
                if k is not None:
                    properties[k] = v
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        table_comment = None
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE TABLE EXTENDED {full_table}")
                ext_rows = _cursor_rows_as_dicts(cursor)
            for r in ext_rows or []:
                cname = r.get("col_name") or r.get("colname")
                if cname == "Comment":
                    table_comment = r.get("data_type")
                    break
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        tables[table_name] = {
            "table_name_original": table_name,
            "column_names_original": column_names,
            "column_types": column_types,
            "column_is_nullable": column_is_nullable,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "unique_columns": unique_columns,
            "partition_columns": partition_columns,
            "table_comment": table_comment,
            "properties": properties,
        }
    debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] complete: {len(tables)} tables")
    return tables


def apply_catalog_descriptions_from_tables_meta(
    schema: SchemaGraph,
    tables_meta: dict[str, dict[str, Any]],
) -> None:
    """Apply catalog table and column comments through the description precedence ladder."""
    for table_name, meta in tables_meta.items():
        table = schema.tables.get(table_name)
        if table is None:
            continue
        table_comment = meta.get("table_comment")
        if table_comment is not None and str(table_comment).strip():
            DescriptionOwner.set_on(table, str(table_comment).strip(), DescriptionOwner.CATALOG)
        column_comments = meta.get("column_comments") or {}
        if not isinstance(column_comments, dict):
            continue
        for col_name, col_comment in column_comments.items():
            col = table.columns.get(str(col_name))
            if col is None:
                continue
            if col_comment is None or not str(col_comment).strip():
                continue
            DescriptionOwner.set_on(col, str(col_comment).strip(), DescriptionOwner.CATALOG)


def _enrich_fk_column_descriptions(schema: SchemaGraph) -> None:
    """Append navigational hints to FK column descriptions."""
    forbidden_tokens = collect_schema_description_neutrality_forbidden_tokens(schema)
    for table in schema.tables.values():
        for col in table.columns.values():
            if not col.fk_target:
                continue
            dst_table_name, _dst_col = col.fk_target
            dst_table = schema.tables.get(dst_table_name)
            if not dst_table:
                continue
            if dst_table_name.lower() in (col.description or "").lower():
                continue
            descriptive_cols = [
                c.name
                for c in dst_table.columns.values()
                if not c.is_primary_key
                and not c.is_foreign_key
                and c.role not in ("identifier", "")
                and c.name not in forbidden_tokens
            ][:3]
            if not descriptive_cols:
                descriptive_cols = [
                    c.name
                    for c in dst_table.columns.values()
                    if not c.is_primary_key and not c.is_foreign_key and c.name not in forbidden_tokens
                ][:3]
            join_target = sanitize_description_text(dst_table_name, forbidden_tokens) or dst_table_name
            if descriptive_cols:
                suffix = f"join {join_target} for {', '.join(descriptive_cols)}"
            else:
                suffix = f"join {join_target}"
            suffix = sanitize_description_text(suffix, forbidden_tokens)
            if not suffix:
                continue
            existing = (col.description or "").rstrip(". ")
            DescriptionOwner.set_on(col, f"{existing} — {suffix}" if existing else suffix, DescriptionOwner.PROFILE)


def _column_name_suggests_duration(name: str) -> bool:
    """Return True when a column name contains a duration-style token."""
    lowered = (name or "").lower()
    return any(token in lowered for token in DURATION_COLUMN_NAME_TOKENS)


def _column_name_suggests_year(name: str) -> bool:
    """Return True when a column name contains a year-like token."""
    lowered = (name or "").lower()
    return any(token in lowered for token in YEAR_LIKE_COLUMN_NAME_TOKENS)


def _column_name_suggests_date(name: str) -> bool:
    """Return True when a column name suggests a date/time value stored as text."""
    lowered = (name or "").lower()
    if lowered.endswith("_date") or lowered.endswith("_at") or lowered.endswith("_time"):
        return True
    if lowered.endswith("_timestamp") or lowered.startswith("date_"):
        return True
    return any(token in lowered for token in DATE_COLUMN_NAME_TOKENS)


def _infer_column_role(col: ColumnMetadata) -> ColumnRole:
    """Infer a column's role from its metadata using heuristic rules. (fallback)."""
    if col.value_type == "boolean":
        return ColumnRole.BOOLEAN

    if col.is_primary_key:
        return ColumnRole.IDENTIFIER

    if col.is_foreign_key:
        return ColumnRole.IDENTIFIER

    bl_match, _bl_truth = _has_boolean_like_values(col)
    if bl_match:
        return ColumnRole.BOOLEAN

    if col.value_type == "date":
        return ColumnRole.TEMPORAL

    if col.value_type == "string" and _column_name_suggests_date(col.name):
        return ColumnRole.TEMPORAL

    if col.value_type == "integer" and _column_name_suggests_duration(col.name):
        if not _column_name_suggests_year(col.name):
            return ColumnRole.TEMPORAL

    if col.value_type == "integer" and _column_name_suggests_year(col.name):
        return ColumnRole.NUMERIC_CATEGORICAL

    if (
        col.value_type == "string"
        and col.distinct_count is not None
        and col.distinct_count > PolicyConfig.FREE_TEXT_CATEGORICAL_MAX_CARDINALITY
    ):
        return ColumnRole.FREE_TEXT

    if col.distinct_ratio is not None and col.distinct_ratio >= PolicyConfig.IDENTIFIER_MIN_UNIQUENESS:
        return ColumnRole.FREE_TEXT

    is_categorical = col.distinct_count <= PolicyConfig.CATEGORICAL_MAX_CARDINALITY or (
        col.distinct_ratio is not None and col.distinct_ratio <= PolicyConfig.CATEGORICAL_MAX_RATIO
    )
    if is_categorical:
        if col.value_type in ("integer", "number"):
            return ColumnRole.NUMERIC_CATEGORICAL
        return ColumnRole.CATEGORICAL

    if col.value_type in ("integer", "number"):
        return ColumnRole.NUMERIC_MEASURE

    return ColumnRole.FREE_TEXT


def array_element_type_from_data_type(data_type: str) -> tuple[bool, str | None]:
    """Detect array or list column types and return element type when. inferable."""
    s = (data_type or "").strip()
    if not s:
        return False, None
    sl = s.lower()
    if re.fullmatch(r"array\s*$", sl):
        return True, "string"
    if sl.startswith("array<") and s.endswith(">"):
        inner = s[6:-1].strip()
        return True, inner.lower() if inner else "string"
    m = re.match(r"^array\s*\(\s*(.+)\s*\)$", s, re.IGNORECASE)
    if m:
        return True, m.group(1).strip().lower()
    if sl.endswith("[]"):
        return True, sl[:-2].strip().lower() or "string"
    if sl.startswith("list(") and s.endswith(")"):
        inner = s[5:-1].strip()
        return True, inner.lower() if inner else "string"
    if "array" in sl and "[" in sl:
        return True, "string"
    return False, None


def looks_like_json_array_values(values: list[str]) -> bool:
    """Detect array columns stored as JSON text from sampled values. Engines without a native array type store arrays as JSON text (MySQL ``JSON``, SQL Server ``NVARCHAR``), so the catalog type string alone cannot distinguish them from scalar text. This inspects sampled distinct values and returns ``True`` only when every non-empty sample parses as a JSON array, so scalar/object JSON columns (and PostgreSQL array literals like ``{a,b}``) are not misclassified."""
    seen = False
    for raw in values or ():
        s = str(raw).strip()
        if not s:
            continue
        if not (s.startswith("[") and s.endswith("]")):
            return False
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            return False
        if not isinstance(parsed, list):
            return False
        seen = True
    return seen


def _validate_column_classification(col: ColumnMetadata, role: str) -> tuple[list[str], list[str]]:
    """Validate an LLM-assigned column role against profiling data."""
    hard_errors = []
    soft_warnings = []

    if role == ColumnRole.BOOLEAN.value and col.distinct_count and col.distinct_count > 2:
        hard_errors.append(f"{col.name}: BOOLEAN requires distinct_count <= 2, got {col.distinct_count}")

    if role == ColumnRole.CATEGORICAL.value and col.distinct_count and col.distinct_count > 1000:
        soft_warnings.append(f"{col.name}: CATEGORICAL with high cardinality ({col.distinct_count})")

    if role == ColumnRole.NUMERIC_MEASURE.value and col.distinct_count and col.distinct_count <= 5:
        soft_warnings.append(f"{col.name}: NUMERIC_MEASURE with low cardinality ({col.distinct_count})")

    if role == ColumnRole.IDENTIFIER.value and not col.is_primary_key and not col.is_foreign_key:
        soft_warnings.append(f"{col.name}: IDENTIFIER on non-PK/FK column")

    if role == ColumnRole.TEMPORAL.value:
        is_date_like_string = col.value_type == "string" and _column_name_suggests_date(col.name)
        is_duration_integer = col.value_type == "integer" and _column_name_suggests_duration(col.name)
        is_year_integer = col.value_type == "integer" and _column_name_suggests_year(col.name)
        if is_date_like_string:
            soft_warnings.append(f"{col.name}: TEMPORAL on string column with date-like name")
        elif is_duration_integer and is_year_integer:
            soft_warnings.append(f"{col.name}: TEMPORAL on year-like integer column, recommend NUMERIC_CATEGORICAL")
        elif is_year_integer:
            soft_warnings.append(f"{col.name}: TEMPORAL on year-like integer column, recommend NUMERIC_CATEGORICAL")

    return hard_errors, soft_warnings


def _coerce_llm_assigned_role(col: ColumnMetadata, requested_role: str) -> str:
    """Return a role compatible with ``col.value_type``, coercing via profile heuristics when needed."""
    role = (requested_role or "").strip().lower()
    if not role:
        return _infer_column_role(col).value
    value_type = (col.value_type or "").strip().lower()
    if not value_type:
        return role
    allowed = ROLE_VALUE_TYPE_COMPAT.get(role)
    if allowed is None or value_type in allowed:
        return role
    fallback = _infer_column_role(col).value
    notify(
        f"Column {col.name}: role {role!r} incompatible with value_type {value_type!r}; using {fallback!r}.",
        stage="schema",
        code=DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED,
        level="info",
        details=(
            ("requested_role", role),
            ("value_type", value_type),
            ("fallback_role", fallback),
        ),
    )
    return fallback


def _build_column_profile_for_llm(col: ColumnMetadata) -> dict[str, Any]:
    """Build a column profile dict for inclusion in the LLM classification prompt."""
    value_type = (col.value_type or "").strip().lower()
    if not value_type:
        value_type = UNKNOWN_VALUE_TYPE
    profile = {
        "name": col.name,
        "value_type": value_type,
        "is_primary_key": col.is_primary_key,
        "is_foreign_key": col.is_foreign_key,
    }
    hints: dict[str, Any] = {}
    if not _column_is_binary_value_type(col):
        if col.distinct_count is not None:
            hints["distinct_count"] = col.distinct_count
        if col.distinct_ratio is not None:
            hints["distinct_ratio"] = round(col.distinct_ratio, 3)
        if col.null_ratio is not None:
            hints["null_ratio"] = round(col.null_ratio, 3)
    if hints:
        profile["profile_hints"] = hints
    return profile


def _column_usability_diagnostics(col: ColumnMetadata) -> tuple[bool, str]:
    """Return whether the column is LLM-classifiable and a short reason string for tracing."""
    if column_has_unknown_value_type(col):
        return False, "unknown_value_type"
    if col.is_primary_key or col.is_foreign_key:
        return True, "key_column"
    if col.usable_override is True:
        return True, "usable_override"
    if col.distinct_count is not None and col.distinct_count <= 1:
        return False, f"distinct_count={col.distinct_count}<=1"
    if col.null_ratio is not None and col.null_ratio >= PolicyConfig.UNUSABLE_NULL_RATIO_THRESHOLD:
        return False, f"null_ratio={col.null_ratio:.3f}>={PolicyConfig.UNUSABLE_NULL_RATIO_THRESHOLD}"
    if col.mode_frequency_ratio >= PolicyConfig.SENTINEL_MODE_FREQUENCY_THRESHOLD:
        return False, (
            f"mode_frequency_ratio={col.mode_frequency_ratio:.3f}>={PolicyConfig.SENTINEL_MODE_FREQUENCY_THRESHOLD}"
        )
    return True, "profile_ok"


def llm_classification_column_scope(schema: SchemaGraph) -> dict[str, frozenset[str]]:
    """Map each table name to the column names sent to the schema LLM classify pass."""
    scope: dict[str, frozenset[str]] = {}
    for table in schema.tables.values():
        scope[table.name] = frozenset(col.name for col in table.columns.values() if col.is_usable)
    return scope


def _debug_trace_classification_scope(
    schema: SchemaGraph, scope: dict[str, frozenset[str]], *, log_sink: Callable[[str], None] | None = None
) -> None:
    """Log profiling usability and LLM payload scope before schema classification."""
    total_columns = 0
    total_usable = 0
    total_unusable = 0
    for tname in sorted(schema.tables):
        table = schema.tables[tname]
        usable_cols: list[str] = []
        unusable_cols: list[str] = []
        sent_profiles: list[dict[str, Any]] = []
        for col_name in sorted(table.columns):
            col = table.columns[col_name]
            total_columns += 1
            usable, reason = _column_usability_diagnostics(col)
            if usable:
                total_usable += 1
                usable_cols.append(col_name)
                if col_name in scope.get(tname, frozenset()):
                    sent_profiles.append(_build_column_profile_for_llm(col))
            else:
                total_unusable += 1
                unusable_cols.append(f"{col_name}({reason})")
        scope_line = (
            f"  classify scope {tname}: columns={len(table.columns)} "
            f"usable={len(usable_cols)} unusable={len(unusable_cols)} "
            f"llm_payload={sorted(scope.get(tname, frozenset()))}"
        )
        debug(f"[apply_column_roles_llm] scope table={tname} " + scope_line[21:])
        if log_sink is not None:
            log_sink(scope_line)
        if unusable_cols:
            detail = f"  classify scope {tname} unusable: {unusable_cols}"
            debug(f"[apply_column_roles_llm] scope table={tname} unusable_detail={unusable_cols}")
            if log_sink is not None:
                log_sink(detail)
        if sent_profiles:
            debug(f"[apply_column_roles_llm] scope table={tname} llm_column_profiles={stable_json(sent_profiles)}")
    summary = (
        f"  classify scope summary: tables={len(schema.tables)} columns={total_columns} "
        f"usable={total_usable} unusable={total_unusable} llm_columns={sum(len(cols) for cols in scope.values())}"
    )
    debug(f"[apply_column_roles_llm] scope summary tables={len(schema.tables)} " + summary[24:])
    if log_sink is not None:
        log_sink(summary)


def _deterministic_column_description(table_name: str, col: ColumnMetadata, role: str) -> str:
    """Build a fixed column description from role and metadata for non- LLM columns."""
    role_key = (role or _infer_column_role(col).value).strip().lower()
    label = col.name.replace("_", " ")
    if role_key == ColumnRole.IDENTIFIER.value:
        if col.is_foreign_key and col.fk_target:
            dst_table = col.fk_target[0]
            return f"foreign key to {dst_table}"
        return f"{label} identifier for {table_name}"
    if role_key == ColumnRole.TEMPORAL.value:
        if (col.value_type or "").strip().lower() == "integer":
            return f"{label} duration in days on {table_name}"
        return f"{label} date or timestamp on {table_name}"
    if role_key == ColumnRole.BOOLEAN.value:
        return f"{label} yes or no flag on {table_name}"
    if role_key == ColumnRole.NUMERIC_MEASURE.value:
        return f"{label} numeric measure on {table_name}"
    if role_key == ColumnRole.NUMERIC_CATEGORICAL.value:
        return f"{label} numeric category code on {table_name}"
    if role_key == ColumnRole.CATEGORICAL.value:
        return f"{label} category value on {table_name}"
    if role_key == ColumnRole.AUDIT.value:
        return f"{label} audit timestamp on {table_name}"
    if role_key == ColumnRole.FREE_TEXT.value:
        return f"{label} free text on {table_name}"
    vt = (col.value_type or col.data_type or "unknown").strip().lower()
    return f"{label} {vt} column on {table_name}"


def _apply_deterministic_unscoped_column_profiles(
    schema: SchemaGraph, scope: dict[str, frozenset[str]], skip_columns: set[tuple[str, str]]
) -> int:
    """Assign heuristic roles and template descriptions to columns excluded from the LLM pass."""
    filled = 0
    for table in schema.tables.values():
        scoped = scope.get(table.name, frozenset())
        for col in table.columns.values():
            if col.name in scoped or (table.name, col.name) in skip_columns:
                continue
            role = _infer_column_role(col)
            if RoleOwner.can_overwrite(col.role_owner, RoleOwner.PROFILE):
                col.role = role.value
                col.role_owner = RoleOwner.PROFILE
            description = _deterministic_column_description(table.name, col, role.value)
            if DescriptionOwner.set_on(col, description, DescriptionOwner.PROFILE):
                filled += 1
            debug(
                f"[apply_column_roles_llm] deterministic_profile {table.name}.{col.name} "
                f"role={role.value} description={description!r}"
            )
    if filled:
        debug(f"[apply_column_roles_llm] deterministic_profile filled={filled} unscoped columns")
    return filled


def _column_classification_description(classification: Any) -> str:
    """Read the column description string from a raw LLM column classification object."""
    if not isinstance(classification, dict):
        return ""

    return SchemaGraph.scrub_schema_prose_for_prompt(str(classification.get("description", "")).strip())


def _normalize_llm_classification_payload(
    result: dict[str, Any],
) -> dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]]:
    """Convert raw LLM classification JSON into validated internal. tuples."""
    valid_table_roles = {"dimension", "fact", "bridge", "unknown"}
    valid_col_roles = {r.value for r in ColumnRole}
    classifications: dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]] = {}
    for table_name, table_data in result.items():
        if not isinstance(table_data, dict):
            continue
        table_role = table_data.get("table_role", "unknown").lower()
        if table_role not in valid_table_roles:
            table_role = "unknown"
        description = str(table_data.get("description", "")).strip()

        description = SchemaGraph.scrub_schema_prose_for_prompt(description)
        columns_data = table_data.get("columns", {})
        column_classifications: dict[str, tuple[str, str, str | None]] = {}
        for col_name, classification in columns_data.items():
            sensitivity: str | None = None
            if isinstance(classification, dict):
                role = classification.get("role", "").lower()
                col_description = _column_classification_description(classification)
                sens_raw = classification.get("sensitivity")
                if sens_raw is not None and str(sens_raw).strip():
                    sv = str(sens_raw).strip().lower()
                    if sv in VALID_SENSITIVITY_LEVELS:
                        sensitivity = sv
            else:
                role = str(classification).lower()
                col_description = ""
            if role not in valid_col_roles:
                role = ColumnRole.FREE_TEXT.value
            column_classifications[col_name] = (role, col_description, sensitivity)
        classifications[table_name] = (table_role, description, column_classifications)
    return classifications


def _merge_classification_payloads(base: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    """Merge refine-stage JSON over base-stage JSON without dropping. base tables or columns."""
    merged = dict(base)
    for tname, tdata in refined.items():
        if not isinstance(tdata, dict) or tname not in merged or not isinstance(merged[tname], dict):
            continue
        mt = dict(merged[tname])
        tr = tdata.get("table_role")
        if isinstance(tr, str) and tr.strip():
            mt["table_role"] = tr
        desc = tdata.get("description")
        if isinstance(desc, str) and desc.strip():
            mt["description"] = desc
        base_cols = mt.get("columns")
        if not isinstance(base_cols, dict):
            base_cols = {}
        ref_cols = tdata.get("columns")
        if isinstance(ref_cols, dict):
            for cname, cdata in ref_cols.items():
                if cname not in base_cols:
                    continue
                if isinstance(cdata, dict) and isinstance(base_cols.get(cname), dict):
                    merged_col = dict(base_cols[cname])
                    for key, val in cdata.items():
                        if key == "description" and isinstance(val, str) and not val.strip():
                            continue
                        merged_col[key] = val
                    base_cols[cname] = merged_col
                else:
                    base_cols[cname] = cdata
        mt["columns"] = base_cols
        merged[tname] = mt
    return merged


def emit_description_enrichment_failed(scope: str, exc: Exception) -> None:
    """Surface a failed LLM description refresh so it is not mistaken for a no-op."""
    notify(
        f"description enrichment failed ({scope}): {exc}",
        stage="schema",
        code=DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED,
        level="warning",
        details=(("scope", scope),),
    )


def emit_description_enrichment_noop(scope: str) -> None:
    """Surface a successful classify pass that produced no description updates."""
    notify(
        f"description enrichment produced no updates ({scope})",
        stage="schema",
        code=DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP,
        level="info",
        details=(("scope", scope),),
    )


def emit_schema_fk_catalog_absent_warning(dialect_name: str) -> None:
    """Warn when catalog reflection yields no foreign keys and join edges may be inferred."""
    notify(
        f"{dialect_name}: catalog exposes no foreign keys; join graph edges may be inferred from samples "
        "or operator overrides rather than enforced catalog metadata.",
        stage="schema",
        code=DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT,
        level="warning",
        details=(("engine", dialect_name),),
    )


def emit_schema_unknown_type_unusable_warnings(schema: SchemaGraph) -> None:
    """Emit construction diagnostics for columns whose physical type is unsupported."""
    for tname in sorted(schema.tables):
        table = schema.tables[tname]
        for col_name in sorted(table.columns):
            col = table.columns[col_name]
            if not column_has_unknown_value_type(col):
                continue
            raw_type = (col.data_type or "unknown").strip() or "unknown"
            notify(
                f"{tname}.{col_name}: unsupported column type {raw_type!r} (excluded from LLM scope).",
                stage="schema",
                code=DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE,
                level="warning",
                details=(
                    ("table", tname),
                    ("column", col_name),
                    ("data_type", raw_type),
                    ("value_type", UNKNOWN_VALUE_TYPE),
                ),
            )


def schema_classification_content_hash(
    schema: SchemaGraph,
    notes_content: str | None,
    column_scope: dict[str, frozenset[str]],
) -> str:
    """Return a stable digest over schema-classification inputs for disk cache lookup."""
    notes_hash = hashlib.sha256((notes_content or "").encode("utf-8")).hexdigest()
    payload = {
        "effective_structural_hash": schema.effective_structural_hash or schema.structural_hash or "",
        "profiling_hash": schema.profiling_hash or "",
        "notes_hash": notes_hash,
        "column_scope": {table: sorted(cols) for table, cols in sorted(column_scope.items())},
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _schema_classification_cache_path(content_hash: str) -> Path | None:
    schema_path = (EngineConfig.SCHEMA_JSON_PATH or "").strip()
    if not schema_path:
        return None
    return Path(schema_path).parent / f"schema_classification_{content_hash[:32]}.json"


def _load_schema_classification_cache(content_hash: str) -> dict[str, Any] | None:
    path = _schema_classification_cache_path(content_hash)
    if path is None or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("content_hash") != content_hash:
        return None
    payload = data.get("classifications")
    return payload if isinstance(payload, dict) else None


def _save_schema_classification_cache(content_hash: str, classifications: dict[str, Any]) -> None:
    path = _schema_classification_cache_path(content_hash)
    if path is None:
        return
    body = stable_json({"content_hash": content_hash, "classifications": classifications})
    with artifact_lock(str(path.parent)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(path)


def llm_classify_schema(
    schema: SchemaGraph,
    notes_content: str | None = None,
    *,
    column_scope: dict[str, frozenset[str]] | None = None,
    cache_payload_out: list[dict[str, Any]] | None = None,
) -> dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]]:
    """Classify table roles, column roles, descriptions, and sensitivity using a profiling-driven base pass, then an unconditional second LLM pass (notes-aware when domain notes are present, otherwise cross- table consistency refinement)."""
    scope = column_scope if column_scope is not None else llm_classification_column_scope(schema)
    content_hash = schema_classification_content_hash(schema, notes_content, scope)
    cached_payload = _load_schema_classification_cache(content_hash)
    if cached_payload is not None:
        if cache_payload_out is not None:
            cache_payload_out.append(cached_payload)
        return _normalize_llm_classification_payload(cached_payload)
    tables_data = []
    for table in schema.tables.values():
        scoped_names = scope.get(table.name, frozenset())
        fks = [",".join(fk.src_cols) + "->" + fk.dst_table + "." + ",".join(fk.dst_cols) for fk in table.foreign_keys]
        column_profiles = [
            _build_column_profile_for_llm(table.columns[col_name])
            for col_name in sorted(scoped_names)
            if col_name in table.columns
        ]
        tables_data.append(
            {
                "table": table.name,
                "fks": fks,
                "columns": column_profiles,
            }
        )
    system_base = SCHEMA_CLASSIFY_SYSTEM
    user = stable_json({"tables": tables_data})
    raw_base = LLMProvider.chat(system_base, user, timeout=360.0, task="schema_base")
    base_payload = safe_json_loads(raw_base)
    if not base_payload or not isinstance(base_payload, dict):
        raise ValueError(f"LLM returned invalid JSON for schema classification (base): {raw_base[:200]}")
    notes_stripped = (notes_content or "").strip()
    final_payload: dict[str, Any] = base_payload
    if notes_stripped:
        user_refine = stable_json({"base_classification": base_payload, "domain_notes": notes_stripped})
        raw_refine = LLMProvider.chat(SCHEMA_NOTES_REFINE_SYSTEM, user_refine, timeout=360.0, task="schema")
        refined_payload = safe_json_loads(raw_refine)
        if refined_payload and isinstance(refined_payload, dict):
            final_payload = _merge_classification_payloads(base_payload, refined_payload)
    else:
        user_refine = stable_json({"base_classification": base_payload})
        raw_refine = LLMProvider.chat(SCHEMA_CONSISTENCY_REFINE_SYSTEM, user_refine, timeout=360.0, task="schema")
        refined_payload = safe_json_loads(raw_refine)
        if refined_payload and isinstance(refined_payload, dict):
            final_payload = _merge_classification_payloads(base_payload, refined_payload)
    if cache_payload_out is not None:
        cache_payload_out.append(final_payload)
    return _normalize_llm_classification_payload(final_payload)


def apply_column_roles_llm(
    schema: SchemaGraph,
    notes_content: str | None = None,
    *,
    skip_columns: set[tuple[str, str]] | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """Apply LLM-inferred roles, descriptions, and sensitivity to the. schema in-place."""
    sink_call: Callable[[str], None] = log_sink if log_sink is not None else notify
    skip_columns = skip_columns or set()
    debug(f"[schema_profiling.apply_column_roles_llm] classifying {len(schema.tables)} tables via LLM (base + refine)")
    total_columns = sum(len(table.columns) for table in schema.tables.values())
    debug(f"[schema_profiling.apply_column_roles_llm] total columns: {total_columns}")
    sink_call(
        f"  Classifying {len(schema.tables)} tables / {total_columns} columns via LLM (this can take a minute)..."
    )
    llm_column_scope = llm_classification_column_scope(schema)
    classification_cache_hash = schema_classification_content_hash(schema, notes_content, llm_column_scope)
    _debug_trace_classification_scope(schema, llm_column_scope, log_sink=sink_call)
    role_counts: dict[str, int] = {}
    table_role_counts: dict[str, int] = {}
    llm_success = 0
    success = False
    last_hard_errors: list[str] = []
    max_attempts = PolicyConfig.MAX_ROLE_CLASSIFICATION_RETRIES + 1
    notes_stripped = (notes_content or "").strip()
    description_owner = DescriptionOwner.NOTES if notes_stripped else DescriptionOwner.LLM_REFINEMENT
    for attempt in range(max_attempts):
        cache_payload_holder: list[dict[str, Any]] = []
        try:
            classifications = llm_classify_schema(
                schema,
                notes_content,
                column_scope=llm_column_scope,
                cache_payload_out=cache_payload_holder,
            )
            all_hard_errors = []
            all_soft_warnings = []
            for table in schema.tables.values():
                scoped_cols = llm_column_scope.get(table.name, frozenset())
                if table.name not in classifications:
                    all_hard_errors.append(f"{table.name}: missing from LLM response")
                    continue
                table_role, table_description, column_classifications = classifications[table.name]
                if not (table_description or "").strip():
                    all_hard_errors.append(f"{table.name}: empty table description")
                for col_name in column_classifications:
                    if col_name not in scoped_cols:
                        all_hard_errors.append(f"{table.name}.{col_name}: unexpected in LLM response")
                for col in table.columns.values():
                    if col.name not in scoped_cols:
                        continue
                    if col.name not in column_classifications:
                        all_hard_errors.append(f"{table.name}.{col.name}: missing from LLM response")
                        continue
                    role, col_description, _sens = column_classifications[col.name]
                    if not (col_description or "").strip():
                        all_hard_errors.append(f"{table.name}.{col.name}: empty column description")
                    hard_errors, soft_warnings = _validate_column_classification(col, role)
                    all_hard_errors.extend([f"{table.name}.{e}" for e in hard_errors])
                    all_soft_warnings.extend([f"{table.name}.{w}" for w in soft_warnings])
            for warning in all_soft_warnings:
                debug(f"[apply_column_roles_llm] WARNING: {warning}")
            if all_hard_errors:
                last_hard_errors = all_hard_errors
                debug(
                    f"[apply_column_roles_llm] {len(all_hard_errors)} hard errors "
                    f"(attempt {attempt + 1}/{max_attempts}): {all_hard_errors}"
                )
                continue
            for table in schema.tables.values():
                scoped_cols = llm_column_scope.get(table.name, frozenset())
                if table.name in classifications:
                    table_role, description, column_classifications = classifications[table.name]
                    if RoleOwner.can_overwrite(table.role_owner, RoleOwner.LLM):
                        table.role = table_role
                        table.role_owner = RoleOwner.LLM
                    DescriptionOwner.set_on(table, description, description_owner)
                    table_role_counts[table_role] = table_role_counts.get(table_role, 0) + 1
                    for col in table.columns.values():
                        if col.name not in scoped_cols or col.name not in column_classifications:
                            continue
                        if (table.name, col.name) in skip_columns:
                            continue
                        role, col_description, sensitivity = column_classifications[col.name]
                        requested_role = role
                        role = _coerce_llm_assigned_role(col, role)
                        role_owner = RoleOwner.PROFILE if role != requested_role else RoleOwner.LLM
                        if RoleOwner.can_overwrite(col.role_owner, role_owner):
                            col.role = role
                            col.role_owner = role_owner
                        DescriptionOwner.set_on(col, col_description, description_owner)
                        SensitivityClassification.apply_to(col, sensitivity)
                        if col.is_primary_key or col.is_foreign_key:
                            if col.role != ColumnRole.IDENTIFIER.value and RoleOwner.can_overwrite(
                                col.role_owner, RoleOwner.PK_FK_COERCION
                            ):
                                debug(
                                    f"[apply_column_roles_llm] override {table.name}.{col.name}: {col.role} → identifier (pk/fk)"
                                )
                                col.role = ColumnRole.IDENTIFIER.value
                                col.role_owner = RoleOwner.PK_FK_COERCION
                        else:
                            bl_pat, bl_truth = _has_boolean_like_values(col)
                            if bl_pat:
                                if col.role != ColumnRole.BOOLEAN.value and RoleOwner.can_overwrite(
                                    col.role_owner, RoleOwner.BOOLEAN_COERCION
                                ):
                                    debug(
                                        f"[apply_column_roles_llm] override {table.name}.{col.name}: {col.role} → boolean (value pattern)"
                                    )
                                    col.role = ColumnRole.BOOLEAN.value
                                    col.role_owner = RoleOwner.BOOLEAN_COERCION
                                    col.boolean_truth_value = bl_truth
                            elif (
                                col.value_type == "date"
                                and col.role != ColumnRole.AUDIT.value
                                and col.role != ColumnRole.TEMPORAL.value
                                and RoleOwner.can_overwrite(col.role_owner, RoleOwner.PROFILE)
                            ):
                                debug(
                                    f"[apply_column_roles_llm] override {table.name}.{col.name}: {col.role} → temporal (date type)"
                                )
                                col.role = ColumnRole.TEMPORAL.value
                                col.role_owner = RoleOwner.PROFILE
                            elif (
                                col.value_type == "integer"
                                and _column_name_suggests_duration(col.name)
                                and not _column_name_suggests_year(col.name)
                                and col.role not in (ColumnRole.TEMPORAL.value, ColumnRole.IDENTIFIER.value)
                                and RoleOwner.can_overwrite(col.role_owner, RoleOwner.PROFILE)
                            ):
                                debug(
                                    f"[apply_column_roles_llm] override {table.name}.{col.name}: {col.role} → temporal (duration integer)"
                                )
                                col.role = ColumnRole.TEMPORAL.value
                                col.role_owner = RoleOwner.PROFILE
                        if col.role is not None:
                            role_counts[col.role] = role_counts.get(col.role, 0) + 1
            success = True
            llm_success = len(schema.tables)
            debug("[apply_column_roles_llm] two-phase classification successful")
            if cache_payload_holder:
                _save_schema_classification_cache(classification_cache_hash, cache_payload_holder[0])
            break
        except Exception as e:
            last_hard_errors = [str(e)]
            debug(f"[apply_column_roles_llm] attempt {attempt + 1}/{max_attempts} failed: {e}")
            continue
    if not success:
        detail_cap = SCHEMA_CLASSIFY_ERROR_DETAIL_CAP
        detail = "; ".join(last_hard_errors[:detail_cap])
        if len(last_hard_errors) > detail_cap:
            detail = f"{detail}; ... ({len(last_hard_errors)} total)"
        role_type_hits = any("BOOLEAN requires" in e for e in last_hard_errors)
        description_hits = any("description" in e.lower() for e in last_hard_errors)
        if role_type_hits and not description_hits:
            category = "role/type compatibility failures"
        elif description_hits and not role_type_hits:
            category = "missing descriptions"
        elif role_type_hits and description_hits:
            category = "role/type compatibility and missing description failures"
        else:
            category = "classification structural failures"
        raise RuntimeError(f"Schema LLM classification failed after {max_attempts} attempt(s); {category}. {detail}")
    _apply_deterministic_unscoped_column_profiles(schema, llm_column_scope, skip_columns)
    debug(f"[apply_column_roles_llm] completed: {llm_success} LLM-classified tables")
    debug(f"[apply_column_roles_llm] table distribution: {table_role_counts}")
    debug(f"[apply_column_roles_llm] column distribution: {role_counts}")
    _enrich_fk_column_descriptions(schema)
    role_value_type_mismatches = 0
    for table in schema.tables.values():
        for col in table.columns.values():
            if (table.name, col.name) in skip_columns:
                continue
            if col.is_primary_key or col.is_foreign_key:
                continue
            current_role = (col.role or "").strip().lower()
            if not current_role:
                continue
            value_type = (col.value_type or "").strip().lower()
            if not value_type:
                continue
            allowed = ROLE_VALUE_TYPE_COMPAT.get(current_role)
            if allowed is None:
                continue
            if value_type in allowed:
                continue
            if col.role_owner == RoleOwner.USER_OVERRIDE:
                rejected_role = current_role
                fallback_role = _infer_column_role(col)
                sink_call(
                    f"User-provided role {rejected_role!r} for column {table.name}.{col.name} "
                    f"is incompatible with value_type {value_type!r}; override discarded."
                )
                debug(
                    f"[apply_column_roles_llm] user override discarded {table.name}.{col.name}: "
                    f"{rejected_role!r} vs {value_type!r} -> {fallback_role.value!r}"
                )
                col.role = fallback_role.value
                col.role_owner = RoleOwner.PROFILE
                role_value_type_mismatches += 1
                continue
            fallback_role = _infer_column_role(col)
            if fallback_role.value == current_role:
                debug(
                    f"[apply_column_roles_llm] role/value_type mismatch no-op {table.name}.{col.name}: "
                    f"{current_role!r} vs {value_type!r} matches heuristic"
                )
                continue
            debug(
                f"[apply_column_roles_llm] role/value_type mismatch {table.name}.{col.name}: "
                f"{current_role!r} vs {value_type!r} -> {fallback_role.value!r}"
            )
            col.role = fallback_role.value
            col.role_owner = RoleOwner.BOOLEAN_COERCION
            role_value_type_mismatches += 1
    if role_value_type_mismatches:
        debug(f"[apply_column_roles_llm] role/value_type mismatches corrected: {role_value_type_mismatches}")


def schema_name_dump_for_business_knowledge(schema: SchemaGraph) -> tuple[str, ...]:
    """Return sorted qualified table and ``table.column`` names for Pass B filtering."""
    names: list[str] = []
    for table_name, table in sorted(schema.tables.items()):
        names.append(str(table_name))
        for col_name in sorted(table.columns.keys()):
            names.append(f"{table_name}.{col_name}")
    return tuple(names)


def filter_schema_anchored_business_knowledge(
    entries: Sequence[BusinessKnowledgeEntry],
    schema: SchemaGraph,
) -> tuple[BusinessKnowledgeEntry, ...]:
    """Drop entries whose key or text matches a schema table or ``table.column`` name."""
    name_set = {n.lower() for n in schema_name_dump_for_business_knowledge(schema)}
    kept: list[BusinessKnowledgeEntry] = []
    for entry in entries:
        key_l = entry.key.lower()
        text_l = entry.text.lower()
        anchored = False
        for name in name_set:
            if name == key_l or name in text_l or f" {name} " in f" {text_l} ":
                anchored = True
                break
            if "." not in name and re.search(rf"\b{re.escape(name)}\b", text_l):
                anchored = True
                break
        if not anchored:
            kept.append(entry)
    return tuple(kept)


def extract_business_knowledge_from_notes(
    notes_content: str | None,
    schema: SchemaGraph,
) -> tuple[BusinessKnowledgeEntry, ...]:
    """Pass B: extract business-knowledge entries from notes, dropping schema-anchored lines. When notes are empty or credentials are unavailable, returns an empty tuple (version-0 BK)."""
    notes_stripped = (notes_content or "").strip()
    if not notes_stripped:
        return ()
    if not EngineConfig.llm_credentials_configured():
        return ()
    schema_names = list(schema_name_dump_for_business_knowledge(schema))
    user_payload = stable_json({"domain_notes": notes_stripped, "schema_names": schema_names})
    raw = LLMProvider.chat(BUSINESS_KNOWLEDGE_NOTES_EXTRACT_SYSTEM, user_payload, timeout=360.0, task="schema")
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, list):
        return ()
    allowed_kinds = {member.value for member in BusinessKnowledgeKind}
    candidates: list[BusinessKnowledgeEntry] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        text_val = str(item.get("text") or "").strip()
        kind = (
            str(item.get("kind") or BusinessKnowledgeKind.GLOSSARY.value).strip()
            or BusinessKnowledgeKind.GLOSSARY.value
        )
        if not key or not text_val or key in seen:
            continue
        if kind not in allowed_kinds:
            kind = BusinessKnowledgeKind.GLOSSARY.value
        try:
            entry = BusinessKnowledgeEntry.normalize(BusinessKnowledgeEntry(key=key, text=text_val, kind=kind))
        except ConfigError:
            continue
        seen.add(entry.key)
        candidates.append(entry)
    return filter_schema_anchored_business_knowledge(tuple(candidates), schema)


def _strip_leading_articles(value: str) -> str:
    """Lowercase strip and remove leading ``a`` / ``an`` prefixes for comparison."""
    s = value.lower().strip()
    for prefix in BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix) :].strip()
    return s


def _coerce_zero_one_column(col: ColumnMetadata) -> bool:
    """Return True if column was coerced to boolean from numeric or string ``0``/``1``."""
    if col.distinct_count != 2 or not col.frequent_values or len(col.frequent_values) != 2:
        return False
    bucket: set[str] = set()
    for v in col.frequent_values:
        s = str(v).strip().lower()
        if s in ("0", "0.0"):
            bucket.add("z")
        elif s in ("1", "1.0"):
            bucket.add("o")
        else:
            return False
    if bucket != {"z", "o"}:
        return False
    if RoleOwner.can_overwrite(col.role_owner, RoleOwner.BOOLEAN_COERCION):
        col.role = ColumnRole.BOOLEAN.value
        col.role_owner = RoleOwner.BOOLEAN_COERCION
        for v in col.frequent_values:
            s = str(v).strip().lower()
            if s in ("1", "1.0"):
                col.boolean_truth_value = str(v).strip()
                break
    return True


def _coerce_antonym_pair_column(col: ColumnMetadata) -> bool:
    """Return True if two string values match configured negation affix rules."""
    if col.distinct_count != 2 or not col.frequent_values or len(col.frequent_values) != 2:
        return False
    raw_a, raw_b = col.frequent_values[0], col.frequent_values[1]
    a = _strip_leading_articles(str(raw_a))
    b = _strip_leading_articles(str(raw_b))
    if len(a) < BOOLEAN_ANTONYM_MIN_STEM_LEN or len(b) < BOOLEAN_ANTONYM_MIN_STEM_LEN:
        return False
    for prefix in BOOLEAN_NEGATION_PREFIXES:
        if len(a) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and b == prefix + a:
            if RoleOwner.can_overwrite(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_a).strip()
            return True
        if len(b) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and a == prefix + b:
            if RoleOwner.can_overwrite(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_b).strip()
            return True
    for suffix in BOOLEAN_NEGATION_SUFFIXES:
        if len(a) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and b == a + suffix:
            if RoleOwner.can_overwrite(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_a).strip()
            return True
        if len(b) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and a == b + suffix:
            if RoleOwner.can_overwrite(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_b).strip()
            return True
    return False


def apply_boolean_coercion_pass(schema: SchemaGraph) -> None:
    """Deterministically promote two-value columns to boolean using. literals and affix rules."""
    for table in schema.tables.values():
        for col in table.columns.values():
            if col.is_primary_key or col.is_foreign_key:
                continue
            bl_pat, bl_truth = _has_boolean_like_values(col)
            if bl_pat:
                if col.role != ColumnRole.BOOLEAN.value and RoleOwner.can_overwrite(
                    col.role_owner, RoleOwner.BOOLEAN_COERCION
                ):
                    col.role = ColumnRole.BOOLEAN.value
                    col.role_owner = RoleOwner.BOOLEAN_COERCION
                    col.boolean_truth_value = bl_truth
                continue
            if _coerce_zero_one_column(col):
                continue
            if _coerce_antonym_pair_column(col):
                continue


def assign_column_ops(schema: SchemaGraph) -> None:
    """Assign valid filter, aggregation, and HAVING ops to each column. based on its final role."""
    debug("[schema_profiling.assign_column_ops] assigning ops to columns")

    null_ops = ["is null", "is not null"]
    string_only_ops = {"like", "ilike", "not like", "not ilike"}
    numeric_only_aggs = {"sum", "avg"}

    for table in schema.tables.values():
        for col in table.columns.values():
            role = col.role
            vt = col.value_type
            string = vt == "string"
            numeric = vt in ("integer", "number")

            if role == ColumnRole.AUDIT.value:
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
                col.valid_aggregations = ["count"]
                if vt == "date":
                    col.valid_where_ops = [
                        "=",
                        "!=",
                        "<",
                        "<=",
                        ">",
                        ">=",
                        "between",
                        "in",
                        "not in",
                    ] + null_ops
                elif string:
                    col.valid_where_ops = [
                        "=",
                        "!=",
                        "in",
                        "not in",
                        "like",
                        "ilike",
                        "not like",
                        "not ilike",
                    ] + null_ops
                else:
                    col.valid_where_ops = ["=", "!=", "in", "not in"] + null_ops
            elif col.is_primary_key or col.is_foreign_key or role == ColumnRole.IDENTIFIER.value:
                col.valid_where_ops = [
                    "=",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "between",
                    "in",
                    "not in",
                ] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.CATEGORICAL.value:
                col.valid_where_ops = [
                    "=",
                    "!=",
                    "in",
                    "not in",
                    "like",
                    "ilike",
                    "not like",
                    "not ilike",
                ] + null_ops
                col.valid_aggregations = ["count", "min", "max"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.NUMERIC_CATEGORICAL.value:
                col.valid_where_ops = [
                    "=",
                    "!=",
                    "in",
                    "not in",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "between",
                ] + null_ops
                col.valid_aggregations = ["count", "min", "max"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.NUMERIC_MEASURE.value:
                col.valid_where_ops = [
                    "=",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "between",
                    "in",
                    "not in",
                ] + null_ops
                col.valid_aggregations = ["sum", "avg", "min", "max", "count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.TEMPORAL.value:
                col.valid_where_ops = [
                    "=",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "between",
                    "in",
                    "not in",
                ] + null_ops
                col.valid_aggregations = ["min", "max", "count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.BOOLEAN.value:
                col.valid_where_ops = ["=", "!=", "in", "not in"] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.FREE_TEXT.value:
                col.valid_where_ops = [
                    "like",
                    "ilike",
                    "not like",
                    "not ilike",
                ] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = []
            else:
                col.valid_where_ops = ["=", "!="] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]

            is_arr, elt = array_element_type_from_data_type(col.data_type)
            if not is_arr and looks_like_json_array_values(col.frequent_values):
                is_arr, elt = True, "string"
            if not is_arr and normalize_column_type(col.data_type or "") in JSON_COLUMN_TYPE_TOKENS:
                is_arr, elt = True, "string"
            if is_arr and role != ColumnRole.AUDIT.value:
                col.element_type = elt or "string"
                col.valid_where_ops = ["contains"] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif is_arr:
                col.element_type = elt or "string"

            if not string:
                col.valid_where_ops = [op for op in col.valid_where_ops if op not in string_only_ops]
            if not numeric:
                col.valid_aggregations = [agg for agg in col.valid_aggregations if agg not in numeric_only_aggs]

            if not col.is_filterable:
                if role == ColumnRole.FREE_TEXT.value:
                    pattern_ops = {
                        "like",
                        "ilike",
                        "not like",
                        "not ilike",
                        "is null",
                        "is not null",
                    }
                    col.valid_where_ops = [op for op in col.valid_where_ops if op in pattern_ops]
                else:
                    col.valid_where_ops = []

    debug("[schema_profiling.assign_column_ops] completed")


def extract_tables_from_catalog(
    spark: Any,
    catalog: str,
    schema: str,
    *,
    allow_objects: frozenset[str] | None = None,
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> dict[str, dict[str, Any]]:
    """Extract full table metadata from a Databricks Unity Catalog. schema."""
    tables = {}

    table_list = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()

    allow_lower = frozenset(str(x).lower() for x in allow_objects) if allow_objects else None
    for row in table_list:
        table_name = row["tableName"]
        if allow_lower is not None and str(table_name).lower() not in allow_lower:
            continue
        full_table = f"{catalog}.{schema}.{table_name}"

        debug(f"[schema_profiling.extract_tables_from_catalog] extracting: {full_table}")

        cols = spark.sql(f"DESCRIBE TABLE {full_table}").collect()

        column_names = []
        column_types = []
        column_is_nullable: list[bool] = []
        null_map = structural_constraints_index.column_nullability.get(str(table_name).lower(), {})

        for col in cols:
            col_name = col["col_name"]

            if col_name.startswith("#"):
                break

            column_names.append(col_name)
            column_types.append(col["data_type"])
            if col_name in null_map:
                column_is_nullable.append(bool(null_map[col_name]))
            else:
                column_is_nullable.append(True)

        bundle = structural_constraints_index.tables.get(str(table_name).lower())
        use_info_schema = bundle is not None and bool(
            bundle.primary_keys or bundle.foreign_keys or bundle.unique_columns
        )
        if use_info_schema and bundle is not None:
            primary_keys = list(bundle.primary_keys)
            foreign_keys = _tables_meta_foreign_key_dicts_from_edges(bundle.foreign_keys)
            unique_columns = list(bundle.unique_columns)
        else:
            primary_keys = []
            foreign_keys = []
            unique_columns = []
        partition_columns: list[str] = []
        try:
            create_result = spark.sql(f"SHOW CREATE TABLE {full_table}").collect()
            if create_result:
                create_stmt = create_result[0]["createtab_stmt"]
                partition_columns = _parse_partition_columns_from_create_stmt(create_stmt)
                if not use_info_schema:
                    pk_dd, fk_dd, uq_dd = _parse_catalog_constraints_from_ddl(create_stmt)
                    primary_keys = pk_dd
                    foreign_keys = fk_dd
                    unique_columns = uq_dd
                debug(f"[schema_profiling.extract_tables_from_catalog] ddl_found: {full_table}")
        except Exception as e:
            debug(f"[schema_profiling.extract_tables_from_catalog] ddl_error: {full_table} {e}")

        if not partition_columns:
            partition_columns = _extract_partition_columns_from_describe_detail_spark(spark, full_table)

        try:
            props = spark.sql(f"SHOW TBLPROPERTIES {full_table}").collect()
            properties = {p["key"]: p["value"] for p in props}
        except Exception:
            properties = {}

        try:
            extended = spark.sql(f"DESCRIBE TABLE EXTENDED {full_table}").collect()
            table_comment = None
            for row in extended:
                if row["col_name"] == "Comment":
                    table_comment = row["data_type"]
                    break
        except Exception:
            table_comment = None

        tables[table_name] = {
            "table_name_original": table_name,
            "column_names_original": column_names,
            "column_types": column_types,
            "column_is_nullable": column_is_nullable,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "unique_columns": unique_columns,
            "partition_columns": partition_columns,
            "table_comment": table_comment,
            "properties": properties,
        }

    debug(f"[schema_profiling.extract_tables_from_catalog] complete: {len(tables)} tables")
    return tables


def _extract_partition_columns_from_describe_detail_spark(spark: Any, full_table: str) -> list[str]:
    """Extract partition column names via DESCRIBE DETAIL, with. INFORMATION_SCHEMA fallback."""
    try:
        detail_df = spark.sql(f"DESCRIBE DETAIL {full_table}")
        row = detail_df.collect()
        if row:
            r = row[0]
            cols = r.get("partitionColumns") or r.get("partition_columns")
            if isinstance(cols, list) and cols:
                return [str(c) for c in cols]
    except Exception as e:
        debug(f"[schema_profiling._extract_partition_from_detail] DESCRIBE DETAIL failed: {e}")

    try:
        parts = full_table.split(".")
        if len(parts) >= 3:
            catalog_name, schema_name, table_name = parts[0], parts[1], parts[2]
            info_schema = f"{catalog_name}.information_schema.columns"
            q = f"""
                SELECT column_name FROM {info_schema}
                WHERE table_catalog = '{catalog_name}'
                  AND table_schema = '{schema_name}'
                  AND table_name = '{table_name}'
                  AND partition_ordinal_position IS NOT NULL
                ORDER BY partition_ordinal_position
            """
            info_rows = spark.sql(q).collect()
            return [str(r["column_name"]) for r in info_rows if r.get("column_name")]
    except Exception as e:
        debug(f"[schema_profiling._extract_partition_from_detail] INFORMATION_SCHEMA failed: {e}")

    return []


def _extract_partition_columns_from_describe_detail_sql_connector(connection: Any, full_table: str) -> list[str]:
    """Extract partition column names via DESCRIBE DETAIL, with. INFORMATION_SCHEMA fallback."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE DETAIL {full_table}")
            rows = _cursor_rows_as_dicts(cursor)
        if rows:
            r = rows[0]
            cols = r.get("partitionColumns") or r.get("partition_columns")
            if isinstance(cols, list) and cols:
                return [str(c) for c in cols]
    except Exception as e:
        debug(f"[schema_profiling._extract_partition_sql_connector] DESCRIBE DETAIL failed: {e}")

    try:
        parts = full_table.split(".")
        if len(parts) >= 3:
            catalog_name, schema_name, table_name = parts[0], parts[1], parts[2]
            info_schema = f"{catalog_name}.information_schema.columns"
            q = f"""
                SELECT column_name FROM {info_schema}
                WHERE table_catalog = '{catalog_name}'
                  AND table_schema = '{schema_name}'
                  AND table_name = '{table_name}'
                  AND partition_ordinal_position IS NOT NULL
                ORDER BY partition_ordinal_position
            """
            with connection.cursor() as cursor:
                cursor.execute(q)
                info_rows = _cursor_rows_as_dicts(cursor)
            return [str(r["column_name"]) for r in info_rows if r.get("column_name")]
    except Exception as e:
        debug(f"[schema_profiling._extract_partition_sql_connector] INFORMATION_SCHEMA failed: {e}")

    return []


def _parse_partition_columns_from_create_stmt(create_stmt: str) -> list[str]:
    """Extract partition column names from a CREATE TABLE DDL string."""
    match = re.search(r"PARTITIONED\s+BY\s*\(([^)]+)\)", create_stmt, re.IGNORECASE)
    if not match:
        return []
    return [c.strip().strip("`").strip('"') for c in match.group(1).split(",")]


def _partition_column_names_from_create_ddl(create_stmt_sql: str) -> list[str]:
    """Extract declarative partition columns from Hive-style or. PostgreSQL CREATE DDL."""
    spark_cols = _parse_partition_columns_from_create_stmt(create_stmt_sql)
    if spark_cols:
        return spark_cols
    match = re.search(
        r"\bPARTITION\s+BY\s+(?:RANGE|LIST|HASH)\s*\(\s*([^)]+)\s*\)", create_stmt_sql, re.IGNORECASE | re.DOTALL
    )
    if not match:
        return []
    return [c.strip().strip("`").strip('"') for c in match.group(1).split(",") if c.strip()]


def _parse_catalog_constraints_from_ddl(create_stmt: str) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Parse PRIMARY KEY, FOREIGN KEY, and single-column UNIQUE constraint fragments from DDL text."""
    primary_keys = []
    foreign_keys = []
    unique_columns: list[str] = []

    pk_pattern = r"(?:CONSTRAINT\s+\w+\s+)?PRIMARY\s+KEY\s*\(([^)]+)\)"
    pk_matches = re.findall(pk_pattern, create_stmt, re.IGNORECASE | re.DOTALL)
    for match in pk_matches:
        cols = [c.strip().strip("`").strip('"') for c in match.split(",")]
        primary_keys.extend(cols)

    ref_table_segment = r"((?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))*)"
    fk_pattern = (
        r"(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+" + ref_table_segment + r"\s*\(([^)]+)\)"
    )
    fk_matches = re.findall(fk_pattern, create_stmt, re.IGNORECASE | re.DOTALL)
    for match in fk_matches:
        src_cols = [c.strip().strip("`").strip('"') for c in match[0].split(",")]
        ref_table = _trailing_table_identifier(match[1])
        ref_cols = [c.strip().strip("`").strip('"') for c in match[2].split(",")]

        foreign_keys.append({"src_cols": src_cols, "dst_table": ref_table, "dst_cols": ref_cols})

    for m in re.finditer(r"UNIQUE\s*\(([^)]+)\)", create_stmt, re.IGNORECASE):
        ucols = [c.strip().strip("`").strip('"') for c in m.group(1).split(",") if c.strip()]
        if len(ucols) == 1:
            unique_columns.append(ucols[0])

    return primary_keys, foreign_keys, unique_columns


def _balanced_paren_span(sql_text: str, open_paren_idx: int) -> tuple[int, int] | None:
    """Return ``(open_paren_idx, index_after_closing_paren)`` when. parentheses balance from *open_paren_idx*. Depth scanning does not model nested quotes; unusual DDL with parentheses inside literal defaults may mis-scan."""
    if open_paren_idx < 0 or open_paren_idx >= len(sql_text) or sql_text[open_paren_idx] != "(":
        return None
    depth = 0
    i = open_paren_idx
    while i < len(sql_text):
        ch = sql_text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return open_paren_idx, i + 1
        i += 1
    return None


def _parse_sql_file_regex_reflect(sql_content: str) -> dict[str, dict[str, Any]]:
    """Extract CREATE TABLE shapes using bracket balancing when. structured parsers return nothing."""
    tables: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+|TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"([\w.`\"\[\]]+(?:\.[\w.`\"\[\]]+)?)\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(sql_content):
        raw = m.group(1).strip().strip('`"')
        if "." in raw:
            tname = raw.split(".")[-1].strip('`"')
        else:
            tname = raw
        if not tname:
            continue
        open_idx = m.end() - 1
        span = _balanced_paren_span(sql_content, open_idx)
        if span is None:
            continue
        _o, end_after = span
        inner = sql_content[open_idx + 1 : end_after - 1]
        full_chunk = sql_content[m.start() : end_after]
        tables[tname] = _table_metadata_dict_from_ddl_parts(tname, inner, full_chunk)
    return tables


def _schema_graph_has_structural_foreign_keys(sg: SchemaGraph) -> bool:
    """Return True when any reflected table already carries at least. one. FK edge."""
    for tbl in sg.tables.values():
        if tbl.foreign_keys:
            return True
    return False


def _canonicalize_llm_ddl_table_row(tinfo: dict[str, Any]) -> dict[str, Any]:
    """Map LLM DDL JSON (with optional ``column_not_null`` / ``column_unique`` arrays) to the canonical metadata shape used by deterministic DDL parsing."""
    out = dict(tinfo)
    cols = out.get("column_names_original") or []
    if not isinstance(cols, list):
        cols = []
    col_names = [str(c) for c in cols]
    n = len(col_names)
    nn_raw = out.get("column_not_null")
    uq_raw = out.get("column_unique")
    if isinstance(nn_raw, list):
        nn_bool = [bool(x) for x in nn_raw[:n]]
        while len(nn_bool) < n:
            nn_bool.append(False)
    else:
        nn_bool = [False] * n
    out["column_is_nullable"] = [not x for x in nn_bool]
    unique_columns: list[str] = []
    if isinstance(uq_raw, list):
        for i, flag in enumerate(uq_raw):
            if i >= n:
                break
            try:
                is_uq = bool(flag)
            except (TypeError, ValueError):
                is_uq = False
            if is_uq and i < len(col_names):
                unique_columns.append(col_names[i])
    out["unique_columns"] = list(dict.fromkeys(unique_columns))
    out["column_names_original"] = col_names
    out.pop("column_not_null", None)
    out.pop("column_unique", None)
    if "partition_columns" not in out or not isinstance(out.get("partition_columns"), list):
        out["partition_columns"] = []
    return out


def _parse_sql_file_via_llm(sql_content: str) -> dict[str, dict[str, Any]]:
    """Invoke JSON-mode DDL parsing when deterministic parsers produced. nothing."""
    debug("[schema_profiling.parse_sql_file] structured parsers returned 0 tables, falling back to LLM")

    system = """You are a deterministic SQL parser. Extract CREATE TABLE and ALTER TABLE metadata and output ONLY valid JSON. Be precise and consistent. Follow the output format exactly."""

    user = stable_json(
        {
            "task": "Parse CREATE TABLE and ALTER TABLE ADD COLUMN / ADD CONSTRAINT from the SQL and merge into one metadata dict per table in file order",
            "sql_content": sql_content,
            "output_format": {
                "tables": {
                    "table1": {
                        "table_name_original": "exact table name from SQL (without schema prefix)",
                        "column_names_original": ["col1", "col2"],
                        "column_types": ["TYPE1", "TYPE2"],
                        "column_not_null": [True, False],
                        "column_unique": [True, False],
                        "primary_keys": ["col1"],
                        "foreign_keys": [
                            {
                                "src_cols": ["col1"],
                                "dst_table": "table2",
                                "dst_cols": ["col1"],
                            }
                        ],
                    }
                }
            },
            "rules": [
                "Process statements in file order: each CREATE TABLE starts a table entry; ALTER TABLE updates the matching table",
                "Apply ALTER TABLE ADD COLUMN by appending column name and type",
                "Apply ALTER TABLE ADD CONSTRAINT for PRIMARY KEY, FOREIGN KEY, and single-column UNIQUE",
                "Skip ALTER TABLE when the table was never created",
                "Extract ALL CREATE TABLE statements; merge ALTER TABLE into those tables",
                "Preserve exact table and column names (case-sensitive)",
                "Strip schema prefixes from table names (e.g., 'public.users' → 'users')",
                "Remove quotes from identifiers (e.g., '\"user_id\"' → 'user_id')",
                "Capture PRIMARY KEY constraints (inline, CONSTRAINT in CREATE, and ALTER ADD CONSTRAINT)",
                "Capture FOREIGN KEY constraints with all source/destination columns",
                "Set column_not_null[i] = true when the column has NOT NULL or is part of an inline PRIMARY KEY",
                "Set column_unique[i] = true when the column has UNIQUE (inline or single-column UNIQUE in CREATE TABLE / ALTER TABLE)",
                "Normalize data types to uppercase (e.g., 'varchar(50)' → 'VARCHAR(50)')",
                "Convert SERIAL → INTEGER, BIGSERIAL → BIGINT",
                "Use empty arrays [] for tables with no PKs or FKs",
                "Ignore CHECK constraints, DEFAULT values, and multi-column UNIQUE except as needed for FK/PK",
                "Handle multi-line statements and SQL comments (-- and /* */)",
                "Output ONLY the JSON object, no markdown code blocks, no explanations",
            ],
            "examples": [
                {
                    "input": "CREATE TABLE public.table1 (column1 SERIAL PRIMARY KEY, column2 VARCHAR(100));",
                    "output": {
                        "tables": {
                            "table1": {
                                "table_name_original": "table1",
                                "column_names_original": ["column1", "column2"],
                                "column_types": ["INTEGER", "VARCHAR(100)"],
                                "column_not_null": [True, False],
                                "column_unique": [True, False],
                                "primary_keys": ["column1"],
                                "foreign_keys": [],
                            }
                        }
                    },
                },
                {
                    "input": "CREATE TABLE table2 (column1 INT, column2 INT, FOREIGN KEY (column2) REFERENCES table1(column1));",
                    "output": {
                        "tables": {
                            "table2": {
                                "table_name_original": "table2",
                                "column_names_original": ["column1", "column2"],
                                "column_types": ["INT", "INT"],
                                "column_not_null": [False, False],
                                "column_unique": [False, False],
                                "primary_keys": [],
                                "foreign_keys": [
                                    {
                                        "src_cols": ["column2"],
                                        "dst_table": "table1",
                                        "dst_cols": ["column1"],
                                    }
                                ],
                            }
                        }
                    },
                },
            ],
        }
    )

    response = LLMProvider.chat(system, user, task="ddl")
    parsed = safe_json_loads(response)

    if not isinstance(parsed, dict) or "tables" not in parsed:
        debug("[schema_profiling.parse_sql_file] llm also failed, returning empty")
        return {}

    tables = parsed["tables"]
    debug(f"[schema_profiling.parse_sql_file] llm parsed: {len(tables)} tables")

    if diagnostic_debug_enabled():
        for tname, tinfo in tables.items():
            if isinstance(tinfo, dict):
                debug(
                    f"[schema_profiling.parse_sql_file] table: {tname} cols={len(tinfo.get('column_names_original', []))}"
                )

    if not isinstance(tables, dict):
        return {}

    out_tables: dict[str, dict[str, Any]] = {}
    for tname, tinfo in tables.items():
        if isinstance(tinfo, dict):
            out_tables[str(tname)] = _canonicalize_llm_ddl_table_row(tinfo)
    return out_tables


def parse_sql_file(sql_path: Path, *, reflected_schema: SchemaGraph | None = None) -> dict[str, dict[str, Any]]:
    """Parse CREATE TABLE and in-order ALTER TABLE DDL from a SQL file, with conditional LLM fallback."""
    with open(sql_path, encoding="utf-8-sig") as f:
        sql_content = f.read()

    debug(f"[schema_profiling.parse_sql_file] reading: {len(sql_content)} chars")

    def _finalize(tables: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if tables:
            _enrich_ddl_comments_from_create_statements(tables, sql_content)
            _merge_ddl_comment_on_statements(tables, sql_content)
        return tables

    tables = _parse_sql_file_fallback(sql_content)
    if tables:
        debug(f"[schema_profiling.parse_sql_file] ddl_parser parsed: {len(tables)} tables")
        return _finalize(tables)

    tables = _parse_sql_file_regex_reflect(sql_content)
    if tables:
        debug(f"[schema_profiling.parse_sql_file] regex_reflect parsed: {len(tables)} tables")
        return _finalize(tables)

    if reflected_schema is not None and _schema_graph_has_structural_foreign_keys(reflected_schema):
        debug(
            "[schema_profiling.parse_sql_file] deterministic parsers returned 0 tables; "
            "reflection already has FK edges — skipping DDL LLM fallback"
        )
        return {}

    return _finalize(_parse_sql_file_via_llm(sql_content))


def _parse_sql_file_fallback(sql_content: str) -> dict[str, dict[str, Any]]:
    """Parse CREATE TABLE and ALTER TABLE metadata from DDL using the. active engine profile."""
    dialect_cls = DialectRegistry.get_dialect_class(EngineConfig.TYPE)
    dialect_stub = dialect_cls.__new__(dialect_cls)
    token = dialect_stub.sql_file_parse_dialect
    if token == "postgres":
        return _parse_sql_file_pglast_postgres(sql_content)
    return _parse_sql_file_sqlglot(sql_content, token)


def _extract_column_block_from_create(create_expr: Any) -> str:
    """Extract the inner content of the column definition block from a. sqlglot Create."""
    schema = create_expr.this if create_expr.this else create_expr.expression
    if schema is None:
        return ""
    expressions = getattr(schema, "expressions", None)
    if expressions:
        return ", ".join(e.sql() for e in expressions if hasattr(e, "sql"))
    schema_sql = schema.sql()
    if schema_sql.startswith("(") and schema_sql.endswith(")"):
        return str(schema_sql[1:-1].strip())
    full_sql = create_expr.sql()
    match = re.search(r"\(([\s\S]*)\)\s*(?:PARTITIONED|STORED|LOCATION|AS|$)", full_sql)
    if match:
        return str(match.group(1).strip())
    return str(schema_sql.strip())


def _parse_column_name_and_sql_type(line: str) -> tuple[str, str] | None:
    """Split a single DDL column line into name and full SQL type. tokens."""
    parts = line.split()
    if len(parts) < 2:
        return None
    col_name = parts[0].strip("`").strip('"')
    chunks: list[str] = []
    depth = 0
    i = 1
    while i < len(parts):
        tok = parts[i]
        if depth == 0 and tok.upper() in COLUMN_DEFINITION_STOP_WORDS:
            break
        chunks.append(tok)
        depth += tok.count("(") - tok.count(")")
        i += 1
        if depth <= 0 and chunks:
            break
    if not chunks:
        return None
    return col_name, " ".join(chunks)


def _split_by_top_level_comma(s: str) -> list[str]:
    """Split a string by commas that are outside parentheses."""
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for char in s:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            segment = "".join(current).strip()
            if segment:
                result.append(segment)
            current = []
        else:
            current.append(char)
    segment = "".join(current).strip()
    if segment:
        result.append(segment)
    return result


def _infer_column_line_nullable(line_upper: str) -> bool:
    """Infer whether a single-column DDL line allows NULLs. Inline ``PRIMARY KEY`` implies NOT NULL in SQL; ``NOT NULL`` is parsed explicitly."""
    if re.search(r"\bNOT\s+NULL\b", line_upper):
        return False
    if re.search(r"\bPRIMARY\s+KEY\b", line_upper):
        return False
    return True


def _unescape_sql_string_literal(text: str) -> str:
    return text.replace("''", "'").replace('""', '"')


def _extract_sql_quoted_literal(fragment: str) -> str | None:
    fragment = fragment.strip()
    if fragment.startswith("'") and fragment.endswith("'") and len(fragment) >= 2:
        return _unescape_sql_string_literal(fragment[1:-1])
    if fragment.startswith('"') and fragment.endswith('"') and len(fragment) >= 2:
        return _unescape_sql_string_literal(fragment[1:-1])
    return None


def _extract_inline_comment_suffix(line: str) -> str | None:
    match = re.search(
        r"\bCOMMENT\b\s+('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    literal = _extract_sql_quoted_literal(match.group(1))
    return literal.strip() if literal else None


def _extract_create_table_comment(full_create_sql: str) -> str | None:
    match = re.search(
        r"\)\s*COMMENT\s*=?\s*('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")",
        full_create_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    literal = _extract_sql_quoted_literal(match.group(1))
    return literal.strip() if literal else None


def _normalize_ddl_table_name(raw: str) -> str:
    name = raw.strip().strip('`"[]')
    if "." in name:
        return name.split(".")[-1].strip('`"[]')
    return name


def _merge_ddl_comment_on_statements(tables: dict[str, dict[str, Any]], sql_content: str) -> None:
    """Merge standalone ``COMMENT ON`` DDL into parsed table metadata."""
    for match in re.finditer(
        r"COMMENT\s+ON\s+TABLE\s+([\w.`\"\[\]]+(?:\.[\w.`\"\[\]]+)?)\s+IS\s+('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")",
        sql_content,
        re.IGNORECASE | re.DOTALL,
    ):
        tname = _normalize_ddl_table_name(match.group(1))
        literal = _extract_sql_quoted_literal(match.group(2))
        if not tname or not literal:
            continue
        entry = tables.setdefault(
            tname,
            {
                "table_name_original": tname,
                "column_names_original": [],
                "column_types": [],
                "primary_keys": [],
                "foreign_keys": [],
            },
        )
        entry["table_comment"] = literal.strip()
    for match in re.finditer(
        r"COMMENT\s+ON\s+COLUMN\s+([\w.`\"\[\]]+(?:\.[\w.`\"\[\]]+)?)\s+IS\s+('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")",
        sql_content,
        re.IGNORECASE | re.DOTALL,
    ):
        ref = match.group(1)
        literal = _extract_sql_quoted_literal(match.group(2))
        if not literal:
            continue
        parts = ref.replace('"', "").replace("`", "").split(".")
        if len(parts) < 2:
            continue
        tname = _normalize_ddl_table_name(parts[-2])
        cname = parts[-1].strip('`"[]')
        entry = tables.setdefault(
            tname,
            {
                "table_name_original": tname,
                "column_names_original": [],
                "column_types": [],
                "primary_keys": [],
                "foreign_keys": [],
            },
        )
        comments = entry.setdefault("column_comments", {})
        if isinstance(comments, dict):
            comments[cname] = literal.strip()


def _enrich_ddl_comments_from_create_statements(tables: dict[str, dict[str, Any]], sql_content: str) -> None:
    """Merge inline CREATE TABLE comment metadata into parser output."""
    pattern = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+|TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"([\w.`\"\[\]]+(?:\.[\w.`\"\[\]]+)?)\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(sql_content):
        tname = _normalize_ddl_table_name(match.group(1))
        if not tname or tname not in tables:
            continue
        open_idx = match.end() - 1
        span = _balanced_paren_span(sql_content, open_idx)
        if span is None:
            continue
        _open_idx, end_after = span
        inner = sql_content[open_idx + 1 : end_after - 1]
        stmt_end = sql_content.find(";", end_after)
        if stmt_end == -1:
            stmt_end = len(sql_content)
        create_stmt = sql_content[match.start() : stmt_end]
        parts = _table_metadata_dict_from_ddl_parts(tname, inner, create_stmt)
        entry = tables[tname]
        table_comment = parts.get("table_comment")
        if table_comment:
            entry["table_comment"] = table_comment
        col_comments = parts.get("column_comments") or {}
        if col_comments:
            merged = dict(entry.get("column_comments") or {})
            merged.update(col_comments)
            entry["column_comments"] = merged


def _parse_columns_and_constraints(
    col_block: str,
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]], list[str], list[bool], dict[str, str]]:
    """Parse column definitions and constraints from a column block. string."""
    lines = [line.strip() for line in _split_by_top_level_comma(col_block)]

    columns = []
    types = []
    pks = []
    fks = []
    unique_cols: list[str] = []
    column_is_nullable: list[bool] = []
    column_comments: dict[str, str] = {}

    for line in lines:
        line_upper = line.upper()

        if line_upper.startswith("PRIMARY KEY"):
            pk_cols = _extract_pk_columns(line)
            pks.extend(pk_cols)
            continue

        if line_upper.startswith("FOREIGN KEY"):
            fk_def = _extract_fk_definition(line)
            if fk_def:
                fks.append(fk_def)
            continue

        if line_upper.startswith("UNIQUE") or (line_upper.startswith("CONSTRAINT") and "UNIQUE" in line_upper):
            m = re.search(r"UNIQUE\s*\(([^)]+)\)", line, re.IGNORECASE)
            if m:
                for c in m.group(1).split(","):
                    cn = c.strip().strip("`").strip('"')
                    if cn:
                        unique_cols.append(cn)
            continue

        parsed = _parse_column_name_and_sql_type(line)
        if parsed is None:
            continue
        col_name, col_type = parsed

        columns.append(col_name)
        types.append(col_type)
        column_is_nullable.append(_infer_column_line_nullable(line_upper))
        inline_comment = _extract_inline_comment_suffix(line)
        if inline_comment:
            column_comments[col_name] = inline_comment

        type_word_count = len(col_type.split())
        line_tokens = line.split()
        tail_tokens = line_tokens[1 + type_word_count :] if len(line_tokens) > 1 + type_word_count else []
        if "PRIMARY KEY" in line_upper:
            pks.append(col_name)
        elif any(t.upper() == "UNIQUE" for t in tail_tokens):
            unique_cols.append(col_name)
        else:
            inline_ref = re.search(
                r"\bREFERENCES\s+((?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))*)\s*\(([^)]+)\)",
                line,
                re.IGNORECASE,
            )
            if inline_ref:
                fks.append(
                    {
                        "src_cols": [col_name],
                        "dst_table": _trailing_table_identifier(inline_ref.group(1)),
                        "dst_cols": [c.strip().strip("`").strip('"') for c in inline_ref.group(2).split(",")],
                    }
                )

    col_index = {name: idx for idx, name in enumerate(columns)}
    for pk_name in pks:
        idx = col_index.get(pk_name)
        if idx is not None:
            column_is_nullable[idx] = False

    return columns, types, pks, fks, unique_cols, column_is_nullable, column_comments


def _extract_pk_columns(line: str) -> list[str]:
    """Extract column names from a PRIMARY KEY (col1, col2) definition. line."""
    match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", line, re.IGNORECASE)
    if match:
        return [c.strip().strip("`").strip('"') for c in match.group(1).split(",")]
    return []


def _extract_fk_definition(line: str) -> dict[str, Any] | None:
    """Extract a FOREIGN KEY definition from a DDL constraint line."""
    ref_table_segment = r"((?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))*)"
    match = re.search(
        r"FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+" + ref_table_segment + r"\s*\(([^)]+)\)", line, re.IGNORECASE
    )
    if match:
        return {
            "src_cols": [c.strip().strip("`").strip('"') for c in match.group(1).split(",")],
            "dst_table": _trailing_table_identifier(match.group(2)),
            "dst_cols": [c.strip().strip("`").strip('"') for c in match.group(3).split(",")],
        }
    return None


def _table_metadata_dict_from_ddl_parts(table_name: str, col_block: str, full_create_sql: str) -> dict[str, Any]:
    """Build a single-table schema metadata dict from column text and. full CREATE DDL."""
    columns, types, pks, fks, uniqs, col_nullable, column_comments = _parse_columns_and_constraints(col_block)
    cat_pks, cat_fks, cat_uniqs = _parse_catalog_constraints_from_ddl(full_create_sql)
    merged_pks = list(dict.fromkeys(pks + cat_pks))
    merged_uniqs = list(dict.fromkeys(uniqs + cat_uniqs))
    merged_fks = list(fks)
    seen_fk = {(tuple(fk.get("src_cols", [])), fk.get("dst_table"), tuple(fk.get("dst_cols", []))) for fk in merged_fks}
    for fk in cat_fks:
        key = (tuple(fk.get("src_cols", [])), fk.get("dst_table"), tuple(fk.get("dst_cols", [])))
        if key not in seen_fk:
            seen_fk.add(key)
            merged_fks.append(fk)
    partition_cols = _partition_column_names_from_create_ddl(full_create_sql)
    table_comment = _extract_create_table_comment(full_create_sql)
    return {
        "table_name_original": table_name,
        "column_names_original": columns,
        "column_types": types,
        "column_is_nullable": col_nullable,
        "primary_keys": merged_pks,
        "foreign_keys": merged_fks,
        "unique_columns": merged_uniqs,
        "partition_columns": partition_cols,
        "table_comment": table_comment,
        "column_comments": column_comments,
    }


def _pglast_create_table_name(relation: Any) -> str | None:
    """Resolve an unqualified table name from a pglast `RangeVar`."""
    if relation is None:
        return None
    relname = getattr(relation, "relname", None)
    if relname is None:
        return None
    if isinstance(relname, str):
        name = relname.strip()
    else:
        name = str(relname).strip()
    if not name:
        return None
    if "." in name:
        return name.split(".")[-1].strip('"').strip("'")
    return name.strip('"').strip("'")


def _table_meta_append_column(tmeta: dict[str, Any], col_name: str, col_type: str) -> None:
    """Append a column and type to table DDL metadata when the column. is. not already present."""
    columns: list[str] = tmeta["column_names_original"]
    types: list[str] = tmeta["column_types"]
    nulls: list[bool] = tmeta.setdefault("column_is_nullable", [True] * len(columns))
    if len(nulls) < len(columns):
        nulls.extend([True] * (len(columns) - len(nulls)))
    if col_name in columns:
        return
    columns.append(col_name)
    types.append(col_type)
    nulls.append(True)


def _table_meta_extend_primary_keys(tmeta: dict[str, Any], names: list[str]) -> None:
    """Append primary-key column names without duplicates."""
    pks: list[str] = tmeta["primary_keys"]
    for n in names:
        if n not in pks:
            pks.append(n)


def _table_meta_append_foreign_key(tmeta: dict[str, Any], fk: dict[str, Any]) -> None:
    """Append a foreign-key edge dict if an identical edge is not. already recorded."""
    fks: list[dict[str, Any]] = tmeta["foreign_keys"]
    if fk in fks:
        return
    fks.append(fk)


def _table_meta_extend_unique_columns(tmeta: dict[str, Any], names: list[str]) -> None:
    """Append unique column names while preserving first-seen order."""
    uniq: list[str] = tmeta["unique_columns"]
    seen = set(uniq)
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)


def _pglast_string_sval(node: Any) -> str | None:
    """Read a pglast `String` node value as a plain identifier string."""
    if node is None:
        return None
    sval = getattr(node, "sval", None)
    if isinstance(sval, str) and sval:
        return sval
    return None


def _pglast_pk_constraint_column_names(constraint: Any) -> list[str]:
    """Extract PRIMARY KEY column names from a pglast `Constraint` node."""
    keys = getattr(constraint, "keys", None)
    if not keys:
        return []
    out: list[str] = []
    for item in keys:
        s = _pglast_string_sval(item)
        if s:
            out.append(s)
    return out


def _pglast_fk_constraint_to_dict(constraint: Any) -> dict[str, Any] | None:
    """Build a `foreign_keys` entry dict from a pglast FOREIGN KEY. `Constraint`."""
    pktable = getattr(constraint, "pktable", None)
    dst_table = _pglast_create_table_name(pktable)
    if not dst_table:
        return None
    src_cols: list[str] = []
    for item in getattr(constraint, "fk_attrs", None) or ():
        s = _pglast_string_sval(item)
        if s:
            src_cols.append(s)
    dst_cols: list[str] = []
    for item in getattr(constraint, "pk_attrs", None) or ():
        s = _pglast_string_sval(item)
        if s:
            dst_cols.append(s)
    if not src_cols or not dst_cols:
        return None
    return {"src_cols": src_cols, "dst_table": dst_table, "dst_cols": dst_cols}


def _pglast_column_def_name_and_type(column_def: Any) -> tuple[str, str] | None:
    """Resolve column name and type SQL text from a pglast `ColumnDef`."""
    if RawStream is None:
        return None
    colname = getattr(column_def, "colname", None)
    if not isinstance(colname, str) or not colname.strip():
        return None
    type_name = getattr(column_def, "typeName", None)
    if type_name is None:
        return None
    type_sql = _pglast_raw_stream_render(type_name)
    if not type_sql:
        return None
    return colname.strip(), type_sql


def _pglast_apply_alter_constraint(tmeta: dict[str, Any], constraint: Any) -> None:
    """Merge PRIMARY KEY, FOREIGN KEY, or single-column UNIQUE from a. pglast `Constraint`."""
    contype = getattr(constraint, "contype", None)
    if contype is None or ConstrType is None:
        return
    if contype == ConstrType.CONSTR_PRIMARY:
        _table_meta_extend_primary_keys(tmeta, _pglast_pk_constraint_column_names(constraint))
        return
    if contype == ConstrType.CONSTR_FOREIGN:
        fk = _pglast_fk_constraint_to_dict(constraint)
        if fk:
            _table_meta_append_foreign_key(tmeta, fk)
        return
    if contype == ConstrType.CONSTR_UNIQUE:
        cols = _pglast_pk_constraint_column_names(constraint)
        if len(cols) == 1:
            _table_meta_extend_unique_columns(tmeta, cols)


def _apply_pglast_alter_table_statement(tables: dict[str, dict[str, Any]], stmt: Any) -> None:
    """Apply a pglast `AlterTableStmt` to entries in `tables`."""
    if AlterTableType is None or RawStream is None:
        return
    relation = getattr(stmt, "relation", None)
    table_name = _pglast_create_table_name(relation)
    if not table_name:
        return
    tmeta = tables.get(table_name)
    if tmeta is None:
        debug(f"[schema_profiling._apply_pglast_alter_table_statement] skip alter, unknown table: {table_name}")
        return
    for cmd in getattr(stmt, "cmds", None) or ():
        subtype = getattr(cmd, "subtype", None)
        if subtype == AlterTableType.AT_AddColumn:
            col_def = getattr(cmd, "def_", None)
            if col_def is None:
                continue
            parsed = _pglast_column_def_name_and_type(col_def)
            if parsed:
                _table_meta_append_column(tmeta, parsed[0], parsed[1])
            continue
        if subtype == AlterTableType.AT_AddConstraint:
            constr = getattr(cmd, "def_", None)
            if constr is not None:
                _pglast_apply_alter_constraint(tmeta, constr)


def _sqlglot_alter_target_table_name(alter: Any) -> str | None:
    """Resolve the unqualified table name from a sqlglot `Alter` expression."""
    this = getattr(alter, "this", None)
    if this is None:
        return None
    name = getattr(this, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().strip('`"')
    text = str(this).split(".")[-1].strip().strip('`"')
    return text or None


def _sqlglot_column_def_name_type(col_def: Any, *, dialect_token: str) -> tuple[str, str] | None:
    """Extract column name and dialect SQL type text from a sqlglot. `ColumnDef`."""
    ident = getattr(col_def, "this", None)
    col_name = getattr(ident, "name", None) if ident is not None else None
    if not isinstance(col_name, str) or not col_name.strip():
        return None
    kind = getattr(col_def, "args", {}).get("kind")
    if kind is None:
        return None
    type_sql = kind.sql(dialect=dialect_token).strip().upper()
    if not type_sql:
        return None
    return col_name.strip(), type_sql


def _sqlglot_schema_fk_reference(reference: Any) -> tuple[str, list[str]] | None:
    """Parse `REFERENCES tbl(cols)` from a sqlglot. `ForeignKey.reference`."""
    schema = getattr(reference, "this", None)
    if schema is None:
        return None
    table_part = getattr(schema, "this", None)
    dst_table = getattr(table_part, "name", None) if table_part is not None else None
    if not isinstance(dst_table, str) or not dst_table.strip():
        inner = getattr(table_part, "this", None)
        dst_table = getattr(inner, "name", None) if inner is not None else None
    if not isinstance(dst_table, str) or not dst_table.strip():
        return None
    dst_cols: list[str] = []
    for expr in getattr(schema, "expressions", None) or ():
        n = getattr(expr, "name", None)
        if isinstance(n, str) and n.strip():
            dst_cols.append(n.strip().strip('`"'))
    if not dst_cols:
        return None
    return dst_table.strip().strip('`"'), dst_cols


def _sqlglot_apply_constraint_node(tmeta: dict[str, Any], inner: Any) -> None:
    """Merge PRIMARY KEY, FOREIGN KEY, or single-column UNIQUE from a. sqlglot constraint expression."""
    if isinstance(inner, sqlglot.exp.PrimaryKey):
        names: list[str] = []
        for e in getattr(inner, "expressions", None) or ():
            n = getattr(e, "name", None)
            if isinstance(n, str) and n.strip():
                names.append(n.strip().strip('`"'))
        _table_meta_extend_primary_keys(tmeta, names)
        return
    if isinstance(inner, sqlglot.exp.ForeignKey):
        src_cols: list[str] = []
        for e in getattr(inner, "expressions", None) or ():
            n = getattr(e, "name", None)
            if isinstance(n, str) and n.strip():
                src_cols.append(n.strip().strip('`"'))
        ref = getattr(inner, "args", {}).get("reference")
        parsed = _sqlglot_schema_fk_reference(ref) if ref is not None else None
        if not src_cols or parsed is None:
            return
        dst_table, dst_cols = parsed
        _table_meta_append_foreign_key(tmeta, {"src_cols": src_cols, "dst_table": dst_table, "dst_cols": dst_cols})
        return
    if isinstance(inner, sqlglot.exp.UniqueColumnConstraint):
        schema = getattr(inner, "this", None)
        cols: list[str] = []
        if schema is not None:
            for e in getattr(schema, "expressions", None) or ():
                n = getattr(e, "name", None)
                if isinstance(n, str) and n.strip():
                    cols.append(n.strip().strip('`"'))
        if len(cols) == 1:
            _table_meta_extend_unique_columns(tmeta, cols)


def _sqlglot_apply_alter_action(tmeta: dict[str, Any], action: Any, *, dialect_token: str) -> None:
    """Apply one sqlglot ALTER action to table metadata."""
    if isinstance(action, sqlglot.exp.ColumnDef):
        parsed = _sqlglot_column_def_name_type(action, dialect_token=dialect_token)
        if parsed:
            _table_meta_append_column(tmeta, parsed[0], parsed[1])
        return
    if isinstance(action, sqlglot.exp.Schema):
        for e in getattr(action, "expressions", None) or ():
            if isinstance(e, sqlglot.exp.ColumnDef):
                _sqlglot_apply_alter_action(tmeta, e, dialect_token=dialect_token)
        return
    if isinstance(action, sqlglot.exp.AddConstraint):
        for c in getattr(action, "expressions", None) or ():
            if not isinstance(c, sqlglot.exp.Constraint):
                continue
            for inner in getattr(c, "expressions", None) or ():
                _sqlglot_apply_constraint_node(tmeta, inner)


def _apply_sqlglot_alter_table(tables: dict[str, dict[str, Any]], alter: Any, *, dialect_token: str) -> None:
    """Apply a sqlglot `Alter` (TABLE) statement to `tables`."""
    if getattr(alter, "kind", None) != "TABLE":
        return
    table_name = _sqlglot_alter_target_table_name(alter)
    if not table_name:
        return
    tmeta = tables.get(table_name)
    if tmeta is None:
        debug(f"[schema_profiling._apply_sqlglot_alter_table] skip alter, unknown table: {table_name}")
        return
    for action in alter.actions:
        _sqlglot_apply_alter_action(tmeta, action, dialect_token=dialect_token)


def _parse_sql_file_pglast_postgres(sql_content: str) -> dict[str, dict[str, Any]]:
    """Parse PostgreSQL-flavour CREATE TABLE nodes from DDL text using. pglast."""
    if (
        not _PG_LAST_SQL_AVAILABLE
        or _pglast_module is None
        or CreateStmt is None
        or RawStream is None
        or AlterTableStmt is None
    ):
        debug("[schema_profiling._parse_sql_file_pglast_postgres] pglast not installed")
        return {}
    try:
        raw_statements = _pglast_module.parse_sql(sql_content)
    except Exception as exc:
        debug(f"[schema_profiling._parse_sql_file_pglast_postgres] parse failed: {exc}")
        return {}

    tables: dict[str, dict[str, Any]] = {}
    for raw in raw_statements:
        stmt = getattr(raw, "stmt", None)
        if isinstance(stmt, CreateStmt):
            if not stmt.relation or not stmt.tableElts:
                continue
            table_name = _pglast_create_table_name(stmt.relation)
            if not table_name:
                continue
            debug(f"[schema_profiling._parse_sql_file_pglast_postgres] parsing: {table_name}")
            col_block = ", ".join(_pglast_raw_stream_render(elt) for elt in stmt.tableElts)
            full_sql = _pglast_raw_stream_render(stmt)
            tables[table_name] = _table_metadata_dict_from_ddl_parts(table_name, col_block, full_sql)
            continue
        if isinstance(stmt, AlterTableStmt):
            _apply_pglast_alter_table_statement(tables, stmt)

    debug(f"[schema_profiling._parse_sql_file_pglast_postgres] complete: {len(tables)} tables")
    return tables


def _short_ddl_table_name(table_ref: Any, dialect_token: str) -> str:
    """Return the bare table name from a sqlglot table reference (strip catalog/schema qualifiers)."""
    if table_ref is None:
        return ""
    if hasattr(table_ref, "this") and table_ref.this is not None:
        leaf = table_ref.this
        table_name = getattr(leaf, "name", None) or str(leaf)
    else:
        table_name = getattr(table_ref, "name", None) or str(table_ref)
    raw = str(table_name).strip('`"[]')
    if "." in raw or dialect_token in ("bigquery", "spark", "databricks"):
        raw = raw.split(".")[-1].strip('`"[]')
    return raw


def _parse_sql_file_sqlglot(sql_content: str, dialect_token: str) -> dict[str, dict[str, Any]]:
    """Parse CREATE TABLE statements from DDL text using the given. sqlglot dialect token."""
    try:
        statements = sqlglot.parse(sql_content, dialect=dialect_token)
    except Exception as exc:
        debug(f"[schema_profiling._parse_sql_file_sqlglot] parse failed ({dialect_token}): {exc}")
        return {}

    tables: dict[str, dict[str, Any]] = {}
    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, sqlglot.exp.Create) and stmt.this:
            table_name = _short_ddl_table_name(stmt.this, dialect_token)
            if not table_name:
                continue
            debug(f"[schema_profiling._parse_sql_file_sqlglot] parsing ({dialect_token}): {table_name}")
            col_block = _extract_column_block_from_create(stmt)
            full_stmt = stmt.sql(dialect=dialect_token)
            tables[table_name] = _table_metadata_dict_from_ddl_parts(table_name, col_block, full_stmt)
            continue
        if isinstance(stmt, sqlglot.exp.Alter):
            _apply_sqlglot_alter_table(tables, stmt, dialect_token=dialect_token)

    debug(f"[schema_profiling._parse_sql_file_sqlglot] complete ({dialect_token}): {len(tables)} tables")
    return tables


def value_overlap_stats_for_columns(
    left: ColumnMetadata,
    right: ColumnMetadata,
) -> tuple[int, float] | None:
    """Return intersection size and overlap ratio relative to the smaller ``value_overlap_sample``."""
    s1, s2, _ = normalized_value_overlap_sets(left, right)
    if not s1 or not s2:
        return None
    smaller = min(len(s1), len(s2))
    if smaller == 0:
        return None
    inter = len(s1 & s2)
    return inter, inter / smaller


def compute_semantic_profile_join_neighbors(sg: SchemaGraph) -> None:
    """Populate ``semantic_join_neighbors`` on physical columns from ``value_overlap_sample``. A pair ``(A.x, B.y)`` qualifies only when every gate below is satisfied: both endpoints have ``value_type == "string"`` (numeric value-overlap is forbidden because identifier ranges coincide too easily); at least one endpoint is a primary key OR has ``is_unique=True`` (an anchored side); the two tables are not already connected by any foreign key edge that joins these two specific columns; and the existing statistical floors ``PolicyConfig.SEMANTIC_JOIN_MIN_INTERSECTION``, ``PolicyConfig.SEMANTIC_JOIN_MIN_DISTINCT``, and ``PolicyConfig.SEMANTIC_JOIN_MIN_OVERLAP_RATIO`` all pass on the stored ascending distinct samples. Idempotent: clears then recomputes."""
    min_ratio = PolicyConfig.SEMANTIC_JOIN_MIN_OVERLAP_RATIO
    min_distinct = int(PolicyConfig.SEMANTIC_JOIN_MIN_DISTINCT)
    min_intersection = int(PolicyConfig.SEMANTIC_JOIN_MIN_INTERSECTION)
    user_neighbors_by_table: dict[str, list[tuple[str, str, str, str]]] = {}
    for tname, tbl in sg.tables.items():
        snapshot = list(getattr(tbl, "_user_semantic_neighbors", []) or [])
        if snapshot:
            user_neighbors_by_table[tname] = snapshot
    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            col.semantic_join_neighbors = []

    fk_column_pairs: set[tuple[str, str, str, str]] = set()
    for tname, tbl in sg.tables.items():
        for fk in tbl.foreign_keys:
            for sc, dc in zip(fk.src_cols, fk.dst_cols, strict=False):
                fk_column_pairs.add((tname, sc, fk.dst_table, dc))
                fk_column_pairs.add((fk.dst_table, dc, tname, sc))

    entries: list[tuple[str, str, ColumnMetadata]] = []
    for tname, tbl in sg.tables.items():
        for cname, cmeta in tbl.columns.items():
            if column_has_unknown_value_type(cmeta):
                continue
            if (cmeta.value_type or "").strip().lower() != "string":
                continue
            if cmeta.value_overlap_sample:
                entries.append((tname, cname, cmeta))

    for i, (t1, c1, m1) in enumerate(entries):
        if not m1.value_overlap_sample:
            continue
        anchor1 = bool(m1.is_primary_key) or bool(m1.is_unique)
        for t2, c2, m2 in entries[i + 1 :]:
            if t1 == t2:
                continue
            anchor2 = bool(m2.is_primary_key) or bool(m2.is_unique)
            if not (anchor1 or anchor2):
                continue
            if (t1, c1, t2, c2) in fk_column_pairs:
                continue
            if not m2.value_overlap_sample:
                continue

            s1, s2, _ = normalized_value_overlap_sets(m1, m2)
            if not s1 or not s2:
                continue
            smaller = min(len(s1), len(s2))
            if smaller < min_distinct:
                continue
            inter = len(s1 & s2)
            if inter < min_intersection:
                continue
            if inter / smaller < min_ratio:
                continue
            m1.semantic_join_neighbors.append((t2, c2))
            m2.semantic_join_neighbors.append((t1, c1))

    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            col.semantic_join_neighbors = sorted(set(col.semantic_join_neighbors), key=lambda p: (p[0], p[1]))

    for owner_table, quads in user_neighbors_by_table.items():
        owner_meta = sg.tables.get(owner_table)
        if owner_meta is None:
            continue
        for quad in quads:
            src_t, src_c, dst_t, dst_c = quad
            src_tbl = sg.tables.get(src_t)
            dst_tbl = sg.tables.get(dst_t)
            if src_tbl is None or dst_tbl is None:
                continue
            src_col = src_tbl.columns.get(src_c)
            dst_col = dst_tbl.columns.get(dst_c)
            if src_col is None or dst_col is None:
                continue
            if (dst_t, dst_c) not in src_col.semantic_join_neighbors:
                src_col.semantic_join_neighbors.append((dst_t, dst_c))
            if (src_t, src_c) not in dst_col.semantic_join_neighbors:
                dst_col.semantic_join_neighbors.append((src_t, src_c))
    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            col.semantic_join_neighbors = sorted(set(col.semantic_join_neighbors), key=lambda p: (p[0], p[1]))


def replay_user_semantic_neighbors_to_columns(sg: SchemaGraph) -> None:
    """Mirror every quad in :attr:`TableMetadata._user_semantic_neighbors` onto the per-column ``semantic_join_neighbors`` lists. The quad list on each table is the single authoritative store for user-overridden semantic edges; the per-column lists are a derived read view consumed by join-graph traversal. Call this helper after appending new quads (for example in :func:`apply_schema_overrides_to_graph`) so the per-column projection stays in sync without needing to re-run profile-derived overlap discovery. Idempotent: existing per-column entries are not duplicated and the lists are returned in a stable sort order."""
    for tbl in sg.tables.values():
        for quad in list(getattr(tbl, "_user_semantic_neighbors", []) or []):
            if not isinstance(quad, tuple) or len(quad) != 4:
                continue
            src_t, src_c, dst_t, dst_c = quad
            src_tbl = sg.tables.get(src_t)
            dst_tbl = sg.tables.get(dst_t)
            if src_tbl is None or dst_tbl is None:
                continue
            src_col = src_tbl.columns.get(src_c)
            dst_col = dst_tbl.columns.get(dst_c)
            if src_col is None or dst_col is None:
                continue
            if (dst_t, dst_c) not in src_col.semantic_join_neighbors:
                src_col.semantic_join_neighbors.append((dst_t, dst_c))
            if (src_t, src_c) not in dst_col.semantic_join_neighbors:
                dst_col.semantic_join_neighbors.append((src_t, src_c))
    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            col.semantic_join_neighbors = sorted(set(col.semantic_join_neighbors), key=lambda p: (p[0], p[1]))


def collect_schema_description_neutrality_forbidden_tokens(graph: SchemaGraph) -> frozenset[str]:
    """Collect physical catalog labels that must not appear in model- facing descriptions."""
    tokens: set[str] = set()
    for table in graph.tables.values():
        original = (table.original_name or "").strip()
        if original and original.lower() != table.name.lower():
            tokens.add(original)
        for col in table.columns.values():
            col_original = (col.original_name or "").strip()
            if col_original and col_original.lower() != col.name.lower():
                tokens.add(col_original)
    return frozenset(tokens)


def description_neutrality_violations(text: str, forbidden_tokens: frozenset[str]) -> list[str]:
    """Return forbidden tokens present in *text* using identifier word boundaries."""
    cleaned = str(text or "").strip()
    if not cleaned or not forbidden_tokens:
        return []
    hits: list[str] = []
    for token in sorted(forbidden_tokens, key=len, reverse=True):
        if not token:
            continue
        if re.search(rf"\b{re.escape(token)}\b", cleaned, flags=re.IGNORECASE):
            hits.append(token)
    return hits


def sanitize_description_text(text: str, forbidden_tokens: frozenset[str]) -> str:
    """Remove forbidden tokens from description prose while preserving readable remnants."""
    cleaned = str(text or "").strip()
    if not cleaned or not forbidden_tokens:
        return cleaned
    for token in sorted(forbidden_tokens, key=len, reverse=True):
        if not token:
            continue
        cleaned = re.sub(rf"\b{re.escape(token)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r.,;:-—")
    return cleaned


def sanitize_schema_graph_descriptions(graph: SchemaGraph, forbidden_tokens: frozenset[str]) -> None:
    """Strip neutrality violations from table and column descriptions in-place."""
    if not forbidden_tokens:
        return
    for table in graph.tables.values():
        if table.description:
            table.description = sanitize_description_text(table.description, forbidden_tokens)
            if not table.description:
                table.description_owner = None
        for col in table.columns.values():
            if col.description:
                col.description = sanitize_description_text(col.description, forbidden_tokens)
                if not col.description:
                    col.description_owner = None
