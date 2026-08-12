"""Column profiling, LLM classification, DDL parsing, and catalog extraction."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import sqlglot
from sqlalchemy import text

from ._config import EngineConfig, PolicyConfig
from ._constants import (
    AETHERSPACES_SEGMENT,
    BOOLEAN_ANTONYM_MIN_STEM_LEN,
    DEFAULT_RANDOM_SEED,
    DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED,
    DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED,
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED,
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED,
    DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT,
    DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED,
    DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE,
    DOMAIN_KNOWLEDGE_ENTRY_KEYS,
    DOMAIN_KNOWLEDGE_TOP_KEYS,
    JSON_COLUMN_TYPE_TOKENS,
    KNOWLEDGE_NOTES_COVERAGE_ENTRY_KEYS,
    KNOWLEDGE_NOTES_EXTRACT_MAX_ATTEMPTS,
    KNOWLEDGE_NOTES_RECORD_KEYS,
    KNOWLEDGE_NOTES_TOP_KEYS,
    MASTER_AETHERSPACE_NAME,
    MASTER_AETHERSPACE_UID,
    ROLE_VALUE_TYPE_COMPAT,
    SCHEMA_CLASSIFY_ERROR_DETAIL_CAP,
    STRUCTURAL_KNOWLEDGE_FACT_KEYS,
    STRUCTURAL_KNOWLEDGE_TOP_KEYS,
    UNKNOWN_VALUE_TYPE,
    VALID_SENSITIVITY_LEVELS,
)
from ._constants_runtime import (
    BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES,
    BOOLEAN_NEGATION_PREFIXES,
    BOOLEAN_NEGATION_SUFFIXES,
    BOOLEAN_TRUTH_PATTERN_MAP,
    COLUMN_DEFINITION_STOP_WORDS,
    KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM,
    KNOWLEDGE_NOTES_EXTRACT_SYSTEM,
    SCHEMA_CLASSIFY_SYSTEM,
    SCHEMA_CONSISTENCY_REFINE_SYSTEM,
    SCHEMA_ENTITY_ENRICH_SYSTEM,
)
from ._contracts_base import (
    ConfigError,
    DomainKnowledgeEntry,
    DomainKnowledgeKind,
    DomainKnowledgeState,
    KnowledgeScope,
    NotesCoverageEntry,
    NotesExtractionLedger,
    NotesExtractionResult,
    OverlapComparison,
    SensitivityClassification,
    SensitivityRatchetReport,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
    TableKind,
)
from ._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    DescriptionOwner,
    FKEdge,
    InferenceTag,
    RoleOwner,
    SchemaGraph,
    TableMetadata,
)
from ._knowledge_staleness import (
    knowledge_artifact_save_stamps,
    parse_structural_items,
    resolve_structural_knowledge_for_schema,
)
from ._llm_provider import LLMProvider
from ._schema_graph import register_apply_column_roles_llm_hook
from ._utils import (
    column_has_unknown_value_type,
    cost_cap_active,
    debug,
    description_neutrality_violations,
    diagnostic_debug_enabled,
    domain_knowledge_digest,
    effective_profile_timeout_ms,
    llm_usage_build_scope,
    load_domain_knowledge_artifact,
    normalize_column_type,
    normalize_text_value,
    notify,
    safe_json_loads,
    stable_json,
)
from ._utils_artifacts import (
    artifact_lock,
    save_domain_knowledge_artifact,
    write_json_atomic,
)

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


def _dialect_registry() -> Any:
    import importlib

    return importlib.import_module("aetherdialect._dialect").DialectRegistry


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
    return effective_profile_timeout_ms()


def apply_profile_timeout_to_dialect(dialect: Any, profile_timeout_ms: int | None) -> None:
    """Stamp per-member ``profile_timeout_ms`` onto *dialect* for schema profiling."""
    if profile_timeout_ms is None:
        return
    dialect.profile_timeout_ms = int(profile_timeout_ms)


def _stamp_profile_timeout_from_engine(dialect: Any, engine: Any) -> None:
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
    """Check if a column's top-K values match a known boolean-like pattern."""
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
    dialect: Any,
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
    dialect: Any,
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
    dialect: Any,
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
        return str(cast_expr)

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
    dialect: Any, qcol: str, qtbl: str, limit: int, *, sample_clause: str = "", use_subquery: bool = False
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


def _apply_collation_overlap_semantics(dialect: Any, col: ColumnMetadata) -> None:
    """Resolve per-column collation semantics used by overlap comparison."""
    col.is_case_insensitive_collation = dialect.column_is_case_insensitive_collation(col)
    col.overlap_comparison = "case_folded" if col.is_case_insensitive_collation else "exact"


def _profile_column(
    dialect: Any,
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
        and col_name.rsplit("_", 1)[-1].lower() == "name"
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


def _profile_composite_descriptive(dialect: Any, engine: Any, table: TableMetadata) -> None:
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
    dialect: Any,
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


def profile_schema(engine: Any, schema: SchemaGraph, dialect: Any) -> None:
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
    dialect: Any,
    table_kind: TableKind = TableKind.TABLE,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile a single column from a Databricks table via Spark SQL and update metadata in-place."""
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
    spark: Any, catalog: str, schema_name: str, table: TableMetadata, *, dialect: Any
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
    dialect: Any,
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


def profile_schema_spark(spark: Any, catalog: str, schema_name: str, schema: SchemaGraph, *, dialect: Any) -> None:
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


def cursor_rows_as_dicts(cursor: Any) -> list[dict[str, Any]]:
    """Convert cursor result rows to a list of dicts keyed by column name."""
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


def tables_meta_foreign_key_dicts_from_edges(edges: list[FKEdge]) -> list[dict[str, Any]]:
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
    dialect: Any,
    table_kind: TableKind = TableKind.TABLE,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile a single column via databricks-sql-connector and update metadata in-place."""
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
            rows = cursor_rows_as_dicts(cursor)
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
                minmax_rows = cursor_rows_as_dicts(cursor)
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
                freq_rows = cursor_rows_as_dicts(cursor)
                col.frequent_values = collect_profiling_frequent_values(
                    [r["v"] for r in freq_rows if r and r.get("v") is not None]
                )
            if deep_query_budget.allow():
                mode_sql = _build_mode_sql(qcol, full_table, sample_clause=sample_clause, use_subquery=use_subquery)
                cursor.execute(mode_sql)
                mode_rows = cursor_rows_as_dicts(cursor)
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
                sem_rows = cursor_rows_as_dicts(cursor)
                col.value_overlap_sample = [str(r["v"]) for r in sem_rows if r.get("v") is not None]
    except Exception as exc:
        _record_column_profile_failure(table_name, col, exc)


def _profile_composite_descriptive_sql_connector(
    connection: Any, catalog: str, schema_name: str, table: TableMetadata, *, dialect: Any
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
                    rows = cursor_rows_as_dicts(cursor)
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
    dialect: Any,
    deep_query_budget: Any | None = None,
) -> None:
    """Profile all columns in a Databricks table via databricks-sql- connector."""
    debug(f"[schema_profiling._profile_table_sql_connector] profiling {table.name}")
    if deep_query_budget is None:
        deep_query_budget = _new_profiling_deep_query_budget(PolicyConfig.PROFILING_SCHEMA_DEEP_QUERY_BUDGET)
    full_table = dialect.qualified_table_ref(table.name)
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {full_table}")
        rows = cursor_rows_as_dicts(cursor)
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
    connection: Any, catalog: str, schema_name: str, schema: SchemaGraph, *, dialect: Any
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
    """Detect array or list column types and return element type when inferable."""
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

    if role == ColumnRole.TEMPORAL.value and col.value_type not in ("date", "integer", "number"):
        hard_errors.append(f"{col.name}: TEMPORAL requires date or numeric value_type, got {col.value_type}")

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
    """Convert raw LLM classification JSON into validated internal tuples."""
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
    """Merge refine-stage JSON over base-stage JSON without dropping base tables or columns."""
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
    structural_knowledge: Sequence[StructuralKnowledgeFact] | None = None,
) -> str:
    """Return a stable digest over schema-classification inputs for disk cache lookup."""
    if structural_knowledge is not None:
        enrich_payload = [f.to_dict() for f in structural_knowledge]
        enrich_hash = hashlib.sha256(stable_json(enrich_payload).encode("utf-8")).hexdigest()
    else:
        enrich_hash = hashlib.sha256((notes_content or "").encode("utf-8")).hexdigest()
    payload = {
        "effective_structural_hash": schema.effective_structural_hash or schema.structural_hash or "",
        "profiling_hash": schema.profiling_hash or "",
        "enrichment_hash": enrich_hash,
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


def _classification_payload_from_schema(
    schema: SchemaGraph,
    scope: dict[str, frozenset[str]],
    *,
    prefer_base_descriptions: bool = False,
) -> dict[str, Any]:
    """Build a base_classification-shaped payload from live roles and descriptions."""
    out: dict[str, Any] = {}
    for table in schema.tables.values():
        scoped_names = scope.get(table.name, frozenset())
        columns: dict[str, Any] = {}
        for col_name in sorted(scoped_names):
            col = table.columns.get(col_name)
            if col is None:
                continue
            sens = col.sensitivity
            sens_val: str | None = None
            if sens is not None and sens != SensitivityClassification.NONE:
                sens_val = sens.value if hasattr(sens, "value") else str(sens)
            if prefer_base_descriptions:
                col_desc = (col.base_description or col.description or "").strip()
            else:
                col_desc = (col.description or col.base_description or "").strip()
            columns[col_name] = {
                "role": (col.role or ColumnRole.FREE_TEXT.value),
                "description": col_desc,
                "sensitivity": sens_val,
            }
        if prefer_base_descriptions:
            table_desc = (table.base_description or table.description or "").strip()
        else:
            table_desc = (table.description or table.base_description or "").strip()
        out[table.name] = {
            "table_role": (table.role or "unknown"),
            "description": table_desc,
            "columns": columns,
        }
    return out


def _structural_facts_payload(facts: Sequence[StructuralKnowledgeFact]) -> list[dict[str, str]]:
    return [f.to_dict() for f in facts if str(f.text or "").strip()]


def _scope_entity_names(scope: Mapping[str, frozenset[str]]) -> frozenset[str]:
    names: set[str] = set()
    for table_name, column_names in scope.items():
        names.add(str(table_name))
        for column_name in column_names:
            names.add(f"{table_name}.{column_name}")
    return frozenset(names)


def _route_structural_facts_to_entities(
    facts: Sequence[StructuralKnowledgeFact],
) -> dict[str, tuple[StructuralKnowledgeFact, ...]]:
    routes: dict[str, list[StructuralKnowledgeFact]] = {}
    for fact in facts:
        for entity in fact.referenced_entities:
            routes.setdefault(str(entity), []).append(fact)
    return {entity: tuple(routed) for entity, routed in routes.items()}


def _allowed_schema_tokens_for_entity_enrich(
    entity: str,
    routed_facts: Sequence[StructuralKnowledgeFact],
    scope_entity_names: frozenset[str],
) -> frozenset[str]:
    allowed = set(scope_entity_names)
    allowed.add(entity)
    if "." in entity:
        allowed.add(entity.split(".", 1)[0])
    for fact in routed_facts:
        allowed.update(fact.referenced_entities)
    return frozenset(allowed)


def _forbidden_schema_tokens_for_entity_enrich(
    schema: SchemaGraph,
    allowed_tokens: frozenset[str],
) -> frozenset[str]:
    all_names = set(schema_name_dump_for_domain_knowledge(schema))
    return frozenset(name for name in all_names if name not in allowed_tokens)


def _raise_if_entity_enrich_description_names_forbidden(
    description: str,
    forbidden_tokens: frozenset[str],
    *,
    entity: str,
) -> None:
    hits = description_neutrality_violations(description, forbidden_tokens)
    if hits:
        raise ConfigError(
            f"entity enrichment for {entity!r} names a schema identifier outside its routed reference set: {hits[0]!r}"
        )


def _llm_enrich_classification_payload_per_entity(
    base_payload: dict[str, Any],
    facts: Sequence[StructuralKnowledgeFact],
    scope: Mapping[str, frozenset[str]],
    schema: SchemaGraph,
) -> dict[str, Any]:
    """Enrich descriptions per entity using only facts routed to that entity."""
    facts_list = tuple(StructuralKnowledgeFact.normalize(f) for f in facts if str(f.text or "").strip())
    if not facts_list:
        return dict(base_payload)
    scope_names = _scope_entity_names(scope)
    routes = _route_structural_facts_to_entities(facts_list)
    merged = copy.deepcopy(base_payload)
    for entity in sorted(routes):
        if entity not in scope_names:
            continue
        routed = routes[entity]
        allowed_tokens = _allowed_schema_tokens_for_entity_enrich(entity, routed, scope_names)
        forbidden_tokens = _forbidden_schema_tokens_for_entity_enrich(schema, allowed_tokens)
        if "." in entity:
            table_name, column_name = entity.split(".", 1)
            table_data = merged.get(table_name)
            if not isinstance(table_data, dict):
                continue
            columns = table_data.get("columns")
            if not isinstance(columns, dict) or column_name not in columns:
                continue
            column_data = columns[column_name]
            if not isinstance(column_data, dict):
                continue
            user_payload = stable_json(
                {
                    "entity": entity,
                    "entity_type": "column",
                    "base_classification": column_data,
                    "structural_facts": _structural_facts_payload(routed),
                }
            )
            with llm_usage_build_scope():
                raw = LLMProvider.chat(SCHEMA_ENTITY_ENRICH_SYSTEM, user_payload, timeout=360.0, task="schema")
            enriched = safe_json_loads(raw)
            if not isinstance(enriched, dict):
                raise ValueError(f"entity enrichment returned non-object JSON for {entity!r}")
            description = str(enriched.get("description") or "").strip()
            if description:
                _raise_if_entity_enrich_description_names_forbidden(description, forbidden_tokens, entity=entity)
                columns[column_name] = {**column_data, **enriched}
            continue
        table_data = merged.get(entity)
        if not isinstance(table_data, dict):
            continue
        user_payload = stable_json(
            {
                "entity": entity,
                "entity_type": "table",
                "base_classification": {
                    "table_role": table_data.get("table_role"),
                    "description": table_data.get("description"),
                },
                "structural_facts": _structural_facts_payload(routed),
            }
        )
        with llm_usage_build_scope():
            raw = LLMProvider.chat(SCHEMA_ENTITY_ENRICH_SYSTEM, user_payload, timeout=360.0, task="schema")
        enriched = safe_json_loads(raw)
        if not isinstance(enriched, dict):
            raise ValueError(f"entity enrichment returned non-object JSON for {entity!r}")
        description = str(enriched.get("description") or "").strip()
        if description:
            _raise_if_entity_enrich_description_names_forbidden(description, forbidden_tokens, entity=entity)
        table_role = enriched.get("table_role")
        if isinstance(table_role, str) and table_role.strip():
            table_data["table_role"] = table_role
        if description:
            table_data["description"] = description
        merged[entity] = table_data
    return merged


def llm_enrich_schema_from_structural_knowledge(
    schema: SchemaGraph,
    facts: Sequence[StructuralKnowledgeFact],
    *,
    column_scope: dict[str, frozenset[str]] | None = None,
    prefer_base_descriptions: bool = True,
) -> dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]]:
    """Enrich and scope descriptions for an AetherSpace subset. Always calls the refine/scope LLM. When *facts* is empty, the model only scopes descriptions to in-scope entities. ``prefer_base_descriptions=True`` starts from profile ``base_description`` (space-notes enrich path). ``prefer_base_descriptions=False`` starts from current descriptions (inherit master notes-enriched prose, then scope)."""
    scope = column_scope if column_scope is not None else llm_classification_column_scope(schema)
    facts_list = tuple(StructuralKnowledgeFact.normalize(f) for f in facts if str(f.text or "").strip())
    base_payload = _classification_payload_from_schema(schema, scope, prefer_base_descriptions=prefer_base_descriptions)
    final_payload = _llm_enrich_classification_payload_per_entity(base_payload, facts_list, scope, schema)
    return _normalize_llm_classification_payload(final_payload)


def llm_classify_schema(
    schema: SchemaGraph,
    notes_content: str | None = None,
    *,
    structural_knowledge: Sequence[StructuralKnowledgeFact] | None = None,
    column_scope: dict[str, frozenset[str]] | None = None,
    cache_payload_out: list[dict[str, Any]] | None = None,
) -> dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]]:
    """Classify roles and base descriptions from profiling, then enrich from structural facts when present (otherwise cross-table consistency). Notes text is not passed to the enrich pass — extract structural facts first and pass them as ``structural_knowledge``."""
    scope = column_scope if column_scope is not None else llm_classification_column_scope(schema)
    facts = tuple(structural_knowledge) if structural_knowledge is not None else ()
    content_hash = schema_classification_content_hash(
        schema,
        notes_content,
        scope,
        structural_knowledge=facts if structural_knowledge is not None else None,
    )
    cached_payload = _load_schema_classification_cache(content_hash)
    if cached_payload is not None:
        if cache_payload_out is not None:
            cache_payload_out.append(cached_payload)
        return _normalize_llm_classification_payload(cached_payload)
    with llm_usage_build_scope():
        tables_data = []
        for table_name in sorted(schema.tables):
            table = schema.tables[table_name]
            scoped_names = scope.get(table.name, frozenset())
            fks = sorted(
                ",".join(sorted(fk.src_cols)) + "->" + fk.dst_table + "." + ",".join(sorted(fk.dst_cols))
                for fk in table.foreign_keys
            )
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
        if not isinstance(base_payload, dict):
            raise ValueError(f"LLM returned invalid JSON for schema classification (base): {str(raw_base)[:200]}")
        _stamp_base_descriptions_from_payload(schema, base_payload, scope)
        final_payload: dict[str, Any] = base_payload
        if facts:
            final_payload = _llm_enrich_classification_payload_per_entity(base_payload, facts, scope, schema)
        else:
            user_refine = stable_json({"base_classification": base_payload})
            raw_refine = LLMProvider.chat(SCHEMA_CONSISTENCY_REFINE_SYSTEM, user_refine, timeout=360.0, task="schema")
            refined_payload = safe_json_loads(raw_refine)
            if not isinstance(refined_payload, dict):
                raise ValueError("schema consistency refine returned non-object JSON")
            final_payload = _merge_classification_payloads(base_payload, refined_payload)
        if cache_payload_out is not None:
            cache_payload_out.append(final_payload)
        return _normalize_llm_classification_payload(final_payload)


def _stamp_base_descriptions_from_payload(
    schema: SchemaGraph,
    base_payload: dict[str, Any],
    scope: dict[str, frozenset[str]],
) -> None:
    """Persist pre-notes descriptions onto ``base_description`` without clobbering user overrides."""
    normalized = _normalize_llm_classification_payload(base_payload)
    for table_name, (table_role, table_description, column_classifications) in normalized.items():
        del table_role
        table = schema.tables.get(table_name)
        if table is None:
            continue
        if table.description_owner != DescriptionOwner.USER_OVERRIDE:
            text = (table_description or "").strip()
            if text:
                table.base_description = text
        scoped = scope.get(table_name, frozenset())
        for col_name, (_role, col_description, _sens) in column_classifications.items():
            if col_name not in scoped or col_name not in table.columns:
                continue
            col = table.columns[col_name]
            if col.description_owner == DescriptionOwner.USER_OVERRIDE:
                continue
            ctext = (col_description or "").strip()
            if ctext:
                col.base_description = ctext


def restore_descriptions_from_base(schema: SchemaGraph) -> None:
    """Reset live descriptions to persisted base descriptions, preserving user overrides."""
    for table in schema.tables.values():
        if table.description_owner != DescriptionOwner.USER_OVERRIDE and table.base_description.strip():
            table.description = table.base_description
            table.description_owner = DescriptionOwner.LLM_REFINEMENT
        for col in table.columns.values():
            if col.description_owner != DescriptionOwner.USER_OVERRIDE and col.base_description.strip():
                col.description = col.base_description
                col.description_owner = DescriptionOwner.LLM_REFINEMENT


def apply_column_roles_llm(
    schema: SchemaGraph,
    notes_content: str | None = None,
    *,
    skip_columns: set[tuple[str, str]] | None = None,
    log_sink: Callable[[str], None] | None = None,
    artifacts_dir: str | None = None,
    structural_knowledge: Sequence[StructuralKnowledgeFact] | None = None,
    skip_structural_extraction: bool = False,
) -> None:
    """Apply LLM-inferred roles, descriptions, and sensitivity to the schema in-place."""
    sink_call: Callable[[str], None] = log_sink if log_sink is not None else notify
    skip_columns = skip_columns or set()
    debug(f"[schema_profiling.apply_column_roles_llm] classifying {len(schema.tables)} tables via LLM (base + refine)")
    total_columns = sum(len(table.columns) for table in schema.tables.values())
    debug(f"[schema_profiling.apply_column_roles_llm] total columns: {total_columns}")
    sink_call(
        f"  Classifying {len(schema.tables)} tables / {total_columns} columns via LLM (this can take a minute)..."
    )
    llm_column_scope = llm_classification_column_scope(schema)
    notes_stripped = (notes_content or "").strip()
    structural_facts: tuple[StructuralKnowledgeFact, ...] = ()
    if skip_structural_extraction:
        structural_facts = tuple(
            structural_knowledge
            if structural_knowledge is not None
            else getattr(schema, "structural_knowledge", ()) or ()
        )
    elif notes_stripped:
        structural_facts = resolve_structural_knowledge_for_schema(
            schema,
            notes_content,
            artifacts_dir=artifacts_dir,
            extract_knowledge_from_notes=extract_knowledge_from_notes,
        )
        schema.structural_knowledge = structural_facts
    classification_cache_hash = schema_classification_content_hash(
        schema,
        notes_content,
        llm_column_scope,
        structural_knowledge=structural_facts,
    )
    _debug_trace_classification_scope(schema, llm_column_scope, log_sink=sink_call)
    role_counts: dict[str, int] = {}
    table_role_counts: dict[str, int] = {}
    llm_success = 0
    success = False
    last_hard_errors: list[str] = []
    max_attempts = PolicyConfig.MAX_ROLE_CLASSIFICATION_RETRIES + 1
    description_owner = DescriptionOwner.NOTES if notes_stripped else DescriptionOwner.LLM_REFINEMENT
    for attempt in range(max_attempts):
        cache_payload_out: list[dict[str, Any]] = []
        try:
            classifications = llm_classify_schema(
                schema,
                notes_content,
                structural_knowledge=structural_facts,
                column_scope=llm_column_scope,
                cache_payload_out=cache_payload_out,
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
                        if col.role is not None:
                            role_counts[col.role] = role_counts.get(col.role, 0) + 1
            success = True
            llm_success = len(schema.tables)
            debug("[apply_column_roles_llm] two-phase classification successful")
            if cache_payload_out:
                _save_schema_classification_cache(classification_cache_hash, cache_payload_out[0])
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


def schema_name_dump_for_domain_knowledge(schema: SchemaGraph) -> tuple[str, ...]:
    """Return sorted qualified table and ``table.column`` names (structural id list)."""
    names: list[str] = []
    for table_name, table in sorted(schema.tables.items()):
        names.append(str(table_name))
        for col_name in sorted(table.columns.keys()):
            names.append(f"{table_name}.{col_name}")
    return tuple(names)


def filter_schema_anchored_domain_knowledge(
    entries: Sequence[DomainKnowledgeEntry],
    schema: SchemaGraph,
) -> tuple[DomainKnowledgeEntry, ...]:
    """Security filter: drop entries that name schema-known sensitive columns in text."""
    return tuple(entry for entry in entries if not DomainKnowledgeEntry.sensitive_column_references(entry.text, schema))


def filter_schema_anchored_structural_knowledge(
    facts: Sequence[StructuralKnowledgeFact],
    schema: SchemaGraph,
) -> tuple[StructuralKnowledgeFact, ...]:
    """Security filter: drop structural facts that name schema-known sensitive columns in text."""
    return tuple(fact for fact in facts if not DomainKnowledgeEntry.sensitive_column_references(fact.text, schema))


def _reject_unexpected_keys(obj: dict[str, Any], allowed: frozenset[str], *, context: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise ValueError(f"{context} has unexpected keys: {sorted(extra)}")


def _normalize_domain_knowledge_notes_items(parsed: Sequence[Any]) -> tuple[DomainKnowledgeEntry, ...]:
    """Strict-normalize LLM entry objects into domain-knowledge entries (exact key/kind/text)."""
    allowed_kinds = {member.value for member in DomainKnowledgeKind}
    candidates: list[DomainKnowledgeEntry] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("domain knowledge notes item must be an object")
        _reject_unexpected_keys(item, DOMAIN_KNOWLEDGE_ENTRY_KEYS, context="domain knowledge notes entry")
        if "key" not in item or "kind" not in item or "text" not in item:
            raise ValueError("domain knowledge notes entry requires key, kind, and text")
        key = str(item.get("key") or "").strip()
        text_val = str(item.get("text") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not key or not text_val or not kind:
            raise ValueError("domain knowledge notes entry key, kind, and text must be non-empty")
        if kind not in allowed_kinds:
            raise ValueError(f"unknown domain knowledge kind: {kind!r}")
        if "referenced_entities" not in item:
            raise ValueError("domain knowledge notes entry requires referenced_entities")
        raw_referenced = item.get("referenced_entities")
        if not isinstance(raw_referenced, list) or not all(isinstance(r, str) for r in raw_referenced):
            raise ValueError("domain knowledge notes entry referenced_entities must be a list of strings")
        referenced_entities = frozenset(r.strip() for r in raw_referenced if r.strip())
        entry = DomainKnowledgeEntry.normalize(
            DomainKnowledgeEntry(key=key, text=text_val, kind=kind, referenced_entities=referenced_entities)
        )
        dedupe = f"{entry.kind}::{entry.key}::{entry.text.lower()}"
        if dedupe in seen:
            continue
        seen.add(dedupe)
        candidates.append(entry)
    return tuple(candidates)


def _coerce_domain_knowledge_notes_items(parsed: Any) -> list[Any]:
    """Accept only ``{"entries": [...]}`` with no other top-level keys."""
    if not isinstance(parsed, dict):
        raise ValueError('domain knowledge notes JSON must be {"entries": [...]}')
    _reject_unexpected_keys(parsed, DOMAIN_KNOWLEDGE_TOP_KEYS, context="domain knowledge notes payload")
    entries = parsed.get("entries")
    if not isinstance(entries, list):
        raise ValueError("domain knowledge notes entries must be a list")
    return entries


def _evaluate_domain_knowledge_notes_raw(
    raw: str,
    schema: SchemaGraph,
) -> tuple[str, tuple[DomainKnowledgeEntry, ...]]:
    """Parse/normalize/filter one LLM raw string. ``ok`` includes legitimate empty entries; ``invalid_shape`` on parse errors; ``empty_after_filter`` when every entry names a hidden column."""
    parsed = safe_json_loads(raw)
    try:
        items = _coerce_domain_knowledge_notes_items(parsed)
        if len(items) == 0:
            return "ok", ()
        candidates = _normalize_domain_knowledge_notes_items(items)
        for entry in candidates:
            undeclared = DomainKnowledgeEntry.undeclared_schema_identifier_references(
                entry.text, entry.referenced_entities, schema
            )
            if undeclared:
                raise ConfigError(
                    f"domain knowledge entry {entry.key!r} text names schema identifier(s) "
                    f"not declared in referenced_entities: {undeclared[0]!r}"
                )
    except (ValueError, ConfigError, TypeError):
        return "invalid_shape", ()
    kept = filter_schema_anchored_domain_knowledge(candidates, schema)
    if not kept:
        return "empty_after_filter", ()
    return "ok", kept


def _domain_knowledge_notes_repair_message(status: str) -> str:
    if status == "invalid_shape":
        return 'previous_raw must parse to JSON {"entries": [{"key","kind","text"}, ...]} with exactly those top-level and entry keys; kind must be glossary|policy|metric|synonym|caveat.'
    if status == "empty_after_filter":
        return (
            "previous entries were dropped by the security filter because they named specific relation or field identifiers. "
            "Re-emit definitions, policies, metrics, synonyms, and caveats from domain_notes without naming those identifiers; prefer sparse concept-slug keys (not one entry per relation); omit pure schema inventory."
        )
    return "previous_raw failed validation; emit a corrected JSON object."


_EXTRACTION_MEMO: dict[tuple[str, tuple[str, ...]], NotesExtractionResult] = {}


def assert_notes_coverage_total(notes_content: str, ledger: NotesExtractionLedger) -> None:
    """Raise when coverage spans do not partition *notes_content* exactly."""
    notes = str(notes_content or "")
    if not notes.strip():
        return
    if not ledger.entries:
        raise ConfigError("notes extraction coverage ledger is empty")
    pos = 0
    for entry in ledger.entries:
        span = entry.span
        if not span:
            raise ConfigError("notes extraction coverage span must be non-empty")
        idx = notes.find(span, pos)
        if idx < 0:
            raise ConfigError(f"notes extraction coverage span not found in notes: {span!r}")
        if idx > pos and notes[pos:idx].strip():
            raise ConfigError("notes extraction coverage leaves uncovered content")
        pos = idx + len(span)
    if notes[pos:].strip():
        raise ConfigError("notes extraction coverage does not reach end of notes")


def _normalize_notes_coverage_items(parsed: Sequence[Any]) -> tuple[NotesCoverageEntry, ...]:
    out: list[NotesCoverageEntry] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("coverage item must be an object")
        _reject_unexpected_keys(item, KNOWLEDGE_NOTES_COVERAGE_ENTRY_KEYS, context="notes coverage item")
        span = str(item.get("span") or "")
        if span == "":
            raise ValueError("coverage item span must be non-empty")
        disposition = str(item.get("disposition") or "").strip().lower()
        if disposition not in {"fact", "no_fact"}:
            raise ValueError("coverage item requires disposition fact|no_fact")
        record_index_raw = item.get("record_index")
        record_index: int | None
        if disposition == "fact":
            if not isinstance(record_index_raw, int) or record_index_raw < 0:
                raise ValueError("coverage fact disposition requires non-negative record_index")
            record_index = record_index_raw
        else:
            if record_index_raw is not None:
                raise ValueError("coverage no_fact disposition must not include record_index")
            record_index = None
        out.append(NotesCoverageEntry(span=span, disposition=disposition, record_index=record_index))
    return tuple(out)


def _knowledge_record_from_notes_item(
    item: Mapping[str, Any],
    *,
    allowed_entities: frozenset[str],
    schema: SchemaGraph,
) -> tuple[DomainKnowledgeEntry | None, StructuralKnowledgeFact | None]:
    if not isinstance(item, dict):
        raise ValueError("knowledge notes record must be an object")
    _reject_unexpected_keys(item, KNOWLEDGE_NOTES_RECORD_KEYS, context="knowledge notes record")
    if "kind" not in item or "text" not in item or "referenced_entities" not in item:
        raise ValueError("knowledge notes record requires kind, text, and referenced_entities")
    kind = str(item.get("kind") or "").strip().lower()
    text_val = str(item.get("text") or "").strip()
    if not kind or not text_val:
        raise ValueError("knowledge notes record kind and text must be non-empty")
    raw_referenced = item.get("referenced_entities")
    if not isinstance(raw_referenced, list) or not all(isinstance(r, str) for r in raw_referenced):
        raise ValueError("knowledge notes record referenced_entities must be a list of strings")
    referenced_entities = frozenset(r.strip() for r in raw_referenced if r.strip())
    if referenced_entities:
        unknown = sorted(r for r in referenced_entities if r not in allowed_entities)
        if unknown:
            raise ValueError(f"knowledge notes referenced_entities not in schema_names whitelist: {unknown}")
        if kind == "residual":
            raise ValueError("structural knowledge kind residual is not supported")
        allowed_kinds = {member.value for member in StructuralKnowledgeKind}
        if kind not in allowed_kinds:
            raise ValueError(f"unknown structural knowledge kind: {kind!r}")
        payload_raw = item.get("payload")
        if payload_raw is None:
            payload: dict[str, Any] | None = None if "payload" not in item else {}
        elif isinstance(payload_raw, dict):
            payload = payload_raw
        else:
            raise ValueError("knowledge notes record payload must be an object when present")
        fact = StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(
                kind=kind,
                text=text_val,
                referenced_entities=referenced_entities,
                payload=payload,
            )
        )
        return None, fact
    allowed_kinds = {member.value for member in DomainKnowledgeKind}
    if kind not in allowed_kinds:
        raise ValueError(f"unknown domain knowledge kind: {kind!r}")
    key = str(item.get("key") or "").strip()
    if not key:
        raise ValueError("unanchored knowledge notes record requires non-empty key")
    entry = DomainKnowledgeEntry.normalize(
        DomainKnowledgeEntry(key=key, text=text_val, kind=kind, referenced_entities=referenced_entities)
    )
    undeclared = DomainKnowledgeEntry.undeclared_schema_identifier_references(
        entry.text, entry.referenced_entities, schema
    )
    if undeclared:
        raise ValueError(
            f"domain knowledge entry {entry.key!r} text names schema identifier(s) "
            f"not declared in referenced_entities: {undeclared[0]!r}"
        )
    return entry, None


def _strict_parse_knowledge_notes_payload(
    parsed: Any,
    *,
    allowed_entities: frozenset[str],
    schema: SchemaGraph,
) -> tuple[
    tuple[DomainKnowledgeEntry, ...],
    tuple[StructuralKnowledgeFact, ...],
    NotesExtractionLedger,
    tuple[tuple[str, DomainKnowledgeEntry | StructuralKnowledgeFact], ...],
]:
    if not isinstance(parsed, dict):
        raise ValueError('knowledge notes JSON must be {"records": [...], "coverage": [...]}')
    _reject_unexpected_keys(parsed, KNOWLEDGE_NOTES_TOP_KEYS, context="knowledge notes payload")
    records_raw = parsed.get("records")
    coverage_raw = parsed.get("coverage")
    if not isinstance(records_raw, list):
        raise ValueError("knowledge notes records must be a list")
    if not isinstance(coverage_raw, list):
        raise ValueError("knowledge notes coverage must be a list")
    ledger = NotesExtractionLedger(entries=_normalize_notes_coverage_items(coverage_raw))
    domain_entries: list[DomainKnowledgeEntry] = []
    structural_facts: list[StructuralKnowledgeFact] = []
    record_stream: list[tuple[str, DomainKnowledgeEntry | StructuralKnowledgeFact]] = []
    seen_domain: set[str] = set()
    seen_structural: set[str] = set()
    for item in records_raw:
        entry, fact = _knowledge_record_from_notes_item(item, allowed_entities=allowed_entities, schema=schema)
        if entry is not None:
            record_stream.append(("domain", entry))
            dedupe = f"{entry.kind}::{entry.key}::{entry.text.lower()}"
            if dedupe in seen_domain:
                continue
            seen_domain.add(dedupe)
            domain_entries.append(entry)
        elif fact is not None:
            record_stream.append(("structural", fact))
            dedupe = f"{fact.kind}::{fact.text.lower()}"
            if dedupe in seen_structural:
                continue
            seen_structural.add(dedupe)
            structural_facts.append(fact)
    for coverage_entry in ledger.entries:
        if coverage_entry.disposition == "fact":
            if coverage_entry.record_index is None or coverage_entry.record_index >= len(records_raw):
                raise ValueError("coverage record_index out of range for records list")
    return tuple(domain_entries), tuple(structural_facts), ledger, tuple(record_stream)


def _evaluate_knowledge_notes_raw(
    raw: str,
    *,
    notes_content: str,
    allowed_entities: frozenset[str],
    schema: SchemaGraph,
) -> tuple[str, NotesExtractionResult]:
    parsed = safe_json_loads(raw)
    try:
        domain_entries, structural_facts, ledger, record_stream = _strict_parse_knowledge_notes_payload(
            parsed, allowed_entities=allowed_entities, schema=schema
        )
        assert_notes_coverage_total(notes_content, ledger)
    except (ValueError, ConfigError, TypeError):
        return "invalid_shape", NotesExtractionResult((), (), NotesExtractionLedger(()))
    filtered_domain = filter_schema_anchored_domain_knowledge(domain_entries, schema)
    filtered_structural = filter_schema_anchored_structural_knowledge(structural_facts, schema)
    if (domain_entries and not filtered_domain) or (structural_facts and not filtered_structural):
        return "empty_after_filter", NotesExtractionResult((), (), NotesExtractionLedger(()))
    return (
        "ok",
        NotesExtractionResult(
            domain_knowledge=filtered_domain,
            structural_knowledge=filtered_structural,
            ledger=ledger,
            record_stream=record_stream,
        ),
    )


def _knowledge_notes_repair_message(status: str) -> str:
    if status == "invalid_shape":
        return (
            'previous_raw must parse to JSON {"records":[{"key","kind","text","referenced_entities","payload"?},...],'
            '"coverage":[{"span","disposition","record_index"?},...]} with coverage partitioning domain_notes exactly; '
            "anchored records use structural kinds with non-empty referenced_entities from schema_names; "
            "unanchored records use glossary|policy|metric|synonym|caveat with empty referenced_entities and non-empty key."
        )
    if status == "empty_after_filter":
        return (
            "previous records were dropped by the security filter because they named hidden identifiers. "
            "Re-emit facts without naming those identifiers."
        )
    return "previous_raw failed validation; emit a corrected JSON object."


def extract_knowledge_from_notes(
    notes_content: str | None,
    schema: SchemaGraph,
) -> NotesExtractionResult:
    """Single-pass notes extraction: domain and structural knowledge plus a coverage ledger."""
    notes_stripped = (notes_content or "").strip()
    if not notes_stripped:
        return NotesExtractionResult((), (), NotesExtractionLedger(()))
    if not EngineConfig.llm_credentials_configured():
        return NotesExtractionResult((), (), NotesExtractionLedger(()))
    schema_names = schema_name_dump_for_domain_knowledge(schema)
    memo_key = (hashlib.sha256(notes_stripped.encode("utf-8")).hexdigest(), schema_names)
    cached = _EXTRACTION_MEMO.get(memo_key)
    if cached is not None:
        return cached
    allowed_entities = frozenset(schema_names)
    user_payload = stable_json({"domain_notes": notes_stripped, "schema_names": list(schema_names)})
    max_attempts = min(3, max(1, int(KNOWLEDGE_NOTES_EXTRACT_MAX_ATTEMPTS)))
    last_status = "invalid_shape"
    last_raw = "{}"
    with llm_usage_build_scope():
        for attempt in range(max_attempts):
            if attempt == 0:
                raw = LLMProvider.chat(
                    KNOWLEDGE_NOTES_EXTRACT_SYSTEM,
                    user_payload,
                    timeout=360.0,
                    task="schema",
                )
            else:
                repair_payload = stable_json(
                    {
                        "domain_notes": notes_stripped,
                        "schema_names": list(schema_names),
                        "validation_error": _knowledge_notes_repair_message(last_status),
                        "previous_raw": last_raw,
                    }
                )
                debug(
                    f"[extract_knowledge_from_notes] repair attempt={attempt + 1}/{max_attempts} status={last_status}"
                )
                raw = LLMProvider.chat(
                    KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM,
                    repair_payload,
                    timeout=360.0,
                    task="schema",
                )
            last_raw = raw if isinstance(raw, str) else str(raw)
            status, result = _evaluate_knowledge_notes_raw(
                last_raw,
                notes_content=notes_stripped,
                allowed_entities=allowed_entities,
                schema=schema,
            )
            last_status = status
            debug(
                f"[extract_knowledge_from_notes] attempt={attempt + 1}/{max_attempts} "
                f"status={status} dk={len(result.domain_knowledge)} sk={len(result.structural_knowledge)}"
            )
            notify(
                f"Knowledge notes extract attempt {attempt + 1}/{max_attempts}: "
                f"status={status} dk={len(result.domain_knowledge)} sk={len(result.structural_knowledge)}",
                stage="schema",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
            )
            if status == "ok":
                _EXTRACTION_MEMO[memo_key] = result
                return result
    debug(f"[extract_knowledge_from_notes] exhausted attempts={max_attempts} last_status={last_status}")
    notify(
        f"Knowledge notes extract exhausted ({max_attempts} attempts); last_status={last_status}",
        stage="schema",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    empty = NotesExtractionResult((), (), NotesExtractionLedger(()))
    _EXTRACTION_MEMO[memo_key] = empty
    return empty


def extract_domain_knowledge_from_notes(
    notes_content: str | None,
    schema: SchemaGraph,
) -> tuple[DomainKnowledgeEntry, ...]:
    """Thin wrapper over :func:`extract_knowledge_from_notes` returning domain knowledge only."""
    return extract_knowledge_from_notes(notes_content, schema).domain_knowledge


def _structural_fact_from_notes_item(
    item: Mapping[str, Any],
    *,
    allowed_entities: frozenset[str] | None = None,
) -> StructuralKnowledgeFact:
    """Parse one structural-knowledge notes fact object; raises on invalid shape."""
    if not isinstance(item, dict):
        raise ValueError("structural knowledge notes fact must be an object")
    _reject_unexpected_keys(item, STRUCTURAL_KNOWLEDGE_FACT_KEYS, context="structural knowledge notes fact")
    if "kind" not in item or "text" not in item or "referenced_entities" not in item:
        raise ValueError("structural knowledge notes fact requires kind, text, and referenced_entities")
    kind = str(item.get("kind") or "").strip().lower()
    text_val = str(item.get("text") or "").strip()
    if not kind or not text_val:
        raise ValueError("structural knowledge notes fact kind and text must be non-empty")
    if kind == "residual":
        raise ValueError("structural knowledge kind residual is not supported")
    allowed_kinds = {member.value for member in StructuralKnowledgeKind}
    if kind not in allowed_kinds:
        raise ValueError(f"unknown structural knowledge kind: {kind!r}")
    raw_referenced = item.get("referenced_entities")
    if not isinstance(raw_referenced, list) or not all(isinstance(r, str) for r in raw_referenced):
        raise ValueError("structural knowledge notes fact referenced_entities must be a list of strings")
    referenced_entities = frozenset(r.strip() for r in raw_referenced if r.strip())
    if not referenced_entities:
        raise ValueError("structural knowledge notes fact referenced_entities must be non-empty")
    if allowed_entities is not None:
        unknown = sorted(r for r in referenced_entities if r not in allowed_entities)
        if unknown:
            raise ValueError(f"structural knowledge referenced_entities not in schema_names whitelist: {unknown}")
    payload_raw = item.get("payload")
    if payload_raw is None:
        payload: dict[str, Any] | None = None if "payload" not in item else {}
    elif isinstance(payload_raw, dict):
        payload = payload_raw
    else:
        raise ValueError("structural knowledge notes fact payload must be an object when present")
    return StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=kind,
            text=text_val,
            referenced_entities=referenced_entities,
            payload=payload,
        )
    )


def _strict_parse_structural_facts_payload(
    parsed: Any,
    *,
    allowed_entities: frozenset[str] | None = None,
) -> tuple[StructuralKnowledgeFact, ...]:
    """Accept only ``{"facts": [...]}`` with exact keys, known kinds, and non-empty reference sets."""
    if not isinstance(parsed, dict):
        raise ValueError('structural knowledge notes JSON must be {"facts": [...]}')
    _reject_unexpected_keys(parsed, STRUCTURAL_KNOWLEDGE_TOP_KEYS, context="structural knowledge notes payload")
    facts_raw = parsed.get("facts")
    if not isinstance(facts_raw, list):
        raise ValueError("structural knowledge notes facts must be a list")
    out: list[StructuralKnowledgeFact] = []
    seen: set[str] = set()
    for item in facts_raw:
        fact = _structural_fact_from_notes_item(item, allowed_entities=allowed_entities)
        dedupe = f"{fact.kind}::{fact.text.lower()}"
        if dedupe in seen:
            continue
        seen.add(dedupe)
        out.append(fact)
    return tuple(out)


def _evaluate_structural_knowledge_notes_raw(
    raw: str,
    *,
    allowed_entities: frozenset[str] | None = None,
) -> tuple[str, tuple[StructuralKnowledgeFact, ...]]:
    """Parse one structural-knowledge LLM raw string. ``ok`` includes legitimate empty facts; ``invalid_shape`` on parse errors."""
    parsed = safe_json_loads(raw)
    try:
        facts = _strict_parse_structural_facts_payload(parsed, allowed_entities=allowed_entities)
    except (ValueError, ConfigError, TypeError):
        return "invalid_shape", ()
    return "ok", facts


def _structural_knowledge_notes_repair_message(status: str) -> str:
    if status == "invalid_shape":
        return 'previous_raw must parse to JSON {"facts": [{"kind","text","referenced_entities":[...],"payload"?}, ...]} with exactly those top-level and fact keys (referenced_entities required and non-empty, drawn from schema_names); kind must be one of relation|field|join|grain|cardinality|lifecycle|declared_value_set|sentinel_semantics|unit_of_measure|relation_shape|term_binding|period_convention|concept_absence.'
    return "previous_raw failed validation; emit a corrected JSON object."


def extract_structural_knowledge_from_notes(
    notes_content: str | None,
    schema: SchemaGraph,
) -> tuple[StructuralKnowledgeFact, ...]:
    """Thin wrapper over :func:`extract_knowledge_from_notes` returning structural knowledge only."""
    return extract_knowledge_from_notes(notes_content, schema).structural_knowledge


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
    """Deterministically promote two-value columns to boolean using literals and affix rules."""
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
    """Assign valid filter, aggregation, and HAVING ops to each column based on its final role."""
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


def rerun_column_classifier(
    sg: SchemaGraph,
    notes_content: str | None,
    *,
    skip_columns: set[tuple[str, str]] | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """Re-run the LLM column-role classifier and boolean coercion over *sg* in place."""
    debug(f"[schema.rerun_column_classifier] reclassifying {len(sg.tables)} tables")
    restore_descriptions_from_base(sg)
    apply_column_roles_llm(sg, notes_content=notes_content, skip_columns=skip_columns, log_sink=log_sink)
    apply_boolean_coercion_pass(sg)
    assign_column_ops(sg)


def profile_table_clone(dialect: Any, table: TableMetadata, notes_content: str | None) -> TableMetadata | None:
    """Deep-copy *table*, profile and classify it, and return the clone or ``None`` on failure."""
    clone = copy.deepcopy(table)
    tmp_sg = SchemaGraph(tables={clone.name: clone}, join_paths_multi={})
    try:
        dialect.profile_schema(tmp_sg)
        apply_column_roles_llm(tmp_sg, notes_content=notes_content)
        apply_boolean_coercion_pass(tmp_sg)
        assign_column_ops(tmp_sg)
    except Exception as exc:
        table.profile_failed = True
        notify(
            f"profile clone failed for table {table.name!r}: {exc}",
            stage="schema",
            code=DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED,
            level="warning",
            details=(("table", table.name),),
        )
        return None
    return clone


def extract_partition_columns_from_describe_detail_spark(spark: Any, full_table: str) -> list[str]:
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


def extract_partition_columns_from_describe_detail_sql_connector(connection: Any, full_table: str) -> list[str]:
    """Extract partition column names via DESCRIBE DETAIL, with. INFORMATION_SCHEMA fallback."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE DETAIL {full_table}")
            rows = cursor_rows_as_dicts(cursor)
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
                info_rows = cursor_rows_as_dicts(cursor)
            return [str(r["column_name"]) for r in info_rows if r.get("column_name")]
    except Exception as e:
        debug(f"[schema_profiling._extract_partition_sql_connector] INFORMATION_SCHEMA failed: {e}")

    return []


def parse_partition_columns_from_create_stmt(create_stmt: str) -> list[str]:
    """Extract partition column names from a CREATE TABLE DDL string."""
    match = re.search(r"PARTITIONED\s+BY\s*\(([^)]+)\)", create_stmt, re.IGNORECASE)
    if not match:
        return []
    return [c.strip().strip("`").strip('"') for c in match.group(1).split(",")]


def _partition_column_names_from_create_ddl(create_stmt_sql: str) -> list[str]:
    """Extract declarative partition columns from Hive-style or. PostgreSQL CREATE DDL."""
    spark_cols = parse_partition_columns_from_create_stmt(create_stmt_sql)
    if spark_cols:
        return spark_cols
    match = re.search(
        r"\bPARTITION\s+BY\s+(?:RANGE|LIST|HASH)\s*\(\s*([^)]+)\s*\)", create_stmt_sql, re.IGNORECASE | re.DOTALL
    )
    if not match:
        return []
    return [c.strip().strip("`").strip('"') for c in match.group(1).split(",") if c.strip()]


def parse_catalog_constraints_from_ddl(create_stmt: str) -> tuple[list[str], list[dict[str, Any]], list[str]]:
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
    """Return ``(open_paren_idx, index_after_closing_paren)`` when parentheses balance from *open_paren_idx*. Depth scanning does not model nested quotes; unusual DDL with parentheses inside literal defaults may mis-scan."""
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
    """Extract CREATE TABLE shapes using bracket balancing when structured parsers return nothing."""
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
    """Return True when any reflected table already carries at least one. FK edge."""
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
    """Invoke JSON-mode DDL parsing when deterministic parsers produced nothing."""
    debug("[schema_profiling.parse_sql_file] structured parsers returned 0 tables, falling back to LLM")

    system = """You are a deterministic SQL parser. Extract CREATE TABLE and ALTER TABLE metadata and output ONLY valid JSON. Be precise and consistent. Follow the output format exactly."""

    user = stable_json(
        {
            "task": "Parse CREATE TABLE and ALTER TABLE ADD COLUMN / ADD CONSTRAINT from the SQL and merge into one metadata dict per table in file order",
            "sql_content": sql_content,
            "output_format": {
                "tables": {
                    "table": {
                        "table_name_original": "exact table name from SQL (without schema prefix)",
                        "column_names_original": ["column", "other_column"],
                        "column_types": ["TYPE1", "TYPE2"],
                        "column_not_null": [True, False],
                        "column_unique": [True, False],
                        "primary_keys": ["column"],
                        "foreign_keys": [
                            {
                                "src_cols": ["column"],
                                "dst_table": "other_table",
                                "dst_cols": ["column"],
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
                    "input": "CREATE TABLE public.table (column SERIAL PRIMARY KEY, other_column VARCHAR(100));",
                    "output": {
                        "tables": {
                            "table": {
                                "table_name_original": "table",
                                "column_names_original": ["column", "other_column"],
                                "column_types": ["INTEGER", "VARCHAR(100)"],
                                "column_not_null": [True, False],
                                "column_unique": [True, False],
                                "primary_keys": ["column"],
                                "foreign_keys": [],
                            }
                        }
                    },
                },
                {
                    "input": "CREATE TABLE other_table (column INT, other_column INT, FOREIGN KEY (other_column) REFERENCES table(column));",
                    "output": {
                        "tables": {
                            "other_table": {
                                "table_name_original": "other_table",
                                "column_names_original": ["column", "other_column"],
                                "column_types": ["INT", "INT"],
                                "column_not_null": [False, False],
                                "column_unique": [False, False],
                                "primary_keys": [],
                                "foreign_keys": [
                                    {
                                        "src_cols": ["other_column"],
                                        "dst_table": "table",
                                        "dst_cols": ["column"],
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

    if not isinstance(parsed, dict):
        raise ValueError(f"[schema_profiling.parse_sql_file] LLM JSON is not an object; got {type(parsed).__name__}")
    if "tables" not in parsed:
        raise ValueError("[schema_profiling.parse_sql_file] LLM JSON missing 'tables' key")

    tables = parsed["tables"]
    if not isinstance(tables, dict):
        raise ValueError(f"[schema_profiling.parse_sql_file] 'tables' must be a dict; got {type(tables).__name__}")
    debug(f"[schema_profiling.parse_sql_file] llm parsed: {len(tables)} tables")

    if diagnostic_debug_enabled():
        for tname, tinfo in tables.items():
            if isinstance(tinfo, dict):
                debug(
                    f"[schema_profiling.parse_sql_file] table: {tname} cols={len(tinfo.get('column_names_original', []))}"
                )

    out_tables: dict[str, dict[str, Any]] = {}
    for tname, tinfo in tables.items():
        if isinstance(tinfo, dict):
            out_tables[str(tname)] = _canonicalize_llm_ddl_table_row(tinfo)
        else:
            raise ValueError(
                f"[schema_profiling.parse_sql_file] table entry {tname!r} must be a dict; got {type(tinfo).__name__}"
            )
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
            "[schema_profiling.parse_sql_file] deterministic parsers returned 0 tables; reflection already has FK edges — skipping DDL LLM fallback"
        )
        return {}

    return _finalize(_parse_sql_file_via_llm(sql_content))


def _parse_sql_file_fallback(sql_content: str) -> dict[str, dict[str, Any]]:
    """Parse CREATE TABLE and ALTER TABLE metadata from DDL using the active engine profile."""
    dialect_cls = _dialect_registry().get_dialect_class(EngineConfig.TYPE)
    dialect_stub = dialect_cls.__new__(dialect_cls)
    token = dialect_stub.sql_file_parse_dialect
    if token == "postgres":
        return _parse_sql_file_pglast_postgres(sql_content)
    return _parse_sql_file_sqlglot(sql_content, token)


def _extract_column_block_from_create(create_expr: Any) -> str:
    """Extract the inner content of the column definition block from a sqlglot Create."""
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
    """Split a single DDL column line into name and full SQL type tokens."""
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
    """Parse column definitions and constraints from a column block string."""
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
    """Extract column names from a PRIMARY KEY (column, other_column) definition line."""
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
    """Build a single-table schema metadata dict from column text and full CREATE DDL."""
    columns, types, pks, fks, uniqs, col_nullable, column_comments = _parse_columns_and_constraints(col_block)
    cat_pks, cat_fks, cat_uniqs = parse_catalog_constraints_from_ddl(full_create_sql)
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
    """Append a column and type to table DDL metadata when the column is not already present."""
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
    """Append a foreign-key edge dict if an identical edge is not already recorded."""
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
    """Merge PRIMARY KEY, FOREIGN KEY, or single-column UNIQUE from a pglast `Constraint`."""
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
    """Merge PRIMARY KEY, FOREIGN KEY, or single-column UNIQUE from a sqlglot constraint expression."""
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
    """Parse PostgreSQL-flavour CREATE TABLE nodes from DDL text using pglast."""
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
    """Parse CREATE TABLE statements from DDL text using the given sqlglot dialect token."""
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


def replay_user_semantic_neighbors_to_columns(sg: SchemaGraph) -> None:
    """Mirror every quad in :attr:`TableMetadata._user_semantic_neighbors` onto the per-column ``semantic_join_neighbors`` lists. The quad list on each table is the single authoritative store for user-overridden semantic edges; the per-column lists are a derived read view consumed by join-graph traversal. Call this helper after appending new quads (for example in :func:`apply_structure_to_graph`) so the per-column projection stays in sync without needing to re-run profile-derived overlap discovery. Idempotent: existing per-column entries are not duplicated and the lists are returned in a stable sort order."""
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


def infer_view_same_name_key_edges(sg: SchemaGraph) -> int:
    """Add structural FK edges between views that share identically named ``*_id`` columns. Profile-derived semantic neighbors only accept string columns, so numeric key columns shared across analytical views never become join edges. This helper adds those edges for views-only graphs so multi- view routing stays possible. Returns the number of edges appended."""
    view_names = sorted(name for name, table in sg.tables.items() if table.kind == TableKind.VIEW)
    if len(view_names) < 2:
        return 0
    if any(table.kind != TableKind.VIEW for table in sg.tables.values()):
        return 0
    by_column: dict[str, list[str]] = {}
    for name in view_names:
        for col_name in sg.tables[name].columns:
            if not col_name.lower().endswith("_id"):
                continue
            by_column.setdefault(col_name, []).append(name)
    added = 0
    for col_name, owners in sorted(by_column.items()):
        if len(owners) < 2:
            continue
        dest = owners[0]
        for src in owners[1:]:
            src_tbl = sg.tables[src]
            if any(
                edge.dst_table == dest and edge.src_cols == [col_name] and edge.dst_cols == [col_name]
                for edge in src_tbl.foreign_keys
            ):
                continue
            src_tbl.foreign_keys.append(
                FKEdge(
                    src_table=src,
                    src_cols=[col_name],
                    dst_table=dest,
                    dst_cols=[col_name],
                    inference_tag=InferenceTag.SUFFIX,
                )
            )
            added += 1
    return added


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


def out_of_scope_description_tokens(full_graph: SchemaGraph, scope: KnowledgeScope) -> frozenset[str]:
    """Identifier vocabulary present on *full_graph* but absent from *scope* — the names a scoped description must never contain. A bare column name shared with an in-scope column is left out (ambiguous, and the in-scope reading is legitimate)."""
    in_scope_column_names = {qc.split(".", 1)[1].strip().lower() for qc in scope.columns if "." in qc}
    tokens: set[str] = set()
    for table_name, table in full_graph.tables.items():
        if not scope.contains(table_name):
            tokens.add(table_name)
            original = (table.original_name or "").strip()
            if original:
                tokens.add(original)
        for column_name, column in table.columns.items():
            if scope.contains(f"{table_name}.{column_name}"):
                continue
            if column_name.strip().lower() in in_scope_column_names:
                continue
            tokens.add(column_name)
            col_original = (column.original_name or "").strip()
            if col_original and col_original.strip().lower() not in in_scope_column_names:
                tokens.add(col_original)
    return frozenset(tokens)


def raise_if_schema_graph_descriptions_name_out_of_scope_entities(
    scoped_graph: SchemaGraph,
    tokens: frozenset[str],
    *,
    context: str = "scoped description",
) -> None:
    """Hard-fail when a table/column description on *scoped_graph* contains a token from *tokens* (computed via :func:`out_of_scope_description_tokens`). *scoped_graph* is assumed already narrowed to the caller's own entity set."""
    if not tokens:
        return
    for table_name, table in scoped_graph.tables.items():
        if table.description:
            hits = description_neutrality_violations(table.description, tokens)
            if hits:
                raise ConfigError(f"{context} for table {table_name!r} names an out-of-scope identifier: {hits[0]!r}")
        for column_name, column in table.columns.items():
            if column.description:
                hits = description_neutrality_violations(column.description, tokens)
                if hits:
                    raise ConfigError(
                        f"{context} for column {table_name}.{column_name} names an out-of-scope identifier: {hits[0]!r}"
                    )


def raise_if_flat_descriptions_name_out_of_scope_entities(
    table_descriptions: Mapping[str, str],
    column_meta: Mapping[str, Mapping[str, Any]],
    tokens: frozenset[str],
    *,
    context: str = "scoped description",
) -> None:
    """Hard-fail when a persisted-snapshot description string names a token from *tokens*. Same check as :func:`raise_if_schema_graph_descriptions_name_out_of_scope_entities` but over the flat ``table_descriptions``/``column_meta`` dict shape a space snapshot stores."""
    if not tokens:
        return
    for table_name, desc in table_descriptions.items():
        hits = description_neutrality_violations(str(desc or ""), tokens)
        if hits:
            raise ConfigError(f"{context} for table {table_name!r} names an out-of-scope identifier: {hits[0]!r}")
    for qualified_column, meta in column_meta.items():
        column_desc = meta.get("description") if isinstance(meta, Mapping) else None
        hits = description_neutrality_violations(str(column_desc or ""), tokens)
        if hits:
            raise ConfigError(
                f"{context} for column {qualified_column!r} names an out-of-scope identifier: {hits[0]!r}"
            )


def clear_descriptions_naming_out_of_scope_entities(graph: SchemaGraph, tokens: frozenset[str]) -> int:
    """Blank any table/column description on *graph* that names a token from *tokens*. Deterministic fail-closed fallback for callers with no LLM path to re-derive a clean description; returns the number of descriptions cleared."""
    if not tokens:
        return 0
    cleared = 0
    for table in graph.tables.values():
        if table.description and description_neutrality_violations(table.description, tokens):
            table.description = ""
            cleared += 1
        for column in table.columns.values():
            if column.description and description_neutrality_violations(column.description, tokens):
                column.description = ""
                cleared += 1
    return cleared


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


def column_overlap_comparison_mode(left: ColumnMetadata, right: ColumnMetadata) -> OverlapComparison:
    """Return the overlap comparison rule when pairing two profiled columns."""
    if left.is_case_insensitive_collation or right.is_case_insensitive_collation:
        return OverlapComparison.CASE_FOLDED
    return OverlapComparison.EXACT


def _normalize_overlap_sample_value(value: object, *, case_fold: bool, rtrim_pad: bool = False) -> str | None:
    if value is None:
        return None
    s = normalize_text_value(str(value))
    if rtrim_pad:
        s = s.rstrip()
    else:
        s = s.strip()
    return s.casefold() if case_fold else s


def normalized_value_overlap_sets(
    left: ColumnMetadata,
    right: ColumnMetadata,
    *,
    record_comparison: bool = True,
) -> tuple[set[str], set[str], OverlapComparison]:
    """Normalize overlap samples for a pair, optionally recording the comparison rule used."""
    mode = column_overlap_comparison_mode(left, right)
    fold = mode == "case_folded"
    rtrim_pad = left.is_fixed_width_text or right.is_fixed_width_text
    left_set = {
        normalized
        for v in left.value_overlap_sample or []
        if (normalized := _normalize_overlap_sample_value(v, case_fold=fold, rtrim_pad=rtrim_pad)) is not None
    }
    right_set = {
        normalized
        for v in right.value_overlap_sample or []
        if (normalized := _normalize_overlap_sample_value(v, case_fold=fold, rtrim_pad=rtrim_pad)) is not None
    }
    if record_comparison:
        left.overlap_comparison = mode
        right.overlap_comparison = mode
    return left_set, right_set, mode


def value_overlap_ratio_for_columns(left: ColumnMetadata, right: ColumnMetadata) -> float:
    """Return overlap ratio for two columns using collation-aware sample comparison."""
    left_set, right_set, _ = normalized_value_overlap_sets(left, right)
    if not left_set or not right_set:
        return 0.0
    inter = len(left_set & right_set)
    return inter / float(min(len(left_set), len(right_set)))


def _col_value_overlap_frozen(col: ColumnMetadata) -> frozenset[str]:
    vals = col.value_overlap_sample or []
    cleaned = {normalize_text_value(str(v).strip()) for v in vals if v is not None}
    cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
    return frozenset(sorted(cleaned)[:cap])


def profiling_value_overlap(older: SchemaGraph, newer: SchemaGraph) -> float:
    """Aggregate Jaccard overlap of value-overlap samples on shared ``(table, column)`` keys."""
    inter = 0
    union = 0
    for t in older.tables:
        if t not in newer.tables:
            continue
        ot = older.tables[t]
        nt = newer.tables[t]
        for c in ot.columns:
            if c not in nt.columns:
                continue
            a = _col_value_overlap_frozen(ot.columns[c])
            b = _col_value_overlap_frozen(nt.columns[c])
            u = len(a | b)
            if u == 0:
                continue
            union += u
            inter += len(a & b)
    if union == 0:
        return 1.0
    return inter / union


def column_eligible_for_space_allowlist(col: ColumnMetadata | None) -> bool:
    """Return False for columns that must never appear on an aetherspace allow-list."""
    if col is None or col.is_denied:
        return False
    if col.sensitivity in (SensitivityClassification.HIDDEN, SensitivityClassification.RESTRICTED):
        return False
    return True


def filter_space_snapshot_sensitive_columns(
    snapshot: Mapping[str, Any],
    schema_graph: SchemaGraph,
) -> dict[str, Any]:
    """Drop HIDDEN/RESTRICTED/denied columns from a space snapshot's allow-list and column_meta."""
    out = dict(snapshot)
    raw_columns = out.get("columns") or ()
    kept_columns: list[str] = []
    if isinstance(raw_columns, (list, tuple)):
        for spec in raw_columns:
            raw = str(spec).strip()
            if "." not in raw:
                continue
            table_name, column_name = raw.split(".", 1)
            table_name = table_name.strip()
            column_name = column_name.strip()
            if not table_name or not column_name:
                continue
            tm = schema_graph.tables.get(table_name)
            col_meta = tm.columns.get(column_name) if tm is not None else None
            if not column_eligible_for_space_allowlist(col_meta):
                continue
            kept_columns.append(f"{table_name}.{column_name}")
    out["columns"] = sorted(set(kept_columns))
    raw_meta = out.get("column_meta")
    if isinstance(raw_meta, Mapping):
        filtered_meta: dict[str, Any] = {}
        for key, value in raw_meta.items():
            raw = str(key).strip()
            if "." not in raw:
                continue
            table_name, column_name = raw.split(".", 1)
            table_name = table_name.strip()
            column_name = column_name.strip()
            if not table_name or not column_name:
                continue
            tm = schema_graph.tables.get(table_name)
            col_meta = tm.columns.get(column_name) if tm is not None else None
            if not column_eligible_for_space_allowlist(col_meta):
                continue
            filtered_meta[f"{table_name}.{column_name}"] = value
        out["column_meta"] = filtered_meta
    return out


def _sensitivity_rank(value: SensitivityClassification) -> int:
    if value == SensitivityClassification.HIDDEN:
        return 2
    if value == SensitivityClassification.RESTRICTED:
        return 1
    return 0


def snapshot_column_sensitivities(schema: SchemaGraph) -> dict[str, SensitivityClassification]:
    """Return a ``table.column`` → sensitivity map for every column on *schema*."""
    out: dict[str, SensitivityClassification] = {}
    for table_name, table in schema.tables.items():
        for column_name, column in table.columns.items():
            out[f"{table_name}.{column_name}"] = column.sensitivity
    return out


def sensitivity_increased_columns(
    before: Mapping[str, SensitivityClassification],
    schema: SchemaGraph,
) -> frozenset[str]:
    """Return qualified columns whose sensitivity strictness increased since *before*."""
    increased: set[str] = set()
    for table_name, table in schema.tables.items():
        for column_name, column in table.columns.items():
            qualified = f"{table_name}.{column_name}"
            new_rank = _sensitivity_rank(column.sensitivity)
            old_rank = _sensitivity_rank(before.get(qualified, SensitivityClassification.NONE))
            if new_rank > old_rank:
                increased.add(qualified)
    return frozenset(increased)


def text_mentions_sensitive_column(text: str, schema: SchemaGraph) -> bool:
    """Return True when *text* names a schema-known sensitive column."""
    return bool(DomainKnowledgeEntry.sensitive_column_references(text, schema))


_sensitivity_ratchet_artifact_scrub: Callable[[str, SchemaGraph, str], tuple[int, int]] | None = None


def register_sensitivity_ratchet_artifact_scrub(
    fn: Callable[[str, SchemaGraph, str], tuple[int, int]],
) -> None:
    """Register the template-store scrub callback from :mod:`aetherdialect._templates` at import time."""
    global _sensitivity_ratchet_artifact_scrub
    _sensitivity_ratchet_artifact_scrub = fn


def _parse_domain_knowledge_items(items: Sequence[Any]) -> tuple[DomainKnowledgeEntry, ...]:
    out: list[DomainKnowledgeEntry] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or "").strip()
        text = str(raw.get("text") or "").strip()
        kind = str(raw.get("kind") or "glossary").strip() or "glossary"
        if not key or not text or key in seen:
            continue
        try:
            entry = DomainKnowledgeEntry.normalize(DomainKnowledgeEntry(key=key, text=text, kind=kind))
        except (ConfigError, ValueError, TypeError):
            continue
        seen.add(entry.key)
        out.append(entry)
    return tuple(out)


def _scrub_space_snapshot(snapshot: dict[str, Any], schema: SchemaGraph) -> dict[str, Any]:
    out = filter_space_snapshot_sensitive_columns(snapshot, schema)
    dk_items = out.get("domain_knowledge")
    if isinstance(dk_items, list):
        entries = _parse_domain_knowledge_items(dk_items)
        scrubbed = filter_schema_anchored_domain_knowledge(entries, schema)
        out["domain_knowledge"] = [{"key": e.key, "kind": e.kind, "text": e.text} for e in scrubbed]
        out["domain_knowledge_digest"] = domain_knowledge_digest(scrubbed)
    sk_items = out.get("structural_knowledge")
    if isinstance(sk_items, list):
        facts = parse_structural_items(sk_items)
        scrubbed_facts = filter_schema_anchored_structural_knowledge(facts, schema)
        out["structural_knowledge"] = [f.to_dict() for f in scrubbed_facts]
    return out


def on_sensitivity_classification_change(
    schema: SchemaGraph,
    changed_columns: frozenset[str] | set[str],
    *,
    artifacts_dir: str | None = None,
    domain_knowledge: Sequence[DomainKnowledgeEntry] | None = None,
) -> SensitivityRatchetReport:
    """Re-scrub knowledge artifacts and drop dependents that name newly sensitive columns."""
    _ = changed_columns
    report = SensitivityRatchetReport()

    structural = tuple(getattr(schema, "structural_knowledge", ()) or ())
    if structural:
        scrubbed_structural = filter_schema_anchored_structural_knowledge(structural, schema)
        report = SensitivityRatchetReport(
            structural_dropped=len(structural) - len(scrubbed_structural),
            domain_knowledge_dropped=report.domain_knowledge_dropped,
            space_snapshots_updated=report.space_snapshots_updated,
            templates_dropped=report.templates_dropped,
            feedback_rows_dropped=report.feedback_rows_dropped,
            domain_knowledge_entries=report.domain_knowledge_entries,
        )
        schema.structural_knowledge = scrubbed_structural

    scrubbed_dk: tuple[DomainKnowledgeEntry, ...] | None = None
    if domain_knowledge is not None:
        scrubbed_dk = filter_schema_anchored_domain_knowledge(domain_knowledge, schema)
        report = SensitivityRatchetReport(
            domain_knowledge_dropped=len(domain_knowledge) - len(scrubbed_dk),
            structural_dropped=report.structural_dropped,
            space_snapshots_updated=report.space_snapshots_updated,
            templates_dropped=report.templates_dropped,
            feedback_rows_dropped=report.feedback_rows_dropped,
            domain_knowledge_entries=scrubbed_dk,
        )

    if not artifacts_dir:
        return report

    adir = os.path.abspath(str(artifacts_dir))
    with artifact_lock(adir):
        loaded = load_domain_knowledge_artifact(adir, schema, require_notes_match=False)
        if loaded is not None:
            persisted = filter_schema_anchored_domain_knowledge(loaded, schema)
            dk_dropped = len(loaded) - len(persisted)
            stamps = knowledge_artifact_save_stamps(schema)
            save_domain_knowledge_artifact(adir, persisted, **stamps)
            scrubbed_dk = persisted
            report = SensitivityRatchetReport(
                domain_knowledge_dropped=report.domain_knowledge_dropped + dk_dropped,
                structural_dropped=report.structural_dropped,
                space_snapshots_updated=report.space_snapshots_updated,
                templates_dropped=report.templates_dropped,
                feedback_rows_dropped=report.feedback_rows_dropped,
                domain_knowledge_entries=scrubbed_dk,
            )
        elif scrubbed_dk is not None:
            stamps = knowledge_artifact_save_stamps(schema)
            save_domain_knowledge_artifact(adir, scrubbed_dk, **stamps)
            report = SensitivityRatchetReport(
                domain_knowledge_dropped=report.domain_knowledge_dropped,
                structural_dropped=report.structural_dropped,
                space_snapshots_updated=report.space_snapshots_updated,
                templates_dropped=report.templates_dropped,
                feedback_rows_dropped=report.feedback_rows_dropped,
                domain_knowledge_entries=scrubbed_dk,
            )

        root = os.path.join(adir, AETHERSPACES_SEGMENT)
        if os.path.isdir(root):
            for entry in os.listdir(root):
                if not entry.endswith(".json"):
                    continue
                stem = entry[: -len(".json")]
                if not stem or stem in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
                    continue
                path = os.path.join(root, entry)
                try:
                    with open(path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                scrubbed = _scrub_space_snapshot(payload, schema)
                if scrubbed != payload:
                    write_json_atomic(path, scrubbed)
                    report = SensitivityRatchetReport(
                        domain_knowledge_dropped=report.domain_knowledge_dropped,
                        structural_dropped=report.structural_dropped,
                        space_snapshots_updated=report.space_snapshots_updated + 1,
                        templates_dropped=report.templates_dropped,
                        feedback_rows_dropped=report.feedback_rows_dropped,
                        domain_knowledge_entries=report.domain_knowledge_entries,
                    )

        graph_id = str(schema.schema_graph_id or "")
        scrub_hook = _sensitivity_ratchet_artifact_scrub
        if graph_id and scrub_hook is not None:
            templates_dropped, feedback_dropped = scrub_hook(adir, schema, graph_id)
            report = SensitivityRatchetReport(
                domain_knowledge_dropped=report.domain_knowledge_dropped,
                structural_dropped=report.structural_dropped,
                space_snapshots_updated=report.space_snapshots_updated,
                templates_dropped=report.templates_dropped + templates_dropped,
                feedback_rows_dropped=report.feedback_rows_dropped + feedback_dropped,
                domain_knowledge_entries=report.domain_knowledge_entries,
            )

    if scrubbed_dk is not None:
        DomainKnowledgeState.validate_entries(scrubbed_dk, schema)
    return report


def _profile_table_clone_roles(schema: SchemaGraph, notes_content: str | None = None) -> None:
    apply_column_roles_llm(schema, notes_content=notes_content)
    apply_boolean_coercion_pass(schema)
    assign_column_ops(schema)


register_apply_column_roles_llm_hook(_profile_table_clone_roles)
