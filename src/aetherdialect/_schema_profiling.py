"""Profile columns and tables, LLM and heuristic roles, DDL parsing, Unity constraints, and Databricks partition hints."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import sqlglot
from sqlalchemy import text

from . import _core_utils
from ._config import (
    BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES,
    BOOLEAN_ANTONYM_MIN_STEM_LEN,
    BOOLEAN_NEGATION_PREFIXES,
    BOOLEAN_NEGATION_SUFFIXES,
    BOOLEAN_TRUTH_PATTERN_MAP,
    COLUMN_DEFINITION_STOP_WORDS,
    NAME_COLUMN_PATTERN,
    PROFILING_TOP_K,
    ROLE_VALUE_TYPE_COMPAT,
    VALID_SENSITIVITY_LEVELS,
    EngineConfig,
    PolicyConfig,
    QSimConfig,
    cost_cap_active,
    diagnostic_debug_enabled,
    DIAGNOSTIC_CODE_ENGINE_INFO,
)

SCHEMA_NOTES_REFINE_SYSTEM: str = (
    "You refine base_classification using domain_notes.\n\n"
    "The base_classification was produced from profiling statistics and FK topology alone.\n"
    "Apply domain_notes to tighten descriptions, adjust table_role and column roles where notes explicitly require, "
    'and set column sensitivity to "pii" only when domain_notes explicitly mark PII for that column or category.\n'
    "Preserve substantive keywords and meaning from base descriptions and hints.\n"
    "Override table_role, column role, hint, description, or sensitivity only when domain_notes are explicit; "
    "when notes are silent, keep the base values.\n"
    "Do not remove tables or columns from base_classification. Do not add new tables or columns.\n"
    "Emit the full merged JSON with the same shape as base_classification.\n"
    "Reason internally, output only JSON:\n"
    '{"table1": {"table_role": "...", "description": "...", '
    '"columns": {"col1": {"role": "...", "hint": "...", "sensitivity": null}, ...}}, ...}'
)

SCHEMA_CONSISTENCY_REFINE_SYSTEM: str = (
    "You receive base_classification JSON describing every table and column. The user message contains only "
    "base_classification under that key.\n\n"
    "Preserve the base output unless you detect a genuine cross-table inconsistency — for example the same "
    "column name and SQL data type assigned different roles in different tables. When you fix such an "
    "inconsistency, align the conflicting entries to the role that best matches the shared name, type, and "
    "FK topology.\n\n"
    "Do not invent new descriptions: keep each table description and column hint from the base unless a "
    "detected inconsistency forces a minimal coordinated rewrite.\n"
    "Do not change sensitivity values from the base.\n"
    "Do not change column roles when the base assignment is already internally consistent.\n"
    "Do not remove tables or columns from base_classification. Do not add new tables or columns.\n\n"
    "Emit JSON identical in shape to base_classification.\n"
    "Reason internally, output only JSON:\n"
    '{"table1": {"table_role": "...", "description": "...", '
    '"columns": {"col1": {"role": "...", "hint": "...", "sensitivity": null}, ...}}, ...}'
)

from ._contracts_base import (
    CatalogStructuralConstraintsIndex,
    ColumnMetadata,
    ColumnRole,
    DescriptionOwner,
    FKEdge,
    RoleOwner,
    SchemaGraph,
    TableMetadata,
    TableRole,
    can_overwrite_role,
    set_description,
    set_sensitivity,
    SensitivityClassification,
)
from ._core_utils import debug, llm_chat, safe_json_loads, stable_json

if TYPE_CHECKING:
    from ._dialect import Dialect

try:
    import pglast
    from pglast.ast import AlterTableStmt, CreateStmt
    from pglast.enums import AlterTableType, ConstrType
    from pglast.stream import RawStream
except ImportError:
    pglast = None
    AlterTableStmt = None
    CreateStmt = None
    AlterTableType = None
    ConstrType = None
    RawStream = None

_PG_LAST_SQL_AVAILABLE: bool = pglast is not None


def _maybe_set_profile_statement_timeout(conn: Any, dialect_or_engine: Any) -> None:
    """Apply ``PolicyConfig.PROFILE_TIMEOUT_MS`` via ``SET LOCAL`` on PostgreSQL profiling connections."""

    name = getattr(dialect_or_engine, "name", None)
    if name is None:
        d = getattr(dialect_or_engine, "dialect", None)
        name = getattr(d, "name", "") if d is not None else ""
    if str(name).lower() not in ("postgresql", "postgres"):
        return
    tm = PolicyConfig.PROFILE_TIMEOUT_MS
    if not cost_cap_active(tm):
        return
    conn.execute(text(f"SET LOCAL statement_timeout = {int(tm)}"))


def collect_profiling_topk_values(raw: list[Any] | None) -> list[str]:
    """Return distinct non-empty profiling values in first-seen order, capped at ``PROFILING_TOP_K``."""

    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for v in raw:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= PROFILING_TOP_K:
            break
    return out


def _has_boolean_like_values(col: ColumnMetadata) -> tuple[bool, str | None]:
    """
    Check if a column's top-K values match a known boolean-like pattern.

    Args:

        col: The `ColumnMetadata` to inspect.

    Returns:

        ``(matched, truth_literal)`` where *truth_literal* is the stored affirmative token when matched.
    """

    if col.distinct_count != 2:
        return False, None
    if not col.top_k_values or len(col.top_k_values) != 2:
        return False, None
    values_lower = frozenset(str(v).lower().strip() for v in col.top_k_values)
    truth_norm = BOOLEAN_TRUTH_PATTERN_MAP.get(values_lower)
    if truth_norm is None:
        return False, None
    for raw in col.top_k_values:
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


def _profile_column(
    dialect: Dialect,
    engine: Any,
    col: ColumnMetadata,
    table_name: str,
    row_count: int,
    sample_threshold: int = None,
    sample_size: int = None,
    *,
    table_kind: Literal["table", "view"] = "table",
) -> None:
    """
    Profile a single column and update its metadata in-place.

    Args:

        dialect: Active dialect controlling sampling SQL shape.

        engine: SQLAlchemy engine connected to the target database.

        col: The `ColumnMetadata` to update.

        table_name: The name of the table containing the column.

        row_count: Total row count for the table (used for sample size calculation).

        sample_threshold: Row count above which sampling is used; defaults to `QSimConfig.PROFILING_SAMPLE_THRESHOLD`.

        sample_size: Number of rows to sample; defaults to `QSimConfig.PROFILING_SAMPLE_SIZE`.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_column] profiling {table_name}.{col.name}")
    if sample_threshold is None:
        sample_threshold = QSimConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = QSimConfig.PROFILING_SAMPLE_SIZE

    col.row_count = row_count
    use_sample = row_count > sample_threshold

    try:
        with engine.connect() as conn:
            _maybe_set_profile_statement_timeout(conn, dialect)
            sample_clause = dialect.profiling_stats_sample_suffix(
                use_sample=use_sample,
                row_count=row_count,
                sample_size=sample_size,
                random_seed=QSimConfig.RANDOM_SEED,
                table_kind=table_kind,
            )
            use_subquery = use_sample and dialect.profiling_stats_use_subquery_when_sampling(table_kind)

            if use_sample and not use_subquery:
                stats_sql = f"""
                    SELECT 
                        COUNT(*) as cnt,
                        COUNT(DISTINCT "{col.name}") as dist,
                        COUNT(*) - COUNT("{col.name}") as nulls
                    FROM "{table_name}" {sample_clause}
                """
            elif use_sample:
                stats_sql = f"""
                    SELECT 
                        COUNT(*) as cnt,
                        COUNT(DISTINCT "{col.name}") as dist,
                        COUNT(*) - COUNT("{col.name}") as nulls
                    FROM (SELECT "{col.name}" FROM "{table_name}" {sample_clause}) t
                """
            else:
                stats_sql = f"""
                    SELECT 
                        COUNT(*) as cnt,
                        COUNT(DISTINCT "{col.name}") as dist,
                        COUNT(*) - COUNT("{col.name}") as nulls
                    FROM "{table_name}"
                """

            result = conn.execute(text(stats_sql)).fetchone()
            cnt = result[0] or 1
            dist = result[1] or 0
            nulls = result[2] or 0

            col.distinct_count = dist
            col.distinct_ratio = dist / cnt if cnt > 0 else 0.0
            col.null_ratio = nulls / cnt if cnt > 0 else 0.0
            col.distinct_from_sample = bool(use_sample)

            if col.value_type in ("integer", "number") or col.value_type == "date":
                minmax_sql = f'SELECT MIN("{col.name}"), MAX("{col.name}") FROM "{table_name}"'
                minmax_result = conn.execute(text(minmax_sql)).fetchone()
                if minmax_result:
                    col.min_val = str(minmax_result[0]) if minmax_result[0] is not None else None
                    col.max_val = str(minmax_result[1]) if minmax_result[1] is not None else None

            topk_sql = f"""
                SELECT DISTINCT "{col.name}" AS v
                FROM "{table_name}"
                WHERE "{col.name}" IS NOT NULL
                ORDER BY "{col.name}" ASC
                LIMIT {PolicyConfig.CATEGORICAL_SAMPLE_SIZE}
            """
            topk_result = conn.execute(text(topk_sql)).fetchall()
            col.top_k_values = collect_profiling_topk_values([row[0] for row in topk_result if row[0] is not None])
            mode_sql = f"""
                SELECT MAX(c) FROM (
                    SELECT COUNT(*) AS c FROM "{table_name}"
                    WHERE "{col.name}" IS NOT NULL
                    GROUP BY "{col.name}"
                ) s
            """
            mode_row = conn.execute(text(mode_sql)).fetchone()
            non_null = max(0, cnt - nulls)
            if mode_row and mode_row[0] is not None and non_null > 0:
                top_freq = mode_row[0] or 0
                col.mode_frequency_ratio = float(top_freq) / float(non_null) if top_freq else 0.0
            else:
                col.mode_frequency_ratio = 0.0

            if _column_semantic_distinct_eligible(col) and not use_sample:
                cap = PolicyConfig.SEMANTIC_JOIN_ASC_DISTINCT_LIMIT
                sem_sql = (
                    f'SELECT DISTINCT CAST("{col.name}" AS TEXT) AS v FROM "{table_name}" '
                    f'WHERE "{col.name}" IS NOT NULL ORDER BY v ASC LIMIT {cap}'
                )
                sem_rows = conn.execute(text(sem_sql)).fetchall()
                col.semantic_distinct_values = [str(r[0]) for r in sem_rows if r[0] is not None]
    except Exception as e:
        debug(f"[schema_profiling.profile_column] failed: {table_name}.{col.name}: {e}")


def _column_semantic_distinct_eligible(col: ColumnMetadata) -> bool:
    """Return True when distinct-value sampling supports semantic join overlap checks."""
    if col.distinct_count <= 0:
        return False
    if col.distinct_count > PolicyConfig.SEMANTIC_JOIN_ASC_DISTINCT_LIMIT * 20:
        return False
    if _is_boolean_like_column(col):
        return False
    vt = (col.value_type or "").lower()
    if vt in ("blob", "binary"):
        return False
    if vt in ("string", "categorical", "free_text", "identifier"):
        return True
    if vt in ("integer", "number") and col.distinct_count <= PolicyConfig.CATEGORICAL_MAX_CARDINALITY:
        return True
    return False


def _profile_composite_descriptive(
    engine: Any,
    table: TableMetadata,
) -> None:
    """
    Compute composite distinct ratios for name-like column pairs.

    Args:

        engine: SQLAlchemy engine connected to the target database.

        table: The `TableMetadata` to update in-place.

    Returns:

        Return value.
    """
    name_cols = [
        col_name
        for col_name, col_meta in table.columns.items()
        if (col_meta.value_type or "").lower() == "string"
        and NAME_COLUMN_PATTERN.search(col_name)
        and not col_meta.is_primary_key
        and not col_meta.is_foreign_key
    ]
    if len(name_cols) < 2:
        return
    row_count = table.row_count or 0
    if row_count == 0:
        return

    try:
        with engine.connect() as conn:
            _maybe_set_profile_statement_timeout(conn, engine)
            for i in range(len(name_cols)):
                for j in range(i + 1, len(name_cols)):
                    c1, c2 = name_cols[i], name_cols[j]
                    sql = f'SELECT COUNT(DISTINCT CONCAT("{c1}", \' \', "{c2}")) FROM "{table.name}"'
                    composite_distinct = conn.execute(text(sql)).scalar() or 0
                    ratio = composite_distinct / row_count
                    table.composite_descriptive_ratios[(c1, c2)] = ratio
                    debug(
                        f"[schema_profiling._profile_composite_descriptive] "
                        f"{table.name}.({c1}, {c2}) composite_ratio={ratio:.4f}"
                    )
    except Exception as exc:
        debug(f"[schema_profiling._profile_composite_descriptive] failed for {table.name}: {exc}")


def _profile_table(dialect: Dialect, engine: Any, table: TableMetadata) -> None:
    """
    Profile all columns in a table and update metadata in-place.

    Args:

        dialect: Active dialect for sampling and statistics SQL.

        engine: SQLAlchemy engine connected to the target database.

        table: The `TableMetadata` to update (`row_count` and all column stats).

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_table] profiling {table.name} ({len(table.columns)} columns)")
    try:
        with engine.connect() as conn:
            _maybe_set_profile_statement_timeout(conn, dialect)
            count_sql = f'SELECT COUNT(*) FROM "{table.name}"'
            row_count = conn.execute(text(count_sql)).scalar() or 0
            table.row_count = row_count
    except Exception as e:
        debug(f"[schema_profiling.profile_table] row count failed: {table.name}: {e}")
        row_count = 0
        table.row_count = 0

    for col in table.columns.values():
        _profile_column(dialect, engine, col, table.name, row_count, table_kind=table.kind)

    _profile_composite_descriptive(engine, table)

    debug(f"[schema_profiling.profile_table] completed: {table.name}")


def profile_schema(engine: Any, schema: SchemaGraph, dialect: Dialect) -> None:
    """
    Profile all tables in a schema and update metadata in-place.

    Args:

        engine: SQLAlchemy engine connected to the target database.

        schema: The `SchemaGraph` whose tables will be profiled.

        dialect: Active dialect for per-engine sampling behavior.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_schema] profiling {len(schema.tables)} tables")
    total = len(schema.tables)
    for idx, table in enumerate(schema.tables.values(), start=1):
        _core_utils.notify(
            f"  Profiling [{idx}/{total}] {table.name}",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("table", table.name), ("index", str(idx)), ("total", str(total))),
        )
        _profile_table(dialect, engine, table)
    debug("[schema_profiling.profile_schema] completed")


def _profile_column_spark(
    spark,
    catalog: str,
    schema_name: str,
    col: ColumnMetadata,
    table_name: str,
    row_count: int,
    sample_threshold: int = None,
    sample_size: int = None,
) -> None:
    """
    Profile a single column from a Databricks table via Spark SQL and update metadata in-place.

    Args:

        spark: Active `SparkSession`.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name within the catalog.

        col: The `ColumnMetadata` to update.

        table_name: The table name within the schema.

        row_count: Total row count used for sampling decisions.

        sample_threshold: Row count above which sampling is used; defaults to `QSimConfig.PROFILING_SAMPLE_THRESHOLD`.

        sample_size: Number of rows to sample; defaults to `QSimConfig.PROFILING_SAMPLE_SIZE`.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_column_spark] profiling {table_name}.{col.name}")
    if sample_threshold is None:
        sample_threshold = QSimConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = QSimConfig.PROFILING_SAMPLE_SIZE

    col.row_count = row_count
    use_sample = row_count > sample_threshold

    try:
        full_table = f"`{catalog}`.`{schema_name}`.`{table_name}`"

        if use_sample:
            sample_clause = f"TABLESAMPLE ({sample_size} ROWS)"
        else:
            sample_clause = ""

        if use_sample:
            stats_sql = f"""
                SELECT 
                    COUNT(*) as cnt,
                    COUNT(DISTINCT `{col.name}`) as dist,
                    COUNT(*) - COUNT(`{col.name}`) as nulls
                FROM {full_table} {sample_clause}
            """
        else:
            stats_sql = f"""
                SELECT 
                    COUNT(*) as cnt,
                    COUNT(DISTINCT `{col.name}`) as dist,
                    COUNT(*) - COUNT(`{col.name}`) as nulls
                FROM {full_table}
            """

        result = spark.sql(stats_sql).collect()[0]
        cnt = result["cnt"] or 1
        dist = result["dist"] or 0
        nulls = result["nulls"] or 0

        col.distinct_count = dist
        col.distinct_ratio = dist / cnt if cnt > 0 else 0.0
        col.null_ratio = nulls / cnt if cnt > 0 else 0.0
        col.distinct_from_sample = bool(use_sample)

        if col.value_type in ("integer", "number") or col.value_type == "date":
            minmax_sql = f"SELECT MIN(`{col.name}`), MAX(`{col.name}`) FROM {full_table}"
            minmax_result = spark.sql(minmax_sql).collect()[0]
            if minmax_result:
                col.min_val = str(minmax_result[0]) if minmax_result[0] is not None else None
                col.max_val = str(minmax_result[1]) if minmax_result[1] is not None else None

        topk_sql = f"""
            SELECT v FROM (
                SELECT DISTINCT `{col.name}` AS v
                FROM {full_table}
                WHERE `{col.name}` IS NOT NULL
            ) t
            ORDER BY v ASC
            LIMIT {PolicyConfig.CATEGORICAL_SAMPLE_SIZE}
        """
        topk_result = spark.sql(topk_sql).collect()
        col.top_k_values = collect_profiling_topk_values(
            [row["v"] for row in topk_result if row["v"] is not None],
        )
        mode_sql = f"""
            SELECT MAX(freq) AS mx FROM (
                SELECT COUNT(*) AS freq FROM {full_table}
                WHERE `{col.name}` IS NOT NULL
                GROUP BY `{col.name}`
            ) g
        """
        mode_rows = spark.sql(mode_sql).collect()
        non_null = max(0, cnt - nulls)
        if mode_rows and mode_rows[0]["mx"] is not None and non_null > 0:
            top_freq = mode_rows[0]["mx"] or 0
            col.mode_frequency_ratio = float(top_freq) / float(non_null) if top_freq else 0.0
        else:
            col.mode_frequency_ratio = 0.0

        if _column_semantic_distinct_eligible(col):
            cap = PolicyConfig.SEMANTIC_JOIN_ASC_DISTINCT_LIMIT
            sem_sample = f" TABLESAMPLE ({sample_size} ROWS)" if use_sample else ""
            sem_sql = (
                f"SELECT DISTINCT CAST(`{col.name}` AS STRING) AS v FROM {full_table}{sem_sample} "
                f"WHERE `{col.name}` IS NOT NULL ORDER BY v ASC LIMIT {cap}"
            )
            sem_rows = spark.sql(sem_sql).collect()
            col.semantic_distinct_values = [str(r["v"]) for r in sem_rows if r.get("v") is not None]
    except Exception as e:
        debug(f"[schema_profiling.profile_column_spark] failed: {table_name}.{col.name}: {e}")


def _profile_composite_descriptive_spark(
    spark,
    catalog: str,
    schema_name: str,
    table: TableMetadata,
) -> None:
    """
    Compute composite distinct ratios for name-like column pairs via Spark.

    Args:

        spark: Active `SparkSession`.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name within the catalog.

        table: The `TableMetadata` to update in-place.

    Returns:

        Return value.
    """
    name_cols = [
        col_name
        for col_name, col_meta in table.columns.items()
        if (col_meta.value_type or "").lower() == "string"
        and NAME_COLUMN_PATTERN.search(col_name)
        and not col_meta.is_primary_key
        and not col_meta.is_foreign_key
    ]
    if len(name_cols) < 2:
        return
    row_count = table.row_count or 0
    if row_count == 0:
        return
    full_table = f"`{catalog}`.`{schema_name}`.`{table.name}`"
    try:
        for i in range(len(name_cols)):
            for j in range(i + 1, len(name_cols)):
                c1, c2 = name_cols[i], name_cols[j]
                sql = f"SELECT COUNT(DISTINCT CONCAT(`{c1}`, ' ', `{c2}`)) FROM {full_table}"
                composite_distinct = spark.sql(sql).collect()[0][0] or 0
                ratio = composite_distinct / row_count
                table.composite_descriptive_ratios[(c1, c2)] = ratio
                debug(
                    f"[schema_profiling._profile_composite_descriptive_spark] "
                    f"{table.name}.({c1}, {c2}) composite_ratio={ratio:.4f}"
                )
    except Exception as exc:
        debug(f"[schema_profiling._profile_composite_descriptive_spark] failed for {table.name}: {exc}")


def _profile_table_spark(spark, catalog: str, schema_name: str, table: TableMetadata) -> None:
    """
    Profile all columns in a Databricks table via Spark queries.

    Args:

        spark: Active `SparkSession`.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name.

        table: The `TableMetadata` to update.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_table_spark] profiling {table.name} ({len(table.columns)} columns)")
    try:
        full_table = f"`{catalog}`.`{schema_name}`.`{table.name}`"
        count_sql = f"SELECT COUNT(*) FROM {full_table}"
        row_count = spark.sql(count_sql).collect()[0][0] or 0
        table.row_count = row_count
    except Exception as e:
        debug(f"[schema_profiling.profile_table_spark] row count failed: {table.name}: {e}")
        row_count = 0
        table.row_count = 0

    for col in table.columns.values():
        _profile_column_spark(spark, catalog, schema_name, col, table.name, row_count)

    _profile_composite_descriptive_spark(spark, catalog, schema_name, table)

    debug(f"[schema_profiling.profile_table_spark] completed: {table.name}")


def profile_schema_spark(spark, catalog: str, schema_name: str, schema: SchemaGraph) -> None:
    """
    Profile all tables in a Databricks schema via Spark queries.

    Args:

        spark: Active `SparkSession`.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name.

        schema: The `SchemaGraph` whose tables will be profiled.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_schema_spark] profiling {len(schema.tables)} tables")
    total = len(schema.tables)
    for idx, table in enumerate(schema.tables.values(), start=1):
        _core_utils.notify(
            f"  Profiling [{idx}/{total}] {table.name}",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("table", table.name), ("index", str(idx)), ("total", str(total))),
        )
        _profile_table_spark(spark, catalog, schema_name, table)
    debug("[schema_profiling.profile_schema_spark] completed")


def _cursor_rows_as_dicts(cursor) -> list[dict]:
    """
    Convert cursor result rows to a list of dicts keyed by column name.

    Args:

        cursor: Description.

    Returns:

        missing.
    """
    if not cursor.description:
        return []
    col_names = [d[0] for d in cursor.description]
    return [dict(zip(col_names, row, strict=True)) for row in cursor.fetchall()]


def _unity_trailing_table_identifier(ref: str) -> str:
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


def _tables_meta_foreign_key_dicts_from_edges(edges: list[FKEdge]) -> list[dict[str, Any]]:
    """Convert :class:`FKEdge` instances into ``tables_meta`` foreign-key dict entries."""

    out: list[dict[str, Any]] = []
    for e in edges:
        out.append({"src_cols": list(e.src_cols), "dst_table": str(e.dst_table), "dst_cols": list(e.dst_cols)})
    return out


def _profile_column_sql_connector(
    connection,
    catalog: str,
    schema_name: str,
    col: ColumnMetadata,
    table_name: str,
    row_count: int,
    sample_threshold: int = None,
    sample_size: int = None,
) -> None:
    """
    Profile a single column via databricks-sql-connector and update metadata in-place.

    Args:

        connection: Active `databricks.sql` connection.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name within the catalog.

        col: The `ColumnMetadata` to update.

        table_name: The table name within the schema.

        row_count: Total row count used for sampling decisions.

        sample_threshold: Row count above which sampling is used; defaults to `QSimConfig.PROFILING_SAMPLE_THRESHOLD`.

        sample_size: Number of rows to sample; defaults to `QSimConfig.PROFILING_SAMPLE_SIZE`.

    Returns:

        Return value.
    """
    if sample_threshold is None:
        sample_threshold = QSimConfig.PROFILING_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = QSimConfig.PROFILING_SAMPLE_SIZE
    col.row_count = row_count
    use_sample = row_count > sample_threshold
    full_table = f"`{catalog}`.`{schema_name}`.`{table_name}`"
    sample_clause = f"TABLESAMPLE ({sample_size} ROWS)" if use_sample else ""
    try:
        with connection.cursor() as cursor:
            if use_sample:
                stats_sql = f"""
                    SELECT
                        COUNT(*) as cnt,
                        COUNT(DISTINCT `{col.name}`) as dist,
                        COUNT(*) - COUNT(`{col.name}`) as nulls
                    FROM {full_table} {sample_clause}
                """
            else:
                stats_sql = f"""
                    SELECT
                        COUNT(*) as cnt,
                        COUNT(DISTINCT `{col.name}`) as dist,
                        COUNT(*) - COUNT(`{col.name}`) as nulls
                    FROM {full_table}
                """
            cursor.execute(stats_sql)
            rows = _cursor_rows_as_dicts(cursor)
            stats_cnt = 0
            stats_nulls = 0
            if rows:
                r = rows[0]
                cnt = r.get("cnt") or 1
                dist = r.get("dist") or 0
                nulls = r.get("nulls") or 0
                col.distinct_count = dist
                col.distinct_ratio = dist / cnt if cnt > 0 else 0.0
                col.null_ratio = nulls / cnt if cnt > 0 else 0.0
                col.distinct_from_sample = bool(use_sample)
                stats_cnt = cnt
                stats_nulls = nulls
            if col.value_type in ("integer", "number") or col.value_type == "date":
                minmax_sql = f"SELECT MIN(`{col.name}`) as mn, MAX(`{col.name}`) as mx FROM {full_table}"
                cursor.execute(minmax_sql)
                minmax_rows = _cursor_rows_as_dicts(cursor)
                if minmax_rows:
                    r = minmax_rows[0]
                    col.min_val = str(r["mn"]) if r.get("mn") is not None else None
                    col.max_val = str(r["mx"]) if r.get("mx") is not None else None
            topk_sql = f"""
                SELECT v AS topval FROM (
                    SELECT DISTINCT `{col.name}` AS v
                    FROM {full_table}
                    WHERE `{col.name}` IS NOT NULL
                ) t
                ORDER BY topval ASC
                LIMIT {PolicyConfig.CATEGORICAL_SAMPLE_SIZE}
            """
            cursor.execute(topk_sql)
            topk_rows = _cursor_rows_as_dicts(cursor)
            col.top_k_values = collect_profiling_topk_values(
                [r["topval"] for r in topk_rows if r and r.get("topval") is not None],
            )
            mode_sql = f"""
                SELECT MAX(freq) AS mx FROM (
                    SELECT COUNT(*) AS freq FROM {full_table}
                    WHERE `{col.name}` IS NOT NULL
                    GROUP BY `{col.name}`
                ) g
            """
            cursor.execute(mode_sql)
            mode_rows = _cursor_rows_as_dicts(cursor)
            non_null = max(0, stats_cnt - stats_nulls)
            if mode_rows and mode_rows[0].get("mx") is not None and non_null > 0:
                top_freq = mode_rows[0].get("mx") or 0
                col.mode_frequency_ratio = float(top_freq) / float(non_null) if top_freq else 0.0
            else:
                col.mode_frequency_ratio = 0.0
            if _column_semantic_distinct_eligible(col):
                cap = PolicyConfig.SEMANTIC_JOIN_ASC_DISTINCT_LIMIT
                sem_sample = f" TABLESAMPLE ({sample_size} ROWS)" if use_sample else ""
                sem_sql = (
                    f"SELECT DISTINCT CAST(`{col.name}` AS STRING) AS v FROM {full_table}{sem_sample} "
                    f"WHERE `{col.name}` IS NOT NULL ORDER BY v ASC LIMIT {cap}"
                )
                cursor.execute(sem_sql)
                sem_rows = _cursor_rows_as_dicts(cursor)
                col.semantic_distinct_values = [str(r["v"]) for r in sem_rows if r.get("v") is not None]
    except Exception as e:
        debug(f"[schema_profiling._profile_column_sql_connector] failed: {table_name}.{col.name}: {e}")


def _profile_composite_descriptive_sql_connector(
    connection,
    catalog: str,
    schema_name: str,
    table: TableMetadata,
) -> None:
    """
    Compute composite distinct ratios for name-like column pairs via SQL connector.

    Args:

        connection: Active `databricks.sql` connection.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name within the catalog.

        table: The `TableMetadata` to update in-place.

    Returns:

        Return value.
    """
    name_cols = [
        col_name
        for col_name, col_meta in table.columns.items()
        if (col_meta.value_type or "").lower() == "string"
        and NAME_COLUMN_PATTERN.search(col_name)
        and not col_meta.is_primary_key
        and not col_meta.is_foreign_key
    ]
    if len(name_cols) < 2:
        return
    row_count = table.row_count or 0
    if row_count == 0:
        return
    full_table = f"`{catalog}`.`{schema_name}`.`{table.name}`"
    try:
        with connection.cursor() as cursor:
            for i in range(len(name_cols)):
                for j in range(i + 1, len(name_cols)):
                    c1, c2 = name_cols[i], name_cols[j]
                    sql = f"SELECT COUNT(DISTINCT CONCAT(`{c1}`, ' ', `{c2}`)) FROM {full_table}"
                    cursor.execute(sql)
                    rows = _cursor_rows_as_dicts(cursor)
                    composite_distinct = 0
                    if rows and rows[0]:
                        composite_distinct = list(rows[0].values())[0] or 0
                    ratio = composite_distinct / row_count
                    table.composite_descriptive_ratios[(c1, c2)] = ratio
                    debug(
                        f"[schema_profiling._profile_composite_descriptive_sql_connector] "
                        f"{table.name}.({c1}, {c2}) composite_ratio={ratio:.4f}"
                    )
    except Exception as exc:
        debug(f"[schema_profiling._profile_composite_descriptive_sql_connector] failed for {table.name}: {exc}")


def _profile_table_sql_connector(
    connection,
    catalog: str,
    schema_name: str,
    table: TableMetadata,
) -> None:
    """
    Profile all columns in a Databricks table via databricks-sql- connector.

    Args:

        connection: Active `databricks.sql` connection.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name.

        table: The `TableMetadata` to update.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling._profile_table_sql_connector] profiling {table.name}")
    full_table = f"`{catalog}`.`{schema_name}`.`{table.name}`"
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {full_table}")
            rows = _cursor_rows_as_dicts(cursor)
            row_count = rows[0].get(list(rows[0].keys())[0], 0) or 0 if rows else 0
    except Exception as e:
        debug(f"[schema_profiling._profile_table_sql_connector] row count failed: {table.name}: {e}")
        row_count = 0
    table.row_count = row_count
    for col in table.columns.values():
        _profile_column_sql_connector(connection, catalog, schema_name, col, table.name, row_count)
    _profile_composite_descriptive_sql_connector(connection, catalog, schema_name, table)
    debug(f"[schema_profiling._profile_table_sql_connector] completed: {table.name}")


def profile_schema_sql_connector(
    connection,
    catalog: str,
    schema_name: str,
    schema: SchemaGraph,
) -> None:
    """
    Profile all tables in a Databricks schema via databricks-sql- connector.

    Args:

        connection: Active `databricks.sql` connection.

        catalog: The Unity Catalog name.

        schema_name: The schema (database) name.

        schema: The `SchemaGraph` whose tables will be profiled.

    Returns:

        Return value.
    """
    debug(f"[schema_profiling.profile_schema_sql_connector] profiling {len(schema.tables)} tables")
    total = len(schema.tables)
    for idx, table in enumerate(schema.tables.values(), start=1):
        _core_utils.notify(
            f"  Profiling [{idx}/{total}] {table.name}",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("table", table.name), ("index", str(idx)), ("total", str(total))),
        )
        _profile_table_sql_connector(connection, catalog, schema_name, table)
    debug("[schema_profiling.profile_schema_sql_connector] completed")


def extract_tables_from_catalog_sql_connector(
    connection,
    catalog: str,
    schema: str,
    *,
    allow_objects: frozenset[str] | None = None,
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> dict[str, dict]:
    """
    Extract full table metadata from a Databricks Unity Catalog schema via SQL connector.

    Args:

        connection: Active `databricks.sql` connection.

        catalog: The catalog name.

        schema: The schema (database) name.

        allow_objects: When set, restrict catalog extraction to these relation names (case-insensitive).

        structural_constraints_index: PK, FK, and single-column UNIQUE metadata from the active dialect.

    Returns:

        `primary_keys`, `foreign_keys`, `table_comment`, and `properties`.
    """
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
        for col in cols:
            cname = col.get("col_name") or col.get("colname")
            if not cname or str(cname).startswith("#"):
                break
            column_names.append(cname)
            column_types.append(col.get("data_type") or "STRING")
        bundle = structural_constraints_index.tables.get(str(table_name).lower())
        use_info_schema = bundle is not None and bool(
            bundle.primary_keys or bundle.foreign_keys or bundle.unique_columns
        )
        if use_info_schema:
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
                        pk_dd, fk_dd, uq_dd = _parse_unity_catalog_constraints(stmt)
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
        except Exception:
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
        except Exception:
            pass
        tables[table_name] = {
            "table_name_original": table_name,
            "column_names_original": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "unique_columns": unique_columns,
            "partition_columns": partition_columns,
            "table_comment": table_comment,
            "properties": properties,
        }
    debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] complete: {len(tables)} tables")
    return tables


def _enrich_fk_column_descriptions(schema: SchemaGraph) -> None:
    """
    Append navigational hints to FK column descriptions.

    Args:

        schema: The `SchemaGraph` to update in-place.

    Returns:

        Return value.
    """
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
                if not c.is_primary_key and not c.is_foreign_key and c.role not in ("identifier", "")
            ][:3]
            if not descriptive_cols:
                descriptive_cols = [
                    c.name for c in dst_table.columns.values() if not c.is_primary_key and not c.is_foreign_key
                ][:3]
            if descriptive_cols:
                suffix = f"join {dst_table_name} for {', '.join(descriptive_cols)}"
            else:
                suffix = f"join {dst_table_name}"
            existing = (col.description or "").rstrip(". ")
            set_description(
                col,
                f"{existing} — {suffix}" if existing else suffix,
                DescriptionOwner.PROFILE,
            )


def _infer_column_role(col: ColumnMetadata) -> ColumnRole:
    """
    Infer a column's role from its metadata using heuristic rules (fallback).

    Args:

        col: The `ColumnMetadata` to classify.

    Returns:

        The inferred `ColumnRole` enum value.
    """
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

    if col.distinct_ratio >= PolicyConfig.IDENTIFIER_MIN_UNIQUENESS:
        return ColumnRole.FREE_TEXT

    is_categorical = (
        col.distinct_count <= PolicyConfig.CATEGORICAL_MAX_CARDINALITY
        or col.distinct_ratio <= PolicyConfig.CATEGORICAL_MAX_RATIO
    )
    if is_categorical:
        if col.value_type in ("integer", "number"):
            return ColumnRole.NUMERIC_CATEGORICAL
        return ColumnRole.CATEGORICAL

    if col.value_type in ("integer", "number"):
        return ColumnRole.NUMERIC_MEASURE

    return ColumnRole.FREE_TEXT


def _array_element_type_from_data_type(data_type: str) -> tuple[bool, str | None]:
    """
    Detect array or list column types and return element type when inferable.

    Args:

        data_type: SQL or catalog type string (e.g. `ARRAY<INT>`).

    Returns:

        Tuple `(is_array, element_type)` where `element_type` may be `None`.
    """
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
    if "array" in sl and "[" in sl:
        return True, "string"
    return False, None


def _validate_column_classification(col: ColumnMetadata, role: str) -> tuple[list[str], list[str]]:
    """
    Validate an LLM-assigned column role against profiling data.

    Args:

        col: The `ColumnMetadata` with profiling statistics.

        role: The role string assigned by the LLM.

    Returns:

        `(hard_errors, soft_warnings)` where hard errors reject the classification and soft warnings are log-only.
    """
    hard_errors = []
    soft_warnings = []

    is_numeric = col.value_type in ("integer", "number")
    is_temporal = col.value_type == "date"
    col_name_lower = col.name.lower()
    dtype = (col.data_type or "").upper()

    if role == ColumnRole.NUMERIC_MEASURE.value and not is_numeric:
        hard_errors.append(f"{col.name}: NUMERIC_MEASURE requires numeric type, got '{col.data_type}'")

    if role == ColumnRole.NUMERIC_CATEGORICAL.value and not is_numeric:
        hard_errors.append(f"{col.name}: NUMERIC_CATEGORICAL requires numeric type, got '{col.data_type}'")

    if role == ColumnRole.TEMPORAL.value and not is_temporal:
        if "year" in col_name_lower or "DOMAIN" in dtype or "YEAR" in dtype:
            soft_warnings.append(f"{col.name}: TEMPORAL on year/domain column, recommend NUMERIC_CATEGORICAL")
        else:
            hard_errors.append(f"{col.name}: TEMPORAL requires date/time type, got '{col.data_type}'")

    if role == ColumnRole.BOOLEAN.value and col.distinct_count and col.distinct_count > 2:
        hard_errors.append(f"{col.name}: BOOLEAN requires distinct_count <= 2, got {col.distinct_count}")

    if role == ColumnRole.CATEGORICAL.value and col.distinct_count and col.distinct_count > 1000:
        soft_warnings.append(f"{col.name}: CATEGORICAL with high cardinality ({col.distinct_count})")

    if role == ColumnRole.NUMERIC_MEASURE.value and col.distinct_count and col.distinct_count <= 5:
        soft_warnings.append(f"{col.name}: NUMERIC_MEASURE with low cardinality ({col.distinct_count})")

    if role == ColumnRole.IDENTIFIER.value and not col.is_primary_key and not col.is_foreign_key:
        soft_warnings.append(f"{col.name}: IDENTIFIER on non-PK/FK column")

    return hard_errors, soft_warnings


def _build_column_profile_for_llm(col: ColumnMetadata) -> dict:
    """
    Build a column profile dict for inclusion in the LLM classification prompt.

    Args:

        col: The `ColumnMetadata` to summarise.

    Returns:

        Column profile dict with ``name``, ``data_type``, PK / FK flags, and an optional ``profile_hints`` sub-dict that may include ``top_values`` when :data:`PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES` is positive and the column passes categorical sampling qualifiers.
    """
    profile = {
        "name": col.name,
        "data_type": col.data_type,
        "is_primary_key": col.is_primary_key,
        "is_foreign_key": col.is_foreign_key,
    }
    hints: dict = {}
    if col.distinct_count is not None:
        hints["distinct_count"] = col.distinct_count
    if col.distinct_ratio is not None:
        hints["distinct_ratio"] = round(col.distinct_ratio, 3)
    if col.null_ratio is not None:
        hints["null_ratio"] = round(col.null_ratio, 3)
    sample_n = PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES
    ratio_cap = PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_DISTINCT_RATIO_MAX
    if (
        sample_n > 0
        and col.top_k_values
        and not col.is_denied
        and col.sensitivity == SensitivityClassification.NONE
        and col.distinct_ratio is not None
        and col.distinct_ratio <= ratio_cap
    ):
        vt = (col.value_type or "").lower()
        dtype = (col.data_type or "").lower()
        string_like = vt == "string" or any(
            dtype.startswith(p) for p in ("varchar", "char", "text", "string", "nvarchar")
        )
        if string_like:
            hints["top_values"] = list(col.top_k_values[:sample_n])
    if hints:
        profile["profile_hints"] = hints
    return profile


def _normalize_llm_classification_payload(
    result: dict[str, Any],
) -> dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]]:
    """
    Convert raw LLM classification JSON into validated internal tuples.

    Args:

        result: Parsed JSON object whose keys are table names.

    Returns:

        Mapping from table name to ``(table_role, description, column_map)`` where each column maps to ``(role, hint, sensitivity)``.
    """

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
        columns_data = table_data.get("columns", {})
        column_classifications: dict[str, tuple[str, str, str | None]] = {}
        for col_name, classification in columns_data.items():
            sensitivity: str | None = None
            if isinstance(classification, dict):
                role = classification.get("role", "").lower()
                hint = str(classification.get("hint", "")).strip()
                sens_raw = classification.get("sensitivity")
                if sens_raw is not None and str(sens_raw).strip():
                    sv = str(sens_raw).strip().lower()
                    if sv in VALID_SENSITIVITY_LEVELS:
                        sensitivity = sv
            else:
                role = str(classification).lower()
                hint = ""
            if role not in valid_col_roles:
                role = ColumnRole.FREE_TEXT.value
            column_classifications[col_name] = (role, hint, sensitivity)
        classifications[table_name] = (table_role, description, column_classifications)
    return classifications


def _merge_classification_payloads(base: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    """
    Merge refine-stage JSON over base-stage JSON without dropping base tables or columns.

    Args:

        base: Stage-one classification payload.

        refined: Stage-two classification payload overlapping base keys.

    Returns:

        Merged payload suitable for :func:`_normalize_llm_classification_payload`.
    """

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
                if cname in base_cols:
                    base_cols[cname] = cdata
        mt["columns"] = base_cols
        merged[tname] = mt
    return merged


def _llm_classify_schema(
    schema: SchemaGraph,
    notes_content: str | None = None,
) -> dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]]:
    """
    Classify table roles, column roles, hints, and sensitivity using a profiling-driven base pass,
    then an unconditional second LLM pass (notes-aware when domain notes are present, otherwise
    cross-table consistency refinement).
    """
    tables_data = []
    for table in schema.tables.values():
        fks = [",".join(fk.src_cols) + "->" + fk.dst_table + "." + ",".join(fk.dst_cols) for fk in table.foreign_keys]
        column_profiles = [_build_column_profile_for_llm(col) for col in table.columns.values()]
        tables_data.append(
            {
                "table": table.name,
                "fks": fks,
                "columns": column_profiles,
            }
        )
    system_base = (
        "Classify every table's role and every column's role in this schema.\n\n"
        "TABLE ROLES:\n"
        "- dimension: reference/lookup table referenced by others, descriptive attributes\n"
        "- fact: transactional/event table with FKs to dimensions, contains measures\n"
        "- bridge: junction table for many-to-many, mostly FKs, few own columns\n"
        "- unknown: cannot confidently classify\n"
        "Use FK topology: tables referenced by many others are dimension; tables with many outbound FKs are fact; tables with only 2+ FKs and minimal columns are bridge.\n\n"
        "COLUMN ROLE DECISION PRIORITY (evaluate in order, first match wins):\n"
        "1. is_primary_key or is_foreign_key → identifier\n"
        "2. data_type is date/time/timestamp → temporal\n"
        "3. name suggests binary state (is_*, has_*, active) and distinct_count = 2 → boolean\n"
        "4. numeric and name suggests quantity/amount/duration/size/distance/price/count → numeric_measure (integer or decimal — data type does not restrict this)\n"
        "5. numeric and name suggests code/rating/level/rank/status/tier/type → numeric_categorical\n"
        "6. numeric with no clear name signal → numeric_measure (default for numeric)\n"
        "7. text and very high distinct_ratio → free_text\n"
        "8. text → categorical (default for text)\n\n"
        "AMBIGUOUS TWO-VALUE COLUMNS:\n"
        "When a column has exactly two sampled categorical values (e.g. positive/negative-style pairs), do not assume boolean unless name, type, FK topology, and profile_hints clearly support a flag.\n\n"
        "PROFILE HINTS (supporting evidence only — never override name/type signals):\n"
        "Each column may include a profile_hints object with distinct_count, distinct_ratio, null_ratio, and optionally top_values (a capped list of frequent sample values for qualifying low-cardinality string columns). Use these to confirm or disambiguate when name and type are ambiguous.\n"
        "Do NOT use profile_hints as the primary reason to choose a role.\n\n"
        "CROSS-TABLE CONSISTENCY:\n"
        "- Columns with the same name and data type across tables MUST receive the same role.\n"
        "- Deduce roles from names, types, FK topology, and profile_hints using the priority above.\n\n"
        "COLUMN HINTS:\n"
        "For each column, provide a short semantic hint (max 8 words) describing what the column represents in business terms. The hint should help map natural language to the correct column.\n"
        "Role-based guidance for hints:\n"
        "- identifier columns: describe what entity the ID refers to.\n"
        "- numeric_measure columns: state the unit or what is measured.\n"
        "- categorical columns: mention common category values or groupings.\n"
        "- temporal columns: state what event the date/time marks.\n"
        "- boolean columns: describe the yes/no condition.\n"
        "- FK columns: MUST state what business data the target table provides when joined. Name the key descriptive columns on the target table (e.g. 'links to target_table for name, title, description').\n\n"
        "TABLE DESCRIPTIONS:\n"
        "For each table provide a one-line business purpose that includes: (a) what entity or event the table represents, (b) which related tables it connects to via foreign keys, and (c) the notable descriptive or measure columns it provides that users commonly ask about.\n\n"
        "SENSITIVITY (per column, optional):\n"
        'Include "sensitivity" in each column object: always null in this pass.\n'
        "A later second-pass refine step may set \"pii\" only when domain notes explicitly require it.\n\n"
        "Reason internally, output only JSON:\n"
        '{"table1": {"table_role": "...", "description": "...", '
        '"columns": {"col1": {"role": "...", "hint": "...", "sensitivity": null}, ...}}, ...}'
    )
    user = stable_json({"tables": tables_data})
    raw_base = llm_chat(system_base, user, timeout=360.0, task="schema_base")
    base_payload = safe_json_loads(raw_base)
    if not base_payload or not isinstance(base_payload, dict):
        raise ValueError(f"LLM returned invalid JSON for schema classification (base): {raw_base[:200]}")
    notes_stripped = (notes_content or "").strip()
    final_payload: dict[str, Any] = base_payload
    if notes_stripped:
        user_refine = stable_json({"base_classification": base_payload, "domain_notes": notes_stripped})
        raw_refine = llm_chat(SCHEMA_NOTES_REFINE_SYSTEM, user_refine, timeout=360.0, task="schema")
        refined_payload = safe_json_loads(raw_refine)
        if refined_payload and isinstance(refined_payload, dict):
            final_payload = _merge_classification_payloads(base_payload, refined_payload)
    else:
        user_refine = stable_json({"base_classification": base_payload})
        raw_refine = llm_chat(SCHEMA_CONSISTENCY_REFINE_SYSTEM, user_refine, timeout=360.0, task="schema")
        refined_payload = safe_json_loads(raw_refine)
        if refined_payload and isinstance(refined_payload, dict):
            final_payload = _merge_classification_payloads(base_payload, refined_payload)
    return _normalize_llm_classification_payload(final_payload)


def apply_column_roles_llm(
    schema: SchemaGraph,
    notes_content: str | None = None,
    *,
    skip_columns: set[tuple[str, str]] | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """
    Apply LLM-inferred roles, descriptions, and sensitivity to the schema in-place.

    Args:

        schema: The `SchemaGraph` to update in-place.

        notes_content: Optional domain notes forwarded to the classification LLM.

        skip_columns: Optional set of ``(table_name, column_name)`` pairs whose existing role/description/sensitivity must be preserved as-is. The classifier still queries the LLM for these columns (full schema context) but applied results are discarded for any pair listed here.

        log_sink: Optional callback for user-facing classification status lines; defaults to :func:`_core_utils.notify`.

    Returns:

        Return value.
    """

    sink_call: Callable[[str], None] = log_sink if log_sink is not None else _core_utils.notify
    skip_columns = skip_columns or set()
    debug(f"[schema_profiling.apply_column_roles_llm] classifying {len(schema.tables)} tables via LLM (base + refine)")
    total_columns = sum(len(table.columns) for table in schema.tables.values())
    debug(f"[schema_profiling.apply_column_roles_llm] total columns: {total_columns}")
    sink_call(
        f"  Classifying {len(schema.tables)} tables / {total_columns} columns via LLM (this can take a minute)..."
    )
    role_counts: dict[str, int] = {}
    table_role_counts: dict[str, int] = {}
    llm_success = 0
    llm_fallback = 0
    success = False
    for attempt in range(QSimConfig.MAX_ROLE_CLASSIFICATION_RETRIES + 1):
        try:
            classifications = _llm_classify_schema(schema, notes_content)
            all_hard_errors = []
            all_soft_warnings = []
            for table in schema.tables.values():
                if table.name not in classifications:
                    all_hard_errors.append(f"{table.name}: missing from LLM response")
                    continue
                table_role, _desc, column_classifications = classifications[table.name]
                for col in table.columns.values():
                    if col.name not in column_classifications:
                        all_hard_errors.append(f"{table.name}.{col.name}: missing from LLM response")
                        continue
                    role, _hint, _sens = column_classifications[col.name]
                    hard_errors, soft_warnings = _validate_column_classification(col, role)
                    all_hard_errors.extend([f"{table.name}.{e}" for e in hard_errors])
                    all_soft_warnings.extend([f"{table.name}.{w}" for w in soft_warnings])
            for warning in all_soft_warnings:
                debug(f"[apply_column_roles_llm] WARNING: {warning}")
            if all_hard_errors:
                debug(
                    f"[apply_column_roles_llm] {len(all_hard_errors)} hard errors (attempt {attempt + 1}): {all_hard_errors[:5]}"
                )
                continue
            for table in schema.tables.values():
                if table.name in classifications:
                    table_role, description, column_classifications = classifications[table.name]
                    if can_overwrite_role(table.role_owner, RoleOwner.LLM):
                        table.role = table_role
                        table.role_owner = RoleOwner.LLM
                    set_description(table, description, DescriptionOwner.LLM_REFINEMENT)
                    table_role_counts[table_role] = table_role_counts.get(table_role, 0) + 1
                    for col in table.columns.values():
                        if col.name in column_classifications:
                            if (table.name, col.name) in skip_columns:
                                continue
                            role, hint, sensitivity = column_classifications[col.name]
                            if can_overwrite_role(col.role_owner, RoleOwner.LLM):
                                col.role = role
                                col.role_owner = RoleOwner.LLM
                            set_description(col, hint, DescriptionOwner.LLM_REFINEMENT)
                            set_sensitivity(col, sensitivity)
                            if col.is_primary_key or col.is_foreign_key:
                                if col.role != ColumnRole.IDENTIFIER.value and can_overwrite_role(
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
                                    if col.role != ColumnRole.BOOLEAN.value and can_overwrite_role(
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
                                    and can_overwrite_role(col.role_owner, RoleOwner.PROFILE)
                                ):
                                    debug(
                                        f"[apply_column_roles_llm] override {table.name}.{col.name}: {col.role} → temporal (date type)"
                                    )
                                    col.role = ColumnRole.TEMPORAL.value
                                    col.role_owner = RoleOwner.PROFILE
                            role_counts[col.role] = role_counts.get(col.role, 0) + 1
            success = True
            llm_success = len(schema.tables)
            debug("[apply_column_roles_llm] two-phase classification successful")
            break
        except Exception as e:
            debug(f"[apply_column_roles_llm] attempt {attempt + 1} failed: {e}")
            continue
    if not success:
        debug("[apply_column_roles_llm] LLM failed, using heuristic fallback for all tables")
        for table in schema.tables.values():
            llm_fallback += 1
            fk_out = len(table.foreign_keys)
            fk_in = sum(1 for t in schema.tables.values() for fk in t.foreign_keys if fk.dst_table == table.name)
            if fk_out >= 2:
                table_role = TableRole.FACT.value
            elif fk_out == 0 and fk_in >= 1:
                table_role = TableRole.DIMENSION.value
            elif fk_out == 2 and fk_in == 0 and len(table.columns) <= 4:
                table_role = TableRole.BRIDGE.value
            else:
                table_role = TableRole.UNKNOWN.value
            if can_overwrite_role(table.role_owner, RoleOwner.PROFILE):
                table.role = table_role
                table.role_owner = RoleOwner.PROFILE
            table_role_counts[table_role] = table_role_counts.get(table_role, 0) + 1
            for col in table.columns.values():
                if (table.name, col.name) in skip_columns:
                    continue
                role = _infer_column_role(col)
                if can_overwrite_role(col.role_owner, RoleOwner.PROFILE):
                    col.role = role.value
                    col.role_owner = RoleOwner.PROFILE
                role_counts[role.value] = role_counts.get(role.value, 0) + 1
    debug(f"[apply_column_roles_llm] completed: {llm_success} LLM, {llm_fallback} fallback")
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


def _strip_leading_articles(value: str) -> str:
    """Lowercase strip and remove leading ``a`` / ``an`` prefixes for comparison."""

    s = value.lower().strip()
    for prefix in BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix) :].strip()
    return s


def _coerce_zero_one_column(col: ColumnMetadata) -> bool:
    """Return True if column was coerced to boolean from numeric or string ``0``/``1``."""

    if col.distinct_count != 2 or not col.top_k_values or len(col.top_k_values) != 2:
        return False
    bucket: set[str] = set()
    for v in col.top_k_values:
        s = str(v).strip().lower()
        if s in ("0", "0.0"):
            bucket.add("z")
        elif s in ("1", "1.0"):
            bucket.add("o")
        else:
            return False
    if bucket != {"z", "o"}:
        return False
    if can_overwrite_role(col.role_owner, RoleOwner.BOOLEAN_COERCION):
        col.role = ColumnRole.BOOLEAN.value
        col.role_owner = RoleOwner.BOOLEAN_COERCION
        for v in col.top_k_values:
            s = str(v).strip().lower()
            if s in ("1", "1.0"):
                col.boolean_truth_value = str(v).strip()
                break
    return True


def _coerce_antonym_pair_column(col: ColumnMetadata) -> bool:
    """Return True if two string values match configured negation affix rules."""

    if col.distinct_count != 2 or not col.top_k_values or len(col.top_k_values) != 2:
        return False
    raw_a, raw_b = col.top_k_values[0], col.top_k_values[1]
    a = _strip_leading_articles(str(raw_a))
    b = _strip_leading_articles(str(raw_b))
    if len(a) < BOOLEAN_ANTONYM_MIN_STEM_LEN or len(b) < BOOLEAN_ANTONYM_MIN_STEM_LEN:
        return False
    for prefix in BOOLEAN_NEGATION_PREFIXES:
        if len(a) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and b == prefix + a:
            if can_overwrite_role(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_a).strip()
            return True
        if len(b) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and a == prefix + b:
            if can_overwrite_role(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_b).strip()
            return True
    for suffix in BOOLEAN_NEGATION_SUFFIXES:
        if len(a) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and b == a + suffix:
            if can_overwrite_role(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_a).strip()
            return True
        if len(b) >= BOOLEAN_ANTONYM_MIN_STEM_LEN and a == b + suffix:
            if can_overwrite_role(col.role_owner, RoleOwner.BOOLEAN_COERCION):
                col.role = ColumnRole.BOOLEAN.value
                col.role_owner = RoleOwner.BOOLEAN_COERCION
                col.boolean_truth_value = str(raw_b).strip()
            return True
    return False


def apply_boolean_coercion_pass(schema: SchemaGraph) -> None:
    """
    Deterministically promote two-value columns to boolean using literals and affix rules.

    Args:

        schema: Graph whose column roles and boolean mappings may be updated in-place.

    Returns:

        None.
    """

    for table in schema.tables.values():
        for col in table.columns.values():
            if col.is_primary_key or col.is_foreign_key:
                continue
            bl_pat, bl_truth = _has_boolean_like_values(col)
            if bl_pat:
                if col.role != ColumnRole.BOOLEAN.value and can_overwrite_role(
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
    """
    Assign valid filter, aggregation, and HAVING ops to each column based on its final role.

    Args:

        schema: The `SchemaGraph` to update in-place.

    Returns:

        Return value.
    """
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
                    col.valid_filter_ops = [
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
                    col.valid_filter_ops = [
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
                    col.valid_filter_ops = ["=", "!=", "in", "not in"] + null_ops
            elif col.is_primary_key or col.is_foreign_key or role == ColumnRole.IDENTIFIER.value:
                col.valid_filter_ops = [
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
                col.valid_filter_ops = [
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
                col.valid_filter_ops = [
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
                col.valid_filter_ops = [
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
                col.valid_filter_ops = [
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
                col.valid_filter_ops = ["=", "!=", "in", "not in"] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif role == ColumnRole.FREE_TEXT.value:
                col.valid_filter_ops = [
                    "like",
                    "ilike",
                    "not like",
                    "not ilike",
                ] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = []
            else:
                col.valid_filter_ops = ["=", "!="] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]

            is_arr, elt = _array_element_type_from_data_type(col.data_type)
            if is_arr and role != ColumnRole.AUDIT.value:
                col.element_type = elt or "string"
                col.valid_filter_ops = ["contains"] + null_ops
                col.valid_aggregations = ["count"]
                col.valid_having_ops = ["=", "!=", "<", "<=", ">", ">="]
            elif is_arr:
                col.element_type = elt or "string"

            if not string:
                col.valid_filter_ops = [op for op in col.valid_filter_ops if op not in string_only_ops]
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
                    col.valid_filter_ops = [op for op in col.valid_filter_ops if op in pattern_ops]
                else:
                    col.valid_filter_ops = []

    debug("[schema_profiling.assign_column_ops] completed")


def extract_tables_from_catalog(
    spark,
    catalog: str,
    schema: str,
    *,
    allow_objects: frozenset[str] | None = None,
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> dict[str, dict]:
    """
    Extract full table metadata from a Databricks Unity Catalog schema.

    Args:

        spark: Active `SparkSession`.

        catalog: The catalog name.

        schema: The schema (database) name.

        allow_objects: When set, restrict catalog extraction to these relation names (case-insensitive).

        structural_constraints_index: PK, FK, and single-column UNIQUE metadata from the active dialect.

    Returns:

        `primary_keys`, `foreign_keys`, `table_comment`, and `properties`.
    """
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

        for col in cols:
            col_name = col["col_name"]

            if col_name.startswith("#"):
                break

            column_names.append(col_name)
            column_types.append(col["data_type"])

        bundle = structural_constraints_index.tables.get(str(table_name).lower())
        use_info_schema = bundle is not None and bool(
            bundle.primary_keys or bundle.foreign_keys or bundle.unique_columns
        )
        if use_info_schema:
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
                    pk_dd, fk_dd, uq_dd = _parse_unity_catalog_constraints(create_stmt)
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
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "unique_columns": unique_columns,
            "partition_columns": partition_columns,
            "table_comment": table_comment,
            "properties": properties,
        }

    debug(f"[schema_profiling.extract_tables_from_catalog] complete: {len(tables)} tables")
    return tables


def _extract_partition_columns_from_describe_detail_spark(spark, full_table: str) -> list[str]:
    """
    Extract partition column names via DESCRIBE DETAIL, with INFORMATION_SCHEMA fallback.

    Args:

        spark: Active `SparkSession`.

        full_table: Fully qualified table name (`catalog.schema.table`).

    Returns:

        List of partition column name strings.
    """
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


def _extract_partition_columns_from_describe_detail_sql_connector(connection, full_table: str) -> list[str]:
    """
    Extract partition column names via DESCRIBE DETAIL, with INFORMATION_SCHEMA fallback.

    Args:

        connection: Active `databricks.sql` connection.

        full_table: Fully qualified table name (`catalog.schema.table`).

    Returns:

        List of partition column name strings.
    """
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
    """
    Extract partition column names from a CREATE TABLE DDL string.

    Args:

        create_stmt: The raw CREATE TABLE DDL string.

    Returns:

        List of partition column name strings, or empty list if not found.
    """
    match = re.search(r"PARTITIONED\s+BY\s*\(([^)]+)\)", create_stmt, re.IGNORECASE)
    if not match:
        return []
    return [c.strip().strip("`").strip('"') for c in match.group(1).split(",")]


def _partition_column_names_from_create_ddl(create_stmt_sql: str) -> list[str]:
    """
    Extract declarative partition columns from Hive-style or PostgreSQL CREATE DDL.

    Args:

        create_stmt_sql: Description.

    Returns:

        Return value.
    """
    spark_cols = _parse_partition_columns_from_create_stmt(create_stmt_sql)
    if spark_cols:
        return spark_cols
    match = re.search(
        r"\bPARTITION\s+BY\s+(?:RANGE|LIST|HASH)\s*\(\s*([^)]+)\s*\)",
        create_stmt_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    return [c.strip().strip("`").strip('"') for c in match.group(1).split(",") if c.strip()]


def _parse_unity_catalog_constraints(
    create_stmt: str,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Parse PRIMARY KEY, FOREIGN KEY, and single-column UNIQUE constraint fragments from Unity DDL text.

    Args:

        create_stmt: ``SHOW CREATE TABLE`` statement body or equivalent DDL string.

    Returns:

        Tuple of primary-key column names in encounter order, foreign-key dicts shaped for ``tables_meta``, and single-column unique column names.
    """
    primary_keys = []
    foreign_keys = []
    unique_columns: list[str] = []

    pk_pattern = r"(?:CONSTRAINT\s+\w+\s+)?PRIMARY\s+KEY\s*\(([^)]+)\)"
    pk_matches = re.findall(pk_pattern, create_stmt, re.IGNORECASE | re.DOTALL)
    for match in pk_matches:
        cols = [c.strip().strip("`").strip('"') for c in match.split(",")]
        primary_keys.extend(cols)

    ref_table_segment = (
        r"((?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))*)"
    )
    fk_pattern = (
        r"(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+"
        + ref_table_segment
        + r"\s*\(([^)]+)\)"
    )
    fk_matches = re.findall(fk_pattern, create_stmt, re.IGNORECASE | re.DOTALL)
    for match in fk_matches:
        src_cols = [c.strip().strip("`").strip('"') for c in match[0].split(",")]
        ref_table = _unity_trailing_table_identifier(match[1])
        ref_cols = [c.strip().strip("`").strip('"') for c in match[2].split(",")]

        foreign_keys.append({"src_cols": src_cols, "dst_table": ref_table, "dst_cols": ref_cols})

    for m in re.finditer(r"UNIQUE\s*\(([^)]+)\)", create_stmt, re.IGNORECASE):
        ucols = [c.strip().strip("`").strip('"') for c in m.group(1).split(",") if c.strip()]
        if len(ucols) == 1:
            unique_columns.append(ucols[0])

    return primary_keys, foreign_keys, unique_columns


def _balanced_paren_span(sql_text: str, open_paren_idx: int) -> tuple[int, int] | None:
    """
    Return ``(open_paren_idx, index_after_closing_paren)`` when parentheses balance from *open_paren_idx*.

    Depth scanning does not model nested quotes; unusual DDL with parentheses inside literal defaults may mis-scan.

    Args:

        sql_text: Full DDL source.

        open_paren_idx: Index of the opening ``(`` to match.

    Returns:

        Spans for a balanced block, or ``None`` when no matching ``)`` exists.
    """

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
    """
    Extract CREATE TABLE shapes using bracket balancing when structured parsers return nothing.

    Args:

        sql_content: Full DDL file contents.

    Returns:

        Table metadata dicts keyed by short table name.
    """

    tables: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+|TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"([\w.`\"\[\]]+(?:\.[\w.`\"\[\]]+)?)\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(sql_content):
        raw = m.group(1).strip().strip("`\"")
        if "." in raw:
            tname = raw.split(".")[-1].strip("`\"")
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
    """
    Return True when any reflected table already carries at least one FK edge.

    Args:

        sg: Schema graph after catalog reflection.

    Returns:

        Whether DDL LLM augmentation for FK discovery may be skipped.
    """

    for tbl in sg.tables.values():
        if tbl.foreign_keys:
            return True
    return False


def _canonicalize_llm_ddl_table_row(tinfo: dict[str, Any]) -> dict[str, Any]:
    """
    Map LLM DDL JSON (with optional ``column_not_null`` / ``column_unique`` arrays) to the
    canonical metadata shape used by deterministic DDL parsing.
    """

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


def _parse_sql_file_via_llm(sql_content: str) -> dict[str, dict]:
    """
    Invoke JSON-mode DDL parsing when deterministic parsers produced nothing.

    Args:

        sql_content: Full DDL file contents.

    Returns:

        Parsed table metadata dict or empty dict when the LLM yields unusable JSON.
    """

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

    response = llm_chat(system, user, task="ddl")
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


def parse_sql_file(
    sql_path: Path,
    *,
    reflected_schema: SchemaGraph | None = None,
) -> dict[str, dict]:
    """
    Parse CREATE TABLE and in-order ALTER TABLE DDL from a SQL file, with conditional LLM fallback.

    Args:

        sql_path: Path to the SQL file to parse.

        reflected_schema: When structural FK edges already exist on this graph, deterministic-parse emptiness skips the DDL LLM.

    Returns:

        Table metadata dicts keyed by short table name for merge paths that consume parsed DDL.
    """
    with open(sql_path, encoding="utf-8-sig") as f:
        sql_content = f.read()

    debug(f"[schema_profiling.parse_sql_file] reading: {len(sql_content)} chars")

    tables = _parse_sql_file_fallback(sql_content)
    if tables:
        debug(f"[schema_profiling.parse_sql_file] ddl_parser parsed: {len(tables)} tables")
        return tables

    tables = _parse_sql_file_regex_reflect(sql_content)
    if tables:
        debug(f"[schema_profiling.parse_sql_file] regex_reflect parsed: {len(tables)} tables")
        return tables

    if reflected_schema is not None and _schema_graph_has_structural_foreign_keys(reflected_schema):
        debug(
            "[schema_profiling.parse_sql_file] deterministic parsers returned 0 tables; "
            "reflection already has FK edges — skipping DDL LLM fallback",
        )
        return {}

    return _parse_sql_file_via_llm(sql_content)


def _parse_sql_file_fallback(sql_content: str) -> dict[str, dict]:
    """
    Parse CREATE TABLE and ALTER TABLE metadata from DDL using the active engine profile.

    Args:

        sql_content: Full SQL file contents.

    Returns:

        TABLE nodes exist.
    """
    if EngineConfig.TYPE == "databricks":
        return _parse_sql_file_sqlglot_spark(sql_content)
    return _parse_sql_file_pglast_postgres(sql_content)


def _extract_column_block_from_create(create_expr: Any) -> str:
    """
    Extract the inner content of the column definition block from a sqlglot Create.

    Args:

        create_expr: A parsed sqlglot `Create` expression.

    Returns:

        The column definition string with surrounding parentheses stripped.
    """
    schema = create_expr.this if create_expr.this else create_expr.expression
    if schema is None:
        return ""
    expressions = getattr(schema, "expressions", None)
    if expressions:
        return ", ".join(e.sql() for e in expressions if hasattr(e, "sql"))
    schema_sql = schema.sql()
    if schema_sql.startswith("(") and schema_sql.endswith(")"):
        return schema_sql[1:-1].strip()
    full_sql = create_expr.sql()
    match = re.search(r"\(([\s\S]*)\)\s*(?:PARTITIONED|STORED|LOCATION|AS|$)", full_sql)
    if match:
        return match.group(1).strip()
    return schema_sql.strip()


def _parse_column_name_and_sql_type(line: str) -> tuple[str, str] | None:
    """
    Split a single DDL column line into name and full SQL type tokens.

    Args:

        line: One logical column definition (no top-level comma).

    Returns:

        `(column_name, type_sql)` or `None` when the line is not a column.
    """
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
    """
    Split a string by commas that are outside parentheses.

    Args:

        s: The string to split (e.g. a CREATE TABLE column block).

    Returns:

        List of non-empty segments between top-level commas.
    """
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
    """
    Infer whether a single-column DDL line allows NULLs.

    Inline ``PRIMARY KEY`` implies NOT NULL in SQL; ``NOT NULL`` is parsed explicitly.
    """

    if re.search(r"\bNOT\s+NULL\b", line_upper):
        return False
    if re.search(r"\bPRIMARY\s+KEY\b", line_upper):
        return False
    return True


def _parse_columns_and_constraints(
    col_block: str,
) -> tuple[list, list, list, list, list, list]:
    """
    Parse column definitions and constraints from a column block string.

    Args:

        col_block: Description.

    Returns:

        Tuple `(columns, types, pks, fks, unique_cols, column_is_nullable)` where
        ``column_is_nullable`` is a bool list parallel to ``columns`` (``False`` = NOT NULL
        or inline PRIMARY KEY). Table-level ``PRIMARY KEY (…)`` constraints force those
        columns to ``False`` in a post-pass.
    """
    lines = [line.strip() for line in _split_by_top_level_comma(col_block)]

    columns = []
    types = []
    pks = []
    fks = []
    unique_cols: list[str] = []
    column_is_nullable: list[bool] = []

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

        type_word_count = len(col_type.split())
        line_tokens = line.split()
        tail_tokens = line_tokens[1 + type_word_count :] if len(line_tokens) > 1 + type_word_count else []
        if "PRIMARY KEY" in line_upper:
            pks.append(col_name)
        elif any(t.upper() == "UNIQUE" for t in tail_tokens):
            unique_cols.append(col_name)

    col_index = {name: idx for idx, name in enumerate(columns)}
    for pk_name in pks:
        idx = col_index.get(pk_name)
        if idx is not None:
            column_is_nullable[idx] = False

    return columns, types, pks, fks, unique_cols, column_is_nullable


def _extract_pk_columns(line: str) -> list[str]:
    """
    Extract column names from a PRIMARY KEY (col1, col2) definition line.

    Args:

        line: A single DDL line starting with `PRIMARY KEY`.

    Returns:

        List of unquoted column name strings.
    """
    match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", line, re.IGNORECASE)
    if match:
        return [c.strip().strip("`").strip('"') for c in match.group(1).split(",")]
    return []


def _extract_fk_definition(line: str) -> dict:
    """
    Extract a FOREIGN KEY definition from a DDL constraint line.

    Args:

        line: A single DDL line starting with `FOREIGN KEY`.

    Returns:

        the pattern does not match.
    """
    match = re.search(
        r"FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)",
        line,
        re.IGNORECASE,
    )
    if match:
        return {
            "src_cols": [c.strip().strip("`").strip('"') for c in match.group(1).split(",")],
            "dst_table": match.group(2).strip("`").strip('"'),
            "dst_cols": [c.strip().strip("`").strip('"') for c in match.group(3).split(",")],
        }
    return None


def _table_metadata_dict_from_ddl_parts(
    table_name: str,
    col_block: str,
    full_create_sql: str,
) -> dict[str, Any]:
    """
    Build a single-table schema metadata dict from column text and full CREATE DDL.

    Args:

        table_name: Description.

        col_block: Description.

        full_create_sql: Description.

    Returns:

        Return value.
    """
    columns, types, pks, fks, uniqs, col_nullable = _parse_columns_and_constraints(col_block)
    partition_cols = _partition_column_names_from_create_ddl(full_create_sql)
    return {
        "table_name_original": table_name,
        "column_names_original": columns,
        "column_types": types,
        "column_is_nullable": col_nullable,
        "primary_keys": pks,
        "foreign_keys": fks,
        "unique_columns": list(dict.fromkeys(uniqs)),
        "partition_columns": partition_cols,
    }


def _pglast_create_table_name(relation: Any) -> str | None:
    """
    Resolve an unqualified table name from a pglast `RangeVar`.

    Args:

        relation: Description.

    Returns:

        Return value.
    """
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
    """
    Append a column and type to table DDL metadata when the column is not already present.

    Args:

        tmeta: Single-table dict from `_table_metadata_dict_from_ddl_parts`.

        col_name: Column name.

        col_type: SQL type string.

    Returns:

        Return value.
    """
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
    """
    Append primary-key column names without duplicates.

    Args:

        tmeta: Single-table metadata dict.

        names: PK column names to add.

    Returns:

        Return value.
    """
    pks: list[str] = tmeta["primary_keys"]
    for n in names:
        if n not in pks:
            pks.append(n)


def _table_meta_append_foreign_key(tmeta: dict[str, Any], fk: dict[str, Any]) -> None:
    """
    Append a foreign-key edge dict if an identical edge is not already recorded.

    Args:

        tmeta: Single-table metadata dict.

        fk: Dict with `src_cols`, `dst_table`, `dst_cols`.

    Returns:

        Return value.
    """
    fks: list[dict[str, Any]] = tmeta["foreign_keys"]
    if fk in fks:
        return
    fks.append(fk)


def _table_meta_extend_unique_columns(tmeta: dict[str, Any], names: list[str]) -> None:
    """
    Append unique column names while preserving first-seen order.

    Args:

        tmeta: Single-table metadata dict.

        names: Column names participating in UNIQUE.

    Returns:

        Return value.
    """
    uniq: list[str] = tmeta["unique_columns"]
    seen = set(uniq)
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)


def _pglast_string_sval(node: Any) -> str | None:
    """
    Read a pglast `String` node value as a plain identifier string.

    Args:

        node: AST node that may carry `sval`.

    Returns:

        The string or `None` when not present.
    """
    if node is None:
        return None
    sval = getattr(node, "sval", None)
    if isinstance(sval, str) and sval:
        return sval
    return None


def _pglast_pk_constraint_column_names(constraint: Any) -> list[str]:
    """
    Extract PRIMARY KEY column names from a pglast `Constraint` node.

    Args:

        constraint: A pglast `Constraint` with `contype` PRIMARY.

    Returns:

        Ordered list of PK column names.
    """
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
    """
    Build a `foreign_keys` entry dict from a pglast FOREIGN KEY `Constraint`.

    Args:

        constraint: A pglast `Constraint` with `contype` FOREIGN.

    Returns:

        FK dict or `None` when required pieces are missing.
    """
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
    """
    Resolve column name and type SQL text from a pglast `ColumnDef`.

    Args:

        column_def: pglast `ColumnDef`.

    Returns:

        `(name, type_sql)` or `None`.
    """
    if RawStream is None:
        return None
    colname = getattr(column_def, "colname", None)
    if not isinstance(colname, str) or not colname.strip():
        return None
    type_name = getattr(column_def, "typeName", None)
    if type_name is None:
        return None
    type_sql = RawStream()(type_name).strip()
    if not type_sql:
        return None
    return colname.strip(), type_sql


def _pglast_apply_alter_constraint(tmeta: dict[str, Any], constraint: Any) -> None:
    """
    Merge PRIMARY KEY, FOREIGN KEY, or single-column UNIQUE from a pglast `Constraint`.

    Args:

        tmeta: Target table metadata dict.

        constraint: pglast `Constraint` from ALTER ADD CONSTRAINT.

    Returns:

        Return value.
    """
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
    """
    Apply a pglast `AlterTableStmt` to entries in `tables`.

    Args:

        tables: Mutable map of table name to metadata dict.

        stmt: pglast `AlterTableStmt`.

    Returns:

        Return value.
    """
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
    """
    Resolve the unqualified table name from a sqlglot `Alter` expression.

    Args:

        alter: Parsed `sqlglot.exp.Alter` node.

    Returns:

        Table name or `None`.
    """
    this = getattr(alter, "this", None)
    if this is None:
        return None
    name = getattr(this, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().strip('`"')
    text = str(this).split(".")[-1].strip().strip('`"')
    return text or None


def _sqlglot_column_def_name_type(col_def: Any) -> tuple[str, str] | None:
    """
    Extract column name and Spark SQL type text from a sqlglot `ColumnDef`.

    Args:

        col_def: sqlglot `ColumnDef`.

    Returns:

        `(name, type_sql)` or `None`.
    """
    ident = getattr(col_def, "this", None)
    col_name = getattr(ident, "name", None) if ident is not None else None
    if not isinstance(col_name, str) or not col_name.strip():
        return None
    kind = getattr(col_def, "args", {}).get("kind")
    if kind is None:
        return None
    type_sql = kind.sql(dialect="spark").strip().upper()
    if not type_sql:
        return None
    return col_name.strip(), type_sql


def _sqlglot_schema_fk_reference(reference: Any) -> tuple[str, list[str]] | None:
    """
    Parse `REFERENCES tbl(cols)` from a sqlglot `ForeignKey.reference`.

    Args:

        reference: sqlglot `Reference` node.

    Returns:

        `(dst_table, dst_cols)` or `None`.
    """
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
    """
    Merge PRIMARY KEY, FOREIGN KEY, or single-column UNIQUE from a sqlglot constraint expression.

    Args:

        tmeta: Target table metadata.

        inner: `PrimaryKey`, `ForeignKey`, or `UniqueColumnConstraint`.

    Returns:

        Return value.
    """
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
        _table_meta_append_foreign_key(
            tmeta,
            {"src_cols": src_cols, "dst_table": dst_table, "dst_cols": dst_cols},
        )
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


def _sqlglot_apply_alter_action(tmeta: dict[str, Any], action: Any) -> None:
    """
    Apply one sqlglot ALTER action to table metadata.

    Args:

        tmeta: Target table metadata.

        action: `ColumnDef`, `Schema`, or `AddConstraint`.

    Returns:

        Return value.
    """
    if isinstance(action, sqlglot.exp.ColumnDef):
        parsed = _sqlglot_column_def_name_type(action)
        if parsed:
            _table_meta_append_column(tmeta, parsed[0], parsed[1])
        return
    if isinstance(action, sqlglot.exp.Schema):
        for e in getattr(action, "expressions", None) or ():
            if isinstance(e, sqlglot.exp.ColumnDef):
                _sqlglot_apply_alter_action(tmeta, e)
        return
    if isinstance(action, sqlglot.exp.AddConstraint):
        for c in getattr(action, "expressions", None) or ():
            if not isinstance(c, sqlglot.exp.Constraint):
                continue
            for inner in getattr(c, "expressions", None) or ():
                _sqlglot_apply_constraint_node(tmeta, inner)


def _apply_sqlglot_alter_table(tables: dict[str, dict[str, Any]], alter: Any) -> None:
    """
    Apply a sqlglot `Alter` (TABLE) statement to `tables`.

    Args:

        tables: Mutable map of table name to metadata.

        alter: sqlglot `Alter`.

    Returns:

        Return value.
    """
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
        _sqlglot_apply_alter_action(tmeta, action)


def _parse_sql_file_pglast_postgres(sql_content: str) -> dict[str, dict]:
    """
    Parse PostgreSQL-flavour CREATE TABLE nodes from DDL text using pglast.

    Args:

        sql_content: Description.

    Returns:

        Return value.
    """
    if (
        not _PG_LAST_SQL_AVAILABLE
        or pglast is None
        or CreateStmt is None
        or RawStream is None
        or AlterTableStmt is None
    ):
        debug("[schema_profiling._parse_sql_file_pglast_postgres] pglast not installed")
        return {}
    try:
        raw_statements = pglast.parse_sql(sql_content)
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
            col_block = ", ".join(RawStream()(elt) for elt in stmt.tableElts)
            full_sql = RawStream()(stmt)
            tables[table_name] = _table_metadata_dict_from_ddl_parts(table_name, col_block, full_sql)
            continue
        if isinstance(stmt, AlterTableStmt):
            _apply_pglast_alter_table_statement(tables, stmt)

    debug(f"[schema_profiling._parse_sql_file_pglast_postgres] complete: {len(tables)} tables")
    return tables


def _parse_sql_file_sqlglot_spark(sql_content: str) -> dict[str, dict]:
    """
    Parse Spark-style CREATE TABLE statements from DDL text using sqlglot.

    Args:

        sql_content: Description.

    Returns:

        Return value.
    """
    try:
        statements = sqlglot.parse(sql_content, dialect="spark")
    except Exception as exc:
        debug(f"[schema_profiling._parse_sql_file_sqlglot_spark] parse failed: {exc}")
        return {}

    tables: dict[str, dict[str, Any]] = {}
    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, sqlglot.exp.Create) and stmt.this:
            table_ref = stmt.this
            if hasattr(table_ref, "this") and table_ref.this is not None:
                table_name = getattr(table_ref.this, "name", None) or str(table_ref.this)
            else:
                table_name = getattr(table_ref, "name", None) or str(table_ref)
            if not table_name or "." in str(table_name):
                table_name = str(table_ref).split(".")[-1].strip('`"')
            table_name = str(table_name).strip('`"')
            if not table_name:
                continue
            debug(f"[schema_profiling._parse_sql_file_sqlglot_spark] parsing: {table_name}")
            col_block = _extract_column_block_from_create(stmt)
            full_stmt = stmt.sql(dialect="spark")
            tables[table_name] = _table_metadata_dict_from_ddl_parts(table_name, col_block, full_stmt)
            continue
        if isinstance(stmt, sqlglot.exp.Alter):
            _apply_sqlglot_alter_table(tables, stmt)

    debug(f"[schema_profiling._parse_sql_file_sqlglot_spark] complete: {len(tables)} tables")
    return tables


def compute_semantic_profile_join_neighbors(sg: SchemaGraph) -> None:
    """
    Populate ``semantic_join_neighbors`` on physical columns from ``semantic_distinct_values``.

    A pair ``(A.x, B.y)`` qualifies only when every gate below is satisfied: both endpoints have ``value_type == "string"`` (numeric value-overlap is forbidden because identifier ranges coincide too easily); at least one endpoint is a primary key OR has ``is_unique=True`` (an anchored side); the two tables are not already connected by any foreign key edge that joins these two specific columns; and the existing statistical floors ``PolicyConfig.SEMANTIC_JOIN_MIN_INTERSECTION``, ``PolicyConfig.SEMANTIC_JOIN_MIN_DISTINCT``, and ``PolicyConfig.SEMANTIC_JOIN_MIN_OVERLAP_RATIO`` all pass on the stored ascending distinct samples.

    Idempotent: clears then recomputes.
    """
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
            if (cmeta.value_type or "").strip().lower() != "string":
                continue
            if cmeta.semantic_distinct_values:
                entries.append((tname, cname, cmeta))

    for i, (t1, c1, m1) in enumerate(entries):
        s1 = set(m1.semantic_distinct_values)
        if not s1:
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
            s2 = set(m2.semantic_distinct_values)
            if not s2:
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
            if len(quad) != 4:
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


def replay_user_semantic_neighbors_to_columns(sg: SchemaGraph) -> None:
    """
    Mirror every quad in :attr:`TableMetadata._user_semantic_neighbors` onto the per-column ``semantic_join_neighbors`` lists.

    The quad list on each table is the single authoritative store for user-overridden semantic edges; the per-column lists are a derived read view consumed by join-graph traversal. Call this helper after appending new quads (for example in :func:`apply_schema_overrides_to_graph`) so the per-column projection stays in sync without needing to re-run profile-derived overlap discovery.

    Idempotent: existing per-column entries are not duplicated and the lists are returned in a stable sort order.
    """
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
