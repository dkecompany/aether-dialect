"""Sqlglot-backed database engine dialects: MySQL, MariaDB, DuckDB, SQLite, Redshift, Snowflake, SQL Server, Oracle, BigQuery, and Databricks."""

from __future__ import annotations

import csv
import hashlib
import importlib
import os
import re
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from ._config import (
    BigQueryRuntimeConfig,
    ConfigError,
    CsvRuntimeConfig,
    DatabricksRuntimeConfig,
    DuckDBRuntimeConfig,
    EngineConfig,
    EngineRuntimeConfig,
    MariaDBRuntimeConfig,
    MySQLRuntimeConfig,
    OracleRuntimeConfig,
    PolicyConfig,
    RedshiftRuntimeConfig,
    SnowflakeRuntimeConfig,
    SQLiteRuntimeConfig,
    SQLServerRuntimeConfig,
)
from ._constants import (
    BIGQUERY_QUERY_LOG_AVAILABILITY_SQL,
    BIGQUERY_QUERY_LOG_FETCH_SQL,
    BOOL_LITERALS,
    BQ_DEFAULT_PARTITION_LOOKBACK_DAYS,
    DUCKDB_PROFILING_SAMPLE_PREDICATE,
    INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL,
    MYSQL_DATE_WINDOW_SUBDAY_TRUNC_FORMAT,
    MYSQL_DATE_WINDOW_TRUNC_FORMAT,
    MYSQL_NO_BACKSLASH_ESCAPES_SQL_MODE_TOKEN,
    MYSQL_PROFILING_SAMPLE_PREDICATE,
    MYSQL_QUERY_LOG_AVAILABILITY_SQL,
    MYSQL_QUERY_LOG_FETCH_SQL,
    NAMED_PLACEHOLDER_RE,
    ORACLE_QUERY_LOG_AVAILABILITY_SQL,
    ORACLE_QUERY_LOG_FETCH_SQL,
    REDSHIFT_PROFILING_SAMPLE_PREDICATE,
    REDSHIFT_QUERY_LOG_AVAILABILITY_SQL,
    REDSHIFT_QUERY_LOG_FETCH_SQL,
    SNOWFLAKE_QUERY_LOG_AVAILABILITY_SQL,
    SNOWFLAKE_QUERY_LOG_FETCH_SQL,
    SQL_BIND_TOKEN_RE,
    SQLITE_PROFILING_SAMPLE_PREDICATE,
    SQLSERVER_QUERY_LOG_AVAILABILITY_SQL,
    SQLSERVER_QUERY_LOG_FETCH_SQL,
    SQLSERVER_QUERY_STORE_AVAILABILITY_SQL,
    SQLSERVER_QUERY_STORE_FETCH_SQL,
    SQLSERVER_SHOWPLAN_ROW_CACHE_MAX,
    UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL,
    UNITY_INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL,
    UNITY_INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL,
    UNITY_INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL,
    UNITY_INFORMATION_SCHEMA_TABLES_TABLE_TYPE_SQL,
    UNITY_INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL,
    UPLOAD_INGEST_ENGINE_NAMES,
    UPLOAD_STORE_FILENAME,
)
from ._contracts_base import (
    DatabasePingFailed,
    DataQualityReport,
    EngineContext,
    PredicateGroup,
    SchemaInclude,
    SchemaRole,
    SqlDiagnostic,
    SqlDiagnosticCode,
    StatementTimeoutError,
    TableKind,
)
from ._contracts_core import AccessError, ResultReaderKind, RuntimeIntent
from ._contracts_schema import (
    CatalogStructuralConstraintsIndex,
    ColumnMetadata,
    CsvSourceSelection,
    SchemaDiff,
    SchemaGraph,
    UploadIngestResult,
)
from ._data_quality import (
    PreparedRelation,
    excel_cell_to_text,
    infer_duckdb_column_type,
    parse_source_selections,
    pinned_names_from_schema_graph,
    prepare_relations_for_paths,
    validate_upload_sources,
)
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._dialect_sqlglot_helper import (
    ConnectorResultBackend,
    OracleResultBackend,
    PartitionSqlAdapter,
    ResultBackend,
    SqlalchemyExecutionMixin,
    SqlAlchemyResultBackend,
    SqlglotEngineDialect,
    SqlglotParseMixin,
    SqlServerResultBackend,
)
from ._schema_finalize import (
    apply_diff,
    finalize_with_structure,
    migrate_sidecar_for_diff,
)
from ._schema_graph import (
    allow_objects_lower_set,
    assign_schema_graph_hashes,
    diff_schemas,
    load_schema_graph_snapshot,
    raise_if_schema_unusable,
)
from ._schema_reflect import (
    load_or_create_schema_bigquery,
    load_or_create_schema_databricks,
    load_or_create_schema_duckdb,
    load_or_create_schema_mysql,
    load_or_create_schema_oracle,
    load_or_create_schema_redshift,
    load_or_create_schema_snowflake,
    load_or_create_schema_sqlite,
    load_or_create_schema_sqlserver,
    save_schema_to_cache,
    tables_meta_to_schema_graph,
)
from ._sql_gen import databricks_unqualify_agg_arg_sql
from ._utils import (
    bound_engine_runtime_config,
    cost_cap_active,
    debug,
    diagnostic_debug_enabled,
    effective_explain_timeout_ms,
    effective_statement_timeout_ms,
    engine_connect_likely_transient,
    progress,
    reconcile_execute_bind_params,
    refuse_unsafe_sql_string_literal_content,
    require_driver,
    sha256,
)
from ._utils_artifacts import (
    artifact_lock,
    read_artifact_manifest,
    read_gzip_json,
    write_artifact_manifest,
)


@dataclass(frozen=True)
class NoOpQueryLogSource:
    """Query-log source that reports unavailable and returns no historical SQL."""

    @staticmethod
    def _stable_sql_text_for_history(sql_text: str) -> str:
        """Replace inline numeric and single-quoted literals so hashed snapshots stay stable."""
        s = re.sub(r"\b\d+\.\d+\b", "<num>", sql_text)
        s = re.sub(r"\b\d+\b", "<num>", s)
        s = re.sub(r"'(?:[^']|'')*'", "<str>", s)
        return s

    @staticmethod
    def _bind_int_named_params(sql: str, params: dict[str, int]) -> str:
        """Substitute ``:name`` placeholders with integer literals for driver-agnostic execution."""
        out = sql
        for key, val in params.items():
            out = out.replace(f":{key}", str(int(val)))
        return out

    @staticmethod
    def _query_log_fetch_rows(conn: Any, stmt: str) -> list[tuple[Any, ...]]:
        """Execute *stmt* on *conn* and return rows, swallowing driver errors."""
        try:
            cur = conn.cursor()
        except Exception:
            return []
        try:
            cur.execute(stmt)
            rows = cur.fetchall() or []
        except Exception:
            try:
                cur.close()
            except (OSError, AttributeError, TypeError):
                pass
            return []
        try:
            cur.close()
        except (OSError, AttributeError, TypeError):
            pass
        return list(rows)

    @staticmethod
    def _query_log_sql_texts(rows: list[tuple[Any, ...]]) -> list[str]:
        """Extract and stabilize SQL text from the first column of each row."""
        out: list[str] = []
        for row in rows:
            if not row:
                continue
            raw_q = row[0]
            if raw_q is None:
                continue
            out.append(NoOpQueryLogSource._stable_sql_text_for_history(str(raw_q)))
        return out

    def is_available(self, conn: Any) -> bool:
        """Return False because the engine has no query-history catalog."""
        _ = conn
        return False

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Return an empty list because query history is not supported."""
        _ = conn, lookback_days, max_queries, min_runs, user_filter
        return []


@dataclass(frozen=True)
class DatabricksQueryLogSource:
    """Databricks query history fetcher."""

    def is_available(self, conn: Any) -> bool:
        """Return True when a Databricks session handle is present."""
        del conn
        return True

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
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
            except (OSError, AttributeError, TypeError):
                pass
            return []
        try:
            cur.close()
        except (OSError, AttributeError, TypeError):
            pass
        out: list[str] = []
        for row in rows:
            if not row:
                continue
            raw_q = row[0]
            if raw_q is None:
                continue
            out.append(NoOpQueryLogSource._stable_sql_text_for_history(str(raw_q)))
        return out


@dataclass(frozen=True)
class MySQLQueryLogSource:
    """MySQL performance_schema-backed query log."""

    def is_available(self, conn: Any) -> bool:
        """Return True when ``events_statements_history`` consumer is enabled."""
        if conn is None:
            return False
        rows = NoOpQueryLogSource._query_log_fetch_rows(conn, MYSQL_QUERY_LOG_AVAILABILITY_SQL)
        return bool(rows)

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Fetch recent statement digests from performance_schema."""
        del min_runs, user_filter
        lookback_microseconds = int(lookback_days) * 86400 * 1_000_000
        stmt = NoOpQueryLogSource._bind_int_named_params(
            MYSQL_QUERY_LOG_FETCH_SQL,
            {
                "lookback_microseconds": lookback_microseconds,
                "max_queries": int(max_queries),
            },
        )
        return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, stmt))


@dataclass(frozen=True)
class RedshiftQueryLogSource:
    """Redshift ``svl_qlog`` query log."""

    def is_available(self, conn: Any) -> bool:
        """Return True when ``stl_query`` is readable."""
        if conn is None:
            return False
        rows = NoOpQueryLogSource._query_log_fetch_rows(conn, REDSHIFT_QUERY_LOG_AVAILABILITY_SQL)
        return bool(rows)

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Fetch recent query texts from ``svl_qlog``."""
        del min_runs, user_filter
        stmt = NoOpQueryLogSource._bind_int_named_params(
            REDSHIFT_QUERY_LOG_FETCH_SQL, {"lookback_days": int(lookback_days), "max_queries": int(max_queries)}
        )
        return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, stmt))


@dataclass(frozen=True)
class SQLServerQueryLogSource:
    """SQL Server Query Store or ``sys.dm_exec_query_stats`` query log."""

    def _query_store_available(self, conn: Any) -> bool:
        """Return True when Query Store is enabled for the current database."""
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(SQLSERVER_QUERY_STORE_AVAILABILITY_SQL)
            rows = cur.fetchall() or []
            cur.close()
            return bool(rows)
        except Exception:
            return False

    def is_available(self, conn: Any) -> bool:
        """Return True when Query Store or DMVs are readable."""
        if conn is None:
            return False
        if self._query_store_available(conn):
            return True
        try:
            cur = conn.cursor()
            cur.execute(SQLSERVER_QUERY_LOG_AVAILABILITY_SQL)
            cur.close()
            return True
        except Exception:
            return False

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Fetch recent statement texts from Query Store, falling back to DMVs."""
        del min_runs, user_filter
        if self._query_store_available(conn):
            stmt = NoOpQueryLogSource._bind_int_named_params(
                SQLSERVER_QUERY_STORE_FETCH_SQL.replace(
                    "SELECT DISTINCT", f"SELECT DISTINCT TOP ({int(max_queries)})", 1
                ),
                {"lookback_days": int(lookback_days)},
            )
            return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, stmt))
        dmv_stmt = SQLSERVER_QUERY_LOG_FETCH_SQL.replace(
            "SELECT DISTINCT", f"SELECT DISTINCT TOP ({int(max_queries)})", 1
        )
        return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, dmv_stmt))


@dataclass(frozen=True)
class OracleQueryLogSource:
    """Oracle ``V$SQL`` query log with privilege probe."""

    def is_available(self, conn: Any) -> bool:
        """Return True when ``V$SQL`` is readable for the session."""
        if conn is None:
            return False
        rows = NoOpQueryLogSource._query_log_fetch_rows(conn, ORACLE_QUERY_LOG_AVAILABILITY_SQL)
        return bool(rows)

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Fetch recent statement texts from ``V$SQL``."""
        del min_runs, user_filter
        stmt = NoOpQueryLogSource._bind_int_named_params(
            ORACLE_QUERY_LOG_FETCH_SQL,
            {"lookback_days": int(lookback_days), "max_queries": int(max_queries)},
        )
        return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, stmt))


@dataclass(frozen=True)
class SnowflakeQueryLogSource:
    """Snowflake ``INFORMATION_SCHEMA.QUERY_HISTORY`` fetcher."""

    def is_available(self, conn: Any) -> bool:
        """Return True when query history is readable."""
        if conn is None:
            return False
        rows = NoOpQueryLogSource._query_log_fetch_rows(conn, SNOWFLAKE_QUERY_LOG_AVAILABILITY_SQL)
        return bool(rows)

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Fetch successful query texts from Snowflake query history."""
        del min_runs, user_filter
        stmt = NoOpQueryLogSource._bind_int_named_params(
            SNOWFLAKE_QUERY_LOG_FETCH_SQL, {"lookback_days": int(lookback_days), "max_queries": int(max_queries)}
        )
        return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, stmt))


@dataclass(frozen=True)
class BigQueryQueryLogSource:
    """BigQuery ``INFORMATION_SCHEMA.JOBS`` query log."""

    def is_available(self, conn: Any) -> bool:
        """Return True when the project jobs view is readable."""
        if conn is None:
            return False
        project = str(getattr(bound_engine_runtime_config(), "PROJECT", None) or "").strip()
        if not project:
            return False
        stmt = BIGQUERY_QUERY_LOG_AVAILABILITY_SQL.format(project=project)
        rows = NoOpQueryLogSource._query_log_fetch_rows(conn, stmt)
        return bool(rows)

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Fetch completed job query texts from ``INFORMATION_SCHEMA.JOBS``."""
        del min_runs, user_filter
        project = str(getattr(bound_engine_runtime_config(), "PROJECT", None) or "").strip()
        if not project:
            return []
        stmt = NoOpQueryLogSource._bind_int_named_params(
            BIGQUERY_QUERY_LOG_FETCH_SQL.format(project=project),
            {"lookback_days": int(lookback_days), "max_queries": int(max_queries)},
        )
        return NoOpQueryLogSource._query_log_sql_texts(NoOpQueryLogSource._query_log_fetch_rows(conn, stmt))


class DatabricksConnectorBackend(ConnectorResultBackend):
    """Databricks SQL warehouse connector cursor backend."""

    kind = ResultReaderKind.CONNECTOR

    def __init__(self, connection: Any) -> None:
        super().__init__(connection)

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        _ = max_rows, max_bytes

        def _prepare(cursor: Any) -> None:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                try:
                    self._connection.set_query_timeout(int(timeout_ms) // 1000)
                except (OSError, AttributeError, TypeError):
                    pass

        def _execute_sql(cursor: Any, exec_params: dict[str, Any] | None) -> None:
            bound_sql, bound_params = SqlglotEngineDialect.convert_colon_binds_to_pyformat(sql, exec_params)
            if bound_params:
                cursor.execute(bound_sql, bound_params)
            else:
                cursor.execute(bound_sql)

        yield from self._connector_fetch_rows_batched(
            sql,
            params,
            batch_rows=batch_rows,
            timeout_ms=timeout_ms,
            prepare_cursor=_prepare,
            execute_sql=_execute_sql,
        )


class DatabricksSparkBackend(ResultBackend):
    """PySpark / DatabricksSession SQL backend."""

    kind = ResultReaderKind.SPARK

    def __init__(self, spark: Any) -> None:
        self._spark = spark

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        _ = max_rows, max_bytes, batch_rows
        if timeout_ms is not None and cost_cap_active(timeout_ms):
            self._spark.conf.set("spark.databricks.sql.statementTimeout", f"{int(timeout_ms)}ms")
        tm_ex = effective_explain_timeout_ms()
        if tm_ex is not None and "EXPLAIN" in sql.upper():
            self._spark.conf.set("spark.databricks.sql.statementTimeout", f"{int(tm_ex)}ms")
        bound_sql, bound_params = SqlglotEngineDialect.convert_colon_binds_to_pyformat(sql, params)
        df = self._spark.sql(bound_sql, bound_params) if bound_params else self._spark.sql(sql)
        rows = [tuple(row) for row in df.collect()]
        if rows:
            yield tuple(rows)


class DatabricksSqlAlchemyBackend(SqlAlchemyResultBackend):
    """SQLAlchemy engine backend for Databricks warehouse URLs."""

    kind = ResultReaderKind.SQLALCHEMY


class BigQueryClientBackend(ResultBackend):
    """Google-cloud-bigquery Client query result backend."""

    kind = ResultReaderKind.BQ_CLIENT

    @staticmethod
    def _bq_scalar_query_parameters(params: dict[str, Any]) -> list[Any]:
        """Map bind values to BigQuery ``ScalarQueryParameter`` instances."""
        import google.cloud.bigquery

        out: list[Any] = []
        for key, val in params.items():
            if isinstance(val, bool):
                bq_type = "BOOL"
            elif isinstance(val, int):
                bq_type = "INT64"
            elif isinstance(val, float):
                bq_type = "FLOAT64"
            else:
                bq_type = "STRING"
                if val is not None and not isinstance(val, str):
                    val = str(val)
            out.append(google.cloud.bigquery.ScalarQueryParameter(str(key), bq_type, val))
        return out

    @staticmethod
    def _bq_bind_params_from_sql(sql: str, params: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return only parameters referenced as ``@name`` tokens in *sql*."""
        if not params:
            return None
        bound = {key: val for key, val in params.items() if re.search(rf"@{re.escape(key)}\b", sql)}
        return bound or None

    def __init__(
        self,
        client: Any,
        *,
        maximum_bytes_billed: int | None = None,
        job_timeout_ms: int | None = None,
        dialect_name: str = "",
    ) -> None:
        self._client = client
        self._maximum_bytes_billed = maximum_bytes_billed
        self._job_timeout_ms = job_timeout_ms
        self._dialect_name = dialect_name

    def _job_config(self, *, timeout_ms: int | None = None, params: dict[str, Any] | None = None) -> Any:
        import google.cloud.bigquery

        job_config = google.cloud.bigquery.QueryJobConfig(use_query_cache=False)
        if self._maximum_bytes_billed is not None:
            job_config.maximum_bytes_billed = int(self._maximum_bytes_billed)
        tm = timeout_ms if timeout_ms is not None else self._job_timeout_ms
        if tm is not None:
            job_config.job_timeout_ms = int(tm)
        if params:
            job_config.query_parameters = BigQueryClientBackend._bq_scalar_query_parameters(params)
        return job_config

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        if diagnostic_debug_enabled():
            debug(f"[{self._dialect_name}.execute] sql=\n{sql}")
        del max_rows, max_bytes
        try:
            job = self._client.query(sql, job_config=self._job_config(timeout_ms=timeout_ms, params=params))
            rows = job.result()

            def _iter_rows() -> Iterator[tuple[tuple[Any, ...], ...]]:
                batch: list[tuple[Any, ...]] = []
                for row in rows:
                    batch.append(tuple(row.values()))
                    if len(batch) >= batch_rows:
                        yield tuple(batch)
                        batch = []
                if batch:
                    yield tuple(batch)

            yield from _iter_rows()
        except Exception as e:
            err = str(e)
            if Dialect.is_permission_denied_error(err):
                raise AccessError("execute", err, reason="warehouse") from e
            raise


class BigQueryStorageBackend(ResultBackend):
    """BigQuery Storage API reader with client fallback."""

    kind = ResultReaderKind.BQ_STORAGE

    def __init__(
        self,
        client: Any,
        storage_client: Any,
        *,
        maximum_bytes_billed: int | None = None,
        job_timeout_ms: int | None = None,
        dialect_name: str = "",
    ) -> None:
        self._client_backend = BigQueryClientBackend(
            client, maximum_bytes_billed=maximum_bytes_billed, job_timeout_ms=job_timeout_ms, dialect_name=dialect_name
        )
        self._storage_client = storage_client

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        if diagnostic_debug_enabled():
            debug(f"[{self._client_backend._dialect_name}.execute] sql=\n{sql}")
        try:
            job = self._client_backend._client.query(
                sql, job_config=self._client_backend._job_config(timeout_ms=timeout_ms, params=params)
            )
            rows_iter = job.result(bqstorage_client=self._storage_client)

            def _iter_rows() -> Iterator[tuple[tuple[Any, ...], ...]]:
                batch: list[tuple[Any, ...]] = []
                for row in rows_iter:
                    batch.append(tuple(row.values()))
                    if len(batch) >= batch_rows:
                        yield tuple(batch)
                        batch = []
                if batch:
                    yield tuple(batch)

            yield from _iter_rows()
        except Exception:
            yield from self._client_backend.fetch_rows_batched(
                sql,
                params,
                batch_rows=batch_rows,
                max_rows=max_rows,
                max_bytes=max_bytes,
                timeout_ms=timeout_ms,
            )

    def fetch_arrow_table(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> Any:
        if diagnostic_debug_enabled():
            debug(f"[{self._client_backend._dialect_name}.execute] sql=\n{sql}")
        job = self._client_backend._client.query(
            sql, job_config=self._client_backend._job_config(timeout_ms=timeout_ms, params=params)
        )
        rows_iter = job.result(bqstorage_client=self._storage_client)
        to_arrow = getattr(rows_iter, "to_arrow", None)
        if callable(to_arrow):
            return to_arrow()
        raise RuntimeError("BigQuery storage result iterator does not expose to_arrow")


class SnowflakeArrowBackend(ResultBackend):
    """Snowflake connector Arrow or Snowpark collect backend."""

    kind = ResultReaderKind.SNOWFLAKE_ARROW

    @staticmethod
    def _format_result_backend_error(exc: Exception) -> str:
        """Build a diagnostic message from a result-backend fetch failure."""
        parts: list[str] = []
        msg = str(exc).strip()
        if msg:
            parts.append(msg)
        else:
            rep = repr(exc).strip()
            if rep:
                parts.append(rep)
        type_name = type(exc).__name__
        if type_name and not any(type_name in part for part in parts):
            parts.append(f"({type_name})")
        for attr in ("errno", "sqlstate", "sfqid"):
            val = getattr(exc, attr, None)
            if val:
                parts.append(f"{attr}={val}")
        return " ".join(parts) if parts else "Unknown error"

    @staticmethod
    def _arrow_table_to_tuples(table: Any) -> list[tuple[Any, ...]]:
        """Convert a PyArrow table to row tuples."""
        if table is None:
            return []
        num_cols = table.num_columns
        if num_cols == 0:
            return []
        columns = [table.column(i).to_pylist() for i in range(num_cols)]
        return list(zip(*columns, strict=True))

    def __init__(self, *, snowpark: Any | None = None, connection: Any | None = None) -> None:
        self._snowpark = snowpark
        self._connection = connection

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        _ = max_rows, max_bytes
        if self._snowpark is not None:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                secs = max(1, int(timeout_ms) // 1000)
                self._snowpark.sql(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}").collect()
            bound_sql, bound_params = SqlglotEngineDialect.convert_colon_binds_to_pyformat(sql, params)
            collected = (
                self._snowpark.sql(bound_sql, bound_params).collect()
                if bound_params
                else self._snowpark.sql(sql).collect()
            )
            rows: list[tuple[Any, ...]] = []
            for row in collected:
                if hasattr(row, "__iter__") and not isinstance(row, (str, bytes, dict)):
                    rows.append(tuple(row))
                elif hasattr(row, "asDict"):
                    rows.append(tuple(row.asDict().values()))
                else:
                    rows.append(row)
            if rows:
                yield tuple(rows)
            return
        if self._connection is None:
            raise RuntimeError("SnowflakeArrowBackend has no snowpark session or connection")
        cursor = self._connection.cursor()
        try:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                secs = max(1, int(timeout_ms) // 1000)
                cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}")
            bound_sql, bound_params = SqlglotEngineDialect.convert_colon_binds_to_pyformat(sql, params)
            if bound_params:
                cursor.execute(bound_sql, bound_params)
            else:
                cursor.execute(sql)
            if hasattr(cursor, "fetch_arrow_all"):
                try:
                    tuples = SnowflakeArrowBackend._arrow_table_to_tuples(cursor.fetch_arrow_all())
                    if tuples:
                        yield tuple(tuples)
                    return
                except (OSError, AttributeError, TypeError):
                    pass
            yield from ResultBackend.iter_fetchmany_batches(cursor.fetchmany, batch_rows)
        finally:
            cursor.close()

    def fetch_arrow_table(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> Any:
        if self._connection is None:
            raise RuntimeError("SnowflakeArrowBackend has no connection for Arrow fetch")
        cursor = self._connection.cursor()
        try:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                secs = max(1, int(timeout_ms) // 1000)
                cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}")
            bound_sql, bound_params = SqlglotEngineDialect.convert_colon_binds_to_pyformat(sql, params)
            if bound_params:
                cursor.execute(bound_sql, bound_params)
            else:
                cursor.execute(sql)
            fetch_arrow = getattr(cursor, "fetch_arrow_all", None)
            if not callable(fetch_arrow):
                raise RuntimeError("Snowflake cursor does not expose fetch_arrow_all")
            return fetch_arrow()
        finally:
            cursor.close()

    def cancel_statement(self) -> None:
        connection = self._connection
        if connection is None:
            return
        cancel = getattr(connection, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except (OSError, AttributeError, RuntimeError, TypeError):
                pass


class MySQLConnectorBackend(ConnectorResultBackend):
    """Pymysql DB-API backend for MySQL and MariaDB."""

    def __init__(self, connection: Any, *, reopen: Callable[[], Any] | None = None) -> None:
        super().__init__(connection, reopen=reopen)
        self._query_connection_id: int | None = None

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        def _prepare(cursor: Any) -> None:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_ms)}")

        def _execute(cursor: Any, exec_params: dict[str, Any] | None) -> None:
            cursor.execute(sql, exec_params or {})
            thread_id = getattr(self._connection, "thread_id", None)
            if callable(thread_id):
                self._query_connection_id = int(thread_id())

        yield from self._connector_fetch_rows_batched(
            sql,
            params,
            batch_rows=batch_rows,
            timeout_ms=timeout_ms,
            prepare_cursor=_prepare,
            execute_sql=_execute,
        )

    def cancel_statement(self) -> None:
        conn_id = self._query_connection_id
        if conn_id is None:
            return
        cursor = self._connection.cursor()
        try:
            cursor.execute(f"KILL QUERY {int(conn_id)}")
        except (OSError, AttributeError, RuntimeError, TypeError):
            pass


class RedshiftConnectorBackend(ConnectorResultBackend):
    """redshift_connector DB-API backend."""

    def __init__(self, connection: Any, *, reopen: Callable[[], Any] | None = None) -> None:
        super().__init__(connection, reopen=reopen)

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        def _prepare(cursor: Any) -> None:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                cursor.execute(f"SET statement_timeout = {int(timeout_ms)}")

        yield from self._connector_fetch_rows_batched(
            sql,
            params,
            batch_rows=batch_rows,
            timeout_ms=timeout_ms,
            prepare_cursor=_prepare,
        )


class DuckDBNativeBackend(ConnectorResultBackend):
    """Native duckdb connection backend."""

    def __init__(self, connection: Any, *, reopen: Callable[[], Any] | None = None) -> None:
        super().__init__(connection, reopen=reopen)

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        _ = timeout_ms, max_rows, max_bytes

        def _stream() -> Iterator[tuple[tuple[Any, ...], ...]]:
            reconciled = reconcile_execute_bind_params(sql, params)
            if reconciled is not None:
                exec_params = reconciled
            elif params and NAMED_PLACEHOLDER_RE.search(sql):
                exec_params = dict(params)
            else:
                exec_params = {}
            if exec_params and (SQL_BIND_TOKEN_RE.search(sql) or NAMED_PLACEHOLDER_RE.search(sql)):
                bound_sql, bound_list = SqlglotEngineDialect.bind_colon_parameters_for_duckdb(sql, exec_params)
                result = self._connection.execute(bound_sql, bound_list)
            elif exec_params:
                result = self._connection.execute(sql, exec_params)
            else:
                result = self._connection.execute(sql)
            fetchmany = getattr(result, "fetchmany", None)
            if callable(fetchmany):
                yield from ResultBackend.iter_fetchmany_batches(fetchmany, batch_rows)
                return
            rows = result.fetchall() if hasattr(result, "fetchall") else result
            tuples = [tuple(row) for row in (rows or [])]
            if tuples:
                yield tuple(tuples)

        def _collect() -> list[tuple[tuple[Any, ...], ...]]:
            return list(_stream())

        yield from cast(list[tuple[tuple[Any, ...], ...]], self._run_with_connection_retry(_collect))

    def cancel_statement(self) -> None:
        interrupt = getattr(self._connection, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except (OSError, AttributeError, RuntimeError, TypeError):
                pass


class SQLiteNativeBackend(ConnectorResultBackend):
    """Stdlib sqlite3 connection backend."""

    def __init__(self, connection: Any, *, reopen: Callable[[], Any] | None = None) -> None:
        super().__init__(connection, reopen=reopen)

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        _ = timeout_ms, max_rows, max_bytes
        yield from self._connector_fetch_rows_batched(
            sql,
            params,
            batch_rows=batch_rows,
            timeout_ms=timeout_ms,
        )


class _EmbeddedDuckDBResult:
    """Row result wrapper for a native duckdb execute call."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def fetchall(self) -> list[Any]:
        rows = self._result.fetchall() if hasattr(self._result, "fetchall") else self._result
        return list(rows or [])

    def fetchone(self) -> Any | None:
        rows = self.fetchall()
        return rows[0] if rows else None

    def scalar(self) -> Any:
        rows = self.fetchall()
        if not rows:
            return None
        row = rows[0]
        return row[0] if isinstance(row, tuple) else row


class _EmbeddedDuckDBConnection:
    """SQLAlchemy-shaped connection wrapper over one native duckdb handle."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @staticmethod
    def _bind_colon_parameters(sql: str, parameters: dict[str, Any]) -> tuple[str, list[Any]]:
        """Convert SQLAlchemy ``:name`` placeholders to duckdb positional binds."""
        return SqlglotEngineDialect.bind_colon_parameters_for_duckdb(sql, parameters)

    def execute(self, statement: Any, parameters: dict[str, Any] | None = None) -> _EmbeddedDuckDBResult:
        sql = statement.text if hasattr(statement, "text") else str(statement)
        params = parameters or {}
        if params:
            bound_sql, bound_params = _EmbeddedDuckDBConnection._bind_colon_parameters(sql, params)
            result = self._connection.execute(bound_sql, bound_params)
        else:
            result = self._connection.execute(sql)
        return _EmbeddedDuckDBResult(result)

    def __enter__(self) -> _EmbeddedDuckDBConnection:
        return self

    def __exit__(self, *_args: object) -> Literal[False]:
        return False


class _EmbeddedEnginePool:
    """Marker object identifying an embedded shared-connection engine facade."""


class _EmbeddedDuckDBEngine:
    """Minimal engine facade sharing one native duckdb connection for reflection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self.pool = _EmbeddedEnginePool()

    def connect(self) -> _EmbeddedDuckDBConnection:
        return _EmbeddedDuckDBConnection(self._connection)

    def raw_connection(self) -> Any:
        return self._connection

    def dispose(self) -> None:
        return None


class MySQLDialect(SqlglotEngineDialect):
    """MySQL dialect using sqlglot read=mysql and SQLAlchemy+pymysql execution."""

    name: str = "mysql"
    sqlglot_dialect: ClassVar[str] = "mysql"
    registry_canonical_rank: ClassVar[int] = 3
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @staticmethod
    def _open_mysql_connector(config: MySQLRuntimeConfig) -> Any:
        """Open a pymysql connection from runtime config."""
        import pymysql

        return pymysql.connect(
            host=str(config.HOST),
            port=int(config.PORT),
            user=str(config.USER),
            password=str(config.PASSWORD or ""),
            database=str(config.DATABASE or ""),
            charset="utf8mb4",
        )

    @staticmethod
    def _mysql_date_window_period_start_sql(unit: str, *, clock: str) -> str:
        """Return MySQL/MariaDB SQL for the start of the current calendar period."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "week":
            return f"DATE_SUB({clock}, INTERVAL WEEKDAY({clock}) DAY)"
        if unit_norm in MYSQL_DATE_WINDOW_SUBDAY_TRUNC_FORMAT:
            fmt = MYSQL_DATE_WINDOW_SUBDAY_TRUNC_FORMAT[unit_norm]
            return f"DATE_FORMAT({clock}, '{fmt}')"
        if unit_norm in MYSQL_DATE_WINDOW_TRUNC_FORMAT:
            fmt = MYSQL_DATE_WINDOW_TRUNC_FORMAT[unit_norm]
            return f"DATE_FORMAT({clock}, '{fmt}')"
        if unit_norm == "day":
            return clock
        if unit_norm == "quarter":
            return f"DATE_SUB(DATE_FORMAT({clock}, '%Y-%m-01'), INTERVAL ((MONTH({clock}) - 1) % 3) MONTH)"
        if unit_norm == "half_year":
            return f"IF(MONTH({clock}) <= 6, DATE_FORMAT({clock}, '%Y-01-01'), DATE_FORMAT({clock}, '%Y-07-01'))"
        return clock

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return MySQL's binary UTF-8 collation for reproducible ordering."""
        return "utf8mb4_bin"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Prefer pymysql connector backend with SQLAlchemy fallback."""
        self._native_connection: Any | None = None
        if sqlalchemy_engine is not None:
            super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        elif getattr(config, "PASSWORD", None) and getattr(config, "DATABASE", None):
            super().__init__(config, sqlalchemy_engine=None, open_sqlalchemy_engine=False)
            self._select_native_backend()
            if self._native_connection is None:
                self._open_sqlalchemy_engine()
        else:
            super().__init__(config, sqlalchemy_engine=None)
        SqlalchemyExecutionMixin.assert_one_live_handle(self)

    def _select_native_backend(self) -> None:
        """Attach pymysql connector when credentials are configured."""
        if not getattr(self.config, "PASSWORD", None) or not getattr(self.config, "DATABASE", None):
            return
        try:
            config = cast(MySQLRuntimeConfig, self.config)

            def _reopen() -> Any:
                return MySQLDialect._open_mysql_connector(config)

            self._native_connection = _reopen()
            self._backend = MySQLConnectorBackend(self._native_connection, reopen=_reopen)
        except Exception as exc:
            debug(f"[MySQLDialect._select_native_backend] pymysql unavailable: {exc!r}")

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True because MySQL exposes ``KILL QUERY`` for the active connection."""
        return True

    @property
    def logical_engine_name(self) -> str:
        return "MySQL"

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active MySQL row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return ResultReaderKind.SQLALCHEMY

    def can_explain(self) -> bool:
        """Return True when pymysql or SQLAlchemy can run EXPLAIN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=self._native_connection
        )

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a backtick-quoted ``database.table`` reference."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def catalog_name(self) -> str | None:
        """Return None so table qualification does not repeat the database name."""
        return None

    def _qualify_uses_backtick_identifiers(self) -> bool:
        """Return True because MySQL qualification uses backticks."""
        return True

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load InnoDB PK, FK, and UNIQUE metadata from ``information_schema``."""
        schema_name = self.schema_name()
        return SqlglotEngineDialect.structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=getattr(self, "_native_connection", None),
        )

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return MySQL ``CAST(… AS CHAR)`` for overlap sampling."""
        return f"CAST({expr} AS CHAR)"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from MySQL information_schema or SQL file fallback."""
        with self._temporary_reflection_engine() as reflection_engine:
            return load_or_create_schema_mysql(
                reflection_engine,
                include=include,
                allow_objects=allow_objects,
                deny_objects=deny_objects,
                schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
                sql_file=sql_file,
            )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return MySQL ``EXPLAIN FORMAT=JSON`` wrapper."""
        return f"EXPLAIN FORMAT=JSON {finalized_sql}"

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[Any], str]:
        """Parse MySQL ``EXPLAIN FORMAT=JSON`` rows into estimates and soft diagnostics."""
        _ = schema
        payload = rows[0][0] if rows else None
        diags = SqlglotEngineDialect.mysql_diagnostics_from_explain_json(payload, schema=schema)
        est_rows, est_bytes = SqlglotEngineDialect.mysql_root_plan_estimates(payload)
        return est_rows, est_bytes, diags, str(payload or "")

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return MySQL planner row estimate from ``EXPLAIN FORMAT=JSON``."""
        backend = self.result_backend
        if backend is None:
            return None
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            explain_sql = self.explain_statement_sql(finalized)
            rows = backend.fetch_rows(explain_sql, {})
            est_rows, _, _, _ = self.parse_explain_plan(list(rows), schema=schema)
            return est_rows
        except Exception:
            return None

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return MySQL ``MAX_EXECUTION_TIME`` session hint for profiling."""
        return f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_ms)}"

    def session_timezone_sql(self) -> str | None:
        """Return MySQL session time-zone lookup SQL."""
        return "SELECT @@session.time_zone"

    def query_log_source(self) -> Any | None:
        """Return the MySQL performance_schema query-log source."""
        return MySQLQueryLogSource()

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render MySQL date comparison using ``TIMESTAMPDIFF``. MySQL has no interval-typed ``date - date`` result, so a difference of two date columns cannot be compared against an ``INTERVAL`` literal. Use the scalar ``TIMESTAMPDIFF(unit, earlier, later)`` form (mirrors the SQL Server ``DATEDIFF`` shape), comparing the integer unit count directly."""
        unit_token = unit.upper()
        if minuend_sql and subtrahend_sql:
            sql = f"TIMESTAMPDIFF({unit_token}, {subtrahend_sql}, {minuend_sql}) {op} {amount}"
        else:
            sql = f"TIMESTAMPDIFF({unit_token}, {left_expr}, NOW()) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_containment(self, column_sql: str, value_param: str, value_type: str) -> str | None:
        """Render MySQL JSON-array containment with ``JSON_CONTAINS``."""
        _ = value_type
        return f"JSON_CONTAINS({column_sql}, JSON_QUOTE(CAST({value_param} AS CHAR)), '$')"

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render MySQL JSON array membership with native containment."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind in ("json_text_array", "native_array"):
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect=self.sqlglot_dialect,
            )
        norm_param = f"LOWER(TRIM(BOTH '%' FROM CAST(:{param_key} AS CHAR)))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS CHAR))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render MySQL ``JSON_TABLE`` unnest for SELECT list."""
        sql = f"JSON_TABLE({column_sql}, '$[*]' COLUMNS({alias} TEXT PATH '$')) AS jt"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render MySQL date window boundaries with ``DATE_SUB`` / ``DATE_ADD``."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            period_sql = MySQLDialect._mysql_date_window_period_start_sql(unit, clock=clock)
            sql = f"{column} {op} {period_sql}"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} DATE_SUB({clock}, INTERVAL {scaled} {plural_unit})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_trunc(self, unit: str, expr_sql: str) -> str:
        """Render MySQL calendar truncation with ISO week semantics."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "week":
            sql = f"DATE_SUB({expr_sql}, INTERVAL WEEKDAY({expr_sql}) DAY)"
        elif unit_norm == "quarter":
            sql = f"DATE_SUB(DATE_FORMAT({expr_sql}, '%Y-%m-01'), INTERVAL ((MONTH({expr_sql}) - 1) % 3) MONTH)"
        elif unit_norm == "half_year":
            sql = (
                f"IF(MONTH({expr_sql}) <= 6, DATE_FORMAT({expr_sql}, '%Y-01-01'), DATE_FORMAT({expr_sql}, '%Y-07-01'))"
            )
        elif unit_norm in MYSQL_DATE_WINDOW_TRUNC_FORMAT:
            fmt = MYSQL_DATE_WINDOW_TRUNC_FORMAT[unit_norm]
            sql = f"DATE_FORMAT({expr_sql}, '{fmt}')"
        elif unit_norm == "day":
            sql = f"DATE({expr_sql})"
        else:
            return super().render_date_trunc(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_extract(self, unit: str, expr_sql: str) -> str:
        """Render MySQL calendar extraction with ISO week numbering."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "week":
            sql = f"WEEK({expr_sql}, 3)"
        elif unit_norm == "quarter":
            sql = f"QUARTER({expr_sql})"
        elif unit_norm == "half_year":
            sql = f"IF(MONTH({expr_sql}) <= 6, 1, 2)"
        elif unit_norm == "dow":
            sql = f"WEEKDAY({expr_sql}) + 1"
        else:
            return super().render_extract(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on MySQL."""
        return f"LOWER({expr})"

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the MySQL clock expression for relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "CURRENT_TIMESTAMP()"
        return "CURRENT_DATE()"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return MySQL current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a MySQL ``WHERE RAND(seed)`` suffix for statistics."""
        _ = table_kind
        if not use_sample:
            return ""
        ratio = max(0.0001, min(1.0, sample_size / max(row_count, 1)))
        return f"WHERE {MYSQL_PROFILING_SAMPLE_PREDICATE.format(ratio=ratio, seed=random_seed)}"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """MySQL samples via a ``WHERE RAND()`` predicate inside a subquery."""
        _ = table_kind
        return True

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append MySQL partition predicates from ``information_schema.partitions`` metadata."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(sql, schema, intent)

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"GROUP_CONCAT({expr_sql} ORDER BY {order_by_sql} SEPARATOR {sep_sql})"
        return f"GROUP_CONCAT({expr_sql} SEPARATOR {sep_sql})"

    @property
    def supports_median(self) -> bool:
        return False

    @property
    def mysql_string_backslash_escapes(self) -> bool:
        """Return True when MySQL string literals honor backslash escape sequences."""
        conn = getattr(self, "_native_connection", None)
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT @@sql_mode")
                row = cursor.fetchone()
                if row and row[0]:
                    sql_mode = str(row[0]).upper()
                    return MYSQL_NO_BACKSLASH_ESCAPES_SQL_MODE_TOKEN not in sql_mode
            except (OSError, AttributeError, TypeError):
                pass
        return True

    def escape_string_literal(self, value: str) -> str:
        """Escape a MySQL string literal body, honoring server backslash-escape mode."""
        s = str(value)
        refuse_unsafe_sql_string_literal_content(s)
        if self.mysql_string_backslash_escapes:
            s = s.replace("\\", "\\\\")
        return s.replace("'", "''")

    def quote_string_literal(self, text: str) -> str:
        """Render a MySQL string literal with backslash and quote escaping."""
        return f"'{self.escape_string_literal(text)}'"


class MariaDBDialect(MySQLDialect):
    """MariaDB dialect that reuses MySQL SQL generation, execution, and reflection."""

    name: str = "mariadb"
    sqlglot_dialect: ClassVar[str] = "mysql"
    registry_canonical_rank: ClassVar[int] = 4

    @property
    def logical_engine_name(self) -> str:
        return "MariaDB"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from MariaDB information_schema or SQL file fallback."""
        with self._temporary_reflection_engine() as reflection_engine:
            return load_or_create_schema_mysql(
                reflection_engine,
                include=include,
                allow_objects=allow_objects,
                deny_objects=deny_objects,
                schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
                sql_file=sql_file,
            )


class RedshiftDialect(SqlglotEngineDialect):
    """Redshift dialect using sqlglot read=redshift and SQLAlchemy execution."""

    name: str = "redshift"
    sqlglot_dialect: ClassVar[str] = "redshift"
    registry_canonical_rank: ClassVar[int] = 7
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @staticmethod
    def _redshift_pruning_columns(table_meta: Any) -> list[str]:
        """Return Redshift sortkey and distkey columns for partition predicate injection."""
        cols: list[str] = []
        for sk in list(getattr(table_meta, "sortkey", []) or []):
            if sk:
                cols.append(str(sk))
        distkey = getattr(table_meta, "distkey", None)
        if distkey:
            cols.append(str(distkey))
        return cols

    @staticmethod
    def _open_redshift_connector(config: RedshiftRuntimeConfig) -> Any:
        """Open a redshift_connector connection from runtime config."""
        import redshift_connector

        kwargs: dict[str, Any] = {
            "host": str(config.HOST),
            "port": int(config.PORT),
            "database": str(config.DATABASE or ""),
            "user": str(config.USER),
        }
        if config.PASSWORD:
            kwargs["password"] = str(config.PASSWORD)
        kwargs["ssl"] = True
        return redshift_connector.connect(**kwargs)

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return Redshift's portable binary collation for reproducible ordering."""
        return "C"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Prefer redshift_connector backend with SQLAlchemy fallback."""
        self._native_connection: Any | None = None
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        self._select_native_backend()

    def _select_native_backend(self) -> None:
        """Attach redshift_connector when credentials are configured."""
        if not isinstance(self.config, RedshiftRuntimeConfig):
            return
        if not self.config.PASSWORD and not self.config.USE_IAM:
            return
        try:
            config = self.config

            def _reopen() -> Any:
                return RedshiftDialect._open_redshift_connector(config)

            self._native_connection = _reopen()
            self._backend = RedshiftConnectorBackend(self._native_connection, reopen=_reopen)
        except Exception as exc:
            debug(f"[RedshiftDialect._select_native_backend] redshift_connector unavailable: {exc!r}")

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return False because Redshift has no portable in-flight cancel hook."""
        return False

    @property
    def logical_engine_name(self) -> str:
        return "Redshift"

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active Redshift row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return ResultReaderKind.SQLALCHEMY

    def can_explain(self) -> bool:
        """Return True when redshift_connector or SQLAlchemy can run EXPLAIN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=self._native_connection
        )

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a double-quoted ``schema.table`` reference."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from ``information_schema`` and ``svv_foreign_keys``."""
        schema_name = self.schema_name()
        return SqlglotEngineDialect.structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=getattr(self, "_native_connection", None),
        )

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Normalize Redshift ``DATETRUNC`` emission to ``DATE_TRUNC``."""
        if stage != "post_substitute":
            return sql
        return SqlglotParseMixin.normalize_datetrunc_sql(sql, sqlglot_dialect=self.sqlglot_dialect)

    @property
    def supports_ilike(self) -> bool:
        """Redshift supports ``ILIKE`` via Postgres-compatible syntax."""
        return True

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from Redshift catalog views or SQL file fallback."""
        return load_or_create_schema_redshift(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
        )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return plain Redshift ``EXPLAIN`` wrapper."""
        return f"EXPLAIN {finalized_sql}"

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[Any], str]:
        """Parse Redshift ``EXPLAIN`` text rows into estimates and soft diagnostics."""
        _ = schema
        text_payload = "\n".join(str(r[0]) for r in rows if r and r[0] is not None)
        diags = SqlglotEngineDialect.redshift_diagnostics_from_explain_text(text_payload)
        est_rows, est_bytes = SqlglotEngineDialect.redshift_root_plan_estimates(text_payload)
        return est_rows, est_bytes, diags, text_payload

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return Redshift planner row estimate from ``EXPLAIN``."""
        eng = getattr(self, "engine", None)
        if eng is None:
            return None
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            explain_sql = self.explain_statement_sql(finalized)
            with eng.connect() as conn:
                rows = conn.execute(text(explain_sql), {}).fetchall()
            est_rows, _, _, _ = self.parse_explain_plan(list(rows), schema=schema)
            return est_rows
        except Exception:
            return None

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return Redshift ``statement_timeout`` for profiling sessions."""
        return f"SET statement_timeout = {int(timeout_ms)}"

    def session_timezone_sql(self) -> str | None:
        """Return Redshift session time-zone lookup SQL."""
        return "SELECT current_setting('timezone')"

    def query_log_source(self) -> Any | None:
        """Return the Redshift ``svl_qlog`` query-log source."""
        return RedshiftQueryLogSource()

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render Redshift interval date-difference comparison."""
        _ = minuend_sql, subtrahend_sql
        scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
        sql = f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render Redshift array membership, branching on column storage kind."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect=self.sqlglot_dialect,
            )
        norm_param = f"LOWER(BTRIM(CAST(:{param_key} AS VARCHAR), ' ' || CHR(34) || CHR(39)))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS VARCHAR))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Redshift SUPER unnest via lateral alias."""
        sql = f"{column_sql} AS arr, arr.{alias} AS {alias}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render Redshift date window boundaries."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', {clock})"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} {clock} - INTERVAL '{scaled} {plural_unit}'"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Redshift."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return Redshift current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a Redshift seeded hash-bucket ``WHERE`` suffix for statistics."""
        _ = table_kind
        if not use_sample:
            return ""
        ratio = max(0.0001, min(1.0, sample_size / max(row_count, 1)))
        return f"WHERE {REDSHIFT_PROFILING_SAMPLE_PREDICATE.format(ratio=ratio, seed=random_seed)}"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Redshift samples via a ``WHERE RANDOM()`` predicate inside a subquery."""
        _ = table_kind
        return True

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append sortkey and distkey predicates when schema and intent are available."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(
            sql, schema, intent, column_selector=RedshiftDialect._redshift_pruning_columns
        )


class SnowflakeDialect(SqlglotEngineDialect):
    """Snowflake dialect using sqlglot read=snowflake and SQLAlchemy execution."""

    name: str = "snowflake"
    sqlglot_dialect: ClassVar[str] = "snowflake"
    registry_canonical_rank: ClassVar[int] = 9
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @staticmethod
    def _snowflake_cluster_columns(table_meta: Any) -> list[str]:
        """Return clustering column names for Snowflake partition predicate injection."""
        key = getattr(table_meta, "clustering_key", None)
        if key:
            return [str(key)]
        fields = getattr(table_meta, "clustering_fields", None) or []
        return [str(c) for c in fields if c]

    @property
    def supports_ilike(self) -> bool:
        """Return True because Snowflake exposes ``ILIKE``."""
        return True

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True because the Snowflake connector exposes ``cancel``."""
        return True

    @property
    def logical_engine_name(self) -> str:
        return "Snowflake"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Open connector, SQLAlchemy, or Snowpark backends in priority order."""
        self._snowpark_session: Any | None = None
        self._snowflake_connection: Any | None = None
        if sqlalchemy_engine is not None:
            super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
            self._select_result_backend()
        else:
            require_driver(self.name)
            Dialect.__init__(self, config)
            self.engine = None
            self._backend = None
            if not self._try_connector_backend():
                self._open_sqlalchemy_engine()
                self._ensure_result_backend()
                if self._backend is None:
                    self._try_snowpark_backend()
        SqlalchemyExecutionMixin.assert_one_live_handle(self)

    def _try_connector_backend(self) -> bool:
        """Attach snowflake.connector when account credentials are configured."""
        config = cast(SnowflakeRuntimeConfig, self.config)
        if not (config.ACCOUNT and config.USER):
            return False
        try:
            import snowflake.connector

            conn_kwargs: dict[str, Any] = {
                "account": str(config.ACCOUNT),
                "user": str(config.USER),
            }
            if config.DATABASE:
                conn_kwargs["database"] = str(config.DATABASE)
            if config.SCHEMA:
                conn_kwargs["schema"] = str(config.SCHEMA)
            if config.WAREHOUSE:
                conn_kwargs["warehouse"] = str(config.WAREHOUSE)
            if config.ROLE:
                conn_kwargs["role"] = str(config.ROLE)
            if config.AUTHENTICATOR:
                conn_kwargs["authenticator"] = str(config.AUTHENTICATOR)
            if config.has_password_auth():
                conn_kwargs["password"] = str(config.PASSWORD or "")
            conn_kwargs.update(config.connect_args())
            self._snowflake_connection = snowflake.connector.connect(**conn_kwargs)
            self._backend = SnowflakeArrowBackend(connection=self._snowflake_connection)
            return True
        except Exception as exc:
            debug(f"[SnowflakeDialect.__init__] snowflake.connector unavailable: {exc!r}")
            return False

    def _try_snowpark_backend(self) -> bool:
        """Attach an active Snowpark session when reachable."""
        if not SnowflakeRuntimeConfig.snowpark_session_reachable():
            return False
        try:
            from snowflake.snowpark.context import get_active_session

            self._snowpark_session = get_active_session()
            self._backend = SnowflakeArrowBackend(snowpark=self._snowpark_session)
            return True
        except Exception as exc:
            debug(f"[SnowflakeDialect.__init__] Snowpark session unavailable: {exc!r}")
            return False

    def _select_result_backend(self) -> None:
        """Attach the active Snowflake row-fetch backend from connector, engine, or Snowpark."""
        if self._try_connector_backend():
            self.engine = None
            return
        if getattr(self, "engine", None) is not None:
            self._ensure_result_backend()
            if self._backend is not None:
                return
        self._try_snowpark_backend()

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active Snowflake row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return ResultReaderKind.SQLALCHEMY

    def quote_table_column(self, table: str, column: str) -> str:
        """Emit double-quoted Snowflake identifiers."""
        return Dialect.sqlglot_quote_table_column(table, column, self.sqlglot_dialect)

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a double-quoted ``database.schema.table`` reference."""
        _ = kind
        config = getattr(self, "config", None)
        database = str(getattr(config, "DATABASE", None) or "") if config is not None else ""
        schema = self.schema_name()
        if database and schema:
            return f"{self.quote_identifier(database)}.{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def _qualify_tables_for_execution(self, sql: str) -> str:
        """Qualify bare tables with quoted uppercase Snowflake three- part names."""
        sch = self.schema_name()
        cat = self.catalog_name()
        if not sch or not sql or not sql.strip():
            return sql
        try:
            parsed = sqlglot.parse_one(sql, read=self.sqlglot_dialect)
        except Exception:
            return sql
        cte_names_lower: set[str] = set()
        for cte in parsed.find_all(sqlglot.exp.CTE):
            alias = cte.alias_or_name
            if alias:
                cte_names_lower.add(str(alias).lower())
        for table in parsed.find_all(sqlglot.exp.Table):
            raw_name = str(table.name or "")
            if not raw_name:
                continue
            if raw_name.lower() in cte_names_lower:
                continue
            if table.args.get("db") or table.args.get("catalog"):
                continue
            table.set("this", sqlglot.exp.to_identifier(raw_name.upper(), quoted=True))
            table.set("db", sqlglot.exp.to_identifier(str(sch).upper(), quoted=True))
            if cat:
                table.set("catalog", sqlglot.exp.to_identifier(str(cat).upper(), quoted=True))
        try:
            return parsed.sql(dialect=self.sqlglot_dialect)
        except Exception:
            return sql

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from Snowflake ``information_schema``."""
        schema_name = self.schema_name()
        return SqlglotEngineDialect.structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=getattr(self, "_snowflake_connection", None),
        )

    def can_explain(self) -> bool:
        """Return True when Snowflake connector, Snowpark, or SQLAlchemy can run EXPLAIN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self,
            sqlalchemy_engine=self.engine,
            native_connection=getattr(self, "_snowflake_connection", None),
            spark_session=getattr(self, "_snowpark_session", None),
        )

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return Snowflake ``CAST(… AS VARCHAR)`` for overlap sampling."""
        return f"CAST({expr} AS VARCHAR)"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from Snowflake INFORMATION_SCHEMA or SQL file fallback."""
        with self._temporary_reflection_engine() as reflection_engine:
            return load_or_create_schema_snowflake(
                reflection_engine,
                include=include,
                allow_objects=allow_objects,
                deny_objects=deny_objects,
                schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
                sql_file=sql_file,
            )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return Snowflake ``EXPLAIN USING JSON`` wrapper."""
        return f"EXPLAIN USING JSON {finalized_sql}"

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[SqlDiagnostic], str]:
        """Parse Snowflake ``EXPLAIN USING JSON`` rows into estimates and soft diagnostics."""
        _ = schema
        payload = rows[0][0] if rows else None
        diags = SqlglotEngineDialect.snowflake_diagnostics_from_explain_json(payload)
        est_rows, est_bytes = SqlglotEngineDialect.snowflake_root_plan_estimates(payload)
        return est_rows, est_bytes, diags, str(payload or "")

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run Snowflake ``EXPLAIN USING JSON`` via the active result backend."""
        finalized = self.finalize_render(sql, params or {}, schema=schema, intent=intent)
        explain_sql = self.explain_statement_sql(finalized)
        try:
            backend = self.result_backend
            if backend is None:
                return True, [], ""
            tm = effective_explain_timeout_ms()
            rows = backend.fetch_rows(explain_sql, params, timeout_ms=tm)
            est_rows, est_bytes, soft_diags, plan_text = self.parse_explain_plan(list(rows), schema=schema)
            failed, why = Dialect.explain_cost_gate_violation(est_rows, est_bytes, dialect=self)
            if failed:
                return (
                    False,
                    soft_diags + [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                    why,
                )
            return True, soft_diags, plan_text
        except Exception as e:
            err = SnowflakeArrowBackend._format_result_backend_error(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)], err)

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return Snowflake planner row estimate from ``EXPLAIN USING JSON``."""
        backend = self.result_backend
        if backend is None and getattr(self, "engine", None) is None:
            return None
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            explain_sql = self.explain_statement_sql(finalized)
            if backend is not None:
                rows = backend.fetch_rows(explain_sql)
            else:
                if self.engine is None:
                    return None
                with self.engine.connect() as conn:
                    rows = conn.execute(text(explain_sql)).fetchall()
            est_rows, _, _, _ = self.parse_explain_plan(list(rows), schema=schema)
            return est_rows
        except Exception:
            return None

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return Snowflake session statement timeout for profiling."""
        secs = max(1, int(timeout_ms) // 1000)
        return f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}"

    def session_timezone_sql(self) -> str | None:
        """Return Snowflake session time-zone lookup SQL."""
        return "SELECT CURRENT_TIMEZONE()"

    def query_log_source(self) -> Any | None:
        """Return the Snowflake query-history query-log source."""
        return SnowflakeQueryLogSource()

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append cluster-key predicates when schema and intent carry date signals."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(
            sql, schema, intent, column_selector=SnowflakeDialect._snowflake_cluster_columns
        )

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render Snowflake date comparison using ``DATEDIFF``. Snowflake ``DATEDIFF(part, start, end)`` returns an integer count, so a two-column difference must pass the columns as separate ``start``/``end`` arguments (``date - date`` is not a valid Snowflake interval). Falls back to comparing a single date column against ``CURRENT_DATE()``."""
        if minuend_sql and subtrahend_sql:
            sql = f"DATEDIFF('{unit}', {subtrahend_sql}, {minuend_sql}) {op} {amount}"
        else:
            sql = f"DATEDIFF('{unit}', {left_expr}, CURRENT_DATE()) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render Snowflake array membership with trimmed, case- insensitive element comparison."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect=self.sqlglot_dialect,
            )
        trim_set = "CONCAT(' ', CHR(34), CHR(39))"
        norm_bind = f"LOWER(TRIM(CAST(:{param_key} AS VARCHAR), {trim_set}))"
        xform = f"TRANSFORM({column_sql}, _ac_x -> LOWER(TRIM(CAST(_ac_x AS VARCHAR), {trim_set})))"
        sql = f"({column_sql} IS NOT NULL AND ARRAY_CONTAINS({norm_bind}::VARIANT, {xform}))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Snowflake ``LATERAL FLATTEN`` unnest."""
        sql = f"LATERAL FLATTEN(INPUT => {column_sql}) {alias}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render Snowflake date window boundaries."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', {clock})"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            _ = plural_unit
            sql = f"{column} {op} DATEADD({unit}, -{scaled}, {clock})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Snowflake."""
        return f"LOWER({expr})"

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the Snowflake clock expression for relative date- window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "CURRENT_TIMESTAMP()"
        return "CURRENT_DATE()"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return Snowflake current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a Snowflake seeded ``SAMPLE`` suffix for statistics."""
        _ = row_count, table_kind
        if not use_sample:
            return ""
        pct = max(0.01, min(100.0, 100.0 * sample_size / max(row_count, 1)))
        return f"SAMPLE ({pct:.2f}) SEED ({random_seed})"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Snowflake ``SAMPLE`` applies directly on the scanned table."""
        _ = table_kind
        return False

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"LISTAGG({expr_sql}, {sep_sql}) WITHIN GROUP (ORDER BY {order_by_sql})"
        return f"LISTAGG({expr_sql}, {sep_sql})"

    def render_median(self, expr_sql: str) -> str:
        return f"MEDIAN({expr_sql})"


class SQLServerDialect(SqlglotEngineDialect):
    """SQL Server dialect using sqlglot read=tsql and SQLAlchemy+pyodbc execution."""

    name: str = "sqlserver"
    sqlglot_dialect: ClassVar[str] = "tsql"
    registry_canonical_rank: ClassVar[int] = 5
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True
    registry_array_contains_excluded: ClassVar[bool] = True

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return SQL Server's binary collation for reproducible ordering."""
        return "Latin1_General_BIN2"

    @property
    def supports_ilike(self) -> bool:
        """Return False because T-SQL has no ``ILIKE`` operator."""
        return False

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"STRING_AGG({expr_sql}, {sep_sql}) WITHIN GROUP (ORDER BY {order_by_sql})"
        return f"STRING_AGG({expr_sql}, {sep_sql})"

    def render_date_trunc(self, unit: str, expr_sql: str) -> str:
        """Render SQL Server calendar truncation with ISO week semantics."""
        unit_norm = (unit or "").strip().lower()
        cast_expr = f"CAST({expr_sql} AS date)"
        if unit_norm == "week":
            sql = f"DATEADD(day, -((DATEPART(weekday, {cast_expr}) + 5) % 7), {cast_expr})"
        elif unit_norm == "quarter":
            sql = f"DATEADD(quarter, DATEDIFF(quarter, 0, {cast_expr}), 0)"
        elif unit_norm == "half_year":
            sql = (
                f"CASE WHEN MONTH({cast_expr}) <= 6 "
                f"THEN DATEFROMPARTS(YEAR({cast_expr}), 1, 1) "
                f"ELSE DATEFROMPARTS(YEAR({cast_expr}), 7, 1) END"
            )
        elif unit_norm == "month":
            sql = f"DATEADD(month, DATEDIFF(month, 0, {cast_expr}), 0)"
        elif unit_norm == "year":
            sql = f"DATEFROMPARTS(YEAR({cast_expr}), 1, 1)"
        else:
            return super().render_date_trunc(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_extract(self, unit: str, expr_sql: str) -> str:
        """Render SQL Server calendar extraction with native quarter and half-year units."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "half_year":
            sql = f"CASE WHEN MONTH({expr_sql}) <= 6 THEN 1 ELSE 2 END"
        elif unit_norm == "quarter":
            sql = f"DATEPART(quarter, {expr_sql})"
        elif unit_norm == "week":
            sql = f"DATEPART(iso_week, {expr_sql})"
        else:
            return super().render_extract(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Attach SQL Server ODBC backend and SHOWPLAN diagnose cache."""
        self._showplan_row_cache: OrderedDict[str, float | None] = OrderedDict()
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True because SQL Server exposes ``KILL`` for the active session."""
        return True

    @property
    def logical_engine_name(self) -> str:
        return "SQL Server"

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a bracket-quoted ``schema.table`` reference."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from SQL Server ``information_schema``."""
        schema_name = self.schema_name()
        return SqlglotEngineDialect.structural_constraints_index_for_schema(
            self, schema_name, engine=getattr(self, "engine", None)
        )

    def can_explain(self) -> bool:
        """Return True when SQLAlchemy or ODBC can run SHOWPLAN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(self, sqlalchemy_engine=self.engine)

    def _ensure_result_backend(self) -> None:
        """Attach a SQL Server ODBC backend with driver-level command timeouts."""
        if self._backend is not None:
            return
        if getattr(self, "engine", None) is not None:
            self._backend = SqlServerResultBackend(self.engine, dialect_name=self.__class__.__name__)

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return ``None`` because T-SQL has no portable session statement-timeout statement. SQL Server statement timeouts are enforced via the ODBC driver command timeout in :class:`SqlServerResultBackend`."""
        _ = timeout_ms
        return None

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return T-SQL ``CAST(… AS NVARCHAR(4000))`` for overlap sampling."""
        return f"CAST({expr} AS NVARCHAR(4000))"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from ``sys.*`` catalog views or SQL file fallback."""
        return load_or_create_schema_sqlserver(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
        )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return finalized SQL for ``SHOWPLAN_ALL`` batches."""
        return finalized_sql

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[SqlDiagnostic], str]:
        """Parse SQL Server ``SHOWPLAN_ALL`` rows into estimates and soft diagnostics."""
        _ = schema
        diags = SqlglotEngineDialect.sqlserver_diagnostics_from_showplan_rows(rows)
        est_rows, est_bytes = SqlglotEngineDialect.sqlserver_root_plan_estimates(rows)
        plan_text = "\n".join(" | ".join(str(c) if c is not None else "" for c in row) for row in rows)
        return est_rows, est_bytes, diags, plan_text

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run T-SQL ``SHOWPLAN_ALL`` via separate driver batches."""
        finalized = self.finalize_render(sql, params or {}, schema=schema, intent=intent)
        try:
            if self.engine is None:
                return (
                    False,
                    [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message="no SQLAlchemy engine")],
                    "no SQLAlchemy engine",
                )
            raw = self.engine.raw_connection()
            try:
                cursor = raw.cursor()
                cursor.execute("SET SHOWPLAN_ALL ON")
                cursor.execute(finalized)
                rows = cursor.fetchall()
                cursor.execute("SET SHOWPLAN_ALL OFF")
                cursor.close()
                raw.commit()
            finally:
                raw.close()
            est_rows, est_bytes, soft_diags, plan_text = self.parse_explain_plan(list(rows), schema=schema)
            xml_diags: list[SqlDiagnostic] = []
            try:
                raw_xml = self.engine.raw_connection()
                try:
                    xml_cursor = raw_xml.cursor()
                    xml_cursor.execute("SET SHOWPLAN_XML ON")
                    xml_cursor.execute(finalized)
                    xml_rows = xml_cursor.fetchall()
                    xml_cursor.execute("SET SHOWPLAN_XML OFF")
                    xml_cursor.close()
                    raw_xml.commit()
                    xml_text = "\n".join(str(c) for row in xml_rows for c in row if c is not None)
                    xml_diags = SqlglotEngineDialect.sqlserver_diagnostics_from_showplan_xml(xml_text)
                finally:
                    raw_xml.close()
            except Exception:
                xml_diags = []
            merged_diags = list(soft_diags) + xml_diags
            failed, why = Dialect.explain_cost_gate_violation(est_rows, est_bytes, dialect=self)
            if failed:
                return (
                    False,
                    merged_diags + [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                    why,
                )
            cache_key = finalized.strip()
            if est_rows is not None:
                self._showplan_row_cache[cache_key] = est_rows
                self._showplan_row_cache.move_to_end(cache_key)
                while len(self._showplan_row_cache) > SQLSERVER_SHOWPLAN_ROW_CACHE_MAX:
                    self._showplan_row_cache.popitem(last=False)
            return True, merged_diags, plan_text
        except Exception as e:
            err = str(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)], err)

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return SQL Server planner row estimate from cached SHOWPLAN diagnose."""
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            cache_key = finalized.strip()
            if cache_key in self._showplan_row_cache:
                self._showplan_row_cache.move_to_end(cache_key)
                return self._showplan_row_cache[cache_key]
            ok, _, _ = self.explain_diagnose(sql_text, {}, schema=schema, intent=intent)
            if not ok:
                return None
            return self._showplan_row_cache.get(cache_key)
        except Exception:
            return None

    def query_log_source(self) -> Any | None:
        """Return the SQL Server DMV query-log source."""
        return SQLServerQueryLogSource()

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a deterministic ordered row cap when T-SQL ``TABLESAMPLE`` is unseeded."""
        _ = row_count, random_seed, table_kind
        if not use_sample:
            return ""
        return self.profiling_ordered_limit_sample_suffix(sample_size)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Ordered-limit sampling scans a deterministic subquery."""
        _ = table_kind
        return True

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via the active result backend with driver bind maps."""
        backend = self.result_backend
        if backend is None:
            return super().execute(sql, params)
        tm = effective_statement_timeout_ms()
        return backend.fetch_rows(sql, params, timeout_ms=tm)

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append partition-key predicates when schema and intent carry filter signals."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(sql, schema, intent)

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render T-SQL ``DATEDIFF`` comparison against ``GETDATE()``."""
        _ = left_expr
        unit_token = unit.upper()
        if minuend_sql and subtrahend_sql:
            sql = f"DATEDIFF({unit_token}, {subtrahend_sql}, {minuend_sql}) {op} {amount}"
        else:
            sql = f"DATEDIFF({unit_token}, {left_expr}, GETDATE()) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_integer_days(self, base_sql: str, sign: str, offset_sql: str) -> str:
        """Render T-SQL date plus or minus an integer day count."""
        if sign == "+":
            sql = f"DATEADD(day, {offset_sql}, {base_sql})"
        else:
            sql = f"DATEADD(day, -({offset_sql}), {base_sql})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render SQL Server membership without substring JSON search; capability gate excludes contains."""
        _ = value_type
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "unknown"
        if kind == "json_text_array":
            raise ValueError("json containment is not supported for dialect 'sqlserver'")
        norm_param = f"LOWER(LTRIM(RTRIM(CAST(:{param_key} AS NVARCHAR(MAX)))))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS NVARCHAR(MAX)))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render SQL Server ``OPENJSON`` unnest for SELECT list."""
        sql = f"j.value AS {alias} FROM OPENJSON({column_sql}) j"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window_anchor_literal(self, anchor: datetime, unit: str) -> str:
        """Render a bound anchor instant as a T-SQL date/timestamp literal."""
        aware = anchor if anchor.tzinfo is not None else anchor.replace(tzinfo=UTC)
        if Dialect.relative_window_uses_timestamp(unit):
            text = aware.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
            return f"CAST(N'{text}' AS DATETIME2)"
        return f"CAST('{aware.date().isoformat()}' AS DATE)"

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render T-SQL date window boundaries with ``DATEADD``."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            sql = f"{column} {op} DATEADD({unit}, DATEDIFF({unit}, 0, {clock}), 0)"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            _ = plural_unit
            sql = f"{column} {op} DATEADD({unit}, -{scaled}, {clock})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on SQL Server."""
        return f"LOWER({expr})"

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the T-SQL clock expression for relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "GETDATE()"
        return "CAST(GETDATE() AS DATE)"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return T-SQL current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)


class OracleDialect(SqlglotEngineDialect):
    """Oracle dialect using sqlglot read=oracle and SQLAlchemy+oracledb execution."""

    name: str = "oracle"
    sqlglot_dialect: ClassVar[str] = "oracle"
    registry_canonical_rank: ClassVar[int] = 11
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True
    registry_array_contains_excluded: ClassVar[bool] = True

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return Oracle binary collation for reproducible ordering."""
        return "BINARY"

    @property
    def supports_ilike(self) -> bool:
        """Return False because Oracle has no ``ILIKE`` operator."""
        return False

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True because python-oracledb exposes ``Connection.cancel``."""
        return True

    @property
    def logical_engine_name(self) -> str:
        return "Oracle"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Attach Oracle SQLAlchemy backend with optional thick-mode client init."""
        if isinstance(config, OracleRuntimeConfig):
            config.ensure_driver_mode()
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)

    def _ensure_result_backend(self) -> None:
        """Attach an Oracle backend with driver-level ``call_timeout`` support."""
        if self._backend is not None:
            return
        if getattr(self, "engine", None) is not None:
            self._backend = OracleResultBackend(self.engine, dialect_name=self.__class__.__name__)

    def can_explain(self) -> bool:
        """Return True when SQLAlchemy can run ``EXPLAIN PLAN``."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(self, sqlalchemy_engine=self.engine)

    def quote_identifier(self, ident: str) -> str:
        """Quote an Oracle identifier using the dictionary-folded uppercase form."""
        return Dialect.sqlglot_quote_identifier(str(ident).upper(), self.sqlglot_dialect)

    def quote_table_column(self, table: str, column: str) -> str:
        """Emit double-quoted Oracle ``table.column`` with dictionary uppercase fold."""
        return f"{self.quote_identifier(table)}.{self.quote_identifier(column)}"

    def schema_name(self) -> str:
        """Return the Oracle owner/schema in dictionary-folded uppercase."""
        return str(super().schema_name() or "").upper()

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a double-quoted ``schema.table`` reference."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from Oracle dictionary views."""
        schema_name = self.schema_name()
        return SqlglotEngineDialect.structural_constraints_index_for_schema(
            self, schema_name, engine=getattr(self, "engine", None)
        )

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return Oracle ``TO_CHAR`` for overlap sampling."""
        return f"TO_CHAR({expr})"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from Oracle ``ALL_*`` views or SQL file fallback."""
        return load_or_create_schema_oracle(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
            schema_name=self.schema_name(),
        )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return ``EXPLAIN PLAN FOR`` wrapper for Oracle."""
        return f"EXPLAIN PLAN FOR {finalized_sql}"

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[SqlDiagnostic], str]:
        """Parse Oracle plan-table cardinality and cost rows into estimates."""
        _ = schema
        est_rows: float | None = None
        est_bytes: float | None = None
        soft_diags: list[SqlDiagnostic] = []
        if rows:
            try:
                cardinality = rows[0][0]
                if cardinality is not None:
                    est_rows = float(cardinality)
            except (TypeError, ValueError, IndexError):
                pass
        plan_text = "\n".join(" | ".join(str(c) if c is not None else "" for c in row) for row in rows)
        return est_rows, est_bytes, soft_diags, plan_text

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run Oracle ``EXPLAIN PLAN FOR`` then read root plan cardinality."""
        finalized = self.finalize_render(sql, params or {}, schema=schema, intent=intent)
        explain_sql = self.explain_statement_sql(finalized)
        try:
            if self.engine is None:
                return (
                    False,
                    [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message="no SQLAlchemy engine")],
                    "no SQLAlchemy engine",
                )
            with self.engine.begin() as conn:
                conn.execute(text(explain_sql))
                rows = conn.execute(
                    text(
                        "SELECT cardinality, cost FROM plan_table "
                        "WHERE id = 0 ORDER BY timestamp DESC FETCH FIRST 1 ROW ONLY"
                    )
                ).fetchall()
            est_rows, est_bytes, soft_diags, plan_text = self.parse_explain_plan(list(rows), schema=schema)
            failed, why = Dialect.explain_cost_gate_violation(est_rows, est_bytes, dialect=self)
            if failed:
                return (
                    False,
                    soft_diags + [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                    why,
                )
            return True, soft_diags, plan_text
        except Exception as e:
            err = str(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)], err)

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return Oracle planner row estimate from ``EXPLAIN PLAN``."""
        try:
            ok, _, _ = self.explain_diagnose(sql_text, {}, schema=schema, intent=intent)
            if not ok:
                return None
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            if self.engine is None:
                return None
            with self.engine.begin() as conn:
                conn.execute(text(self.explain_statement_sql(finalized)))
                rows = conn.execute(
                    text(
                        "SELECT cardinality FROM plan_table WHERE id = 0 ORDER BY timestamp DESC FETCH FIRST 1 ROW ONLY"
                    )
                ).fetchall()
            est_rows, _, _, _ = self.parse_explain_plan(list(rows), schema=schema)
            return est_rows
        except Exception:
            return None

    def query_log_source(self) -> Any | None:
        """Return the Oracle ``V$SQL`` query-log source."""
        return OracleQueryLogSource()

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return ``None`` because Oracle statement timeouts use python- oracledb ``call_timeout``."""
        _ = timeout_ms
        return None

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a deterministic ordered row cap when Oracle ``SAMPLE`` is unseeded."""
        _ = row_count, random_seed, table_kind
        if not use_sample:
            return ""
        return self.profiling_ordered_limit_sample_suffix(sample_size)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Ordered-limit sampling scans a deterministic subquery."""
        _ = table_kind
        return True

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via the active result backend with driver bind maps."""
        backend = self.result_backend
        if backend is None:
            return super().execute(sql, params)
        tm = effective_statement_timeout_ms()
        return backend.fetch_rows(sql, params, timeout_ms=tm)

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append partition-key predicates when schema and intent carry filter signals."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(sql, schema, intent)

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"LISTAGG({expr_sql}, {sep_sql}) WITHIN GROUP (ORDER BY {order_by_sql})"
        return f"LISTAGG({expr_sql}, {sep_sql})"

    def render_date_trunc(self, unit: str, expr_sql: str) -> str:
        """Render Oracle calendar truncation with ``TRUNC``."""
        unit_norm = (unit or "").strip().lower()
        trunc_fmt = {
            "day": "DD",
            "week": "IW",
            "month": "MM",
            "quarter": "Q",
            "year": "YYYY",
            "hour": "HH24",
            "minute": "MI",
        }
        if unit_norm == "half_year":
            sql = (
                f"CASE WHEN EXTRACT(MONTH FROM {expr_sql}) <= 6 "
                f"THEN TRUNC({expr_sql}, 'YYYY') "
                f"ELSE ADD_MONTHS(TRUNC({expr_sql}, 'YYYY'), 6) END"
            )
        elif unit_norm in trunc_fmt:
            sql = f"TRUNC({expr_sql}, '{trunc_fmt[unit_norm]}')"
        else:
            return super().render_date_trunc(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_extract(self, unit: str, expr_sql: str) -> str:
        """Render Oracle calendar extraction."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "half_year":
            sql = f"CASE WHEN EXTRACT(MONTH FROM {expr_sql}) <= 6 THEN 1 ELSE 2 END"
        elif unit_norm == "week":
            sql = f"TO_NUMBER(TO_CHAR({expr_sql}, 'IW'))"
        elif unit_norm == "quarter":
            sql = f"TO_NUMBER(TO_CHAR({expr_sql}, 'Q'))"
        else:
            return super().render_extract(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render Oracle date difference comparisons."""
        unit_norm = (unit or "").strip().lower()
        later = minuend_sql or "SYSDATE"
        earlier = subtrahend_sql or left_expr
        if unit_norm in {"day", "hour", "minute", "second"}:
            day_diff = f"({later} - {earlier})"
            scale = {"day": 1.0, "hour": 24.0, "minute": 1440.0, "second": 86400.0}[unit_norm]
            sql = f"({day_diff} * {scale}) {op} {amount}"
        elif unit_norm in {"month", "quarter", "year", "half_year"}:
            months = f"MONTHS_BETWEEN({later}, {earlier})"
            divisor = {"month": 1.0, "quarter": 3.0, "half_year": 6.0, "year": 12.0}[unit_norm]
            sql = f"({months} / {divisor}) {op} {amount}"
        elif unit_norm == "week":
            sql = f"(({later} - {earlier}) / 7) {op} {amount}"
        else:
            sql = f"({later} - {earlier}) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_integer_days(self, base_sql: str, sign: str, offset_sql: str) -> str:
        """Render Oracle date plus or minus an integer day count."""
        if sign == "+":
            sql = f"({base_sql} + {offset_sql})"
        else:
            sql = f"({base_sql} - {offset_sql})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render Oracle membership without substring JSON search; capability gate excludes contains."""
        _ = value_type
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "unknown"
        if kind == "json_text_array":
            raise ValueError("json containment is not supported for dialect 'oracle'")
        norm_param = f"LOWER(TRIM(BOTH FROM CAST(:{param_key} AS VARCHAR2(4000))))"
        sql = f"{norm_param} = LOWER(TO_CHAR({column_sql}))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Oracle ``JSON_TABLE`` unnest for SELECT list."""
        sql = f"jt.{alias} FROM JSON_TABLE({column_sql}, '$[*]' COLUMNS({alias} VARCHAR2(4000) PATH '$')) jt"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window_anchor_literal(self, anchor: datetime, unit: str) -> str:
        """Render a bound anchor instant as an Oracle date/timestamp literal."""
        aware = anchor if anchor.tzinfo is not None else anchor.replace(tzinfo=UTC)
        if Dialect.relative_window_uses_timestamp(unit):
            text = aware.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
            return f"TO_TIMESTAMP('{text}', 'YYYY-MM-DD HH24:MI:SS')"
        return f"DATE '{aware.date().isoformat()}'"

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render Oracle date window boundaries with ``ADD_MONTHS`` / day arithmetic."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        unit_norm = (unit or "").strip().lower()
        if amount == 0:
            sql = f"{column} {op} {self.render_date_trunc(unit_norm, clock)}"
        elif unit_norm in {"month", "quarter", "half_year", "year"}:
            months = amount * {"month": 1, "quarter": 3, "half_year": 6, "year": 12}[unit_norm]
            sql = f"{column} {op} ADD_MONTHS({clock}, -{months})"
        elif unit_norm == "week":
            sql = f"{column} {op} ({clock} - {amount * 7})"
        else:
            scaled, _plural = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} ({clock} - {scaled})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Oracle."""
        return f"LOWER({expr})"

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the Oracle clock expression for relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "SYSTIMESTAMP"
        return "TRUNC(SYSDATE)"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return Oracle current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)


class BigQueryDialect(SqlglotEngineDialect):
    """BigQuery dialect using sqlglot read=bigquery and google-cloud- bigquery execution."""

    name: str = "bigquery"
    sqlglot_dialect: ClassVar[str] = "bigquery"
    registry_canonical_rank: ClassVar[int] = 10
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @staticmethod
    def _bq_partition_columns(table_meta: Any) -> list[str]:
        """Return BigQuery partition column names for predicate injection."""
        return list(getattr(table_meta, "partition_columns", []) or [])

    @property
    def integer_division_truncates(self) -> bool:
        """Return False because BigQuery ``/`` returns floating-point results."""
        return False

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return False because BigQuery jobs cannot be cancelled from this execution path."""
        return False

    @property
    def logical_engine_name(self) -> str:
        return "BigQuery"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Build BigQuery client, optional storage reader, and SQLAlchemy inspect engine."""
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        bq_config = cast(BigQueryRuntimeConfig, config)
        self._bq_client: Any | None = None
        self._bq_storage_client: Any | None = None
        try:
            from google.cloud import bigquery

            credentials: Any | None = None
            if bq_config.has_service_account() and bq_config.CREDENTIALS_PATH:
                from google.oauth2 import service_account

                from_service_account: Any = service_account.Credentials.from_service_account_file
                credentials = from_service_account(str(bq_config.CREDENTIALS_PATH))
            self._bq_client = bigquery.Client(
                project=str(bq_config.PROJECT or ""), credentials=credentials, location=str(bq_config.LOCATION or "US")
            )
            try:
                bq_storage = importlib.import_module("google.cloud.bigquery_storage")
                read_client_ctor: Any = bq_storage.BigQueryReadClient
                self._bq_storage_client = read_client_ctor(credentials=credentials)
            except ImportError:
                self._bq_storage_client = None
        except Exception as exc:
            debug(f"[BigQueryDialect.__init__] BigQuery client unavailable: {exc!r}")
        self._select_result_backend()

    def _effective_max_query_cost_bytes(self) -> float | int | None:
        """Return the active bytes cap: dialect member override, then policy default."""
        override = getattr(self, "max_query_cost_bytes", None)
        if override is not None:
            return cast(float | int, override)
        return PolicyConfig.MAX_QUERY_COST_BYTES

    def _bq_job_limits(self) -> tuple[int | None, int | None]:
        """Return optional maximum bytes billed and job timeout for BigQuery jobs."""
        cap = self._effective_max_query_cost_bytes()
        max_bytes = int(cap) if cost_cap_active(cap) and cap is not None else None
        timeout_ms = effective_statement_timeout_ms()
        return max_bytes, timeout_ms

    def _select_result_backend(self) -> None:
        """Attach the preferred BigQuery row-fetch backend."""
        max_bytes, timeout_ms = self._bq_job_limits()
        if getattr(self, "_bq_storage_client", None) is not None and self._bq_client is not None:
            self._backend = BigQueryStorageBackend(
                self._bq_client,
                self._bq_storage_client,
                maximum_bytes_billed=max_bytes,
                job_timeout_ms=timeout_ms,
                dialect_name=self.__class__.__name__,
            )
            return
        if self._bq_client is not None:
            self._backend = BigQueryClientBackend(
                self._bq_client,
                maximum_bytes_billed=max_bytes,
                job_timeout_ms=timeout_ms,
                dialect_name=self.__class__.__name__,
            )
            return
        if getattr(self, "engine", None) is not None:
            self._backend = SqlAlchemyResultBackend(self.engine, dialect_name=self.__class__.__name__)

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active BigQuery row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return ResultReaderKind.SQLALCHEMY

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a backtick-quoted ``project.dataset.table`` reference."""
        _ = kind
        config = getattr(self, "config", None)
        project = str(getattr(config, "PROJECT", None) or "") if config is not None else ""
        dataset = (
            str(getattr(config, "DATASET", None) or getattr(config, "SCHEMA", None) or "") if config is not None else ""
        )
        if project and dataset:
            return f"{self.quote_identifier(project)}.{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        if dataset:
            return f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def _qualify_uses_backtick_identifiers(self) -> bool:
        """Return True because BigQuery qualification uses backticks."""
        return True

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """BigQuery does not expose live FK metadata; return an empty index. Operators must supply join edges via ``EngineContext.sql_file`` DDL and/or ``foreign_keys_add`` overrides; profiling-time suffix/composite/semantic inference may still add edges when PK anchors exist."""
        return CatalogStructuralConstraintsIndex.empty()

    def can_explain(self) -> bool:
        """Return True when the BigQuery client or SQLAlchemy engine can dry-run."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=getattr(self, "_bq_client", None)
        )

    def apply_execute_cost_limits(self, target: Any) -> None:
        """Apply ``maximum_bytes_billed`` to a BigQuery query job config when configured."""
        cap = self._effective_max_query_cost_bytes()
        if not cost_cap_active(cap) or cap is None:
            return
        if hasattr(target, "maximum_bytes_billed"):
            target.maximum_bytes_billed = int(cap)

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return BigQuery ``CAST(… AS STRING)`` for overlap sampling."""
        return f"CAST({expr} AS STRING)"

    def pre_execute_rewrite(self, sql: str) -> str:
        """Rewrite ``:name`` placeholders to BigQuery ``@name`` form."""
        if not sql or not sql.strip():
            return sql
        try:
            tree = sqlglot.parse_one(sql, dialect=self.sqlglot_dialect)
        except Exception:
            return sql
        for node in tree.find_all(sqlglot.exp.Placeholder):
            key = node.name or node.this
            if isinstance(key, str) and key:
                node.replace(sqlglot.exp.Parameter(this=key))
        try:
            return tree.sql(dialect=self.sqlglot_dialect)
        except Exception:
            return sql

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from BigQuery INFORMATION_SCHEMA or SQL file fallback."""
        return load_or_create_schema_bigquery(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
        )

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Validate SQL via BigQuery dry-run query job and enforce bytes-billed caps."""
        finalized = self.finalize_render(sql, params or {}, schema=schema, intent=intent)
        if self._bq_client is None:
            return True, [], ""
        try:
            import google.cloud.bigquery

            max_bytes, timeout_ms = self._bq_job_limits()
            job_config = google.cloud.bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            if max_bytes is not None:
                job_config.maximum_bytes_billed = int(max_bytes)
            if timeout_ms is not None:
                job_config.job_timeout_ms = int(timeout_ms)
            job = self._bq_client.query(finalized, job_config=job_config)
            est_bytes = float(job.total_bytes_processed or 0)
            failed, why = Dialect.explain_cost_gate_violation(None, est_bytes, dialect=self)
            require_part: list[str] = []
            partition_filter_present = True
            if schema is not None:
                for tname in intent.tables if intent is not None and intent.tables else []:
                    meta = schema.tables.get(tname)
                    if meta is not None and getattr(meta, "require_partition_filter", False):
                        require_part.append(tname)
                        part_cols = list(getattr(meta, "partition_columns", []) or [])
                        if part_cols:
                            part_col = part_cols[0].lower()
                            if part_col not in finalized.lower():
                                partition_filter_present = False
            soft_diags = SqlglotEngineDialect.bigquery_diagnostics_from_dry_run(
                est_bytes,
                partition_filter_present=partition_filter_present,
                require_partition_filter_tables=require_part,
            )
            if failed:
                return (
                    False,
                    soft_diags + [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                    why,
                )
            return True, soft_diags, ""
        except Exception as e:
            err = str(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)], err)

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """BigQuery dry-run does not expose row estimates; always return ``None``."""
        _ = sql_text, schema, intent
        return None

    def query_log_source(self) -> Any | None:
        """Return the BigQuery ``INFORMATION_SCHEMA.JOBS`` query-log source."""
        return BigQueryQueryLogSource()

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return SHA-256 over information_schema columns only (no UNIQUE probe on BigQuery)."""
        _ = schema_context
        try:
            schema_name = str(getattr(self.config, "DATASET", None) or getattr(self.config, "SCHEMA", None) or "")
            if not schema_name:
                return ""
            backend = getattr(self, "_backend", None)
            if backend is not None:
                esc = schema_name.replace("'", "''")
                probe_sql = INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL.replace(":s", f"'{esc}'")
                rows = backend.fetch_rows(probe_sql)
                payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
                return sha256(payload_cols)
            if self.engine is None:
                return ""
            with self.engine.connect() as conn:
                rows = conn.execute(text(INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL), {"s": schema_name}).fetchall()
            payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
            return sha256(payload_cols)
        except Exception as exc:
            debug(f"[{self.__class__.__name__}.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via the active BigQuery result backend."""
        backend = getattr(self, "_backend", None)
        if backend is None:
            return super().execute(sql, params)
        bind = BigQueryClientBackend._bq_bind_params_from_sql(sql, params)
        tm = effective_statement_timeout_ms()
        return cast(list[tuple[Any, ...]], backend.fetch_rows(sql, bind, timeout_ms=tm))

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append partition guard predicates using shared partition helpers."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        pruned = adapter.inject_partition_predicates(
            sql, schema, intent, column_selector=BigQueryDialect._bq_partition_columns
        )

        def _default_guard(table_name: str, part_col: str) -> str:
            qual = self.quote_table_column(table_name, part_col)
            return f"{qual} >= DATE_SUB(CURRENT_DATE(), INTERVAL {BQ_DEFAULT_PARTITION_LOOKBACK_DAYS} DAY)"

        out = PartitionSqlAdapter.append_required_partition_filter_guard(
            pruned,
            schema=schema,
            intent=intent,
            sqlglot_dialect=self.sqlglot_dialect,
            column_selector=BigQueryDialect._bq_partition_columns,
            default_predicate_sql=_default_guard,
            intent_equality_for_column=self._partition_predicate_from_intent,
        )
        Dialect.trace_finalize_render_stage("inject_pruning_predicates", sql, out)
        return out

    def _partition_predicate_from_intent(self, intent: RuntimeIntent, table_name: str, part_col: str) -> str | None:
        for fp in PredicateGroup.where_leaves(intent.where) or []:
            term = (fp.left_expr.primary_term or "").strip() if fp.left_expr else ""
            if not term:
                continue
            col_part = term.rsplit(".", 1)[-1].strip().lower()
            if col_part != part_col.lower():
                continue
            if fp.raw_value is not None:
                qual = self.quote_table_column(table_name, part_col)
                return f"{qual} = {self.quote_string_literal(str(fp.raw_value))}"
        return None

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render BigQuery date comparison using ``DATE_DIFF``. BigQuery ``DATE_DIFF(end, start, part)`` returns an integer, so a two-column difference passes the columns as separate ``end``/``start`` arguments (``date - date`` is not valid in BigQuery). Falls back to comparing a single date column against ``CURRENT_DATE()``."""
        unit_token = unit.upper()
        if minuend_sql and subtrahend_sql:
            sql = f"DATE_DIFF({minuend_sql}, {subtrahend_sql}, {unit_token}) {op} {amount}"
        else:
            sql = f"DATE_DIFF(CURRENT_DATE(), {left_expr}, {unit_token}) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render BigQuery array membership with native containment when available."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect=self.sqlglot_dialect,
                param_prefix="@",
            )
        needle = f"CONCAT('\"', LOWER(@{param_key}), '\"')"
        sql = f"STRPOS(LOWER(TO_JSON_STRING({column_sql})), {needle}) > 0"
        return sql

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render BigQuery ``UNNEST`` for SELECT list."""
        sql = f"UNNEST({column_sql}) AS {alias}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def _bigquery_date_window_column_sql(self, column: str, unit: str) -> str:
        """Return the column expression for BigQuery date-window comparisons."""
        if Dialect.relative_window_uses_timestamp(unit):
            return column
        return f"DATE({column})"

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render BigQuery date window boundaries with matching date/timestamp clocks."""
        col = self._bigquery_date_window_column_sql(column, unit)
        unit_norm = (unit or "").strip().lower()
        clock = self.date_window_clock_sql(unit_norm, anchor=anchor)
        if amount == 0:
            sql = f"{col} {op} DATE_TRUNC({clock}, {unit_norm.upper()})"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            _ = plural_unit
            if Dialect.relative_window_uses_timestamp(unit_norm):
                sql = f"{col} {op} TIMESTAMP_SUB({clock}, INTERVAL {scaled} {unit_norm.upper()})"
            else:
                sql = f"{col} {op} DATE_SUB({clock}, INTERVAL {scaled} {unit_norm.upper()})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_trunc(self, unit: str, expr_sql: str) -> str:
        """Render BigQuery calendar truncation with ISO week semantics."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "week":
            sql = f"DATE_TRUNC({expr_sql}, WEEK(MONDAY))"
        elif unit_norm == "quarter":
            sql = f"DATE_TRUNC({expr_sql}, QUARTER)"
        elif unit_norm == "half_year":
            sql = (
                f"IF(EXTRACT(MONTH FROM {expr_sql}) <= 6, "
                f"DATE_TRUNC({expr_sql}, YEAR), "
                f"DATE(EXTRACT(YEAR FROM {expr_sql}), 7, 1))"
            )
        else:
            sql = f"DATE_TRUNC({expr_sql}, {unit_norm.upper()})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_extract(self, unit: str, expr_sql: str) -> str:
        """Render BigQuery calendar extraction with native quarter and half-year units."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "half_year":
            sql = f"IF(EXTRACT(MONTH FROM {expr_sql}) <= 6, 1, 2)"
        elif unit_norm == "quarter":
            sql = f"EXTRACT(QUARTER FROM {expr_sql})"
        elif unit_norm == "week":
            sql = f"EXTRACT(ISOWEEK FROM {expr_sql})"
        else:
            return super().render_extract(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window_inclusive_upper(
        self, left_rendered: str, unit: str, *, anchor: datetime | None = None
    ) -> str:
        """Render BigQuery inclusive upper bound with the same column normalization as the lower bound."""
        col = self._bigquery_date_window_column_sql(left_rendered, unit)
        anchor_sql = self.date_window_upper_bound_sql(unit, anchor=anchor)
        return f"{col} <= {anchor_sql}"

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on BigQuery."""
        return f"LOWER({expr})"

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the BigQuery clock expression for relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "CURRENT_TIMESTAMP()"
        return "CURRENT_DATE()"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return BigQuery current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a deterministic ordered row cap when BigQuery ``TABLESAMPLE`` is unseeded."""
        _ = row_count, random_seed, table_kind
        if not use_sample:
            return ""
        return self.profiling_ordered_limit_sample_suffix(sample_size)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Ordered-limit sampling scans a deterministic subquery."""
        _ = table_kind
        return True


class DatabricksDialect(SqlglotParseMixin, Dialect):
    """Databricks / Spark SQL dialect using EXPLAIN and optional native SQL connector."""

    name: str = "databricks"
    sqlglot_dialect: ClassVar[str] = "databricks"
    registry_canonical_rank: ClassVar[int] = 8
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @staticmethod
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

    @staticmethod
    def databricks_normalize_datetrunc_sql(sql: str) -> str:
        """Rewrite parsed ``Anonymous`` ``DATETRUNC`` call sites and ``DATEADD`` tokens for Spark emission."""
        out = SqlglotParseMixin.normalize_datetrunc_sql(sql, sqlglot_dialect="databricks")
        out = re.sub(r"\bDATEADD\s*\(", "date_columndd(", out, flags=re.IGNORECASE)
        return out

    @staticmethod
    def _spark_calendar_months_threshold(unit: str, amount: int) -> int:
        """Return a month count for Spark ``MONTHS_BETWEEN`` calendar comparisons."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "month":
            return amount
        if unit_norm == "quarter":
            return amount * 3
        if unit_norm == "half_year":
            return amount * 6
        if unit_norm == "year":
            return amount * 12
        raise ValueError(f"unsupported calendar unit for months threshold: {unit!r}")

    def finalize_like_predicate_sql(self, sql: str) -> str:
        """Round-trip LIKE/ILIKE predicates so sqlglot-backed parsers accept ESCAPE clauses."""
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    @property
    def supports_ilike(self) -> bool:
        """Return True because Spark SQL exposes ``ILIKE``."""
        return True

    @property
    def integer_division_truncates(self) -> bool:
        """Return True because Spark integer division truncates."""
        return True

    @property
    def supports_ordered_string_agg(self) -> bool:
        return False

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        return f"array_join(collect_list({expr_sql}), {sep_sql})"

    def render_median(self, expr_sql: str) -> str:
        return f"median({expr_sql})"

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return Spark ``CAST(… AS STRING)`` for overlap sampling."""
        return f"CAST({expr} AS STRING)"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Open a native Databricks SQL connection or fall back to a PySpark session. When warehouse credentials are configured (``server_hostname``, ``http_path``, ``access_token``), the ``databricks-sql- connector`` is preferred.  If the connector import or connection attempt fails, the dialect falls back to a cluster-local ``SparkSession``.  A ``RuntimeError`` is raised only when **neither** backend can be established."""
        require_driver("databricks")
        super().__init__(config)

        self.connection: Any | None = None
        self.spark: Any | None = None
        self.engine: Any | None = None
        self._backend: ResultBackend | None = None
        db_config = cast(DatabricksRuntimeConfig, config)

        if sqlalchemy_engine is not None:
            self.engine = sqlalchemy_engine
            debug("[DatabricksDialect.__init__] using caller-provided SQLAlchemy engine")
            self._select_result_backend()
            SqlalchemyExecutionMixin.assert_one_live_handle(self)
            return

        connector_error: str | None = None
        last_connector_exc: BaseException | None = None

        if db_config.has_native_connection():
            try:
                import databricks.sql

                progress("  Connecting to Databricks SQL warehouse (cold start can take several minutes)...")
                connect_started = time.monotonic()
                self.connection = databricks.sql.connect(
                    server_hostname=db_config.SERVER_HOSTNAME,
                    http_path=db_config.HTTP_PATH,
                    access_token=db_config.ACCESS_TOKEN,
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
                        raise DatabasePingFailed("Databricks warehouse warmup probe failed after connect.") from exc
                    raise
                progress(f"  Warehouse ready in {time.monotonic() - connect_started:.1f}s.")
                debug("[DatabricksDialect.__init__] using databricks-sql-connector (warehouse)")
                self._select_result_backend()
                SqlalchemyExecutionMixin.assert_one_live_handle(self)
                return

            msg = (
                "databricks-sql-connector failed to open a warehouse session "
                f"({connector_error}). Verify the warehouse is reachable and the "
                "access token is valid; warehouses can take several minutes to cold-start."
            )
            if last_connector_exc is not None and engine_connect_likely_transient(last_connector_exc):
                raise DatabasePingFailed(msg) from last_connector_exc
            debug(f"[DatabricksDialect.__init__] {msg}")

        if self.connection is None:
            url = db_config.sqlalchemy_url()
            if url and self.engine is None:
                try:
                    self.engine = create_engine(url, future=True)
                except Exception as exc:
                    debug(f"[DatabricksDialect.__init__] SQLAlchemy engine not created: {exc}")
            if self.engine is not None:
                self._select_result_backend()
                SqlalchemyExecutionMixin.assert_one_live_handle(self)
                return
            self._init_spark_fallback(connector_error)
            self._select_result_backend()
        SqlalchemyExecutionMixin.assert_one_live_handle(self)

    def _init_spark_fallback(self, connector_error: str | None) -> None:
        """Attempt to initialise a Spark session as the execution backend. Tries ``databricks.connect.DatabricksSession`` first to honour the installed ``databricks-connect`` build of ``pyspark``, which hard-rejects ``SparkSession.builder.getOrCreate()``. Falls back to ``pyspark.sql.SparkSession`` only when ``databricks.connect`` is not importable. Raises :class:`ConfigError` with the canonical missing-credential hint when neither path yields a session."""
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

    def _select_result_backend(self) -> None:
        """Attach the active Databricks row-fetch backend from connector, engine, or Spark."""
        if self.connection is not None:
            self._backend = DatabricksConnectorBackend(self.connection)
            return
        if self.engine is not None:
            self._backend = DatabricksSqlAlchemyBackend(self.engine, dialect_name=self.__class__.__name__)
            return
        if self.spark is not None:
            self._backend = DatabricksSparkBackend(self.spark)

    def _databricks_fetch_dict_rows(self, sql: str) -> list[dict[str, Any]]:
        """Execute an information_schema query and return normalized row dicts."""
        backend = self._backend
        if isinstance(backend, DatabricksConnectorBackend) and self.connection is not None:
            with self.connection.cursor() as cur:
                return SqlglotEngineDialect.information_schema_connector_fetchall_dict_rows(cur, sql)
        if isinstance(backend, DatabricksSparkBackend) and self.spark is not None:
            return SqlglotEngineDialect.information_schema_spark_collect_normalized_dicts(self.spark, sql)
        if isinstance(backend, DatabricksSqlAlchemyBackend) and self.engine is not None:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                if not result.cursor.description:
                    return []
                col_names = [d[0] for d in result.cursor.description]
                return [
                    SqlglotEngineDialect.information_schema_normalize_row(dict(zip(col_names, row, strict=True)))
                    for row in (result.fetchall() or [])
                ]
        return []

    @property
    def result_backend(self) -> ResultBackend | None:
        """Return the active row-fetch backend for this dialect instance."""
        return self._backend

    def _collect_rows(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via the active backend and return row tuples."""
        backend = self._backend
        if backend is None:
            return []
        tm = effective_statement_timeout_ms()
        return backend.fetch_rows(sql, params, timeout_ms=tm)

    def _collect_explain_text(self, explain_sql: str) -> str:
        """Run an EXPLAIN statement and return newline-joined first- column text."""
        return self._backend.fetch_first_column_text(explain_sql) if self._backend is not None else ""

    def _explain_result_from_text(self, text_payload: str) -> tuple[bool, list[SqlDiagnostic], str]:
        """Parse EXPLAIN text into cost-gate outcome and soft diagnostics."""
        er, eb = SqlglotEngineDialect.databricks_plan_stats_from_explain_text(text_payload)
        failed, why = Dialect.explain_cost_gate_violation(er, eb, dialect=self)
        if failed:
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)], why)
        return True, SqlglotEngineDialect.databricks_diagnostics_from_explain_text(text_payload), ""

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run Spark/Databricks ``EXPLAIN`` and return ``(ok, diagnostics, raw_message)``. ``ok`` is False only on hard validation failures. A permission- denied error disables EXPLAIN for the remainder of this dialect instance and is reported as ``ok=True`` with no diagnostics so the caller can proceed without treating missing privileges as invalid SQL. Soft plan-shape findings (suspected cartesian joins, zero-row estimates) are emitted as :class:`SqlDiagnostic` entries with codes from ``SOFT_DIAGNOSTIC_CODES`` in ``_config`` so callers may apply confidence penalties without rejecting the SQL."""
        finalized = self.finalize_render(sql, params or {}, schema=schema, intent=intent)
        explain_sql = f"EXPLAIN COST {finalized}"
        if self._backend is None:
            return True, [], ""
        try:
            text_payload = self._collect_explain_text(explain_sql)
            return self._explain_result_from_text(text_payload)
        except Exception as e:
            err = str(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)], err)

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a Unity Catalog three-part backtick-quoted table reference."""
        _ = kind
        catalog = self.catalog_name()
        schema = self.schema_name()
        if catalog:
            return f"{self.quote_identifier(catalog)}.{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"

    def _qualify_uses_backtick_identifiers(self) -> bool:
        """Return True because Spark SQL table qualification uses backticks."""
        return True

    def pre_execute_rewrite(self, sql: str) -> str:
        """Apply Databricks-specific SQL rewrites before parameter substitution."""
        return super().pre_execute_rewrite(sql)

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Rewrite parsed ``DATETRUNC`` call sites to Spark ``DATE_TRUNC`` ordering."""
        if stage != "post_substitute":
            return sql
        return DatabricksDialect.databricks_normalize_datetrunc_sql(sql)

    def profile_schema_dispatch(self, sg: SchemaGraph) -> None:
        """Profile tables using the active Databricks native backend chain."""
        super().profile_schema_dispatch(sg)

    def apply_execute_cost_limits(self, target: Any) -> None:
        """Apply Databricks execute-time cost caps when the driver supports them."""
        super().apply_execute_cost_limits(target)

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
        """Produce executable SQL through the shared Databricks render pipeline."""
        return super().finalize_render(
            sql_param,
            params,
            schema=schema,
            intent=intent,
            execution_sql_override=execution_sql_override,
            structural_defaults=structural_defaults,
        )

    def can_explain(self) -> bool:
        """Return True when SQLAlchemy, the native connector, or Spark can run EXPLAIN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=self.connection, spark_session=self.spark
        )

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append Delta partition predicates when schema and intent are available."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=DatabricksDialect._dbr_format_partition_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(sql, schema, intent)

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active Databricks row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        if getattr(self, "engine", None) is not None:
            return ResultReaderKind.SQLALCHEMY
        if getattr(self, "connection", None) is not None:
            return ResultReaderKind.CONNECTOR
        return ResultReaderKind.SPARK

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return Spark current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def normalize_window_agg_sql_frag(self, frag: str) -> str:
        """Unqualify aggregate arguments inside window functions for Spark SQL."""
        return databricks_unqualify_agg_arg_sql(frag)

    def plan_rows_from_explain_text(self, payload: str) -> float | None:
        """Extract a coarse row-count estimate from Spark/Databricks ``EXPLAIN COST`` text."""
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

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return Databricks planner row estimate from ``EXPLAIN COST``."""
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            explain_sql = f"EXPLAIN COST {finalized}"
            text_payload = self._collect_explain_text(explain_sql)
            return self.plan_rows_from_explain_text(text_payload)
        except Exception:
            return None

    def query_log_source(self) -> Any | None:
        """Return the Databricks ``system.query.history`` query-log source."""
        return DatabricksQueryLogSource()

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute finalized Spark SQL via the warehouse connector or Spark session."""
        _ = params
        if diagnostic_debug_enabled():
            debug(f"[DatabricksDialect.execute] sql=\n{sql}")

        try:
            if self._backend is None:
                raise RuntimeError("DatabricksDialect has no result backend")
            return self._collect_rows(sql, params)
        except Exception as e:
            err = str(e)
            if Dialect.is_permission_denied_error(err):
                raise AccessError("execute", err, reason="warehouse") from e
            el = err.lower()
            if "timeout" in el and ("statement" in el or "cancel" in el or "deadline" in el):
                raise StatementTimeoutError(err) from e
            raise

    def table_kinds_map(self) -> dict[str, str]:
        """Return lowercased Unity relation name to ``information_schema.tables.table_type`` string."""
        catalog = self.catalog_name() or ""
        schema_name = self.schema_name()
        types: dict[str, str] = {}
        if not catalog or not schema_name:
            return types
        esc_cat = catalog.replace("`", "``")
        lit = str(schema_name).replace("'", "''")
        q = UNITY_INFORMATION_SCHEMA_TABLES_TABLE_TYPE_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        try:
            for row in self._databricks_fetch_dict_rows(q):
                key_t = row.get("t") or row.get("table_name")
                if key_t is None:
                    continue
                types[str(key_t).lower()] = str(row.get("table_type") or row.get("TABLE_TYPE") or "")
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.table_kinds_map] failed: {exc!r}")
        return types

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and single-column UNIQUE metadata from Unity ``information_schema``."""
        catalog = self.catalog_name() or ""
        schema_name = self.schema_name()
        if not catalog or not schema_name:
            return CatalogStructuralConstraintsIndex.empty()
        esc_cat = catalog.replace("`", "``")
        lit = str(schema_name).replace("'", "''")
        tc_sql = UNITY_INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        kcu_sql = UNITY_INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        rc_sql = UNITY_INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        cols_sql = UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL.format(catalog_esc=esc_cat, schema_lit=lit)
        try:
            t_rows = self._databricks_fetch_dict_rows(tc_sql)
            k_rows = self._databricks_fetch_dict_rows(kcu_sql)
            r_rows = self._databricks_fetch_dict_rows(rc_sql)
            c_rows = self._databricks_fetch_dict_rows(cols_sql)
            idx = SqlglotEngineDialect.structural_constraints_index_from_information_schema_rows(t_rows, k_rows, r_rows)
            idx.column_nullability = SqlglotEngineDialect.column_nullability_from_information_schema_rows(c_rows)
            return idx
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.structural_constraints_index] failed: {exc!r}")
            return CatalogStructuralConstraintsIndex.empty()

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a graph from Unity Catalog or DDL fallback."""
        return load_or_create_schema_databricks(
            spark_session=self.spark,
            connection=self.connection,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            table_kinds_map=self.table_kinds_map(),
            structural_constraints_index=self.structural_constraints_index(),
            sql_file=sql_file,
        )

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return SHA-256 over ``information_schema.columns`` rows for the configured catalog and schema on Databricks. Tries the SQL connector first, then falls back to a Spark session. Always returns ``""`` rather than raising so build_schema_graph degrades to fingerprint validation when the probe cannot run."""
        _ = schema_context
        try:
            db_config = cast(DatabricksRuntimeConfig, self.config)
            catalog = str(db_config.CATALOG or "")
            schema_name = str(db_config.SCHEMA or "")
            if not catalog or not schema_name:
                return ""
            esc_cat = catalog.replace("`", "``")
            esc_sch = schema_name.replace("'", "''")
            cols_sql = UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL.format(catalog_esc=esc_cat, schema_lit=esc_sch)
            unique_sql = UNITY_INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL.format(
                catalog_esc=esc_cat, schema_lit=esc_sch
            )
            if self._backend is None:
                return ""
            rows = self._collect_rows(cols_sql)
            uniq_rows: list[tuple[Any, ...]] = []
            try:
                uniq_rows = self._collect_rows(unique_sql)
            except Exception as uexc:
                debug(f"[dialect.DatabricksDialect.compute_ddl_probe] unique probe failed: {uexc!r}")
            payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
            payload_uniq = "\n".join("|".join("" if c is None else str(c) for c in r) for r in uniq_rows)
            return sha256(payload_cols + "\n##UNIQUE##\n" + payload_uniq)
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""

    def profile_schema(self, sg: SchemaGraph) -> None:
        """Profile Databricks tables via the native connector, Spark, or SQLAlchemy fallback."""
        self.profile_schema_dispatch(sg)

    def refresh_full_table_distinct_for_pk_inference(
        self, table_name: str, col_name: str, *, table_kind: TableKind = TableKind.TABLE
    ) -> tuple[int, int, float] | None:
        """Run full-table statistics for PK inference after sampled profiling."""
        try:
            full_table = self.qualified_table_ref(table_name, kind=table_kind)
            q_col = self.quote_identifier(col_name)
            sql = (
                f"SELECT COUNT(*) AS cnt, COUNT(DISTINCT {q_col}) AS dist, "
                f"COUNT(*) - COUNT({q_col}) AS nulls FROM {full_table}"
            )
            if self._backend is not None:
                rows = self._collect_rows(sql)
                if not rows:
                    return None
                row = rows[0]
                cnt = int(row[0] or 0)
                dist = int(row[1] or 0)
                nulls = int(row[2] or 0)
                nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
                return (dist, cnt, nr)
            return None
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.refresh_full_table_distinct_for_pk_inference] failed: {exc!r}")
            return None

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render Spark calendar-aware date-difference comparisons."""
        unit_norm = (unit or "").strip().lower()
        if minuend_sql and subtrahend_sql:
            if unit_norm == "day":
                sql = f"DATEDIFF({minuend_sql}, {subtrahend_sql}) {op} {amount}"
            elif unit_norm == "week":
                sql = f"WEEKS_BETWEEN({minuend_sql}, {subtrahend_sql}) {op} {amount}"
            elif unit_norm in ("month", "quarter", "half_year", "year"):
                months = DatabricksDialect._spark_calendar_months_threshold(unit, amount)
                sql = f"MONTHS_BETWEEN({minuend_sql}, {subtrahend_sql}) {op} {months}"
            else:
                scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
                sql = f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"
        elif unit_norm == "day":
            sql = f"({left_expr}) {op} {amount}"
        elif unit_norm == "week":
            sql = f"({left_expr}) {op} {amount * 7}"
        elif unit_norm in ("month", "quarter", "half_year", "year"):
            sql = f"({left_expr}) {op} {DatabricksDialect._spark_calendar_months_threshold(unit, amount)}"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            sql = f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_containment(self, column_sql: str, value_param: str, value_type: str) -> str | None:
        """Render Spark JSON-array containment with ``array_contains``."""
        _ = value_type
        return f"array_contains(from_json({column_sql}, 'array<string>'), CAST({value_param} AS STRING))"

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render Databricks array membership with trimmed element comparison."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect=self.sqlglot_dialect,
            )
        trim_set = "CONCAT(' ', chr(34), chr(39))"
        norm_bind = f"LOWER(TRIM(CAST(:{param_key} AS STRING), {trim_set}))"
        xform = f"TRANSFORM({column_sql}, _ac_x -> LOWER(TRIM(CAST(_ac_x AS STRING), {trim_set})))"
        sql = f"({column_sql} IS NOT NULL AND ARRAY_CONTAINS({xform}, {norm_bind}))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Spark ``EXPLODE`` for SELECT list."""
        sql = f"EXPLODE({column_sql}) AS {alias}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    @property
    def supports_unnest_select_item(self) -> bool:
        """Return True because Spark ``EXPLODE`` is valid as a SELECT- list generator."""
        return True

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render Spark date window boundaries."""
        unit_norm = (unit or "").strip().lower()
        clock = self.date_window_clock_sql(unit_norm, anchor=anchor)
        if amount == 0:
            if unit_norm == "week":
                sql = f"{column} {op} DATE_TRUNC('WEEK', {clock})"
            else:
                sql = f"{column} {op} DATE_TRUNC('{unit_norm}', {clock})"
        elif unit_norm == "week":
            scaled, _ = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} DATE_TRUNC('WEEK', {clock} - INTERVAL '{scaled} weeks')"
        elif unit_norm == "day":
            sql = f"{column} {op} DATE_ADD({clock}, {amount} * -1)"
        elif unit_norm == "month":
            sql = f"{column} {op} ADD_MONTHS({clock}, -{amount})"
        elif unit_norm == "quarter":
            sql = f"{column} {op} ADD_MONTHS({clock}, -{amount * 3})"
        elif unit_norm == "half_year":
            sql = f"{column} {op} ADD_MONTHS({clock}, -{amount * 6})"
        elif unit_norm == "year":
            sql = f"{column} {op} ADD_MONTHS({clock}, -{amount * 12})"
        elif Dialect.relative_window_uses_timestamp(unit_norm):
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} ({clock} - INTERVAL '{scaled} {plural_unit}')"
        else:
            sql = f"{column} {op} DATE_ADD({clock}, {amount} * -1)"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the Spark clock expression for relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "current_timestamp()"
        return "current_date()"

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Spark."""
        return f"LOWER({expr})"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a Spark ``TABLESAMPLE … REPEATABLE`` suffix for statistics."""
        _ = table_kind
        if not use_sample:
            return ""
        pct = max(0.01, min(100.0, 100.0 * sample_size / max(row_count, 1)))
        return f"TABLESAMPLE ({pct:.2f} PERCENT) REPEATABLE ({random_seed})"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Spark ``TABLESAMPLE`` applies directly on the scanned table."""
        _ = table_kind
        return False

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return ``None`` because Spark SQL has no portable session statement-timeout statement. Databricks statement timeouts are enforced via ``spark.databricks.sql.statementTimeout`` in :class:`DatabricksSparkBackend` and connector query timeouts in :class:`DatabricksConnectorBackend`."""
        _ = timeout_ms
        return None


class DuckDBDialect(SqlglotEngineDialect):
    """DuckDB dialect using sqlglot read=duckdb and SQLAlchemy execution over duckdb_engine."""

    name: str = "duckdb"
    sqlglot_dialect: ClassVar[str] = "duckdb"
    registry_canonical_rank: ClassVar[int] = 1
    registry_native_backend: ClassVar[bool] = True
    registry_embedded: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True

    @staticmethod
    def _import_duckdb_module() -> Any:
        require_driver("duckdb")
        return importlib.import_module("duckdb")

    @staticmethod
    def _open_duckdb_connection(config: DuckDBRuntimeConfig, *, connection: Any | None = None) -> Any:
        """Return an existing or newly opened native duckdb connection."""
        if connection is not None:
            return connection
        return DuckDBDialect._import_duckdb_module().connect(str(config.DATABASE_PATH or ":memory:"))

    @staticmethod
    def create_duckdb_sqlalchemy_engine(connection: Any) -> Any:
        """Build a reflection engine over a single duckdb connection."""
        DuckDBDialect._import_duckdb_module()
        return _EmbeddedDuckDBEngine(connection)

    @staticmethod
    def extract_static_pool_connection(engine: Any) -> Any | None:
        """Return the underlying DBAPI connection from a shared embedded engine."""
        if isinstance(engine, _EmbeddedDuckDBEngine):
            return engine.raw_connection()
        pool = getattr(engine, "pool", None)
        if not isinstance(pool, StaticPool):
            return None
        try:
            return engine.raw_connection()
        except Exception:
            return None

    @staticmethod
    def _resolve_embedded_native_connection(
        config: EngineRuntimeConfig, sqlalchemy_engine: Any | None, native_connection: Any | None, *, open_new: Any
    ) -> tuple[Any, bool]:
        """Resolve the single native connection for an embedded dialect."""
        if native_connection is not None:
            return native_connection, False
        attached = type(config).NATIVE_CONNECTION
        if attached is not None:
            return attached, False
        if sqlalchemy_engine is not None:
            pooled = DuckDBDialect.extract_static_pool_connection(sqlalchemy_engine)
            if pooled is not None:
                return pooled, False
        return open_new(), True

    @staticmethod
    def _embedded_sqlalchemy_engine_for_connection(
        connection: Any, engine_name: Literal["duckdb", "sqlite"], sqlalchemy_engine: Any | None
    ) -> tuple[Any, bool]:
        """Reuse or build the SQLAlchemy engine for an embedded native connection."""
        if sqlalchemy_engine is not None:
            pooled = DuckDBDialect.extract_static_pool_connection(sqlalchemy_engine)
            if pooled is connection:
                return sqlalchemy_engine, False
        if engine_name == "duckdb":
            return DuckDBDialect.create_duckdb_sqlalchemy_engine(connection), True
        return SQLiteDialect.create_sqlite_sqlalchemy_engine(connection), True

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return DuckDB's portable binary collation for reproducible ordering."""
        return "C"

    @property
    def integer_division_truncates(self) -> bool:
        """Return False because DuckDB promotes integer division to double."""
        return False

    def render_median(self, expr_sql: str) -> str:
        return f"median({expr_sql})"

    def session_timezone_sql(self) -> str | None:
        """Return DuckDB session time-zone lookup SQL."""
        return "SELECT current_setting('TimeZone')"

    def __init__(
        self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None, *, native_connection: Any | None = None
    ) -> None:
        """Open one native duckdb connection for reflection and execution."""
        self._native_connection: Any | None = None
        self._owns_native_connection = False
        self._owns_sqlalchemy_engine = False
        if getattr(config, "ENGINE_NAME", None) == "duckdb":
            try:
                duckdb_config = cast(DuckDBRuntimeConfig, config)
                connection, owns_connection = DuckDBDialect._resolve_embedded_native_connection(
                    duckdb_config,
                    sqlalchemy_engine,
                    native_connection,
                    open_new=lambda: DuckDBDialect._open_duckdb_connection(duckdb_config),
                )
                engine, owns_engine = DuckDBDialect._embedded_sqlalchemy_engine_for_connection(
                    connection, "duckdb", sqlalchemy_engine
                )
                self._native_connection = connection
                self._owns_native_connection = owns_connection
                self._owns_sqlalchemy_engine = owns_engine
                super().__init__(config, sqlalchemy_engine=engine)
                reopen = (lambda: DuckDBDialect._open_duckdb_connection(duckdb_config)) if owns_connection else None
                self._backend = DuckDBNativeBackend(connection, reopen=reopen)
                return
            except Exception as exc:
                debug(f"[DuckDBDialect.__init__] duckdb unavailable: {exc!r}")
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        self._select_native_backend()

    def _select_native_backend(self) -> None:
        """Attach native duckdb when the database path is configured."""
        if self._native_connection is not None:
            return
        if getattr(self.config, "ENGINE_NAME", None) != "duckdb":
            return
        try:
            duck_config = cast(DuckDBRuntimeConfig, self.config)

            def _reopen() -> Any:
                return DuckDBDialect._open_duckdb_connection(duck_config)

            self._native_connection = _reopen()
            self._owns_native_connection = True
            self._backend = DuckDBNativeBackend(self._native_connection, reopen=_reopen)
        except Exception as exc:
            debug(f"[DuckDBDialect._select_native_backend] duckdb unavailable: {exc!r}")

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True because DuckDB exposes ``interrupt`` on native connections."""
        return True

    @property
    def logical_engine_name(self) -> str:
        return "DuckDB"

    def dispose_native_connection(self) -> None:
        """Close dialect-owned native and SQLAlchemy resources without touching injected handles."""
        if self._owns_native_connection and self._native_connection is not None:
            self._native_connection.close()
            self._native_connection = None
        if self._owns_sqlalchemy_engine:
            engine = self.engine
            if engine is not None:
                engine.dispose()
                self.engine = None
        self._backend = None

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active DuckDB row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return ResultReaderKind.SQLALCHEMY

    def can_explain(self) -> bool:
        """Return True when native duckdb or SQLAlchemy can run EXPLAIN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=self._native_connection
        )

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from DuckDB ``information_schema``."""
        schema_name = self.schema_name()
        return SqlglotEngineDialect.structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=getattr(self, "_native_connection", None),
        )

    def query_log_source(self) -> Any | None:
        """Return a documented no-op query-log source for DuckDB."""
        return NoOpQueryLogSource()

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Normalize DuckDB ``DATETRUNC`` emission to ``DATE_TRUNC``."""
        if stage != "post_substitute":
            return sql
        return SqlglotParseMixin.normalize_datetrunc_sql(sql, sqlglot_dialect=self.sqlglot_dialect)

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append partition predicates from ``TableMetadata.partition_columns`` when present."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return adapter.inject_partition_predicates(sql, schema, intent)

    @property
    def supports_ilike(self) -> bool:
        """Return True because DuckDB supports the ILIKE operator."""
        return True

    @property
    def supports_unnest_select_item(self) -> bool:
        """Return True because DuckDB allows UNNEST directly in the SELECT list."""
        return True

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return DuckDB CAST(... AS VARCHAR) for overlap sampling."""
        return f"CAST({expr} AS VARCHAR)"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from DuckDB reflection or SQL file fallback."""
        schema_name = str(getattr(self.config, "SCHEMA", None) or "main")
        return load_or_create_schema_duckdb(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
            schema_name=schema_name,
        )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return the DuckDB EXPLAIN wrapper."""
        return f"EXPLAIN {finalized_sql}"

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[Any], str]:
        """Parse DuckDB EXPLAIN rows into estimates and soft diagnostics."""
        _ = schema
        plan_text = "\n".join(str(cell) for row in rows for cell in row) if rows else ""
        diags = SqlglotEngineDialect.duckdb_diagnostics_from_explain_text(plan_text)
        est_rows, est_bytes = SqlglotEngineDialect.duckdb_root_plan_estimates(plan_text)
        return est_rows, est_bytes, diags, plan_text

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render a DuckDB date_diff comparison."""
        unit_token = unit.lower()
        if minuend_sql and subtrahend_sql:
            sql = f"date_diff('{unit_token}', {subtrahend_sql}, {minuend_sql}) {op} {amount}"
        else:
            sql = f"date_diff('{unit_token}', {left_expr}, current_date) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_containment(self, column_sql: str, value_param: str, value_type: str) -> str | None:
        """Render DuckDB JSON-array containment with ``list_contains``."""
        _ = value_type
        return f"list_contains(CAST({column_sql} AS JSON), CAST({value_param} AS VARCHAR))"

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render DuckDB list membership with case-insensitive comparison."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect=self.sqlglot_dialect,
            )
        norm_param = f"LOWER(TRIM(BOTH '%' FROM :{param_key}))"
        sql = f"list_contains(list_transform({column_sql}, x -> lower(x)), {norm_param})"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render DuckDB UNNEST for a SELECT list item."""
        sql = f"UNNEST({column_sql}) AS {alias}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render a DuckDB relative date-window boundary using DATE_TRUNC and INTERVAL."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', {clock})"
        else:
            scaled, plural = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} ({clock} - INTERVAL '{scaled} {plural}')"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression in LOWER(...) for case-insensitive comparison."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the inclusive upper-bound timestamp expression for DuckDB."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a DuckDB seeded ``USING SAMPLE`` suffix when sampling, else an empty suffix."""
        _ = table_kind
        if not use_sample or row_count <= 0:
            return ""
        pct = max(min(100.0 * float(sample_size) / float(row_count), 100.0), 0.0001)
        return DUCKDB_PROFILING_SAMPLE_PREDICATE.format(pct=pct, seed=random_seed)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Return True so distinct/null stats scan a sampled subquery."""
        _ = table_kind
        return True


class SQLiteDialect(SqlglotEngineDialect):
    """SQLite dialect using sqlglot read=sqlite and SQLAlchemy execution over the bundled pysqlite driver."""

    name: str = "sqlite"
    sqlglot_dialect: ClassVar[str] = "sqlite"
    registry_canonical_rank: ClassVar[int] = 0
    registry_native_backend: ClassVar[bool] = True
    registry_embedded: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_statistical_agg_excluded: ClassVar[bool] = True
    registry_array_contains_excluded: ClassVar[bool] = True

    @staticmethod
    def _open_sqlite_connection(config: SQLiteRuntimeConfig, *, connection: Any | None = None) -> Any:
        """Return an existing or newly opened stdlib sqlite3 connection."""
        if connection is not None:
            return connection
        import sqlite3

        return sqlite3.connect(str(config.DATABASE_PATH or ":memory:"), check_same_thread=False)

    @staticmethod
    def create_sqlite_sqlalchemy_engine(connection: Any) -> Any:
        """Build a SQLAlchemy engine over a single sqlite3 connection."""
        return create_engine(
            "sqlite:///",
            creator=lambda: connection,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            future=True,
        )

    @property
    def supports_ordered_string_agg(self) -> bool:
        return False

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return False because SQLite has no portable in-flight cancel hook."""
        return False

    @property
    def logical_engine_name(self) -> str:
        return "SQLite"

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"GROUP_CONCAT({expr_sql})"
        return f"GROUP_CONCAT({expr_sql}, {sep_sql})"

    def __init__(
        self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None, *, native_connection: Any | None = None
    ) -> None:
        """Open one native sqlite3 connection for reflection and execution."""
        self._native_connection: Any | None = None
        self._owns_native_connection = False
        self._owns_sqlalchemy_engine = False
        if getattr(config, "ENGINE_NAME", None) == "sqlite":
            try:
                sqlite_config = cast(SQLiteRuntimeConfig, config)
                connection, owns_connection = DuckDBDialect._resolve_embedded_native_connection(
                    sqlite_config,
                    sqlalchemy_engine,
                    native_connection,
                    open_new=lambda: SQLiteDialect._open_sqlite_connection(sqlite_config),
                )
                engine, owns_engine = DuckDBDialect._embedded_sqlalchemy_engine_for_connection(
                    connection, "sqlite", sqlalchemy_engine
                )
                self._native_connection = connection
                self._owns_native_connection = owns_connection
                self._owns_sqlalchemy_engine = owns_engine
                super().__init__(config, sqlalchemy_engine=engine)
                self._backend = SQLiteNativeBackend(connection)
                return
            except Exception as exc:
                debug(f"[SQLiteDialect.__init__] sqlite3 unavailable: {exc!r}")
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        self._select_native_backend()

    def _select_native_backend(self) -> None:
        """Attach stdlib sqlite3 when the database path is configured."""
        if self._native_connection is not None:
            return
        if getattr(self.config, "ENGINE_NAME", None) != "sqlite":
            return
        try:
            sqlite_config = cast(SQLiteRuntimeConfig, self.config)
            self._native_connection = SQLiteDialect._open_sqlite_connection(sqlite_config)
            self._owns_native_connection = True
            self._backend = SQLiteNativeBackend(self._native_connection)
        except Exception as exc:
            debug(f"[SQLiteDialect._select_native_backend] sqlite3 unavailable: {exc!r}")

    def dispose_native_connection(self) -> None:
        """Close dialect-owned native and SQLAlchemy resources without touching injected handles."""
        if self._owns_native_connection and self._native_connection is not None:
            self._native_connection.close()
            self._native_connection = None
        if self._owns_sqlalchemy_engine:
            engine = self.engine
            if engine is not None:
                engine.dispose()
                self.engine = None
        self._backend = None

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the active SQLite row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return ResultReaderKind.SQLALCHEMY

    def can_explain(self) -> bool:
        """Return True when sqlite3 or SQLAlchemy can run EXPLAIN QUERY PLAN."""
        return SqlalchemyExecutionMixin.can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=self._native_connection
        )

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load FK metadata from ``PRAGMA foreign_key_list`` when foreign keys are enabled."""
        return SqlglotEngineDialect.sqlite_structural_constraints_index(getattr(self, "engine", None))

    def query_log_source(self) -> Any | None:
        """Return a documented no-op query-log source for SQLite."""
        return NoOpQueryLogSource()

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return SQLite CAST(... AS TEXT) for overlap sampling."""
        return f"CAST({expr} AS TEXT)"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from SQLite reflection or SQL file fallback."""
        return load_or_create_schema_sqlite(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
        )

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return a DDL fingerprint from sqlite_master and PRAGMA table_info rows."""
        _ = schema_context
        eng = getattr(self, "engine", None)
        if eng is None:
            return ""
        try:
            with eng.connect() as conn:
                table_rows = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
                ).fetchall()
                payload: list[str] = []
                for table_name in table_rows:
                    info_rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
                    for info in info_rows:
                        payload.append(f"{table_name}|{info[1]}|{info[0]}|{info[2]}|{info[3]}")
            return sha256("\n".join(sorted(payload)))
        except Exception:
            return ""

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return the SQLite EXPLAIN QUERY PLAN wrapper."""
        return f"EXPLAIN QUERY PLAN {finalized_sql}"

    def parse_explain_plan(
        self, rows: list[Any], *, schema: SchemaGraph | None = None
    ) -> tuple[float | None, float | None, list[Any], str]:
        """Parse SQLite EXPLAIN QUERY PLAN rows into soft diagnostics with no cardinality estimate."""
        _ = schema
        plan_text = "\n".join(str(cell) for row in rows for cell in row) if rows else ""
        diags = SqlglotEngineDialect.sqlite_diagnostics_from_query_plan(plan_text)
        return None, None, diags, plan_text

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return None because SQLite provides no planner row-count estimate."""
        _ = (sql_text, schema, intent)
        return None

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render a SQLite day-grain date difference using julianday."""
        if minuend_sql and subtrahend_sql:
            sql = f"(julianday({minuend_sql}) - julianday({subtrahend_sql})) {op} {amount}"
        else:
            sql = f"(julianday('now') - julianday({left_expr})) {op} {amount}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render SQLite membership without substring JSON search; capability gate excludes contains."""
        _ = value_type
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "unknown"
        if kind == "json_text_array":
            raise ValueError("json containment is not supported for dialect 'sqlite'")
        norm_param = f"LOWER(TRIM(CAST(:{param_key} AS TEXT)))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS TEXT))"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render SQLite json_each as the unnest source for a SELECT list item."""
        sql = f"json_each({column_sql}) AS {alias}"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window_anchor_literal(self, anchor: datetime, unit: str) -> str:
        """Render a bound anchor instant as a SQLite date/timestamp literal."""
        aware = anchor if anchor.tzinfo is not None else anchor.replace(tzinfo=UTC)
        if Dialect.relative_window_uses_timestamp(unit):
            text = aware.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
            return f"datetime('{text}')"
        return f"date('{aware.date().isoformat()}')"

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render a SQLite relative date-window boundary using the date modifier syntax."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            sql = f"{column} {op} {clock}"
        else:
            scaled, plural = Dialect.format_interval_unit(unit, amount)
            if anchor is not None:
                fn = "datetime" if Dialect.relative_window_uses_timestamp(unit) else "date"
                sql = f"{column} {op} {fn}({clock}, '-{scaled} {plural}')"
            elif Dialect.relative_window_uses_timestamp(unit):
                sql = f"{column} {op} datetime('now', '-{scaled} {plural}')"
            else:
                sql = f"{column} {op} date('now', '-{scaled} {plural}')"
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_trunc(self, unit: str, expr_sql: str) -> str:
        """Render SQLite calendar truncation with ISO week semantics."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "week":
            sql = f"date({expr_sql}, '-' || ((strftime('%w', {expr_sql}) + 6) % 7) || ' days')"
        elif unit_norm == "quarter":
            sql = (
                f"date({expr_sql}, 'start of month', '-' || "
                f"(((cast(strftime('%m', {expr_sql}) as integer) - 1) % 3)) || ' months')"
            )
        elif unit_norm == "half_year":
            sql = (
                f"CASE WHEN cast(strftime('%m', {expr_sql}) as integer) <= 6 "
                f"THEN date({expr_sql}, 'start of year') "
                f"ELSE date({expr_sql}, 'start of year', '+6 months') END"
            )
        elif unit_norm == "month":
            sql = f"date({expr_sql}, 'start of month')"
        elif unit_norm == "year":
            sql = f"date({expr_sql}, 'start of year')"
        else:
            return super().render_date_trunc(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_extract(self, unit: str, expr_sql: str) -> str:
        """Render SQLite calendar extraction with native quarter and half-year units."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "half_year":
            sql = f"CASE WHEN cast(strftime('%m', {expr_sql}) as integer) <= 6 THEN 1 ELSE 2 END"
        elif unit_norm == "quarter":
            sql = f"((cast(strftime('%m', {expr_sql}) as integer) - 1) / 3) + 1"
        elif unit_norm == "week":
            sql = f"cast(strftime('%W', {expr_sql}) as integer)"
        else:
            return super().render_extract(unit, expr_sql)
        return Dialect.emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression in LOWER(...) for case-insensitive comparison."""
        return f"LOWER({expr})"

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the SQLite clock expression for relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "datetime('now')"
        return "date('now')"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the inclusive upper-bound expression for SQLite."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a SQLite seeded hash-bucket ``WHERE`` suffix when sampling."""
        _ = table_kind
        if not use_sample:
            return ""
        ratio = max(0.0001, min(1.0, sample_size / max(row_count, 1)))
        return f"WHERE {SQLITE_PROFILING_SAMPLE_PREDICATE.format(ratio=ratio, seed=random_seed)}"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """SQLite samples via a ``WHERE random()`` predicate inside a subquery."""
        _ = table_kind
        return True


DialectRegistry.register_dialect("mysql", MySQLDialect, MySQLRuntimeConfig)
DialectRegistry.register_dialect("mariadb", MariaDBDialect, MariaDBRuntimeConfig)
DialectRegistry.register_dialect("duckdb", DuckDBDialect, DuckDBRuntimeConfig)
DialectRegistry.register_dialect("sqlite", SQLiteDialect, SQLiteRuntimeConfig)
DialectRegistry.register_dialect("redshift", RedshiftDialect, RedshiftRuntimeConfig)
DialectRegistry.register_dialect("snowflake", SnowflakeDialect, SnowflakeRuntimeConfig)
DialectRegistry.register_dialect("sqlserver", SQLServerDialect, SQLServerRuntimeConfig)
DialectRegistry.register_dialect("oracle", OracleDialect, OracleRuntimeConfig)
DialectRegistry.register_dialect("bigquery", BigQueryDialect, BigQueryRuntimeConfig)
DialectRegistry.register_dialect("databricks", DatabricksDialect, DatabricksRuntimeConfig)


class CsvDialect(DuckDBDialect):
    """CSV/Excel files loaded into an in-memory DuckDB database per session."""

    name: str = "csv"
    sqlglot_dialect: ClassVar[str] = "duckdb"
    registry_canonical_rank: ClassVar[int] = 2
    registry_statistical_agg_excluded: ClassVar[bool] = True
    registry_window_frames_excluded: ClassVar[bool] = True
    registry_array_contains_excluded: ClassVar[bool] = True

    @staticmethod
    def _normalize_header_columns(headers: Sequence[str], *, source: Path) -> list[str]:
        columns = [str(h).strip() for h in headers]
        if not columns or all(not col for col in columns):
            raise ConfigError(f"csv file missing header row: {source}")
        seen: set[str] = set()
        for col in columns:
            if not col:
                raise ConfigError(f"csv file has empty header column in {source}")
            if col in seen:
                raise ConfigError(f"csv duplicate header column {col!r} in {source}")
            seen.add(col)
        return columns

    @staticmethod
    def _read_csv_header(path: Path) -> list[str]:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ConfigError(f"csv file is empty: {path}") from exc
        return CsvDialect._normalize_header_columns(row, source=path)

    @staticmethod
    def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ConfigError(f"csv file missing header row: {path}")
            columns = CsvDialect._normalize_header_columns(reader.fieldnames, source=path)
            for raw in reader:
                yield {col: str(raw.get(col) or "") for col in columns}

    @staticmethod
    def _read_xlsx_header(path: Path) -> list[str]:
        require_driver("csv")
        openpyxl = importlib.import_module("openpyxl")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            worksheet = workbook.active
            row = next(worksheet.iter_rows(min_row=1, max_row=1))
        except StopIteration as exc:
            raise ConfigError(f"csv file is empty: {path}") from exc
        finally:
            workbook.close()
        headers = [excel_cell_to_text(cell) for cell in row]
        return CsvDialect._normalize_header_columns(headers, source=path)

    @staticmethod
    def _iter_xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
        require_driver("csv")
        openpyxl = importlib.import_module("openpyxl")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            worksheet = workbook.active
            row_iter = worksheet.iter_rows()
            try:
                header_row = next(row_iter)
            except StopIteration as exc:
                raise ConfigError(f"csv file is empty: {path}") from exc
            columns = CsvDialect._normalize_header_columns(
                [excel_cell_to_text(cell) for cell in header_row], source=path
            )
            for row_cells in row_iter:
                if row_cells is None:
                    continue
                row_map = {
                    col: excel_cell_to_text(row_cells[idx]) if idx < len(row_cells) else ""
                    for idx, col in enumerate(columns)
                }
                if any(str(v).strip() for v in row_map.values()):
                    yield row_map
        finally:
            workbook.close()

    @staticmethod
    def _read_source_header(path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return CsvDialect._read_csv_header(path)
        if suffix == ".xlsx":
            return CsvDialect._read_xlsx_header(path)
        raise ConfigError(f"csv unsupported file type: {path}")

    @staticmethod
    def _iter_source_rows(path: Path) -> Iterator[dict[str, str]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            yield from CsvDialect._iter_csv_rows(path)
        elif suffix == ".xlsx":
            yield from CsvDialect._iter_xlsx_rows(path)
        else:
            raise ConfigError(f"csv unsupported file type: {path}")

    @staticmethod
    def _looks_boolean(value: str) -> bool:
        return value.strip().lower() in BOOL_LITERALS

    @staticmethod
    def _looks_integer(value: str) -> bool:
        text = value.strip()
        if not text or "." in text or "e" in text.lower():
            return False
        int(text)
        return True

    @staticmethod
    def _looks_number(value: str) -> bool:
        float(value.strip())
        return True

    @staticmethod
    def infer_duckdb_column_type(samples: Sequence[str]) -> str:
        return infer_duckdb_column_type(samples)

    @staticmethod
    def _coerce_typed_cell(value: str | None, duckdb_type: str) -> object:
        if value is None or str(value).strip() == "":
            return None
        text = str(value).strip()
        upper = duckdb_type.upper()
        if "BOOL" in upper:
            return text.lower() in ("1", "true", "t", "yes")
        if any(token in upper for token in ("TIMESTAMP", "DATETIME")):
            return datetime.fromisoformat(text)
        if "DATE" in upper:
            return date.fromisoformat(text[:10])
        if any(token in upper for token in ("INT", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT")):
            return int(text)
        if any(token in upper for token in ("DECIMAL", "NUMERIC")):
            return Decimal(text)
        if any(token in upper for token in ("DOUBLE", "FLOAT", "REAL")):
            return float(text)
        return text

    @staticmethod
    def _load_locked_column_types(schema_json_path: str | Path | None) -> dict[str, dict[str, str]]:
        path = str(schema_json_path or "").strip()
        if not path or not os.path.isfile(path):
            return {}
        try:
            payload = read_gzip_json(path)
            graph = SchemaGraph.from_dict(payload)
        except Exception as exc:
            debug(f"[csv._load_locked_column_types] cache read failed: {exc!r}")
            return {}
        locked: dict[str, dict[str, str]] = {}
        for table_name, table in graph.tables.items():
            locked[table_name.lower()] = {
                col_name: str(col.data_type or "VARCHAR") for col_name, col in table.columns.items()
            }
        return locked

    @staticmethod
    def _column_types_for_source(
        path: Path, *, locked_types: Mapping[str, str] | None = None, sample_limit: int = 512
    ) -> tuple[list[str], list[str]]:
        columns = CsvDialect._read_source_header(path)
        locked = dict(locked_types or {})
        samples: dict[str, list[str]] = {col: [] for col in columns}
        for row in CsvDialect._iter_source_rows(path):
            for col in columns:
                bucket = samples[col]
                if len(bucket) >= sample_limit:
                    continue
                bucket.append(row.get(col, ""))
        types: list[str] = []
        for col in columns:
            locked_type = locked.get(col)
            types.append(locked_type if locked_type else infer_duckdb_column_type(samples[col]))
        return columns, types

    @staticmethod
    def _csv_schema_pins(schema_json_path: str | None = None) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
        cached = load_schema_graph_snapshot(schema_json_path or EngineConfig.SCHEMA_JSON_PATH)
        return pinned_names_from_schema_graph(cached)

    @staticmethod
    def _csv_relations_for_config(
        config: CsvRuntimeConfig, *, schema_json_path: str | None = None
    ) -> list[PreparedRelation]:
        paths = config.resolve_source_files()
        table_pins, column_pins = CsvDialect._csv_schema_pins(schema_json_path)
        return prepare_relations_for_paths(
            paths,
            pinned_table_names=table_pins,
            pinned_column_names=column_pins,
            apply_auto_correct=True,
            source_selections=parse_source_selections(config.SOURCE_SELECTIONS),
        )

    @staticmethod
    def _tables_meta_from_relations(
        relations: Sequence[PreparedRelation],
        *,
        locked_by_table: Mapping[str, Mapping[str, str]] | None = None,
        allow_objects: frozenset[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        allow_lower = allow_objects_lower_set(allow_objects)
        locked = locked_by_table or {}
        tables_meta: dict[str, dict[str, Any]] = {}
        for relation in relations:
            if allow_lower is not None and relation.relation_name.lower() not in allow_lower:
                continue
            types = list(relation.column_types)
            locked_types = locked.get(relation.relation_name.lower(), {})
            for idx, col in enumerate(relation.columns):
                if col in locked_types:
                    types[idx] = locked_types[col]
            column_comments: dict[str, str] = {}
            for idx, col in enumerate(relation.columns):
                unit = relation.column_unit_labels[idx] if idx < len(relation.column_unit_labels) else ""
                if unit:
                    column_comments[col] = f"Values in {unit} (unit affix stripped on ingest)."
            tables_meta[relation.relation_name] = {
                "column_names_original": list(relation.columns),
                "original_column_labels": list(relation.original_column_labels),
                "original_table_label": relation.original_table_label,
                "column_types": types,
                "column_comments": column_comments,
                "column_is_nullable": [True] * len(relation.columns),
                "primary_keys": [],
                "unique_columns": [],
                "foreign_keys": [],
            }
        return tables_meta

    @staticmethod
    def _create_table_sql(table: str, columns: Sequence[str], types: Sequence[str]) -> str:
        quote = Dialect.sqlglot_quote_identifier
        col_defs = ", ".join(f"{quote(col)} {duckdb_type}" for col, duckdb_type in zip(columns, types, strict=True))
        return f"CREATE TABLE {quote(table)} ({col_defs})"

    @staticmethod
    def _insert_rows(
        connection: Any, table: str, columns: Sequence[str], types: Sequence[str], rows: Sequence[Mapping[str, str]]
    ) -> None:
        if not rows:
            return
        quote = Dialect.sqlglot_quote_identifier
        col_sql = ", ".join(quote(col) for col in columns)
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f"INSERT INTO {quote(table)} ({col_sql}) VALUES ({placeholders})"
        for row in rows:
            values = [
                CsvDialect._coerce_typed_cell(row.get(col), duckdb_type)
                for col, duckdb_type in zip(columns, types, strict=True)
            ]
            connection.execute(insert_sql, values)

    @staticmethod
    def _load_prepared_relation_into_connection(connection: Any, relation: PreparedRelation) -> None:
        locked_by_table = CsvDialect._load_locked_column_types(EngineConfig.SCHEMA_JSON_PATH)
        locked_types = locked_by_table.get(relation.relation_name.lower(), {})
        types: list[str] = []
        for idx, col in enumerate(relation.columns):
            locked_type = locked_types.get(col)
            types.append(locked_type if locked_type else relation.column_types[idx])
        quote = Dialect.sqlglot_quote_identifier
        connection.execute(f"DROP TABLE IF EXISTS {quote(relation.relation_name)}")
        connection.execute(CsvDialect._create_table_sql(relation.relation_name, relation.columns, types))
        CsvDialect._insert_rows(connection, relation.relation_name, relation.columns, types, list(relation.rows))

    @staticmethod
    def load_prepared_relation_into_native_connection(connection: Any, relation: PreparedRelation) -> None:
        """Materialise one validated upload relation into an embedded native connection."""
        CsvDialect._load_prepared_relation_into_connection(connection, relation)

    @staticmethod
    def _normalize_upload_paths(paths: Sequence[str | os.PathLike[str]]) -> tuple[Path, ...]:
        return tuple(Path(os.fspath(path)) for path in paths)

    @staticmethod
    def _normalize_upload_selections(
        source_selections: Mapping[str, CsvSourceSelection | Mapping[str, Any]] | None,
    ) -> dict[str, CsvSourceSelection]:
        if not source_selections:
            return {}
        out: dict[str, CsvSourceSelection] = {}
        raw: dict[str, Mapping[str, Any]] = {}
        for name, body in source_selections.items():
            if isinstance(body, CsvSourceSelection):
                out[str(name)] = body
            else:
                raw[str(name)] = body
        if raw:
            out.update(parse_source_selections(raw))
        return out

    @staticmethod
    def _upload_selection_to_confirmed_dict(selection: CsvSourceSelection) -> dict[str, Any]:
        confirmed: dict[str, Any] = {}
        if selection.sheet:
            confirmed["sheet"] = selection.sheet
        if selection.header_row is not None:
            confirmed["header_row"] = selection.header_row
        if selection.skip_rows:
            confirmed["skip_rows"] = selection.skip_rows
        if selection.table_range:
            confirmed["table_range"] = selection.table_range
        if selection.merge_regions:
            confirmed["merge_regions"] = list(selection.merge_regions)
        if selection.append_regions:
            confirmed["append_regions"] = list(selection.append_regions)
        if selection.column_transforms:
            confirmed["column_transforms"] = [dict(item) for item in selection.column_transforms]
        return confirmed

    @staticmethod
    def _confirmed_upload_selections_payload(
        selections: Mapping[str, CsvSourceSelection],
    ) -> dict[str, dict[str, Any]]:
        return {
            name: CsvDialect._upload_selection_to_confirmed_dict(selection) for name, selection in selections.items()
        }

    @staticmethod
    def _resolve_upload_engine_type(engine: Any) -> str:
        engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
        if engine_type:
            return engine_type
        runtime_cfg = getattr(engine, "_runtime_config", None)
        return str(getattr(runtime_cfg, "engine", "") or "").strip().lower()

    @staticmethod
    def _resolve_upload_native_connection(engine: Any) -> Any:
        dialect = getattr(engine, "_dialect", None)
        connection = getattr(dialect, "_native_connection", None) if dialect is not None else None
        if connection is None:
            connection = getattr(engine, "_native_connection", None)
        if connection is None:
            raise ConfigError("ingest_upload_sources requires an embedded engine with a native connection")
        return connection

    @staticmethod
    def _upload_validation_config_error(message: str, data_quality_report: object) -> ConfigError:
        exc = ConfigError(message)
        cast(Any, exc).data_quality_report = data_quality_report
        return exc

    @staticmethod
    def _sync_upload_engine_config_globals(engine: Any) -> None:
        engine_type = CsvDialect._resolve_upload_engine_type(engine)
        dialect = getattr(engine, "_dialect", None)
        config = getattr(dialect, "config", None)
        if config is None or isinstance(config, type):
            raise ConfigError(f"ingest_upload_sources requires a configured {engine_type!r} engine member")
        EngineConfig.TYPE = engine_type
        EngineConfig.RUNTIME = config

    @staticmethod
    def _refresh_engine_schema_after_ingest(engine: Any) -> tuple[SchemaGraph, SchemaDiff | None]:
        dialect = getattr(engine, "_dialect", None)
        if dialect is None:
            raise ConfigError("ingest_upload_sources requires a live engine dialect")
        cached_sg = getattr(engine, "_schema_graph", None)
        if not isinstance(cached_sg, SchemaGraph):
            raise ConfigError("ingest_upload_sources requires an initialized schema graph")
        runtime_cfg = getattr(engine, "_runtime_config", None)
        schema_context = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
        if schema_context is None:
            schema_context = EngineContext()
        notes_sha = str(getattr(cached_sg, "notes_sha256", "") or "")
        schema_json_path = str(getattr(engine, "_artifacts_dir", Path(".")) / "schema_graph.json.gz")
        EngineConfig.SCHEMA_JSON_PATH = schema_json_path
        CsvDialect._sync_upload_engine_config_globals(engine)
        new_struct = dialect.reflect_schema_graph()
        if not isinstance(new_struct, SchemaGraph):
            raise ConfigError("ingest_upload_sources failed to reflect the updated member schema")
        schema_diff = diff_schemas(cached_sg, new_struct)
        if not schema_diff.is_empty:
            apply_diff(
                cached_sg,
                new_struct,
                schema_diff,
                dialect,
                schema_json_path=schema_json_path,
            )
        cached_sg.notes_sha256 = notes_sha
        raise_if_schema_unusable(cached_sg, schema_context)
        assign_schema_graph_hashes(
            cached_sg,
            schema_context,
            notes_sha,
            schema_role=SchemaRole.coerce(getattr(engine, "_schema_role", SchemaRole.OWNER)),
        )
        save_schema_to_cache(cached_sg, schema_json_path)
        migrate_sidecar_for_diff(schema_json_path, schema_diff)
        finalize_with_structure(cached_sg, schema_json_path, dialect=dialect)
        return cached_sg, schema_diff if not schema_diff.is_empty else None

    @staticmethod
    def ingest_upload_sources_into_engine(
        engine: Any,
        paths: Sequence[str | os.PathLike[str]],
        *,
        source_selections: Mapping[str, CsvSourceSelection | Mapping[str, Any]] | None = None,
        relation_names: Mapping[str, str] | None = None,
        log_sink: Callable[[str], None] | None = None,
    ) -> UploadIngestResult:
        """Validate uploads, materialise relations into *engine*, and return schema delta info."""
        engine_type = CsvDialect._resolve_upload_engine_type(engine)
        if (engine_type or "").strip().lower() not in UPLOAD_INGEST_ENGINE_NAMES:
            raise ConfigError(
                f"ingest_upload_sources requires a duckdb or csv engine member, got {engine_type!r}",
            )
        resolved_paths = CsvDialect._normalize_upload_paths(paths)
        if not resolved_paths:
            raise ConfigError("ingest_upload_sources requires at least one upload path")
        selections = CsvDialect._normalize_upload_selections(source_selections)
        selection_by_name = {path.name: selections[path.name] for path in resolved_paths if path.name in selections}
        report = validate_upload_sources(
            resolved_paths,
            log_sink=log_sink,
            source_selections=selection_by_name,
        )
        if report.requires_review and not selections:
            raise CsvDialect._upload_validation_config_error(
                f"{report.narrative} "
                "Call inspect_tabular_upload and pass source_selections with the accepted interpretation.",
                report,
            )
        if not report.ok:
            raise CsvDialect._upload_validation_config_error(report.narrative, report)
        if selections:
            report = DataQualityReport(
                ok=report.ok,
                issues=report.issues,
                narrative=report.narrative,
                suggested_selections=report.suggested_selections,
                confirmed_selections=CsvDialect._confirmed_upload_selections_payload(selections),
            )
        cached_sg = getattr(engine, "_schema_graph", None)
        if not isinstance(cached_sg, SchemaGraph):
            raise ConfigError("ingest_upload_sources requires an initialized schema graph")
        table_pins, column_pins = pinned_names_from_schema_graph(cached_sg)
        if relation_names:
            table_pins.update({str(key): str(value) for key, value in relation_names.items()})
        relations = prepare_relations_for_paths(
            resolved_paths,
            pinned_table_names=table_pins,
            pinned_column_names=column_pins,
            source_selections=selection_by_name,
        )
        schema_json_path = str(getattr(engine, "_artifacts_dir", Path(".")) / "schema_graph.json.gz")
        EngineConfig.SCHEMA_JSON_PATH = schema_json_path
        CsvDialect._sync_upload_engine_config_globals(engine)
        connection = CsvDialect._resolve_upload_native_connection(engine)
        for relation in relations:
            CsvDialect.load_prepared_relation_into_native_connection(connection, relation)
        relation_names_created = tuple(relation.relation_name for relation in relations)
        artifacts_dir = str(getattr(engine, "_artifacts_dir", "") or "").strip()
        if artifacts_dir:
            ingest_probe = CsvDialect._csv_source_probe_payload(resolved_paths)
            prior = read_artifact_manifest(artifacts_dir)
            config_probe = prior.source_probe if prior is not None else ""
            if not config_probe:
                try:
                    csv_config = cast(CsvRuntimeConfig, getattr(getattr(engine, "_dialect", None), "config", None))
                    if isinstance(csv_config, CsvRuntimeConfig):
                        config_probe = CsvDialect._csv_source_probe_payload(csv_config.resolve_source_files())
                except (ConfigError, OSError, TypeError, ValueError):
                    config_probe = ""
            combined = CsvDialect._csv_combined_store_probe(config_probe, ingest_probe)
            write_artifact_manifest(
                artifacts_dir,
                structural_hash=prior.structural_hash if prior is not None else "",
                profiling_hash=prior.profiling_hash if prior is not None else "",
                scope_hash=prior.scope_hash if prior is not None else "",
                effective_structural_hash=prior.effective_structural_hash if prior is not None else "",
                schema_graph_id=prior.schema_graph_id if prior is not None else "",
                notes_hash=prior.notes_hash if prior is not None else "",
                semantic_edges_hash=prior.semantic_edges_hash if prior is not None else "",
                last_migration_tier=prior.last_migration_tier if prior is not None else "",
                last_migration_at=prior.last_migration_at if prior is not None else "",
                last_action="csv_ingest_write_through",
                source_probe=config_probe,
                store_fingerprint=combined,
                ingest_source_probe=ingest_probe,
            )
        updated_schema, schema_diff = CsvDialect._refresh_engine_schema_after_ingest(engine)
        engine._schema_graph = updated_schema
        engine._schema_stats = updated_schema.refresh_schema_stats()
        engine._data_quality_report = report
        return UploadIngestResult(
            relation_names=relation_names_created,
            report=report,
            schema_diff=schema_diff,
        )

    @staticmethod
    def _csv_artifacts_dir_from_schema_json() -> str | None:
        schema_json = str(EngineConfig.SCHEMA_JSON_PATH or "").strip()
        if not schema_json:
            return None
        path = Path(schema_json).expanduser()
        if path.name != "schema_graph.json.gz":
            return None
        parent = path.resolve().parent
        if not parent.is_dir() and not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                return None
        return str(parent)

    @staticmethod
    def _csv_upload_store_path(artifacts_dir: str) -> Path:
        return Path(artifacts_dir) / UPLOAD_STORE_FILENAME

    @staticmethod
    def _csv_combined_store_probe(config_probe: str, ingest_probe: str = "") -> str:
        parts = [str(config_probe or "").strip()]
        ingest = str(ingest_probe or "").strip()
        if ingest:
            parts.append(ingest)
        return "\n".join(parts)

    @staticmethod
    def _stamp_csv_store_manifest(artifacts_dir: str, *, source_probe: str, store_fingerprint: str) -> None:
        prior = read_artifact_manifest(artifacts_dir)
        write_artifact_manifest(
            artifacts_dir,
            structural_hash=prior.structural_hash if prior is not None else "",
            profiling_hash=prior.profiling_hash if prior is not None else "",
            scope_hash=prior.scope_hash if prior is not None else "",
            effective_structural_hash=prior.effective_structural_hash if prior is not None else "",
            schema_graph_id=prior.schema_graph_id if prior is not None else "",
            notes_hash=prior.notes_hash if prior is not None else "",
            semantic_edges_hash=prior.semantic_edges_hash if prior is not None else "",
            last_migration_tier=prior.last_migration_tier if prior is not None else "",
            last_migration_at=prior.last_migration_at if prior is not None else "",
            last_action="csv_store_refresh",
            source_probe=source_probe,
            store_fingerprint=store_fingerprint,
            ingest_source_probe=prior.ingest_source_probe if prior is not None else "",
        )

    @staticmethod
    def _build_csv_memory_connection(config: CsvRuntimeConfig) -> Any:
        """Open an in-memory or durable DuckDB store for CSV/Excel sources."""
        duckdb_mod = DuckDBDialect._import_duckdb_module()
        artifacts_dir = CsvDialect._csv_artifacts_dir_from_schema_json()
        paths = tuple(config.resolve_source_files())
        config_probe = CsvDialect._csv_source_probe_payload(paths)
        if artifacts_dir is None:
            connection = duckdb_mod.connect(":memory:")
            for relation in CsvDialect._csv_relations_for_config(config):
                CsvDialect._load_prepared_relation_into_connection(connection, relation)
            return connection
        store_path = CsvDialect._csv_upload_store_path(artifacts_dir)
        with artifact_lock(artifacts_dir):
            prior = read_artifact_manifest(artifacts_dir)
            ingest_probe = prior.ingest_source_probe if prior is not None else ""
            desired = CsvDialect._csv_combined_store_probe(config_probe, ingest_probe)
            if (
                store_path.is_file()
                and prior is not None
                and prior.store_fingerprint
                and prior.store_fingerprint == desired
            ):
                try:
                    return duckdb_mod.connect(str(store_path))
                except Exception as exc:
                    debug(f"[CsvDialect._build_csv_memory_connection] corrupt store reopen: {exc!r}")
                    try:
                        store_path.unlink()
                    except OSError:
                        pass
            tmp_path = Path(tempfile.mkstemp(prefix=".upload_store_", suffix=".duckdb.tmp", dir=artifacts_dir)[1])
            try:
                connection = duckdb_mod.connect(str(tmp_path))
                for relation in CsvDialect._csv_relations_for_config(config):
                    CsvDialect._load_prepared_relation_into_connection(connection, relation)
                connection.close()
                os.replace(tmp_path, store_path)
            except Exception:
                try:
                    if tmp_path.is_file():
                        tmp_path.unlink()
                except OSError:
                    pass
                raise
            CsvDialect._stamp_csv_store_manifest(
                artifacts_dir,
                source_probe=config_probe,
                store_fingerprint=desired,
            )
            return duckdb_mod.connect(str(store_path))

    @staticmethod
    def _load_source_into_connection(
        connection: Any, path: Path, *, locked_types: Mapping[str, str] | None = None
    ) -> None:
        _ = locked_types
        table_pins, column_pins = CsvDialect._csv_schema_pins()
        for relation in prepare_relations_for_paths(
            [path],
            pinned_table_names=table_pins,
            pinned_column_names=column_pins,
            apply_auto_correct=True,
            source_selections=parse_source_selections(CsvRuntimeConfig.SOURCE_SELECTIONS),
        ):
            CsvDialect._load_prepared_relation_into_connection(connection, relation)

    @staticmethod
    def _csv_source_probe_payload(paths: Sequence[Path]) -> str:
        parts: list[str] = []
        for path in paths:
            stat = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            identity_path = os.path.normcase(os.path.abspath(os.fspath(path)))
            parts.append(f"{identity_path}|{stat.st_mtime_ns}|{digest}")
        return "\n".join(sorted(parts))

    def __init__(
        self, config: CsvRuntimeConfig, sqlalchemy_engine: Any | None = None, *, native_connection: Any | None = None
    ) -> None:
        self._schema_json_path = str(EngineConfig.SCHEMA_JSON_PATH)
        self._native_connection: Any | None = None
        self._owns_native_connection = False
        self._owns_sqlalchemy_engine = False
        try:
            connection, owns_connection = DuckDBDialect._resolve_embedded_native_connection(
                config,
                sqlalchemy_engine,
                native_connection,
                open_new=lambda: CsvDialect._build_csv_memory_connection(config),
            )
            engine, owns_engine = DuckDBDialect._embedded_sqlalchemy_engine_for_connection(
                connection, "duckdb", sqlalchemy_engine
            )
            self._native_connection = connection
            self._owns_native_connection = owns_connection
            self._owns_sqlalchemy_engine = owns_engine
            super(DuckDBDialect, self).__init__(config, sqlalchemy_engine=engine)
            self._backend = DuckDBNativeBackend(connection)
            return
        except Exception as exc:
            debug(f"[CsvDialect.__init__] csv duckdb unavailable: {exc!r}")
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine, native_connection=native_connection)

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from CSV/Excel headers and inferred column types."""
        _ = sql_file
        if include == SchemaInclude.VIEWS:
            return tables_meta_to_schema_graph({}, object_kind=TableKind.VIEW)
        csv_config = cast(CsvRuntimeConfig, self.config)
        locked_by_table = CsvDialect._load_locked_column_types(self._schema_json_path)
        relations = CsvDialect._csv_relations_for_config(csv_config, schema_json_path=self._schema_json_path)
        tables_meta = CsvDialect._tables_meta_from_relations(
            relations, locked_by_table=locked_by_table, allow_objects=allow_objects
        )
        return tables_meta_to_schema_graph(tables_meta, object_kind=TableKind.TABLE)

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return a cache fingerprint from source file mtimes and content hashes."""
        _ = schema_context
        try:
            csv_config = cast(CsvRuntimeConfig, self.config)
            paths = csv_config.resolve_source_files()
            return sha256(CsvDialect._csv_source_probe_payload(paths))
        except Exception as exc:
            debug(f"[CsvDialect.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""


DialectRegistry.register_dialect("csv", CsvDialect, CsvRuntimeConfig)
