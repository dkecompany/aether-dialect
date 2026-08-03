"""Sqlglot-backed database engine dialects: MySQL, MariaDB, DuckDB, SQLite, Redshift, Snowflake, SQL Server, BigQuery, and Databricks."""

from __future__ import annotations

import csv
import hashlib
import importlib
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

import duckdb

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
    PolicyConfig,
    RedshiftRuntimeConfig,
    SnowflakeRuntimeConfig,
    SQLiteRuntimeConfig,
    SQLServerRuntimeConfig,
)
from ._constants import (
    BOOL_LITERALS,
    BQ_DEFAULT_PARTITION_LOOKBACK_DAYS,
    DUCKDB_PROFILING_SAMPLE_PREDICATE,
    INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL,
    MYSQL_PROFILING_SAMPLE_PREDICATE,
    REDSHIFT_PROFILING_SAMPLE_PREDICATE,
    SQL_BIND_TOKEN_RE,
    SQLITE_PROFILING_SAMPLE_PREDICATE,
    UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL,
    UNITY_INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL,
    UNITY_INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL,
    UNITY_INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL,
    UNITY_INFORMATION_SCHEMA_TABLES_TABLE_TYPE_SQL,
    UNITY_INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL,
    ResultReaderKind,
    is_upload_ingest_engine,
)
from ._contracts_base import (
    AccessError,
    DataQualityReport,
    DatabasePingFailed,
    EngineContext,
    SchemaInclude,
    SqlDiagnostic,
    SqlDiagnosticCode,
    StatementTimeoutError,
    where_leaves,
)
from ._contracts_core import RuntimeIntent
from ._contracts_schema import (
    CatalogStructuralConstraintsIndex,
    ColumnMetadata,
    CsvSourceSelection,
    SchemaDiff,
    SchemaGraph,
    UploadIngestResult,
)
from ._core_utils import (
    cost_cap_active,
    debug,
    diagnostic_debug_enabled,
    effective_explain_timeout_ms,
    engine_connect_likely_transient,
    progress,
    read_gzip_json,
    reconcile_execute_bind_params,
    sha256,
)
from ._data_quality import (
    PreparedRelation,
    parse_source_selections,
    pinned_names_from_schema_graph,
    prepare_relations_for_paths,
    validate_upload_sources,
)
from ._dialect import (
    Dialect,
    emit_via_ast,
    explain_cost_gate_violation,
    format_interval_unit,
    is_permission_denied_error,
    register_dialect,
    sqlglot_quote_identifier,
    sqlglot_quote_table_column,
    trace_finalize_render_stage,
    unit_to_approx_days,
)
from ._dialect_sqlglot_helper import (
    PartitionSqlAdapter,
    ResultBackend,
    SqlAlchemyResultBackend,
    SqlglotEngineDialect,
    SqlglotParseMixin,
    SqlServerResultBackend,
    append_required_partition_filter_guard,
    array_storage_kind,
    bigquery_diagnostics_from_dry_run,
    can_explain_for_backends,
    column_nullability_from_information_schema_rows,
    databricks_diagnostics_from_explain_text,
    databricks_plan_stats_from_explain_text,
    duckdb_diagnostics_from_explain_text,
    duckdb_root_plan_estimates,
    information_schema_connector_fetchall_dict_rows,
    information_schema_normalize_row,
    information_schema_spark_collect_normalized_dicts,
    inject_partition_predicates,
    mysql_diagnostics_from_explain_json,
    mysql_root_plan_estimates,
    normalize_datetrunc_sql,
    quoted_json_element_token_predicate,
    redshift_diagnostics_from_explain_text,
    redshift_root_plan_estimates,
    snowflake_diagnostics_from_explain_json,
    snowflake_root_plan_estimates,
    sqlite_diagnostics_from_query_plan,
    sqlite_structural_constraints_index,
    sqlserver_diagnostics_from_showplan_rows,
    sqlserver_diagnostics_from_showplan_xml,
    sqlserver_root_plan_estimates,
    structural_constraints_index_for_schema,
    structural_constraints_index_from_information_schema_rows,
)
from ._schema_build import (
    load_or_create_schema_bigquery,
    load_or_create_schema_duckdb,
    load_or_create_schema_mysql,
    load_or_create_schema_redshift,
    load_or_create_schema_snowflake,
    load_or_create_schema_sqlite,
    load_or_create_schema_sqlserver,
    tables_meta_to_schema_graph,
)
from ._schema_graph import (
    allow_objects_lower_set,
    assign_schema_graph_hashes,
    diff_schemas,
    load_schema_graph_snapshot,
    raise_if_schema_unusable,
)
from ._schema_overrides import (
    apply_diff,
    finalize_with_overrides,
    load_or_create_schema_databricks,
    migrate_sidecar_for_diff,
    save_schema_to_cache,
)
from ._sql_gen import databricks_unqualify_agg_arg_sql
from ._sql_to_intent import (
    BigQueryQueryLogSource,
    DatabricksQueryLogSource,
    MySQLQueryLogSource,
    NoOpQueryLogSource,
    RedshiftQueryLogSource,
    SnowflakeQueryLogSource,
    SQLServerQueryLogSource,
    databricks_plan_rows_from_explain_text,
)


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


def _arrow_table_to_tuples(table: Any) -> list[tuple[Any, ...]]:
    """Convert a PyArrow table to row tuples."""
    if table is None:
        return []
    num_cols = table.num_columns
    if num_cols == 0:
        return []
    columns = [table.column(i).to_pylist() for i in range(num_cols)]
    return list(zip(*columns, strict=True))


class DatabricksConnectorBackend(ResultBackend):
    """Databricks SQL warehouse connector cursor backend."""

    kind = "connector"

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        _ = params
        if timeout_ms is not None and cost_cap_active(timeout_ms):
            try:
                self._connection.set_query_timeout(int(timeout_ms) // 1000)
            except Exception:
                pass
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall() or []
            return [tuple(row) for row in rows]
        finally:
            cursor.close()


class DatabricksSparkBackend(ResultBackend):
    """PySpark / DatabricksSession SQL backend."""

    kind = "spark"

    def __init__(self, spark: Any) -> None:
        self._spark = spark

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        _ = params
        if timeout_ms is not None and cost_cap_active(timeout_ms):
            self._spark.conf.set("spark.databricks.sql.statementTimeout", f"{int(timeout_ms)}ms")
        tm_ex = effective_explain_timeout_ms()
        if tm_ex is not None and "EXPLAIN" in sql.upper():
            self._spark.conf.set("spark.databricks.sql.statementTimeout", f"{int(tm_ex)}ms")
        df = self._spark.sql(sql)
        return [tuple(row) for row in df.collect()]


class DatabricksSqlAlchemyBackend(SqlAlchemyResultBackend):
    """SQLAlchemy engine backend for Databricks warehouse URLs."""

    kind = "sqlalchemy"


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


def _bq_bind_params_from_sql(sql: str, params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only parameters referenced as ``@name`` tokens in *sql*."""
    if not params:
        return None
    bound = {key: val for key, val in params.items() if re.search(rf"@{re.escape(key)}\b", sql)}
    return bound or None


class BigQueryClientBackend(ResultBackend):
    """Google-cloud-bigquery Client query result backend."""

    kind = "bq_client"

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
            job_config.query_parameters = _bq_scalar_query_parameters(params)
        return job_config

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        if diagnostic_debug_enabled():
            debug(f"[{self._dialect_name}.execute] sql=\n{sql}")
        try:
            job = self._client.query(sql, job_config=self._job_config(timeout_ms=timeout_ms, params=params))
            rows = job.result()
            return [tuple(row.values()) for row in rows]
        except Exception as e:
            err = str(e)
            if is_permission_denied_error(err):
                raise AccessError("execute", err) from e
            raise


class BigQueryStorageBackend(ResultBackend):
    """BigQuery Storage API reader with client fallback."""

    kind = "bq_storage"

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

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        if diagnostic_debug_enabled():
            debug(f"[{self._client_backend._dialect_name}.execute] sql=\n{sql}")
        try:
            job = self._client_backend._client.query(
                sql, job_config=self._client_backend._job_config(timeout_ms=timeout_ms, params=params)
            )
            rows_iter = job.result(bqstorage_client=self._storage_client)
            return [tuple(row.values()) for row in rows_iter]
        except Exception:
            return self._client_backend.fetch_rows(sql, params, timeout_ms=timeout_ms)

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

    kind = "snowflake_arrow"

    def __init__(self, *, snowpark: Any | None = None, connection: Any | None = None) -> None:
        self._snowpark = snowpark
        self._connection = connection

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        _ = params
        if timeout_ms is not None and cost_cap_active(timeout_ms):
            secs = max(1, int(timeout_ms) // 1000)
            if self._snowpark is not None:
                self._snowpark.sql(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}").collect()
            elif self._connection is not None:
                cur = self._connection.cursor()
                try:
                    cur.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}")
                finally:
                    cur.close()
        if self._snowpark is not None:
            collected = self._snowpark.sql(sql).collect()
            out: list[tuple[Any, ...]] = []
            for row in collected:
                if hasattr(row, "__iter__") and not isinstance(row, (str, bytes, dict)):
                    out.append(tuple(row))
                elif hasattr(row, "asDict"):
                    out.append(tuple(row.asDict().values()))
                else:
                    out.append(row)
            return out
        if self._connection is None:
            raise RuntimeError("SnowflakeArrowBackend has no snowpark session or connection")
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql)
            if hasattr(cursor, "fetch_arrow_all"):
                try:
                    return _arrow_table_to_tuples(cursor.fetch_arrow_all())
                except Exception:
                    pass
            rows = cursor.fetchall() or []
            return [tuple(row) for row in rows]
        finally:
            cursor.close()

    def fetch_arrow_table(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> Any:
        _ = params
        if timeout_ms is not None and cost_cap_active(timeout_ms):
            secs = max(1, int(timeout_ms) // 1000)
            if self._snowpark is not None:
                self._snowpark.sql(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}").collect()
            elif self._connection is not None:
                cur = self._connection.cursor()
                try:
                    cur.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {secs}")
                finally:
                    cur.close()
        if self._connection is None:
            raise RuntimeError("SnowflakeArrowBackend has no connection for Arrow fetch")
        cursor = self._connection.cursor()
        try:
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
            except Exception:
                pass


class MySQLConnectorBackend(ResultBackend):
    """Pymysql DB-API backend for MySQL and MariaDB."""

    kind = "connector"

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        cursor = self._connection.cursor()
        try:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_ms)}")
            exec_params = reconcile_execute_bind_params(sql, params) or {}
            cursor.execute(sql, exec_params)
            rows = cursor.fetchall() or []
            return [tuple(row) for row in rows]
        finally:
            cursor.close()


class RedshiftConnectorBackend(ResultBackend):
    """redshift_connector DB-API backend."""

    kind = "connector"

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        cursor = self._connection.cursor()
        try:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                cursor.execute(f"SET statement_timeout = {int(timeout_ms)}")
            exec_params = reconcile_execute_bind_params(sql, params) or {}
            cursor.execute(sql, exec_params)
            rows = cursor.fetchall() or []
            return [tuple(row) for row in rows]
        finally:
            cursor.close()


class DuckDBNativeBackend(ResultBackend):
    """Native duckdb connection backend."""

    kind = "connector"

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        _ = timeout_ms
        bind_map = reconcile_execute_bind_params(sql, params) or {}
        if bind_map:
            result = self._connection.execute(sql, bind_map)
        else:
            result = self._connection.execute(sql)
        rows = result.fetchall() if hasattr(result, "fetchall") else result
        return [tuple(row) for row in (rows or [])]


class SQLiteNativeBackend(ResultBackend):
    """Stdlib sqlite3 connection backend."""

    kind = "connector"

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        _ = timeout_ms
        cursor = self._connection.cursor()
        try:
            exec_params = reconcile_execute_bind_params(sql, params) or {}
            if exec_params:
                cursor.execute(sql, exec_params)
            else:
                cursor.execute(sql)
            rows = cursor.fetchall() or []
            return [tuple(row) for row in rows]
        finally:
            cursor.close()


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


def _open_duckdb_connection(config: DuckDBRuntimeConfig, *, connection: Any | None = None) -> Any:
    """Return an existing or newly opened native duckdb connection."""
    if connection is not None:
        return connection

    return duckdb.connect(str(config.DATABASE_PATH or ":memory:"))


def _open_sqlite_connection(config: SQLiteRuntimeConfig, *, connection: Any | None = None) -> Any:
    """Return an existing or newly opened stdlib sqlite3 connection."""
    if connection is not None:
        return connection
    import sqlite3

    return sqlite3.connect(str(config.DATABASE_PATH or ":memory:"), check_same_thread=False)


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

    def execute(self, statement: Any, parameters: dict[str, Any] | None = None) -> _EmbeddedDuckDBResult:
        sql = statement.text if hasattr(statement, "text") else str(statement)
        params = parameters or {}
        if params:
            bound_sql, bound_params = _bind_colon_parameters(sql, params)
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


def _bind_colon_parameters(sql: str, parameters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Convert SQLAlchemy ``:name`` placeholders to duckdb positional binds."""
    ordered: list[Any] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in parameters:
            return match.group(0)
        ordered.append(parameters[key])
        return "?"

    bound_sql = re.sub(r":(\w+)", _replace, sql)
    return bound_sql, ordered


def create_duckdb_sqlalchemy_engine(connection: Any) -> Any:
    """Build a reflection engine over a single duckdb connection."""
    try:
        import duckdb as _duckdb
    except ImportError as exc:
        raise ConfigError("DuckDB requires the 'duckdb' package.") from exc
    _ = _duckdb
    return _EmbeddedDuckDBEngine(connection)


def create_sqlite_sqlalchemy_engine(connection: Any) -> Any:
    """Build a SQLAlchemy engine over a single sqlite3 connection."""
    return create_engine(
        "sqlite:///",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        future=True,
    )


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


def _resolve_embedded_native_connection(
    config: EngineRuntimeConfig, sqlalchemy_engine: Any | None, native_connection: Any | None, *, open_new: Any
) -> tuple[Any, bool]:
    """Resolve the single native connection for an embedded dialect."""
    if native_connection is not None:
        return native_connection, False
    attached = getattr(config, "NATIVE_CONNECTION", None)
    if attached is not None:
        return attached, False
    if sqlalchemy_engine is not None:
        pooled = extract_static_pool_connection(sqlalchemy_engine)
        if pooled is not None:
            return pooled, False
    return open_new(), True


def _embedded_sqlalchemy_engine_for_connection(
    connection: Any, engine_name: Literal["duckdb", "sqlite"], sqlalchemy_engine: Any | None
) -> tuple[Any, bool]:
    """Reuse or build the SQLAlchemy engine for an embedded native connection."""
    if sqlalchemy_engine is not None:
        pooled = extract_static_pool_connection(sqlalchemy_engine)
        if pooled is connection:
            return sqlalchemy_engine, False
    if engine_name == "duckdb":
        return create_duckdb_sqlalchemy_engine(connection), True
    return create_sqlite_sqlalchemy_engine(connection), True


class MySQLDialect(SqlglotEngineDialect):
    """MySQL dialect using sqlglot read=mysql and SQLAlchemy+pymysql execution."""

    name: str = "mysql"
    sqlglot_dialect: ClassVar[str] = "mysql"
    registry_canonical_rank: ClassVar[int] = 3
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Prefer pymysql connector backend with SQLAlchemy fallback."""
        self._native_connection: Any | None = None
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        self._select_native_backend()

    def _select_native_backend(self) -> None:
        """Attach pymysql connector when credentials are configured."""
        if not isinstance(self.config, MySQLRuntimeConfig):
            return
        if not self.config.PASSWORD or not self.config.DATABASE:
            return
        try:
            self._native_connection = _open_mysql_connector(self.config)
            self._backend = MySQLConnectorBackend(self._native_connection)
        except Exception as exc:
            debug(f"[MySQLDialect._select_native_backend] pymysql unavailable: {exc!r}")

    @property
    def result_reader_kind(
        self,
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active MySQL row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return "sqlalchemy"

    def can_explain(self) -> bool:
        """Return True when pymysql or SQLAlchemy can run EXPLAIN."""
        return can_explain_for_backends(self, sqlalchemy_engine=self.engine, native_connection=self._native_connection)

    def qualified_table_ref(self, table: str, kind: Literal["table", "view"] = "table") -> str:
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
        return structural_constraints_index_for_schema(
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
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from MySQL information_schema or SQL file fallback."""
        return load_or_create_schema_mysql(
            self.engine,
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
        diags = mysql_diagnostics_from_explain_json(payload, schema=schema)
        est_rows, est_bytes = mysql_root_plan_estimates(payload)
        return est_rows, est_bytes, diags, str(payload or "")

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return MySQL planner row estimate from ``EXPLAIN FORMAT=JSON``."""
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
        """Return MySQL ``MAX_EXECUTION_TIME`` session hint for profiling."""
        return f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_ms)}"

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
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render MySQL JSON array membership with case-insensitive element matching."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind in ("json_text_array", "native_array"):
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql, param_key=param_key, position_fn="INSTR", value_cast="CHAR"
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        norm_param = f"LOWER(TRIM(BOTH '%' FROM CAST(:{param_key} AS CHAR)))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS CHAR))"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render MySQL ``JSON_TABLE`` unnest for SELECT list."""
        sql = f"JSON_TABLE({column_sql}, '$[*]' COLUMNS({alias} TEXT PATH '$')) AS jt"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render MySQL date window boundaries with ``DATE_SUB`` / ``DATE_ADD``."""
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', CURRENT_DATE())"
        else:
            scaled, plural_unit = format_interval_unit(unit, amount)
            sql = f"{column} {op} DATE_SUB(CURRENT_DATE(), INTERVAL {scaled} {plural_unit})"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on MySQL."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return MySQL current timestamp or date for inclusive window upper bounds."""
        if unit in ("hour", "minute", "second"):
            return "CURRENT_TIMESTAMP"
        return "CURRENT_DATE"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a MySQL ``WHERE RAND(seed)`` suffix for statistics."""
        _ = table_kind
        if not use_sample:
            return ""
        ratio = max(0.0001, min(1.0, sample_size / max(row_count, 1)))
        return f"WHERE {MYSQL_PROFILING_SAMPLE_PREDICATE.format(ratio=ratio, seed=random_seed)}"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
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
        return inject_partition_predicates(adapter, sql, schema, intent)

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"GROUP_CONCAT({expr_sql} ORDER BY {order_by_sql} SEPARATOR {sep_sql})"
        return f"GROUP_CONCAT({expr_sql} SEPARATOR {sep_sql})"

    @property
    def supports_median(self) -> bool:
        return False

    def quote_string_literal(self, text: str) -> str:
        """Render a MySQL string literal with backslash and quote escaping."""
        s = str(text).replace("\\", "\\\\").replace("'", "''")
        return f"'{s}'"


class MariaDBDialect(MySQLDialect):
    """MariaDB dialect that reuses MySQL SQL generation, execution, and reflection."""

    name: str = "mariadb"
    sqlglot_dialect: ClassVar[str] = "mysql"
    registry_canonical_rank: ClassVar[int] = 4

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from MariaDB information_schema or SQL file fallback."""
        return load_or_create_schema_mysql(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
        )


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


class RedshiftDialect(SqlglotEngineDialect):
    """Redshift dialect using sqlglot read=redshift and SQLAlchemy execution."""

    name: str = "redshift"
    sqlglot_dialect: ClassVar[str] = "redshift"
    registry_canonical_rank: ClassVar[int] = 7
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

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
            self._native_connection = _open_redshift_connector(self.config)
            self._backend = RedshiftConnectorBackend(self._native_connection)
        except Exception as exc:
            debug(f"[RedshiftDialect._select_native_backend] redshift_connector unavailable: {exc!r}")

    @property
    def result_reader_kind(
        self,
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active Redshift row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return "sqlalchemy"

    def can_explain(self) -> bool:
        """Return True when redshift_connector or SQLAlchemy can run EXPLAIN."""
        return can_explain_for_backends(self, sqlalchemy_engine=self.engine, native_connection=self._native_connection)

    def qualified_table_ref(self, table: str, kind: Literal["table", "view"] = "table") -> str:
        """Return a double-quoted ``schema.table`` reference."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from ``information_schema`` and ``svv_foreign_keys``."""
        schema_name = self.schema_name()
        return structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=getattr(self, "_native_connection", None),
        )

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Normalize Redshift ``DATETRUNC`` emission to ``DATE_TRUNC``."""
        if stage != "post_substitute":
            return sql
        return normalize_datetrunc_sql(sql, sqlglot_dialect=self.sqlglot_dialect)

    @property
    def supports_ilike(self) -> bool:
        """Redshift supports ``ILIKE`` via Postgres-compatible syntax."""
        return True

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
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
        diags = redshift_diagnostics_from_explain_text(text_payload)
        est_rows, est_bytes = redshift_root_plan_estimates(text_payload)
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

    def query_log_source(self) -> Any | None:
        """Return the Redshift ``svl_qlog`` query-log source."""
        return RedshiftQueryLogSource()

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render Redshift interval date-difference comparison."""
        _ = minuend_sql, subtrahend_sql
        scaled, plural_unit = format_interval_unit(unit, amount)
        sql = f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render Redshift array membership, branching on column storage kind."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql, param_key=param_key, position_fn="STRPOS", value_cast="VARCHAR"
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        if kind == "native_array":
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql, param_key=param_key, position_fn="STRPOS", value_cast="VARCHAR"
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        norm_param = f"LOWER(BTRIM(CAST(:{param_key} AS VARCHAR), ' ' || CHR(34) || CHR(39)))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS VARCHAR))"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Redshift SUPER unnest via lateral alias."""
        sql = f"{column_sql} AS arr, arr.{alias} AS {alias}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render Redshift date window boundaries."""
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', CURRENT_DATE)"
        else:
            scaled, plural_unit = format_interval_unit(unit, amount)
            sql = f"{column} {op} CURRENT_DATE - INTERVAL '{scaled} {plural_unit}'"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Redshift."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return Redshift current timestamp or date for inclusive window upper bounds."""
        if unit in ("hour", "minute", "second"):
            return "CURRENT_TIMESTAMP"
        return "CURRENT_DATE"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a Redshift seeded hash-bucket ``WHERE`` suffix for statistics."""
        _ = table_kind
        if not use_sample:
            return ""
        ratio = max(0.0001, min(1.0, sample_size / max(row_count, 1)))
        return f"WHERE {REDSHIFT_PROFILING_SAMPLE_PREDICATE.format(ratio=ratio, seed=random_seed)}"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
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
        return inject_partition_predicates(adapter, sql, schema, intent, column_selector=_redshift_pruning_columns)


def _snowflake_cluster_columns(table_meta: Any) -> list[str]:
    """Return clustering column names for Snowflake partition predicate injection."""
    key = getattr(table_meta, "clustering_key", None)
    if key:
        return [str(key)]
    fields = getattr(table_meta, "clustering_fields", None) or []
    return [str(c) for c in fields if c]


class SnowflakeDialect(SqlglotEngineDialect):
    """Snowflake dialect using sqlglot read=snowflake and SQLAlchemy execution."""

    name: str = "snowflake"
    sqlglot_dialect: ClassVar[str] = "snowflake"
    registry_canonical_rank: ClassVar[int] = 9
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @property
    def supports_ilike(self) -> bool:
        """Return True because Snowflake exposes ``ILIKE``."""
        return True

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Open connector, SQLAlchemy, or Snowpark backends in priority order."""
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)
        self._snowpark_session: Any | None = None
        self._snowflake_connection: Any | None = None
        self._select_result_backend()

    def _select_result_backend(self) -> None:
        """Attach the active Snowflake row-fetch backend from connector, engine, or Snowpark."""
        config = cast(SnowflakeRuntimeConfig, self.config)
        if config.ACCOUNT and config.USER:
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
                return
            except Exception as exc:
                debug(f"[SnowflakeDialect.__init__] snowflake.connector unavailable: {exc!r}")
        if getattr(self, "engine", None) is not None:
            self._ensure_result_backend()
            if self._backend is not None:
                return
        if SnowflakeRuntimeConfig.snowpark_session_reachable():
            try:
                from snowflake.snowpark.context import get_active_session

                self._snowpark_session = get_active_session()
                self._backend = SnowflakeArrowBackend(snowpark=self._snowpark_session)
            except Exception as exc:
                debug(f"[SnowflakeDialect.__init__] Snowpark session unavailable: {exc!r}")

    @property
    def result_reader_kind(
        self,
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active Snowflake row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return "sqlalchemy"

    def quote_table_column(self, table: str, column: str) -> str:
        """Emit unquoted uppercase Snowflake identifiers by default."""
        return sqlglot_quote_table_column(table, column, self.sqlglot_dialect, quoted=False)

    def qualified_table_ref(self, table: str, kind: Literal["table", "view"] = "table") -> str:
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
        """Qualify bare tables with unquoted uppercase Snowflake three- part names."""
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
            table.set("this", sqlglot.exp.to_identifier(raw_name.upper(), quoted=False))
            table.set("db", sqlglot.exp.to_identifier(str(sch).upper(), quoted=False))
            if cat:
                table.set("catalog", sqlglot.exp.to_identifier(str(cat).upper(), quoted=False))
        try:
            return parsed.sql(dialect=self.sqlglot_dialect)
        except Exception:
            return sql

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from Snowflake ``information_schema``."""
        schema_name = self.schema_name()
        return structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=getattr(self, "_snowflake_connection", None),
        )

    def can_explain(self) -> bool:
        """Return True when Snowflake connector, Snowpark, or SQLAlchemy can run EXPLAIN."""
        return can_explain_for_backends(
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
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from Snowflake INFORMATION_SCHEMA or SQL file fallback."""
        return load_or_create_schema_snowflake(
            self.engine,
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
        diags = snowflake_diagnostics_from_explain_json(payload)
        est_rows, est_bytes = snowflake_root_plan_estimates(payload)
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
            if tm is not None and backend.kind == "sqlalchemy" and self.engine is not None:
                ms = int(tm)
                with self.engine.begin() as conn:
                    conn.execute(text(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {max(1, ms // 1000)}"))
            rows = backend.fetch_rows(explain_sql, params, timeout_ms=tm)
            est_rows, est_bytes, soft_diags, plan_text = self.parse_explain_plan(list(rows), schema=schema)
            if cost_cap_active(None) and (est_rows is not None or est_bytes is not None):
                failed, why = explain_cost_gate_violation(est_rows, est_bytes, dialect=self)
                if failed:
                    return (
                        False,
                        soft_diags + [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                        why,
                    )
            return True, soft_diags, plan_text
        except Exception as e:
            err = _format_result_backend_error(e)
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
        return inject_partition_predicates(adapter, sql, schema, intent, column_selector=_snowflake_cluster_columns)

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render Snowflake date comparison using ``DATEDIFF``. Snowflake ``DATEDIFF(part, start, end)`` returns an integer count, so a two-column difference must pass the columns as separate ``start``/``end`` arguments (``date - date`` is not a valid Snowflake interval). Falls back to comparing a single date column against ``CURRENT_DATE()``."""
        if minuend_sql and subtrahend_sql:
            sql = f"DATEDIFF('{unit}', {subtrahend_sql}, {minuend_sql}) {op} {amount}"
        else:
            sql = f"DATEDIFF('{unit}', {left_expr}, CURRENT_DATE()) {op} {amount}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render Snowflake array membership with trimmed, case- insensitive element comparison."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql, param_key=param_key, position_fn="STRPOS", value_cast="VARCHAR"
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        trim_set = "CONCAT(' ', CHR(34), CHR(39))"
        norm_bind = f"LOWER(TRIM(CAST(:{param_key} AS VARCHAR), {trim_set}))"
        xform = f"TRANSFORM({column_sql}, _ac_x -> LOWER(TRIM(CAST(_ac_x AS VARCHAR), {trim_set})))"
        sql = f"({column_sql} IS NOT NULL AND ARRAY_CONTAINS({norm_bind}::VARIANT, {xform}))"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Snowflake ``LATERAL FLATTEN`` unnest."""
        sql = f"LATERAL FLATTEN(INPUT => {column_sql}) {alias}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render Snowflake date window boundaries."""
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', CURRENT_DATE())"
        else:
            scaled, plural_unit = format_interval_unit(unit, amount)
            _ = plural_unit
            sql = f"{column} {op} DATEADD({unit}, -{scaled}, CURRENT_DATE())"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Snowflake."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return Snowflake current timestamp or date for inclusive window upper bounds."""
        if unit in ("hour", "minute", "second"):
            return "CURRENT_TIMESTAMP()"
        return "CURRENT_DATE()"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a Snowflake seeded ``SAMPLE`` suffix for statistics."""
        _ = row_count, table_kind
        if not use_sample:
            return ""
        pct = max(0.01, min(100.0, 100.0 * sample_size / max(row_count, 1)))
        return f"SAMPLE ({pct:.2f}) SEED ({random_seed})"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
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

    @property
    def supports_ilike(self) -> bool:
        """Return True because SQL Server exposes ``ILIKE`` (case- insensitive ``LIKE``)."""
        return True

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        if order_by_sql:
            return f"STRING_AGG({expr_sql}, {sep_sql}) WITHIN GROUP (ORDER BY {order_by_sql})"
        return f"STRING_AGG({expr_sql}, {sep_sql})"

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None):
        """Attach SQL Server ODBC backend and SHOWPLAN diagnose cache."""
        self._showplan_row_cache: dict[str, float | None] = {}
        super().__init__(config, sqlalchemy_engine=sqlalchemy_engine)

    def qualified_table_ref(self, table: str, kind: Literal["table", "view"] = "table") -> str:
        """Return a bracket-quoted ``schema.table`` reference."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from SQL Server ``information_schema``."""
        schema_name = self.schema_name()
        return structural_constraints_index_for_schema(self, schema_name, engine=getattr(self, "engine", None))

    def can_explain(self) -> bool:
        """Return True when SQLAlchemy or ODBC can run SHOWPLAN."""
        return can_explain_for_backends(self, sqlalchemy_engine=self.engine)

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
        include: SchemaInclude = "tables",
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
        diags = sqlserver_diagnostics_from_showplan_rows(rows)
        est_rows, est_bytes = sqlserver_root_plan_estimates(rows)
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
                    xml_diags = sqlserver_diagnostics_from_showplan_xml(xml_text)
                finally:
                    raw_xml.close()
            except Exception:
                xml_diags = []
            merged_diags = list(soft_diags) + xml_diags
            if cost_cap_active(None) and (est_rows is not None or est_bytes is not None):
                failed, why = explain_cost_gate_violation(est_rows, est_bytes, dialect=self)
                if failed:
                    return (
                        False,
                        merged_diags + [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                        why,
                    )
            cache_key = finalized.strip()
            if est_rows is not None:
                self._showplan_row_cache[cache_key] = est_rows
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
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a deterministic ordered row cap when T-SQL ``TABLESAMPLE`` is unseeded."""
        _ = row_count, random_seed, table_kind
        if not use_sample:
            return ""
        return self.profiling_ordered_limit_sample_suffix(sample_size)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
        """Ordered-limit sampling scans a deterministic subquery."""
        _ = table_kind
        return True

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via the active result backend, including ``sp_executesql`` batches."""
        backend = self.result_backend
        if backend is None:
            return super().execute(sql, params)
        tm = PolicyConfig.STATEMENT_TIMEOUT_MS if cost_cap_active(PolicyConfig.STATEMENT_TIMEOUT_MS) else None
        if params:
            return backend.fetch_rows("EXEC sp_executesql :stmt", {"stmt": sql}, timeout_ms=tm)
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
        return inject_partition_predicates(adapter, sql, schema, intent)

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
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_integer_days(self, base_sql: str, sign: str, offset_sql: str) -> str:
        """Render T-SQL date plus or minus an integer day count."""
        if sign == "+":
            sql = f"DATEADD(day, {offset_sql}, {base_sql})"
        else:
            sql = f"DATEADD(day, -({offset_sql}), {base_sql})"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render SQL Server JSON-array membership as a validator-safe scalar predicate. SQL Server has no scalar "array contains" function: ``OPENJSON`` is table-valued and requires a subquery / ``CROSS APPLY`` that the structural SQL validator forbids (no ``EXISTS`` / subqueries). Arrays are stored as JSON text, so membership is tested as a quoted-token match, which is precise for JSON string elements (the surrounding quotes prevent prefix false positives such as ``Trailers`` matching ``Trailers Extended``)."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind in ("json_text_array", "native_array"):
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql, param_key=param_key, position_fn="CHARINDEX", value_cast="NVARCHAR(MAX)"
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        norm_param = f"LOWER(LTRIM(RTRIM(CAST(:{param_key} AS NVARCHAR(MAX)))))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS NVARCHAR(MAX)))"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render SQL Server ``OPENJSON`` unnest for SELECT list."""
        sql = f"j.value AS {alias} FROM OPENJSON({column_sql}) j"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render T-SQL date window boundaries with ``DATEADD``."""
        if amount == 0:
            sql = f"{column} {op} DATEADD({unit}, DATEDIFF({unit}, 0, GETDATE()), 0)"
        else:
            scaled, plural_unit = format_interval_unit(unit, amount)
            _ = plural_unit
            sql = f"{column} {op} DATEADD({unit}, -{scaled}, GETDATE())"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on SQL Server."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return T-SQL current timestamp or date for inclusive window upper bounds."""
        if unit in ("hour", "minute", "second"):
            return "GETDATE()"
        return "CAST(GETDATE() AS DATE)"


def _bq_partition_columns(table_meta: Any) -> list[str]:
    """Return BigQuery partition column names for predicate injection."""
    return list(getattr(table_meta, "partition_columns", []) or [])


class BigQueryDialect(SqlglotEngineDialect):
    """BigQuery dialect using sqlglot read=bigquery and google-cloud- bigquery execution."""

    name: str = "bigquery"
    sqlglot_dialect: ClassVar[str] = "bigquery"
    registry_canonical_rank: ClassVar[int] = 10
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

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

    def _bq_job_limits(self) -> tuple[int | None, int | None]:
        """Return optional maximum bytes billed and job timeout for BigQuery jobs."""
        cap = PolicyConfig.MAX_QUERY_COST_BYTES
        max_bytes = int(cap) if cost_cap_active(cap) and cap is not None else None
        timeout_ms = (
            int(PolicyConfig.STATEMENT_TIMEOUT_MS)
            if cost_cap_active(PolicyConfig.STATEMENT_TIMEOUT_MS) and PolicyConfig.STATEMENT_TIMEOUT_MS is not None
            else None
        )
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
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active BigQuery row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return "sqlalchemy"

    def qualified_table_ref(self, table: str, kind: Literal["table", "view"] = "table") -> str:
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
        return can_explain_for_backends(
            self, sqlalchemy_engine=self.engine, native_connection=getattr(self, "_bq_client", None)
        )

    def apply_execute_cost_limits(self, target: Any) -> None:
        """Apply ``maximum_bytes_billed`` to a BigQuery query job config when configured."""
        cap = PolicyConfig.MAX_QUERY_COST_BYTES
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
        include: SchemaInclude = "tables",
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
            failed, why = explain_cost_gate_violation(None, est_bytes, dialect=self)
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
            soft_diags = bigquery_diagnostics_from_dry_run(
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
        bind = _bq_bind_params_from_sql(sql, params)
        tm = PolicyConfig.STATEMENT_TIMEOUT_MS if cost_cap_active(PolicyConfig.STATEMENT_TIMEOUT_MS) else None
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
        pruned = inject_partition_predicates(adapter, sql, schema, intent, column_selector=_bq_partition_columns)

        def _default_guard(table_name: str, part_col: str) -> str:
            qual = self.quote_table_column(table_name, part_col)
            return f"{qual} >= DATE_SUB(CURRENT_DATE(), INTERVAL {BQ_DEFAULT_PARTITION_LOOKBACK_DAYS} DAY)"

        out = append_required_partition_filter_guard(
            pruned,
            schema=schema,
            intent=intent,
            sqlglot_dialect=self.sqlglot_dialect,
            column_selector=_bq_partition_columns,
            default_predicate_sql=_default_guard,
            intent_equality_for_column=self._partition_predicate_from_intent,
        )
        trace_finalize_render_stage("inject_pruning_predicates", sql, out)
        return out

    def _partition_predicate_from_intent(self, intent: RuntimeIntent, table_name: str, part_col: str) -> str | None:
        for fp in where_leaves(intent.where) or []:
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
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render BigQuery array membership as a scalar JSON text match."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            norm_param = f"LOWER(TRIM(@{param_key}))"
            needle = f"CONCAT('\"', {norm_param}, '\"')"
            sql = f"STRPOS(LOWER(CAST({column_sql} AS STRING)), {needle}) > 0"
            return sql
        needle = f"CONCAT('\"', LOWER(@{param_key}), '\"')"
        sql = f"STRPOS(LOWER(TO_JSON_STRING({column_sql})), {needle}) > 0"
        return sql

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render BigQuery ``UNNEST`` for SELECT list."""
        sql = f"UNNEST({column_sql}) AS {alias}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render BigQuery date window boundaries. The column is normalized with ``DATE(...)`` so a ``TIMESTAMP`` column compares cleanly against the ``DATE``-typed boundary; BigQuery does not implicitly coerce ``TIMESTAMP`` and ``DATE`` operands."""
        col = f"DATE({column})"
        if amount == 0:
            sql = f"{col} {op} DATE_TRUNC(CURRENT_DATE(), {unit.upper()})"
        else:
            scaled, plural_unit = format_interval_unit(unit, amount)
            _ = plural_unit
            sql = f"{col} {op} DATE_SUB(CURRENT_DATE(), INTERVAL {scaled} {unit.upper()})"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window_inclusive_upper(self, left_rendered: str, unit: str) -> str:
        """Render BigQuery inclusive upper bound with the same DATE cast as the lower bound."""
        anchor = self.date_window_upper_bound_sql(unit)
        return f"DATE({left_rendered}) <= {anchor}"

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on BigQuery."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return BigQuery current timestamp or date for inclusive window upper bounds."""
        if unit in ("hour", "minute", "second"):
            return "CURRENT_TIMESTAMP()"
        return "CURRENT_DATE()"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a deterministic ordered row cap when BigQuery ``TABLESAMPLE`` is unseeded."""
        _ = row_count, random_seed, table_kind
        if not use_sample:
            return ""
        return self.profiling_ordered_limit_sample_suffix(sample_size)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
        """Ordered-limit sampling scans a deterministic subquery."""
        _ = table_kind
        return True


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


def databricks_normalize_datetrunc_sql(sql: str) -> str:
    """Rewrite parsed ``Anonymous`` ``DATETRUNC`` call sites and legacy ``DATEADD`` tokens for Spark emission."""
    out = normalize_datetrunc_sql(sql, sqlglot_dialect="databricks")
    out = re.sub(r"\bDATEADD\s*\(", "date_columndd(", out, flags=re.IGNORECASE)
    return out


class DatabricksDialect(SqlglotParseMixin, Dialect):
    """Databricks / Spark SQL dialect using EXPLAIN and optional native SQL connector."""

    name: str = "databricks"
    sqlglot_dialect: ClassVar[str] = "databricks"
    registry_canonical_rank: ClassVar[int] = 8
    registry_native_backend: ClassVar[bool] = True
    registry_structural_index: ClassVar[bool] = True
    registry_qualified_table_ref: ClassVar[bool] = True

    @property
    def supports_ilike(self) -> bool:
        """Return True because Spark SQL exposes ``ILIKE``."""
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
        """Open a native Databricks SQL connection or fall back to a. PySpark session. When warehouse credentials are configured (``server_hostname``, ``http_path``, ``access_token``), the ``databricks-sql-connector`` is preferred.  If the connector import or connection attempt fails, the dialect falls back to a cluster-local ``SparkSession``.  A ``RuntimeError`` is raised only when **neither** backend can be established."""
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
                url = db_config.sqlalchemy_url()
                if url:
                    try:
                        self.engine = create_engine(url, future=True)
                    except Exception as exc:
                        debug(f"[DatabricksDialect.__init__] SQLAlchemy engine not created: {exc}")
                debug("[DatabricksDialect.__init__] using databricks-sql-connector (warehouse)")
                self._select_result_backend()
                return

            msg = (
                "databricks-sql-connector failed to open a warehouse session "
                f"({connector_error}). Verify the warehouse is reachable and the "
                "access token is valid; warehouses can take several minutes to "
                "cold-start."
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
                return
            self._init_spark_fallback(connector_error)
            self._select_result_backend()

    def _init_spark_fallback(self, connector_error: str | None) -> None:
        """Attempt to initialise a Spark session as the execution. backend. Tries ``databricks.connect.DatabricksSession`` first to honour the installed ``databricks-connect`` build of ``pyspark``, which hard-rejects ``SparkSession.builder.getOrCreate()``. Falls back to ``pyspark.sql.SparkSession`` only when ``databricks.connect`` is not importable. Raises:class:`ConfigError` with the canonical missing-credential hint when neither path yields a session."""
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
                return information_schema_connector_fetchall_dict_rows(cur, sql)
        if isinstance(backend, DatabricksSparkBackend) and self.spark is not None:
            return information_schema_spark_collect_normalized_dicts(self.spark, sql)
        if isinstance(backend, DatabricksSqlAlchemyBackend) and self.engine is not None:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                if not result.cursor.description:
                    return []
                col_names = [d[0] for d in result.cursor.description]
                return [
                    information_schema_normalize_row(dict(zip(col_names, row, strict=True)))
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
        tm = PolicyConfig.STATEMENT_TIMEOUT_MS if cost_cap_active(PolicyConfig.STATEMENT_TIMEOUT_MS) else None
        return backend.fetch_rows(sql, params, timeout_ms=tm)

    def _collect_explain_text(self, explain_sql: str) -> str:
        """Run an EXPLAIN statement and return newline-joined first- column text."""
        return self._backend.fetch_first_column_text(explain_sql) if self._backend is not None else ""

    def _explain_result_from_text(self, text_payload: str) -> tuple[bool, list[SqlDiagnostic], str]:
        """Parse EXPLAIN text into cost-gate outcome and soft diagnostics."""
        er, eb = databricks_plan_stats_from_explain_text(text_payload)
        failed, why = explain_cost_gate_violation(er, eb, dialect=self)
        if failed:
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)], why)
        return True, databricks_diagnostics_from_explain_text(text_payload), ""

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run Spark/Databricks ``EXPLAIN`` and return ``(ok, diagnostics, raw_message)``. ``ok`` is False only on hard validation failures. A permission- denied error disables EXPLAIN for the remainder of this dialect instance and is reported as ``ok=True`` with no diagnostics so the caller can proceed without treating missing privileges as invalid SQL. Soft plan- shape findings (suspected cartesian joins, zero-row estimates) are emitted as :class:`SqlDiagnostic` entries with codes from ``SOFT_DIAGNOSTIC_CODES`` in ``_config`` so callers may apply confidence penalties without rejecting the SQL."""
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

    def qualified_table_ref(self, table: str, kind: Literal["table", "view"] = "table") -> str:
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
        return databricks_normalize_datetrunc_sql(sql)

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
        return can_explain_for_backends(
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
            format_literal=_dbr_format_partition_literal,
            sqlglot_dialect=self.sqlglot_dialect,
        )
        return inject_partition_predicates(adapter, sql, schema, intent)

    @property
    def result_reader_kind(
        self,
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active Databricks row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        if getattr(self, "engine", None) is not None:
            return "sqlalchemy"
        if getattr(self, "connection", None) is not None:
            return "connector"
        return "spark"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return Spark current timestamp or date for inclusive window upper bounds."""
        if unit in ("hour", "minute", "second"):
            return "current_timestamp()"
        return "current_date()"

    def normalize_window_agg_sql_frag(self, frag: str) -> str:
        """Unqualify aggregate arguments inside window functions for Spark SQL."""
        return databricks_unqualify_agg_arg_sql(frag)

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return Databricks planner row estimate from ``EXPLAIN COST``."""
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            explain_sql = f"EXPLAIN COST {finalized}"
            text_payload = self._collect_explain_text(explain_sql)
            return databricks_plan_rows_from_explain_text(text_payload)
        except Exception:
            return None

    def query_log_source(self) -> Any | None:
        """Return the Databricks ``system.query.history`` query-log source."""
        return DatabricksQueryLogSource()

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute finalized Spark SQL via the warehouse connector or. Spark session."""
        _ = params
        if diagnostic_debug_enabled():
            debug(f"[DatabricksDialect.execute] sql=\n{sql}")

        try:
            if self._backend is None:
                raise RuntimeError("DatabricksDialect has no result backend")
            return self._collect_rows(sql, params)
        except Exception as e:
            err = str(e)
            if is_permission_denied_error(err):
                raise AccessError("execute", err) from e
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
            idx = structural_constraints_index_from_information_schema_rows(t_rows, k_rows, r_rows)
            idx.column_nullability = column_nullability_from_information_schema_rows(c_rows)
            return idx
        except Exception as exc:
            debug(f"[dialect.DatabricksDialect.structural_constraints_index] failed: {exc!r}")
            return CatalogStructuralConstraintsIndex.empty()

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
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
        """Return SHA-256 over ``information_schema.columns`` rows for the configured catalog and schema on Databricks. Tries the SQL connector first, then falls back to a Spark session. Always returns ``""`` rather than raising so build_schema_graph degrades to the legacy fingerprint validation when the probe cannot run."""
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
        self, table_name: str, col_name: str, *, table_kind: Literal["table", "view"] = "table"
    ) -> tuple[int, int, float] | None:
        """Run full-table statistics for PK inference after sampled. profiling."""
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
        """Render Spark ``DATEDIFF``-based date-difference comparison. When *minuend_sql* and *subtrahend_sql* are available the method emits ``DATEDIFF(minuend, subtrahend)`` which returns an integer, avoiding the INTERVAL-vs-INT type mismatch that raw date subtraction causes on Databricks/Spark."""
        days = unit_to_approx_days(unit, amount)
        if minuend_sql and subtrahend_sql:
            sql = f"DATEDIFF({minuend_sql}, {subtrahend_sql}) {op} {days}"
        else:
            sql = f"({left_expr}) {op} {days}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render Databricks array membership with trimmed element. comparison."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql, param_key=param_key, position_fn="STRPOS", value_cast="STRING"
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        trim_set = "CONCAT(' ', chr(34), chr(39))"
        norm_bind = f"LOWER(TRIM(CAST(:{param_key} AS STRING), {trim_set}))"
        xform = f"TRANSFORM({column_sql}, _ac_x -> LOWER(TRIM(CAST(_ac_x AS STRING), {trim_set})))"
        sql = f"({column_sql} IS NOT NULL AND ARRAY_CONTAINS({xform}, {norm_bind}))"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render Spark ``EXPLODE`` for SELECT list."""
        sql = f"EXPLODE({column_sql}) AS {alias}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    @property
    def supports_unnest_select_item(self) -> bool:
        """Return True because Spark ``EXPLODE`` is valid as a SELECT- list generator."""
        return True

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render Spark date window boundaries."""
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
            scaled, plural_unit = format_interval_unit(unit, amount)
            sql = f"{column} {op} (current_timestamp() - INTERVAL '{scaled} {plural_unit}')"
        else:
            sql = f"{column} {op} date_sub(current_date(), {amount})"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive comparison on Spark."""
        return f"LOWER(TRIM({expr}))"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a Spark ``TABLESAMPLE … REPEATABLE`` suffix for statistics."""
        _ = table_kind
        if not use_sample:
            return ""
        pct = max(0.01, min(100.0, 100.0 * sample_size / max(row_count, 1)))
        return f"TABLESAMPLE ({pct:.2f} PERCENT) REPEATABLE ({random_seed})"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
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

    def render_median(self, expr_sql: str) -> str:
        return f"median({expr_sql})"

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
                connection, owns_connection = _resolve_embedded_native_connection(
                    duckdb_config,
                    sqlalchemy_engine,
                    native_connection,
                    open_new=lambda: _open_duckdb_connection(duckdb_config),
                )
                engine, owns_engine = _embedded_sqlalchemy_engine_for_connection(
                    connection, "duckdb", sqlalchemy_engine
                )
                self._native_connection = connection
                self._owns_native_connection = owns_connection
                self._owns_sqlalchemy_engine = owns_engine
                super().__init__(config, sqlalchemy_engine=engine)
                self._backend = DuckDBNativeBackend(connection)
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
            self._native_connection = _open_duckdb_connection(duck_config)
            self._owns_native_connection = True
            self._backend = DuckDBNativeBackend(self._native_connection)
        except Exception as exc:
            debug(f"[DuckDBDialect._select_native_backend] duckdb unavailable: {exc!r}")

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
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active DuckDB row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return "sqlalchemy"

    def can_explain(self) -> bool:
        """Return True when native duckdb or SQLAlchemy can run EXPLAIN."""
        return can_explain_for_backends(self, sqlalchemy_engine=self.engine, native_connection=self._native_connection)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load PK, FK, and UNIQUE metadata from DuckDB ``information_schema``."""
        schema_name = self.schema_name()
        return structural_constraints_index_for_schema(
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
        return normalize_datetrunc_sql(sql, sqlglot_dialect=self.sqlglot_dialect)

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
        return inject_partition_predicates(adapter, sql, schema, intent)

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
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from DuckDB reflection or SQL file fallback."""
        return load_or_create_schema_duckdb(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
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
        diags = duckdb_diagnostics_from_explain_text(plan_text)
        est_rows, est_bytes = duckdb_root_plan_estimates(plan_text)
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
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render DuckDB list membership with case-insensitive comparison."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql,
                param_key=param_key,
                position_fn="INSTR",
                value_cast="VARCHAR",
                needle_style="sqlite_pipe",
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        norm_param = f"LOWER(TRIM(BOTH '%' FROM :{param_key}))"
        sql = f"list_contains(list_transform({column_sql}, x -> lower(x)), {norm_param})"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render DuckDB UNNEST for a SELECT list item."""
        sql = f"UNNEST({column_sql}) AS {alias}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render a DuckDB relative date-window boundary using DATE_TRUNC and INTERVAL."""
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', current_date)"
        else:
            scaled, plural = format_interval_unit(unit, amount)
            sql = f"{column} {op} (current_date - INTERVAL '{scaled} {plural}')"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression in LOWER(...) for case-insensitive comparison."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return the inclusive upper-bound timestamp expression for DuckDB."""
        if unit in ("hour", "minute", "second"):
            return "current_timestamp"
        return "current_date"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a DuckDB seeded ``USING SAMPLE`` suffix when sampling, else an empty suffix."""
        _ = table_kind
        if not use_sample or row_count <= 0:
            return ""
        pct = max(min(100.0 * float(sample_size) / float(row_count), 100.0), 0.0001)
        return DUCKDB_PROFILING_SAMPLE_PREDICATE.format(pct=pct, seed=random_seed)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
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

    @property
    def supports_ordered_string_agg(self) -> bool:
        return False

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        return f"GROUP_CONCAT({expr_sql})"

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
                connection, owns_connection = _resolve_embedded_native_connection(
                    sqlite_config,
                    sqlalchemy_engine,
                    native_connection,
                    open_new=lambda: _open_sqlite_connection(sqlite_config),
                )
                engine, owns_engine = _embedded_sqlalchemy_engine_for_connection(
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
            self._native_connection = _open_sqlite_connection(sqlite_config)
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
    ) -> Literal["sqlalchemy", "spark", "connector", "bq_client", "bq_storage", "snowflake_arrow"]:
        """Return the active SQLite row-fetch backend kind."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return cast(ResultReaderKind, backend.kind)
        return "sqlalchemy"

    def can_explain(self) -> bool:
        """Return True when sqlite3 or SQLAlchemy can run EXPLAIN QUERY PLAN."""
        return can_explain_for_backends(self, sqlalchemy_engine=self.engine, native_connection=self._native_connection)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Load FK metadata from ``PRAGMA foreign_key_list`` when foreign keys are enabled."""
        return sqlite_structural_constraints_index(getattr(self, "engine", None))

    def query_log_source(self) -> Any | None:
        """Return a documented no-op query-log source for SQLite."""
        return NoOpQueryLogSource()

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return SQLite CAST(... AS TEXT) for overlap sampling."""
        return f"CAST({expr} AS TEXT)"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = "tables",
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
        diags = sqlite_diagnostics_from_query_plan(plan_text)
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
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_contains(
        self, column_sql: str, param_key: str, *, column_meta: ColumnMetadata | None = None
    ) -> str:
        """Render SQLite JSON array membership as a validator-safe scalar predicate."""
        kind = array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind in ("json_text_array", "native_array"):
            sql = quoted_json_element_token_predicate(
                column_sql=column_sql,
                param_key=param_key,
                position_fn="INSTR",
                value_cast="TEXT",
                needle_style="sqlite_pipe",
            )
            return emit_via_ast(sql, self.sqlglot_dialect)
        norm_param = f"LOWER(TRIM(CAST(:{param_key} AS TEXT), '\"''))"
        sql = f"{norm_param} = LOWER(CAST({column_sql} AS TEXT))"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render SQLite json_each as the unnest source for a SELECT list item."""
        sql = f"json_each({column_sql}) AS {alias}"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_date_window(self, column: str, op: str, unit: str, amount: int) -> str:
        """Render a SQLite relative date-window boundary using the date modifier syntax."""
        if amount == 0:
            sql = f"{column} {op} date('now')"
        else:
            scaled, plural = format_interval_unit(unit, amount)
            sql = f"{column} {op} date('now', '-{scaled} {plural}')"
        return emit_via_ast(sql, self.sqlglot_dialect)

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression in LOWER(...) for case-insensitive comparison."""
        return f"LOWER({expr})"

    def date_window_upper_bound_sql(self, unit: str) -> str:
        """Return the inclusive upper-bound expression for SQLite."""
        if unit in ("hour", "minute", "second"):
            return "datetime('now')"
        return "date('now')"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a SQLite seeded hash-bucket ``WHERE`` suffix when sampling."""
        _ = table_kind
        if not use_sample:
            return ""
        ratio = max(0.0001, min(1.0, sample_size / max(row_count, 1)))
        return f"WHERE {SQLITE_PROFILING_SAMPLE_PREDICATE.format(ratio=ratio, seed=random_seed)}"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: Literal["table", "view"] = "table") -> bool:
        """SQLite samples via a ``WHERE random()`` predicate inside a subquery."""
        _ = table_kind
        return True


register_dialect("mysql", MySQLDialect, MySQLRuntimeConfig)
register_dialect("mariadb", MariaDBDialect, MariaDBRuntimeConfig)
register_dialect("duckdb", DuckDBDialect, DuckDBRuntimeConfig)
register_dialect("sqlite", SQLiteDialect, SQLiteRuntimeConfig)
register_dialect("redshift", RedshiftDialect, RedshiftRuntimeConfig)
register_dialect("snowflake", SnowflakeDialect, SnowflakeRuntimeConfig)
register_dialect("sqlserver", SQLServerDialect, SQLServerRuntimeConfig)
register_dialect("bigquery", BigQueryDialect, BigQueryRuntimeConfig)
register_dialect("databricks", DatabricksDialect, DatabricksRuntimeConfig)


_quote_ident = sqlglot_quote_identifier


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


def _read_csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            row = next(reader)
        except StopIteration as exc:
            raise ConfigError(f"csv file is empty: {path}") from exc
    return _normalize_header_columns(row, source=path)


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ConfigError(f"csv file missing header row: {path}")
        columns = _normalize_header_columns(reader.fieldnames, source=path)
        for raw in reader:
            yield {col: str(raw.get(col) or "") for col in columns}


def _read_xlsx_header(path: Path) -> list[str]:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        row = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True))
    except StopIteration as exc:
        raise ConfigError(f"csv file is empty: {path}") from exc
    finally:
        workbook.close()
    headers = ["" if cell is None else str(cell) for cell in row]
    return _normalize_header_columns(headers, source=path)


def _iter_xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        row_iter = worksheet.iter_rows(values_only=True)
        try:
            header_row = next(row_iter)
        except StopIteration as exc:
            raise ConfigError(f"csv file is empty: {path}") from exc
        columns = _normalize_header_columns(["" if cell is None else str(cell) for cell in header_row], source=path)
        for values in row_iter:
            if values is None:
                continue
            row_map = {
                col: "" if idx >= len(values) or values[idx] is None else str(values[idx])
                for idx, col in enumerate(columns)
            }
            if any(str(v).strip() for v in row_map.values()):
                yield row_map
    finally:
        workbook.close()


def _read_source_header(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_header(path)
    if suffix == ".xlsx":
        return _read_xlsx_header(path)
    raise ConfigError(f"csv unsupported file type: {path}")


def _iter_source_rows(path: Path) -> Iterator[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _iter_csv_rows(path)
    elif suffix == ".xlsx":
        yield from _iter_xlsx_rows(path)
    else:
        raise ConfigError(f"csv unsupported file type: {path}")


def _looks_boolean(value: str) -> bool:
    return value.strip().lower() in BOOL_LITERALS


def _looks_integer(value: str) -> bool:
    text = value.strip()
    if not text or "." in text or "e" in text.lower():
        return False
    int(text)
    return True


def _looks_number(value: str) -> bool:
    float(value.strip())
    return True


def _infer_duckdb_column_type(samples: Sequence[str]) -> str:
    non_empty = [str(v).strip() for v in samples if str(v).strip()]
    if not non_empty:
        return "VARCHAR"
    if all(_looks_boolean(v) for v in non_empty):
        return "BOOLEAN"
    try:
        if all(_looks_integer(v) for v in non_empty):
            return "INTEGER"
    except ValueError:
        pass
    try:
        if all(_looks_number(v) for v in non_empty):
            return "DOUBLE"
    except ValueError:
        pass
    return "VARCHAR"


def _coerce_typed_cell(value: str | None, duckdb_type: str) -> object:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    upper = duckdb_type.upper()
    if "BOOL" in upper:
        return text.lower() in ("1", "true", "t", "yes")
    if any(token in upper for token in ("INT", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT")):
        return int(text)
    if any(token in upper for token in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL")):
        return float(text)
    return text


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


def _column_types_for_source(
    path: Path, *, locked_types: Mapping[str, str] | None = None, sample_limit: int = 512
) -> tuple[list[str], list[str]]:
    columns = _read_source_header(path)
    locked = dict(locked_types or {})
    samples: dict[str, list[str]] = {col: [] for col in columns}
    for row in _iter_source_rows(path):
        for col in columns:
            bucket = samples[col]
            if len(bucket) >= sample_limit:
                continue
            bucket.append(row.get(col, ""))
    types: list[str] = []
    for col in columns:
        locked_type = locked.get(col)
        types.append(locked_type if locked_type else _infer_duckdb_column_type(samples[col]))
    return columns, types


def _csv_schema_pins(schema_json_path: str | None = None) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    cached = load_schema_graph_snapshot(schema_json_path or EngineConfig.SCHEMA_JSON_PATH)
    return pinned_names_from_schema_graph(cached)


def _csv_relations_for_config(
    config: CsvRuntimeConfig, *, schema_json_path: str | None = None
) -> list[PreparedRelation]:
    paths = config.resolve_source_files()
    table_pins, column_pins = _csv_schema_pins(schema_json_path)
    return prepare_relations_for_paths(
        paths,
        pinned_table_names=table_pins,
        pinned_column_names=column_pins,
        apply_auto_correct=True,
        source_selections=parse_source_selections(config.SOURCE_SELECTIONS),
    )


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
        tables_meta[relation.relation_name] = {
            "column_names_original": list(relation.columns),
            "original_column_labels": list(relation.original_column_labels),
            "original_table_label": relation.original_table_label,
            "column_types": types,
            "column_is_nullable": [True] * len(relation.columns),
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
        }
    return tables_meta


def _load_prepared_relation_into_connection(connection: Any, relation: PreparedRelation) -> None:
    locked_by_table = _load_locked_column_types(EngineConfig.SCHEMA_JSON_PATH)
    locked_types = locked_by_table.get(relation.relation_name.lower(), {})
    types: list[str] = []
    for idx, col in enumerate(relation.columns):
        locked_type = locked_types.get(col)
        types.append(locked_type if locked_type else relation.column_types[idx])
    connection.execute(f"DROP TABLE IF EXISTS {_quote_ident(relation.relation_name)}")
    connection.execute(_create_table_sql(relation.relation_name, relation.columns, types))
    _insert_rows(connection, relation.relation_name, relation.columns, types, list(relation.rows))


def load_prepared_relation_into_native_connection(connection: Any, relation: PreparedRelation) -> None:
    """Materialise one validated upload relation into an embedded native connection."""
    _load_prepared_relation_into_connection(connection, relation)


def _normalize_upload_paths(paths: Sequence[str | os.PathLike[str]]) -> tuple[Path, ...]:
    return tuple(Path(os.fspath(path)) for path in paths)


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
    return confirmed


def _confirmed_upload_selections_payload(
    selections: Mapping[str, CsvSourceSelection],
) -> dict[str, dict[str, Any]]:
    return {name: _upload_selection_to_confirmed_dict(selection) for name, selection in selections.items()}


def _resolve_upload_engine_type(engine: Any) -> str:
    engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
    if engine_type:
        return engine_type
    runtime_cfg = getattr(engine, "_runtime_config", None)
    return str(getattr(runtime_cfg, "engine", "") or "").strip().lower()


def _resolve_upload_native_connection(engine: Any) -> Any:
    dialect = getattr(engine, "_dialect", None)
    connection = getattr(dialect, "_native_connection", None) if dialect is not None else None
    if connection is None:
        connection = getattr(engine, "_native_connection", None)
    if connection is None:
        raise ConfigError("ingest_upload_sources requires an embedded engine with a native connection")
    return connection


def _upload_validation_config_error(message: str, data_quality_report: object) -> ConfigError:
    exc = ConfigError(message)
    setattr(exc, "data_quality_report", data_quality_report)
    return exc


def _sync_upload_engine_config_globals(engine: Any) -> None:
    engine_type = _resolve_upload_engine_type(engine)
    EngineConfig.TYPE = engine_type
    if engine_type == "duckdb":
        EngineConfig.RUNTIME = DuckDBRuntimeConfig
    elif engine_type == "csv":
        EngineConfig.RUNTIME = CsvRuntimeConfig
    else:
        raise ConfigError(f"ingest_upload_sources requires a duckdb or csv engine member, got {engine_type!r}")


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
    _sync_upload_engine_config_globals(engine)
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
        schema_role=str(getattr(engine, "_schema_role", "owner") or "owner"),
    )
    save_schema_to_cache(cached_sg, schema_json_path)
    migrate_sidecar_for_diff(schema_json_path, schema_diff)
    finalize_with_overrides(cached_sg, schema_json_path, dialect=dialect)
    return cached_sg, schema_diff if not schema_diff.is_empty else None


def ingest_upload_sources_into_engine(
    engine: Any,
    paths: Sequence[str | os.PathLike[str]],
    *,
    source_selections: Mapping[str, CsvSourceSelection | Mapping[str, Any]] | None = None,
    relation_names: Mapping[str, str] | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> UploadIngestResult:
    """Validate uploads, materialise relations into *engine*, and return schema delta info."""
    engine_type = _resolve_upload_engine_type(engine)
    if not is_upload_ingest_engine(engine_type):
        raise ConfigError(
            f"ingest_upload_sources requires a duckdb or csv engine member, got {engine_type!r}",
        )
    resolved_paths = _normalize_upload_paths(paths)
    if not resolved_paths:
        raise ConfigError("ingest_upload_sources requires at least one upload path")
    selections = _normalize_upload_selections(source_selections)
    selection_by_name = {path.name: selections[path.name] for path in resolved_paths if path.name in selections}
    report = validate_upload_sources(
        resolved_paths,
        log_sink=log_sink,
        source_selections=selection_by_name,
    )
    if report.requires_review and not selections:
        raise _upload_validation_config_error(
            f"{report.narrative} "
            "Call inspect_tabular_upload and pass source_selections with the accepted interpretation.",
            report,
        )
    if not report.ok:
        raise _upload_validation_config_error(report.narrative, report)
    if selections:
        report = DataQualityReport(
            ok=report.ok,
            issues=report.issues,
            narrative=report.narrative,
            suggested_selections=report.suggested_selections,
            confirmed_selections=_confirmed_upload_selections_payload(selections),
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
    _sync_upload_engine_config_globals(engine)
    connection = _resolve_upload_native_connection(engine)
    for relation in relations:
        load_prepared_relation_into_native_connection(connection, relation)
    relation_names_created = tuple(relation.relation_name for relation in relations)
    updated_schema, schema_diff = _refresh_engine_schema_after_ingest(engine)
    engine._schema_graph = updated_schema
    engine._schema_stats = updated_schema.refresh_schema_stats()
    engine._data_quality_report = report
    return UploadIngestResult(
        relation_names=relation_names_created,
        report=report,
        schema_diff=schema_diff,
    )


def _build_csv_memory_connection(config: CsvRuntimeConfig) -> Any:
    duckdb = importlib.import_module("duckdb")
    connection = duckdb.connect(":memory:")
    for relation in _csv_relations_for_config(config):
        _load_prepared_relation_into_connection(connection, relation)
    return connection


def _create_table_sql(table: str, columns: Sequence[str], types: Sequence[str]) -> str:
    col_defs = ", ".join(f"{_quote_ident(col)} {duckdb_type}" for col, duckdb_type in zip(columns, types, strict=True))
    return f"CREATE TABLE {_quote_ident(table)} ({col_defs})"


def _insert_rows(
    connection: Any, table: str, columns: Sequence[str], types: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    if not rows:
        return
    col_sql = ", ".join(_quote_ident(col) for col in columns)
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO {_quote_ident(table)} ({col_sql}) VALUES ({placeholders})"
    for row in rows:
        values = [
            _coerce_typed_cell(row.get(col), duckdb_type) for col, duckdb_type in zip(columns, types, strict=True)
        ]
        connection.execute(insert_sql, values)


def _load_source_into_connection(connection: Any, path: Path, *, locked_types: Mapping[str, str] | None = None) -> None:
    _ = locked_types
    table_pins, column_pins = _csv_schema_pins()
    for relation in prepare_relations_for_paths(
        [path],
        pinned_table_names=table_pins,
        pinned_column_names=column_pins,
        apply_auto_correct=True,
        source_selections=parse_source_selections(CsvRuntimeConfig.SOURCE_SELECTIONS),
    ):
        _load_prepared_relation_into_connection(connection, relation)


def _csv_source_probe_payload(paths: Sequence[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(f"{path.resolve()}|{stat.st_mtime_ns}|{digest}")
    return "\n".join(sorted(parts))


class CsvDialect(DuckDBDialect):
    """CSV/Excel files loaded into an in-memory DuckDB database per session."""

    name: str = "csv"
    sqlglot_dialect: ClassVar[str] = "duckdb"
    registry_canonical_rank: ClassVar[int] = 2
    registry_statistical_agg_excluded: ClassVar[bool] = True
    registry_window_frames_excluded: ClassVar[bool] = True
    registry_array_contains_excluded: ClassVar[bool] = True

    def __init__(
        self, config: CsvRuntimeConfig, sqlalchemy_engine: Any | None = None, *, native_connection: Any | None = None
    ) -> None:
        self._schema_json_path = str(EngineConfig.SCHEMA_JSON_PATH)
        self._native_connection: Any | None = None
        self._owns_native_connection = False
        self._owns_sqlalchemy_engine = False
        try:
            connection, owns_connection = _resolve_embedded_native_connection(
                config, sqlalchemy_engine, native_connection, open_new=lambda: _build_csv_memory_connection(config)
            )
            engine, owns_engine = _embedded_sqlalchemy_engine_for_connection(connection, "duckdb", sqlalchemy_engine)
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
        include: SchemaInclude = "tables",
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from CSV/Excel headers and inferred column types."""
        _ = sql_file
        if include == "views":
            return tables_meta_to_schema_graph({}, object_kind="view")
        csv_config = cast(CsvRuntimeConfig, self.config)
        locked_by_table = _load_locked_column_types(self._schema_json_path)
        relations = _csv_relations_for_config(csv_config, schema_json_path=self._schema_json_path)
        tables_meta = _tables_meta_from_relations(
            relations, locked_by_table=locked_by_table, allow_objects=allow_objects
        )
        return tables_meta_to_schema_graph(tables_meta, object_kind="table")

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return a cache fingerprint from source file mtimes and content hashes."""
        _ = schema_context
        try:
            csv_config = cast(CsvRuntimeConfig, self.config)
            paths = csv_config.resolve_source_files()
            return sha256(_csv_source_probe_payload(paths))
        except Exception as exc:
            debug(f"[CsvDialect.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""


register_dialect("csv", CsvDialect, CsvRuntimeConfig)
