"""Tests for engine-correct execution limits and timeouts (Phase G)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aetherdialect._config import PolicyConfig
from aetherdialect._dialect_sqlglot_engines import (
    BigQueryClientBackend,
    BigQueryDialect,
    DatabricksConnectorBackend,
    DatabricksDialect,
    DatabricksSqlAlchemyBackend,
    DuckDBNativeBackend,
    MySQLConnectorBackend,
    MySQLDialect,
    RedshiftConnectorBackend,
    SnowflakeArrowBackend,
    SQLiteNativeBackend,
    SQLServerDialect,
)
from aetherdialect._dialect_sqlglot_helper import (
    SqlAlchemyResultBackend,
    SqlServerResultBackend,
)


def test_sqlalchemy_backend_uses_dialect_timeout_sql() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [(1,)]
    backend = SqlAlchemyResultBackend(
        engine,
        dialect_name="MySQLDialect",
        timeout_sql_provider=lambda ms: f"SET SESSION MAX_EXECUTION_TIME = {ms}",
    )
    rows = backend.fetch_rows("SELECT 1", timeout_ms=5000)
    assert rows == [(1,)]
    assert "MAX_EXECUTION_TIME" in str(conn.execute.call_args_list[0].args[0])


def test_mysql_dialect_timeout_sql() -> None:
    dialect = object.__new__(MySQLDialect)
    sql = MySQLDialect.profile_statement_timeout_sql(dialect, 12_000)
    assert sql is not None
    assert "MAX_EXECUTION_TIME" in sql


def test_snowflake_arrow_backend_sets_session_timeout() -> None:
    conn = MagicMock()
    cursor = MagicMock(spec=["execute", "fetchall", "close"])
    cursor.fetchall.return_value = [(3,)]
    conn.cursor.return_value = cursor
    backend = SnowflakeArrowBackend(connection=conn)
    rows = backend.fetch_rows("SELECT 3", timeout_ms=30_000)
    assert rows == [(3,)]
    timeout_calls = [str(c.args[0]) for c in cursor.execute.call_args_list]
    assert any("STATEMENT_TIMEOUT_IN_SECONDS" in c for c in timeout_calls)


def test_databricks_connector_backend_honors_timeout() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = [(1,)]
    backend = DatabricksConnectorBackend(conn)
    rows = backend.fetch_rows("SELECT 1", timeout_ms=10_000)
    assert rows == [(1,)]
    conn.set_query_timeout.assert_called_once()


def test_bigquery_client_backend_applies_job_limits() -> None:
    client = MagicMock()
    job = MagicMock()
    client.query.return_value = job
    job.result.return_value = iter([{"a": 1}])
    backend = BigQueryClientBackend(
        client,
        maximum_bytes_billed=1_000_000,
        job_timeout_ms=5_000,
        dialect_name="BigQueryDialect",
    )
    with patch.object(backend, "_job_config") as mock_cfg:
        cfg = MagicMock()
        mock_cfg.return_value = cfg
        rows = backend.fetch_rows("SELECT 1", timeout_ms=2_000)
    assert rows == [(1,)]
    mock_cfg.assert_called_once_with(timeout_ms=2_000, params=None)


def test_sqlserver_execute_routes_sp_executesql_through_backend() -> None:
    dialect = object.__new__(SQLServerDialect)
    mock_backend = MagicMock()
    mock_backend.fetch_rows.return_value = [(7,)]
    dialect._backend = mock_backend
    rows = SQLServerDialect.execute(dialect, "SELECT :x", {"x": 1})
    assert rows == [(7,)]
    mock_backend.fetch_rows.assert_called_once()
    assert mock_backend.fetch_rows.call_args.args[0] == "EXEC sp_executesql :stmt"


def test_sqlserver_profile_timeout_returns_none() -> None:
    dialect = object.__new__(SQLServerDialect)
    assert SQLServerDialect.profile_statement_timeout_sql(dialect, 30_000) is None


def test_sqlserver_result_backend_sets_driver_timeout() -> None:
    engine = MagicMock()
    conn = MagicMock()
    raw = MagicMock()
    conn.connection = raw
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [(2,)]
    backend = SqlServerResultBackend(engine, dialect_name="SQLServerDialect")
    rows = backend.fetch_rows("SELECT 2", timeout_ms=8000)
    assert rows == [(2,)]
    assert raw.timeout == 8


def test_policy_config_apply_environment_reads_execution_caps() -> None:
    PolicyConfig.apply_environment(
        {
            "AETHERDIALECT_MAX_QUERY_COST_BYTES": "999",
            "AETHERDIALECT_STATEMENT_TIMEOUT_MS": "45000",
        }
    )
    assert PolicyConfig.MAX_QUERY_COST_BYTES == 999.0
    assert PolicyConfig.STATEMENT_TIMEOUT_MS == 45000


def test_databricks_profile_timeout_returns_none() -> None:
    dialect = object.__new__(DatabricksDialect)
    assert DatabricksDialect.profile_statement_timeout_sql(dialect, 30_000) is None


def test_databricks_sqlalchemy_backend_does_not_set_statement_timeout() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [(1,)]
    backend = DatabricksSqlAlchemyBackend(engine, dialect_name="DatabricksDialect")
    rows = backend.fetch_rows("SELECT 1", timeout_ms=5000)
    assert rows == [(1,)]
    executed = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert not any("statement_timeout" in sql for sql in executed)


def test_mysql_connector_backend_honors_timeout() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = [(2,)]
    backend = MySQLConnectorBackend(conn)
    rows = backend.fetch_rows("SELECT 2", timeout_ms=12_000)
    assert rows == [(2,)]
    assert any("MAX_EXECUTION_TIME" in str(c.args[0]) for c in cursor.execute.call_args_list)


def test_redshift_connector_backend_honors_timeout() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = [(3,)]
    backend = RedshiftConnectorBackend(conn)
    rows = backend.fetch_rows("SELECT 3", timeout_ms=8000)
    assert rows == [(3,)]
    assert any("statement_timeout" in str(c.args[0]) for c in cursor.execute.call_args_list)


def test_duckdb_native_backend_fetch_rows() -> None:
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = [(4,)]
    backend = DuckDBNativeBackend(connection)
    assert backend.fetch_rows("SELECT 4") == [(4,)]


def test_duckdb_native_backend_fetch_rows_passes_params() -> None:
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = [(1,)]
    backend = DuckDBNativeBackend(connection)
    params = {"p1": "horror"}
    assert backend.fetch_rows("SELECT 1 WHERE name = $p1", params) == [(1,)]
    connection.execute.assert_called_once_with("SELECT 1 WHERE name = $p1", params)


def test_sqlite_native_backend_fetch_rows() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = [(5,)]
    backend = SQLiteNativeBackend(conn)
    assert backend.fetch_rows("SELECT 5") == [(5,)]


def test_bigquery_apply_execute_cost_limits_sets_bytes_cap() -> None:
    dialect = object.__new__(BigQueryDialect)
    job_config = MagicMock()
    with patch("aetherdialect._dialect_sqlglot_engines.PolicyConfig.MAX_QUERY_COST_BYTES", 2048):
        with patch("aetherdialect._dialect_sqlglot_engines.cost_cap_active", return_value=True):
            BigQueryDialect.apply_execute_cost_limits(dialect, job_config)
    assert job_config.maximum_bytes_billed == 2048
