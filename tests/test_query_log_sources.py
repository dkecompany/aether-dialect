"""Tests for dialect query-log source implementations."""

from __future__ import annotations

from aetherdialect._sql_to_intent import (
    BigQueryQueryLogSource,
    MySQLQueryLogSource,
    RedshiftQueryLogSource,
    SnowflakeQueryLogSource,
    SQLServerQueryLogSource,
    fetch_query_log,
)


def test_mysql_query_log_source_fetch_stabilizes_literals() -> None:
    """MySQL fetch masks literals in returned SQL texts."""
    src = MySQLQueryLogSource()
    captured: list[tuple] = []

    class _Cur:
        def execute(self, stmt: str) -> None:
            captured.append((stmt,))

        def fetchall(self) -> list[tuple[str]]:
            return [("SELECT * FROM t WHERE id = 42 AND name = 'alice'",)]

        def close(self) -> None:
            pass

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    assert src.is_available(_Conn()) is True
    out = src.fetch(_Conn(), lookback_days=1, max_queries=5, min_runs=1, user_filter=None)
    assert out == ["SELECT * FROM t WHERE id = <num> AND name = <str>"]
    assert ":lookback_microseconds" not in captured[0][0]


def test_redshift_query_log_source_unavailable_on_error() -> None:
    """Redshift availability probe returns False when the catalog errors."""
    src = RedshiftQueryLogSource()

    class _Cur:
        def execute(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("denied")

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    assert src.is_available(_Conn()) is False


def test_sqlserver_query_log_source_available_on_empty_probe() -> None:
    """SQL Server treats an empty DMV probe as available."""
    src = SQLServerQueryLogSource()

    class _Cur:
        def execute(self, *_a: object, **_k: object) -> None:
            pass

        def close(self) -> None:
            pass

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    assert src.is_available(_Conn()) is True


def test_snowflake_query_log_source_fetch() -> None:
    """Snowflake fetch returns stabilized query texts."""
    src = SnowflakeQueryLogSource()

    class _Cur:
        def execute(self, _stmt: str) -> None:
            pass

        def fetchall(self) -> list[tuple[str]]:
            return [("SELECT 1",)]

        def close(self) -> None:
            pass

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    assert src.fetch(_Conn(), lookback_days=7, max_queries=3, min_runs=1, user_filter=None) == ["SELECT <num>"]


def test_bigquery_query_log_source_requires_project(monkeypatch) -> None:
    """BigQuery fetch returns empty when no project is configured."""
    from aetherdialect._config import BigQueryRuntimeConfig, EngineConfig

    monkeypatch.setattr(EngineConfig, "TYPE", "bigquery")
    monkeypatch.setattr(EngineConfig, "RUNTIME", BigQueryRuntimeConfig)
    monkeypatch.setattr(BigQueryRuntimeConfig, "PROJECT", "")

    src = BigQueryQueryLogSource()

    class _Conn:
        def cursor(self) -> object:
            raise AssertionError("cursor should not be called without project")

    assert src.is_available(_Conn()) is False
    assert src.fetch(_Conn(), lookback_days=1, max_queries=1, min_runs=1, user_filter=None) == []


def test_fetch_query_log_dispatches_mysql(monkeypatch) -> None:
    """``fetch_query_log`` routes mysql to the dialect query-log source."""
    from aetherdialect._config import EngineConfig, MySQLRuntimeConfig

    monkeypatch.setattr(EngineConfig, "TYPE", "mysql")
    monkeypatch.setattr(EngineConfig, "RUNTIME", MySQLRuntimeConfig)

    class _Cur:
        def execute(self, _stmt: str) -> None:
            pass

        def fetchone(self) -> tuple[int]:
            return (1,)

        def fetchall(self) -> list[tuple[str]]:
            return [("SELECT 1",)]

        def close(self) -> None:
            pass

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    out = fetch_query_log("mysql", _Conn(), lookback_days=1, max_queries=5, min_runs=1, user_filter=None)
    assert out == ["SELECT <num>"]
