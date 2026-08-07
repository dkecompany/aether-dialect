"""Native connector backends reopen once on connection-level failures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import DatabasePingFailed
from aetherdialect._dialect_sqlglot_helper import ConnectorResultBackend, SqlalchemyExecutionMixin

ResultBackendSupport = SqlalchemyExecutionMixin


class _RecordingConnectorBackend(ConnectorResultBackend):
    def __init__(self, connection: MagicMock, *, reopen: MagicMock, execute_fn) -> None:
        super().__init__(connection, reopen=reopen)
        self._execute_fn = execute_fn

    def _fetch_rows_batched_impl(
        self,
        sql: str,
        params: dict | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ):
        _ = params, batch_rows, max_rows, max_bytes, timeout_ms
        rows = self.fetch_rows(sql, params, timeout_ms=timeout_ms)
        if rows:
            yield tuple(rows)

    def fetch_rows(self, sql: str, params: dict | None = None, *, timeout_ms: int | None = None) -> list[tuple]:
        _ = params, timeout_ms

        def _run() -> list[tuple]:
            return self._execute_fn(self._connection, sql)

        return self._run_with_connection_retry(_run)


@pytest.mark.fast
def test_dead_connection_reopened_once() -> None:
    calls = {"execute": 0}
    connection = MagicMock()
    reopen = MagicMock(side_effect=lambda: MagicMock())

    def execute_fn(conn: MagicMock, sql: str) -> list[tuple]:
        _ = conn, sql
        calls["execute"] += 1
        if calls["execute"] == 1:
            raise ConnectionError("server has gone away")
        return [(1,)]

    backend = _RecordingConnectorBackend(connection, reopen=reopen, execute_fn=execute_fn)
    rows = backend.fetch_rows("SELECT 1")
    assert rows == [(1,)]
    assert calls["execute"] == 2
    reopen.assert_called_once()


@pytest.mark.fast
def test_statement_error_is_not_retried() -> None:
    calls = {"execute": 0, "reopen": 0}
    connection = MagicMock()
    reopen = MagicMock(side_effect=lambda: MagicMock())

    class _StatementError(Exception):
        pass

    def execute_fn(conn: MagicMock, sql: str) -> list[tuple]:
        _ = conn, sql
        calls["execute"] += 1
        raise _StatementError("syntax error at or near SELECT")

    backend = _RecordingConnectorBackend(connection, reopen=reopen, execute_fn=execute_fn)
    with pytest.raises(_StatementError):
        backend.fetch_rows("SELECT bad")
    assert calls["execute"] == 1
    assert calls["reopen"] == 0
    reopen.assert_not_called()


@pytest.mark.fast
def test_second_connection_failure_is_retryable() -> None:
    connection = MagicMock()
    reopen = MagicMock(side_effect=lambda: MagicMock())

    def execute_fn(conn: MagicMock, sql: str) -> list[tuple]:
        _ = conn, sql
        raise ConnectionError("connection reset by peer")

    backend = _RecordingConnectorBackend(connection, reopen=reopen, execute_fn=execute_fn)
    with pytest.raises(DatabasePingFailed):
        backend.fetch_rows("SELECT 1")
    assert reopen.call_count == 1
    assert ResultBackendSupport.is_connection_level_error(ConnectionError("connection reset by peer"))
