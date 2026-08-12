"""Worker thread lifetime and pooled-connection hygiene."""

from __future__ import annotations

import threading

import pytest

from aetherdialect._contracts_base import FederationCapExceededError
from aetherdialect._utils_artifacts import clear_connection_poison, is_connection_poisoned


@pytest.mark.fast
def test_timed_out_worker_connection_not_reused() -> None:
    """A stuck worker poisons its connection so the pool must not reuse it."""
    from aetherdialect._federation_execute import _execute_coordinator_sql_with_timeout

    hang = threading.Event()
    release = threading.Event()
    pool_returned = False

    class _FakeResult:
        def fetchdf(self) -> object:
            return None

    class _FakeConn:
        def execute(self, sql: str, params: dict[str, object] | None = None) -> _FakeResult:
            hang.set()
            release.wait(timeout=30.0)
            return _FakeResult()

        def interrupt(self) -> None:
            pass

    conn = _FakeConn()
    clear_connection_poison(conn)

    worker_error: list[BaseException] = []

    def _run() -> None:
        try:
            _execute_coordinator_sql_with_timeout(conn, "SELECT 1", {}, timeout_ms=50)
        except BaseException as exc:
            worker_error.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    assert hang.wait(timeout=5.0)
    worker.join(timeout=10.0)

    assert worker_error
    exc = worker_error[0]
    assert isinstance(exc, FederationCapExceededError)
    assert exc.source_id == "coordinator"
    assert "coordinator" in str(exc).lower()
    assert is_connection_poisoned(conn), "timed-out worker connection must be poisoned"

    if not is_connection_poisoned(conn):
        pool_returned = True
    assert not pool_returned

    release.set()
