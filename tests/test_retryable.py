"""Retryable error classification for transient failures."""

from __future__ import annotations

from aetherdialect._contracts_base import (
    DatabasePingFailed,
    LlmTransientFailure,
    RetryableError,
    StatementTimeoutError,
)


def test_retryable_types_register_under_marker() -> None:
    assert issubclass(DatabasePingFailed, RetryableError)
    assert issubclass(LlmTransientFailure, RetryableError)
    assert issubclass(StatementTimeoutError, RetryableError)


def test_instances_match_marker() -> None:
    assert isinstance(DatabasePingFailed("x"), RetryableError)
    assert isinstance(LlmTransientFailure("y"), RetryableError)
    assert isinstance(StatementTimeoutError("z"), RetryableError)


def test_engine_connect_likely_transient_message_heuristic() -> None:
    from aetherdialect._core_utils import engine_connect_likely_transient

    assert engine_connect_likely_transient(Exception("connection reset by peer"))
    assert engine_connect_likely_transient(Exception("Error 503: temporarily unavailable"))
    assert not engine_connect_likely_transient(Exception("invalid access token format"))
