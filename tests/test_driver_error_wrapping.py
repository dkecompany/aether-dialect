"""Driver execution errors are wrapped in library exceptions."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if "aetherdialect" not in sys.modules:
    _pkg = types.ModuleType("aetherdialect")
    _pkg.__path__ = [str(_SRC / "aetherdialect")]
    _pkg.__package__ = "aetherdialect"
    sys.modules["aetherdialect"] = _pkg

pre_fix_failure: str | None = None

_DRIVER_SECRET = "driver-secret-token-xyzzy-42"


class _ExplodingBackend:
    kind = None

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
        _ = sql, params, batch_rows, max_rows, max_bytes, timeout_ms
        raise RuntimeError(_DRIVER_SECRET)


def test_driver_error_becomes_library_error() -> None:
    global pre_fix_failure
    from aetherdialect._contracts_base import AetherError, DatabaseExecutionError
    from aetherdialect._dialect_sqlglot_helper import ResultBackend

    backend = _ExplodingBackend()
    backend.fetch_rows_batched = ResultBackend.fetch_rows_batched.__get__(backend, _ExplodingBackend)

    with pytest.raises(DatabaseExecutionError) as caught:
        list(
            backend.fetch_rows_batched(
                "SELECT 1",
                batch_rows=100,
            )
        )
    wrapped = caught.value
    if not isinstance(wrapped, AetherError):
        pre_fix_failure = f"expected AetherError subclass, got {type(wrapped)!r}"
    assert isinstance(wrapped, AetherError), pre_fix_failure
    assert wrapped.__cause__ is not None, pre_fix_failure
    assert _DRIVER_SECRET in str(wrapped.__cause__), pre_fix_failure


def test_raw_driver_text_not_in_message() -> None:
    global pre_fix_failure
    from aetherdialect._contracts_base import DatabaseExecutionError
    from aetherdialect._dialect_sqlglot_helper import ResultBackend

    backend = _ExplodingBackend()
    backend.fetch_rows_batched = ResultBackend.fetch_rows_batched.__get__(backend, _ExplodingBackend)

    with pytest.raises(DatabaseExecutionError) as caught:
        list(backend.fetch_rows_batched("SELECT 1", batch_rows=100))
    wrapped = caught.value
    if _DRIVER_SECRET in str(wrapped):
        pre_fix_failure = "raw driver text leaked into user-facing DatabaseExecutionError message"
    assert _DRIVER_SECRET not in str(wrapped), pre_fix_failure
    assert wrapped.driver_detail is not None, pre_fix_failure
    assert wrapped.driver_detail.get("message") == _DRIVER_SECRET, pre_fix_failure
