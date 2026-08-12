"""Async cancel dispatches through ContextVar-preserving worker threads."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, cast

import pytest

from aetherdialect import AsyncPipelineSession

_PROBE: ContextVar[str | None] = ContextVar("async_cancel_probe", default=None)


@pytest.mark.fast
def test_cancel_during_ask_no_context_token_error() -> None:
    seen: dict[str, str | None] = {}

    class _Inner:
        def __init__(self) -> None:
            self._calls = 0

        def cancel(self) -> bool:
            self._calls += 1
            seen[f"cancel{self._calls}"] = _PROBE.get()
            return self._calls == 1

    ap = AsyncPipelineSession(cast(Any, _Inner()))
    token = _PROBE.set("cancel-bound")

    async def _run() -> None:
        assert await ap.cancel() is True
        assert await ap.cancel() is False

    try:
        asyncio.run(_run())
    finally:
        _PROBE.reset(token)

    assert seen["cancel1"] == "cancel-bound"
    assert seen["cancel2"] == "cancel-bound"
