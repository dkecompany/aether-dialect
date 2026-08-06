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
        def cancel(self) -> bool:
            seen["cancel"] = _PROBE.get()
            return True

        def cancel_active_federation_turn(self) -> bool:
            seen["fed"] = _PROBE.get()
            return False

    ap = AsyncPipelineSession(cast(Any, _Inner()))
    token = _PROBE.set("cancel-bound")

    async def _run() -> None:
        assert await ap.cancel() is True
        assert await ap.cancel_active_federation_turn() is False

    try:
        asyncio.run(_run())
    finally:
        _PROBE.reset(token)

    assert seen["cancel"] == "cancel-bound"
    assert seen["fed"] == "cancel-bound"
