"""AsyncPipelineSession propagates ContextVars into worker threads."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, cast

import pytest

from aetherdialect import AsyncPipelineSession

_PROBE: ContextVar[str | None] = ContextVar("async_session_probe", default=None)


@pytest.mark.fast
def test_ask_step_cleanup_same_context() -> None:
    seen: dict[str, str | None] = {}

    class _Inner:
        def ask(self, question: str):
            seen["ask"] = _PROBE.get()
            return type("S", (), {"done": True, "kind": "result"})()

        def step(self, response: str | None = None):
            seen["step"] = _PROBE.get()
            return type("S", (), {"done": True, "kind": "result"})()

        def reset(self) -> None:
            return None

        def awaiting_prompt(self) -> bool:
            return False

        def cancel(self) -> bool:
            seen["cancel"] = _PROBE.get()
            return True

    ap = AsyncPipelineSession(cast(Any, _Inner()))
    token = _PROBE.set("bound-value")

    async def _run() -> None:
        await ap.ask("q")
        await ap.step("y")
        await ap.cancel()

    try:
        asyncio.run(_run())
    finally:
        _PROBE.reset(token)

    assert seen["ask"] == "bound-value"
    assert seen["step"] == "bound-value"
    assert seen["cancel"] == "bound-value"
