"""Live smoke tests for the public ``Text2SQL`` / ``PipelineSession`` façade."""

from __future__ import annotations

import asyncio

import pytest

from aetherdialect._contracts_base import AuditEvent


@pytest.mark.live
def test_session_ask_until_done_happy_path(t2s) -> None:
    """``ask_until_done`` completes a simple list query without manual stepping."""

    with t2s.session() as session:
        step = session.ask_until_done("list two film titles", on_confirm="y")
    assert step.done
    assert step.error is None


@pytest.mark.live
def test_asession_ask_until_done_parity(t2s) -> None:
    """Async façade mirrors ``ask_until_done``."""

    async def _run() -> None:
        async with t2s.asession() as session:
            step = await session.ask_until_done("list two film titles", on_confirm="y")
        assert step.done and step.error is None

    asyncio.run(_run())


@pytest.mark.live
def test_audit_sink_receives_turn_events(t2s) -> None:
    """Audit sink sees ``ask_begin`` and ``ask_done`` for a completed turn."""

    events: list[AuditEvent] = []

    def sink(ev: AuditEvent) -> None:
        events.append(ev)

    t2s._audit_sink = sink  # type: ignore[attr-defined]
    with t2s.session() as session:
        session.ask_until_done("list one film title", on_confirm="y")
    kinds = [e.event_type for e in events]
    assert "ask_begin" in kinds and "ask_done" in kinds
