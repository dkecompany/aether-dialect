"""Turn correlation id is shared across audit, diagnostics, and phase events."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants_runtime import ASK_PHASE_B
from aetherdialect._contracts_core import PhaseProgressEvent, SessionStep
from aetherdialect._main_session import PipelineSession
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    emit_ask_phase,
    mint_turn_id,
    notify,
    pop_turn_id,
    pop_turn_timing,
    push_turn_id,
    push_turn_timing,
)


def _session_owner(*, phase_callback=None) -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    owner._phase_callback = phase_callback
    return owner


def _detail_value(details: tuple[tuple[str, str], ...], key: str) -> str | None:
    for k, v in details:
        if k == key:
            return v
    return None


@pytest.mark.fast
def test_ask_audit_and_diagnostics_share_turn_id() -> None:
    phase_events: list[PhaseProgressEvent] = []
    owner = _session_owner(phase_callback=lambda ev: phase_events.append(ev))
    session = PipelineSession(owner)

    def ask_side_effect(*_args: object, **_kwargs: object) -> None:
        notify("turn diagnostic", stage="test", code="ENGINE_INFO")
        emit_ask_phase(ASK_PHASE_B)

    with patch(
        "aetherdialect._main_init.MainInitOps.interactive_run_once",
        side_effect=ask_side_effect,
    ):
        step = session.ask("how many rentals")

    audit_events = owner._audit_emit.call_args_list
    turn_ids = {
        _detail_value(call.kwargs.get("details") or (), "turn_id")
        for call in audit_events
        if _detail_value(call.kwargs.get("details") or (), "turn_id") is not None
    }
    assert len(turn_ids) == 1
    turn_id = next(iter(turn_ids))

    diag_turn_ids = {
        _detail_value(d.details, "turn_id") for d in step.diagnostics if _detail_value(d.details, "turn_id") is not None
    }
    assert diag_turn_ids == {turn_id}

    phase_turn_ids = {ev.turn_id for ev in phase_events if ev.turn_id is not None}
    assert phase_turn_ids == {turn_id}

    ask_begin = next(call for call in audit_events if call.args[0] == "ask_begin")
    assert _detail_value(ask_begin.kwargs.get("details") or (), "turn_id") == turn_id
    audit_top_level_turn_ids = {call.kwargs.get("turn_id") for call in audit_events if call.kwargs.get("turn_id")}
    assert audit_top_level_turn_ids == {turn_id}


@pytest.mark.fast
def test_session_step_carries_turn_id() -> None:
    step = SessionStep(done=True, prompt=None, kind="result", turn_id="abc-123")
    assert step.turn_id == "abc-123"


@pytest.mark.fast
def test_terminal_mk_step_sets_turn_id_and_elapsed_ms() -> None:
    owner = _session_owner()
    session = PipelineSession(owner)
    turn_id = mint_turn_id()
    turn_id_token = push_turn_id(turn_id)
    timing_tokens = push_turn_timing()
    try:
        step = session._mk_step(done=True, prompt=None, kind="result")
    finally:
        pop_turn_timing(*timing_tokens)
        pop_turn_id(turn_id_token)
    assert step.turn_id == turn_id
    assert step.elapsed_ms is not None and step.elapsed_ms >= 0
