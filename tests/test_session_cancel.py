"""Unified PipelineSession.cancel() for federation and non-federation turns."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import SessionTurnCancelledError
from aetherdialect._contracts_core import FederationExecutionContext
from aetherdialect._main_execution import (
    MainExecutionOps,
    PipelineSession,
)


@pytest.mark.fast
def test_cancel_delegates_federation_context() -> None:
    session = PipelineSession(MagicMock())
    ctx = FederationExecutionContext(plan_id="test-plan")
    session._active_federation_execution_context = ctx
    session._session_busy = True
    session._turn_cancel_event = threading.Event()

    assert session.cancel() is True
    assert ctx.cancelled is True
    assert session._turn_cancel_event.is_set()


@pytest.mark.fast
def test_cancel_sets_turn_event_when_busy() -> None:
    session = PipelineSession(MagicMock())
    session._session_busy = True
    session._turn_cancel_event = threading.Event()
    session._active_federation_execution_context = None

    assert session.cancel() is True
    assert session._turn_cancel_event.is_set()


@pytest.mark.fast
def test_cancel_returns_false_when_idle() -> None:
    session = PipelineSession(MagicMock())
    assert session.cancel() is False


@pytest.mark.fast
def test_cancel_active_federation_turn_delegates_to_cancel() -> None:
    session = PipelineSession(MagicMock())
    session._session_busy = True
    session._turn_cancel_event = threading.Event()

    with patch.object(session, "cancel", return_value=True) as mock_cancel:
        assert session.cancel_active_federation_turn() is True
    mock_cancel.assert_called_once_with()


@pytest.mark.fast
def test_session_turn_cancelled_raises_in_interactive_run_once() -> None:
    with patch("aetherdialect._main_execution.session_turn_cancelled", return_value=True):
        with pytest.raises(SessionTurnCancelledError):
            MainExecutionOps.interactive_run_once(question="how many customers?", pipeline_session=MagicMock())


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    return owner


@pytest.mark.fast
def test_cancel_while_suspended_frees_session_for_ask() -> None:
    from aetherdialect._constants import AUDIT_EVENT_ASK_CANCELLED, SESSION_KIND_ERROR
    from aetherdialect._pipeline import (
        PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
        PipelineSuspended,
    )

    owner = _session_owner()
    sess = PipelineSession(owner)
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "confirm?", None)

    with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once", side_effect=suspended):
        step = sess.ask("first question")

    assert step.done is False
    assert sess.awaiting_prompt() is True

    assert sess.cancel() is True
    assert sess.awaiting_prompt() is False

    cancel_step = sess.step()
    assert cancel_step.done is True
    assert cancel_step.kind == SESSION_KIND_ERROR
    assert cancel_step.status == "cancelled"

    audit_names = [call.args[0] for call in owner._audit_emit.call_args_list]
    assert AUDIT_EVENT_ASK_CANCELLED in audit_names

    def _complete_turn(*_args: object, **_kwargs: object) -> None:
        sess.note_turn_outcome(outcome="success")

    with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once", side_effect=_complete_turn):
        second = sess.ask("second question")

    assert second.done is True
