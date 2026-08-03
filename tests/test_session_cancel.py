"""Unified PipelineSession.cancel() for federation and non-federation turns."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import SessionTurnCancelledError
from aetherdialect._contracts_core import FederationExecutionContext
from aetherdialect._main_execution import PipelineSession, interactive_run_once


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
            interactive_run_once(question="how many customers?", pipeline_session=MagicMock())
