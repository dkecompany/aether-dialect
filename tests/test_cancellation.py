"""Statement cancellation hooks and unsupported-engine diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_CANCEL_NOT_SUPPORTED
from aetherdialect._contracts_base import FederationTurnCancelledError, SessionTurnCancelledError
from aetherdialect._dialect import Dialect
from aetherdialect._pipeline import _raise_federation_turn_cancelled


@pytest.mark.fast
def test_cancel_calls_dialect_hook() -> None:
    dialect = MagicMock()
    dialect.supports_statement_cancellation = True
    Dialect.cancel_in_flight_statement(dialect)
    dialect.cancel_statement.assert_called_once()


@pytest.mark.fast
def test_unsupported_cancel_reports() -> None:
    dialect = MagicMock()
    dialect.supports_statement_cancellation = False
    dialect.logical_engine_name = "BigQuery"
    with patch("aetherdialect._dialect.notify") as notify_mock:
        Dialect.cancel_in_flight_statement(dialect)
    notify_mock.assert_called_once()
    assert notify_mock.call_args.kwargs["code"] == DIAGNOSTIC_CODE_CANCEL_NOT_SUPPORTED
    assert "BigQuery" in notify_mock.call_args.args[0]
    dialect.cancel_statement.assert_not_called()


@pytest.mark.fast
def test_cancelled_turn_reports_cancelled_not_failed() -> None:
    from aetherdialect._main_execution import PipelineSession

    owner = MagicMock()
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = MagicMock()
    owner._artifacts_dir = None
    owner._pipeline_writer_lock = None
    owner._ask_phase_callback = None
    owner._active_space_name = "master"
    owner.dialect = "duckdb"
    owner._runtime_config = owner._runtime_config
    session = PipelineSession(owner, mode="writer")
    session._session_busy = True

    session._turn_llm_scope_tok = None
    with (
        patch(
            "aetherdialect._main_execution.MainExecutionOps.interactive_run_once",
            side_effect=SessionTurnCancelledError("Turn cancelled."),
        ),
        patch("aetherdialect._main_execution.push_session_turn_cancel"),
        patch("aetherdialect._main_execution.pop_session_turn_cancel"),
        patch("aetherdialect._main_execution.push_ask_phase_callback"),
        patch("aetherdialect._main_execution.pop_ask_phase_callback"),
        patch("aetherdialect._main_execution.llm_usage_session_scope"),
        patch("aetherdialect._main_execution.snapshot_llm_usage_records", return_value=[]),
        patch.object(session, "_resources", return_value=(None, {}, {}, [], {})),
        patch.object(session, "_owner_engine_identity", return_value=MagicMock()),
        patch("aetherdialect._main_execution.push_engine_identity"),
        patch("aetherdialect._main_execution.pop_engine_identity"),
        patch("aetherdialect._main_execution.push_engine_limits"),
        patch("aetherdialect._main_execution.pop_engine_limits"),
    ):
        step = session._drive_question_turn("how many rows?")

    assert step.done is True
    assert step.status == "cancelled"
    assert step.status != "execution_other_error"


@pytest.mark.fast
def test_federation_cancel_path_uses_cancel_helper() -> None:
    dialect = MagicMock()
    dialect.supports_statement_cancellation = True
    with (
        patch("aetherdialect._pipeline.Dialect.cancel_in_flight_statement") as cancel_mock,
        patch("aetherdialect._pipeline.notify"),
        pytest.raises(FederationTurnCancelledError),
    ):
        _raise_federation_turn_cancelled(
            source_id="a",
            phase="member",
            dialect=dialect,
        )
    cancel_mock.assert_called_once_with(dialect)
