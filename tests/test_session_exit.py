"""PipelineSession context-manager exit semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._main_execution import PipelineSession
from aetherdialect._templates import TemplateOps


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._sandbox_closed = False
    owner._artifacts_dir = None
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = None
    owner._dialect = MagicMock()
    owner._dialect.supports_statement_cancellation = True
    return owner


@pytest.mark.fast
def test_exit_cancels_in_flight_statement() -> None:
    owner = _session_owner()
    session = PipelineSession(owner, mode="writer")
    session._session_busy = True

    with (
        patch.object(session, "cancel", wraps=session.cancel) as cancel_mock,
        patch("aetherdialect._main_execution.Dialect.cancel_in_flight_statement") as dialect_cancel,
    ):
        session.__exit__(None, None, None)

    cancel_mock.assert_called_once()
    dialect_cancel.assert_called_once_with(owner._dialect)


@pytest.mark.fast
def test_exit_leaves_engine_usable() -> None:
    owner = _session_owner()
    session = PipelineSession(owner, mode="reader")

    with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once"):
        with session:
            session.ask("first question")
        session.ask("second question after exit")

    owner.close.assert_not_called()
