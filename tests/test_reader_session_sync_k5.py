"""Reader pipeline lock and atomic session-busy guard."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import SessionActiveError
from aetherdialect._main_execution import PipelineSession
from aetherdialect._templates import empty_template_store


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._sandbox_closed = False
    owner._artifacts_dir = None
    owner._pipeline_writer_lock = threading.Lock()
    return owner


@pytest.mark.fast
def test_concurrent_ask_raises_session_active_error() -> None:
    """Two threads cannot both enter a turn; the second must get SessionActiveError."""
    session = PipelineSession(_session_owner(), mode="writer")
    inside_drive = threading.Event()
    release_drive = threading.Event()
    results: list[str] = []
    orig_reset = PipelineSession._reset_after_turn

    def gated_reset(self: PipelineSession) -> None:
        inside_drive.set()
        release_drive.wait(timeout=5)
        orig_reset(self)

    def block_run(*_args: object, **_kwargs: object) -> None:
        release_drive.wait(timeout=5)

    def run_ask() -> None:
        with patch("aetherdialect._main_execution.interactive_run_once", side_effect=block_run):
            session.ask("question")

    def run_second_ask() -> None:
        assert inside_drive.wait(timeout=5)
        try:
            session.ask("second")
        except SessionActiveError:
            results.append("session_active")

    with patch.object(PipelineSession, "_reset_after_turn", gated_reset):
        first = threading.Thread(target=run_ask)
        second = threading.Thread(target=run_second_ask)
        first.start()
        second.start()
        second.join(timeout=10)
        release_drive.set()
        first.join(timeout=10)

    assert results.count("session_active") == 1


@pytest.mark.fast
def test_reader_turn_holds_pipeline_lock() -> None:
    """Reader-mode turns acquire the owner pipeline lock for synchronized reload."""
    owner = _session_owner()
    lock = MagicMock()
    owner._pipeline_writer_lock = lock

    session = PipelineSession(owner, mode="reader")
    with patch("aetherdialect._main_execution.interactive_run_once"):
        session.ask("show rows")

    lock.__enter__.assert_called_once()
