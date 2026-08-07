"""Writer-lock scope and concurrent reader-turn behaviour."""

from __future__ import annotations

import threading
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
    owner._pipeline_writer_lock = threading.Lock()
    return owner


@pytest.mark.fast
def test_two_reader_turns_overlap() -> None:
    """Reader turns do not hold the writer lock and may overlap across model/DB work."""
    owner = _session_owner()
    lock = owner._pipeline_writer_lock
    overlap = threading.Event()
    first_inside = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    lock_states: list[bool] = []

    def block_first(*_args: object, **_kwargs: object) -> None:
        first_inside.set()
        lock_states.append(lock.locked())
        release_first.wait(timeout=5.0)

    def block_second(*_args: object, **_kwargs: object) -> None:
        second_started.set()
        lock_states.append(lock.locked())
        overlap.set()

    session_a = PipelineSession(owner, mode="reader")
    session_b = PipelineSession(owner, mode="reader")

    def run_first() -> None:
        with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once", side_effect=block_first):
            session_a.ask("first reader question")

    def run_second() -> None:
        assert first_inside.wait(timeout=5.0)
        with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once", side_effect=block_second):
            session_b.ask("second reader question")

    first = threading.Thread(target=run_first)
    second = threading.Thread(target=run_second)
    first.start()
    second.start()
    assert second_started.wait(timeout=5.0), "second reader turn should start while first is in flight"
    release_first.set()
    overlap.set()
    first.join(timeout=10.0)
    second.join(timeout=10.0)
    assert lock_states == [False, False], f"reader turns must not hold writer lock during work: {lock_states}"


@pytest.mark.fast
def test_writer_lock_released_across_model_call() -> None:
    """Writer turns release the pipeline lock before model and database work."""
    owner = _session_owner()
    lock = owner._pipeline_writer_lock

    lock_states: list[bool] = []

    def capture_lock_state(*_args: object, **_kwargs: object) -> None:
        lock_states.append(lock.locked())

    session = PipelineSession(owner, mode="writer")
    with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once", side_effect=capture_lock_state):
        session.ask("writer question")

    assert lock_states == [False], f"writer lock must be released during interactive_run_once, got {lock_states}"
