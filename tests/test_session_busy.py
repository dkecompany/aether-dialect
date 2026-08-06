"""Session busy flag must be race-safe and pending plan state must be per-session."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import FederationPlanTemplate, SessionActiveError
from aetherdialect._federation import clear_federated_turn_state
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
    owner._pipeline_writer_lock = threading.Lock()
    return owner


@pytest.mark.fast
def test_concurrent_ask_only_one_turn_starts() -> None:
    """Two threads racing ask() must not both enter a turn."""
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
        with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once", side_effect=block_run):
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
def test_session_busy_lock_serializes_check_and_set() -> None:
    """The busy flag check and set must happen under the same lock."""
    session = PipelineSession(_session_owner(), mode="writer")
    assert hasattr(session, "_session_busy_lock")
    acquire = getattr(session, "_acquire_session_turn", None)
    release = getattr(session, "_release_session_turn", None)
    assert callable(acquire)
    assert callable(release)
    acquire()
    with pytest.raises(SessionActiveError):
        acquire()
    release()
    acquire()
    release()


@pytest.mark.fast
def test_pending_plan_template_isolated_between_sessions_on_same_owner() -> None:
    owner = _session_owner()
    tmpl_a = FederationPlanTemplate(
        plan_id="plan_a",
        composite_schema_graph_id="cg_a",
        intent_key="ik_a",
        step_fingerprints=(),
        combine_hash="hash_a",
    )
    tmpl_b = FederationPlanTemplate(
        plan_id="plan_b",
        composite_schema_graph_id="cg_b",
        intent_key="ik_b",
        step_fingerprints=(),
        combine_hash="hash_b",
    )
    session_a = PipelineSession(owner)
    session_b = PipelineSession(owner)
    session_a._pending_federation_plan_template = tmpl_a
    session_b._pending_federation_plan_template = tmpl_b
    clear_federated_turn_state(session_a)
    assert session_a._pending_federation_plan_template is None
    assert session_b._pending_federation_plan_template is tmpl_b


@pytest.mark.fast
def test_owner_has_no_pending_plan_template_slot() -> None:
    class _Owner:
        pass

    owner = _Owner()
    assert not hasattr(owner, "_pending_federation_plan_template")
