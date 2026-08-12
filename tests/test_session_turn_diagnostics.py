"""Terminal SessionStep diagnostics accumulate across suspend steps within a turn."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import PIPELINE_SUSPEND_ID_INTENT_CONFIRM, SESSION_KIND_RESULT
from aetherdialect._contracts_core import PipelineSuspended
from aetherdialect._main_session import PipelineSession
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import notify


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    return owner


SUSPEND_DIAG_CODE = "TEST_SUSPEND_PHASE_DIAG"
COMPLETE_DIAG_CODE = "TEST_COMPLETE_PHASE_DIAG"


@pytest.mark.fast
def test_terminal_step_includes_suspend_phase_diagnostics() -> None:
    """A turn that suspends once then completes carries suspend diagnostics on the terminal step."""
    session = PipelineSession(_session_owner())
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "confirm?", None)

    def ask_side_effect(*_args: object, **_kwargs: object) -> None:
        notify("suspend-phase diagnostic", stage="intent", code=SUSPEND_DIAG_CODE)
        raise suspended

    with patch("aetherdialect._main_init.MainInitOps.interactive_run_once", side_effect=ask_side_effect):
        suspend_step = session.ask("show rows")

    assert suspend_step.done is False
    suspend_codes = {d.code for d in suspend_step.diagnostics}
    assert SUSPEND_DIAG_CODE in suspend_codes

    def resume_side_effect(*_args: object, **_kwargs: object) -> None:
        notify("complete-phase diagnostic", stage="execution", code=COMPLETE_DIAG_CODE)

    with patch(
        "aetherdialect._main_interactive.MainInteractiveOps.dispatch_pipeline_resume", side_effect=resume_side_effect
    ):
        terminal_step = session.step("y")

    assert terminal_step.done is True
    assert terminal_step.kind == SESSION_KIND_RESULT
    terminal_codes = {d.code for d in terminal_step.diagnostics}
    assert SUSPEND_DIAG_CODE in terminal_codes
    assert COMPLETE_DIAG_CODE in terminal_codes


@pytest.mark.fast
def test_suspend_step_carries_only_step_local_diagnostics() -> None:
    """Non-terminal suspend steps expose diagnostics from that step only, not later phases."""
    session = PipelineSession(_session_owner())
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "confirm?", None)

    def ask_side_effect(*_args: object, **_kwargs: object) -> None:
        notify("suspend-only", stage="intent", code=SUSPEND_DIAG_CODE)
        raise suspended

    with patch("aetherdialect._main_init.MainInitOps.interactive_run_once", side_effect=ask_side_effect):
        suspend_step = session.ask("show rows")

    suspend_codes = {d.code for d in suspend_step.diagnostics}
    assert SUSPEND_DIAG_CODE in suspend_codes
    assert COMPLETE_DIAG_CODE not in suspend_codes

    def resume_side_effect(*_args: object, **_kwargs: object) -> None:
        notify("complete-only", stage="execution", code=COMPLETE_DIAG_CODE)

    with patch(
        "aetherdialect._main_interactive.MainInteractiveOps.dispatch_pipeline_resume", side_effect=resume_side_effect
    ):
        terminal_step = session.step("y")

    terminal_codes = {d.code for d in terminal_step.diagnostics}
    assert SUSPEND_DIAG_CODE in terminal_codes
    assert COMPLETE_DIAG_CODE in terminal_codes
    # Suspend-step diagnostics stay local to that step.
    assert COMPLETE_DIAG_CODE not in suspend_codes
