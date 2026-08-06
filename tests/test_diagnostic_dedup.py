"""Repeated diagnostic conditions collapse to one row with a count."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import Diagnostic, DiagnosticSeverity
from aetherdialect._main_execution import PipelineSession


@pytest.mark.fast
def test_repeated_condition_reported_once_with_count() -> None:
    first = Diagnostic(
        stage="execution",
        level=DiagnosticSeverity.ERROR,
        code="FEDERATION_MEMBER_FAILED",
        message="first wording",
        phase="execution",
        subject="member_a",
        count=1,
    )
    second = Diagnostic(
        stage="execution",
        level=DiagnosticSeverity.ERROR,
        code="FEDERATION_MEMBER_FAILED",
        message="different wording",
        phase="execution",
        subject="member_a",
        count=1,
    )
    session = object.__new__(PipelineSession)
    session._turn_accumulated_diagnostics = []
    session._extend_turn_accumulated_diagnostics((first, second))
    assert len(session._turn_accumulated_diagnostics) == 1
    retained = session._turn_accumulated_diagnostics[0]
    assert retained.count == 2
    assert retained.phase == "execution"
    assert retained.subject == "member_a"
