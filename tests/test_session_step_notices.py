"""SessionStep.notices carries structured bookkeeping separate from message."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import SessionNotice, SessionStep


@pytest.mark.fast
def test_session_notice_fields() -> None:
    notice = SessionNotice(code="SAVED_LINE", level="info", message="Saved.")
    assert notice.code == "SAVED_LINE"
    assert notice.level == "info"
    assert notice.message == "Saved."


@pytest.mark.fast
def test_session_step_notices_default_empty() -> None:
    step = SessionStep(done=True, prompt=None, kind="ok")
    assert step.notices == ()


@pytest.mark.fast
def test_session_step_notices_can_be_set() -> None:
    notice = SessionNotice(code="SAVED_LINE", level="info", message="Saved.")
    step = SessionStep(done=True, prompt=None, kind="ok", notices=(notice,))
    assert step.notices == (notice,)
