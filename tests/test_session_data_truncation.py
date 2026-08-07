"""SessionStep.data_truncated signals row-cap truncation on tabular results."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import SessionStep


@pytest.mark.fast
def test_session_step_data_truncated_defaults_false() -> None:
    step = SessionStep(done=True, prompt=None, kind="ok")
    assert step.data_truncated is False


@pytest.mark.fast
def test_session_step_data_truncated_can_be_set_true() -> None:
    step = SessionStep(done=True, prompt=None, kind="ok", data_truncated=True)
    assert step.data_truncated is True
