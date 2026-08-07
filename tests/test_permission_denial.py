"""Permission denial is detectable by exception type and SessionStep fields."""

from __future__ import annotations

import dataclasses

from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE, REFUSAL_NOT_AVAILABLE_IN_CONTEXT
from aetherdialect._contracts_base import AccessError, SessionStep


def test_denial_detectable_by_exception_type() -> None:
    exc = AccessError("execute", PERMISSION_DENIED_USER_MESSAGE)
    assert isinstance(exc, AccessError)
    assert exc.operation == "execute"


def test_denial_detectable_by_step_status_and_code() -> None:
    fields = {f.name for f in dataclasses.fields(SessionStep)}
    assert "refusal_code" in fields
    step = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        message=PERMISSION_DENIED_USER_MESSAGE,
        status="permission_denied",
        refusal_code=REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    )
    assert step.status == "permission_denied"
    assert step.refusal_code == REFUSAL_NOT_AVAILABLE_IN_CONTEXT
