"""Permission denial is detectable by exception type and SessionStep fields."""

from __future__ import annotations

import dataclasses

from aetherdialect._constants import REFUSAL_NOT_AVAILABLE_IN_CONTEXT
from aetherdialect._constants_runtime import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_core import AccessError, SessionError, SessionOutcome, SessionStep


def test_denial_detectable_by_exception_type() -> None:
    exc = AccessError("execute", PERMISSION_DENIED_USER_MESSAGE)
    assert isinstance(exc, AccessError)
    assert exc.operation == "execute"


def test_denial_detectable_by_step_status_and_code() -> None:
    fields = {f.name for f in dataclasses.fields(SessionStep)}
    assert "error" in fields
    assert "refusal_code" not in fields
    step = SessionStep(
        done=True,
        prompt=None,
        kind="error",
        error=SessionError(code=SessionOutcome.FORBIDDEN, detail_code=REFUSAL_NOT_AVAILABLE_IN_CONTEXT),
    )
    assert step.error is not None and step.error.code.value == "forbidden"
    assert step.error.detail_code == REFUSAL_NOT_AVAILABLE_IN_CONTEXT
