"""Federation malformed-member and join fan-out diagnostic codes must be emitted."""

from __future__ import annotations

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
    DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
)
from aetherdialect._contracts_base import (
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMemberExecutionError,
)
from aetherdialect._main_execution import _federation_error_diagnostics


@pytest.mark.fast
def test_malformed_member_answer_emits_dedicated_diagnostic_code() -> None:
    exc = FederationMalformedMemberAnswerError(
        "projection mismatch",
        source_id="b",
        phase="member",
    )
    diags = _federation_error_diagnostics(exc)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER
    assert diags[0].code != DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED
    assert diags[0].source_id == "b"


@pytest.mark.fast
def test_join_fan_out_emits_dedicated_diagnostic_code() -> None:
    exc = FederationJoinFanOutError(
        "join multiplied rows",
        source_id="b",
        phase="coordinator",
    )
    diags = _federation_error_diagnostics(exc)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT
    assert diags[0].source_id == "b"
    assert diags[0].details
    phase = dict(diags[0].details).get("phase")
    assert phase == "coordinator"


@pytest.mark.fast
def test_generic_member_execution_still_emits_member_failed_code() -> None:
    exc = FederationMemberExecutionError("query failed", source_id="a", phase="member")
    diags = _federation_error_diagnostics(exc)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED
