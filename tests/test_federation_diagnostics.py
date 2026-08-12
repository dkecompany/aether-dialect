"""Federation cap, malformed-member, and join fan-out diagnostics: emission and catalogue."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED,
    DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
    DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
)
from aetherdialect._contracts_base import (
    FederationCapExceededError,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMemberExecutionError,
)
from aetherdialect._main_execution import MainExecutionOps

_REPO = Path(__file__).resolve().parents[1]
_API_REFERENCE = _REPO / "docs" / "API_REFERENCE.md"
_TROUBLESHOOTING = _REPO / "docs" / "TROUBLESHOOTING.md"

_L37_FEDERATION_DIAGNOSTIC_CODES = frozenset(
    {
        DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED,
        DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
        DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
    }
)


@pytest.mark.fast
def test_cap_exceeded_emits_dedicated_diagnostic_code() -> None:
    exc = FederationCapExceededError(
        "row cap exceeded",
        source_id="storefront",
        limit_key="row_cap",
    )
    diags = MainExecutionOps.federation_error_diagnostics(exc)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED
    assert diags[0].source_id == "storefront"
    assert dict(diags[0].details).get("limit_key") == "row_cap"


@pytest.mark.fast
def test_malformed_member_answer_emits_dedicated_diagnostic_code() -> None:
    exc = FederationMalformedMemberAnswerError(
        "projection mismatch",
        source_id="b",
        phase="member",
    )
    diags = MainExecutionOps.federation_error_diagnostics(exc)
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
    diags = MainExecutionOps.federation_error_diagnostics(exc)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT
    assert diags[0].source_id == "b"
    assert diags[0].details
    phase = dict(diags[0].details).get("phase")
    assert phase == "coordinator"


@pytest.mark.fast
def test_generic_member_execution_still_emits_member_failed_code() -> None:
    exc = FederationMemberExecutionError("query failed", source_id="a", phase="member")
    diags = MainExecutionOps.federation_error_diagnostics(exc)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED


@pytest.mark.fast
def test_api_reference_documents_l37_federation_diagnostic_codes() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(code for code in _L37_FEDERATION_DIAGNOSTIC_CODES if code not in text)
    assert not missing, f"TROUBLESHOOTING.md federation diagnostics missing L37 codes: {missing}"
