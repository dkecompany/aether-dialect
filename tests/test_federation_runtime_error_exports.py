"""Public exports for attributable federation runtime error subclasses."""

from __future__ import annotations

import pytest

import aetherdialect
from aetherdialect import (
    FederationCapExceededError,
    FederationMemberExecutionError,
    FederationRuntimeError,
)


@pytest.mark.fast
def test_federation_member_execution_error_is_public_and_attributable() -> None:
    assert "FederationMemberExecutionError" in aetherdialect.__all__
    assert issubclass(FederationMemberExecutionError, FederationRuntimeError)
    exc = FederationMemberExecutionError("member failed", source_id="east", phase="member")
    assert isinstance(exc, FederationRuntimeError)
    assert exc.source_id == "east"
    assert exc.phase == "member"
    with pytest.raises(FederationRuntimeError):
        raise exc


@pytest.mark.fast
def test_federation_cap_exceeded_error_is_public_and_attributable() -> None:
    assert "FederationCapExceededError" in aetherdialect.__all__
    assert issubclass(FederationCapExceededError, FederationRuntimeError)
    exc = FederationCapExceededError("too many rows", limit_key="row_cap", source_id="west")
    assert isinstance(exc, FederationRuntimeError)
    assert exc.limit_key == "row_cap"
    assert exc.source_id == "west"
    with pytest.raises(FederationRuntimeError):
        raise exc


@pytest.mark.fast
def test_federation_cap_exceeded_error_coordinator_source_id_defaults_empty() -> None:
    exc = FederationCapExceededError("coordinator cap", limit_key="total_input_row_cap")
    assert exc.limit_key == "total_input_row_cap"
    assert exc.source_id == ""
