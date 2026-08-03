"""Federation errors must not leak physical member labels in user-facing text."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import (
    FederationCapExceededError,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationPartialFailureError,
)
from aetherdialect._federation import federation_user_facing_error_message, probe_federation_member_connections
from aetherdialect._main_execution import PipelineSession, _federation_error_diagnostics


@pytest.mark.fast
def test_cap_exceeded_user_message_omits_physical_source_label() -> None:
    exc = FederationCapExceededError(
        "federation row cap exceeded for source 'secret_member': 100 rows > cap 50",
        limit_key="row_cap",
        source_id="secret_member",
    )
    message = federation_user_facing_error_message(exc)
    assert "secret_member" not in message
    assert exc.source_id == "secret_member"


@pytest.mark.fast
def test_member_execution_user_message_omits_physical_source_label() -> None:
    exc = FederationMemberExecutionError(
        "member secret_member failed: relation orders does not exist",
        source_id="secret_member",
        phase="member",
    )
    message = federation_user_facing_error_message(exc)
    assert "secret_member" not in message
    assert "orders" not in message


@pytest.mark.fast
def test_timeout_cap_user_message_omits_physical_source_label() -> None:
    from aetherdialect._federation import federation_member_timeout_error

    exc = federation_member_timeout_error("secret_member", RuntimeError("statement timeout"))
    message = federation_user_facing_error_message(exc)
    assert "secret_member" not in message
    assert exc.source_id == "secret_member"


@pytest.mark.fast
def test_partial_failure_user_message_is_generic_hint() -> None:
    exc = FederationPartialFailureError(
        "member b failed",
        source_id="b",
        phase="member",
        succeeded=(("a", 2, "2026-01-01T00:00:00+00:00"),),
    )
    message = federation_user_facing_error_message(exc)
    assert "member b" not in message
    assert "b" not in message.split()


@pytest.mark.fast
def test_federation_error_diagnostics_use_sanitized_message() -> None:
    exc = FederationCapExceededError(
        "federation semijoin key cap exceeded for member 'b': distinct keys on 'id' exceed cap 2",
        limit_key="semijoin_key_cap",
        source_id="b",
    )
    diags = _federation_error_diagnostics(exc)
    assert len(diags) == 1
    user_message = federation_user_facing_error_message(exc)
    assert diags[0].message == user_message
    assert "'b'" not in diags[0].message
    detail_map = dict(diags[0].details)
    assert detail_map["source_id"] == "b"
    assert detail_map["message"] == user_message
    assert detail_map["limit_key"] == "semijoin_key_cap"


@pytest.mark.fast
def test_probe_connection_error_omits_registration_key_and_driver_text() -> None:
    engine = MagicMock()
    engine.dialect = "postgresql"
    sa_engine = MagicMock()
    sa_engine.connect.side_effect = RuntimeError("password authentication failed for user postgres")
    engine._execution_engine = sa_engine
    with pytest.raises(FederationMemberProbeError) as exc_info:
        probe_federation_member_connections({"my_secret_db": engine})
    assert exc_info.value.source_id == "my_secret_db"
    message = federation_user_facing_error_message(exc_info.value)
    assert "my_secret_db" not in message
    assert "postgres" not in message
    assert "password" not in message


@pytest.mark.fast
def test_pipeline_session_terminal_error_omits_physical_source_label() -> None:
    owner = MagicMock()
    owner._audit_emit = None
    owner._schema_graph = None
    owner._llm_config = MagicMock(provider="mock")
    session = PipelineSession(owner)
    exc = FederationCapExceededError(
        "federation coordinator total input row cap exceeded for source 'b': 10 rows > cap 5",
        limit_key="total_input_row_cap",
        source_id="b",
    )
    step = session._terminal_error_from_exception(exc)
    assert step.error is not None
    assert "'b'" not in step.error
    assert step.federation_source_id == "b"
    assert step.federation_limit_key == "total_input_row_cap"
