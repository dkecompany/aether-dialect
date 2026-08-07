"""RBAC scope vs warehouse AccessError outcomes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import AccessError, EngineContext
from aetherdialect._dialect import Dialect
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import _execution_scope_gate_active


@pytest.mark.fast
def test_default_owner_gate_inactive() -> None:
    assert _execution_scope_gate_active(EngineContext(), None, "owner", context_name="master") is False


@pytest.mark.fast
def test_scope_gate_uses_contact_admin() -> None:
    err = AccessError("execute", PERMISSION_DENIED_USER_MESSAGE, reason="scope")
    assert err.reason == "scope"
    assert str(err) == PERMISSION_DENIED_USER_MESSAGE


@pytest.mark.fast
def test_owner_warehouse_access_error_not_contact_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    noted: dict[str, object] = {}

    def _capture(choice_port: object, **kwargs: object) -> None:
        noted.update(kwargs)

    monkeypatch.setattr("aetherdialect._main_execution.note_interactive_turn", _capture)
    owner = SimpleNamespace(_schema_role="owner")
    port = SimpleNamespace(_owner=owner)
    MainExecutionOps._note_access_error_turn(
        port, AccessError("execute", "warehouse privilege missing", reason="warehouse")
    )
    assert noted.get("outcome") == "validation_failed"
    assert noted.get("error") == "warehouse privilege missing"


@pytest.mark.fast
def test_consumer_warehouse_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    noted: dict[str, object] = {}

    def _capture(choice_port: object, **kwargs: object) -> None:
        noted.update(kwargs)

    monkeypatch.setattr("aetherdialect._main_execution.note_interactive_turn", _capture)
    owner = SimpleNamespace(_schema_role="consumer")
    port = SimpleNamespace(_owner=owner)
    MainExecutionOps._note_access_error_turn(port, AccessError("execute", "secret table name xyz", reason="warehouse"))
    assert noted.get("outcome") == "permission_denied"
    assert noted.get("error") is None


@pytest.mark.fast
def test_undefined_table_not_permission_denied() -> None:
    assert Dialect.is_permission_denied_error("ERROR: undefinedtable: foo") is False
    assert Dialect.is_permission_denied_error("SQLSTATE 42P01") is False


@pytest.mark.fast
def test_insufficient_privilege_still_matches() -> None:
    assert Dialect.is_permission_denied_error("insufficient privilege to access table") is True
