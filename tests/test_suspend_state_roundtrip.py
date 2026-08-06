"""Suspend state export/restore round-trip."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from aetherdialect._constants import SUSPEND_STATE_FORMAT_VERSION
from aetherdialect._contracts_base import ConfigError
from aetherdialect._main_execution import MainExecutionOps, PipelineSession, PipelineSuspended


class _DummyPayload:
    def __init__(self, suspended_at: datetime | None = None) -> None:
        self.suspended_at = suspended_at or datetime.now(UTC)
        self.kind = "sql_confirm"


@pytest.mark.fast
def test_missing_payload_raises() -> None:
    with pytest.raises(ConfigError, match="hollow|missing"):
        MainExecutionOps.deserialize_suspended_state(
            {
                "format_version": SUSPEND_STATE_FORMAT_VERSION,
                "state_id": "sql_confirm",
                "message": "m",
                "choice_queue": [],
            }
        )


@pytest.mark.fast
def test_export_restore_continues_step(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = SimpleNamespace(
        limits=SimpleNamespace(suspended_session_ttl_seconds=3600),
        _schema_graph=None,
        _active_space_name="master",
        session=lambda **kwargs: PipelineSession(
            owner,
            mode=kwargs.get("mode", "writer"),
            space_name=kwargs.get("space", kwargs.get("space_name", "master")),
            data_row_cap=kwargs.get("data_row_cap"),
        ),
    )

    def fake_serialize(state_id: str, payload: object) -> dict:
        return {"type": "sql_confirm", "marker": "ok"}

    def fake_deserialize(state_id: str, raw: dict, *, owner=None) -> _DummyPayload:
        assert raw.get("marker") == "ok"
        return _DummyPayload()

    monkeypatch.setattr(MainExecutionOps, "_serialize_pipeline_suspend_payload", fake_serialize)
    monkeypatch.setattr(MainExecutionOps, "_deserialize_pipeline_suspend_payload", fake_deserialize)

    sess = PipelineSession(owner, mode="writer")
    sess._suspended = PipelineSuspended("sql_confirm", "confirm?", _DummyPayload())
    sess._session_busy = True
    sess._turn_question = "how many?"
    exported = sess.export_serialized_state()
    assert exported["format_version"] == SUSPEND_STATE_FORMAT_VERSION
    assert "payload" in exported
    assert "suspended_at" in exported
    assert exported["policy_ttl_seconds"] == 3600

    restored = PipelineSession.restore_serialized_state(owner, exported)
    assert restored.awaiting_prompt() is True
    assert restored._turn_question == "how many?"
    assert restored._restored_policy_ttl_seconds == 3600
