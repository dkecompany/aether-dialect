"""Write-queue drain must persist template stores before truncating the queue."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import WRITE_QUEUE_FILENAME
from aetherdialect._contracts_core import FeedbackKind, QuestionFeedbackEntry, RejectionBucket, WriteQueueEvent
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import emit_write_queue_event


@pytest.mark.fast
def test_crash_after_apply_before_save_keeps_queue(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    graph_id = "sg_drain000000000001__abcd1234"
    store = TemplateOps.empty_template_store(graph_id)
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.schema_graph_id = graph_id
    owner._store = store
    owner._templates = TemplateOps.store_to_templates(store)
    owner._rejected = {}
    owner._dialect = None

    ts = datetime.now(UTC).isoformat()
    entry = QuestionFeedbackEntry(
        summary="s",
        buckets=(RejectionBucket.OTHER,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash=graph_id,
        intent_structural_hash="ik",
        intent_payload="{}",
        created_at=ts,
        updated_at=ts,
    )
    ev = WriteQueueEvent(
        kind="feedback_record",
        schema_graph_id=graph_id,
        schema_hash=graph_id,
        produced_at=ts,
        payload=(("q_norm", "q1"), ("entry_json", json.dumps(entry.to_dict()))),
    )
    emit_write_queue_event(str(tmp_path), ev, space_name="master")
    queue_path = tmp_path / WRITE_QUEUE_FILENAME
    original_bytes = queue_path.read_bytes()

    def _save_raises(_store: object) -> None:
        raise OSError("simulated crash before persist")

    monkeypatch.setattr("aetherdialect._templates_ops.TemplateOps.save_template_store", _save_raises)

    applied = MainExecutionOps.drain_write_queue(owner, str(tmp_path))
    assert applied == 0
    assert queue_path.read_bytes() == original_bytes
    assert "q1" not in store.question_feedback
