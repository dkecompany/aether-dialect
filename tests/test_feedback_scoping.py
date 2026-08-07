"""Question feedback lookups are scoped to the current schema graph identity."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_core import FeedbackKind, QuestionFeedbackEntry, RejectionBucket, RuntimeIntent
from aetherdialect._pipeline import should_skip_intent_confirmation
from aetherdialect._templates import TemplateOps


def _intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _rejection_row(*, graph_id: str, summary: str = "bad join") -> dict:
    return QuestionFeedbackEntry(
        summary=summary,
        buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash=graph_id,
        intent_structural_hash="ih",
        intent_payload="{}",
        created_at="t1",
        updated_at="t2",
    ).to_dict() | {"schema_graph_id": graph_id}


@pytest.mark.fast
def test_rejection_from_previous_schema_does_not_block() -> None:
    old_id = "sg_old000000000005__eeee5555"
    current_id = "sg_new000000000006__ffff6666"
    store = {"question_feedback": {"q1": [_rejection_row(graph_id=old_id)]}}
    assert should_skip_intent_confirmation(_intent(), store, "q1", None, schema_graph_id=current_id) is True


@pytest.mark.fast
def test_join_hint_from_previous_schema_not_injected() -> None:
    old_id = "sg_old000000000007__gggg7777"
    current_id = "sg_new000000000008__hhhh8888"
    store = {
        "question_feedback": {
            "q1": [_rejection_row(graph_id=old_id, summary="stale join hint")],
        }
    }
    hints = TemplateOps.lookup_join_feedback_for_question(store, "q1", schema_graph_id=current_id)
    assert hints == []
