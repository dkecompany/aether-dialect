"""Join feedback avoid-hints hard-filter LLM join-choice scopes."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._constants import JOIN_CHOICE_SCOPE_MAIN
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    FeedbackKind,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._sql_gen import get_join_choice_from_llm
from aetherdialect._templates import TemplateOps


def _feedback_row(candidate_id: str, *, effective_hash: str = "h") -> dict[str, object]:
    intent = RuntimeIntent(
        tables=["a", "b"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        chosen_join_candidate_id=candidate_id,
    )
    _, intent_payload = TemplateOps._compute_intent_structural_signature(intent)
    return QuestionFeedbackEntry(
        summary="wrong join path",
        buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash=effective_hash,
        intent_structural_hash="ih",
        intent_payload=intent_payload,
        created_at="t0",
        updated_at="t0",
    ).to_dict()


@pytest.mark.fast
def test_avoided_candidate_not_offered_to_llm() -> None:
    """Prior wrong-join feedback removes candidate ids from join-choice LLM scopes."""
    question = "orders and customers"
    store: dict[str, object] = {"question_feedback": {question: [_feedback_row("J01")]}}
    avoided = TemplateOps.lookup_join_avoid_candidate_ids_for_question(store, question)
    assert avoided == frozenset({"J01"})

    scopes = [
        {
            "scope": JOIN_CHOICE_SCOPE_MAIN,
            "tables": ["a", "b"],
            "candidates": [
                {
                    "candidate_id": "J01",
                    "join_path_signature": ["a.x->b.y"],
                    "candidate_tier": "base",
                },
                {
                    "candidate_id": "J02",
                    "join_path_signature": ["c.x->d.y"],
                    "candidate_tier": "base",
                },
            ],
        }
    ]
    captured_users: list[str] = []

    def _fake_llm(_system: str, user: str, **kwargs: object) -> dict[str, str]:
        captured_users.append(user)
        return {"choices": {JOIN_CHOICE_SCOPE_MAIN: "J02"}}

    with patch("aetherdialect._sql_gen.LLMProvider.json", side_effect=_fake_llm):
        got = get_join_choice_from_llm(
            question,
            "SELECT 1",
            llm_scopes=scopes,
            preset_choices={},
            accept_na_by_scope={JOIN_CHOICE_SCOPE_MAIN: False},
            avoided_candidate_ids=avoided,
        )

    assert got[JOIN_CHOICE_SCOPE_MAIN] == "J02"
    assert captured_users
    payload = json.loads(captured_users[0])
    offered_ids = [
        str(c.get("candidate_id"))
        for scope in payload.get("scopes", [])
        for c in scope.get("candidates", [])
        if isinstance(c, dict)
    ]
    assert "J01" not in offered_ids
    assert "J02" in offered_ids
