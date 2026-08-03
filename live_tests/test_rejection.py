"""Rejection feedback loop deterministic + seeded live pipeline tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._contracts_core import FeedbackKind, GenerationPath, UserFeedbackRejectSuspendContext
from aetherdialect._dialect import get_dialect
from aetherdialect._pipeline import complete_user_feedback_reject
from aetherdialect._templates import _compute_intent_structural_signature
from aetherdialect._utils import intent_key

from ._seed_helpers import (
    assert_new_rejected_template,
    intent_customer_first_names,
    isolated_runner,
    seed_rejected,
    seed_template,
    snapshot_store,
)

_REJECTION_CASES = [
    ("RJ-001", "list customer first names rj001", "wrong columns selected"),
    ("RJ-002", "list customer first names rj002", "too many rows"),
    ("RJ-003", "list customer first names rj003", "wrong intent"),
]


@pytest.mark.parametrize(
    ("scenario_id", "q_norm", "reject_reason"),
    _REJECTION_CASES,
    ids=[case[0] for case in _REJECTION_CASES],
)
@patch("aetherdialect._templates.llm_credentials_configured", return_value=False)
def test_seeded_rejection_feedback(
    _mock_no_llm,
    schema,
    schema_terms,
    t2s,
    scenario_id: str,
    q_norm: str,
    reject_reason: str,
) -> None:
    """Pre-seed an accepted template, then drive ``complete_user_feedback_reject`` directly. Asserts the rejection bookkeeping writes ``question_feedback`` for this question with the canonicalised reason recorded."""
    intent = intent_customer_first_names()
    sql = "SELECT customer.customer_id, customer.first_name FROM customer"
    ish, _ = _compute_intent_structural_signature(intent)
    with isolated_runner(schema, schema_terms, t2s, label=f"rej_{scenario_id.lower()}") as runner:
        tmpl = seed_template(
            runner,
            q_norm=q_norm,
            intent=intent,
            sql=sql,
            trust_level=2,
        )
        before = snapshot_store(runner)
        ctx = UserFeedbackRejectSuspendContext(
            intent=intent,
            sql=sql,
            schema=runner.schema,
            store=runner.store,
            templates=runner.templates,
            rejected={},
            q_norm=q_norm,
            generation_path=GenerationPath.UNION_TEMPLATE_WIDEN,
            matched_template=tmpl,
            matched_rejected_template=None,
            dialect=get_dialect(),
        )
        result = complete_user_feedback_reject(ctx, needs_reason=True, reject_reason=reject_reason)

        assert result is not None, f"[{scenario_id}] complete_user_feedback_reject returned None"
        assert result["normalized_reason"], f"[{scenario_id}] empty normalized_reason"
        after = snapshot_store(runner)
        assert_new_rejected_template(before, after)
        qf_rows = (runner.store.get("question_feedback") or {}).get(q_norm, [])
        hashes = {str(r.get("intent_structural_hash", "")) for r in qf_rows if isinstance(r, dict)}
        kinds = {str(r.get("kind", "")) for r in qf_rows if isinstance(r, dict)}
        assert ish in hashes, f"[{scenario_id}] intent_structural_hash {ish!r} not in {hashes!r}"
        assert FeedbackKind.INTENT_REJECTED.value in kinds, f"[{scenario_id}] kinds={kinds!r}"
        summaries = " ".join(str(row.get("summary", "")) for row in qf_rows if isinstance(row, dict))
        assert result["normalized_reason"] in summaries, (
            f"[{scenario_id}] reason {result['normalized_reason']!r} missing from "
            f"question_feedback summaries={summaries!r}"
        )


def test_seeded_structural_rejection_bucket(schema, schema_terms, t2s) -> None:
    """Seeding a rejection records the intent_key + reason via ``seed_rejected`` (question_feedback). Asserts the legacy ``negative_memory`` store section is absent (memory lives under ``question_feedback``)."""
    ikey = intent_key(intent_customer_first_names())
    with isolated_runner(schema, schema_terms, t2s, label="rej_struct") as runner:
        rt = seed_rejected(
            runner,
            q_norm="list customer first names seeded",
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id, customer.first_name FROM customer WHERE 1 = 0",
            reason="seeded structural rejection",
        )
        assert rt.intent_key == ikey
        assert "seeded structural rejection" in rt.value_history.rejection_reasons
        assert "negative_memory" not in runner.store


def test_seeded_semantic_rejection_bucket(schema, schema_terms, t2s) -> None:
    """Seeding a semantic-style rejection records the reason on the synthetic seed row."""
    ikey = intent_key(intent_customer_first_names())
    with isolated_runner(schema, schema_terms, t2s, label="rej_sem") as runner:
        rt = seed_rejected(
            runner,
            q_norm="list customer first names semantic",
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id, customer.first_name FROM customer",
            reason="seeded semantic rejection",
        )
        assert rt.intent_key == ikey
        assert "seeded semantic rejection" in rt.value_history.rejection_reasons
