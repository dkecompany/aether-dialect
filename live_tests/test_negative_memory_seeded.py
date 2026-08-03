"""Seeded tests for question-level feedback (validation + intent rejections). Each group seeds the artifact it owns and asserts only the observable output of that subsystem: persisted rows under ``question_feedback``, prompt collection via :func:`aetherdialect._templates.collect_question_feedback_for_prompt`, and penalty mapping via :func:`aetherdialect._templates.compute_question_feedback_penalty`."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_core import FeedbackKind, Template
from aetherdialect._contracts_schema import TemplateStats
from aetherdialect._templates import (
    _compute_intent_structural_signature,
    collect_question_feedback_for_prompt,
    compute_question_feedback_penalty,
    record_question_feedback,
    summarize_failure_for_memory,
)

from ._seed_helpers import (
    intent_customer_first_names,
    intent_rental_count_by_store,
    isolated_runner,
    seed_negative_memory,
    seed_rejected,
    seed_template,
)


def _template_penalty(store: dict, template: Template) -> float:
    """Penalty used by the pipeline's confidence path for *template*'s question + keys."""
    q_hist = (template.value_history.questions or [""])[0]
    return compute_question_feedback_penalty(store, q_hist, template.effective_structural_hash)


_FAILURE_Q_NORM = "list customer first names with unparseable filter"
_FAILURE_MESSAGE = "seeded intent validation issue: missing column"


@patch("aetherdialect._templates.llm_credentials_configured", return_value=False)
def test_validation_failure_feedback_surfaces_for_prompt(_mock_no_llm, schema, schema_terms, t2s) -> None:
    """Seeded structural validation row surfaces through ``collect_question_feedback_for_prompt``."""
    intent = intent_customer_first_names()
    ish, _ = _compute_intent_structural_signature(intent)
    sql = "SELECT customer.customer_id, customer.first_name FROM customer"
    with isolated_runner(schema, schema_terms, t2s, label="nm_hint") as runner:
        seed_negative_memory(
            runner,
            intent=intent,
            sql=sql,
            reason=_FAILURE_MESSAGE,
            q_norm=_FAILURE_Q_NORM,
        )
        rows = collect_question_feedback_for_prompt(
            runner.store, _FAILURE_Q_NORM, runner.schema.effective_structural_hash
        )
        assert rows, f"[NM-HINT] expected >=1 row; got {rows!r}"
        assert rows[0].get("kind") == FeedbackKind.VALIDATION_FAILURE.value
        assert rows[0].get("intent_structural_hash") == ish
        flat = " ".join(r.get("summary", "") for r in rows)
        assert _FAILURE_MESSAGE in flat, f"[NM-HINT] expected seeded hint; got {rows!r}"


@patch("aetherdialect._templates.llm_credentials_configured", return_value=False)
def test_validation_failure_rows_scoped_to_schema(_mock_no_llm, schema, schema_terms, t2s) -> None:
    """Structural validation rows do not leak across schema hashes."""
    intent = intent_customer_first_names()
    with isolated_runner(schema, schema_terms, t2s, label="nm_hint_scope") as runner:
        ent = summarize_failure_for_memory(
            question=_FAILURE_Q_NORM,
            intent=intent,
            kind=FeedbackKind.VALIDATION_FAILURE,
            schema_hash="different-schema-hash",
            validator_errors=[_FAILURE_MESSAGE],
        )
        record_question_feedback(runner.store, _FAILURE_Q_NORM, ent)
        rows = collect_question_feedback_for_prompt(
            runner.store, _FAILURE_Q_NORM, runner.schema.effective_structural_hash
        )
        assert rows == [], f"[NM-HINT-SCOPE] expected no hints; got {rows!r}"


_REJECT_Q_NORM = "list customer first names with bad aggregation"


def test_rejected_template_seeds_avoid_example(schema, schema_terms, t2s) -> None:
    """A seeded rejection shows up as an avoid-intent example for matching questions."""
    with isolated_runner(schema, schema_terms, t2s, label="nm_avoid") as runner:
        intent_avoid = intent_customer_first_names()
        ish, _ = _compute_intent_structural_signature(intent_avoid)
        rt = seed_rejected(
            runner,
            q_norm=_REJECT_Q_NORM,
            intent=intent_avoid,
            sql="SELECT customer.customer_id, customer.first_name FROM customer",
            reason="seeded wrong-aggregation rejection",
        )
        rows = collect_question_feedback_for_prompt(
            runner.store, _REJECT_Q_NORM, runner.schema.effective_structural_hash
        )
        assert rows, f"[NM-AVOID] expected >=1 feedback row for {rt.id!r}; got {rows!r}"
        assert rows[0].get("kind") == FeedbackKind.INTENT_REJECTED.value
        assert rows[0].get("intent_structural_hash") == ish
        bucket_csv = (rows[0].get("buckets") or "").strip()
        assert bucket_csv, "[NM-AVOID] expected non-empty buckets field"
        summary = (rows[0].get("summary") or "").strip()
        assert summary, "[NM-AVOID] expected non-empty summary"


def test_negative_memory_penalty_applies_to_matching_template(schema, schema_terms, t2s) -> None:
    """Seeded validation rows increase ``compute_question_feedback_penalty`` for the template's question. One append must add at least ``PEN_BY_THREE_SOURCE_UNIT`` and stay under ``PENALTY_CAP``."""
    rental_intent = intent_rental_count_by_store()
    sql = "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental JOIN inventory ON rental.inventory_id = inventory.inventory_id GROUP BY inventory.store_id"
    q_seed = "rental count per store seeded"
    with isolated_runner(schema, schema_terms, t2s, label="nm_penalty") as runner:
        template = seed_template(
            runner,
            q_norm=q_seed,
            intent=rental_intent,
            sql=sql,
            trust_level=2,
            stats=TemplateStats(accept=4, reject=1),
        )
        baseline = _template_penalty(runner.store, template)
        assert baseline == 0.0, f"[NM-PENALTY-BASE] expected zero baseline penalty; got {baseline}"

        seed_negative_memory(
            runner,
            intent=rental_intent,
            sql=sql,
            reason="seeded structural rejection",
            q_norm=q_seed,
        )
        penalty = _template_penalty(runner.store, template)
        expected = min(PolicyConfig.PENALTY_CAP, PolicyConfig.PEN_BY_THREE_SOURCE_UNIT)
        _penalty_msg = f"[NM-PENALTY] expected {expected}; got {penalty} (unit={PolicyConfig.PEN_BY_THREE_SOURCE_UNIT})"
        assert abs(penalty - expected) < 1e-6, _penalty_msg


def test_negative_memory_penalty_caps_at_policy_limit(schema, schema_terms, t2s) -> None:
    """Repeated failure-log rows are bounded by ``PolicyConfig.PENALTY_CAP``."""
    rental_intent = intent_rental_count_by_store()
    sql = "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental JOIN inventory ON rental.inventory_id = inventory.inventory_id GROUP BY inventory.store_id"
    q_seed = "rental count per store seeded-cap"
    with isolated_runner(schema, schema_terms, t2s, label="nm_penalty_cap") as runner:
        seed_template(
            runner,
            q_norm=q_seed,
            intent=rental_intent,
            sql=sql,
            trust_level=2,
            stats=TemplateStats(accept=4, reject=1),
        )
        seed_negative_memory(
            runner,
            intent=rental_intent,
            sql=sql,
            reason="seeded rejection cap",
            repeats=20,
            q_norm=q_seed,
        )
        pen = compute_question_feedback_penalty(runner.store, q_seed, runner.schema.effective_structural_hash)
        _cap_msg = f"[NM-PENALTY-CAP] penalty exceeds cap: {pen} > {PolicyConfig.PENALTY_CAP}"
        assert pen <= PolicyConfig.PENALTY_CAP + 1e-9, _cap_msg
