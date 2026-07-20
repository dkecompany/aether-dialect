"""Seeded live tests for template trust promotion and per-question feedback."""

from __future__ import annotations

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_core import FeedbackCounts
from aetherdialect._contracts_schema import TemplateStats
from aetherdialect._live_testing import LiveTestRunner
from aetherdialect._templates import (
    insert_template,
    promote_trust,
    record_per_question_feedback,
    record_template_feedback,
)

from ._seed_helpers import (
    intent_customer_first_names,
    isolated_runner,
    seed_template,
    seeded_runner,
    snapshot_store,
)


def _merge_same_template(runner: LiveTestRunner, q_norm: str) -> None:
    """Call ``insert_template`` with the seeded fingerprint to trigger ``_merge_accept``."""
    insert_template(
        runner.store,
        runner.templates,
        runner.schema,
        q_norm,
        intent_customer_first_names(),
        "SELECT customer.customer_id, customer.first_name FROM customer",
    )


def test_trust_promotion_one_to_two_at_threshold(schema, schema_terms, t2s) -> None:
    """Trust ``1 -> 2`` fires on first merge with low reject ratio."""
    with isolated_runner(schema, schema_terms, t2s, label="trust_1_2") as runner:
        threshold = PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS
        prior_accepts = max(threshold - 1, 0)
        seeded = seed_template(
            runner,
            q_norm="list customer first names",
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id, customer.first_name FROM customer",
            trust_level=1,
            stats=TemplateStats(accept=prior_accepts, reject=0),
        )
        seeded.feedback_by_question["list customer first names"] = FeedbackCounts(
            accepts=prior_accepts,
            rejects=0,
            last_path=1,
        )
        _merge_same_template(runner, "list customer first names")
        q_norm = "list customer first names"
        tmpl = runner.templates[seeded.id]
        _trust_msg = f"[TR-TRUST-1-2] expected trust_level == 2 after threshold merge; got {tmpl.trust_level}"
        assert tmpl.trust_level == 2, _trust_msg
        pair = tmpl.feedback_by_question[q_norm]
        _accepts_msg = f"[TR-TRUST-1-2] expected per-question accepts at promotion threshold; got {pair.accepts}"
        assert pair.accepts == PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS, _accepts_msg


def test_trust_promotion_reject_ratio_blocks(schema, schema_terms, t2s) -> None:
    """A reject ratio above ``TRUST_PROMOTE_MAX_REJECT_RATIO`` keeps the template at trust 1."""
    with isolated_runner(schema, schema_terms, t2s, label="trust_block") as runner:
        accept_seed = 1
        reject_seed = max(
            1,
            int(round((accept_seed + 1) * PolicyConfig.TRUST_PROMOTE_MAX_REJECT_RATIO + 1)),
        )
        seeded = seed_template(
            runner,
            q_norm="list customer first names",
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id, customer.first_name FROM customer",
            trust_level=1,
            stats=TemplateStats(accept=accept_seed, reject=reject_seed),
        )
        _merge_same_template(runner, "list customer first names")
        assert runner.templates[seeded.id].trust_level == 1, (
            "[TR-TRUST-BLOCK] reject ratio above TRUST_PROMOTE_MAX_REJECT_RATIO must not promote; got "
            f"trust_level={runner.templates[seeded.id].trust_level} accept={runner.templates[seeded.id].stats.accept} "
            f"reject={runner.templates[seeded.id].stats.reject}"
        )


def test_seeded_promotion_when_per_pair_accepts_meets_threshold(schema, schema_terms, t2s) -> None:
    """A trust=1 template promotes to 2 once per-pair accepts crosses the policy threshold."""
    with seeded_runner(schema, schema_terms, t2s, label="trust_promote_ok", kits=("cold_templates",)) as runner:
        tid = runner.seeded_ids["cold_templates"]["first_names_cold"]
        tmpl = runner.templates[tid]
        q_norm = "list customer first names"
        for _ in range(PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS):
            record_template_feedback(tmpl, accept=True)
            record_per_question_feedback(tmpl, q_norm, accept=True, path=1)

        promoted = promote_trust(tmpl, q_norm)

        assert promoted is True, (
            f"[TC-PROMOTE-OK] expected promote_trust True; trust={tmpl.trust_level} "
            f"counts={tmpl.feedback_by_question.get(q_norm)!r}"
        )
        assert tmpl.trust_level == 2, f"[TC-PROMOTE-OK] expected trust_level=2 after promotion; got {tmpl.trust_level}"


def test_seeded_promotion_blocked_by_excess_reject_ratio(schema, schema_terms, t2s) -> None:
    """Promotion is blocked when the template's overall reject ratio exceeds the policy ceiling."""
    with seeded_runner(
        schema,
        schema_terms,
        t2s,
        label="trust_promote_blocked",
        kits=("cold_templates",),
    ) as runner:
        tid = runner.seeded_ids["cold_templates"]["first_names_cold"]
        tmpl = runner.templates[tid]
        q_norm = "list customer first names"
        for _ in range(PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS):
            record_per_question_feedback(tmpl, q_norm, accept=True, path=1)
        tmpl.stats.accept = 1
        tmpl.stats.reject = 9

        promoted = promote_trust(tmpl, q_norm)

        assert promoted is False, (
            f"[TC-PROMOTE-BLOCKED] expected promote_trust False under reject ratio "
            f"{tmpl.stats.reject}/{tmpl.stats.accept + tmpl.stats.reject}"
        )
        assert tmpl.trust_level == 1, f"[TC-PROMOTE-BLOCKED] trust must remain at 1; got {tmpl.trust_level}"


def test_seeded_reject_increments_pair_counts(schema, schema_terms, t2s) -> None:
    """Recording a reject increments per-pair reject count and overall stats.reject."""
    with seeded_runner(schema, schema_terms, t2s, label="trust_reject", kits=("baseline_templates",)) as runner:
        tid = runner.seeded_ids["baseline_templates"]["first_names"]
        tmpl = runner.templates[tid]
        q_norm = "list customer first names"
        before = snapshot_store(runner)
        before_pair = before["feedback_by_question_by_id"][tid][q_norm]

        record_template_feedback(tmpl, accept=False)
        record_per_question_feedback(tmpl, q_norm, accept=False, path=1)

        after_pair = (
            tmpl.feedback_by_question[q_norm].accepts,
            tmpl.feedback_by_question[q_norm].rejects,
            tmpl.feedback_by_question[q_norm].last_path,
        )
        _reject_pair_msg = f"[TC-REJECT] pair rejects must increment: before={before_pair!r} after={after_pair!r}"
        assert after_pair[1] == before_pair[1] + 1, _reject_pair_msg
        assert tmpl.trust_level == 2, f"[TC-REJECT] trust=2 is terminal; got {tmpl.trust_level}"


def test_seeded_promotion_no_op_without_q_norm(schema, schema_terms, t2s) -> None:
    """``promote_trust`` returns False when ``q_norm`` is empty regardless of stats."""
    with seeded_runner(
        schema,
        schema_terms,
        t2s,
        label="trust_promote_no_qnorm",
        kits=("cold_templates",),
    ) as runner:
        tid = runner.seeded_ids["cold_templates"]["first_names_cold"]
        tmpl = runner.templates[tid]

        assert promote_trust(tmpl, "") is False
        assert tmpl.trust_level == 1
