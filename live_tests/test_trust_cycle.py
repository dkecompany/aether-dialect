"""
Trust-cycle deterministic + seeded live tests.

The bulk of the trust state machine is exercised by directly calling ``record_template_feedback`` / ``record_per_question_feedback`` / ``promote_trust`` against pre-seeded templates so each transition can be asserted in isolation; one live NL sequence remains as an end-to-end smoke.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._live_testing import LiveTestRunner, run_sequence_and_assert
from aetherdialect._templates import (
    promote_trust,
    record_per_question_feedback,
    record_template_feedback,
)

from ._seed_helpers import isolated_runner, seeded_runner, snapshot_store
from .mydb_scenarios import trust_cycle_scenarios

_sequences = trust_cycle_scenarios()


@pytest.fixture(scope="module")
def trust_runner(schema, schema_terms, t2s) -> Iterator[LiveTestRunner]:
    """Runner whose template store stays isolated for the live trust-cycle smoke."""

    with isolated_runner(schema, schema_terms, t2s, label="trust_seq") as runner:
        yield runner


@pytest.mark.live
@pytest.mark.parametrize("seq", _sequences[:1], ids=[_sequences[0].id])
def test_trust_cycle_sequence_smoke(trust_runner: LiveTestRunner, seq) -> None:
    """End-to-end NL smoke retained for one trust-cycle sequence."""

    run_sequence_and_assert(trust_runner, seq)


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
        assert after_pair[1] == before_pair[1] + 1, (
            f"[TC-REJECT] pair rejects must increment: before={before_pair!r} after={after_pair!r}"
        )
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
