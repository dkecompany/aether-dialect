"""
Template reuse sequence live tests.

The module has two groups. The first runs the real NL pipeline against the isolated ``reuse_runner`` fixture (parametrised sequence scenarios). The second runs seeded trust-promotion tests that call ``insert_template`` directly on a fresh in-memory store so the ``promote_trust`` state machine (trust ``0->1``, trust ``1->2`` threshold, reject-ratio block) is pinned without relying on the LLM.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import TemplateStats
from aetherdialect._contracts_core import FeedbackCounts
from aetherdialect._live_testing import LiveTestRunner, run_sequence_and_assert
from aetherdialect._templates import insert_template

from ._seed_helpers import (
    intent_customer_first_names,
    isolated_runner,
    seed_template,
)
from .mydb_scenarios import template_reuse_sequence_scenarios

_sequences = template_reuse_sequence_scenarios()


@pytest.fixture(scope="module")
def reuse_runner(schema, schema_terms, t2s) -> Iterator[LiveTestRunner]:
    """Runner whose template store stays isolated across the NL sequence tests in this module."""

    with isolated_runner(schema, schema_terms, t2s, label="reuse_seq") as runner:
        yield runner


@pytest.mark.live
@pytest.mark.parametrize("seq", _sequences, ids=[s.id for s in _sequences])
def test_template_reuse_sequence(reuse_runner: LiveTestRunner, seq) -> None:
    """Run sequence ensuring template reuse by running single-table before template-reuse steps."""

    run_sequence_and_assert(reuse_runner, seq)


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


@pytest.mark.live
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
        assert tmpl.trust_level == 2, (
            f"[TR-TRUST-1-2] expected trust_level == 2 after threshold merge; got {tmpl.trust_level}"
        )
        pair = tmpl.feedback_by_question[q_norm]
        assert pair.accepts == PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS, (
            f"[TR-TRUST-1-2] expected per-question accepts at promotion threshold; got {pair.accepts}"
        )


@pytest.mark.live
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
