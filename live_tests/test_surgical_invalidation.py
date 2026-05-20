"""Live checks for template-store helpers that prune rows against the current graph."""

from __future__ import annotations

import pytest

from aetherdialect._templates import (
    TemplateRefs,
    reconcile_template_store,
    template_is_live,
    templates_to_store,
)

from ._seed_helpers import seeded_runner


@pytest.mark.live_no_llm
def test_reconcile_empty_store_round_trip(schema) -> None:
    """Reconciliation leaves an empty store unchanged and reports no drops."""

    store: dict = {
        "templates": {},
        "question_feedback": {},
    }
    report = reconcile_template_store(store, schema)
    assert report.dropped_template_ids == ()
    assert report.kept_template_ids == ()


@pytest.mark.live_no_llm
def test_template_is_live_flags_missing_table(schema) -> None:
    """``template_is_live`` reports a missing table against the live graph."""

    refs = TemplateRefs(
        tables=frozenset({"__nonexistent_relation__"}),
        columns=frozenset(),
        fk_edges=frozenset(),
    )
    ok, reasons = template_is_live(refs, schema)
    assert ok is False
    assert any(r.startswith("missing_table:") for r in reasons)


def test_reconcile_preserves_seeded_baseline_templates(schema, schema_terms, t2s) -> None:
    """All four ``baseline_templates`` rows survive reconciliation against the live schema."""

    with seeded_runner(
        schema,
        schema_terms,
        t2s,
        label="reconcile_baseline",
        kits=("baseline_templates",),
    ) as runner:
        seeded_ids = set(runner.seeded_ids["baseline_templates"].values())
        templates_to_store(runner.store, runner.templates)

        report = reconcile_template_store(runner.store, runner.schema)

        kept = set(report.kept_template_ids)
        assert seeded_ids.issubset(kept), (
            f"[RECONCILE-BASELINE] expected all seeded ids preserved; "
            f"missing={sorted(seeded_ids - kept)!r} dropped={list(report.dropped_template_ids)!r}"
        )
        assert report.dropped_template_ids == (), (
            f"[RECONCILE-BASELINE] no drops expected; got {list(report.dropped_template_ids)!r} "
            f"reasons={report.reason_histogram!r}"
        )


def test_reconcile_preserves_seeded_rejected_aggregations(schema, schema_terms, t2s) -> None:
    """``rejected_aggregations`` question_feedback keys survive reconciliation against the live schema."""

    _q_norms = (
        "rentals per store wrong agg",
        "payments per staff wrong agg",
    )
    with seeded_runner(
        schema,
        schema_terms,
        t2s,
        label="reconcile_rej_agg",
        kits=("rejected_aggregations",),
    ) as runner:
        for qn in _q_norms:
            assert qn in (runner.store.get("question_feedback") or {}), runner.store.get("question_feedback")
        report = reconcile_template_store(runner.store, runner.schema)

        assert report.dropped_template_ids == ()
        for qn in _q_norms:
            assert qn in (runner.store.get("question_feedback") or {}), (
                f"[RECONCILE-REJ-AGG] expected feedback key {qn!r} preserved; got {runner.store.get('question_feedback')!r}"
            )
        assert report.dropped_negative_memory_bucket_count == 0
