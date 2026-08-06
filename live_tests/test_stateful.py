"""Stateful sequence live pipeline tests."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_schema import TemplateStats
from aetherdialect._templates import (
    TemplateOps,
)

from ._seed_helpers import (
    intent_customer_first_names,
    intent_rental_count_by_store,
    isolated_runner,
    seed_negative_memory,
    seed_rejected,
    seed_template,
    seeded_runner,
)


@patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=False)
def test_store_persistence_roundtrip(_mock_no_llm, schema, schema_terms, t2s) -> None:
    """Seed a full-shape store on disk, then reload it and assert every section survives. Covers accepted templates and ``question_feedback`` rows (rejections, validation failures, penalties) so regressions in ``save_template_store`` / ``load_template_store`` are caught at the live layer."""
    with isolated_runner(schema, schema_terms, t2s, label="persist_rt") as runner:
        accepted = seed_template(
            runner,
            q_norm="list customer first names persisted",
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id, customer.first_name FROM customer",
            trust_level=2,
            stats=TemplateStats(accept=5, reject=1),
        )
        seed_rejected(
            runner,
            q_norm="list customer first names persisted-rej",
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id, customer.first_name FROM customer WHERE 1 = 0",
            reason="persisted rejection",
        )
        seed_negative_memory(
            runner,
            intent=intent_rental_count_by_store(),
            sql="SELECT store_id, COUNT(*) FROM rental GROUP BY store_id",
            reason="persisted negative memory",
            q_norm="persisted negative memory q",
        )
        seed_negative_memory(
            runner,
            intent=intent_customer_first_names(),
            sql="SELECT customer.customer_id FROM customer",
            reason="intent validation failed seeded",
            q_norm="unparseable seeded request",
        )

        TemplateOps.templates_to_store(runner.store, runner.templates)
        TemplateOps.save_template_store(runner.store)
        reloaded = TemplateOps.load_template_store(runner.schema.effective_structural_hash, runner.schema)

        assert accepted.id in reloaded["templates"], reloaded["templates"].keys()
        reloaded_templates = TemplateOps.store_to_templates(reloaded)
        assert reloaded_templates[accepted.id].trust_level == 2
        assert reloaded_templates[accepted.id].stats == TemplateStats(accept=5, reject=1)

        qf = reloaded.get("question_feedback") or {}
        rows_rej = qf.get("list customer first names persisted-rej") or []
        flat_rej = " ".join(str(r.get("summary", "")) for r in rows_rej if isinstance(r, dict))
        assert "persisted rejection" in flat_rej, qf

        rows_fail = qf.get("unparseable seeded request") or []
        flat_fail = " ".join(str(r.get("summary", "")) for r in rows_fail if isinstance(r, dict))
        assert "intent validation failed seeded" in flat_fail, qf

        rows_neg = qf.get("persisted negative memory q") or []
        flat_neg = " ".join(str(r.get("summary", "")) for r in rows_neg if isinstance(r, dict))
        assert "persisted negative memory" in flat_neg, qf


def test_seeded_per_pair_state_survives_save_load(schema, schema_terms, t2s) -> None:
    """The ``multi_pair_template`` kit's per-pair feedback dict survives a save/load round-trip. Asserts every seeded ``feedback_by_question`` entry — including pairs with zero accepts — comes back identical after the store is persisted and reloaded."""
    with seeded_runner(
        schema,
        schema_terms,
        t2s,
        label="persist_per_pair",
        kits=("multi_pair_template",),
    ) as runner:
        tid = runner.seeded_ids["multi_pair_template"]["template"]
        seeded_pairs = {
            qn: (fc.accepts, fc.rejects, fc.last_path) for qn, fc in runner.templates[tid].feedback_by_question.items()
        }

        TemplateOps.templates_to_store(runner.store, runner.templates)
        TemplateOps.save_template_store(runner.store)
        reloaded = TemplateOps.load_template_store(runner.schema.effective_structural_hash, runner.schema)
        reloaded_templates = TemplateOps.store_to_templates(reloaded)

        _missing_template_msg = f"[PER-PAIR] template {tid!r} missing after reload; got {sorted(reloaded_templates)!r}"
        assert tid in reloaded_templates, _missing_template_msg
        reloaded_pairs = {
            qn: (fc.accepts, fc.rejects, fc.last_path)
            for qn, fc in reloaded_templates[tid].feedback_by_question.items()
        }
        _diverged_msg = f"[PER-PAIR] feedback_by_question diverged: seeded={seeded_pairs!r} reloaded={reloaded_pairs!r}"
        assert reloaded_pairs == seeded_pairs, _diverged_msg
