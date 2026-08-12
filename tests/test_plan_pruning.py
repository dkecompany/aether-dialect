"""Federation plan pruning must inspect every step and sanitize surviving templates."""

from __future__ import annotations

import json
import tempfile

import pytest

from aetherdialect._contracts_schema import FederationPlanTemplate
from aetherdialect._federation_execute import (
    load_federation_plan_templates,
    prune_federation_plan_templates_for_sources,
    save_federation_plan_template,
)
from aetherdialect._federation_manifest import federation_artifact_paths


def _plan(
    plan_id: str,
    *,
    steps: tuple[tuple[str, str], ...],
    member_template_ids: tuple[tuple[str, str], ...] = (),
) -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=plan_id,
        composite_schema_graph_id="cg_test",
        intent_key=f"ik_{plan_id}",
        step_fingerprints=steps,
        combine_hash="combine_hash",
        question="cross-source question",
        accepted_questions=("accepted",),
        format_version="0.2.3",
        member_template_ids=member_template_ids,
        residual_hash="residual_hash",
        join_feedback=(),
        manifest_hash="manifest_hash",
        member_tuple_hash="member_tuple_hash",
    )


@pytest.mark.fast
def test_plan_referencing_removed_source_in_any_step_is_pruned() -> None:
    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(
            fed_dir,
            _plan(
                "second_step_alpha",
                steps=(("beta", "fp_beta"), ("alpha", "fp_alpha")),
                member_template_ids=(("beta", "T0002"),),
            ),
        )
        save_federation_plan_template(
            fed_dir,
            _plan(
                "survives_with_stale_member_ids",
                steps=(("beta", "fp_beta"),),
                member_template_ids=(("beta", "T0002"), ("alpha", "T0001")),
            ),
        )

        prune_federation_plan_templates_for_sources(fed_dir, {"alpha"})

        loaded = load_federation_plan_templates(fed_dir)
        assert "second_step_alpha" not in loaded
        survivor = loaded["survives_with_stale_member_ids"]
        assert survivor.step_fingerprints == (("beta", "fp_beta"),)
        assert survivor.member_template_ids == (("beta", "T0002"),)

        path = federation_artifact_paths(fed_dir)["plan_templates"]
        raw = json.loads(open(path, encoding="utf-8").read())
        assert raw["survives_with_stale_member_ids"]["member_template_ids"] == [["beta", "T0002"]]
