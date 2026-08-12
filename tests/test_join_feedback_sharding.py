"""Federation join feedback sharded out of plan template records."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aetherdialect._constants import (
    FEDERATION_JOIN_FEEDBACK_PREFIX,
    FEDERATION_TEMPLATES_SEGMENT,
)
from aetherdialect._contracts_schema import FederationPlanTemplate
from aetherdialect._federation_execute import (
    load_federation_plan_templates,
    lookup_federation_join_feedback,
    record_federation_join_feedback,
    save_federation_plan_template,
)
from aetherdialect._federation_manifest import federation_artifact_paths
from aetherdialect._templates import TemplateStoreView
from aetherdialect._utils import normalize_question


def _plan_template(plan_id: str = "plan_1", question: str = "join left and right") -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=plan_id,
        composite_schema_graph_id="sg_composite",
        intent_key="ik_1",
        step_fingerprints=(("a", "ik_a"), ("b", "ik_b")),
        combine_hash="combine_hash",
        question=question,
    )


def _plan_file_bytes(federation_dir: str) -> int:
    path = federation_artifact_paths(federation_dir)["plan_templates"]
    return os.path.getsize(path)


@pytest.mark.fast
def test_plan_file_size_independent_of_rejection_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        template = _plan_template()
        save_federation_plan_template(tmp, template)
        record_federation_join_feedback(tmp, template.plan_id, "wrong join path once")
        size_one = _plan_file_bytes(tmp)

        for i in range(100):
            record_federation_join_feedback(tmp, template.plan_id, f"wrong join path rejection {i}")
        size_many = _plan_file_bytes(tmp)

        assert size_one == size_many
        assert lookup_federation_join_feedback(tmp, template.plan_id)
        loaded = load_federation_plan_templates(tmp)[template.plan_id]
        assert not loaded.join_feedback

        q_norm = normalize_question(template.question)
        part = TemplateStoreView.question_feedback_partition_number(q_norm)
        feedback_dir = os.path.join(tmp, FEDERATION_TEMPLATES_SEGMENT, "feedback")
        shard_name = f"{FEDERATION_JOIN_FEEDBACK_PREFIX}{part:02x}.json.gz"
        assert os.path.isfile(os.path.join(feedback_dir, shard_name))

        plan_payload = json.loads(open(federation_artifact_paths(tmp)["plan_templates"], encoding="utf-8").read())
        for row in plan_payload.values():
            assert "join_feedback" not in row
