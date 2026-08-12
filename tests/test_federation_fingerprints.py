"""Federation plan step fingerprints must include member join paths."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SelectCol, SourceStep
from aetherdialect._federation_execute import federation_plan_step_fingerprints
from aetherdialect._templates import TemplateRefs
from aetherdialect._utils_intent import intent_key


def _member_join_intent(*, signature: list[str], candidate_id: str = "J01") -> RuntimeIntent:
    return RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_candidate_id=candidate_id,
        chosen_join_path_signature=list(signature),
    )


def _plan_with_member_join(signature: list[str], *, candidate_id: str = "J01") -> FederatedPlan:
    sub_intent = _member_join_intent(signature=signature, candidate_id=candidate_id)
    return FederatedPlan(
        steps=(SourceStep(source_id="west", sub_intent=sub_intent),),
    )


@pytest.mark.fast
def test_step_fingerprints_differ_on_member_join_path() -> None:
    direct_sig = ["child.parent_id->parent.id"]
    bridge_sig = ["child.bridge_id->bridge.id", "bridge.parent_id->parent.id"]

    plan_direct = _plan_with_member_join(direct_sig)
    plan_bridge = _plan_with_member_join(bridge_sig, candidate_id="J02")

    assert intent_key(plan_direct.steps[0].sub_intent) == intent_key(plan_bridge.steps[0].sub_intent)
    assert TemplateRefs.join_fingerprint_from_runtime_intent(
        plan_direct.steps[0].sub_intent
    ) != TemplateRefs.join_fingerprint_from_runtime_intent(plan_bridge.steps[0].sub_intent)

    fps_direct = federation_plan_step_fingerprints(plan_direct, intent_key_fn=intent_key)
    fps_bridge = federation_plan_step_fingerprints(plan_bridge, intent_key_fn=intent_key)

    assert fps_direct != fps_bridge
    assert fps_direct[0][0] == fps_bridge[0][0] == "west"
    assert fps_direct[0][1] != fps_bridge[0][1]


@pytest.mark.fast
def test_step_fingerprint_includes_join_path_segment_hash() -> None:
    plan = _plan_with_member_join(["child.parent_id->parent.id"])
    join_fp = TemplateRefs.join_fingerprint_from_runtime_intent(plan.steps[0].sub_intent)
    fps = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key)
    assert join_fp
    assert join_fp in fps[0][1]
