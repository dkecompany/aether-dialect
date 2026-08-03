"""Federation intake and question-level plan replay must pass the plan match gate and bind parameters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import FederationPlanTemplate
from aetherdialect._contracts_core import FederatedPlan, JoinSpec, RuntimeIntent, SourceStep
from aetherdialect._federation import federation_plan_step_fingerprints
from aetherdialect._pipeline import try_federation_plan_intake_reuse
from aetherdialect._utils import intent_key


@pytest.mark.fast
def test_intake_reuse_calls_plan_match_gate() -> None:
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="sg1",
        intent_key="ik",
        step_fingerprints=(("a", "fp"),),
        combine_hash="hash1",
        member_template_ids=(("a", "tmpl1"),),
    )
    composite = MagicMock()
    composite.schema_graph_id = "sg1"
    ref_tmpl = MagicMock()
    ref_tmpl.id = "tmpl1"
    ref_tmpl.value_history.questions = ["norm q"]
    ref_tmpl.value_history.param_values = [{"p1": 42}]
    with (
        patch(
            "aetherdialect._pipeline.lookup_federation_plan_template_for_question",
            return_value=cached,
        ),
        patch(
            "aetherdialect._pipeline._member_template_for_plan_template",
            return_value=ref_tmpl,
        ),
        patch(
            "aetherdialect._pipeline.federation_plan_matches_template",
            return_value=False,
        ) as mock_match,
        patch(
            "aetherdialect._pipeline.plan_federated_intent",
            return_value=FederatedPlan(steps=()),
        ),
    ):
        out = try_federation_plan_intake_reuse(
            "norm q",
            composite,
            MagicMock(),
            federation_dir="/fed",
            federation_manifest=MagicMock(),
            stores_by_source={"a": {"templates": {"tmpl1": ref_tmpl}}},
            member_graphs={"a": composite},
        )
    assert out is None
    mock_match.assert_called_once()


@pytest.mark.fast
def test_intake_reuse_binds_stored_param_values() -> None:
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="sg1",
        intent_key="ik",
        step_fingerprints=(("a", "fp"),),
        combine_hash="hash1",
        member_template_ids=(("a", "tmpl1"),),
    )
    composite = MagicMock()
    composite.schema_graph_id = "sg1"
    ref_tmpl = MagicMock()
    ref_tmpl.id = "tmpl1"
    ref_tmpl.value_history.questions = ["norm q"]
    ref_tmpl.value_history.param_values = [{"p1": 99}]
    captured_params: list[dict] = []

    def _capture_reuse(_q: str, _tmpl: object, params: dict, *_args: object, **_kwargs: object) -> None:
        captured_params.append(dict(params))
        return None

    with (
        patch(
            "aetherdialect._pipeline.lookup_federation_plan_template_for_question",
            return_value=cached,
        ),
        patch(
            "aetherdialect._pipeline._member_template_for_plan_template",
            return_value=ref_tmpl,
        ),
        patch("aetherdialect._pipeline._try_federation_plan_question_reuse", side_effect=_capture_reuse),
    ):
        try_federation_plan_intake_reuse(
            "norm q",
            composite,
            MagicMock(),
            federation_dir="/fed",
            federation_manifest=MagicMock(),
            stores_by_source={"a": {"templates": {"tmpl1": ref_tmpl}}},
            member_graphs={"a": composite},
        )
    assert captured_params == [{"p1": 99}]


@pytest.mark.fast
def test_question_reuse_rejects_stale_step_fingerprints() -> None:
    from aetherdialect._pipeline import _try_federation_plan_question_reuse

    composite = MagicMock()
    composite.schema_graph_id = "sg_composite"
    tmpl = MagicMock()
    tmpl.id = "tmpl1"
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="sg_composite",
        intent_key="ik",
        step_fingerprints=(("a", "stale_fp"),),
        combine_hash="hash1",
        member_template_ids=(("a", tmpl.id),),
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=intent),),
        combine=JoinSpec(
            left_source="a",
            right_source="b",
            left_key="id",
            right_key="id",
            logical_key="id",
            kind="inner",
        ),
    )
    with (
        patch(
            "aetherdialect._pipeline._resolve_federation_plan_template_for_reuse",
            return_value=cached,
        ),
        patch("aetherdialect._pipeline.plan_federated_intent", return_value=plan),
        patch(
            "aetherdialect._pipeline.federation_plan_step_fingerprints",
            return_value=federation_plan_step_fingerprints(plan, intent_key_fn=intent_key),
        ),
    ):
        out = _try_federation_plan_question_reuse(
            "norm_q",
            tmpl,
            {},
            composite,
            MagicMock(),
            federation_dir="/fed",
            federation_manifest=MagicMock(),
            stores_by_source={"a": {"templates": {tmpl.id: tmpl}}},
            member_graphs={"a": composite},
        )
    assert out is None
