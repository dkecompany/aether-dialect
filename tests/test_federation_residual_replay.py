"""Federation replay distinguishes combine shape from residual clauses."""

from aetherdialect._contracts_base import FederationPlanTemplate, OrderByCol
from aetherdialect._contracts_core import FederatedPlan, JoinSpec, ResidualSpec
from aetherdialect._federation import (
    federation_plan_combine_hash,
    federation_plan_matches_template,
    federation_plan_residual_hash,
)
from aetherdialect._intent_process import NormalizedExpr


def _join_plan(*, residual: ResidualSpec | None) -> FederatedPlan:
    return FederatedPlan(
        steps=(),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="inner",
            ),
        ),
        residual=residual,
    )


def test_identical_combine_shape_with_different_residual_does_not_replay() -> None:
    residual_a = ResidualSpec(
        order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column("left_t.id")),),
    )
    residual_b = ResidualSpec(
        order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column("right_t.id")),),
    )
    plan_a = _join_plan(residual=residual_a)
    plan_b = _join_plan(residual=residual_b)
    assert federation_plan_combine_hash(plan_a) == federation_plan_combine_hash(plan_b)
    assert federation_plan_residual_hash(plan_a) != federation_plan_residual_hash(plan_b)

    template = FederationPlanTemplate(
        plan_id="plan_a",
        composite_schema_graph_id="composite",
        intent_key="intent",
        step_fingerprints=(),
        combine_hash=federation_plan_combine_hash(plan_a),
        residual_hash=federation_plan_residual_hash(plan_a),
    )
    assert federation_plan_matches_template(plan_a, template, step_fingerprints=())
    assert not federation_plan_matches_template(plan_b, template, step_fingerprints=())
