"""Residual predicate columns must appear in coordinator combine projections."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import FederatedPlan, ResidualSpec, RuntimeIntent, SourceStep
from aetherdialect._federation_plan import combine_select_column_names


@pytest.mark.fast
def test_combine_select_harvests_residual_where_column() -> None:
    sub_intent = RuntimeIntent(
        tables=[],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(
            SourceStep(source_id="a", sub_intent=sub_intent),
            SourceStep(source_id="b", sub_intent=sub_intent),
        ),
        residual=ResidualSpec(
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("filter_col"),
                        op="=",
                        value_type="string",
                        param_key="p1",
                        raw_value="1",
                    ),
                ],
            ),
        ),
    )
    cols = combine_select_column_names(plan)
    assert cols is not None
    assert "filter_col" in cols
