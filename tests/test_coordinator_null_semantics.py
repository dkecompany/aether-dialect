"""Coordinator residual GROUP BY and DISTINCT null semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_base import NormalizedExpr, OrderByCol
from aetherdialect._contracts_core import (
    FederatedPlan,
    JoinSpec,
    ResidualSpec,
    RuntimeIntent,
    SelectCol,
    SourceStep,
)
from aetherdialect._federation_manifest import federation_residual_column_headers
from aetherdialect._federation_plan import render_federation_glue

_LEFT_TABLE = "left_t"
_RIGHT_TABLE = "right_t"
_LEFT_SOURCE = "a"
_RIGHT_SOURCE = "b"
_STATUS_COL = "status"


def _nullable_status_frames() -> dict[str, pd.DataFrame]:
    return {
        _LEFT_SOURCE: pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                _STATUS_COL: [None, None, "active", "inactive", None],
            },
        ),
        _RIGHT_SOURCE: pd.DataFrame({"id": [1, 2, 3, 4, 5]}),
    }


def _join_steps() -> tuple[SourceStep, SourceStep]:
    return (
        SourceStep(
            source_id=_LEFT_SOURCE,
            sub_intent=RuntimeIntent(
                tables=[_LEFT_TABLE],
                grain="many",
                select_cols=[
                    SelectCol(expr=NormalizedExpr.from_column(f"{_LEFT_TABLE}.id")),
                    SelectCol(expr=NormalizedExpr.from_column(f"{_LEFT_TABLE}.{_STATUS_COL}")),
                ],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            ),
            projected_keys=("id", _STATUS_COL),
        ),
        SourceStep(
            source_id=_RIGHT_SOURCE,
            sub_intent=RuntimeIntent(
                tables=[_RIGHT_TABLE],
                grain="many",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{_RIGHT_TABLE}.id"))],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            ),
            projected_keys=("id",),
        ),
    )


def _join_plan(*, residual: ResidualSpec, grain: str) -> FederatedPlan:
    return FederatedPlan(
        steps=_join_steps(),
        combine=(
            JoinSpec(
                left_source=_LEFT_SOURCE,
                right_source=_RIGHT_SOURCE,
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="inner",
            ),
        ),
        residual=residual,
        grain=grain,
        scope_sources=frozenset({_LEFT_SOURCE, _RIGHT_SOURCE}),
    )


def _execute_coordinator_glue(frames: dict[str, pd.DataFrame], plan: FederatedPlan) -> pd.DataFrame:
    """Run coordinator DuckDB glue for *plan* over in-memory member frames."""
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect(":memory:")
    try:
        step_ids = {source_id: f"src_{source_id}" for source_id in frames}
        for source_id, frame in frames.items():
            conn.register(step_ids[source_id], frame)
        glue = render_federation_glue(plan, step_ids)
        return conn.execute(glue).fetchdf()
    finally:
        conn.close()


@pytest.mark.fast
def test_nulls_group_together_and_distinct_once() -> None:
    """NULLs must collapse to one GROUP BY bucket and one DISTINCT value at the coordinator."""
    frames = _nullable_status_frames()

    grouped_plan = _join_plan(
        grain="grouped",
        residual=ResidualSpec(
            select_cols=(
                SelectCol(expr=NormalizedExpr.from_column(_STATUS_COL)),
                SelectCol(expr=NormalizedExpr.from_agg("count", "*")),
            ),
            group_by_cols=(NormalizedExpr.from_column(_STATUS_COL),),
            order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column(_STATUS_COL), direction="ASC"),),
        ),
    )
    grouped = _execute_coordinator_glue(frames, grouped_plan)
    status_col, count_col = federation_residual_column_headers(grouped_plan)
    assert len(grouped) == 3
    null_group = grouped[grouped[status_col].isna()]
    assert len(null_group) == 1
    assert int(null_group.iloc[0][count_col]) == 3

    distinct_plan = _join_plan(
        grain="many",
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column(_STATUS_COL)),),
            distinct_select_index=0,
        ),
    )
    distinct = _execute_coordinator_glue(frames, distinct_plan)
    distinct_status_col = federation_residual_column_headers(distinct_plan)[0]
    assert len(distinct) == 3
    assert distinct[distinct_status_col].isna().sum() == 1
