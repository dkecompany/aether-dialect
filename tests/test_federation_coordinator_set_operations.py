"""Tests for cross-source semi-join and anti-join execution at the coordinator."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationCapExceededError
from aetherdialect._contracts_core import FederatedPlan, JoinSpec, RuntimeCteStep, RuntimeIntent, SelectCol, SourceStep
from aetherdialect._federation import (
    _apply_coordinator_probe_joins,
    execute_federation_coordinator,
    render_federation_glue,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._sql_gen import anti_join_presence_column


def _cross_source_probe_plan(fed, *, emission: str, probe_name: str) -> FederatedPlan:
    probe_cte = RuntimeCteStep(
        cte_name=probe_name,
        emission=emission,
        tables=[fed.right_table],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.right_table}.id"))],
        output_columns=["id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    return FederatedPlan(
        steps=(
            SourceStep(
                source_id=fed.left_source,
                sub_intent=RuntimeIntent(
                    tables=[fed.left_table, probe_name],
                    grain="many",
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.left_table}.id"))],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                    cte_steps=[probe_cte],
                ),
            ),
            SourceStep(
                source_id=fed.right_source,
                sub_intent=RuntimeIntent(
                    tables=[fed.right_table],
                    grain="many",
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.right_table}.id"))],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
        combine=(
            JoinSpec(
                left_source=fed.left_source,
                right_source=fed.right_source,
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="left",
            ),
        ),
        lifted_probe_ctes=(probe_cte,),
        grain="many",
        scope_sources=frozenset({fed.left_source, fed.right_source}),
    )


@pytest.mark.fast
def test_cross_source_semi_join_executes_at_coordinator(two_member_federation) -> None:
    fed = two_member_federation
    probe_name = "matching_right"
    plan = _cross_source_probe_plan(fed, emission="semi_join", probe_name=probe_name)
    step_ids = {fed.left_source: "src_a", fed.right_source: "src_b"}
    glue = render_federation_glue(plan, step_ids, schema=fed.composite)
    assert "INNER JOIN" in glue.upper()

    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.register("src_a", pd.DataFrame({"id": [1, 2, 3]}))
    conn.register("src_b", pd.DataFrame({"id": [2, 3]}))
    source_by_table = {fed.left_table: fed.left_source, fed.right_table: fed.right_source}
    lifted_sql = _apply_coordinator_probe_joins(
        "SELECT id FROM src_a",
        plan.lifted_probe_ctes,
        step_ids,
        source_by_table,
    )
    rows = conn.execute(lifted_sql).fetchall()
    assert sorted(row[0] for row in rows) == [2, 3]


@pytest.mark.fast
def test_cross_source_anti_join_executes_at_coordinator(two_member_federation) -> None:
    fed = two_member_federation
    probe_name = "absent_right"
    plan = _cross_source_probe_plan(fed, emission="anti_join", probe_name=probe_name)
    step_ids = {fed.left_source: "src_a", fed.right_source: "src_b"}
    glue = render_federation_glue(plan, step_ids, schema=fed.composite)
    marker = anti_join_presence_column(probe_name)
    assert "IS NULL" in glue.upper()
    assert marker in glue

    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.register("src_a", pd.DataFrame({"id": [1, 2, 3]}))
    conn.register("src_b", pd.DataFrame({"id": [2]}))
    source_by_table = {fed.left_table: fed.left_source, fed.right_table: fed.right_source}
    lifted_sql = _apply_coordinator_probe_joins(
        "SELECT id FROM src_a",
        plan.lifted_probe_ctes,
        step_ids,
        source_by_table,
    )
    rows = conn.execute(lifted_sql).fetchall()
    assert sorted(row[0] for row in rows) == [1, 3]


@pytest.mark.fast
def test_coordinator_probe_semijoin_key_cap_exceeded_raises(two_member_federation) -> None:
    fed = two_member_federation
    probe_name = "matching_right"
    plan = _cross_source_probe_plan(fed, emission="semi_join", probe_name=probe_name)
    frames = {
        fed.left_source: pd.DataFrame({"id": [1, 2, 3]}),
        fed.right_source: pd.DataFrame({"id": list(range(5))}),
    }
    with pytest.raises(FederationCapExceededError, match="semijoin key cap exceeded") as excinfo:
        execute_federation_coordinator(
            frames,
            plan,
            row_cap=100,
            semijoin_key_cap=2,
            schema=fed.composite,
        )
    assert excinfo.value.limit_key == "semijoin_key_cap"
    assert excinfo.value.source_id == fed.right_source
