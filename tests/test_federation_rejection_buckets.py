"""Federation malformed-member and join fan-out rejection buckets."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_base import (
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    NormalizedExpr,
)
from aetherdialect._contracts_core import (
    FederatedPlan,
    JoinSpec,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    execute_federation_coordinator,
    validate_coordinator_join_fan_out,
    validate_member_frame_projection,
)
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, *, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


_MANIFEST = {
    "federation_id": "fed_reject",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_malformed_member_projection_raises_rejection_bucket_with_member_and_phase() -> None:
    step = SourceStep(
        source_id="b",
        sub_intent=RuntimeIntent(
            tables=["right_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        projected_keys=("right_t.id",),
    )
    frame = pd.DataFrame({"id": [1], "extra": [2]})

    with pytest.raises(FederationMalformedMemberAnswerError) as exc_info:
        validate_member_frame_projection(step, frame)

    exc = exc_info.value
    assert exc.rejection_bucket == RejectionBucket.MALFORMED_MEMBER_ANSWER.value
    assert exc.source_id == "b"
    assert exc.phase == "member"


@pytest.mark.fast
def test_coordinator_join_fan_out_raises_rejection_bucket_with_member_and_phase() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    frames = {
        "a": pd.DataFrame({"id": [1, 1]}),
        "b": pd.DataFrame({"id": [1, 1]}),
    }

    with pytest.raises(FederationJoinFanOutError) as exc_info:
        execute_federation_coordinator(frames, plan, row_cap=100)

    exc = exc_info.value
    assert exc.rejection_bucket == RejectionBucket.JOIN_FAN_OUT.value
    assert exc.source_id == "b"
    assert exc.phase == "coordinator"


@pytest.mark.fast
def test_coordinator_join_fan_out_bound_refuses_before_returning_rows() -> None:
    plan = FederatedPlan(
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
    )
    with pytest.raises(FederationJoinFanOutError):
        validate_coordinator_join_fan_out(plan, {"a": 2, "b": 2}, 4)
