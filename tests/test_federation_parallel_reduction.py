"""Parallel member batches must not break semijoin / filter_keys reduction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPreparedStep,
    FederatedStage,
    FederationReducingEdge,
    JoinSpec,
    RuntimeIntent,
    SelectCol,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    compose_composite_graph,
    federation_member_execution_batches,
    parse_federation_manifest,
)
from aetherdialect._pipeline import _execute_federation_steps_parallel
from aetherdialect._schema_graph import recompute_join_paths_multi


def _join_schema() -> SchemaGraph:
    tables = {
        "ta": TableMetadata(
            name="ta",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
        "tb": TableMetadata(
            name="tb",
            columns={
                "a_id": ColumnMetadata(
                    name="a_id",
                    data_type="integer",
                    sensitivity="none",
                    is_unique=True,
                ),
            },
            primary_key=[],
            foreign_keys=[],
            source_id="b",
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def _parallel_manifest() -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_parallel_reduction",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "connection": "conn_a", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "connection": "conn_b", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.a_id", "kind": "inner", "logical_key": "id"},
            ],
            "coordinator": {"max_parallel_members": 2},
        },
        include_derived_roster=True,
    )


def _parallel_reduction_plan() -> tuple[FederatedPlan, SchemaGraph, object]:
    manifest = _parallel_manifest()
    composite = compose_composite_graph(
        {
            "a": SchemaGraph(
                tables={
                    "ta": TableMetadata(
                        name="ta",
                        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                        primary_key=["id"],
                        foreign_keys=[],
                        source_id="a",
                    )
                },
                join_paths_multi=recompute_join_paths_multi({}),
            ),
            "b": SchemaGraph(
                tables={
                    "tb": TableMetadata(
                        name="tb",
                        columns={
                            "a_id": ColumnMetadata(
                                name="a_id",
                                data_type="integer",
                                sensitivity="none",
                                is_unique=True,
                            ),
                        },
                        primary_key=[],
                        foreign_keys=[],
                        source_id="b",
                    )
                },
                join_paths_multi=recompute_join_paths_multi({}),
            ),
        },
        manifest,
    )
    intent_a = RuntimeIntent(
        tables=["ta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent_b = RuntimeIntent(
        tables=["tb"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("tb.a_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    step_a = SourceStep(source_id="a", sub_intent=intent_a, projected_keys=("id",))
    step_b = SourceStep(source_id="b", sub_intent=intent_b, projected_keys=("a_id",))
    reducing_edge = FederationReducingEdge(
        driving_source_id="a",
        target_source_id="b",
        driving_key="id",
        target_key="a_id",
        edge_kind="filter_keys",
    )
    plan = FederatedPlan(
        steps=(step_a, step_b),
        combine=JoinSpec(
            left_source="a",
            right_source="b",
            left_key="id",
            right_key="a_id",
            logical_key="id",
            kind="inner",
        ),
        stages=(
            FederatedStage(stage_id="member_a", kind="member", source_ids=("a",)),
            FederatedStage(
                stage_id="member_b",
                kind="member",
                source_ids=("b",),
                reducing_edges=(reducing_edge,),
            ),
        ),
        scope_sources=frozenset({"a", "b"}),
    )
    return plan, composite, manifest


@pytest.mark.fast
def test_parallel_batch_applies_reduction_from_driving_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """Target member in a parallel batch must see the driving member frame for reduction."""
    plan, composite, manifest = _parallel_reduction_plan()
    prepared_by_source = {
        "a": FederatedPreparedStep(source_id="a", sub_intent=plan.steps[0].sub_intent, sql="SELECT id FROM ta"),
        "b": FederatedPreparedStep(source_id="b", sub_intent=plan.steps[1].sub_intent, sql="SELECT a_id FROM tb"),
    }
    captured_intents: dict[str, RuntimeIntent] = {}

    def _fake_execute_guarded_sql(
        _dialect: object,
        _sql: str,
        _bind: object,
        *,
        intent: RuntimeIntent | None = None,
        **_kwargs: object,
    ) -> list[tuple[int, ...]]:
        assert intent is not None
        table = intent.tables[0] if intent.tables else ""
        captured_intents[table] = intent
        if table == "ta":
            return [(1,), (2,)]
        return [(1,)]

    monkeypatch.setattr(
        "aetherdialect._pipeline.generate_and_validate_sql",
        lambda *_a, **_k: type("Out", (), {"success": True, "sql": "SELECT a_id FROM tb"})(),
    )
    monkeypatch.setattr("aetherdialect._pipeline.execute_guarded_sql", _fake_execute_guarded_sql)
    monkeypatch.setattr("aetherdialect._pipeline.validate_federated_sub_intent", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "aetherdialect._validation_execute.validate_sql",
        lambda *a, **k: (True, None, None, None),
    )
    mock_dialect = MagicMock()
    mock_dialect.finalize_render.return_value = "SELECT 1"
    mock_dialect_streams = MagicMock(return_value=False)
    monkeypatch.setattr("aetherdialect._pipeline.dialect_streams_arrow_to_coordinator", mock_dialect_streams)

    execution_steps = (plan.steps[0], plan.steps[1])

    _execute_federation_steps_parallel(
        execution_steps,
        prepared_by_source=prepared_by_source,
        composite_schema=composite,
        dialect_map={"a": mock_dialect, "b": mock_dialect},
        dialect=mock_dialect,
        manifest=manifest,
        q_norm="q",
        join_candidates={},
        cmap={},
        store=MagicMock(),
        gate_kwargs={},
        plan=plan,
        semijoin_cap=100,
    )

    assert "tb" in captured_intents
    member_intent = captured_intents["tb"]
    leaves = list(member_intent.where.leaves() if member_intent.where else [])
    assert any(fp.op == "in" for fp in leaves), "parallel batch must apply filter_keys reduction to target member"


@pytest.mark.fast
def test_parallel_batch_splits_reducing_driver_and_target() -> None:
    """Reducer and driving member must not share a parallel execution batch."""
    plan, _composite, manifest = _parallel_reduction_plan()
    batches = federation_member_execution_batches(plan.steps, manifest, plan=plan)
    assert len(batches) == 2
    first_ids = {step.source_id for step in batches[0]}
    second_ids = {step.source_id for step in batches[1]}
    assert first_ids == {"a"}
    assert second_ids == {"b"}
