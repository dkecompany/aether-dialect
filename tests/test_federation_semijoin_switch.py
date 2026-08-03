"""Semijoin switch must gate filter_keys pushdown the same as semijoin edges."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aetherdialect._contracts_base import NormalizedExpr, WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import (
    FederatedStage,
    FederationReducingEdge,
    RuntimeIntent,
    SelectCol,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationMappings,
    _collect_member_reducing_edges,
    parse_federation_manifest,
    plan_federated_intent,
)
from aetherdialect._pipeline import _execute_federation_source_step
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


def _join_manifest(*, semijoin_enabled_b: bool = True) -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_semijoin_switch",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {
                    "source_id": "b",
                    "engine": "duckdb",
                    "role": "owner",
                    "limits": {"semijoin_enabled": semijoin_enabled_b},
                },
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.a_id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )


def _cross_source_join_where() -> WhereParam:
    return WhereParam(
        left_expr=NormalizedExpr.from_column("ta.id"),
        op="=",
        right_expr=NormalizedExpr.from_column("tb.a_id"),
        value_type="integer",
    )


@pytest.mark.fast
def test_filter_keys_edges_respect_semijoin_disabled_switch() -> None:
    manifest = _join_manifest(semijoin_enabled_b=False)
    schema = _join_schema()
    sources = {"a", "b"}
    source_by_table = dict(manifest.table_namespace)
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list([_cross_source_join_where()]),
    )
    reducing = _collect_member_reducing_edges(
        manifest,
        FederationMappings(version=2),
        sources,
        intent,
        source_by_table,
        schema=schema,
    )
    b_edges = reducing.get("b", ())
    assert not any(edge.edge_kind == "filter_keys" for edge in b_edges)
    assert not any(edge.edge_kind == "semijoin" for edge in b_edges)


@pytest.mark.fast
def test_filter_keys_pushdown_skipped_when_semijoin_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _join_manifest(semijoin_enabled_b=False)
    schema = _join_schema()
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("tb.a_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    base_plan = plan_federated_intent(intent, schema, manifest)
    filter_edge = FederationReducingEdge(
        driving_source_id="a",
        target_source_id="b",
        driving_key="id",
        target_key="a_id",
        edge_kind="filter_keys",
    )
    member_stage = FederatedStage(
        stage_id="member_b",
        kind="member",
        source_ids=("b",),
        reducing_edges=(filter_edge,),
    )
    plan = replace(base_plan, stages=(member_stage,))
    target_step = SourceStep(source_id="b", sub_intent=replace(intent, tables=["tb"]))
    prep = type(
        "Prep",
        (),
        {
            "sub_intent": target_step.sub_intent,
            "sql": "SELECT a_id FROM tb",
            "structural_defaults": None,
        },
    )()
    executed = {"a": pd.DataFrame({"id": [1, 2]})}
    captured_intents: list[RuntimeIntent] = []

    def _fake_execute_guarded_sql(
        _dialect: object,
        _sql: str,
        _bind: object,
        *,
        intent: RuntimeIntent | None = None,
        **_kwargs: object,
    ) -> list[tuple[int, ...]]:
        assert intent is not None
        captured_intents.append(intent)
        return [(1,), (2,)]

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
    mock_dialect.finalize_render.return_value = "SELECT a_id FROM tb"

    frame = _execute_federation_source_step(
        target_step,
        prepared_by_source={"b": prep},
        composite_schema=schema,
        dialect_map={},
        dialect=mock_dialect,
        manifest=manifest,
        executed=executed,
        plan=plan,
        semijoin_cap=100,
        q_norm="q",
        join_candidates={},
        cmap={},
        store={},
        gate_kwargs={},
    )
    assert frame is not None
    assert captured_intents
    member_intent = captured_intents[0]
    for fp in (member_intent.where.leaves() if member_intent.where else []) or []:
        assert fp.op != "in"
