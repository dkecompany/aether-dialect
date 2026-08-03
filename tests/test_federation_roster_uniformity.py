"""Tests that federated plans share the same envelope across roster sizes."""

from __future__ import annotations

from dataclasses import fields

import pytest

from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _build_source_sub_intent,
    compose_composite_graph,
    parse_federation_manifest,
    parse_federation_mappings,
    plan_federated_intent,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _member_table(table: str, source_id: str) -> TableMetadata:
    return TableMetadata(
        name=table,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _member_graph(table: str, source_id: str) -> SchemaGraph:
    tables = {table: _member_table(table, source_id)}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash="test-profiled",
    )


def _plan_envelope(plan: FederatedPlan) -> dict[str, object]:
    skip = {"steps", "combine", "union_specs", "scope_sources", "stages"}
    return {field.name: getattr(plan, field.name) for field in fields(plan) if field.name not in skip}


_ONE_MEMBER_MANIFEST = {
    "federation_id": "fed_one",
    "sources": [{"source_id": "solo", "engine": "duckdb", "role": "owner"}],
    "table_namespace": {"events": "solo"},
    "cross_source_joins": [],
}

_FOUR_MEMBER_MANIFEST = {
    "federation_id": "fed_four",
    "sources": [
        {"source_id": "m1", "engine": "duckdb", "role": "owner"},
        {"source_id": "m2", "engine": "duckdb", "role": "owner"},
        {"source_id": "m3", "engine": "duckdb", "role": "owner"},
        {"source_id": "m4", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {
        "e1": "m1",
        "e2": "m2",
        "e3": "m3",
        "e4": "m4",
    },
    "cross_source_joins": [],
}

_FOUR_MEMBER_MAPPINGS = {
    "version": 2,
    "logical_tables": [
        {
            "logical": "events",
            "semantics": "union",
            "members": [
                {"source": "m1", "table": "e1", "columns": {"id": "id"}},
                {"source": "m2", "table": "e2", "columns": {"id": "id"}},
                {"source": "m3", "table": "e3", "columns": {"id": "id"}},
                {"source": "m4", "table": "e4", "columns": {"id": "id"}},
            ],
        },
    ],
}


@pytest.mark.fast
def test_one_member_and_four_member_plans_share_envelope_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls: list[bool] = []

    def _track_build(*args, **kwargs):
        build_calls.append(kwargs.get("multi_source", False))
        return _build_source_sub_intent(*args, **kwargs)

    monkeypatch.setattr("aetherdialect._federation._build_source_sub_intent", _track_build)

    one_manifest = parse_federation_manifest(_ONE_MEMBER_MANIFEST, include_derived_roster=True)
    one_composite = compose_composite_graph({"solo": _member_graph("events", "solo")}, one_manifest)
    one_intent = RuntimeIntent(
        tables=["events"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    one_plan = plan_federated_intent(one_intent, one_composite, one_manifest)
    assert len(one_plan.steps) == 1
    assert one_plan.combine is None
    assert build_calls == [False]

    four_manifest = parse_federation_manifest(_FOUR_MEMBER_MANIFEST, include_derived_roster=True)
    four_mappings = parse_federation_mappings(_FOUR_MEMBER_MAPPINGS)
    four_members = {
        "m1": _member_graph("e1", "m1"),
        "m2": _member_graph("e2", "m2"),
        "m3": _member_graph("e3", "m3"),
        "m4": _member_graph("e4", "m4"),
    }
    four_composite = compose_composite_graph(four_members, four_manifest, four_mappings)
    four_intent = RuntimeIntent(
        tables=["events"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    four_plan = plan_federated_intent(four_intent, four_composite, four_manifest, mappings=four_mappings)
    assert len(four_plan.steps) == 4
    assert four_plan.union_specs
    assert four_plan.combine is None
    assert build_calls == [False, True, True, True, True]

    assert _plan_envelope(one_plan) == _plan_envelope(four_plan)
    assert one_plan.scope_sources
    assert four_plan.scope_sources
    assert one_plan.stages == ()
    assert four_plan.stages
