"""Reducing-edge edge_kind must reach the correct rendering injector."""

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
    inject_semijoin_where,
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


def _join_manifest() -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_edge_kind_l25",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
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


def _execution_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reducing_edge: FederationReducingEdge,
) -> tuple[list[bool], list[bool]]:
    manifest = _join_manifest()
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
    member_stage = FederatedStage(
        stage_id="member_b",
        kind="member",
        source_ids=("b",),
        reducing_edges=(reducing_edge,),
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
    semijoin_called: list[bool] = []
    filter_keys_called: list[bool] = []

    def _track_semijoin(
        sub_intent: RuntimeIntent,
        key_column: str,
        keys: object,
        *,
        value_type: str = "string",
    ) -> RuntimeIntent:
        semijoin_called.append(True)
        return inject_semijoin_where(sub_intent, key_column, keys, value_type=value_type)

    def _track_filter_keys(
        sub_intent: RuntimeIntent,
        key_column: str,
        keys: object,
        *,
        value_type: str = "string",
    ) -> RuntimeIntent:
        from aetherdialect._federation import inject_filter_keys_where

        filter_keys_called.append(True)
        return inject_filter_keys_where(sub_intent, key_column, keys, value_type=value_type)

    monkeypatch.setattr("aetherdialect._pipeline.inject_semijoin_where", _track_semijoin)
    monkeypatch.setattr(
        "aetherdialect._pipeline.inject_filter_keys_where",
        _track_filter_keys,
        raising=False,
    )
    monkeypatch.setattr(
        "aetherdialect._pipeline.generate_and_validate_sql",
        lambda *_a, **_k: type("Out", (), {"success": True, "sql": "SELECT a_id FROM tb"})(),
    )

    def _fake_execute_guarded_sql(
        _dialect: object,
        _sql: str,
        _bind: object,
        *,
        intent: RuntimeIntent | None = None,
        **_kwargs: object,
    ) -> list[tuple[int, ...]]:
        return [(1,), (2,)]

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
        executed={"a": pd.DataFrame({"id": [1, 2]})},
        plan=plan,
        semijoin_cap=100,
        q_norm="q",
        join_candidates={},
        cmap={},
        store={},
        gate_kwargs={},
    )
    assert frame is not None
    return semijoin_called, filter_keys_called


@pytest.mark.fast
def test_filter_keys_edge_kind_renders_via_filter_keys_injector(monkeypatch: pytest.MonkeyPatch) -> None:
    """filter_keys reducing edges must call inject_filter_keys_where, not inject_semijoin_where."""
    semijoin_called, filter_keys_called = _execution_harness(
        monkeypatch,
        reducing_edge=FederationReducingEdge(
            driving_source_id="a",
            target_source_id="b",
            driving_key="id",
            target_key="a_id",
            edge_kind="filter_keys",
        ),
    )
    assert filter_keys_called, "filter_keys edge must render through inject_filter_keys_where"
    assert not semijoin_called, "filter_keys edge must not render through inject_semijoin_where"


@pytest.mark.fast
def test_semijoin_edge_kind_renders_via_semijoin_injector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semijoin reducing edges must call inject_semijoin_where, not inject_filter_keys_where."""
    semijoin_called, filter_keys_called = _execution_harness(
        monkeypatch,
        reducing_edge=FederationReducingEdge(
            driving_source_id="a",
            target_source_id="b",
            driving_key="id",
            target_key="a_id",
            edge_kind="semijoin",
        ),
    )
    assert semijoin_called, "semijoin edge must render through inject_semijoin_where"
    assert not filter_keys_called, "semijoin edge must not render through inject_filter_keys_where"


@pytest.mark.fast
def test_collect_member_reducing_edges_assigns_distinct_kinds() -> None:
    """Structural joins yield semijoin; join-covered cross-source filters yield filter_keys."""
    manifest = _join_manifest()
    schema = _join_schema()
    sources = {"a", "b"}
    source_by_table = dict(manifest.table_namespace)

    join_only = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    join_reducing = _collect_member_reducing_edges(
        manifest,
        FederationMappings(version=2),
        sources,
        join_only,
        source_by_table,
        schema=schema,
    )
    b_join_edges = join_reducing.get("b", ())
    assert any(edge.edge_kind == "semijoin" for edge in b_join_edges)
    assert not any(edge.edge_kind == "filter_keys" for edge in b_join_edges)

    filter_intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list([_cross_source_join_where()]),
    )
    filter_reducing = _collect_member_reducing_edges(
        manifest,
        FederationMappings(version=2),
        sources,
        filter_intent,
        source_by_table,
        schema=schema,
    )
    b_filter_edges = filter_reducing.get("b", ())
    assert any(edge.edge_kind == "filter_keys" for edge in b_filter_edges)
