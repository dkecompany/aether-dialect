"""Federation profiled gate and unified description precedence."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import INTERPRET_FIELDS, SCHEMA_FIELD_DESCRIPTION
from aetherdialect._contracts_base import (
    DescriptionOwner,
    FederationMemberUnprofilableError,
    resolve_descriptions,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _member_graph_is_profiled,
    _merge_column_metadata_union_statistics,
    assert_federation_member_graph_profiled,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(*, profiling_hash: str = "", role: str | None = None) -> SchemaGraph:
    col = ColumnMetadata(name="id", data_type="integer", sensitivity="none")
    if role is not None:
        col.role = role
    table = TableMetadata(
        name="entity",
        columns={"id": col},
        primary_key=["id"],
        foreign_keys=[],
    )
    tables = {"entity": table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash=profiling_hash,
    )


@pytest.mark.fast
def test_member_graph_with_role_but_no_profiling_hash_is_unprofiled() -> None:
    graph = _graph(profiling_hash="", role="identifier")
    assert _member_graph_is_profiled(graph) is False


@pytest.mark.fast
def test_member_graph_with_profiling_hash_is_profiled() -> None:
    graph = _graph(profiling_hash="profiled-abc")
    assert _member_graph_is_profiled(graph) is True


@pytest.mark.fast
def test_assert_federation_member_graph_profiled_rejects_role_only_graph() -> None:
    graph = _graph(profiling_hash="", role="identifier")
    with pytest.raises(FederationMemberUnprofilableError, match="not profiled"):
        assert_federation_member_graph_profiled("alpha", graph)


@pytest.mark.fast
def test_merge_union_statistics_does_not_sum_distinct_count() -> None:
    left = ColumnMetadata(name="id", data_type="integer", distinct_count=10)
    right = ColumnMetadata(name="id", data_type="integer", distinct_count=20)
    merged = _merge_column_metadata_union_statistics([left, right])
    assert merged.distinct_count == 0


@pytest.mark.fast
def test_merge_union_statistics_preserves_single_member_distinct_count() -> None:
    col = ColumnMetadata(name="id", data_type="integer", distinct_count=15)
    merged = _merge_column_metadata_union_statistics([col])
    assert merged.distinct_count == 15


@pytest.mark.fast
def test_resolve_descriptions_higher_owner_wins() -> None:
    text, owner = resolve_descriptions(
        ("catalog text", DescriptionOwner.CATALOG),
        ("user text", DescriptionOwner.USER_OVERRIDE),
    )
    assert text == "user text"
    assert owner == DescriptionOwner.USER_OVERRIDE


@pytest.mark.fast
def test_resolve_descriptions_same_owner_conflict_clears() -> None:
    text, owner = resolve_descriptions(
        ("alpha", DescriptionOwner.PROFILE),
        ("beta", DescriptionOwner.PROFILE),
    )
    assert text == ""
    assert owner is None


@pytest.mark.fast
def test_resolve_descriptions_same_owner_unanimous_keeps_text() -> None:
    text, owner = resolve_descriptions(
        ("shared", DescriptionOwner.PROFILE),
        ("shared", DescriptionOwner.PROFILE),
    )
    assert text == "shared"
    assert owner == DescriptionOwner.PROFILE


@pytest.mark.fast
def test_schema_payload_overlay_respects_description_owner_ladder() -> None:
    col = ColumnMetadata(
        name="id",
        data_type="integer",
        description="graph description",
        description_owner=DescriptionOwner.USER_OVERRIDE,
    )
    table = TableMetadata(
        name="orders",
        columns={"id": col},
        primary_key=["id"],
        foreign_keys=[],
        description="table graph",
        description_owner=DescriptionOwner.USER_OVERRIDE,
    )
    graph = SchemaGraph(tables={"orders": table}, join_paths_multi={})
    overlay = {
        "table_descriptions": {"orders": "space table"},
        "column_meta": {"orders.id": {"description": "space column"}},
    }
    payload = json.loads(
        graph.schema_payload_json(INTERPRET_FIELDS, owner_master_scope=True, description_overlay=overlay)
    )
    assert payload["orders"]["description"] == "table graph"
    assert payload["orders"]["columns"]["id"]["description"] == "graph description"


@pytest.mark.fast
def test_schema_payload_overlay_wins_when_owner_outranks_graph() -> None:
    col = ColumnMetadata(
        name="id",
        data_type="integer",
        description="graph description",
        description_owner=DescriptionOwner.CATALOG,
    )
    table = TableMetadata(
        name="orders",
        columns={"id": col},
        primary_key=["id"],
        foreign_keys=[],
        description="table graph",
        description_owner=DescriptionOwner.CATALOG,
    )
    graph = SchemaGraph(tables={"orders": table}, join_paths_multi={})
    overlay = {
        "table_descriptions": {"orders": "space table"},
        "column_meta": {
            "orders.id": {
                "description": "space column",
                "description_owner": DescriptionOwner.USER_OVERRIDE.value,
            }
        },
    }
    payload = json.loads(
        graph.schema_payload_json(INTERPRET_FIELDS, owner_master_scope=True, description_overlay=overlay)
    )
    assert payload["orders"]["description"] == "space table"
    assert payload["orders"]["columns"]["id"]["description"] == "space column"


@pytest.mark.fast
def test_merge_column_metadata_union_statistics_uses_owner_precedence() -> None:
    low = ColumnMetadata(
        name="id",
        data_type="integer",
        description="catalog",
        description_owner=DescriptionOwner.CATALOG,
    )
    high = ColumnMetadata(
        name="id",
        data_type="integer",
        description="profile",
        description_owner=DescriptionOwner.PROFILE,
    )
    merged = _merge_column_metadata_union_statistics([low, high])
    assert merged.description == "profile"
    assert merged.description_owner == DescriptionOwner.PROFILE
