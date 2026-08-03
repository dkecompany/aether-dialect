"""Federation member role and composite scope denial guards."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import FederationContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationConfigError,
    binding_from_member_engine,
    build_federation_manifest_from_members,
    compose_composite_graph,
    parse_federation_manifest,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _table(
    name: str,
    *,
    source_id: str = "",
    columns: dict[str, ColumnMetadata] | None = None,
) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns=columns
        or {
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "email": ColumnMetadata(name="email", data_type="text", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _graph(table: str, *, source_id: str = "", tables: dict[str, TableMetadata] | None = None) -> SchemaGraph:
    table_map = tables or {table: _table(table, source_id=source_id)}
    return SchemaGraph(
        tables=table_map,
        join_paths_multi=recompute_join_paths_multi(table_map),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
        profiling_hash=f"profile_{source_id}",
    )


_MANIFEST = {
    "federation_id": "fed_scope",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"customers": "a", "orders": "b"},
    "cross_source_joins": [],
}


def _orders_table() -> TableMetadata:
    return _table(
        "orders",
        source_id="b",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
    )


def _consumer_member(connection: str = "a") -> MagicMock:
    member = MagicMock()
    member.dialect = "duckdb"
    member._connection = connection
    member._context_name = "master"
    member._schema_role = "consumer"
    member._runtime_config = None
    return member


@pytest.mark.fast
def test_binding_from_member_engine_rejects_consumer_role() -> None:
    with pytest.raises(FederationConfigError, match="must be an owner engine"):
        binding_from_member_engine("a", _consumer_member("a"))


@pytest.mark.fast
def test_build_federation_manifest_from_members_rejects_consumer_role() -> None:
    members = {"a": _consumer_member("a")}
    declaration = parse_federation_manifest({"federation_id": "fed_scope", "cross_source_joins": []})
    member_graphs = {"a": _graph("customers", source_id="a")}
    with pytest.raises(FederationConfigError, match="must be an owner engine"):
        build_federation_manifest_from_members(
            members,
            declaration=declaration,
            member_graphs=member_graphs,
        )


@pytest.mark.fast
def test_parse_federation_manifest_derived_roster_defaults_role_owner() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_scope",
            "sources": [{"source_id": "a", "engine": "duckdb"}],
            "table_namespace": {"customers": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    assert manifest.sources[0].role == "owner"


@pytest.mark.fast
def test_parse_federation_manifest_derived_roster_rejects_consumer_role() -> None:
    with pytest.raises(FederationConfigError, match="must be an owner engine"):
        parse_federation_manifest(
            {
                "federation_id": "fed_scope",
                "sources": [{"source_id": "a", "engine": "duckdb", "role": "consumer"}],
                "table_namespace": {"customers": "a"},
                "cross_source_joins": [],
            },
            include_derived_roster=True,
        )


@pytest.mark.fast
def test_compose_applies_federation_context_deny_objects() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("orders", source_id="b", tables={"orders": _orders_table()}),
    }
    ctx = FederationContext(deny_objects=frozenset({"orders"}))
    composite = compose_composite_graph(members, manifest, master_context=ctx)
    assert "customers" in composite.tables
    assert "orders" not in composite.tables


@pytest.mark.fast
def test_compose_applies_federation_context_deny_columns() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("orders", source_id="b", tables={"orders": _orders_table()}),
    }
    ctx = FederationContext(deny_columns=frozenset({"customers.email"}))
    composite = compose_composite_graph(members, manifest, master_context=ctx)
    assert "email" in composite.deny_columns.get("customers", set())
    assert "email" not in composite.tables["customers"].columns
    assert "id" in composite.tables["customers"].columns
