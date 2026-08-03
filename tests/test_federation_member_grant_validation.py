"""Federation declaration grant validation at composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import FEDERATION_MAPPINGS_VERSION
from aetherdialect._contracts_base import LogicalColumnMapping
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationDeclarationError,
    FederationMappings,
    MemberEffectiveGrants,
    compose_composite_graph,
    introspect_member_effective_grants,
    member_effective_grants_from_graph,
    member_graphs_from_engines,
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


def _graph(
    table: str,
    *,
    source_id: str = "",
    extra_tables: dict[str, TableMetadata] | None = None,
) -> SchemaGraph:
    tables = {table: _table(table, source_id=source_id)}
    if extra_tables:
        tables.update(extra_tables)
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
        profiling_hash=f"profile_{source_id}",
    )


_MANIFEST = {
    "federation_id": "fed_grants",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"customers": "a", "orders": "b"},
    "cross_source_joins": [],
}

_MANIFEST_WITH_SECRET = {
    **_MANIFEST,
    "table_namespace": {**_MANIFEST["table_namespace"], "secret_ledger": "b"},
}


@pytest.mark.fast
def test_compose_rejects_declared_table_not_in_member_effective_grants() -> None:
    manifest = parse_federation_manifest(_MANIFEST_WITH_SECRET, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph(
            "orders",
            source_id="b",
            extra_tables={"secret_ledger": _table("secret_ledger", source_id="b")},
        ),
    }
    grants = {
        "a": MemberEffectiveGrants(tables=frozenset({"customers"})),
        "b": MemberEffectiveGrants(tables=frozenset({"orders"})),
    }
    with pytest.raises(FederationDeclarationError, match=r"member 'b'.*secret_ledger"):
        compose_composite_graph(members, manifest, member_effective_grants=grants)


@pytest.mark.fast
def test_compose_rejects_declared_column_not_in_member_effective_grants() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("customers", source_id="a"), "b": _graph("orders", source_id="b")}
    mappings = FederationMappings(
        version=FEDERATION_MAPPINGS_VERSION,
        logical_columns=(
            LogicalColumnMapping(
                logical="email",
                members=("customers.email",),
                role="attribute",
                unify_in_graph=False,
            ),
        ),
    )
    grants = {
        "a": MemberEffectiveGrants(
            tables=frozenset({"customers"}),
            columns=frozenset({("customers", "id")}),
        ),
        "b": MemberEffectiveGrants(tables=frozenset({"orders"})),
    }
    with pytest.raises(FederationDeclarationError, match=r"member 'a'.*customers\.email"):
        compose_composite_graph(members, manifest, mappings, member_effective_grants=grants)


@pytest.mark.fast
def test_compose_accepts_declared_objects_within_effective_grants() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("customers", source_id="a"), "b": _graph("orders", source_id="b")}
    grants = {
        "a": MemberEffectiveGrants(
            tables=frozenset({"customers"}),
            columns=frozenset({("customers", "id"), ("customers", "email")}),
        ),
        "b": MemberEffectiveGrants(
            tables=frozenset({"orders"}),
            columns=frozenset({("orders", "id"), ("orders", "email")}),
        ),
    }
    composite = compose_composite_graph(members, manifest, member_effective_grants=grants)
    assert "customers" in composite.tables
    assert "orders" in composite.tables


@pytest.mark.fast
def test_introspect_member_effective_grants_uses_dialect_hook() -> None:
    dialect = MagicMock()
    dialect.introspect_effective_grants.return_value = MemberEffectiveGrants(tables=frozenset({"customers"}))
    engine = MagicMock()
    engine._dialect = dialect
    grants = introspect_member_effective_grants(engine)
    assert grants is not None
    assert grants.tables == frozenset({"customers"})
    dialect.introspect_effective_grants.assert_called_once_with()


@pytest.mark.fast
def test_member_graphs_from_engines_stashes_introspected_grants() -> None:
    dialect = MagicMock()
    dialect.introspect_effective_grants.return_value = {"tables": ["customers"]}
    engine = MagicMock()
    engine._schema_graph = _graph("customers", source_id="a")
    engine._dialect = dialect
    graphs = member_graphs_from_engines({"a": engine})
    stashed = getattr(graphs["a"], "_member_effective_grants", None)
    assert stashed is not None
    assert stashed.tables == frozenset({"customers"})


@pytest.mark.fast
def test_member_effective_grants_from_graph_respects_scope_descriptor() -> None:
    graph = _graph("customers", source_id="a")
    graph.deny_columns = {"customers": {"email"}}
    grants = member_effective_grants_from_graph(graph)
    assert "customers" in grants.tables
    assert ("customers", "id") in grants.columns
    assert ("customers", "email") not in grants.columns
