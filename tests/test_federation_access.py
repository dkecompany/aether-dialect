"""Runtime access denial and federation execution-scope composition."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    FederationContext,
    FederationDeclarationError,
    SchemaAccessError,
)
from aetherdialect._contracts_core import AccessError
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import (
    compose_composite_graph,
    validate_federation_scope_against_member_visibility,
)
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import recompute_join_paths_multi, scope_descriptor_for


def _table(name: str, *, source_id: str = "") -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={
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
    tables: dict[str, TableMetadata] | None = None,
    scope: FederationContext | None = None,
) -> SchemaGraph:
    table_map = tables or {table: _table(table, source_id=source_id)}
    graph = SchemaGraph(
        tables=table_map,
        join_paths_multi=recompute_join_paths_multi(table_map),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
        profiling_hash=f"profile_{source_id}",
    )
    if scope is not None:
        graph.scope_descriptor = scope_descriptor_for(scope)
    return graph


_MANIFEST = {
    "federation_id": "fed_scope",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"customers": "a", "rental": "b"},
    "cross_source_joins": [],
}

_SINGLE_SOURCE_MANIFEST = {
    "federation_id": "fed_scope",
    "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
    "table_namespace": {"customers": "a", "rental": "a"},
    "cross_source_joins": [],
}


def _dual_table_graph(source_id: str, *, scope: FederationContext | None = None) -> SchemaGraph:
    tables = {
        "customers": _table("customers", source_id=source_id),
        "rental": _table("rental", source_id=source_id),
    }
    return _graph("customers", source_id=source_id, tables=tables, scope=scope)


@pytest.mark.fast
def test_runtime_access_error_is_schema_access_error() -> None:
    """Integrators catching SchemaAccessError also catch runtime permission denial."""
    err = AccessError("execute", "permission denied for relation secret")
    assert isinstance(err, SchemaAccessError)


@pytest.mark.fast
def test_federation_execution_allow_objects_returns_master_allow_when_set() -> None:
    master = FederationContext(allow_objects=frozenset({"customers"}))
    composite_tables = frozenset({"customers", "rental"})
    assert MainExecutionOps.federation_execution_allow_objects(master, composite_tables) == frozenset({"customers"})


@pytest.mark.fast
def test_federation_execution_allow_objects_uses_composite_when_master_unrestricted() -> None:
    master = FederationContext()
    composite_tables = frozenset({"customers", "rental"})
    assert MainExecutionOps.federation_execution_allow_objects(master, composite_tables) == composite_tables


@pytest.mark.fast
def test_compose_rejects_federation_allow_wider_than_member_allow() -> None:
    manifest = parse_federation_manifest(_SINGLE_SOURCE_MANIFEST, include_derived_roster=True)
    member_graph = _dual_table_graph("a", scope=FederationContext(allow_objects=frozenset({"customers"})))
    composite_tables = {
        "customers": member_graph.tables["customers"],
        "rental": member_graph.tables["rental"],
    }
    composite = SchemaGraph(
        tables=composite_tables,
        join_paths_multi=recompute_join_paths_multi(composite_tables),
        schema_graph_id="sg_composite",
        effective_structural_hash="eff_composite",
    )
    ctx = FederationContext(allow_objects=frozenset({"customers", "rental"}))
    with pytest.raises(FederationDeclarationError, match=r"member_allow_omitted"):
        validate_federation_scope_against_member_visibility(
            ctx,
            {"a": member_graph},
            composite,
            {("a", "customers"): "customers", ("a", "rental"): "rental"},
            FederationMappings(version="0.2.3"),
            manifest=manifest,
        )


@pytest.mark.fast
def test_compose_rejects_federation_allow_on_member_denied_table() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    member_graph = _graph("rental", source_id="b", scope=FederationContext(deny_objects=frozenset({"rental"})))
    composite_tables = {"rental": member_graph.tables["rental"]}
    composite = SchemaGraph(
        tables=composite_tables,
        join_paths_multi=recompute_join_paths_multi(composite_tables),
        schema_graph_id="sg_composite",
        effective_structural_hash="eff_composite",
    )
    ctx = FederationContext(allow_objects=frozenset({"rental"}))
    with pytest.raises(FederationDeclarationError, match=r"member_denied"):
        validate_federation_scope_against_member_visibility(
            ctx,
            {"b": member_graph},
            composite,
            {("b", "rental"): "rental"},
            FederationMappings(version="0.2.3"),
            manifest=manifest,
        )


@pytest.mark.fast
def test_compose_rejects_federation_allow_unknown_to_composite() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("rental", source_id="b"),
    }
    ctx = FederationContext(allow_objects=frozenset({"missing"}))
    with pytest.raises(ConfigError, match=r"allow_objects references unknown table"):
        compose_composite_graph(members, manifest, master_context=ctx)


@pytest.mark.fast
def test_compose_rejects_federation_deny_unknown_to_composite() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("rental", source_id="b"),
    }
    ctx = FederationContext(deny_objects=frozenset({"missing"}))
    with pytest.raises(ConfigError, match=r"deny_objects references unknown table"):
        compose_composite_graph(members, manifest, master_context=ctx)


@pytest.mark.fast
def test_validate_federation_scope_reports_not_on_member_graph() -> None:
    manifest = parse_federation_manifest(_SINGLE_SOURCE_MANIFEST, include_derived_roster=True)
    member_graph = _graph("customers", source_id="a")
    rental_table = _table("rental", source_id="a")
    composite_tables = {"rental": rental_table}
    composite = SchemaGraph(
        tables=composite_tables,
        join_paths_multi=recompute_join_paths_multi(composite_tables),
        schema_graph_id="sg_composite",
        effective_structural_hash="eff_composite",
    )
    ctx = FederationContext(allow_objects=frozenset({"rental"}))
    with pytest.raises(FederationDeclarationError, match=r"not_on_member_graph"):
        validate_federation_scope_against_member_visibility(
            ctx,
            {"a": member_graph},
            composite,
            {("a", "rental"): "rental"},
            FederationMappings(version="0.2.3"),
            manifest=manifest,
        )
