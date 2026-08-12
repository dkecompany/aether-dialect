"""Federation closure: probe, composition scoping, mapping drift, parse guards, merge rules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT
from aetherdialect._contracts_base import FederationConfigError, FederationDeclarationError, FederationMemberProbeError
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import _resolve_composite_table_names, compose_composite_graph
from aetherdialect._federation_execute import probe_federation_member_connections
from aetherdialect._federation_manifest import (
    derive_table_namespace,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


def _graph(
    table_name: str,
    *,
    source_id: str,
    columns: dict[str, ColumnMetadata] | None = None,
    overlap: list[str] | None = None,
) -> SchemaGraph:
    cols = columns or {
        "id": ColumnMetadata(
            name="id",
            data_type="integer",
            sensitivity="none",
            value_overlap_sample=list(overlap or []),
        ),
    }
    table = TableMetadata(
        name=table_name,
        columns=cols,
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table_name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table_name}",
        effective_structural_hash=f"eff_{source_id}_{table_name}",
        profiling_hash=f"profile_{source_id}_{table_name}",
    )


def _multi_table_graph(source_id: str, table_names: list[str]) -> SchemaGraph:
    tables: dict[str, TableMetadata] = {}
    for name in table_names:
        tables[name] = TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}",
        effective_structural_hash=f"eff_{source_id}",
        profiling_hash=f"profile_{source_id}",
    )


@pytest.mark.fast
def test_probe_raises_when_execution_engine_missing() -> None:
    engine = MagicMock(spec=["dialect", "_execution_engine", "_runtime_config", "_dialect"])
    engine.dialect = "postgresql"
    engine._execution_engine = None
    engine._dialect = None
    engine._runtime_config = None
    with pytest.raises(FederationConfigError, match="member_a"):
        probe_federation_member_connections({"member_a": engine})


@pytest.mark.fast
def test_probe_fails_on_missing_declared_column() -> None:
    graph = _graph("orders", source_id="a")
    engine = MagicMock()
    engine.dialect = "postgresql"
    sa_engine = MagicMock()
    conn = MagicMock()
    sa_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    sa_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    engine._execution_engine = sa_engine
    engine._schema_graph = graph
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_probe",
            "sources": [{"source_id": "a", "engine": "postgresql", "role": "owner"}],
            "table_namespace": {"orders": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "orders",
                    "semantics": "replica",
                    "authoritative_source": "a",
                    "members": [
                        {
                            "source": "a",
                            "table": "orders",
                            "columns": {"id": "id", "missing_col": "ghost_col"},
                        },
                    ],
                },
            ],
        },
    )
    with pytest.raises(FederationConfigError, match="ghost_col"):
        probe_federation_member_connections({"a": engine}, manifest=manifest, mappings=mappings)


@pytest.mark.fast
def test_probe_fails_when_live_object_missing_despite_stale_graph() -> None:
    graph = _graph(
        "orders",
        source_id="a",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "ghost_col": ColumnMetadata(name="ghost_col", data_type="text", sensitivity="none"),
        },
    )
    engine = MagicMock()
    engine.dialect = "postgresql"
    sa_engine = MagicMock()
    conn = MagicMock()

    def _execute(stmt: object) -> MagicMock:
        sql = str(stmt)
        if "ghost_col" in sql.lower():
            raise RuntimeError('column "ghost_col" does not exist')
        return MagicMock()

    conn.execute.side_effect = _execute
    sa_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    sa_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    engine._execution_engine = sa_engine
    engine._schema_graph = graph
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_probe_live",
            "sources": [{"source_id": "a", "engine": "postgresql", "role": "owner"}],
            "table_namespace": {"orders": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "orders",
                    "semantics": "replica",
                    "authoritative_source": "a",
                    "members": [
                        {
                            "source": "a",
                            "table": "orders",
                            "columns": {"id": "id", "ghost_col": "ghost_col"},
                        },
                    ],
                },
            ],
        },
    )
    with pytest.raises(FederationMemberProbeError):
        probe_federation_member_connections({"a": engine}, manifest=manifest, mappings=mappings)


@pytest.mark.fast
def test_probe_fails_when_live_table_missing_despite_stale_graph() -> None:
    graph = _graph("orders", source_id="a")
    engine = MagicMock()
    engine.dialect = "postgresql"
    sa_engine = MagicMock()
    conn = MagicMock()
    sa_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    sa_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    engine._execution_engine = sa_engine
    engine._schema_graph = graph
    conn.execute.side_effect = RuntimeError('relation "orders" does not exist')
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_probe_live_table",
            "sources": [{"source_id": "a", "engine": "postgresql", "role": "owner"}],
            "table_namespace": {"orders": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "orders",
                    "semantics": "replica",
                    "authoritative_source": "a",
                    "members": [{"source": "a", "table": "orders", "columns": {"id": "id"}}],
                },
            ],
        },
    )
    with pytest.raises(FederationMemberProbeError):
        probe_federation_member_connections({"a": engine}, manifest=manifest, mappings=mappings)


@pytest.mark.fast
def test_compose_excludes_tables_outside_member_allow_scope() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_scope",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "a": _multi_table_graph("a", ["ta", "extra_a"]),
        "b": _multi_table_graph("b", ["tb"]),
    }
    composite = compose_composite_graph(members, manifest)
    assert "ta" in composite.tables
    assert "tb" in composite.tables
    assert "extra_a" not in composite.tables


@pytest.mark.fast
def test_compose_refuses_on_hard_mapping_drift() -> None:
    members = {
        "a": _graph("entity_a", source_id="a", overlap=["1", "2", "3"]),
        "b": _graph("entity_b", source_id="b", overlap=["9", "8", "7"]),
    }
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_columns": [
                {
                    "logical": "entity_id",
                    "unify_in_graph": True,
                    "members": ["entity_a.id", "entity_b.id"],
                },
            ],
            "logical_tables": [
                {
                    "logical": "entity",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "entity_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "entity_b", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_drift",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "a", "entity_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    token = set_diagnostic_collector([])
    try:
        with pytest.raises(FederationConfigError, match="mapping drift"):
            compose_composite_graph(members, manifest, mappings)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert any(d.code == DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT for d in diags)


@pytest.mark.fast
def test_parse_rejects_join_key_without_unify_in_graph() -> None:
    with pytest.raises(FederationDeclarationError, match="unify_in_graph"):
        parse_federation_mappings(
            {
                "version": "0.2.3",
                "logical_columns": [
                    {
                        "logical": "shared_id",
                        "role": "join_key",
                        "unify_in_graph": False,
                        "members": ["left_t.id", "right_t.id"],
                    },
                ],
            },
        )


@pytest.mark.fast
def test_parse_rejects_unknown_logical_column_role() -> None:
    with pytest.raises(FederationDeclarationError, match="role"):
        parse_federation_mappings(
            {
                "version": "0.2.3",
                "logical_columns": [
                    {
                        "logical": "attr",
                        "role": "attribute",
                        "members": ["orders.id"],
                    },
                ],
            },
        )


@pytest.mark.fast
def test_replica_merge_raises_on_data_type_mismatch() -> None:
    members = {
        "a": _graph(
            "orders_a",
            source_id="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            },
        ),
        "b": _graph(
            "orders_b",
            source_id="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="text", sensitivity="none"),
            },
        ),
    }
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "orders",
                    "semantics": "replica",
                    "authoritative_source": "a",
                    "members": [
                        {"source": "a", "table": "orders_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "orders_b", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_replica",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"orders_a": "a", "orders_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    with pytest.raises(FederationConfigError, match="data_type"):
        compose_composite_graph(members, manifest, mappings)


@pytest.mark.fast
def test_derive_table_namespace_raises_on_duplicate_logical() -> None:
    members = {
        "a": _graph("shared", source_id="a"),
        "b": _graph("shared", source_id="b"),
    }
    with pytest.raises(FederationConfigError, match="table name collision"):
        derive_table_namespace(members)


@pytest.mark.fast
def test_resolve_composite_table_names_raises_on_excess_aliases() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_alias",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"alias_one": "a", "alias_two": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {"a": _graph("phys_only", source_id="a")}
    with pytest.raises(FederationConfigError, match="alias"):
        _resolve_composite_table_names(members, manifest)


@pytest.mark.fast
def test_resolve_composite_table_names_raises_on_excess_unaliased() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_unaliased",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"alias_one": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {"a": _multi_table_graph("a", ["phys_a", "phys_b"])}
    with pytest.raises(FederationConfigError, match="unaliased"):
        _resolve_composite_table_names(members, manifest)
