"""Federation member add/remove reconciles authored declaration and session safety."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_base import FederationConfigError, FederationMappings
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    compose_composite_graph,
    parse_federation_manifest,
    prune_federation_aliases,
    reconcile_authored_declaration_for_members,
    reconcile_federation_member_graphs,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import write_federation_declaration_file
from tests.test_aether_federation_public_surface import _init_bundle, _minimal_member


def _graph(table: str, *, source_id: str, description: str = "") -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
            description=description,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


def _three_member_manifest() -> dict[str, object]:
    return {
        "federation_id": "fed_members",
        "aliases": {
            "alias_c": {"source": "c", "table": "t_c"},
        },
        "cross_source_joins": [
            {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
            {"left": "t_a.id", "right": "t_c.id", "kind": "inner", "logical_key": "id"},
        ],
    }


@pytest.mark.fast
def test_reconcile_authored_declaration_prunes_aliases_for_removed_member() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed",
            "aliases": {
                "alias_c": {"source": "c", "table": "t_c"},
            },
            "cross_source_joins": [
                {"left": "t_a.id", "right": "t_c.id", "kind": "inner", "logical_key": "id"},
            ],
        },
    )
    mappings = FederationMappings(version=1)
    pruned_manifest, pruned_mappings = reconcile_authored_declaration_for_members(
        manifest,
        mappings,
        active_source_ids={"a", "b"},
    )
    assert pruned_manifest.aliases == ()
    assert len(pruned_manifest.cross_source_joins) == 1
    assert pruned_mappings.logical_tables == ()


@pytest.mark.fast
def test_prune_federation_aliases_drops_removed_sources() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed",
            "aliases": {"alias_c": {"source": "c", "table": "t_c"}},
            "cross_source_joins": [],
        },
    )
    pruned = prune_federation_aliases(manifest, active_source_ids={"a", "b"})
    assert pruned.aliases == ()


@pytest.mark.fast
def test_remove_engine_succeeds_when_declaration_names_removed_alias(tmp_path: Path) -> None:
    declaration_path = write_federation_declaration_file(tmp_path, _three_member_manifest(), {"version": 1})
    members = {
        "a": _graph("t_a", source_id="a"),
        "b": _graph("t_b", source_id="b"),
        "c": _graph("t_c", source_id="c"),
    }
    manifest = parse_federation_manifest(_three_member_manifest())
    composite = _graph("t_a", source_id="a")
    bundle = _init_bundle(manifest, composite)
    init_calls: list[dict[str, object]] = []

    def _capture_init(*args: object, **kwargs: object) -> object:
        init_calls.append(dict(kwargs))
        return bundle

    member_a = _minimal_member(connection="a")
    member_a._schema_graph = members["a"]
    member_b = _minimal_member(connection="b")
    member_b._schema_graph = members["b"]
    member_c = _minimal_member(connection="c")
    member_c._schema_graph = members["c"]
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", side_effect=_capture_init):
        fed = AetherFederation(
            "fed_members",
            members={"a": member_a, "b": member_b, "c": member_c},
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )
    fed._members = {"a": member_a, "b": member_b, "c": member_c}
    init_calls.clear()
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", side_effect=_capture_init):
        fed.remove_engine("c")
    assert init_calls
    declaration = init_calls[-1].get("declaration")
    assert isinstance(declaration, tuple)
    pruned_manifest = declaration[0]
    assert all(alias.source != "c" for alias in pruned_manifest.aliases)
    assert "c" not in fed._members


@pytest.mark.fast
def test_reconcile_federation_member_graphs_prefers_live_engine_graph() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"t_a": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    live = _graph("t_a", source_id="a", description="live override")
    disk = _graph("t_a", source_id="a", description="stale disk")
    merged = reconcile_federation_member_graphs({"a": live}, {"a": disk}, manifest)
    assert merged["a"].tables["t_a"].description == "live override"


@pytest.mark.fast
def test_add_engine_refuses_while_writer_lock_held() -> None:
    from tests.test_aether_federation_public_surface import _fed

    fed = _fed()
    fed._pipeline_writer_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="session turn"):
            fed.add_engine("c", _minimal_member(connection="c"))
    finally:
        fed._pipeline_writer_lock.release()


@pytest.mark.fast
def test_remove_engine_refuses_while_writer_lock_held() -> None:
    from tests.test_aether_federation_public_surface import _fed

    fed = _fed()
    fed._pipeline_writer_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="session turn"):
            fed.remove_engine("a")
    finally:
        fed._pipeline_writer_lock.release()
