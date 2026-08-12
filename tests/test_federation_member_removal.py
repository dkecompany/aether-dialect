"""Removing a federation member purges its artifact tree and composite template shards."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import TEMPLATE_STORE_SEGMENT
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    federation_source_artifacts_dir,
    persist_federation_tree,
)
from aetherdialect._federation_manifest import (
    binding_from_member_engine,
    parse_federation_manifest,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import write_federation_declaration_file
from tests.test_aether_federation_public_surface import _init_bundle, _minimal_member


def _three_member_manifest() -> dict[str, object]:
    return {
        "federation_id": "fed_remove",
        "cross_source_joins": [
            {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            {"left": "left_t.id", "right": "extra_t.id", "kind": "inner", "logical_key": "id"},
        ],
    }


def _graph(table: str, *, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


@pytest.mark.fast
def test_remove_engine_purges_member_artifact_tree(tmp_path: Path) -> None:
    manifest_payload = _three_member_manifest()
    declaration_path = write_federation_declaration_file(tmp_path, manifest_payload, {"version": "0.2.3"})
    manifest = parse_federation_manifest(manifest_payload)
    members = {
        "a": _graph("left_t", source_id="a"),
        "b": _graph("right_t", source_id="b"),
        "c": _graph("extra_t", source_id="c"),
    }
    composite = compose_composite_graph(members, manifest)
    fed_dir = tmp_path / "fed"
    persist_federation_tree(
        str(fed_dir),
        manifest=manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=composite,
        member_graphs=members,
    )
    member_c = _minimal_member(connection="c")
    member_c._schema_graph = members["c"]
    binding_c = binding_from_member_engine(member_c)
    member_tree = Path(federation_source_artifacts_dir(str(tmp_path), binding_c, federation_id=manifest.federation_id))
    member_tree.mkdir(parents=True, exist_ok=True)
    (member_tree / "schema_graph.json.gz").write_text("{}", encoding="utf-8")
    composite_store = fed_dir / TEMPLATE_STORE_SEGMENT
    composite_store.mkdir(parents=True, exist_ok=True)
    (composite_store / "stale_shard.json").write_text("{}", encoding="utf-8")

    bundle = _init_bundle(manifest, composite)
    bundle.federation_storage_dir = str(fed_dir)
    member_a = _minimal_member(connection="a")
    member_b = _minimal_member(connection="b")
    for key, member in (("a", member_a), ("b", member_b), ("c", member_c)):
        member._schema_graph = members[key]
        member._artifacts_dir = federation_source_artifacts_dir(
            str(tmp_path),
            binding_from_member_engine(member),
            federation_id=manifest.federation_id,
        )

    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_remove",
            members=(member_a, member_b, member_c),
            declaration=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )
    fed._members = {"a": member_a, "b": member_b, "c": member_c}
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed.remove_engine("c")
    assert "c" not in fed._members
    assert not member_tree.exists()
    assert not composite_store.exists()


@pytest.mark.fast
def test_remove_engine_prunes_aetherspace_snapshots_for_removed_member(tmp_path: Path) -> None:
    from aetherdialect._constants import AETHERSPACE_ARTIFACT_VERSION
    from aetherdialect._main_execution import MainExecutionOps

    manifest_payload = _three_member_manifest()
    declaration_path = write_federation_declaration_file(tmp_path, manifest_payload, {"version": "0.2.3"})
    manifest = replace(
        parse_federation_manifest(manifest_payload),
        table_namespace={"left_t": "a", "right_t": "b", "extra_t": "c"},
    )
    members = {
        "a": _graph("left_t", source_id="a"),
        "b": _graph("right_t", source_id="b"),
        "c": _graph("extra_t", source_id="c"),
    }
    composite = compose_composite_graph(members, manifest)
    fed_dir = tmp_path / "fed"
    persist_federation_tree(
        str(fed_dir),
        manifest=manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=composite,
        member_graphs=members,
    )
    MainExecutionOps.save_aetherspace_snapshot(
        str(fed_dir),
        "scoped",
        {
            "version": AETHERSPACE_ARTIFACT_VERSION,
            "tables": ["left_t", "extra_t"],
            "columns": ["left_t.id", "extra_t.id"],
            "deny_objects": [],
            "deny_columns": [],
            "table_descriptions": {},
            "column_meta": {},
            "notes": None,
            "notes_hash": "",
        },
    )
    bundle = _init_bundle(manifest, composite)
    bundle.federation_storage_dir = str(fed_dir)
    bundle.federation_manifest = manifest
    member_a = _minimal_member(connection="a")
    member_b = _minimal_member(connection="b")
    member_c = _minimal_member(connection="c")
    for key, member in (("a", member_a), ("b", member_b), ("c", member_c)):
        member._schema_graph = members[key]
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_remove",
            members=(member_a, member_b, member_c),
            declaration=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )
    fed._members = {"a": member_a, "b": member_b, "c": member_c}
    fed._federation_manifest = manifest
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed.remove_engine("c")
    snap_path = fed_dir / "aetherspaces" / "scoped.json"
    assert snap_path.is_file()
    payload = json.loads(snap_path.read_text(encoding="utf-8"))
    assert "extra_t" not in payload.get("tables", [])
    assert "extra_t.id" not in payload.get("columns", [])
