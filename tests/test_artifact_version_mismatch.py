"""Version-mismatch handling for persisted artifact loaders."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aetherdialect._constants import (
    AETHERSPACE_ARTIFACT_VERSION,
    FEDERATION_ARTIFACT_FORMAT_VERSION,
    SCHEMA_CONTEXT_CACHE_VERSION,
    STRUCTURE_DOCUMENT_VERSION,
)
from aetherdialect._contracts_base import ConfigError, EngineContext, FederationConfigError
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    load_federation_composite_graph,
    mappings_replay_matches,
    persist_federation_tree,
)
from aetherdialect._federation_manifest import (
    federation_artifact_paths,
    parse_federation_manifest,
)
from aetherdialect._main_execution import (
    MainExecutionOps,
)
from aetherdialect._schema_finalize import load_structure_sidecar, save_structure_sidecar
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_reflect import structure_sidecar_path


def _member_graph(table: str, source_id: str = "") -> SchemaGraph:
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
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


_FED_MANIFEST = {
    "federation_id": "fed_ver",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_aetherspace_snapshot_version_mismatch_distinct_from_missing(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    assert MainExecutionOps.load_aetherspace_snapshot(engine_dir, "films") is None

    current = {
        "version": AETHERSPACE_ARTIFACT_VERSION,
        "tables": ["film"],
        "columns": ["film.film_id"],
    }
    MainExecutionOps.save_aetherspace_snapshot(engine_dir, "films", current)
    loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, "films")
    assert loaded is not None
    assert loaded["tables"] == ["film"]

    wrong = dict(current)
    wrong["version"] = "9.9.9"
    spaces = Path(engine_dir) / "aetherspaces"
    spaces.mkdir(parents=True, exist_ok=True)
    (spaces / "stale.json").write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"version .*" + str(AETHERSPACE_ARTIFACT_VERSION)) as exc_info:
        MainExecutionOps.load_aetherspace_snapshot(engine_dir, "stale")
    msg = str(exc_info.value)
    assert "9.9.9" in msg
    assert str(AETHERSPACE_ARTIFACT_VERSION) in msg
    assert "Delete" in msg


@pytest.mark.fast
def test_schema_context_cache_version_mismatch_distinct_from_missing(tmp_path: Path) -> None:
    adir = str(tmp_path)
    assert MainExecutionOps.load_schema_context_cache(adir) is None

    ctx = EngineContext(allow_objects=frozenset({"public.t"}))
    MainExecutionOps.write_schema_context_cache(adir, ctx)
    loaded = MainExecutionOps.load_schema_context_cache(adir)
    assert loaded is not None
    assert "public.t" in loaded.allow_objects

    cache_path = os.path.join(adir, "schema_context.json")
    with open(cache_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["version"] = "9.9.9"
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    with pytest.raises(ConfigError, match=r"version .*9\.9\.9") as exc_info:
        MainExecutionOps.load_schema_context_cache(adir)
    msg = str(exc_info.value)
    assert str(SCHEMA_CONTEXT_CACHE_VERSION) in msg
    assert "Delete" in msg


@pytest.mark.fast
def test_schema_context_cache_rejects_include_both(tmp_path: Path) -> None:
    from aetherdialect._main_execution import MainExecutionOps

    adir = str(tmp_path)
    cache_path = Path(adir) / "schema_context.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": "0.2.3",
                "include": "both",
                "allow_objects": [],
                "deny_objects": [],
                "deny_columns": [],
                "allow_columns": [],
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="include must be tables or views"):
        MainExecutionOps.load_schema_context_cache(adir)


@pytest.mark.fast
def test_schema_context_cache_v3_is_mismatch(tmp_path: Path) -> None:
    adir = str(tmp_path)
    cache_path = os.path.join(adir, "schema_context.json")
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": 3,
                "include": "tables",
                "allow_objects": ["public.t"],
                "deny_objects": [],
                "deny_columns": [],
                "allow_columns": [],
            },
            fh,
        )
    with pytest.raises(ConfigError, match=r"version .*3") as exc_info:
        MainExecutionOps.load_schema_context_cache(adir)
    assert str(SCHEMA_CONTEXT_CACHE_VERSION) in str(exc_info.value)


@pytest.mark.fast
def test_overrides_sidecar_version_mismatch_distinct_from_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "schema.json.gz"
    cache_path.write_bytes(b"")
    assert load_structure_sidecar(cache_path) is None

    doc = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
    }
    save_structure_sidecar(cache_path, doc, source_schema_hash="abc", metadata_hash="0" * 64)
    loaded = load_structure_sidecar(cache_path)
    assert loaded is not None
    assert loaded["version"] == STRUCTURE_DOCUMENT_VERSION

    sidecar = structure_sidecar_path(cache_path)
    with sidecar.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["version"] = "9.9.9"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"version .*") as exc_info:
        load_structure_sidecar(cache_path)
    msg = str(exc_info.value)
    assert "9.9.9" in msg
    assert str(STRUCTURE_DOCUMENT_VERSION) in msg
    assert "Delete" in msg


@pytest.mark.fast
def test_mappings_replay_matches_version_mismatch_distinct_from_missing() -> None:
    manifest = parse_federation_manifest(_FED_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.3")
    members = {"a": _member_graph("left_t", "a"), "b": _member_graph("right_t", "b")}
    with tempfile.TemporaryDirectory() as tmp:
        assert mappings_replay_matches(tmp, members, manifest, mappings) is False

        composite = compose_composite_graph(members, manifest, mappings)
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        assert mappings_replay_matches(tmp, members, manifest, mappings) is True

        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        wrong = "9.9.9"
        stored["artifact_format_version"] = wrong
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with pytest.raises(FederationConfigError, match=r"artifact_format_version") as exc_info:
            mappings_replay_matches(tmp, members, manifest, mappings)
        msg = str(exc_info.value)
        assert str(wrong) in msg
        assert str(FEDERATION_ARTIFACT_FORMAT_VERSION) in msg
        assert "Delete" in msg


@pytest.mark.fast
def test_load_federation_composite_graph_version_mismatch_distinct_from_missing() -> None:
    manifest = parse_federation_manifest(_FED_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.3")
    members = {"a": _member_graph("left_t", "a"), "b": _member_graph("right_t", "b")}
    with tempfile.TemporaryDirectory() as tmp:
        assert load_federation_composite_graph(tmp) is None

        composite = compose_composite_graph(members, manifest, mappings)
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        loaded = load_federation_composite_graph(tmp)
        assert loaded is not None
        assert "left_t" in loaded.tables

        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        wrong = "9.9.9"
        stored["artifact_format_version"] = wrong
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with pytest.raises(FederationConfigError, match=r"artifact_format_version") as exc_info:
            load_federation_composite_graph(tmp)
        msg = str(exc_info.value)
        assert str(wrong) in msg
        assert str(FEDERATION_ARTIFACT_FORMAT_VERSION) in msg
        assert "Delete" in msg
