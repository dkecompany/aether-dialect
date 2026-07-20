"""Tests for stable schema-graph identity mint, persistence, and upgrade."""

from __future__ import annotations

import gzip
import json
import os
import re

import pytest

from aetherdialect._config import ConfigError
from aetherdialect._constants import SCHEMA_GRAPH_ID_PREFIX
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import (
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._core_utils import read_artifact_manifest, write_gzip_json_atomic
from aetherdialect._schema_graph import (
    assign_schema_graph_hashes,
    derive_deterministic_schema_graph_id,
    mint_schema_graph_id,
    upgrade_artifacts_schema_graph_id,
)


def _table(name: str) -> TableMetadata:
    return TableMetadata(name=name, columns={}, primary_key=[], foreign_keys=[], row_count=1)


def test_owner_mint_matches_pattern() -> None:
    sg = SchemaGraph(join_paths_multi={}, tables={"a": _table("a")})
    ctx = EngineContext()
    assign_schema_graph_hashes(sg, ctx, "", schema_role="owner")
    assert re.fullmatch(SCHEMA_GRAPH_ID_PREFIX + r"[0-9a-f]{16}__.{8}", sg.schema_graph_id)
    first = sg.schema_graph_id
    assign_schema_graph_hashes(sg, ctx, "", schema_role="owner")
    assert sg.schema_graph_id == first


def test_owner_structural_change_mints_new_id() -> None:
    sg = SchemaGraph(join_paths_multi={}, tables={"a": _table("a")})
    ctx = EngineContext()
    assign_schema_graph_hashes(sg, ctx, "", schema_role="owner")
    prior = sg.schema_graph_id
    sg.tables["b"] = _table("b")
    assign_schema_graph_hashes(sg, ctx, "", schema_role="owner")
    assert sg.schema_graph_id != prior


def test_deterministic_derive_stable() -> None:
    eff = "abc123effective"
    structural = "structural999"
    a = derive_deterministic_schema_graph_id(eff, structural)
    b = derive_deterministic_schema_graph_id(eff, structural)
    assert a == b
    assert a.startswith(SCHEMA_GRAPH_ID_PREFIX)
    assert structural[:8] in a


def test_schema_graph_round_trip_dict() -> None:
    sg = SchemaGraph(
        join_paths_multi={},
        tables={"t": _table("t")},
        schema_graph_id="sg_0123456789abcdef__deadbeef",
        effective_structural_hash="eff1",
    )
    restored = SchemaGraph.from_dict(sg.to_dict())
    assert restored.schema_graph_id == sg.schema_graph_id
    assert restored.effective_structural_hash == sg.effective_structural_hash


def test_gzip_cache_round_trip_schema_graph_id(tmp_path) -> None:
    sg = SchemaGraph(join_paths_multi={}, tables={"t": _table("t")})
    ctx = EngineContext()
    assign_schema_graph_hashes(sg, ctx, "", schema_role="owner")
    cache_path = tmp_path / "schema_graph.json.gz"
    payload = sg.to_dict()
    write_gzip_json_atomic(str(cache_path), payload, sort_keys=True)
    with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["schema_graph_id"] == sg.schema_graph_id


def test_upgrade_artifacts_backfills_manifest_and_header(tmp_path, monkeypatch) -> None:
    from aetherdialect._constants import (
        TEMPLATE_STORE_HEADER_FILENAME,
        TEMPLATE_STORE_PARTITION_PREFIX,
        TEMPLATE_STORE_SEGMENT,
    )
    from aetherdialect._core_utils import write_artifact_manifest

    adir = str(tmp_path)
    eff = "legacy_eff_hash_value"
    store_dir = os.path.join(adir, TEMPLATE_STORE_SEGMENT)
    os.makedirs(store_dir, exist_ok=True)
    graph_id = derive_deterministic_schema_graph_id(eff, eff)
    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    write_gzip_json_atomic(
        header_path,
        {"effective_structural_hash": eff, "schema_hash": eff},
        sort_keys=True,
    )
    part_path = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}00.json.gz")
    write_gzip_json_atomic(
        part_path,
        {"T1": {"id": "T1", "effective_structural_hash": eff}},
        sort_keys=True,
    )
    write_artifact_manifest(adir, effective_structural_hash=eff, structural_hash=eff)
    counts = upgrade_artifacts_schema_graph_id(adir)
    assert counts["template_rows"] >= 1
    manifest = read_artifact_manifest(adir)
    assert manifest is not None
    assert manifest.schema_graph_id == graph_id


def test_consumer_requires_pin() -> None:
    sg = SchemaGraph(join_paths_multi={}, tables={"a": _table("a")})
    ctx = EngineContext()
    with pytest.raises(ConfigError, match="pinned schema_graph_id"):
        assign_schema_graph_hashes(sg, ctx, "", schema_role="consumer")


def test_mint_schema_graph_id_format() -> None:
    out = mint_schema_graph_id(seed_hex="0123456789abcdef", structural_hash="abcdef0123456789")
    assert out == f"{SCHEMA_GRAPH_ID_PREFIX}0123456789abcdef__abcdef01"
