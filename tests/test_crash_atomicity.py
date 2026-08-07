"""Crash injection at every artifact write pair: state stays old or fully new and usable."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import MIGRATION_MAP_ACTION_REMAP
from aetherdialect._contracts_base import ColumnRole, SchemaMigrationMap, SchemaMigrationMapEntry
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import read_artifact_manifest, read_gzip_json, write_artifact_manifest
from aetherdialect._federation import (
    FederationMappings,
    federation_artifact_paths,
    load_federation_composite_graph,
    persist_federation_tree,
)
from aetherdialect._schema_overrides import (
    apply_overrides_and_persist,
    load_overrides_sidecar,
    save_schema_to_cache,
)
from aetherdialect._templates import TemplateOps
from tests.federation_helpers import TwoMemberFederation, build_two_member_federation
from tests.test_migration_atomic import _make_template, _schema, _seed_store
from tests.test_schema import _ov_doc
from tests.test_template_store_integrity import _store_dir, _typed_template


def _fail_on_nth_replace(n: int, *, match: str | None = None) -> Callable[..., Any]:
    """Return an ``os.replace`` side effect that raises on the *n*th call."""
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _side_effect(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        calls.append((str(src), str(dst)))
        if match is not None and match not in str(dst):
            return real_replace(src, dst)
        if len(calls) == n:
            raise OSError("simulated crash on second write")
        return real_replace(src, dst)

    return _side_effect


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def _state_matches(before: dict[str, bytes], after: dict[str, bytes], expected_new: dict[str, bytes] | None) -> bool:
    if after == before:
        return True
    if expected_new is not None and after == expected_new:
        return True
    return False


WRITE_PAIRS = (
    pytest.param("template_shard_header", id="template_shard_header"),
    pytest.param("schema_graph_manifest", id="schema_graph_manifest"),
    pytest.param("federation_manifest_mappings_composite", id="federation_manifest_mappings_composite"),
    pytest.param("migration_map_store", id="migration_map_store"),
    pytest.param("override_sidecar_graph", id="override_sidecar_graph"),
)


@pytest.mark.fast
@pytest.mark.parametrize("write_pair", WRITE_PAIRS)
def test_crash_second_write_leaves_coherent_usable_state(
    tmp_path: Path,
    schema_graph: SchemaGraph,
    write_pair: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed second write must leave either the pre-image or a fully applied post-image."""
    before = _snapshot_tree(tmp_path)

    if write_pair == "template_shard_header":
        monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
        store_dir = _store_dir(tmp_path)
        os.makedirs(store_dir, exist_ok=True)
        monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
        graph_id = "sg_test000000000001__abcd1234"
        view = TemplateOps.empty_template_store(graph_id)
        TemplateOps.templates_to_store(view, {"T0001": _typed_template()})
        TemplateOps.save_template_store(view)
        before = _snapshot_tree(tmp_path)
        second = replace(_typed_template(tid="T0002"), trust_level=2)
        TemplateOps.templates_to_store(view, {"T0001": _typed_template(), second.id: second})
        with patch("aetherdialect._templates.os.replace", side_effect=_fail_on_nth_replace(1)):
            with pytest.raises(OSError, match="simulated crash"):
                TemplateOps.save_template_store(view)
        after = _snapshot_tree(tmp_path)
        assert _state_matches(before, after, None)
        loaded = TemplateOps.load_template_store(graph_id, schema=None)
        assert loaded.get_template("T0001") is not None
        assert "T0002" not in loaded.partition_map

    elif write_pair == "schema_graph_manifest":
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        schema_path = artifacts_dir / "schema_graph.json.gz"
        write_artifact_manifest(
            str(artifacts_dir),
            effective_structural_hash="eff_old",
            structural_hash="eff_old",
        )
        sg = deepcopy(schema_graph)
        sg.effective_structural_hash = "eff_old"
        sg.structural_hash = "eff_old"
        save_schema_to_cache(sg, str(schema_path))
        before = _snapshot_tree(tmp_path)
        sg.effective_structural_hash = "eff_new"
        sg.structural_hash = "eff_new"
        with patch(
            "aetherdialect._schema_overrides.write_gzip_json_atomic",
            side_effect=OSError("graph crash"),
        ):
            with pytest.raises(OSError, match="graph crash"):
                save_schema_to_cache(sg, str(schema_path))
        after = _snapshot_tree(tmp_path)
        assert after == before
        payload = read_gzip_json(str(schema_path))
        assert payload.get("effective_structural_hash") == "eff_old"
        manifest = read_artifact_manifest(str(artifacts_dir))
        assert manifest is not None
        assert manifest.effective_structural_hash == "eff_old"

    elif write_pair == "federation_manifest_mappings_composite":
        fed: TwoMemberFederation = build_two_member_federation()
        fed_dir = str(tmp_path / "fed")
        persist_federation_tree(
            fed_dir,
            manifest=fed.manifest,
            mappings=FederationMappings(version="0.2.1"),
            composite=fed.composite,
            member_graphs=fed.member_graphs,
        )
        before = _snapshot_tree(tmp_path)
        new_composite = deepcopy(fed.composite)
        new_composite.effective_structural_hash = "eff_mutated"
        calls = 0
        real_write = persist_federation_tree.__globals__["_write_federation_json_atomic"]

        def _flaky_write(path: str, payload: dict[str, Any]) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("federation second write failed")
            return real_write(path, payload)

        with patch("aetherdialect._federation._write_federation_json_atomic", side_effect=_flaky_write):
            with pytest.raises(OSError, match="federation second write failed"):
                persist_federation_tree(
                    fed_dir,
                    manifest=fed.manifest,
                    mappings=FederationMappings(version="0.2.1"),
                    composite=new_composite,
                    member_graphs=fed.member_graphs,
                )
        after = _snapshot_tree(tmp_path)
        assert _state_matches(before, after, None)
        paths = federation_artifact_paths(fed_dir)
        assert Path(paths["manifest"]).is_file()
        loaded = load_federation_composite_graph(fed_dir)
        assert loaded is not None or Path(paths["composite_schema"]).is_file()

    elif write_pair == "migration_map_store":
        schema = _schema(
            {
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "order_id": ColumnMetadata(
                            name="order_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="integer",
                            value_type="integer",
                            role=ColumnRole.NUMERIC_MEASURE.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="order_id",
                )
            }
        )
        tmpl = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": tmpl})
        before = _snapshot_tree(tmp_path)
        map_obj = SchemaMigrationMap(
            version=1,
            action=MIGRATION_MAP_ACTION_REMAP,
            table_renames=(SchemaMigrationMapEntry(entry_type="table", from_name="orders", to_name="sales_orders"),),
            column_renames=(),
            dropped_tables=(),
            dropped_columns=(),
            added_tables=(),
            added_columns=(),
        )
        with (
            patch("aetherdialect._templates.TemplateOps._apply_schema_rename_migration_to_store", return_value=(1, 0)),
            patch("aetherdialect._templates.TemplateOps._stamp_manifest", side_effect=OSError("stamp failed")),
            patch("aetherdialect._templates.TemplateOps.apply_structural_migration_from_map"),
            patch("aetherdialect._templates.migrate_sidecar_for_diff"),
        ):
            with pytest.raises(OSError, match="stamp failed"):
                TemplateOps.apply_schema_migration_map(
                    map_obj,
                    str(tmp_path),
                    schema,
                    tmp_path / "schema.json.gz",
                )
        after = _snapshot_tree(tmp_path)
        assert _state_matches(before, after, None)
        store_dir = TemplateOps.template_store_dir_for_space(str(tmp_path), "master")
        view = TemplateOps._load_partitioned_view_unlocked(store_dir)
        assert view is not None

    elif write_pair == "override_sidecar_graph":
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        schema_path = artifacts_dir / "schema_graph.json.gz"
        overrides_path = artifacts_dir / "schema_overrides.json"
        overrides_path.write_text(json.dumps(_ov_doc()), encoding="utf-8")
        sg = deepcopy(schema_graph)
        save_schema_to_cache(sg, str(schema_path))
        before = _snapshot_tree(tmp_path)
        with patch(
            "aetherdialect._schema_overrides._write_overrides_sidecar_payload",
            side_effect=OSError("sidecar crash"),
        ):
            with pytest.raises(OSError, match="sidecar crash"):
                apply_overrides_and_persist(sg, overrides_path, schema_json_path=str(schema_path))
        after = _snapshot_tree(tmp_path)
        assert _state_matches(before, after, None)
        payload = read_gzip_json(str(schema_path))
        assert isinstance(payload.get("tables"), dict)
        sidecar = load_overrides_sidecar(str(schema_path))
        assert sidecar is None or isinstance(sidecar, dict)

    else:
        raise AssertionError(f"unknown write pair: {write_pair}")


@pytest.mark.fast
def test_graph_manifest_pair_all_or_nothing(
    tmp_path: Path,
    schema_graph: SchemaGraph,
) -> None:
    """Second live-path replace must not leave a new graph with an old manifest."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    schema_path = artifacts_dir / "schema_graph.json.gz"
    write_artifact_manifest(
        str(artifacts_dir),
        effective_structural_hash="eff_old",
        structural_hash="eff_old",
    )
    sg = deepcopy(schema_graph)
    sg.effective_structural_hash = "eff_old"
    sg.structural_hash = "eff_old"
    save_schema_to_cache(sg, str(schema_path))
    before = _snapshot_tree(tmp_path)

    sg.effective_structural_hash = "eff_new"
    sg.structural_hash = "eff_new"
    live_graph = str(schema_path)
    live_manifest = str(artifacts_dir / "artifact_manifest.json")
    live_replaces = 0
    real_replace = os.replace

    def _side_effect(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal live_replaces
        dst_s = str(dst)
        if dst_s in (live_graph, live_manifest):
            live_replaces += 1
            if live_replaces == 2:
                raise OSError("simulated crash on second commit replace")
        return real_replace(src, dst)

    with patch("aetherdialect._schema_overrides.os.replace", side_effect=_side_effect):
        with pytest.raises(OSError, match="simulated crash"):
            save_schema_to_cache(sg, str(schema_path))

    after = _snapshot_tree(tmp_path)
    assert after == before
    payload = read_gzip_json(str(schema_path))
    assert payload.get("effective_structural_hash") == "eff_old"
    manifest = read_artifact_manifest(str(artifacts_dir))
    assert manifest is not None
    assert manifest.effective_structural_hash == "eff_old"


@pytest.mark.fast
def test_template_mid_commit_keeps_old_or_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mid-commit template save must leave either the pre-image or a fully usable post-image."""
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    store_dir = _store_dir(tmp_path)
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    graph_id = "sg_test000000000001__abcd1234"
    view = TemplateOps.empty_template_store(graph_id)
    first = _typed_template(tid="T0001")
    TemplateOps.templates_to_store(view, {first.id: first})
    TemplateOps.save_template_store(view)

    second = replace(_typed_template(tid="T0002"), trust_level=2)
    TemplateOps.templates_to_store(view, {first.id: first, second.id: second})
    original_replace = os.replace
    calls: list[tuple[str, str]] = []

    def _replace_after_header(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        calls.append((str(src), str(dst)))
        if len(calls) == 2:
            raise OSError("simulated crash after header commit")
        return original_replace(src, dst)

    with patch("aetherdialect._templates.os.replace", side_effect=_replace_after_header):
        with pytest.raises(OSError, match="simulated crash"):
            TemplateOps.save_template_store(view)

    loaded = TemplateOps.load_template_store(graph_id, schema=None)
    has_first = loaded.get_template("T0001") is not None
    has_second = "T0002" in loaded.partition_map and loaded.get_template("T0002") is not None
    old_state = has_first and not has_second
    new_state = has_first and has_second
    assert old_state or new_state
    assert not (has_second and loaded.get_template("T0002") is None)


@pytest.mark.fast
def test_federation_persist_torn_tree_refused_or_old(tmp_path: Path) -> None:
    """Interrupted federation four-file persist must keep the old tree or refuse torn partial state."""
    fed = build_two_member_federation()
    fed_dir = str(tmp_path / "fed")
    persist_federation_tree(
        fed_dir,
        manifest=fed.manifest,
        mappings=FederationMappings(version="0.2.1"),
        composite=fed.composite,
        member_graphs=fed.member_graphs,
    )
    before = _snapshot_tree(tmp_path)
    old_composite = load_federation_composite_graph(fed_dir)
    assert old_composite is not None

    new_composite = deepcopy(fed.composite)
    new_composite.effective_structural_hash = "eff_mutated_n3"
    new_mappings = FederationMappings(version=99)
    paths = federation_artifact_paths(fed_dir)
    live_targets = (
        paths["manifest"],
        paths["mappings"],
        paths["composite_schema"],
        paths["artifact_manifest"],
    )
    live_replaces = 0
    real_replace = os.replace

    def _side_effect(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal live_replaces
        dst_s = str(dst)
        if dst_s in live_targets:
            live_replaces += 1
            if live_replaces == 3:
                raise OSError("simulated crash before federation manifest commit")
        return real_replace(src, dst)

    with patch("aetherdialect._federation.os.replace", side_effect=_side_effect):
        with pytest.raises(OSError, match="simulated crash"):
            persist_federation_tree(
                fed_dir,
                manifest=fed.manifest,
                mappings=new_mappings,
                composite=new_composite,
                member_graphs=fed.member_graphs,
            )

    after = _snapshot_tree(tmp_path)
    loaded = load_federation_composite_graph(fed_dir)
    assert after == before or loaded is None
    if loaded is not None:
        assert str(loaded.effective_structural_hash or "") == str(old_composite.effective_structural_hash or "")
        assert str(loaded.effective_structural_hash or "") != "eff_mutated_n3"
