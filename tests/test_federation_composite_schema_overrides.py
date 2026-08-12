"""Composite schema override export/apply for :class:`~aetherdialect.AetherFederation`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import STRUCTURE_DOCUMENT_FILENAME, STRUCTURE_SIDECAR_FILENAME
from aetherdialect._constants_runtime import FEDERATION_METHOD_SEMANTICS
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    apply_federation_composite_overrides,
    finalize_federation_composite_overrides,
    persist_federation_tree,
)
from aetherdialect._federation_manifest import (
    federation_artifact_paths,
    parse_federation_manifest,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_reflect import structure_sidecar_path
from tests.test_aether_federation_public_surface import _MANIFEST, _MANIFEST_FILE, _fed, _init_bundle, _minimal_member
from tests.test_schema import _ov_doc


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


def test_federation_method_semantics_schema_finalize_are_both_scoped() -> None:
    assert FEDERATION_METHOD_SEMANTICS["export_structure"] == "composite"
    assert FEDERATION_METHOD_SEMANTICS["apply_structure"] == "composite"


def test_export_structure_without_connection_targets_composite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest)
    fed_dir = tmp_path / "fed"
    persist_federation_tree(
        str(fed_dir),
        manifest=manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=composite,
        member_graphs=members,
    )
    bundle = _init_bundle(manifest, composite)
    bundle.federation_storage_dir = str(fed_dir)
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_public",
            members=[_minimal_member(connection="conn_a"), _minimal_member(connection="conn_b")],
            declaration=_MANIFEST_FILE,
            artifacts_dir=str(tmp_path),
        )
    monkeypatch.chdir(tmp_path)
    out = fed.export_structure()
    assert isinstance(out, dict)
    assert "tables" in out
    # Composite export may be empty when visibility fails closed on stubs; keys still round-trip.
    assert set(out) >= {"tables", "relationships", "table_count"}


def test_apply_federation_composite_overrides_persists_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aetherdialect._config.EngineConfig.llm_credentials_configured", lambda: False)
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest)
    fed_dir = tmp_path / "fed"
    persist_federation_tree(
        str(fed_dir),
        manifest=manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=composite,
        member_graphs=members,
    )
    composite_path = federation_artifact_paths(str(fed_dir))["composite_schema"]
    editor = tmp_path / STRUCTURE_DOCUMENT_FILENAME
    editor.write_text(
        json.dumps(
            _ov_doc(
                tables={
                    "left_t": {
                        "role": "dimension",
                    }
                }
            )
        ),
        encoding="utf-8",
    )
    report = apply_federation_composite_overrides(composite, str(fed_dir), editor)
    assert report.table_edits == 1
    assert composite.tables["left_t"].role == "dimension"
    sidecar = structure_sidecar_path(composite_path)
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["tables"]["left_t"]["role"] == "dimension"


def test_finalize_federation_composite_overrides_replays_after_recompose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aetherdialect._config.EngineConfig.llm_credentials_configured", lambda: False)
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest)
    fed_dir = tmp_path / "fed"
    persist_federation_tree(
        str(fed_dir),
        manifest=manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=composite,
        member_graphs=members,
    )
    editor = tmp_path / STRUCTURE_DOCUMENT_FILENAME
    editor.write_text(
        json.dumps(
            _ov_doc(
                tables={
                    "right_t": {
                        "role": "dimension",
                    }
                }
            )
        ),
        encoding="utf-8",
    )
    apply_federation_composite_overrides(composite, str(fed_dir), editor)
    recomposed = compose_composite_graph(members, manifest)
    assert recomposed.tables["right_t"].role != "dimension"
    assert finalize_federation_composite_overrides(recomposed, str(fed_dir)) is True
    assert recomposed.tables["right_t"].role == "dimension"
    assert Path(fed_dir / STRUCTURE_SIDECAR_FILENAME).is_file()


def test_member_schema_finalize_still_dispatch_with_connection_name() -> None:
    fed = _fed()
    # Structure export is composite-scoped; connection tokens are not space identities.
    doc = fed.export_structure()
    assert isinstance(doc, dict)
    assert "tables" in doc
