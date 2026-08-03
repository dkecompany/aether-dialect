"""Behaviour tests for federation declaration constructor and export surface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import FEDERATION_DECLARATION_FILENAME
from aetherdialect._contracts_base import FederationMappings
from aetherdialect._federation import (
    compose_composite_graph,
    federation_declaration_document,
    parse_federation_declaration,
    parse_federation_manifest,
)
from tests.federation_helpers import write_federation_declaration_file
from tests.test_aether_federation_public_surface import (
    _MANIFEST,
    _fed,
    _graph,
    _init_bundle,
    _minimal_member,
)

_MAPPINGS_PAYLOAD = {
    "version": 2,
    "logical_tables": [
        {
            "logical": "payment",
            "semantics": "union",
            "members": [
                {"source": "a", "table": "payment_a", "columns": {"id": "id"}},
                {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
            ],
        }
    ],
    "logical_columns": [],
}


def test_constructor_rejects_legacy_manifest_file_keyword() -> None:
    with pytest.raises(TypeError):
        AetherFederation(
            "fed_public",
            members={"conn_a": _minimal_member(connection="a")},
            manifest_file="/tmp/legacy.json",
        )


def test_export_federation_declaration_writes_authored_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fed = _fed()
    monkeypatch.chdir(tmp_path)
    path = fed.export_federation_declaration()
    assert path.name == FEDERATION_DECLARATION_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "sources" not in payload
    assert "table_namespace" not in payload
    assert payload["federation_id"] == "fed_public"


def test_export_federation_declaration_round_trips_through_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fed = _fed()
    monkeypatch.chdir(tmp_path)
    exported = fed.export_federation_declaration()
    manifest, mappings = parse_federation_declaration(exported.read_text(encoding="utf-8"))
    assert manifest.federation_id == "fed_public"
    assert manifest.cross_source_joins
    assert mappings.logical_tables == ()


def test_federation_declaration_document_omits_derived_roster_fields() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version=2)
    doc = federation_declaration_document(manifest, mappings)
    assert "sources" not in doc
    assert "table_namespace" not in doc
    assert doc["federation_id"] == "fed_public"


def test_apply_federation_declaration_reads_full_document_from_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration_path = write_federation_declaration_file(tmp_path, _MANIFEST, _MAPPINGS_PAYLOAD)
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    bundle = _init_bundle(manifest, composite)
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_public",
            members={"conn_a": _minimal_member(connection="a"), "conn_b": _minimal_member(connection="b")},
            declaration_file=str(declaration_path),
        )
    editor = Path(fed._artifacts_dir) / FEDERATION_DECLARATION_FILENAME
    editor.parent.mkdir(parents=True, exist_ok=True)
    editor.write_text(declaration_path.read_text(encoding="utf-8"), encoding="utf-8")
    with patch.object(AetherFederation, "_recompose") as recompose:
        fed.apply_federation_declaration()
    assert fed._mappings is not None
    assert len(fed._mappings.logical_tables) == 1
    recompose.assert_called_once()


def test_in_memory_mappings_override_declaration_file_on_recompose(tmp_path: Path) -> None:
    declaration_path = write_federation_declaration_file(tmp_path, _MANIFEST, _MAPPINGS_PAYLOAD)
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    bundle = _init_bundle(manifest, composite)
    override_mappings = FederationMappings(version=2, logical_tables=(), logical_columns=())
    captured: dict[str, object] = {}

    def _capture_init(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return bundle

    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_public",
            members={"conn_a": _minimal_member(connection="a"), "conn_b": _minimal_member(connection="b")},
            declaration_file=str(declaration_path),
        )
    fed._mappings = override_mappings
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", side_effect=_capture_init):
        fed._recompose()
    declaration = captured.get("declaration")
    assert isinstance(declaration, tuple)
    assert declaration[1] is override_mappings
