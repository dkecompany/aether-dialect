"""Behaviour tests for federation declaration constructor and export surface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_schema import FederationMappings
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
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
    "version": "0.2.3",
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


def test_constructor_rejects_manifest_file_keyword() -> None:
    with pytest.raises(TypeError):
        AetherFederation(
            "fed_public",
            members=(_minimal_member(connection="a"), _minimal_member(connection="b")),
            manifest_file="/tmp/manifest.json",
        )


def test_export_federation_writes_authored_shape() -> None:
    fed = _fed()
    payload = fed.export_federation()
    assert isinstance(payload, dict)
    assert "sources" not in payload
    assert "table_namespace" not in payload
    assert payload["federation_id"] == "fed_public"


def test_export_federation_round_trips_through_parser() -> None:
    fed = _fed()
    exported = fed.export_federation()
    manifest, mappings = parse_federation_declaration(exported)
    assert manifest.federation_id == "fed_public"
    assert manifest.cross_source_joins
    assert mappings.logical_tables == ()


def test_federation_declaration_document_omits_derived_roster_fields() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.3")
    doc = federation_declaration_document(manifest, mappings)
    assert "sources" not in doc
    assert "table_namespace" not in doc
    assert doc["federation_id"] == "fed_public"


def test_apply_federation_reads_full_document(tmp_path: Path) -> None:
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
            members=[_minimal_member(connection="conn_a"), _minimal_member(connection="conn_b")],
            declaration=str(declaration_path),
        )
    payload = json.loads(declaration_path.read_text(encoding="utf-8"))
    with patch.object(AetherFederation, "_recompose") as recompose:
        fed.apply_federation(payload)
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
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_public",
            members=[_minimal_member(connection="conn_a"), _minimal_member(connection="conn_b")],
            declaration=str(declaration_path),
        )
    assert fed._mappings is not None
