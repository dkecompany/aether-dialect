"""Public API surface behaviour for federation construction, declaration apply, upload inspection, and export parity."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, AetherFederation, MigrationPreview
from aetherdialect._constants import FEDERATION_DECLARATION_FILENAME, FEDERATION_DECLARATION_VERSION
from aetherdialect._contracts_base import FederationDeclarationError, FederationMappings
from aetherdialect._federation import (
    binding_from_member_engine,
    compose_composite_graph,
    federation_declaration_document,
    load_federation_declaration_from_path,
    parse_federation_declaration,
    parse_federation_manifest,
    parse_federation_mappings,
)
from tests.federation_helpers import write_federation_declaration_file
from tests.test_aether_federation_public_surface import (
    _MANIFEST,
    _graph,
    _init_bundle,
    _minimal_member,
)

_MAPPINGS_PAYLOAD = {
    "version": "0.2.1",
    "logical_tables": [],
    "logical_columns": [],
}


@pytest.mark.fast
def test_aether_engine_connection_parameter_sets_handle() -> None:
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
        init.return_value = MagicMock(
            runtime_config=MagicMock(engine="duckdb"),
            llm_config=MagicMock(),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/m1",
            store=MagicMock(),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={},
            schema_role="owner",
            consumer_visible_objects=None,
            context_name="master",
            data_quality_report=None,
        )
        engine = AetherEngine(connection="storefront_pg")
    init.assert_called_once()
    assert init.call_args.kwargs["connection"] == "storefront_pg"
    assert engine._named_connection == "storefront_pg"


@pytest.mark.fast
def test_binding_error_explains_source_id_vs_connection() -> None:
    member = _minimal_member(connection="storefront")
    member._connection = "storefront_pg"
    with pytest.raises(Exception, match="source_id") as exc:
        binding_from_member_engine("storefront", member)
    msg = str(exc.value)
    assert "connection" in msg.lower() or "storefront_pg" in msg


@pytest.mark.fast
def test_federation_member_key_is_source_id_not_toml_connection() -> None:
    member = _minimal_member(connection="storefront")
    member._named_connection = "storefront_pg"
    member._connection = "storefront"
    binding = binding_from_member_engine("storefront", member)
    assert binding.source_id == "storefront"
    assert binding.connection == "storefront_pg"


@pytest.mark.fast
def test_apply_federation_declaration_replaces_apply_federation_mappings() -> None:
    assert hasattr(AetherFederation, "apply_federation_declaration")
    assert not hasattr(AetherFederation, "apply_federation_mappings")


def test_apply_federation_declaration_applies_manifest_sections(
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
            members={"a": _minimal_member(connection="a"), "b": _minimal_member(connection="b")},
            declaration_file=str(declaration_path),
        )
    editor = Path(fed._artifacts_dir) / FEDERATION_DECLARATION_FILENAME
    editor.parent.mkdir(parents=True, exist_ok=True)
    edited = dict(_MANIFEST)
    edited["cross_source_joins"] = [
        {"left": "left_t.id", "right": "right_t.id", "kind": "left", "logical_key": "id"},
    ]
    editor.write_text(
        json.dumps(
            {
                "version": FEDERATION_DECLARATION_VERSION,
                **edited,
                "logical_columns": [],
                "logical_tables": _MAPPINGS_PAYLOAD["logical_tables"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with patch.object(AetherFederation, "_recompose") as recompose:
        fed.apply_federation_declaration()
    recompose.assert_called_once()
    manifest, _ = load_federation_declaration_from_path(fed._declaration_file)
    assert manifest.cross_source_joins[0].kind == "left"


@pytest.mark.fast
def test_parse_federation_declaration_rejects_version_below_minimum() -> None:
    payload = {"version": 0, "federation_id": "fed", "cross_source_joins": []}
    with pytest.raises(FederationDeclarationError, match="version"):
        parse_federation_declaration(payload)


@pytest.mark.fast
def test_parse_federation_mappings_rejects_unknown_version() -> None:
    with pytest.raises(Exception, match="version"):
        parse_federation_mappings({"version": 99, "logical_columns": [], "logical_tables": []})


@pytest.mark.fast
def test_export_declaration_round_trips_empty_logical_sections() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.1")
    doc = federation_declaration_document(manifest, mappings)
    assert "logical_columns" in doc
    assert "logical_tables" in doc
    assert doc["logical_columns"] == []
    assert doc["logical_tables"] == []
    round_tripped, _ = parse_federation_declaration(doc)
    assert round_tripped.federation_id == "fed_public"


@pytest.mark.fast
def test_export_union_omits_empty_authoritative_source() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(
        version="0.2.1",
        logical_tables=(
            __import__("aetherdialect._contracts_base", fromlist=["LogicalTableMapping"]).LogicalTableMapping(
                logical="payment",
                semantics="union",
                authoritative_source="",
                members=(
                    __import__("aetherdialect._contracts_base", fromlist=["LogicalTableMember"]).LogicalTableMember(
                        source="a", table="t", columns={"id": "id"}
                    ),
                ),
            ),
        ),
    )
    doc = federation_declaration_document(manifest, mappings)
    table_entry = doc["logical_tables"][0]
    assert "authoritative_source" not in table_entry
    round_tripped, _ = parse_federation_declaration(doc)
    assert round_tripped.federation_id == "fed_public"


@pytest.mark.fast
def test_inspect_tabular_upload_is_public(tmp_path: Path) -> None:
    from aetherdialect import inspect_tabular_upload

    path = tmp_path / "items.csv"
    path.write_text("id,name\n1,Alice\n", encoding="utf-8")
    report = inspect_tabular_upload(path)
    assert report.ok
    assert hasattr(report, "requires_review")
    assert hasattr(report, "confirmed_selections")


@pytest.mark.fast
def test_aether_engine_exposes_data_quality_report_property() -> None:
    from aetherdialect._contracts_base import DataQualityReport

    report = DataQualityReport(ok=True, issues=(), narrative="ok")
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
        init.return_value = MagicMock(
            runtime_config=MagicMock(engine="csv"),
            llm_config=MagicMock(),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/m3",
            store=MagicMock(),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={},
            schema_role="owner",
            consumer_visible_objects=None,
            context_name="master",
            data_quality_report=report,
        )
        engine = AetherEngine()
    assert engine.data_quality_report is report


@pytest.mark.fast
def test_constructor_accepts_source_selections() -> None:
    selections = {"items.csv": {"header_row": 2}}
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
        init.return_value = MagicMock(
            runtime_config=MagicMock(engine="csv"),
            llm_config=MagicMock(),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/m3b",
            store=MagicMock(),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={},
            schema_role="owner",
            consumer_visible_objects=None,
            context_name="master",
            data_quality_report=None,
        )
        AetherEngine(source_selections=selections)
    assert init.call_args.kwargs["source_selections"] == selections


@pytest.mark.fast
def test_federation_manifest_and_mappings_exports_are_not_public() -> None:
    assert not hasattr(AetherFederation, "export_federation_manifest")
    assert not hasattr(AetherFederation, "export_federation_mappings")


@pytest.mark.fast
def test_aether_federation_apply_migration_map_is_documented() -> None:
    method = AetherFederation.apply_migration_map
    assert method.__doc__ is not None
    assert "federation_migration_map" in method.__doc__


@pytest.mark.fast
def test_aether_engine_preview_migration_map_returns_migration_preview() -> None:
    with patch("aetherdialect.aetherdialect.preview_schema_migration") as preview:
        preview.return_value = MigrationPreview(
            tier="compatible",
            affected_tables=(),
            affected_columns=(),
            skeleton_path="",
        )
        with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
            init.return_value = MagicMock(
                runtime_config=MagicMock(engine="duckdb"),
                llm_config=MagicMock(),
                schema_graph=MagicMock(),
                dialect=MagicMock(),
                artifacts_dir="/tmp/m4",
                store=MagicMock(),
                templates={},
                rejected={},
                schema_terms=set(),
                schema_stats={},
                schema_role="owner",
                consumer_visible_objects=None,
                context_name="master",
                data_quality_report=None,
            )
            engine = AetherEngine()
        result = engine.preview_migration_map()
    assert isinstance(result, MigrationPreview)
