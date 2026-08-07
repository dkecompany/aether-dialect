"""Tests for unified federation declaration parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._constants import FEDERATION_DECLARATION_VERSION, FEDERATION_MAPPINGS_VERSION
from aetherdialect._federation import (
    FederationConfigError,
    FederationDeclarationError,
    load_federation_declaration_from_path,
    parse_federation_declaration,
)

_DECLARATION = {
    "version": FEDERATION_DECLARATION_VERSION,
    "federation_id": "fed_decl",
    "aliases": {},
    "cross_source_joins": [],
    "coordinator": {"row_cap": 1000},
    "logical_tables": [
        {
            "logical": "payments",
            "semantics": "union",
            "members": [
                {
                    "source": "east",
                    "table": "payment",
                    "columns": {"id": "payment_id", "amount": "amount"},
                }
            ],
        }
    ],
}


def test_unified_declaration_parses_manifest_and_mappings() -> None:
    manifest, mappings = parse_federation_declaration(_DECLARATION)
    assert manifest.federation_id == "fed_decl"
    assert len(mappings.logical_tables) == 1
    assert mappings.logical_tables[0].logical == "payments"


def test_unknown_top_level_key_is_rejected() -> None:
    bad = dict(_DECLARATION)
    bad["unexpected"] = True
    with pytest.raises(FederationDeclarationError, match="unknown keys"):
        parse_federation_declaration(bad)


def test_authored_sources_are_rejected_in_declaration() -> None:
    bad = dict(_DECLARATION)
    bad["sources"] = [{"source_id": "east", "engine": "duckdb", "role": "owner"}]
    with pytest.raises(FederationDeclarationError, match="sources are derived"):
        parse_federation_declaration(bad)


def test_authored_table_namespace_is_rejected_in_declaration() -> None:
    bad = dict(_DECLARATION)
    bad["table_namespace"] = {"payments": "east"}
    with pytest.raises(FederationDeclarationError, match="table_namespace is derived"):
        parse_federation_declaration(bad)


def test_absent_logical_sections_yield_empty_mappings() -> None:
    manifest, mappings = parse_federation_declaration(
        {
            "federation_id": "fed_decl",
            "aliases": {},
            "cross_source_joins": [],
            "coordinator": {"row_cap": 1000},
        }
    )
    assert manifest.federation_id == "fed_decl"
    assert mappings.logical_tables == ()
    assert mappings.logical_columns == ()
    assert mappings.version == FEDERATION_MAPPINGS_VERSION


def test_higher_declaration_version_is_refused() -> None:
    bad = dict(_DECLARATION)
    bad["version"] = "9.9.9"
    with pytest.raises(FederationDeclarationError, match="unsupported federation declaration version"):
        parse_federation_declaration(bad)


def test_load_federation_declaration_from_path_reports_declarations_file(tmp_path: Path) -> None:
    path = tmp_path / "federation_declaration.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FederationConfigError, match="declarations file"):
        load_federation_declaration_from_path(str(path))


def test_load_federation_declaration_from_path_parses_valid_document(tmp_path: Path) -> None:
    path = tmp_path / "federation_declaration.json"
    path.write_text(json.dumps(_DECLARATION), encoding="utf-8")
    manifest, mappings = load_federation_declaration_from_path(str(path))
    assert manifest.federation_id == "fed_decl"
    assert mappings.logical_tables
