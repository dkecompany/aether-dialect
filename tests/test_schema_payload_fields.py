"""Tests for unified schema payload field sets and master vs interactive visibility."""

from __future__ import annotations

import json

from aetherdialect._constants import (
    COMPOSE_FIELDS,
    FULL_FIELDS,
    GROUND_FIELDS,
    INTERPRET_FIELDS,
    SCHEMA_FIELD_DERIVED,
    SCHEMA_FIELD_DESCRIPTION,
    SCHEMA_FIELD_ENUM,
    SCHEMA_FIELD_KEYS,
    SCHEMA_FIELD_ROLE,
    SCHEMA_FIELD_TRUTH_VALUE,
    SCHEMA_FIELD_TYPE,
)
from aetherdialect._contracts_base import SensitivityClassification, TableRole
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata


def _make_graph_with_hidden_column() -> SchemaGraph:
    hidden_col = ColumnMetadata(name="secret_col", data_type="varchar", description="hidden metric")
    SensitivityClassification.apply_to(hidden_col, SensitivityClassification.HIDDEN)
    visible_col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
    table = TableMetadata(
        name="tbl_a",
        columns={"id": visible_col, "secret_col": hidden_col},
        primary_key=["id"],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
    )
    sg = SchemaGraph(
        tables={"tbl_a": table},
        enum_values={"status_enum": ("active", "inactive")},
        join_paths_multi={},
        effective_structural_hash="test",
    )
    return sg


def test_interpret_fields_exclude_structure() -> None:
    sg = _make_graph_with_hidden_column()
    payload = json.loads(sg.schema_payload_interpret(owner_master_scope=True))
    col = payload["tbl_a"]["columns"]["secret_col"]
    assert SCHEMA_FIELD_DESCRIPTION in col or "description" in col
    assert "pk" not in col
    assert "type" not in col


def test_compose_fields_exclude_descriptions() -> None:
    sg = _make_graph_with_hidden_column()
    payload = json.loads(sg.schema_payload_compose(["tbl_a"], owner_master_scope=True))
    col = payload["tbl_a"]["columns"]["secret_col"]
    assert "description" not in col
    assert col.get("type") == "string"


def test_master_scope_includes_hidden_column() -> None:
    sg = _make_graph_with_hidden_column()
    payload = json.loads(sg.schema_payload_json(FULL_FIELDS, owner_master_scope=True))
    assert "secret_col" in payload["tbl_a"]["columns"]


def test_interactive_scope_excludes_hidden_column() -> None:
    sg = _make_graph_with_hidden_column()
    payload = json.loads(sg.schema_payload_json(FULL_FIELDS, owner_master_scope=False))
    assert "secret_col" not in payload["tbl_a"]["columns"]


def test_field_set_constants() -> None:
    assert INTERPRET_FIELDS == frozenset({SCHEMA_FIELD_DESCRIPTION, SCHEMA_FIELD_ENUM})
    assert SCHEMA_FIELD_TYPE in GROUND_FIELDS
    assert SCHEMA_FIELD_DESCRIPTION not in COMPOSE_FIELDS
    assert SCHEMA_FIELD_KEYS in COMPOSE_FIELDS
    assert FULL_FIELDS == frozenset(
        {
            SCHEMA_FIELD_DESCRIPTION,
            SCHEMA_FIELD_DERIVED,
            SCHEMA_FIELD_ROLE,
            SCHEMA_FIELD_TYPE,
            SCHEMA_FIELD_TRUTH_VALUE,
            SCHEMA_FIELD_KEYS,
            SCHEMA_FIELD_ENUM,
        }
    )
    assert SCHEMA_FIELD_TRUTH_VALUE in GROUND_FIELDS
    assert SCHEMA_FIELD_ROLE in COMPOSE_FIELDS
