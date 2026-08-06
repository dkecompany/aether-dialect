"""Catalog comments and description precedence wiring."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import DescriptionOwner
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _merge_column_metadata_strictest,
    compose_composite_graph,
    parse_federation_manifest,
    parse_federation_mappings,
    reconcile_composite_classifications,
)
from aetherdialect._schema_build import tables_meta_to_schema_graph
from aetherdialect._schema_catalog import (
    apply_catalog_descriptions_from_tables_meta,
    apply_column_roles_llm,
    parse_sql_file,
)
from tests.federation_helpers import stamp_union_disjointness_profiling


def _payment_union_fixtures(
    *,
    a_description: str = "Payments",
    b_description: str = "Payments",
    a_description_owner: DescriptionOwner | None = DescriptionOwner.CATALOG,
    b_description_owner: DescriptionOwner | None = DescriptionOwner.CATALOG,
) -> tuple[dict, dict, dict]:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_payment_only",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"payment_a": "a", "payment_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "a": SchemaGraph(
            tables={
                "payment": TableMetadata(
                    name="payment",
                    columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="a",
                    description=a_description,
                    description_owner=a_description_owner,
                )
            },
            join_paths_multi={},
        ),
        "b": SchemaGraph(
            tables={
                "payment": TableMetadata(
                    name="payment",
                    columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="b",
                    description=b_description,
                    description_owner=b_description_owner,
                )
            },
            join_paths_multi={},
        ),
    }
    stamp_union_disjointness_profiling(members["a"].tables["payment"], key_col="id", overlap_sample=("a1",))
    stamp_union_disjointness_profiling(members["b"].tables["payment"], key_col="id", overlap_sample=("b1",))
    members["a"].tables["payment"].row_count = 1
    members["b"].tables["payment"].row_count = 1
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    return manifest, members, mappings


@pytest.mark.fast
def test_sql_file_ddl_comments_become_catalog_descriptions(tmp_path) -> None:
    ddl = """
    CREATE TABLE orders (
        id INT PRIMARY KEY COMMENT 'Order surrogate key',
        amount DECIMAL(10,2)
    ) COMMENT='Customer purchase records';
    COMMENT ON COLUMN orders.amount IS 'Purchase amount';
    """
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text(ddl, encoding="utf-8")
    tables = parse_sql_file(sql_path)
    assert tables["orders"]["table_comment"] == "Customer purchase records"
    assert tables["orders"]["column_comments"]["id"] == "Order surrogate key"
    assert tables["orders"]["column_comments"]["amount"] == "Purchase amount"

    sg = tables_meta_to_schema_graph(tables)
    assert sg.tables["orders"].description == "Customer purchase records"
    assert sg.tables["orders"].description_owner == DescriptionOwner.CATALOG
    assert sg.tables["orders"].columns["id"].description == "Order surrogate key"
    assert sg.tables["orders"].columns["amount"].description == "Purchase amount"


@pytest.mark.fast
def test_sql_file_catalog_descriptions_respect_higher_owner(tmp_path) -> None:
    ddl = "CREATE TABLE orders (id INT PRIMARY KEY) COMMENT='Catalog text';"
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text(ddl, encoding="utf-8")
    tables = parse_sql_file(sql_path)
    sg = tables_meta_to_schema_graph({k: {**v, "table_comment": None} for k, v in tables.items()})
    sg.tables["orders"].description = "User text"
    sg.tables["orders"].description_owner = DescriptionOwner.USER_OVERRIDE
    apply_catalog_descriptions_from_tables_meta(sg, tables)
    assert sg.tables["orders"].description == "User text"
    assert sg.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE


@pytest.mark.fast
def test_tables_meta_table_comment_wired_through_set_description() -> None:
    meta = {
        "orders": {
            "column_names_original": ["id"],
            "column_types": ["integer"],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "table_comment": "Customer purchase records",
            "column_comments": {"id": "Surrogate primary key"},
        }
    }
    sg = tables_meta_to_schema_graph(meta)
    assert sg.tables["orders"].description == "Customer purchase records"
    assert sg.tables["orders"].description_owner == DescriptionOwner.CATALOG
    assert sg.tables["orders"].columns["id"].description == "Surrogate primary key"
    assert sg.tables["orders"].columns["id"].description_owner == DescriptionOwner.CATALOG


@pytest.mark.fast
def test_apply_catalog_descriptions_respects_higher_owner() -> None:
    meta = {
        "orders": {
            "column_names_original": ["id"],
            "column_types": ["integer"],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "table_comment": "Catalog text",
        }
    }
    sg = tables_meta_to_schema_graph({k: {**v, "table_comment": None} for k, v in meta.items()})
    sg.tables["orders"].description = "User text"
    sg.tables["orders"].description_owner = DescriptionOwner.USER_OVERRIDE
    apply_catalog_descriptions_from_tables_meta(sg, meta)
    assert sg.tables["orders"].description == "User text"
    assert sg.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE


@pytest.mark.fast
def test_apply_column_roles_llm_with_notes_writes_notes_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    table = TableMetadata(
        name="orders",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", value_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
    )
    sg = SchemaGraph(tables={"orders": table}, join_paths_multi={})

    def _fake_classify(_schema, notes_content=None, *, column_scope=None, **kwargs):
        _ = notes_content, column_scope
        return {
            "orders": (
                "fact",
                "Orders from notes",
                {"id": ("identifier", "Order id from notes", "none")},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_catalog.llm_classify_schema", _fake_classify)
    apply_column_roles_llm(sg, notes_content="Domain notes about orders.")
    assert sg.tables["orders"].description_owner == DescriptionOwner.NOTES
    assert sg.tables["orders"].columns["id"].description_owner == DescriptionOwner.NOTES


@pytest.mark.fast
def test_merge_column_metadata_strictest_uses_description_precedence() -> None:
    low = ColumnMetadata(
        name="id",
        data_type="integer",
        description="catalog",
        description_owner=DescriptionOwner.CATALOG,
    )
    high = ColumnMetadata(
        name="id",
        data_type="integer",
        description="profile",
        description_owner=DescriptionOwner.PROFILE,
    )
    merged = _merge_column_metadata_strictest([low, high])
    assert merged.description == "profile"
    assert merged.description_owner == DescriptionOwner.PROFILE


@pytest.mark.fast
def test_reconcile_composite_preserves_higher_description_owner() -> None:
    manifest, members, mappings = _payment_union_fixtures()
    composite = compose_composite_graph(members, manifest, mappings)
    composite.tables["payment"].description = "Operator override"
    composite.tables["payment"].description_owner = DescriptionOwner.USER_OVERRIDE
    reconcile_composite_classifications(composite, members, mappings)
    assert composite.tables["payment"].description == "Operator override"
    assert composite.tables["payment"].description_owner == DescriptionOwner.USER_OVERRIDE


@pytest.mark.fast
def test_reconcile_composite_llm_conflict_uses_notes_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, members, mappings = _payment_union_fixtures(
        a_description="Member A ledger",
        b_description="Member B ledger",
    )

    def _fake_classify(_schema: SchemaGraph, notes: str | None) -> dict:
        _ = notes
        return {"payment": (None, "Unified payments", {})}

    monkeypatch.setattr("aetherdialect._federation.llm_classify_schema", _fake_classify)
    composite = compose_composite_graph(
        members,
        manifest,
        mappings,
        notes_content="Domain notes about payments.",
        llm_classify=_fake_classify,
    )
    assert composite.tables["payment"].description == "Unified payments"
    assert composite.tables["payment"].description_owner == DescriptionOwner.NOTES
