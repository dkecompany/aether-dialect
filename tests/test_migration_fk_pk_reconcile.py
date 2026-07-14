"""Tests for migration FK/PK reconciliation (Phase J)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import SCHEMA_OVERRIDES_VERSION
from aetherdialect._contracts_base import (
    InferenceTag,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._schema_build import overrides_sidecar_path
from aetherdialect._schema_graph import (
    SchemaDiff,
    TableDiff,
    collapse_redundant_inferences,
)
from aetherdialect._schema_overrides import (
    _resync_column_key_flags,
    apply_diff,
    migrate_sidecar_for_diff,
)

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


def _sg_with_inferred_fk() -> tuple[SchemaGraph, SchemaGraph, SchemaDiff]:
    old_orders = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(name="order_id", data_type="integer", is_primary_key=True),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="varchar",
                value_type="string",
                value_overlap_sample=["9", "8", "7", "6", "5"],
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
                inference_tag=InferenceTag.SUFFIX,
            )
        ],
    )
    old_customers = TableMetadata(
        name="customers",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="varchar",
                value_type="string",
                is_primary_key=True,
                value_overlap_sample=["1", "2", "3", "4", "5"],
            )
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    cached = SchemaGraph(
        tables={"orders": old_orders, "customers": old_customers},
        join_paths_multi={},
        created_at="",
    )
    new_orders = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(name="order_id", data_type="integer", is_primary_key=True),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                value_overlap_sample=["9", "8", "7", "6", "5"],
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[],
    )
    new_customers = TableMetadata(
        name="customers",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                value_overlap_sample=["1", "2", "3", "4", "5"],
            )
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    new_sg = SchemaGraph(
        tables={"orders": new_orders, "customers": new_customers},
        join_paths_multi={},
        created_at="",
    )
    diff = SchemaDiff()
    diff.per_table["orders"] = TableDiff(
        retyped_columns=(("customer_id", "varchar", "integer"),),
        value_type_changed_columns=(("customer_id", "string", "integer"),),
    )
    return cached, new_sg, diff


def test_value_type_change_drops_incompatible_inferred_fk() -> None:
    cached, new_sg, diff = _sg_with_inferred_fk()
    dialect = MagicMock()
    dialect.profile_schema = MagicMock()
    out = apply_diff(cached, new_sg, diff, dialect, notes_content=None)
    fk_edges = [e for e in out.tables["orders"].foreign_keys if e.inference_tag is not None]
    assert fk_edges == []


def test_resync_column_key_flags_after_pk_change() -> None:
    tbl = TableMetadata(
        name="t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "ref_id": ColumnMetadata(name="ref_id", data_type="integer", is_foreign_key=True),
        },
        primary_key=["id"],
        foreign_keys=[
            FKEdge(
                src_table="t",
                src_cols=["ref_id"],
                dst_table="u",
                dst_cols=["id"],
                inference_tag=InferenceTag.SUFFIX,
            )
        ],
    )
    other = TableMetadata(
        name="u",
        columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    sg = SchemaGraph(tables={"t": tbl, "u": other}, join_paths_multi={}, created_at="")
    _resync_column_key_flags(sg)
    assert tbl.columns["id"].is_primary_key is True
    assert tbl.columns["ref_id"].is_foreign_key is True


def test_sidecar_list_endpoint_survives_rename(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json.gz"
    sidecar_path = overrides_sidecar_path(schema_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "version": SCHEMA_OVERRIDES_VERSION,
                "tables": {},
                "foreign_keys_add": [
                    {
                        "from": {"table": "old_t", "columns": ["a", "b"]},
                        "to": {"table": "old_u", "columns": ["x", "y"]},
                        "kind": "structural",
                    }
                ],
                "_internal": {"fk_block_inferred": [], "pk_block_inferred": []},
            }
        ),
        encoding="utf-8",
    )
    diff = SchemaDiff(
        table_renames=(("old_t", "new_t"), ("old_u", "new_u")),
        per_table={
            "new_t": TableDiff(renamed_columns=(("a", "a2"), ("b", "b2"))),
            "new_u": TableDiff(renamed_columns=(("x", "x2"), ("y", "y2"))),
        },
    )
    changed = migrate_sidecar_for_diff(schema_path, diff)
    assert changed is True
    doc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    entry = doc["foreign_keys_add"][0]
    assert entry["from"] == {"table": "new_t", "columns": ["a2", "b2"]}
    assert entry["to"] == {"table": "new_u", "columns": ["x2", "y2"]}


def test_catalog_only_rename_stays_remap_despite_inferred_mismatch() -> None:
    old_c = TableMetadata(
        name="customers",
        columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    old_o = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(name="order_id", data_type="integer", is_primary_key=True),
            "customer_id": ColumnMetadata(name="customer_id", data_type="integer"),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
                inference_tag=None,
            ),
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
                inference_tag=InferenceTag.SUFFIX,
            ),
        ],
    )
    old = SchemaGraph(tables={"customers": old_c, "orders": old_o}, join_paths_multi={}, created_at="")
    new_c = TableMetadata(
        name="clients",
        columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    new_o = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(name="order_id", data_type="integer", is_primary_key=True),
            "customer_id": ColumnMetadata(name="customer_id", data_type="integer"),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="clients",
                dst_cols=["id"],
                inference_tag=None,
            ),
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="clients",
                dst_cols=["id"],
                inference_tag=InferenceTag.SUFFIX,
            ),
        ],
    )
    new = SchemaGraph(tables={"clients": new_c, "orders": new_o}, join_paths_multi={}, created_at="")
    from aetherdialect._core_utils import _fk_maps_consistent

    tmap = {"customers": "clients", "orders": "orders"}
    colmap = {
        "customers": {"id": "id"},
        "orders": {"order_id": "order_id", "customer_id": "customer_id"},
    }
    assert _fk_maps_consistent(old, new, tmap, colmap) is True


def test_revalidate_overlap_removes_redundant_inferred_fk() -> None:
    orders = TableMetadata(
        name="orders",
        columns={"customer_id": ColumnMetadata(name="customer_id", data_type="integer")},
        primary_key=[],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
                inference_tag=InferenceTag.SUFFIX,
            ),
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
                inference_tag=None,
            ),
        ],
    )
    customers = TableMetadata(
        name="customers",
        columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    sg = SchemaGraph(
        tables={"orders": orders, "customers": customers},
        join_paths_multi={},
        created_at="",
    )
    removed = collapse_redundant_inferences(sg, [])
    assert removed >= 1
    inferred = [e for e in orders.foreign_keys if e.inference_tag is not None]
    assert inferred == []
