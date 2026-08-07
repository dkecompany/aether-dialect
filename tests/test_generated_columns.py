"""Generated and identity columns must not drive FK inference or preferred PK inference."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import COMPOSE_FIELDS, GROUND_FIELDS, SCHEMA_FIELD_DERIVED
from aetherdialect._contracts_base import PkInferenceTag
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import infer_missing_fks, infer_missing_pks_from_profile


def _pk_col(
    name: str,
    *,
    distinct: int = 100,
    is_identity: bool = False,
    is_generated: bool = False,
    value_overlap_sample: list[str] | None = None,
) -> ColumnMetadata:
    overlap = value_overlap_sample if value_overlap_sample is not None else ["1", "2", "3", "4", "5"]
    return ColumnMetadata(
        name=name,
        data_type="int",
        value_type="integer",
        distinct_count=distinct,
        distinct_ratio=1.0,
        row_count=distinct,
        is_nullable=False,
        null_ratio=0.0,
        is_identity=is_identity,
        is_generated=is_generated,
        value_overlap_sample=overlap,
    )


def _table(name: str, columns: dict[str, ColumnMetadata], **overrides) -> TableMetadata:
    defaults = dict(
        name=name,
        columns=columns,
        primary_key=[],
        foreign_keys=[],
        row_count=100,
    )
    defaults.update(overrides)
    return TableMetadata(**defaults)


@pytest.mark.fast
def test_identity_not_preferred_as_primary_key() -> None:
    """Non-identity candidates win PK inference; identity is a last resort with provenance."""
    auto_id = _pk_col("id", is_identity=True)
    business_code = _pk_col("business_code")
    tbl = _table("items", {"id": auto_id, "business_code": business_code})
    result = infer_missing_pks_from_profile({"items": tbl})
    assert result == [("items", "business_code")]
    assert tbl.primary_key == ["business_code"]
    assert tbl.columns["business_code"].pk_inference_tag == PkInferenceTag.PROFILE
    assert tbl.columns["id"].pk_inference_tag is None
    assert not tbl.columns["id"].is_primary_key

    only_identity = _pk_col("surrogate_id", is_identity=True)
    lone_tbl = _table("orphan", {"surrogate_id": only_identity})
    lone_result = infer_missing_pks_from_profile({"orphan": lone_tbl})
    assert lone_result == [("orphan", "surrogate_id")]
    assert lone_tbl.columns["surrogate_id"].pk_inference_tag == PkInferenceTag.IDENTITY


@pytest.mark.fast
def test_generated_column_excluded_from_fk_inference() -> None:
    """Computed columns with overlapping values must not infer foreign keys."""
    overlap = ["10", "20", "30", "40", "50"]
    tables = {
        "customer": _table(
            "customer",
            {"customer_id": _pk_col("customer_id", value_overlap_sample=overlap)},
            primary_key=["customer_id"],
        ),
        "orders": _table(
            "orders",
            {
                "order_id": _pk_col("order_id"),
                "customer_id": _pk_col("customer_id", is_generated=True, value_overlap_sample=overlap),
            },
            primary_key=["order_id"],
        ),
    }
    inferred = infer_missing_fks(tables)
    assert inferred == []
    assert not tables["orders"].columns["customer_id"].is_foreign_key

    gen_col = ColumnMetadata(
        name="total_amount",
        data_type="int",
        value_type="integer",
        is_generated=True,
        distinct_count=50,
        value_overlap_sample=["1", "2", "3"],
    )
    plain_col = ColumnMetadata(
        name="plain_id",
        data_type="int",
        value_type="integer",
        distinct_count=50,
        value_overlap_sample=["1", "2", "3"],
    )
    table = TableMetadata(
        name="ledger",
        columns={"total_amount": gen_col, "plain_id": plain_col},
        primary_key=[],
        foreign_keys=[],
        row_count=100,
    )
    graph = SchemaGraph(
        tables={"ledger": table},
        join_paths_multi={},
        effective_structural_hash="test",
    )
    ground_payload = json.loads(graph.schema_payload_json(GROUND_FIELDS, owner_master_scope=True))
    compose_payload = json.loads(graph.schema_payload_json(COMPOSE_FIELDS, owner_master_scope=True))
    assert ground_payload["ledger"]["columns"]["total_amount"][SCHEMA_FIELD_DERIVED] is True
    assert "derived" not in ground_payload["ledger"]["columns"]["plain_id"]
    assert compose_payload["ledger"]["columns"]["total_amount"]["derived"] is True
