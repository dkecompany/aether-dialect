"""NOTES_STRUCTURAL FK admission on catalog-FK-free schemas. Exercises ``admit_join_fk_proposals`` without warehouse credentials."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import StructuralKnowledgeFact, StructuralKnowledgeKind
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, InferenceTag, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import (
    FkAdmissionClassification,
    admit_join_fk_proposals,
    classify_fk_admission_effect,
    recompute_join_paths_multi,
)


def _catalog_fk_free_schema() -> SchemaGraph:
    tables = {
        "orders": TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    value_overlap_sample=["1", "2", "3", "4", "5"],
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
        ),
        "customers": TableMetadata(
            name="customers",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    is_primary_key=True,
                    value_overlap_sample=["1", "2", "3", "4", "5"],
                )
            },
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="notes_structural_gate",
    )


@pytest.mark.not_fast
def test_notes_structural_fk_admission_without_catalog_fks() -> None:
    """Structural join facts admit logical FKs when the catalog supplies none."""
    schema = _catalog_fk_free_schema()
    assert not any(fk.inference_tag is None for tbl in schema.tables.values() for fk in tbl.foreign_keys)

    join_fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="orders reference customers",
            referenced_entities=frozenset({"orders.customer_id", "customers.id"}),
        )
    )
    proposed = FKEdge(
        src_table="orders",
        src_cols=["customer_id"],
        dst_table="customers",
        dst_cols=["id"],
        inference_tag=InferenceTag.NOTES_STRUCTURAL,
    )
    assert classify_fk_admission_effect(schema, proposed) == FkAdmissionClassification.ADDS_REACHABILITY

    fk_add, fk_remove, report = admit_join_fk_proposals(schema, (join_fact,))
    assert fk_add
    assert fk_add[0]["provenance"] == "notes_structural"
    admitted = [entry for entry in report if entry.admitted and not entry.blocked_negative]
    assert any(entry.classification == FkAdmissionClassification.ADDS_REACHABILITY for entry in admitted)
