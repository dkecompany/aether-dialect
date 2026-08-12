"""Join FK admission, declared-path pinning, and fan-out metadata from structural knowledge."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    FkAdmissionClassification,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, InferenceTag, SchemaGraph, TableMetadata
from aetherdialect._knowledge_join import (
    attach_structural_fanout_metadata,
    merge_preserve_tables_with_notes_defaults,
    notes_declares_one_to_many_edge,
)
from aetherdialect._schema_graph import (
    admit_join_fk_proposals,
    classify_fk_admission_effect,
    recompute_join_paths_multi,
)
from aetherdialect._sql_gen import join_hints_multi, pin_join_paths_multi, validate_declared_join_paths_or_raise
from aetherdialect._validation_shape import multiplying_edges_for_table


def _orders_customers_schema() -> SchemaGraph:
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
    sg = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="eff",
    )
    return sg


@pytest.mark.fast
def test_fk_admission_classifies_adds_reachability() -> None:
    schema = _orders_customers_schema()
    edge = FKEdge(
        src_table="orders",
        src_cols=["customer_id"],
        dst_table="customers",
        dst_cols=["id"],
        inference_tag=InferenceTag.NOTES_STRUCTURAL,
    )
    assert classify_fk_admission_effect(schema, edge) == FkAdmissionClassification.ADDS_REACHABILITY


@pytest.mark.fast
def test_admit_join_fk_proposals_structural_and_negative() -> None:
    schema = _orders_customers_schema()
    join_fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="orders reference customers",
            referenced_entities=frozenset({"orders.customer_id", "customers.id"}),
        )
    )
    negative = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="no link",
            referenced_entities=frozenset({"orders.id", "customers.id"}),
            payload={"negative": True, "from": "customers.id", "to": "orders.id"},
        )
    )
    fk_add, fk_remove, report = admit_join_fk_proposals(schema, (join_fact, negative))
    assert fk_add
    assert fk_add[0]["provenance"] == "notes_structural"
    assert fk_remove
    admitted = [r for r in report if r.admitted and not r.blocked_negative]
    assert any(r.classification == FkAdmissionClassification.ADDS_REACHABILITY for r in admitted)


@pytest.mark.fast
def test_declared_path_pinning_collapses_candidates() -> None:
    tables = {
        "a": TableMetadata(
            name="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
                "b_id": ColumnMetadata(name="b_id", data_type="integer"),
            },
            primary_key=["id"],
            foreign_keys=[],
        ),
        "b": TableMetadata(
            name="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
                "c_id": ColumnMetadata(name="c_id", data_type="integer"),
            },
            primary_key=["id"],
            foreign_keys=[
                FKEdge(
                    src_table="b",
                    src_cols=["c_id"],
                    dst_table="c",
                    dst_cols=["id"],
                )
            ],
        ),
        "c": TableMetadata(
            name="c",
            columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    tables["a"].foreign_keys = [
        FKEdge(src_table="a", src_cols=["b_id"], dst_table="b", dst_cols=["id"]),
    ]
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="pin",
    )
    declared_sig = ["a.b_id->b.id", "b.c_id->c.id"]
    fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="use b bridge",
            referenced_entities=frozenset({"a.b_id", "b.id", "b.c_id", "c.id"}),
            payload={"path_signature": declared_sig},
        )
    )
    schema = SchemaGraph(
        tables=schema.tables,
        join_paths_multi=dict(schema.join_paths_multi),
        effective_structural_hash="pin",
        structural_knowledge=(fact,),
    )
    validate_declared_join_paths_or_raise(schema, (fact,))
    pin_join_paths_multi(schema, (fact,))
    hints = join_hints_multi(schema, ["a", "c"], None, virtual_specs={}, include_semantic=False)
    non_j00 = [c for c in hints["candidates"] if c.get("candidate_id") != "J00"]
    assert len(non_j00) == 1
    assert non_j00[0]["join_path_signature"] == declared_sig


@pytest.mark.fast
def test_unmatched_declared_path_raises() -> None:
    schema = _orders_customers_schema()
    fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="bad path",
            referenced_entities=frozenset({"orders.customer_id", "customers.id"}),
            payload={"path_signature": ["orders.customer_id->customers.id", "customers.id->orders.id"]},
        )
    )
    with pytest.raises(ConfigError, match="not found in enumerated candidates"):
        validate_declared_join_paths_or_raise(schema, (fact,))


@pytest.mark.fast
def test_fanout_metadata_ratchet_more_conservative() -> None:
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True, is_unique=True)},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
                "parent_id": ColumnMetadata(
                    name="parent_id",
                    data_type="integer",
                    is_unique=True,
                    distinct_count=10,
                    row_count=10,
                ),
            },
            primary_key=["id"],
            foreign_keys=[
                FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"]),
            ],
        ),
    }
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="fanout",
    )
    cardinality = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.CARDINALITY.value,
            text="child many per parent",
            referenced_entities=frozenset({"parent.id", "child.parent_id"}),
        )
    )
    attach_structural_fanout_metadata(schema, (cardinality,))
    sig = ["child.parent_id->parent.id"]
    assert notes_declares_one_to_many_edge(schema, "child", ["parent_id"], "parent", ["id"])
    schema2 = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="fanout2",
    )
    hits_before_notes = multiplying_edges_for_table(sig, "parent", schema2, from_anchor="child")
    attach_structural_fanout_metadata(schema2, (cardinality,))
    hits_after_notes = multiplying_edges_for_table(sig, "parent", schema2, from_anchor="child")
    assert not hits_before_notes
    assert hits_after_notes


@pytest.mark.fast
def test_merge_preserve_tables_includes_notes_parent_defaults() -> None:
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
                "parent_id": ColumnMetadata(name="parent_id", data_type="integer"),
            },
            primary_key=["id"],
            foreign_keys=[
                FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"]),
            ],
        ),
    }
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="preserve",
    )
    cardinality = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.CARDINALITY.value,
            text="child many per parent",
            referenced_entities=frozenset({"parent.id", "child.parent_id"}),
        )
    )
    attach_structural_fanout_metadata(schema, (cardinality,))
    merged = merge_preserve_tables_with_notes_defaults([], schema, query_tables=["parent", "child"])
    assert "parent" in merged
    assert merge_preserve_tables_with_notes_defaults([], schema, query_tables=["child"]) == []
