"""Extraction proposal artifact persistence, diff, and FK materialization."""

from __future__ import annotations

import json
from pathlib import Path

from aetherdialect._contracts_base import (
    DomainKnowledgeKind,
    KnowledgeExtractionProposal,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._knowledge_staleness import (
    diff_knowledge_extraction_proposals,
    foreign_key_proposals_from_structural_facts,
    knowledge_extraction_proposal_path,
    knowledge_scope_fingerprint,
    load_knowledge_extraction_proposal,
    materialize_fk_proposals_to_overrides,
    save_knowledge_extraction_proposal,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _schema() -> SchemaGraph:
    tables = {
        "orders": TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer"),
                "status": ColumnMetadata(name="status", data_type="text"),
                "customer_id": ColumnMetadata(name="customer_id", data_type="integer"),
            },
            primary_key=["id"],
            foreign_keys=[],
        ),
        "customers": TableMetadata(
            name="customers",
            columns={"id": ColumnMetadata(name="id", data_type="integer")},
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="g_prop",
        effective_structural_hash="eff",
        notes_sha256="notes_hash",
    )


def test_foreign_key_proposals_from_join_facts() -> None:
    fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="orders link customers",
            referenced_entities=frozenset({"orders.customer_id", "customers.id"}),
        )
    )
    proposals = foreign_key_proposals_from_structural_facts((fact,))
    assert proposals == (
        {
            "from": "customers.id",
            "to": "orders.customer_id",
            "kind": "logical",
            "provenance": "notes_structural",
        },
    )


def test_proposal_round_trip_and_diff(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import DomainKnowledgeEntry

    schema = _schema()
    scope_fp = knowledge_scope_fingerprint(schema)
    prior = KnowledgeExtractionProposal(
        domain_knowledge=(
            DomainKnowledgeEntry.normalize(
                DomainKnowledgeEntry(key="arr", text="old arr", kind=DomainKnowledgeKind.GLOSSARY.value)
            ),
        ),
        structural_knowledge=(),
        foreign_keys_add=(),
        coverage={"entries": []},
        notes_sha256="notes_hash",
        scope_fingerprint=scope_fp,
    )
    save_knowledge_extraction_proposal(tmp_path, prior)
    new = KnowledgeExtractionProposal(
        domain_knowledge=(
            DomainKnowledgeEntry.normalize(
                DomainKnowledgeEntry(key="arr", text="new arr", kind=DomainKnowledgeKind.GLOSSARY.value)
            ),
        ),
        structural_knowledge=(),
        foreign_keys_add=({"from": "customers.id", "to": "orders.customer_id", "kind": "logical"},),
        coverage={"entries": [{"span": "ARR means revenue.", "disposition": "fact", "record_index": 0}]},
        notes_sha256="notes_hash",
        scope_fingerprint=scope_fp,
    )
    diff = diff_knowledge_extraction_proposals(prior, new)
    assert diff["status"] == "changed"
    assert diff["domain_knowledge_changed"]
    assert diff["foreign_keys_add_added"]
    loaded = load_knowledge_extraction_proposal(tmp_path, schema, require_stamp_match=True)
    assert loaded is not None
    assert loaded.domain_knowledge[0].text == "old arr"
    assert knowledge_extraction_proposal_path(tmp_path).is_file()


def test_materialize_fk_proposals_to_overrides(tmp_path: Path) -> None:
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"foreign_keys_add": []}), encoding="utf-8")
    proposals = ({"from": "customers.id", "to": "orders.customer_id", "kind": "logical"},)
    changed = materialize_fk_proposals_to_overrides(overrides_path, proposals)
    assert changed is True
    doc = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert doc["foreign_keys_add"][0]["from"] == "customers.id"
    assert materialize_fk_proposals_to_overrides(overrides_path, proposals) is False
