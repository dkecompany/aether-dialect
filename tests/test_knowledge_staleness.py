"""Derived knowledge staleness keys and reference resolution on load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION
from aetherdialect._contracts_base import DomainKnowledgeEntry, StructuralKnowledgeFact, StructuralKnowledgeKind
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._knowledge_staleness import (
    filter_domain_knowledge_by_resolvable_references,
    filter_structural_knowledge_by_resolvable_references,
    knowledge_artifact_save_stamps,
    knowledge_scope_fingerprint,
    load_knowledge_extraction_proposal,
    load_structural_knowledge_artifact,
    save_knowledge_extraction_proposal,
)
from aetherdialect._utils import load_domain_knowledge_artifact
from aetherdialect._utils_artifacts import save_domain_knowledge_artifact, save_structural_knowledge_artifact


def _schema(*tables: str) -> SchemaGraph:
    out: dict[str, TableMetadata] = {}
    for name in tables:
        out[name] = TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")},
            primary_key=["id"],
            foreign_keys=[],
        )
    return SchemaGraph(
        tables=out,
        join_paths_multi={},
        effective_structural_hash="hash-a",
        profiling_hash="prof-a",
        schema_graph_id="g",
        notes_sha256="notes-a",
    )


@pytest.mark.fast
def test_scope_fingerprint_changes_when_profiling_hash_changes() -> None:
    schema = _schema("store")
    first = knowledge_scope_fingerprint(schema)
    schema.profiling_hash = "prof-b"
    second = knowledge_scope_fingerprint(schema)
    assert first != second


@pytest.mark.fast
def test_domain_knowledge_load_drops_unresolvable_references(tmp_path: Path) -> None:
    schema = _schema("shop")
    entries = (
        DomainKnowledgeEntry(key="shop", kind="glossary", text="a shop", referenced_entities=frozenset({"shop"})),
        DomainKnowledgeEntry(
            key="gone", kind="glossary", text="missing table", referenced_entities=frozenset({"payroll"})
        ),
    )
    save_domain_knowledge_artifact(tmp_path, entries, **knowledge_artifact_save_stamps(schema))
    loaded = load_domain_knowledge_artifact(tmp_path, schema)
    assert loaded is not None
    assert [e.key for e in loaded] == ["shop"]


@pytest.mark.fast
def test_domain_knowledge_artifact_misses_when_scope_fingerprint_changes(tmp_path: Path) -> None:
    schema = _schema("shop")
    save_domain_knowledge_artifact(
        tmp_path,
        (DomainKnowledgeEntry(key="shop", kind="glossary", text="a shop", referenced_entities=frozenset({"shop"})),),
        **knowledge_artifact_save_stamps(schema),
    )
    schema.profiling_hash = "changed"
    assert load_domain_knowledge_artifact(tmp_path, schema) is None


@pytest.mark.fast
def test_structural_knowledge_artifact_round_trip(tmp_path: Path) -> None:
    schema = _schema("orders")
    facts = (
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.RELATION.value,
            text="orders header",
            referenced_entities=frozenset({"orders"}),
        ),
    )
    from aetherdialect._utils_artifacts import save_structural_knowledge_artifact

    save_structural_knowledge_artifact(tmp_path, facts, **knowledge_artifact_save_stamps(schema))
    loaded = load_structural_knowledge_artifact(tmp_path, schema)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].text == "orders header"


@pytest.mark.fast
def test_structural_knowledge_artifact_rejects_wrong_format_version(tmp_path: Path) -> None:
    schema = _schema("orders")
    facts = (
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.RELATION.value,
            text="orders header",
            referenced_entities=frozenset({"orders"}),
        ),
    )
    save_structural_knowledge_artifact(tmp_path, facts, **knowledge_artifact_save_stamps(schema))
    artifact_path = tmp_path / "structural_knowledge.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["format_version"] = "0.0.0"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_structural_knowledge_artifact(tmp_path, schema) is None


@pytest.mark.fast
def test_structural_knowledge_artifact_rejects_missing_format_version(tmp_path: Path) -> None:
    schema = _schema("orders")
    facts = (
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.RELATION.value,
            text="orders header",
            referenced_entities=frozenset({"orders"}),
        ),
    )
    save_structural_knowledge_artifact(tmp_path, facts, **knowledge_artifact_save_stamps(schema))
    artifact_path = tmp_path / "structural_knowledge.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    del payload["format_version"]
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_structural_knowledge_artifact(tmp_path, schema) is None


@pytest.mark.fast
def test_extraction_proposal_rejects_wrong_format_version(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import KnowledgeExtractionProposal

    schema = _schema("orders")
    proposal = KnowledgeExtractionProposal(
        domain_knowledge=(),
        structural_knowledge=(),
        foreign_keys_add=(),
        coverage={"entries": []},
        notes_sha256="notes-a",
        scope_fingerprint=knowledge_scope_fingerprint(schema),
    )
    save_knowledge_extraction_proposal(tmp_path, proposal)
    path = tmp_path / "knowledge_extraction_proposal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION
    payload["format_version"] = "0.0.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_knowledge_extraction_proposal(tmp_path, schema, require_stamp_match=True) is None


@pytest.mark.fast
def test_filter_structural_drops_unresolvable_reference() -> None:
    schema = _schema("orders")
    facts = (
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.FIELD.value,
            text="missing column",
            referenced_entities=frozenset({"orders.missing_col"}),
        ),
    )
    kept, dropped = filter_structural_knowledge_by_resolvable_references(facts, schema)
    assert kept == ()
    assert dropped == 1


@pytest.mark.fast
def test_filter_domain_keeps_empty_reference_set() -> None:
    schema = _schema("orders")
    entries = (DomainKnowledgeEntry(key="arr", kind="metric", text="ARR", referenced_entities=frozenset()),)
    kept, dropped = filter_domain_knowledge_by_resolvable_references(entries, schema)
    assert len(kept) == 1
    assert dropped == 0
