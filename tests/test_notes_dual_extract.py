"""Dual notes extraction: glossary lines become BK; table-anchored lines stay out."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import BusinessKnowledgeEntry
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_catalog import (
    extract_business_knowledge_from_notes,
    filter_schema_anchored_business_knowledge,
)


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "payment": TableMetadata(
                name="payment",
                columns={
                    "amount": ColumnMetadata(name="amount", data_type="numeric"),
                    "payment_date": ColumnMetadata(name="payment_date", data_type="date"),
                },
                primary_key=["amount"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g",
    )


@pytest.mark.fast
def test_glossary_lines_become_bk() -> None:
    schema = _schema()
    notes = "ARR means annual recurring revenue.\npayment.amount is the charged total."
    llm_payload = [
        {"key": "arr", "kind": "glossary", "text": "ARR means annual recurring revenue."},
        {"key": "payment_amount", "kind": "glossary", "text": "payment.amount is the charged total."},
    ]
    with patch("aetherdialect._schema_catalog.LLMProvider.chat", return_value=__import__("json").dumps(llm_payload)):
        with patch("aetherdialect._schema_catalog.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_business_knowledge_from_notes(notes, schema)
    keys = {e.key for e in entries}
    assert "arr" in keys
    assert "payment_amount" not in keys
    assert all(isinstance(e, BusinessKnowledgeEntry) for e in entries)


@pytest.mark.fast
def test_table_anchored_lines_not_in_bk() -> None:
    schema = _schema()
    candidates = (
        BusinessKnowledgeEntry(key="pay_tbl", text="The payment table holds receipts."),
        BusinessKnowledgeEntry(key="amt", text="Use payment.amount for totals."),
        BusinessKnowledgeEntry(key="fy", text="Fiscal year starts in July.", kind="policy"),
    )
    kept = filter_schema_anchored_business_knowledge(candidates, schema)
    assert [e.key for e in kept] == ["fy"]


@pytest.mark.fast
def test_no_notes_bk_empty() -> None:
    schema = _schema()
    assert extract_business_knowledge_from_notes(None, schema) == ()
    assert extract_business_knowledge_from_notes("", schema) == ()
    assert extract_business_knowledge_from_notes("   ", schema) == ()
