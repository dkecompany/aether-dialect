"""Single-pass notes extraction: domain and structural knowledge from one record stream."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._constants_runtime import (
    DOMAIN_KNOWLEDGE_REFINER_SYSTEM,
    KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM,
    KNOWLEDGE_NOTES_EXTRACT_SYSTEM,
)
from aetherdialect._contracts_base import DomainKnowledgeEntry
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_profile import (
    assert_notes_coverage_total,
    extract_domain_knowledge_from_notes,
    extract_knowledge_from_notes,
    filter_schema_anchored_domain_knowledge,
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


def _unified_payload(notes: str, records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"records": [], "coverage": [{"span": notes, "disposition": "no_fact"}]}
    coverage: list[dict[str, object]] = []
    pos = 0
    for index, record in enumerate(records):
        text = str(record["text"])
        start = notes.find(text, pos)
        if start < 0:
            raise ValueError(f"record text not found in notes: {text!r}")
        if start > pos:
            coverage.append({"span": notes[pos:start], "disposition": "no_fact"})
        coverage.append({"span": text, "disposition": "fact", "record_index": index})
        pos = start + len(text)
    if pos < len(notes):
        coverage.append({"span": notes[pos:], "disposition": "no_fact"})
    return {"records": records, "coverage": coverage}


@pytest.mark.fast
def test_glossary_and_field_facts_kept_from_notes_extract() -> None:
    schema = _schema()
    notes = "ARR means annual recurring revenue.\npayment.amount is the charged total."
    records = [
        {
            "key": "arr",
            "kind": "glossary",
            "text": "ARR means annual recurring revenue.",
            "referenced_entities": [],
        },
        {
            "kind": "field",
            "text": "payment.amount is the charged total.",
            "referenced_entities": ["payment.amount"],
        },
    ]
    llm_payload = _unified_payload(notes, records)
    with patch("aetherdialect._schema_profile.LLMProvider.chat", return_value=json.dumps(llm_payload)) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            result = extract_knowledge_from_notes(notes, schema)
    dk_keys = {e.key for e in result.domain_knowledge}
    assert dk_keys == {"arr"}
    assert len(result.structural_knowledge) == 1
    assert result.structural_knowledge[0].kind == "field"
    assert chat.call_count == 1
    assert chat.call_args.args[0] == KNOWLEDGE_NOTES_EXTRACT_SYSTEM


@pytest.mark.fast
def test_filter_keeps_operator_facts_including_qualified_names() -> None:
    schema = _schema()
    candidates = (
        DomainKnowledgeEntry(
            key="arr",
            text="ARR means annualized rental revenue from payment.amount.",
            kind="glossary",
        ),
        DomainKnowledgeEntry(
            key="pay_only",
            text="payment.amount is the charged total.",
            kind="glossary",
        ),
        DomainKnowledgeEntry(key="fy", text="Fiscal year starts in July.", kind="policy"),
    )
    kept = filter_schema_anchored_domain_knowledge(candidates, schema)
    assert [e.key for e in kept] == ["arr", "pay_only", "fy"]


@pytest.mark.fast
def test_filter_does_not_scan_inventory_phrasing() -> None:
    """Inventory omission is prompt guidance only; no free-text marker drop."""
    schema = _schema()
    candidates = (
        DomainKnowledgeEntry(key="inv", text="The payment table has columns amount and payment_date."),
        DomainKnowledgeEntry(key="ddl", text="See DDL for the payment primary key."),
        DomainKnowledgeEntry(key="fy", text="Fiscal year starts in July.", kind="policy"),
    )
    kept = filter_schema_anchored_domain_knowledge(candidates, schema)
    assert [e.key for e in kept] == ["inv", "ddl", "fy"]


@pytest.mark.fast
def test_extract_payload_includes_schema_names() -> None:
    schema = _schema()
    notes = "Fiscal year starts in July."
    records = [
        {
            "key": "fy",
            "kind": "policy",
            "text": "Fiscal year starts in July.",
            "referenced_entities": [],
        }
    ]
    with patch(
        "aetherdialect._schema_profile.LLMProvider.chat",
        return_value=json.dumps(_unified_payload(notes, records)),
    ) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    assert [e.key for e in entries] == ["fy"]
    assert chat.call_args.args[0] == KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    user_payload = json.loads(chat.call_args.args[1])
    assert user_payload["domain_notes"] == notes
    assert "payment" in user_payload["schema_names"]
    assert "hidden_columns" not in user_payload
    assert "hidden_columns" not in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert "hidden_columns" not in KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM


@pytest.mark.fast
def test_no_notes_domain_knowledge_empty() -> None:
    schema = _schema()
    assert extract_domain_knowledge_from_notes(None, schema) == ()
    assert extract_domain_knowledge_from_notes("", schema) == ()
    assert extract_domain_knowledge_from_notes("   ", schema) == ()


@pytest.mark.fast
def test_domain_knowledge_accepts_records_object_on_first_try() -> None:
    schema = _schema()
    notes = "Fiscal year starts in July."
    records = [
        {
            "key": "fy",
            "kind": "policy",
            "text": "Fiscal year starts in July.",
            "referenced_entities": [],
        }
    ]
    with patch(
        "aetherdialect._schema_profile.LLMProvider.chat",
        return_value=json.dumps(_unified_payload(notes, records)),
    ) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    assert [e.key for e in entries] == ["fy"]
    assert chat.call_count == 1


@pytest.mark.fast
def test_domain_knowledge_repair_after_wrong_object_shape_then_success() -> None:
    schema = _schema()
    notes = "Fiscal year starts in July."
    records = [
        {
            "key": "fy",
            "kind": "policy",
            "text": "Fiscal year starts in July.",
            "referenced_entities": [],
        }
    ]
    responses = [
        json.dumps({"items": records}),
        json.dumps(_unified_payload(notes, records)),
    ]

    def _chat(system: str, user: str, **_kwargs: object) -> str:
        return responses.pop(0)

    with patch("aetherdialect._schema_profile.LLMProvider.chat", side_effect=_chat) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    assert [e.key for e in entries] == ["fy"]
    assert chat.call_count == 2
    assert chat.call_args_list[0].args[0] == KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert chat.call_args_list[1].args[0] == KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM
    repair_user = json.loads(chat.call_args_list[1].args[1])
    assert '"records"' in repair_user["validation_error"]
    assert repair_user["domain_notes"] == notes


@pytest.mark.fast
def test_domain_knowledge_empty_records_is_success_with_total_coverage() -> None:
    schema = _schema()
    notes = "Only structural inventory; no glossary or policy."
    with patch(
        "aetherdialect._schema_profile.LLMProvider.chat",
        return_value=json.dumps(_unified_payload(notes, [])),
    ) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    assert entries == ()
    assert chat.call_count == 1


@pytest.mark.fast
def test_domain_knowledge_repair_after_hidden_token_wipe() -> None:
    from aetherdialect._contracts_base import SensitivityClassification

    schema = _schema()
    schema.tables["payment"].columns["amount"].sensitivity = SensitivityClassification.HIDDEN
    notes = "payment.amount is hidden and off limits."
    hidden_only = [
        {
            "kind": "field",
            "text": notes,
            "referenced_entities": ["payment.amount"],
        }
    ]
    responses = [
        json.dumps(_unified_payload(notes, hidden_only)),
        json.dumps(_unified_payload(notes, [])),
    ]

    def _chat(system: str, user: str, **_kwargs: object) -> str:
        return responses.pop(0)

    with patch("aetherdialect._schema_profile.LLMProvider.chat", side_effect=_chat) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    assert entries == ()
    assert chat.call_count == 2
    repair_user = json.loads(chat.call_args_list[1].args[1])
    assert "hidden_columns" not in repair_user
    assert "security filter" in repair_user["validation_error"]


@pytest.mark.fast
def test_prompts_instruct_single_pass_and_coverage() -> None:
    assert "single pass" in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert '"coverage"' in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert "schema_names" in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert "Preserve operator wording" in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert "independent of database" not in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert "independent of database" not in KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM
    assert "independent of database" not in DOMAIN_KNOWLEDGE_REFINER_SYSTEM


@pytest.mark.fast
def test_domain_knowledge_empty_only_when_llm_returns_empty() -> None:
    schema = _schema()
    notes = "Some operator notes that might contain policies."

    with patch(
        "aetherdialect._schema_profile.LLMProvider.chat",
        return_value=json.dumps(_unified_payload(notes, [])),
    ) as chat:
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    assert entries == ()
    assert chat.call_count == 1
    assert chat.call_args_list[0].args[0] == KNOWLEDGE_NOTES_EXTRACT_SYSTEM


@pytest.mark.fast
def test_extract_parses_referenced_entities_field() -> None:
    schema = _schema()
    notes = "payment.amount is the charged total."
    records = [
        {
            "key": "charged_total",
            "kind": "glossary",
            "text": "The charged total is the amount billed.",
            "referenced_entities": [],
        },
        {"key": "fy", "kind": "policy", "text": "Fiscal year starts in July.", "referenced_entities": []},
    ]
    notes = str(records[0]["text"]) + "\n" + str(records[1]["text"])
    with patch(
        "aetherdialect._schema_profile.LLMProvider.chat",
        return_value=json.dumps(_unified_payload(notes, records)),
    ):
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            entries = extract_domain_knowledge_from_notes(notes, schema)
    by_key = {e.key: e for e in entries}
    assert by_key["charged_total"].referenced_entities == frozenset()
    assert by_key["fy"].referenced_entities == frozenset()


@pytest.mark.fast
def test_extract_prompts_avoid_banned_neutrality_words() -> None:
    banned = ("hidden", "withheld", "aetherspace", "federation")
    for constant in (KNOWLEDGE_NOTES_EXTRACT_SYSTEM, KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM):
        lowered = constant.lower()
        for word in banned:
            assert word not in lowered


@pytest.mark.fast
def test_prompt_kind_vocab_matches_parser() -> None:
    for kind in ("glossary", "policy", "metric", "synonym", "caveat"):
        assert kind in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    for kind in ("relation", "field", "join", "grain", "cardinality", "lifecycle"):
        assert kind in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert '"records"' in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert '"key"' in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert '"kind"' in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert '"text"' in KNOWLEDGE_NOTES_EXTRACT_SYSTEM
    assert "business" not in KNOWLEDGE_NOTES_EXTRACT_SYSTEM.lower()
    assert "do not rewrite" not in KNOWLEDGE_NOTES_EXTRACT_SYSTEM.lower()


@pytest.mark.fast
def test_single_pass_routes_anchored_records_to_structural() -> None:
    schema = _schema()
    notes = "payment is the payment header."
    records = [
        {
            "kind": "relation",
            "text": "payment is the payment header.",
            "referenced_entities": ["payment"],
        }
    ]
    with patch(
        "aetherdialect._schema_profile.LLMProvider.chat",
        return_value=json.dumps(_unified_payload(notes, records)),
    ):
        with patch("aetherdialect._schema_profile.EngineConfig.llm_credentials_configured", return_value=True):
            result = extract_knowledge_from_notes(notes, schema)
    assert result.domain_knowledge == ()
    assert len(result.structural_knowledge) == 1
    assert result.structural_knowledge[0].kind == "relation"
    assert_notes_coverage_total(notes, result.ledger)
