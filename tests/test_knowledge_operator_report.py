"""Owner-only operator report combining coverage and claim verification."""

from __future__ import annotations

from aetherdialect._contracts_base import (
    ClaimVerificationOutcome,
    ClaimVerificationResult,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from aetherdialect._knowledge_claims import build_knowledge_operator_report
from aetherdialect._schema_profile import NotesCoverageEntry, NotesExtractionLedger


def test_operator_report_maps_spans_to_verification_outcomes() -> None:
    fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.UNIT_OF_MEASURE.value,
            text="amount is dollars",
            referenced_entities=frozenset({"orders.amount"}),
            payload={"unit": "USD", "summable": True},
        )
    )
    ledger = NotesExtractionLedger(
        entries=(
            NotesCoverageEntry(span="amount is dollars", disposition="fact", record_index=0),
            NotesCoverageEntry(span="no fact here", disposition="no_fact"),
        )
    )
    results = (
        ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence="informational only",
        ),
    )
    report = build_knowledge_operator_report(
        ledger=ledger,
        record_stream=(("structural", fact),),
        verification_results=results,
    )
    assert report["claim_summary"]["unverifiable"] == 1
    assert report["spans"][0]["verification"] == "unverifiable"
    assert report["spans"][1]["verification"] == "no_fact"
