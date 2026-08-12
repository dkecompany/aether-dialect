"""Intent interpret/ground prompts pass schema and domain context as structured JSON."""

from __future__ import annotations

import json

import pytest

from aetherdialect._contracts_base import DomainKnowledgeEntry, FailureCategory
from aetherdialect._contracts_core import InterpretPlan
from aetherdialect._contracts_schema import IntentIssue
from aetherdialect._intent_loop import (
    build_intent_ground_prompt,
    build_intent_interpret_prompt,
    build_intent_parse_prompt,
    build_intent_semantic_repair_prompt,
)
from aetherdialect._utils import domain_knowledge_digest, domain_knowledge_scope


@pytest.mark.fast
def test_schema_and_dk_are_structured() -> None:
    domain = {"tables": {"orders": {"columns": {"id": {"type": "integer"}}}}}
    ground = {"orders": {"columns": {"id": {"type": "integer", "role": "identifier"}}}}
    entries = (
        DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="glossary"),
        DomainKnowledgeEntry(key="fy", text="Fiscal year starts in July.", kind="policy"),
    )
    digest = domain_knowledge_digest(entries)
    with domain_knowledge_scope(entries, digest):
        interpret = json.loads(
            build_intent_interpret_prompt(
                "what is arr",
                json.dumps(domain),
                "",
                (),
            )
        )
        ground_prompt = json.loads(
            build_intent_ground_prompt(
                "what is arr",
                InterpretPlan(approach="lookup glossary", tables=("orders",)),
                json.dumps(ground),
                "",
                "[]",
                (),
                (),
            )
        )
        _system, parse_user = build_intent_parse_prompt("what is arr", json.dumps(ground), ["orders"])
        parse_payload = json.loads(parse_user)
        repair = json.loads(
            build_intent_semantic_repair_prompt(
                "what is arr",
                json.dumps({"tables": ["orders"]}),
                [
                    IntentIssue(
                        issue_id="t1",
                        category=FailureCategory.UNKNOWN_COLUMN,
                        message="bad",
                        severity="error",
                    )
                ],
                [],
                json.dumps(ground),
            )
        )
    assert isinstance(interpret["schema_domain"], dict)
    assert isinstance(interpret["domain_context"], list)
    assert all(isinstance(row, dict) for row in interpret["domain_context"])
    assert interpret["domain_context"][0]["key"] == "arr"
    assert isinstance(ground_prompt["schema_literal_json"], dict)
    assert isinstance(ground_prompt["domain_context"], list)
    assert isinstance(parse_payload["schema_summary"], dict)
    assert isinstance(parse_payload["domain_context"], list)
    assert isinstance(repair["schema_info"], dict)
    assert isinstance(repair["current_intent"], dict)
    assert isinstance(repair["domain_context"], list)
