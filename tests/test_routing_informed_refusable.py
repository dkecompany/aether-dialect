"""Routing inventory, insufficient-knowledge meta refusals, and INVALID/RESTRICTED splits."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
    DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
    DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
)
from aetherdialect._constants_runtime import (
    META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND,
    REFUSAL_CATALOGUE,
)
from aetherdialect._contracts_core import QuestionRoute
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import refusal_diagnostic_code_for_outcome, refusal_user_text_for_code
from aetherdialect._utils_intent import build_question_validation_inventory, validate_question


@pytest.mark.fast
def test_build_question_validation_inventory_sorted_unique() -> None:
    inv = build_question_validation_inventory(
        table_names=("payment", "customer", "payment"),
        domain_knowledge_keys=("arr", "arr", "mrr"),
    )
    assert inv == {"table_names": ["customer", "payment"], "domain_knowledge_keys": ["arr", "mrr"]}


@pytest.mark.fast
def test_validate_question_user_payload_includes_inventory() -> None:
    with patch("aetherdialect._utils_intent.LLMProvider.json") as mock_json:
        mock_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "analytical",
            "corrected": "count orders",
        }
        validate_question(
            "count orders",
            table_names=frozenset({"orders", "customer"}),
            domain_knowledge_keys=("arr",),
        )
    payload = json.loads(mock_json.call_args[0][1])
    assert payload["question"] == "count orders"
    assert payload["inventory"]["table_names"] == ["customer", "orders"]
    assert payload["inventory"]["domain_knowledge_keys"] == ["arr"]


@pytest.mark.fast
def test_conversational_invalid_kind() -> None:
    with patch(
        "aetherdialect._utils_intent.LLMProvider.json",
        return_value={
            "valid_database_question": "no",
            "query_type": "conversational",
            "corrected": "hello",
        },
    ):
        result = validate_question("hello")
    assert result.accepted is False
    assert result.route is QuestionRoute.INVALID
    assert result.invalid_kind == "conversational"


@pytest.mark.fast
def test_unmappable_invalid_kind() -> None:
    with patch(
        "aetherdialect._utils_intent.LLMProvider.json",
        return_value={
            "valid_database_question": "no",
            "query_type": "unmappable",
            "corrected": "how do I write a join",
        },
    ):
        result = validate_question("how do I write a join")
    assert result.accepted is False
    assert result.route is QuestionRoute.INVALID
    assert result.invalid_kind == "unmappable"


@pytest.mark.fast
def test_conversational_and_unmappable_refusal_text_distinct() -> None:
    conv = refusal_user_text_for_code(DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY).lower()
    unmappable = refusal_user_text_for_code(DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION).lower()
    assert "pin" not in conv
    assert conv != unmappable
    assert "pin" in unmappable or "table" in unmappable or "column" in unmappable


@pytest.mark.fast
def test_restricted_refusal_code_and_text() -> None:
    assert refusal_diagnostic_code_for_outcome("restricted") == DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED
    text = refusal_user_text_for_code(DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED).lower()
    assert "operation" in text or "not supported" in text
    assert "rephrase" not in text
    assert "schema" not in text
    assert "table" not in text
    assert REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED]["reformulation_hint"] == ""


@pytest.mark.fast
def test_meta_insufficient_knowledge_step() -> None:
    step = MainExecutionOps._meta_insufficient_knowledge_step()
    assert step.error is not None and step.error.code.value == "insufficient_knowledge"
    assert step.error.detail_code == DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE


@pytest.mark.fast
def test_validate_meta_schema_answer_accepts_insufficient_knowledge() -> None:
    MainExecutionOps.validate_meta_schema_answer(
        {"response_kind": META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND},
        {"inventory": {}, "tables": [], "members": [], "relationships": []},
    )


@pytest.mark.fast
def test_invalid_question_outcome_maps_to_unmappable_code() -> None:
    assert refusal_diagnostic_code_for_outcome("invalid_question") == DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION


@pytest.mark.fast
def test_conversational_deny_outcome_code() -> None:
    assert refusal_diagnostic_code_for_outcome("conversational_deny") == DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY
