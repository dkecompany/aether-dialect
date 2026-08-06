"""business_knowledge metadata answers from the active knowledge list."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import META_EMPTY_BUSINESS_KNOWLEDGE_MESSAGE, SESSION_KIND_META
from aetherdialect._contracts_base import BusinessKnowledgeEntry, BusinessKnowledgeHolder, QuestionRoute
from aetherdialect._core_utils import business_knowledge_scope
from aetherdialect._main_execution import MainExecutionOps


def _owner_with_bk() -> MagicMock:
    owner = MagicMock()
    owner._business_knowledge = BusinessKnowledgeHolder()
    return owner


@pytest.mark.fast
def test_prose_message() -> None:
    entries = (
        BusinessKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="metric"),
        BusinessKnowledgeEntry(key="fy", text="Fiscal year starts in July.", kind="policy"),
    )
    owner = _owner_with_bk()
    llm_answer = {
        "response_kind": "business_knowledge",
        "message": "ARR means annual recurring revenue; FY starts in July.",
    }
    with (
        business_knowledge_scope(entries=entries, digest="d1"),
        patch("aetherdialect._main_execution.LLMProvider.json", return_value=llm_answer) as llm_mock,
    ):
        step = MainExecutionOps.answer_metadata_question(
            owner, "what is ARR", QuestionRoute.BUSINESS_KNOWLEDGE, None, None, None
        )
    assert step.message == "ARR means annual recurring revenue; FY starts in July."
    assert step.meta_payload == {"response_kind": "business_knowledge"}
    llm_mock.assert_called_once()
    system, user = llm_mock.call_args[0][:2]
    assert "business knowledge" in system.lower()
    assert "Annual recurring revenue." in user
    assert "Fiscal year starts in July." in user


@pytest.mark.fast
def test_empty_bk_fixed_message() -> None:
    owner = _owner_with_bk()
    with (
        business_knowledge_scope(entries=(), digest=None),
        patch("aetherdialect._main_execution.LLMProvider.json") as llm_mock,
    ):
        step = MainExecutionOps.answer_metadata_question(
            owner, "what is ARR", QuestionRoute.BUSINESS_KNOWLEDGE, None, None, None
        )
    llm_mock.assert_not_called()
    assert step.message == META_EMPTY_BUSINESS_KNOWLEDGE_MESSAGE
    assert step.kind == SESSION_KIND_META
    assert step.meta_payload == {"response_kind": "business_knowledge"}
    assert step.sql is None
    assert step.error is None


@pytest.mark.fast
def test_step_kind_meta_sql_none() -> None:
    entries = (BusinessKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="metric"),)
    owner = _owner_with_bk()
    llm_answer = {"response_kind": "business_knowledge", "message": "ARR is annual recurring revenue."}
    with (
        business_knowledge_scope(entries=entries, digest="d1"),
        patch("aetherdialect._main_execution.LLMProvider.json", return_value=llm_answer),
    ):
        step = MainExecutionOps.answer_metadata_question(
            owner, "define ARR", QuestionRoute.BUSINESS_KNOWLEDGE, None, None, None
        )
    assert step.kind == SESSION_KIND_META
    assert step.done is True
    assert step.sql is None
    assert step.data is None
    assert step.status is None
    assert step.error is None
