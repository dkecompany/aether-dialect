"""domain_knowledge metadata answers from the active knowledge list."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import SESSION_KIND_META
from aetherdialect._constants_runtime import META_EMPTY_DOMAIN_KNOWLEDGE_MESSAGE
from aetherdialect._contracts_base import DomainKnowledgeEntry, DomainKnowledgeHolder
from aetherdialect._contracts_core import QuestionRoute
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import domain_knowledge_scope


def _owner_with_dk() -> MagicMock:
    owner = MagicMock()
    owner._domain_knowledge = DomainKnowledgeHolder()
    return owner


@pytest.mark.fast
def test_prose_message() -> None:
    entries = (
        DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="metric"),
        DomainKnowledgeEntry(key="fy", text="Fiscal year starts in July.", kind="policy"),
    )
    owner = _owner_with_dk()
    llm_answer = {
        "response_kind": "domain_knowledge",
        "message": "ARR means annual recurring revenue; FY starts in July.",
    }
    with (
        domain_knowledge_scope(entries=entries, digest="d1"),
        patch("aetherdialect._llm_provider.LLMProvider.json", return_value=llm_answer) as llm_mock,
    ):
        step = MainExecutionOps.answer_metadata_question(
            owner, "what is ARR", QuestionRoute.DOMAIN_KNOWLEDGE, None, None, None
        )
    assert step.answer == "ARR means annual recurring revenue; FY starts in July."
    assert step.kind == SESSION_KIND_META
    llm_mock.assert_called_once()
    system, user = llm_mock.call_args[0][:2]
    assert "domain knowledge" in system.lower()
    assert "Annual recurring revenue." in user
    assert "Fiscal year starts in July." in user


@pytest.mark.fast
def test_empty_dk_fixed_message() -> None:
    owner = _owner_with_dk()
    with (
        domain_knowledge_scope(entries=(), digest=None),
        patch("aetherdialect._llm_provider.LLMProvider.json") as llm_mock,
    ):
        step = MainExecutionOps.answer_metadata_question(
            owner, "what is ARR", QuestionRoute.DOMAIN_KNOWLEDGE, None, None, None
        )
    llm_mock.assert_not_called()
    assert step.answer == META_EMPTY_DOMAIN_KNOWLEDGE_MESSAGE
    assert step.kind == SESSION_KIND_META
    assert step.sql is None
    assert step.error is None


@pytest.mark.fast
def test_step_kind_meta_sql_none() -> None:
    entries = (DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="metric"),)
    owner = _owner_with_dk()
    llm_answer = {"response_kind": "domain_knowledge", "message": "ARR is annual recurring revenue."}
    with (
        domain_knowledge_scope(entries=entries, digest="d1"),
        patch("aetherdialect._llm_provider.LLMProvider.json", return_value=llm_answer),
    ):
        step = MainExecutionOps.answer_metadata_question(
            owner, "define ARR", QuestionRoute.DOMAIN_KNOWLEDGE, None, None, None
        )
    assert step.kind == SESSION_KIND_META
    assert step.done is True
    assert step.sql is None
    assert step.data is None
    assert step.error is None
    assert step.error is None
