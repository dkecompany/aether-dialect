"""Tests for feedback SQL rendering and failure-memory payload boundaries."""

from __future__ import annotations

import json
from unittest.mock import patch

from aetherdialect._contracts_core import FeedbackKind, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import render_feedback_sql
from aetherdialect._templates import TemplateOps


def _simple_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["film"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _simple_schema() -> SchemaGraph:
    tables = {
        "film": TableMetadata(
            name="film",
            columns={"title": ColumnMetadata(name="title", data_type="text", sensitivity="none")},
            primary_key=["film_id"],
            foreign_keys=[],
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


class TestSummarizeFailureForMemorySqlBoundary:
    def test_llm_payload_omits_sql_and_carries_intent_structure(self) -> None:
        captured: dict = {}

        def _fake_llm(_system: str, user: str, **_kw) -> str:
            captured["user"] = user
            return '{"summary":"wrong table","bucket":"WRONG_TABLES_OR_JOINS"}'

        with patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True):
            with patch("aetherdialect._templates.LLMProvider.chat", side_effect=_fake_llm):
                entry = TemplateOps.summarize_failure_for_memory(
                    question="q",
                    intent=_simple_intent(),
                    kind=FeedbackKind.INTENT_REJECTED,
                    schema_hash="hash",
                    user_reason="wrong table",
                )
        payload = json.loads(captured["user"])
        assert "sql" not in payload
        assert "intent_structure_json" in payload
        assert entry.summary

    def test_persisted_entry_has_no_sql(self) -> None:
        with patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=False):
            entry = TemplateOps.summarize_failure_for_memory(
                question="q",
                intent=_simple_intent(),
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash="hash",
                user_reason="wrong table",
            )
        payload = entry.to_dict()
        assert "sql" not in payload

    def test_render_feedback_sql_is_not_sent_to_failure_memory(self) -> None:
        intent = _simple_intent()
        schema = _simple_schema()
        feedback_sql = render_feedback_sql(intent, schema)
        captured: dict = {}

        def _fake_llm(_system: str, user: str, **_kw) -> str:
            captured["user"] = user
            return '{"summary":"bad","bucket":"OTHER"}'

        with patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True):
            with patch("aetherdialect._templates.LLMProvider.chat", side_effect=_fake_llm):
                entry = TemplateOps.summarize_failure_for_memory(
                    question="q",
                    intent=intent,
                    kind=FeedbackKind.INTENT_REJECTED,
                    schema_hash=schema.effective_structural_hash,
                    user_reason="bad",
                )
        assert feedback_sql
        assert feedback_sql not in captured["user"]
        assert "sql" not in entry.to_dict()
