"""Tests for render_feedback_sql and feedback persistence boundaries."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    FeedbackKind,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._sql_gen import render_feedback_sql
from aetherdialect._templates import summarize_failure_for_memory


def _simple_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )


def _simple_schema() -> SchemaGraph:
    from aetherdialect._contracts_schema import ColumnMetadata, TableMetadata

    film = TableMetadata(
        name="film",
        columns={"title": ColumnMetadata(name="title", data_type="text")},
        foreign_keys=[],
        primary_key="film_id",
    )
    return SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="hash",
        tables={"film": film},
    )


class TestRenderFeedbackSql:
    def test_renders_dialect_neutral_sql(self) -> None:
        sql = render_feedback_sql(_simple_intent(), _simple_schema())
        assert sql
        assert "film" in sql.lower()

    def test_none_schema_returns_none(self) -> None:
        assert render_feedback_sql(_simple_intent(), None) is None

    def test_non_renderable_intent_returns_none(self) -> None:
        broken = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        with patch(
            "aetherdialect._sql_gen.build_deterministic_sql",
            side_effect=RuntimeError("cannot render"),
        ):
            assert render_feedback_sql(broken, _simple_schema()) is None


class TestSummarizeFailureForMemorySqlBoundary:
    def test_llm_payload_includes_sql_when_provided(self) -> None:
        captured: dict = {}

        def _fake_llm(_system: str, user: str, **_kw) -> str:
            captured["user"] = user
            return '{"summary":"wrong table","bucket":"WRONG_TABLES_OR_JOINS"}'

        with patch("aetherdialect._templates.llm_credentials_configured", return_value=True):
            with patch("aetherdialect._templates.llm_chat", side_effect=_fake_llm):
                entry = summarize_failure_for_memory(
                    question="q",
                    intent=_simple_intent(),
                    kind=FeedbackKind.INTENT_REJECTED,
                    schema_hash="hash",
                    user_reason="wrong table",
                    sql="SELECT title FROM film",
                )
        assert "sql" in captured["user"]
        assert entry.summary

    def test_persisted_entry_has_no_sql(self) -> None:
        with patch("aetherdialect._templates.llm_credentials_configured", return_value=False):
            entry = summarize_failure_for_memory(
                question="q",
                intent=_simple_intent(),
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash="hash",
                user_reason="wrong table",
                sql="SELECT title FROM film",
            )
        payload = entry.to_dict()
        assert "sql" not in payload

    def test_render_feedback_sql_used_for_llm_only(self) -> None:
        intent = _simple_intent()
        schema = _simple_schema()
        feedback_sql = render_feedback_sql(intent, schema)
        with patch("aetherdialect._templates.llm_credentials_configured", return_value=False):
            entry = summarize_failure_for_memory(
                question="q",
                intent=intent,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash=schema.effective_structural_hash,
                user_reason="bad",
                sql=feedback_sql,
            )
        assert feedback_sql
        assert "sql" not in entry.to_dict()
