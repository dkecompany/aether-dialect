"""Question feedback collection keys off effective_structural_hash, not schema_graph_id."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    FeedbackKind,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import ColumnMetadata, LogicalIntent, SchemaGraph, TableMetadata
from aetherdialect._intent_loop import full_intent_parse, logical_intent_to_serialisable
from aetherdialect._schema_graph import derive_deterministic_schema_graph_id
from aetherdialect._templates_ops import TemplateOps


def _schema_with_distinct_ids() -> SchemaGraph:
    col = ColumnMetadata(
        name="id",
        data_type="integer",
        is_primary_key=True,
        distinct_count=1,
        distinct_ratio=1.0,
        null_ratio=0.0,
        is_canonical_duplicate=False,
    )
    eff = "eff_feedback_scope_001"
    struct = "struct_feedback_scope_001"
    graph_id = derive_deterministic_schema_graph_id(eff, struct)
    assert graph_id != eff
    return SchemaGraph(
        tables={"t": TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])},
        join_paths_multi={},
        effective_structural_hash=eff,
        structural_hash=struct,
        schema_graph_id=graph_id,
    )


def _stored_feedback_row(effective_hash: str) -> dict[str, object]:
    return QuestionFeedbackEntry(
        summary="prior wrong join",
        buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash=effective_hash,
        intent_structural_hash="ih",
        intent_payload="{}",
        created_at="t0",
        updated_at="t0",
        source="engine",
    ).to_dict()


@pytest.mark.fast
def test_collect_matches_stored_hash() -> None:
    """full_intent_parse collects feedback scoped to effective_structural_hash."""
    sg = _schema_with_distinct_ids()
    eff = sg.effective_structural_hash
    graph_id = sg.schema_graph_id
    question = "count rows on t"
    store: dict[str, object] = {"question_feedback": {question: [_stored_feedback_row(eff)]}}

    assert TemplateOps.collect_question_feedback_for_prompt(store, question, graph_id) == []
    assert len(TemplateOps.collect_question_feedback_for_prompt(store, question, eff)) == 1

    captured_hashes: list[str] = []
    original_collect = TemplateOps.collect_question_feedback_for_prompt

    def _spy_collect(store_arg: object, q_norm: str, schema_hash: str) -> list[dict[str, str]]:
        captured_hashes.append(schema_hash)
        return original_collect(store_arg, q_norm, schema_hash)

    li = logical_intent_to_serialisable(LogicalIntent(tables=("t",), select="id column"))
    interpret_ok = '{"approach":"count rows on t","tables":["t"]}'
    good_ground = json.dumps(li, separators=(",", ":"), sort_keys=True)
    stub_intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        natural_language="nl",
    )

    with (
        patch.object(TemplateOps, "collect_question_feedback_for_prompt", side_effect=_spy_collect),
        patch("aetherdialect._intent_loop.LLMProvider.chat", side_effect=[interpret_ok, good_ground, "{}"]),
        patch(
            "aetherdialect._intent_loop._format_repair_loop",
            return_value=(stub_intent, 0),
        ),
    ):
        full_intent_parse(question, sg, store=store, max_retries=0)

    assert captured_hashes, "expected collect_question_feedback_for_prompt during parse"
    assert captured_hashes[0] == eff
    assert captured_hashes[0] != graph_id
