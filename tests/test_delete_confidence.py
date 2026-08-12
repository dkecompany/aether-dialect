"""Regression tests for delete-confidence scoring and phase timers."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedPreparedStep,
    FeedbackKind,
    GenerationPath,
    LlmUsageRecord,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeIntent,
    SqlGenerationOutcome,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import FederationPlanTemplate, SQLShape, TemplateStats
from aetherdialect._federation_execute import (
    credit_federation_plan_accept,
    load_federation_plan_templates,
    save_federation_plan_template,
)
from aetherdialect._pipeline_generate import (
    credit_federation_accept,
    emit_explain_soft_diagnostics,
)
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    emit_llm_usage_summary_diagnostics,
    phase_timer,
    set_diagnostic_collector,
    summarize_llm_usage_by_phase_and_source,
)


def test_lookup_join_feedback_filters_by_member() -> None:
    store = {
        "question_feedback": {
            "owner::q": [
                QuestionFeedbackEntry(
                    summary="wrong join on owner",
                    buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
                    kind=FeedbackKind.INTENT_REJECTED,
                    effective_structural_hash="h",
                    intent_structural_hash="h",
                    intent_payload="{}",
                    created_at="t1",
                    updated_at="t1",
                    member_source_id="owner",
                ).to_dict(),
            ],
            "consumer::q": [
                QuestionFeedbackEntry(
                    summary="wrong join on consumer",
                    buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
                    kind=FeedbackKind.INTENT_REJECTED,
                    effective_structural_hash="h",
                    intent_structural_hash="h2",
                    intent_payload="{}",
                    created_at="t2",
                    updated_at="t2",
                    member_source_id="consumer",
                ).to_dict(),
            ],
        }
    }
    owner_fb = TemplateOps.lookup_join_feedback_for_question(store, "q", member_source_id="owner")
    assert owner_fb == ["wrong join on owner"]
    consumer_fb = TemplateOps.lookup_join_feedback_for_question(store, "q", member_source_id="consumer")
    assert consumer_fb == ["wrong join on consumer"]


def test_emit_explain_soft_diagnostics_notifies() -> None:
    from aetherdialect._contracts_base import SqlDiagnostic, SqlDiagnosticCode

    diags: list = []
    set_diagnostic_collector(diags)
    emit_explain_soft_diagnostics(
        (
            SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED, message="seq soft"),
            SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE, message="zero soft"),
        )
    )
    assert len(diags) == 2
    assert any("seq soft" in d.message for d in diags)


def test_phase_timer_sets_duration() -> None:
    diags: list = []
    set_diagnostic_collector(diags)
    with phase_timer("intent_parse", source_id="owner"):
        pass
    assert len(diags) == 1
    assert diags[0].duration_ms is not None
    assert diags[0].duration_ms >= 0


def test_llm_usage_summary_by_phase_and_source() -> None:
    records = (
        LlmUsageRecord(
            scope="question",
            block_id=1,
            task="interpret",
            logical_model="gpt",
            api_model="gpt",
            provider="sandbox",
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            cache_write_tokens=None,
            attempt=1,
            elapsed_ms=1,
            phase="interpret",
            source_id="owner",
        ),
        LlmUsageRecord(
            scope="question",
            block_id=1,
            task="join",
            logical_model="gpt",
            api_model="gpt",
            provider="sandbox",
            input_tokens=20,
            cached_input_tokens=0,
            output_tokens=8,
            cache_write_tokens=None,
            attempt=1,
            elapsed_ms=2,
            phase="join",
            source_id="consumer",
        ),
    )
    summary = summarize_llm_usage_by_phase_and_source(records)
    assert ("question", "interpret", "owner", 1, 10, 5) in summary
    assert ("question", "join", "consumer", 1, 20, 8) in summary


def test_emit_llm_usage_summary_diagnostics() -> None:
    diags: list = []
    set_diagnostic_collector(diags)
    emit_llm_usage_summary_diagnostics(
        (
            LlmUsageRecord(
                scope="question",
                block_id=1,
                task="interpret",
                logical_model="gpt",
                api_model="gpt",
                provider="sandbox",
                input_tokens=3,
                cached_input_tokens=0,
                output_tokens=1,
                cache_write_tokens=None,
                attempt=1,
                elapsed_ms=1,
                phase="interpret",
                source_id="owner",
            ),
        )
    )
    assert any("LLM question/interpret" in d.message for d in diags)


@patch("aetherdialect._templates_ops.TemplateOps.save_template_store")
@patch("aetherdialect._pipeline_generate.credit_federation_plan_accept")
def test_credit_federation_accept_credits_member_and_plan(
    mock_plan_credit: pytest.Mock,
    mock_save: pytest.Mock,
) -> None:
    tmpl = Template(
        id="T0001",
        effective_structural_hash="h",
        intent_signature=ConcreteIntent(
            intent_id="t",
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="ik1",
        tables_used=["orders"],
        sql_param="SELECT 1",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(param_values=[{}], questions=[], natural_language=[]),
        stats=TemplateStats(accept=0, reject=0),
        trust_level=1,
    )
    sub_intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    step = FederatedPreparedStep(
        source_id="owner",
        sub_intent=sub_intent,
        sql="SELECT 1",
        matched_template=tmpl,
    )
    store = {"templates": {"T0001": tmpl}, "question_feedback": {}, "next_id": 2}
    credit_federation_accept(
        q_norm="show orders",
        federation_dir="/tmp/fed",
        plan_id="plan1",
        steps=[step],
        stores_by_source={"owner": store},
        schemas_by_source={},
    )
    assert tmpl.stats.accept == 1
    mock_plan_credit.assert_called_once_with(
        "/tmp/fed",
        "plan1",
        "show orders",
        member_template_ids=(("owner", "T0001"),),
        pending_plan_template=None,
    )
    mock_save.assert_called()


def test_credit_federation_plan_accept() -> None:
    with tempfile.TemporaryDirectory() as fed_dir:
        template = FederationPlanTemplate(
            plan_id="plan1",
            composite_schema_graph_id="cg1",
            intent_key="ik1",
            step_fingerprints=(("owner", "sk1"),),
            combine_hash="hash1",
            question="q",
        )
        save_federation_plan_template(fed_dir, template)
        credit_federation_plan_accept(fed_dir, "plan1", "show orders")
        loaded = load_federation_plan_templates(fed_dir)
        assert "show orders" in loaded["plan1"].accepted_questions


def test_federation_plan_generation_path_exists() -> None:
    assert GenerationPath.FEDERATION_PLAN.code == "6"
    assert GenerationPath.parse("6") is GenerationPath.FEDERATION_PLAN


def test_federation_plan_accept_does_not_require_matched_template() -> None:
    gen = SqlGenerationOutcome(
        sql="-- display",
        success=True,
        generation_path=GenerationPath.FEDERATION_PLAN,
        matched_template=None,
        federated_steps=(),
        federation_plan_id="p1",
        federation_dir="/fed",
    )
    assert gen.matched_template is None
    assert gen.generation_path is GenerationPath.FEDERATION_PLAN
