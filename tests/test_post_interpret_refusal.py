"""Post-interpret unknown-entity refusal and uniform refusal terminal egress."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
)
from aetherdialect._constants_runtime import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import FailureCategory, NormalizedExpr
from aetherdialect._contracts_core import (
    IntentSummary,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_loop import full_intent_parse
from aetherdialect._main_session import PipelineSession
from aetherdialect._utils import (
    note_interactive_turn,
    refusal_terminal_cleared_egress_fields,
    reset_diagnostic_collector,
    set_diagnostic_collector,
    stash_intent_parse_refusal,
)


def _customer_schema() -> SchemaGraph:
    col = ColumnMetadata(
        name="id",
        data_type="integer",
        is_primary_key=True,
        distinct_count=1,
        distinct_ratio=1.0,
        null_ratio=0.0,
        is_canonical_duplicate=False,
    )
    return SchemaGraph(
        tables={
            "customer": TableMetadata(
                name="customer",
                columns={"id": col},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="cust_hash",
    )


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    return owner


def _intent_summary() -> IntentSummary:
    return IntentSummary(
        tables=("secret",),
        select_cols=(),
        filters=(),
        group_by=(),
        order_by=(),
        limit=None,
        natural_language="leaked",
    )


def _runtime_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["secret"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("secret.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_refusal_terminal_cleared_egress_fields() -> None:
    assert refusal_terminal_cleared_egress_fields() == {
        "intent_summary": None,
        "interpretation": None,
    }


@pytest.mark.fast
def test_unknown_entity_refuses_before_compose() -> None:
    schema = _customer_schema()
    interpret_unknown = (
        '{"approach":"list payroll rows","tables":["payroll"],'
        '"grounding":[{"ref":"payroll.id","used_for":"identify rows"}]}'
    )
    buf: list = []
    tok = set_diagnostic_collector(buf)
    try:
        with patch("aetherdialect._intent_loop.LLMProvider.chat", return_value=interpret_unknown) as chat:
            out, _warns, _calls, plan = full_intent_parse("show payroll", schema, store=None, max_retries=0)
    finally:
        reset_diagnostic_collector(tok)
    assert out is None
    assert plan is not None
    assert plan.tables == ("payroll",)
    assert chat.call_count == 1
    assert DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED in {getattr(d, "code", "") for d in buf}


@pytest.mark.fast
def test_unknown_entity_turn_outcome_collapses_to_permission_denied() -> None:
    port = MagicMock()
    stash_intent_parse_refusal(DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED, PERMISSION_DENIED_USER_MESSAGE)
    note_interactive_turn(port, outcome="parse_failed", error="Intent parse failed.")
    recorded = port.note_turn_outcome.call_args.kwargs
    assert recorded["outcome"] == "permission_denied"
    assert recorded["error"] is None
    assert recorded["refusal_diagnostic_code"] == DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED


@pytest.mark.fast
@pytest.mark.parametrize(
    ("outcome", "extra"),
    [
        ("permission_denied", {}),
        (
            "parse_failed",
            {
                "refusal_diagnostic_code": DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
            },
        ),
        (
            "validation_failed",
            {
                "refusal_diagnostic_code": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
                "error": "These tables could not be connected: alpha, beta.",
            },
        ),
        (
            "not_available_in_context",
            {
                "failure_kind": FailureCategory.UNKNOWN_TABLE.value,
            },
        ),
    ],
)
def test_refusal_terminal_clears_intent_summary_and_interpretation(outcome: str, extra: dict) -> None:
    session = PipelineSession(_session_owner())
    session._turn_question = "q"
    session._last_turn_outcome = {
        "outcome": outcome,
        "error": extra.get("error"),
        "sql": "SELECT secret FROM t",
        "rows": None,
        "columns": None,
        "rejection_bucket": None,
        "intent": _runtime_intent(),
        "matched_template": None,
        "template_history_index": None,
        "federated_bundle": None,
        "federated_plan": None,
        "generation_path": None,
        "federation_source_id": None,
        "federation_phase": None,
        "federation_succeeded": (),
        "failure_kind": extra.get("failure_kind"),
        "retryable": None,
        "refusal_diagnostic_code": extra.get("refusal_diagnostic_code"),
    }
    with patch.object(session, "_emit_turn_llm_usage", return_value=()):
        step = session._completed_step()
    assert step.intent_summary is None
    assert step.error is not None
    if (
        outcome == "permission_denied"
        or extra.get("refusal_diagnostic_code") == DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED
    ):
        assert step.error.code.value == "forbidden"
