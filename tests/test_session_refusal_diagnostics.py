"""Session-step refusals carry stable diagnostic codes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    REFUSAL_DIAGNOSTIC_CODES,
)
from aetherdialect._contracts_base import (
    AggregateJoinFanOutError,
    ComparisonJoinScopeExceededError,
    FailureCategory,
    NoJoinPathError,
)
from aetherdialect._contracts_schema import IntentIssue
from aetherdialect._contracts_core import GenerationPath, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._main_execution import PipelineSession
from aetherdialect._pipeline import _join_path_failure_outcome
from aetherdialect._refusal_diagnostics import (
    emit_session_refusal_diagnostic,
    refusal_diagnostic_code_for_exception,
    refusal_diagnostic_code_for_federation_reason,
    refusal_diagnostic_code_for_intent_issue,
)
from aetherdialect._templates import empty_template_store


@pytest.mark.fast
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (NoJoinPathError("main query", ["a", "b"]), DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE),
        (
            AggregateJoinFanOutError("main query", "SUM would duplicate rows"),
            DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
        ),
        (
            ComparisonJoinScopeExceededError("main query", "exceeding the limit of 3"),
            DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
        ),
    ],
)
def test_refusal_diagnostic_code_for_exception(exc: Exception, expected: str) -> None:
    assert refusal_diagnostic_code_for_exception(exc) == expected


@pytest.mark.fast
@pytest.mark.parametrize(
    ("issue_id", "expected"),
    [
        ("cte_step_count_exceeded", DIAGNOSTIC_CODE_REFUSAL_CTE_CAP),
        ("cte_reference_depth_exceeded", DIAGNOSTIC_CODE_REFUSAL_CTE_CAP),
        ("comparison_join_hop_ceiling", DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING),
    ],
)
def test_refusal_diagnostic_code_for_intent_issue(issue_id: str, expected: str) -> None:
    issue = IntentIssue(
        issue_id=issue_id,
        category=FailureCategory.CTE_STRUCTURE,
        severity="error",
        message="refused",
    )
    assert refusal_diagnostic_code_for_intent_issue(issue) == expected


@pytest.mark.fast
def test_refusal_diagnostic_code_for_federation_capability_reason() -> None:
    reason = "member capability: where operator 'ilike' is not supported by federation member 'sqlite'"
    assert refusal_diagnostic_code_for_federation_reason(reason) == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    assert refusal_diagnostic_code_for_federation_reason("median is not supported by all federation members") == (
        DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    )


@pytest.mark.fast
def test_refusal_diagnostic_codes_cover_catalogue() -> None:
    assert REFUSAL_DIAGNOSTIC_CODES == frozenset(
        {
            DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
            DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
            DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
            DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
            DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
        }
    )


@pytest.mark.fast
def test_join_path_failure_outcome_emits_refusal_diagnostic() -> None:
    schema = SchemaGraph(
        tables={
            "alpha": TableMetadata(
                name="alpha",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "beta": TableMetadata(
                name="beta",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff",
    )
    intent = RuntimeIntent(
        tables=["alpha", "beta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    exc = NoJoinPathError("main query", ["alpha", "beta"])
    buf: list = []
    tok = set_diagnostic_collector(buf)
    try:
        with (
            patch("aetherdialect._pipeline.print_rephrase_hint"),
            patch("aetherdialect._pipeline.save_template_store"),
        ):
            outcome = _join_path_failure_outcome(
                exc,
                q_norm="show alpha beta",
                intent=intent,
                schema=schema,
                store={},
                generation_path=GenerationPath.FRESH,
                matched_template=None,
                structural_match_templates=(),
                persist_template_learning=False,
            )
        codes = {d.code for d in buf}
    finally:
        reset_diagnostic_collector(tok)
    assert outcome.success is False
    assert outcome.refusal_diagnostic_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    assert DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE in codes


@pytest.mark.fast
def test_emit_session_refusal_diagnostic_records_code() -> None:
    buf: list = []
    tok = set_diagnostic_collector(buf)
    try:
        emit_session_refusal_diagnostic(DIAGNOSTIC_CODE_REFUSAL_CTE_CAP, "too many CTE steps")
        assert len(buf) == 1
        assert buf[0].code == DIAGNOSTIC_CODE_REFUSAL_CTE_CAP
        assert buf[0].level == "error"
    finally:
        reset_diagnostic_collector(tok)


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    return owner


@pytest.mark.fast
def test_validation_failed_terminal_step_carries_refusal_diagnostic() -> None:
    session = PipelineSession(_session_owner())
    session._turn_question = "q"
    session._last_turn_outcome = {
        "outcome": "validation_failed",
        "error": "These tables could not be connected: alpha, beta.",
        "sql": None,
        "rows": None,
        "columns": None,
        "rejection_bucket": None,
        "intent": None,
        "matched_template": None,
        "template_history_index": None,
        "federated_bundle": None,
        "federated_plan": None,
        "generation_path": None,
        "federation_source_id": None,
        "federation_phase": None,
        "federation_succeeded": (),
        "failure_kind": None,
        "retryable": None,
        "refusal_diagnostic_code": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    }
    with patch.object(session, "_emit_turn_llm_usage", return_value=()):
        step = session._completed_step()
    codes = {d.code for d in step.diagnostics}
    assert DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE in codes
    assert step.error == "These tables could not be connected: alpha, beta."
