"""Crafted join-unreachable refusals must reach the interactive terminal step."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    REPHRASE_HINT_MESSAGES,
)
from aetherdialect._contracts_base import FailureCategory
from aetherdialect._contracts_core import NormalizedExpr, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import IntentIssue
from aetherdialect._core_utils import note_interactive_turn
from aetherdialect._intent_process import _refuse_if_join_unreachable_repair_removed_tables
from aetherdialect._intent_repair import refusal_for_join_unreachable_table_removal
from aetherdialect._main_execution import PipelineSession


def _join_unreachable_issue(*, root: str = "orders", target: str = "products") -> IntentIssue:
    return IntentIssue.make(
        issue_id=f"join_unreachable_main_query_{root}_{target}",
        category=FailureCategory.WRONG_JOIN,
        severity="error",
        message=(
            f"main query: no schema join path between '{root}' and '{target}' "
            "(disconnected FK groups; add a bridging foreign_keys_add)."
        ),
        context={
            "root": root,
            "target": target,
            "tables": sorted([root, target]),
            "scope_label": "main query",
        },
    )


def _intent_with_tables(*tables: str) -> RuntimeIntent:
    return RuntimeIntent(
        tables=list(tables),
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{tables[0]}.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_unreachable_join_message_reaches_the_user() -> None:
    before = _intent_with_tables("orders", "products")
    after = _intent_with_tables("orders")
    errors = [_join_unreachable_issue()]
    expected = refusal_for_join_unreachable_table_removal(before, after, errors)
    assert expected is not None

    assert _refuse_if_join_unreachable_repair_removed_tables(before, after, errors, "test_phase")

    choice_port = MagicMock()
    note_interactive_turn(
        choice_port,
        outcome="parse_failed",
        error="Intent parse failed.",
    )
    choice_port.note_turn_outcome.assert_called_once()
    recorded = choice_port.note_turn_outcome.call_args.kwargs
    assert recorded["outcome"] == "parse_failed"
    assert recorded["error"] == expected
    assert recorded["refusal_diagnostic_code"] == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    assert "no foreign key or semantic edge" in recorded["error"]
    assert "orders" in recorded["error"] and "products" in recorded["error"]
    assert recorded["error"] != "Intent parse failed."
    assert recorded["error"] != REPHRASE_HINT_MESSAGES["intent_parse_failed"]


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = MagicMock()
    owner.dialect = "sqlite"
    return owner


@pytest.mark.fast
def test_unreachable_join_refusal_survives_completed_step() -> None:
    session = PipelineSession(_session_owner())
    session._turn_question = "join orders to products"
    crafted = (
        "Tables 'orders' and 'products' cannot be joined: no foreign key or semantic edge "
        "relates them. Repair removed a table to clear the error, which would answer a "
        "different question. Declare foreign_keys_add or a semantic neighbour override when "
        "the relationship is real."
    )
    session._last_turn_outcome = {
        "outcome": "parse_failed",
        "error": crafted,
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session, "_emit_turn_llm_usage", lambda **_: ())
        step = session._completed_step()
    assert step.error == crafted
    assert step.message == crafted
    codes = {d.code for d in step.diagnostics}
    assert DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE in codes or step.error == crafted
