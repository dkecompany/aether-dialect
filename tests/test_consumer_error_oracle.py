"""Consumer-facing terminal steps must scrub sensitivity/deny oracle detail."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants_runtime import (
    PERMISSION_DENIED_USER_MESSAGE,
    REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
)
from aetherdialect._contracts_base import FailureCategory
from aetherdialect._contracts_core import GenerationPath, SqlGenerationOutcome
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_session import PipelineSession


def _consumer_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_role = "consumer"
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = MagicMock()
    owner._runtime_config.engine_context = None
    owner._runtime_config.execution_context = None
    owner._context_name = "master"
    return owner


def _consumer_port(owner: MagicMock) -> MagicMock:
    port = MagicMock()
    port._owner = owner
    port.execution_visible_objects = None
    port.space_tables = None
    port.space_columns = None
    return port


@pytest.mark.fast
def test_sensitive_group_by_error_scrubbed_for_consumer() -> None:
    leaking = "main query: sensitive column users.email cannot be used in GROUP BY"
    gen_out = SqlGenerationOutcome(
        "",
        False,
        GenerationPath.INTENT_DIRECT_MATCH,
        None,
        sql_validation_error=leaking,
        error_kind=FailureCategory.SENSITIVE_GROUP_BY.value,
    )
    owner = _consumer_owner()
    port = _consumer_port(owner)
    snap_post = MagicMock()
    snap_post.q_norm = "count users by email"
    intent = MagicMock()
    intent.sql_param = "SELECT 1"

    with patch(
        "aetherdialect._main_interactive.generate_and_validate_sql",
        return_value=gen_out,
    ):
        with patch("aetherdialect._main_interactive.note_interactive_turn") as note:
            MainExecutionOps._run_sql_phase_after_intent_confirm(
                q_norm="count users by email",
                intent=intent,
                schema=MagicMock(),
                store={},
                templates={},
                rejected={},
                dialect=MagicMock(),
                choice_port=port,
                snap_post=snap_post,
                join_candidates={},
                cmap={},
                cte_join_hints={},
                matched_template=None,
                union_select_cols=None,
                cols_changed=False,
                structural_match_templates=None,
                union_sql_path=None,
                matched_rejected_template=None,
                persist_template_learning=False,
            )
    note.assert_called_once()
    recorded = note.call_args.kwargs
    assert recorded["outcome"] == "permission_denied"
    assert recorded["error"] is None
    assert recorded["sql"] is None
    assert recorded["intent"] is None

    session = PipelineSession(owner, mode="reader")
    session._turn_question = "count users by email"
    session._last_turn_outcome = {
        "outcome": "permission_denied",
        "error": None,
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
        "failure_kind": FailureCategory.SENSITIVE_GROUP_BY.value,
        "retryable": None,
        "refusal_diagnostic_code": None,
    }
    with patch.object(session, "_emit_turn_llm_usage", return_value=()):
        step = session._completed_step()

    assert step.error is not None and step.error.code.value == "forbidden"
    assert step.answer is None
    assert step.sql is None
    serialized = str(step)
    assert "users" not in serialized
    assert "email" not in serialized
    assert leaking not in serialized
    assert REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE not in serialized
    _ = PERMISSION_DENIED_USER_MESSAGE
