"""Suspended-session preview retention, resume re-execution, and TTL expiry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._constants import PIPELINE_SUSPEND_ID_EXECUTE
from aetherdialect._contracts_base import SuspendedSessionExpiredError
from aetherdialect._contracts_core import (
    GenerationPath,
    InteractiveTailSnapshot,
    PipelineSuspended,
    RuntimeIntent,
    SqlExecuteSuspendContext,
    SqlGenerationOutcome,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_session import PipelineSession

_PREVIEW_CAP = 10


def _runtime_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["film"],
        grain="row",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _tail(intent: RuntimeIntent | None = None) -> InteractiveTailSnapshot:
    intent = intent or _runtime_intent()
    return InteractiveTailSnapshot(
        q_norm="how many films",
        intent=intent,
        schema=MagicMock(),
        store={},
        templates={},
        rejected={},
        schema_terms=set(),
        dialect=MagicMock(),
        semantic_warnings=(),
        has_union_match=False,
        cols_changed=False,
        matched_template=None,
        union_select_cols=None,
        structural_match_templates=(),
        ikey="k",
        intent_sim=0.0,
    )


def _gen_out() -> SqlGenerationOutcome:
    matched = MagicMock()
    matched.trust_level = 2
    matched.feedback_by_question = {}
    matched.stats = MagicMock(accept=1, reject=0)
    return SqlGenerationOutcome(
        sql="SELECT film_id FROM film",
        success=True,
        generation_path=GenerationPath.INTENT_DIRECT_MATCH,
        matched_template=matched,
    )


def test_suspend_holds_preview_not_full_result() -> None:
    intent = _runtime_intent()
    tail = _tail(intent)
    gen_out = _gen_out()
    many_rows = [tuple([i]) for i in range(20)]

    ctx = MainExecutionOps._sql_feedback_suspend_context(
        tail,
        gen_out.sql,
        many_rows,
        None,
        gen_out,
        None,
        False,
        intent,
    )

    assert len(ctx.preview_rows) == _PREVIEW_CAP
    assert ctx.preview_rows == tuple(tuple([i]) for i in range(_PREVIEW_CAP))
    assert not hasattr(ctx, "rows")
    assert ctx.sql == gen_out.sql
    assert isinstance(ctx.sql_parameters, tuple)
    assert ctx.suspended_at is not None
    assert ctx.suspended_at.tzinfo is not None


def test_resume_reexecutes() -> None:
    intent = _runtime_intent()
    tail = _tail(intent)
    gen_out = _gen_out()
    execute_ctx = SqlExecuteSuspendContext(
        tail=tail,
        execution_intent=intent,
        sql=gen_out.sql,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=False,
        tmpl_sd=None,
        sql_parameters=(("p", 1),),
        suspended_at=datetime.now(UTC),
    )
    feedback_ctx = MainExecutionOps._sql_feedback_suspend_context(
        tail,
        gen_out.sql,
        [tuple([i]) for i in range(12)],
        None,
        gen_out,
        None,
        False,
        intent,
    )

    with patch(
        "aetherdialect._main_interactive.MainInteractiveOps._run_pipeline_sql_rows",
        return_value=[(99,)],
    ) as run_rows:
        with patch("aetherdialect._main_interactive.MainInteractiveOps._offer_sql_feedback_after_execute"):
            MainExecutionOps._complete_interactive_execute(execute_ctx, "y")
        assert run_rows.call_count == 1

        with patch("aetherdialect._main_interactive.handle_user_feedback"):
            MainExecutionOps._complete_interactive_sql_feedback(feedback_ctx, "y")
        assert run_rows.call_count == 2


def test_expired_suspension_reports() -> None:
    owner = MagicMock()
    owner.limits = EngineLimits(suspended_session_ttl_seconds=60)
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = MagicMock()
    owner._sandbox_closed = False
    owner._pipeline_writer_lock = None
    owner._phase_callback = None
    owner._audit_emit = MagicMock()
    owner._domain_knowledge = None

    intent = _runtime_intent()
    ctx = SqlExecuteSuspendContext(
        tail=_tail(intent),
        execution_intent=intent,
        sql="SELECT film_id FROM film",
        gen_out=_gen_out(),
        matched_rejected_template=None,
        force_feedback=False,
        tmpl_sd=None,
        suspended_at=datetime.now(UTC) - timedelta(seconds=120),
    )
    sess = PipelineSession(owner)
    sess._suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_EXECUTE, "Execute this SQL?", ctx)
    sess._session_busy = True

    with pytest.raises(SuspendedSessionExpiredError):
        sess.step("y")

    assert not sess.awaiting_prompt()
    assert sess._suspended is None
