"""Tests for the separated execute step and standalone execute_sql."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import (
    PIPELINE_SUSPEND_ID_EXECUTE,
    SESSION_KIND_EXECUTE,
)
from aetherdialect._contracts_base import PipelineSuspended
from aetherdialect._contracts_core import (
    GenerationPath,
    InteractiveTailSnapshot,
    RuntimeIntent,
    SqlExecuteSuspendContext,
    SqlGenerationOutcome,
)
from aetherdialect._main_execution import (
    PipelineSession,
    _complete_interactive_execute,
    dispatch_pipeline_resume,
)


class TestExecuteSql:
    def test_happy_path_returns_rows(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            rows = sb.engine.execute_sql("SELECT COUNT(*) FROM film")
        assert rows
        assert int(rows[0][0]) >= 0

    def test_forbidden_statement_rejected_before_execute(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            with pytest.raises(ValueError, match="forbidden|select-only|validation|failed"):
                sb.engine.execute_sql("DELETE FROM film WHERE film_id = 1")

    def test_consumer_out_of_scope_blocked(self) -> None:
        from aetherdialect import EngineContext, Sandbox
        from aetherdialect._constants import CONSUMER_RESTRICTED_ALLOW_OBJECTS
        from aetherdialect._schema_graph import assert_consumer_sql_in_scope

        with Sandbox() as sandbox:
            engine = sandbox.engine(
                EngineContext(allow_objects=CONSUMER_RESTRICTED_ALLOW_OBJECTS),
                role="consumer",
            )
            sql = "SELECT customer_id FROM customer LIMIT 1"
            allowed = assert_consumer_sql_in_scope(
                sql,
                engine._dialect,
                engine._runtime_config.engine_context,
                engine._schema_graph,
                frozenset({"film"}),
            )
            assert allowed is False


class TestSessionExecuteStep:
    def _execute_suspend(self) -> tuple[PipelineSession, PipelineSuspended]:
        intent = RuntimeIntent(
            tables=["film"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        owner = MagicMock()
        owner._audit_emit = MagicMock()
        owner._schema_graph = MagicMock(effective_structural_hash="h")
        dialect = MagicMock()
        dialect.finalize_render.return_value = "SELECT COUNT(*) FROM film"
        tail = InteractiveTailSnapshot(
            q_norm="how many films",
            intent=intent,
            schema=None,
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            dialect=dialect,
            semantic_warnings=(),
            has_union_match=False,
            cols_changed=False,
            matched_template=None,
            union_select_cols=None,
            structural_match_templates=(),
            ikey="k",
            intent_sim=0.0,
        )
        gen_out = SqlGenerationOutcome(
            sql="SELECT COUNT(*) FROM film",
            success=True,
            generation_path=GenerationPath.INTENT_DIRECT_MATCH,
            matched_template=None,
        )
        ctx = SqlExecuteSuspendContext(
            tail=tail,
            execution_intent=intent,
            sql="SELECT COUNT(*) FROM film",
            gen_out=gen_out,
            matched_rejected_template=None,
            force_feedback=False,
            tmpl_sd=None,
            rows=(),
        )
        owner = MagicMock()
        owner._audit_emit = MagicMock()
        owner._schema_graph = MagicMock(effective_structural_hash="h")
        sess = PipelineSession(owner)
        ex = PipelineSuspended(PIPELINE_SUSPEND_ID_EXECUTE, "Execute this SQL?", ctx)
        return sess, ex

    def test_suspend_maps_to_execute_kind(self) -> None:
        sess, ex = self._execute_suspend()
        step = PipelineSession._suspend_to_step(sess, ex)
        assert step.kind == SESSION_KIND_EXECUTE
        assert step.sql == "SELECT COUNT(*) FROM film"

    def test_confirm_resume_runs_execute_path(self) -> None:
        sess, ex = self._execute_suspend()
        sess._choice_queue.append((PIPELINE_SUSPEND_ID_EXECUTE, "y"))
        with patch(
            "aetherdialect._main_execution._offer_sql_feedback_after_execute",
        ) as offer:
            dispatch_pipeline_resume(sess, ex)
        offer.assert_called_once()

    def test_cached_rows_skip_re_execute(self) -> None:
        intent = RuntimeIntent(
            tables=["film"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        tail = InteractiveTailSnapshot(
            q_norm="q",
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
        gen_out = SqlGenerationOutcome(
            sql="SELECT 1",
            success=True,
            generation_path=GenerationPath.INTENT_DIRECT_MATCH,
            matched_template=None,
        )
        ctx = SqlExecuteSuspendContext(
            tail=tail,
            execution_intent=intent,
            sql="SELECT 1",
            gen_out=gen_out,
            matched_rejected_template=None,
            force_feedback=False,
            tmpl_sd=None,
            rows=((42,),),
        )
        with patch("aetherdialect._main_execution._run_pipeline_sql_rows") as run_rows:
            with patch("aetherdialect._main_execution._offer_sql_feedback_after_execute"):
                _complete_interactive_execute(ctx, "y")
        run_rows.assert_not_called()
