"""Tests for permission-denied session response shape and learning suppression."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import AccessError
from aetherdialect._main_execution import (
    MainExecutionOps,
    PipelineSession,
)


class TestPermissionDeniedResponse:
    def test_completed_step_hides_sql_and_intent(self) -> None:
        owner = MagicMock()
        owner._schema_graph = MagicMock()
        owner._schema_graph.effective_structural_hash = "test_hash"
        owner._store = {}
        owner._templates = {}
        owner._rejected = {}
        owner._schema_terms = set()
        owner._runtime_config = MagicMock(llm_execution=MagicMock())
        owner._audit_emit = MagicMock()
        owner._llm_config = MagicMock(provider="mock")
        session = PipelineSession(owner)
        session._last_turn_outcome = {
            "outcome": "permission_denied",
            "sql": "SELECT secret FROM t",
            "intent": MagicMock(),
        }
        session._turn_question = "q"
        session._session_busy = True
        step = session._completed_step()
        assert step.message == PERMISSION_DENIED_USER_MESSAGE
        assert step.sql is None
        assert step.data is None
        assert step.intent_summary is None
        assert step.status == "permission_denied"

    def test_execute_access_error_notes_validation_failed_without_sql(self) -> None:
        port = MagicMock()
        dialect = MagicMock()
        dialect.finalize_render.return_value = "SELECT 1"
        dialect.execute.side_effect = AccessError("execute", "permission denied for relation secret")
        snap_post = MagicMock()
        snap_post.q_norm = "q"
        intent = MagicMock()
        intent.sql_param = "SELECT 1"
        gen_out = SimpleNamespace(
            success=True,
            sql="SELECT 1",
            matched_template=None,
            explain_soft_diagnostics=0,
        )
        with patch(
            "aetherdialect._main_execution.generate_and_validate_sql",
            return_value=gen_out,
        ):
            with patch("aetherdialect._main_execution.note_interactive_turn") as note:
                out = MainExecutionOps._run_sql_phase_after_intent_confirm(
                    q_norm="q",
                    intent=intent,
                    schema=MagicMock(),
                    store={},
                    templates={},
                    rejected={},
                    dialect=dialect,
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
                )
        assert out is None
        note.assert_called_once()
        assert note.call_args.kwargs["outcome"] == "validation_failed"
        assert note.call_args.kwargs["sql"] is None
        assert note.call_args.kwargs["intent"] is None

    def test_feedback_reject_reader_mode_emits_queue_not_inline_save(self, monkeypatch) -> None:
        from aetherdialect._contracts_base import NormalizedExpr
        from aetherdialect._contracts_core import (
            GenerationPath,
            RuntimeIntent,
            SelectCol,
        )
        from aetherdialect._pipeline import complete_user_feedback_reject

        emitted: list[object] = []
        monkeypatch.setattr(
            "aetherdialect._pipeline._emit_reader_write_queue_event",
            lambda _store, ev: emitted.append(ev),
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        ctx = MagicMock()
        ctx.intent = intent
        ctx.sql = "SELECT 1"
        ctx.schema = MagicMock()
        ctx.schema.schema_graph_id = "sg_x"
        ctx.schema.effective_structural_hash = "h"
        ctx.store = {}
        ctx.templates = {}
        ctx.q_norm = "q"
        ctx.matched_template = MagicMock(id="T1")
        ctx.generation_path = GenerationPath.EXACT_QUESTION_REUSE
        ctx.matched_rejected_template = None
        ctx.dialect = None
        ctx.structural_match_templates = None
        complete_user_feedback_reject(
            ctx,
            needs_reason=False,
            reject_reason="",
            choice_port=None,
            persist_template_learning=False,
        )
        assert len(emitted) == 1
