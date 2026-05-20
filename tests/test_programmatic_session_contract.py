"""Contract tests for the agent-shaped programmatic surface (session-only, no engine stdout)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import aetherdialect
from aetherdialect import SessionStep
from aetherdialect.text2sql import Text2SQL, AsyncPipelineSession
from aetherdialect._templates import empty_template_store


def test_public_all_excludes_legacy_types() -> None:
    assert "QueryResult" not in aetherdialect.__all__
    assert "ConversationContext" not in aetherdialect.__all__
    assert "RecentTurn" not in aetherdialect.__all__
    assert "IntentSummary" not in aetherdialect.__all__


def test_text2sql_session_surface() -> None:
    assert hasattr(Text2SQL, "session")
    assert hasattr(Text2SQL, "asession")
    assert not hasattr(Text2SQL, "pipeline_session")
    assert not hasattr(Text2SQL, "apipeline_session")
    assert not hasattr(Text2SQL, "ask")
    assert not hasattr(Text2SQL, "aask")


@patch("aetherdialect._core_utils.diagnostic_debug_enabled", return_value=False)
def test_notify_emits_no_stdio_without_print_listener(_mock_dbg: object, capsys: pytest.CaptureFixture[str]) -> None:
    from aetherdialect._core_utils import notify

    notify("hello from contract test", stage="test")
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""


def test_session_step_agent_fields_exist() -> None:
    s = SessionStep(
        done=False,
        prompt="ok?",
        kind="awaiting_intent_confirm",
        reply_shape="yes_no",
        semantic_warnings=("warn1",),
    )
    assert s.reply_shape == "yes_no"
    assert s.semantic_warnings == ("warn1",)
    assert s.status is None


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    return owner


def test_pipeline_session_ask_typeerror_non_str() -> None:
    from aetherdialect._contracts_base import SessionActiveError
    from aetherdialect._main_execution import PipelineSession

    sess = PipelineSession(_session_owner())
    with pytest.raises(TypeError, match="str"):
        sess.ask(123)  # type: ignore[arg-type]


def test_pipeline_session_ask_blocked_emits_audit() -> None:
    from aetherdialect._contracts_base import SessionActiveError
    from aetherdialect._main_execution import PipelineSession
    from aetherdialect._pipeline import (
        PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
        PipelineSuspended,
    )

    owner = _session_owner()
    sess = PipelineSession(owner)
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        sess.ask("first")
    with pytest.raises(SessionActiveError):
        sess.ask("second")
    names = [c.args[0] for c in owner._audit_emit.call_args_list]
    assert "ask_begin" in names
    assert "ask_blocked" in names


def test_ask_until_done_yes_resumes_to_completion() -> None:
    from aetherdialect._main_execution import PipelineSession
    from aetherdialect._pipeline import (
        PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
        PipelineSuspended,
    )

    owner = _session_owner()
    sess = PipelineSession(owner)
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with (
        patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended),
        patch("aetherdialect._main_execution.dispatch_pipeline_resume") as disp,
    ):
        step = sess.ask_until_done("q", on_confirm="y")
    assert step.done is True
    disp.assert_called_once()


def test_ask_until_done_free_text_suspend_raises() -> None:
    from aetherdialect._contracts_base import SessionActiveError
    from aetherdialect._main_execution import PipelineSession
    from aetherdialect._pipeline import (
        PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
        PipelineSuspended,
    )

    owner = _session_owner()
    sess = PipelineSession(owner)
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_FEEDBACK, "why?", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        with pytest.raises(SessionActiveError, match="free-text"):
            sess.ask_until_done("q", on_confirm="y")


@pytest.mark.parametrize(
    ("state_id", "expected_reply"),
    [
        ("pipeline_suspend_intent_confirm", "yes_no"),
        ("pipeline_suspend_sql", "yes_no"),
        ("pipeline_suspend_direct_reuse", "yes_no"),
        ("pipeline_suspend_user_feedback_reject", "free_text"),
        ("pipeline_suspend_intent_feedback", "free_text"),
    ],
)
def test_suspend_reply_shape_by_kind(state_id: str, expected_reply: str) -> None:
    from aetherdialect._config import (
        PIPELINE_SUSPEND_ID_DIRECT_REUSE,
        PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
        PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
        PIPELINE_SUSPEND_ID_SQL,
        PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    )
    from aetherdialect._contracts_base import PipelineSuspended
    from aetherdialect._main_execution import PipelineSession

    sid_map = {
        "pipeline_suspend_intent_confirm": PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
        "pipeline_suspend_sql": PIPELINE_SUSPEND_ID_SQL,
        "pipeline_suspend_direct_reuse": PIPELINE_SUSPEND_ID_DIRECT_REUSE,
        "pipeline_suspend_user_feedback_reject": PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
        "pipeline_suspend_intent_feedback": PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    }
    owner = _session_owner()
    sess = PipelineSession(owner)
    ex = PipelineSuspended(sid_map[state_id], "m", None)
    step = sess._suspend_to_step(ex)
    assert step.reply_shape == expected_reply


def test_run_interactive_stdout_smoke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from aetherdialect import text2sql as tmod
    from aetherdialect._config import SESSION_KIND_RESULT

    class _SessCM:
        def __init__(self) -> None:
            self.inner = MagicMock()

        def __enter__(self) -> MagicMock:
            self.inner.ask.return_value = SessionStep(
                done=True,
                prompt=None,
                kind=SESSION_KIND_RESULT,
                message="ok",
            )
            return self.inner

        def __exit__(self, *args: object) -> None:
            return None

    cm = _SessCM()
    owner = MagicMock()
    owner._ensure_llm = MagicMock()

    monkeypatch.setattr("builtins.input", lambda *a, **k: "hello world")

    with patch.object(tmod, "PipelineSession", return_value=cm):
        Text2SQL.run_interactive(owner)

    out = capsys.readouterr().out
    assert "Interactive mode" in out
    assert "Enter question" in out


def test_async_pipeline_session_ask_smoke() -> None:
    from aetherdialect._config import SESSION_KIND_RESULT
    from aetherdialect._main_execution import PipelineSession

    owner = _session_owner()
    inner = PipelineSession(owner)

    async def _run() -> None:
        with patch("aetherdialect._main_execution.interactive_run_once", return_value=None):
            ap = AsyncPipelineSession(inner)
            step = await ap.ask("q")
        assert step.done is True
        assert step.kind == SESSION_KIND_RESULT

    asyncio.run(_run())


def test_async_pipeline_session_awaiting_prompt_delegates() -> None:
    from aetherdialect._main_execution import PipelineSession

    owner = _session_owner()
    inner = PipelineSession(owner)
    inner._suspended = MagicMock()

    async def _run() -> None:
        ap = AsyncPipelineSession(inner)
        assert await ap.awaiting_prompt() is True

    asyncio.run(_run())


def test_completed_step_sets_status_after_final_sql_reject() -> None:
    from aetherdialect._contracts_base import FailureCategory
    from aetherdialect._contracts_core import RuntimeIntent
    from aetherdialect._main_execution import PipelineSession

    owner = _session_owner()
    sess = PipelineSession(owner)
    sess._turn_question = "q1"
    ri = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )
    sess._last_turn_outcome = {
        "outcome": "intent_rejected",
        "error": None,
        "sql": "SELECT 1",
        "rows": [(1,)],
        "columns": None,
        "rejection_bucket": "OTHER",
        "intent": ri,
    }
    step = sess._completed_step()
    assert step.status == FailureCategory.RESULT_OKAY_INTENT_WRONG.value
    assert step.done is True


def test_async_pipeline_session_ask_typeerror() -> None:
    from aetherdialect._main_execution import PipelineSession

    owner = _session_owner()
    inner = PipelineSession(owner)

    async def _run() -> None:
        ap = AsyncPipelineSession(inner)
        with pytest.raises(TypeError, match="str"):
            await ap.ask(123)  # type: ignore[arg-type]

    asyncio.run(_run())
