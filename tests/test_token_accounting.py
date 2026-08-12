"""Per-turn LLM token accounting on terminal session steps."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_LLM_TURN_COST
from aetherdialect._contracts_core import LlmTurnUsageSummary
from aetherdialect._main_session import PipelineSession
from aetherdialect._templates_ops import TemplateOps


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._sandbox_closed = False
    owner._artifacts_dir = None
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = None
    owner._llm_config = MagicMock(provider="openai")
    owner._pipeline_writer_lock = None
    return owner


@pytest.mark.fast
def test_turn_reports_usage() -> None:
    """Terminal steps carry structured LLM usage totals for the turn."""
    from aetherdialect._utils import record_llm_usage

    owner = _session_owner()
    session = PipelineSession(owner, mode="writer")

    def run_with_usage(*_args: object, **_kwargs: object) -> None:
        record_llm_usage(
            task="intent",
            logical_model="gpt-5.4-mini",
            api_model="gpt-5.4-mini",
            provider="openai",
            input_tokens=120,
            cached_input_tokens=20,
            output_tokens=30,
            cache_write_tokens=None,
            attempt=1,
            elapsed_ms=5,
        )
        session.note_turn_outcome(
            outcome="success",
            sql="SELECT 1",
            rows=[(1,)],
            columns=("x",),
        )

    with patch("aetherdialect._main_session.llm_execution_scope", lambda *_a, **_k: nullcontext()):
        with patch("aetherdialect._main_init.MainInitOps.interactive_run_once", side_effect=run_with_usage):
            step = session.ask("how many rows")

    assert step.done
    assert step.llm_usage is not None
    assert isinstance(step.llm_usage, LlmTurnUsageSummary)
    assert step.llm_usage.request_count == 1
    assert step.llm_usage.input_tokens == 120
    assert step.llm_usage.cached_input_tokens == 20
    assert step.llm_usage.output_tokens == 30
    assert step.llm_usage.cost_usd is not None

    cost_rows = [d for d in step.diagnostics if d.code == DIAGNOSTIC_CODE_LLM_TURN_COST]
    assert cost_rows, "diagnostics should still carry LLM_TURN_COST"
