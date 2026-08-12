"""LLM usage turn cursor isolates asks without wiping the session accumulator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._main_session import PipelineSession
from aetherdialect._utils import (
    llm_usage_session_scope,
    record_llm_usage,
    reset_llm_usage_accumulator,
    snapshot_llm_usage_records,
)


@pytest.fixture(autouse=True)
def _reset_usage() -> None:
    reset_llm_usage_accumulator()
    yield
    reset_llm_usage_accumulator()


def _record_question_usage(*, task: str = "intent") -> None:
    record_llm_usage(
        task=task,
        logical_model="gpt-5.4-mini",
        api_model="gpt-5.4-mini",
        provider="openai",
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        cache_write_tokens=None,
        attempt=1,
        elapsed_ms=1,
    )


@pytest.mark.fast
def test_two_asks_do_not_double_count_buffer() -> None:
    owner = MagicMock()
    owner._llm_config = MagicMock(provider="openai")
    owner._audit_emit = MagicMock()
    owner._schema_graph = MagicMock(effective_structural_hash="h")
    sess = PipelineSession(owner, mode="writer")

    with llm_usage_session_scope():
        _record_question_usage(task="intent_pass_1")
        sess._turn_llm_usage_start = 0
        sess._emit_turn_llm_usage(question="first ask", diagnostics=())

        # Cursor advanced; prior ask remains in the session buffer for invoice flush.
        assert len(snapshot_llm_usage_records()) == 1
        sess._turn_llm_usage_start = len(snapshot_llm_usage_records())
        _record_question_usage(task="intent_pass_2")
        turn_records = sess._turn_llm_usage_records()

    assert len(turn_records) == 1
    assert turn_records[0].task == "intent_pass_2"
    assert len(snapshot_llm_usage_records()) == 2
