"""Tests for LLM usage accounting and cost reporting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import DIAGNOSTIC_CODE_LLM_TURN_COST
from aetherdialect._contracts_core import LlmUsageRecord
from aetherdialect._llm_provider import LLMProvider
from aetherdialect._utils import (
    llm_call_audit_details,
    llm_call_cost_usd,
    llm_turn_cost_diagnostic,
    llm_usage_build_scope,
    llm_usage_question_scope,
    llm_usage_run_scope,
    llm_usage_session_scope,
    record_llm_usage,
    reset_llm_usage_accumulator,
    set_llm_price_table_override,
    snapshot_llm_usage_records,
)


@pytest.fixture(autouse=True)
def _reset_price_override() -> None:
    reset_llm_usage_accumulator()
    set_llm_price_table_override(None)
    yield
    reset_llm_usage_accumulator()
    set_llm_price_table_override(None)


def test_record_llm_usage_noop_without_accumulator() -> None:
    record_llm_usage(
        task="intent",
        logical_model="gpt-5.4-mini",
        api_model="gpt-5.4-mini",
        provider="openai",
        input_tokens=100,
        cached_input_tokens=10,
        output_tokens=20,
        cache_write_tokens=None,
        attempt=1,
        elapsed_ms=50,
    )
    assert snapshot_llm_usage_records() == ()


def test_session_scope_records_and_scopes() -> None:
    with llm_usage_session_scope():
        with llm_usage_build_scope():
            record_llm_usage(
                task="schema",
                logical_model="gpt-5-mini",
                api_model="gpt-5-mini",
                provider="openai",
                input_tokens=1000,
                cached_input_tokens=0,
                output_tokens=200,
                cache_write_tokens=None,
                attempt=1,
                elapsed_ms=10,
            )
        with llm_usage_question_scope():
            record_llm_usage(
                task="intent",
                logical_model="gpt-5.4-mini",
                api_model="gpt-5.4-mini",
                provider="openai",
                input_tokens=500,
                cached_input_tokens=100,
                output_tokens=50,
                cache_write_tokens=None,
                attempt=1,
                elapsed_ms=20,
            )
    records = snapshot_llm_usage_records()
    assert len(records) == 2
    assert records[0].scope == "build"
    assert records[1].scope == "question"
    assert records[1].cached_input_tokens == 100


def test_run_scope_records() -> None:
    with llm_usage_session_scope():
        with llm_usage_run_scope():
            record_llm_usage(
                task="intent",
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
    records = snapshot_llm_usage_records()
    assert len(records) == 1
    assert records[0].scope == "run"


def test_openai_cost_uses_cached_tokens() -> None:
    record = LlmUsageRecord(
        scope="question",
        block_id=1,
        task="intent",
        logical_model="gpt-5.4-mini",
        api_model="gpt-5.4-mini",
        provider="openai",
        input_tokens=1000,
        cached_input_tokens=400,
        output_tokens=100,
        cache_write_tokens=None,
        attempt=1,
        elapsed_ms=1,
    )
    cost = llm_call_cost_usd(record)
    assert cost is not None
    assert cost > 0


def test_azure_records_have_no_cost() -> None:
    record = LlmUsageRecord(
        scope="question",
        block_id=1,
        task="intent",
        logical_model="gpt-5.4-mini",
        api_model="dep-heavy",
        provider="azure",
        input_tokens=1000,
        cached_input_tokens=100,
        output_tokens=100,
        cache_write_tokens=None,
        attempt=1,
        elapsed_ms=1,
    )
    assert llm_call_cost_usd(record) is None
    diag = llm_turn_cost_diagnostic((record,), provider="azure")
    assert diag is not None
    assert "$" not in diag.message
    assert diag.code == DIAGNOSTIC_CODE_LLM_TURN_COST


def test_turn_diagnostic_reports_unpriced_models() -> None:
    record = LlmUsageRecord(
        scope="question",
        block_id=1,
        task="default",
        logical_model="unknown-model",
        api_model="unknown-model",
        provider="openai",
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        cache_write_tokens=None,
        attempt=1,
        elapsed_ms=1,
    )
    diag = llm_turn_cost_diagnostic((record,), provider="openai")
    assert diag is not None
    assert "unpriced_models" in dict(diag.details)


def test_llm_call_audit_details_include_cost_for_openai() -> None:
    record = LlmUsageRecord(
        scope="question",
        block_id=1,
        task="intent",
        logical_model="gpt-5.4-mini",
        api_model="gpt-5.4-mini",
        provider="openai",
        input_tokens=1000,
        cached_input_tokens=0,
        output_tokens=100,
        cache_write_tokens=None,
        attempt=1,
        elapsed_ms=1,
    )
    details = dict(llm_call_audit_details(record))
    assert details["task"] == "intent"
    assert "cost_usd" in details


def test_llm_chat_records_usage_on_success() -> None:
    usage = MagicMock()
    usage.input_tokens = 12
    usage.output_tokens = 3
    usage.input_tokens_details = MagicMock(cached_tokens=4)
    mock_resp = MagicMock()
    mock_resp.output_text = '{"ok": true}'
    mock_resp.usage = usage
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp

    prev = EngineConfig.LLM_PROVIDER
    EngineConfig.LLM_PROVIDER = "openai"
    try:
        with llm_usage_session_scope():
            with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
                with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
                    with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=mock_client):
                        with patch("aetherdialect._utils.debug"):
                            with patch("aetherdialect._utils.pipeline_trace"):
                                LLMProvider.chat("sys", "usr", max_retries=1, task="intent")
    finally:
        EngineConfig.LLM_PROVIDER = prev

    records = snapshot_llm_usage_records()
    assert len(records) == 1
    record = records[-1]
    assert record.input_tokens == 12
    assert record.cached_input_tokens == 4
    assert record.output_tokens == 3
    assert record.cache_write_tokens == 0


@pytest.mark.fast
def test_llm_chat_records_cache_write_tokens_when_present() -> None:
    usage = MagicMock()
    usage.input_tokens = 20
    usage.output_tokens = 2
    usage.input_tokens_details = MagicMock(cached_tokens=5, cache_write_tokens=7)
    mock_resp = MagicMock()
    mock_resp.output_text = '{"ok": true}'
    mock_resp.usage = usage
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp

    prev = EngineConfig.LLM_PROVIDER
    EngineConfig.LLM_PROVIDER = "openai"
    try:
        with llm_usage_session_scope():
            with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
                with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
                    with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=mock_client):
                        with patch("aetherdialect._utils.debug"):
                            with patch("aetherdialect._utils.pipeline_trace"):
                                LLMProvider.chat("sys", "usr", max_retries=1, task="intent")
    finally:
        EngineConfig.LLM_PROVIDER = prev

    record = snapshot_llm_usage_records()[-1]
    assert record.cache_write_tokens == 7


@pytest.mark.fast
def test_record_llm_usage_sets_turn_id_when_active() -> None:
    from aetherdialect._utils import pop_turn_id, push_turn_id

    with llm_usage_session_scope():
        turn_id = "turn-abc-123"
        token = push_turn_id(turn_id)
        try:
            record_llm_usage(
                task="intent",
                logical_model="gpt-5.4-mini",
                api_model="gpt-5.4-mini",
                provider="openai",
                input_tokens=1,
                cached_input_tokens=0,
                output_tokens=1,
                cache_write_tokens=0,
                attempt=1,
                elapsed_ms=1,
            )
        finally:
            pop_turn_id(token)
    record = snapshot_llm_usage_records()[-1]
    assert record.turn_id == turn_id


def test_invoice_writer_groups_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sandbox_recording as _invoice

    invoice_file = tmp_path / "invoice.txt"
    monkeypatch.setattr(_invoice, "_INVOICE_PATH", invoice_file)

    records = (
        LlmUsageRecord(
            scope="build",
            block_id=1,
            task="schema",
            logical_model="gpt-5-mini",
            api_model="gpt-5-mini",
            provider="openai",
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=10,
            cache_write_tokens=None,
            attempt=1,
            elapsed_ms=1,
        ),
        LlmUsageRecord(
            scope="question",
            block_id=2,
            task="intent",
            logical_model="gpt-5.4-mini",
            api_model="gpt-5.4-mini",
            provider="openai",
            input_tokens=50,
            cached_input_tokens=5,
            output_tokens=5,
            cache_write_tokens=None,
            attempt=1,
            elapsed_ms=1,
        ),
    )
    _invoice.write_invoice_file(records)
    text = invoice_file.read_text(encoding="utf-8")
    assert "[build_1]" in text
    assert "[question_1]" in text
    assert "[run_total]" in text
    assert "questions=1" in text
    assert "note=reported totals are a floor" in text


def test_nested_build_scope_wins_over_turn_question() -> None:
    from aetherdialect._utils import reset_turn_llm_scope, set_turn_llm_scope

    with llm_usage_session_scope():
        tok = set_turn_llm_scope("question")
        try:
            with llm_usage_build_scope():
                record_llm_usage(
                    task="schema",
                    logical_model="gpt-5-mini",
                    api_model="gpt-5-mini",
                    provider="openai",
                    input_tokens=10,
                    cached_input_tokens=0,
                    output_tokens=1,
                    cache_write_tokens=None,
                    attempt=1,
                    elapsed_ms=1,
                )
            record_llm_usage(
                task="intent_compose",
                logical_model="gpt-5.4-mini",
                api_model="gpt-5.4-mini",
                provider="openai",
                input_tokens=20,
                cached_input_tokens=0,
                output_tokens=2,
                cache_write_tokens=None,
                attempt=1,
                elapsed_ms=1,
            )
        finally:
            reset_turn_llm_scope(tok)
    rows = snapshot_llm_usage_records()
    assert rows[0].scope == "build"
    assert rows[1].scope == "question"


def test_prompt_cache_key_always_emitted_and_stage_distinct() -> None:
    intake = LLMProvider.resolve_prompt_cache_key("intake_validate")
    compose = LLMProvider.resolve_prompt_cache_key("intent_compose")
    interpret = LLMProvider.resolve_prompt_cache_key("intent_interpret")
    assert intake is not None
    assert compose is not None
    assert interpret is not None
    assert intake != compose
    assert interpret != compose
