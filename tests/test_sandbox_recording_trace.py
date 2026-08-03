"""Tests for sandbox recording trace output."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._core_utils import StepResult


def test_recording_max_attempts_is_two() -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")
    assert corpus_mod.RECORDING_MAX_ATTEMPTS == 2


def test_clear_results_file_is_empty(tmp_path: Path) -> None:
    import live_tests.conftest as lt_conftest

    results_path = tmp_path / "recording_results.txt"
    prev_results_file = lt_conftest._RESULTS_FILE
    try:
        lt_conftest._RESULTS_FILE = results_path
        lt_conftest._clear_results_file()
        assert results_path.read_text(encoding="utf-8") == ""
    finally:
        lt_conftest._RESULTS_FILE = prev_results_file


def test_failure_trace_append_writes_trace_only(tmp_path: Path) -> None:
    import live_tests.conftest as lt_conftest

    corpus_mod = importlib.import_module("sandbox_corpus")

    results_path = tmp_path / "recording_results.txt"
    prev_results_file = lt_conftest._RESULTS_FILE
    saved_debug = PolicyConfig.DEBUG
    try:
        corpus_mod._begin_eval_results(results_path)
        step = StepResult(
            scenario_id="slot-1",
            question="how many films",
            status="failed",
            error="validation failed",
            captured_logs=["[DEBUG] intent_parse started", "[PIPELINE_TRACE] intent\n{}"],
            duration_seconds=1.25,
        )
        lt_conftest._append_failure_trace(step)
    finally:
        lt_conftest._RESULTS_FILE = prev_results_file

    text = results_path.read_text(encoding="utf-8")
    assert "Live Test Results" not in text
    assert "ALL RESULTS" not in text
    assert "question: how many films" in text
    assert "llm_calls:" in text
    assert "[DEBUG] intent_parse started" in text
    assert "[PIPELINE_TRACE]" in text
    assert PolicyConfig.DEBUG == saved_debug


def test_pass_does_not_append(tmp_path: Path) -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    results_path = tmp_path / "recording_results.txt"
    corpus_mod._begin_eval_results(results_path)
    assert results_path.read_text(encoding="utf-8") == ""


def test_slot_id_for_question_and_feedback() -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    assert (
        corpus_mod.slot_id_for(corpus_mod.RecordingSlot(tier="questions", label="How many books do we have?"))
        == "owner:writer:How many books do we have?"
    )
    assert (
        corpus_mod.slot_id_for(corpus_mod.RecordingSlot(tier="feedback", label="bad sql", kind="feedback"))
        == "feedback:bad sql"
    )


def test_recording_manifest_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    monkeypatch.setattr(corpus_mod, "STAGING", tmp_path)
    monkeypatch.setattr(corpus_mod, "RECORDING_MANIFEST_PATH", tmp_path / "recording_manifest.json")
    rows = [{"slot_id": "owner:writer:Q1", "committed": True, "attempts": 1, "detail": ""}]
    corpus_mod.write_recording_manifest(rows)
    loaded = corpus_mod._load_recording_manifest()
    assert loaded["slots"] == rows
    assert corpus_mod.committed_slot_ids(loaded) == {"owner:writer:Q1"}


def test_upsert_manifest_row_preserves_existing_entries() -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    rows = [
        {"slot_id": "owner:writer:Q1", "committed": True, "label": "Q1"},
        {"slot_id": "owner:writer:Q2", "committed": True, "label": "Q2"},
    ]
    corpus_mod._upsert_manifest_row(
        rows,
        {"slot_id": "owner:writer:Q2", "committed": False, "label": "Q2", "detail": "retry"},
    )
    assert len(rows) == 2
    assert rows[0]["committed"] is True
    assert rows[1]["committed"] is False
    assert rows[1]["detail"] == "retry"
    corpus_mod._upsert_manifest_row(
        rows,
        {"slot_id": "owner:writer:Q3", "committed": True, "label": "Q3"},
    )
    assert len(rows) == 3


def test_clean_generated_paraphrases_applies_copy_rules_and_dedupes() -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    out = corpus_mod._clean_generated_paraphrases(
        "How many rentals happened in 2025?",
        ["How many rentals happened in 2026?", "How many rentals happened in 2026?"],
    )
    assert out == ["How many rentals happened in 2026?"]


def test_build_reverse_param_extraction_rows_swaps_years() -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    class StubLLM:
        @staticmethod
        def mock_fixture_user_key(user: str) -> str:
            return user

    rows = [
        {
            "task": "default",
            "system": "You are a deterministic parameter value extractor for text-to-SQL.",
            "user": '{"matched_question":"How many rentals happened in 2025?","question":"How many rentals happened in 2026?"}',
            "output_text": '{"param_values":{"p1":2026}}',
        }
    ]
    out = corpus_mod._build_reverse_param_extraction_rows(rows, swaps=(("2025", "2026"),), llm_mod=StubLLM())
    assert out == [
        {
            "task": "default",
            "system": "You are a deterministic parameter value extractor for text-to-SQL.",
            "user": '{"matched_question":"How many rentals happened in 2026?","question":"How many rentals happened in 2025?"}',
            "output_text": '{"param_values":{"p1":2025}}',
        }
    ]


def test_param_extraction_forward_rows_from_corpus_matches_canonical() -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    rows = [
        {
            "task": "default",
            "system": "You are a deterministic parameter value extractor for text-to-SQL.",
            "user": '{"matched_question":"How many rentals happened in 2025?","question":"How many rentals happened in 2026?"}',
            "output_text": "{}",
        },
        {
            "task": "default",
            "system": "You are a deterministic parameter value extractor for text-to-SQL.",
            "user": '{"matched_question":"How many rentals happened in 2026?","question":"How many rentals happened in 2025?"}',
            "output_text": "{}",
        },
        {"task": "default", "system": "other", "user": "How many rentals happened in 2025?", "output_text": "{}"},
    ]
    out = corpus_mod._param_extraction_forward_rows_from_corpus(rows, "How many rentals happened in 2025?")
    assert len(out) == 2
    assert all("2025" in row["user"] for row in out)


def test_reuse_reverse_param_rows_committed_detects_existing_keys(tmp_path: Path) -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")

    class StubLLM:
        @staticmethod
        def mock_fixture_user_key(user: str) -> str:
            return user

    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = corpus_mod.FixtureCorpus(fixtures_path)
    forward = [
        {
            "task": "default",
            "system": "You are a deterministic parameter value extractor for text-to-SQL.",
            "user": '{"matched_question":"How many rentals happened in 2025?","question":"How many rentals happened in 2026?"}',
            "output_text": '{"param_values":{"p1":2026}}',
        }
    ]
    reverse = corpus_mod._build_reverse_param_extraction_rows(
        forward,
        swaps=(("2025", "2026"),),
        llm_mod=StubLLM(),
    )
    assert reverse
    assert (
        corpus_mod._reuse_reverse_param_rows_committed(
            corpus,
            forward_rows=forward,
            swaps=(("2025", "2026"),),
            llm_mod=StubLLM(),
        )
        is False
    )
    for row in reverse:
        corpus.fixtures.append(row)
        corpus.seen.add(corpus_mod.fixture_key(row))
    assert (
        corpus_mod._reuse_reverse_param_rows_committed(
            corpus,
            forward_rows=forward,
            swaps=(("2025", "2026"),),
            llm_mod=StubLLM(),
        )
        is True
    )
