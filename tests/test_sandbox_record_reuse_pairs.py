"""Fast tests for --record-reuse-pairs wiring in sandbox_corpus."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture(scope="module")
def sandbox_corpus():
    return importlib.import_module("sandbox_corpus")


@pytest.mark.fast
def test_record_corpus_forwards_record_reuse_pairs_flag(sandbox_corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}
    session = MagicMock()
    session.paraphrase_seeds_collected = []
    session.failed_slots = []

    def _record_all(**kwargs: object) -> bool:
        recorded.update(kwargs)
        return True

    session.record_all.side_effect = _record_all

    monkeypatch.setattr(sandbox_corpus, "ensure_schema_literals", lambda: None)
    monkeypatch.setattr(sandbox_corpus, "ensure_interpret_domain", lambda: None)
    monkeypatch.setattr(sandbox_corpus, "pin_staging_mock_fixture_keys", lambda: None)
    monkeypatch.setattr(sandbox_corpus, "load_staging_questions", lambda: {})
    monkeypatch.setattr(sandbox_corpus, "prepare_recording_environment", lambda: MagicMock())
    monkeypatch.setattr(sandbox_corpus, "teardown_recording_environment", lambda _env: None)
    monkeypatch.setattr(sandbox_corpus, "WarmRecordingPool", lambda _staging: MagicMock())
    monkeypatch.setattr(sandbox_corpus, "RecordingSession", lambda **_kw: session)
    monkeypatch.setattr(sandbox_corpus, "_staging_sandbox_bundle", nullcontext)
    monkeypatch.setattr(sandbox_corpus, "_load_recording_manifest", lambda: {})
    monkeypatch.setattr(sandbox_corpus, "_missing_paraphrase_canonicals", lambda *_a, **_k: [])
    monkeypatch.setattr(sandbox_corpus, "_reuse_fixtures_ready", lambda _c: True)
    monkeypatch.setattr(sandbox_corpus, "_aetherspace_snapshots_ready", lambda: True)
    monkeypatch.setattr(sandbox_corpus, "_migration_fixtures_ready", lambda _c: True)
    monkeypatch.setattr(sandbox_corpus, "finalize_recording_tail", lambda *_a, **_k: True)
    monkeypatch.setattr(sandbox_corpus, "_recording_pipeline_ready", lambda **_k: (True, []))

    sandbox_corpus.record_corpus(record_reuse_pairs=True)

    assert recorded.get("record_reuse_pairs") is True


@pytest.mark.fast
def test_record_all_records_mapped_reuse_pairs_when_flag_set(sandbox_corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    slot = sandbox_corpus.RecordingSlot(
        tier="questions",
        label="How many rentals happened in 2025?",
        kind="question",
        preset="owner_writer",
        mode="writer",
    )
    session = sandbox_corpus.RecordingSession.__new__(sandbox_corpus.RecordingSession)
    session.failed_slots = []
    session.max_attempts = 3
    session.questions = {}
    session.paraphrase_seeds_collected = []
    session.record_slot = MagicMock(return_value=(True, "", 1, MagicMock()))
    session.record_inline_reuse_param_fixtures = MagicMock(return_value=(True, ""))

    monkeypatch.setattr("aetherdialect.aetherdialect._init_log_sink", lambda _line: None)
    monkeypatch.setattr(sandbox_corpus, "_begin_eval_results", lambda *_a, **_k: None)
    monkeypatch.setattr(sandbox_corpus, "_BUILD_VERBOSE", False)

    ok = session.record_all(slots=[slot], record_reuse_pairs=True)

    assert ok is True
    session.record_inline_reuse_param_fixtures.assert_called_once_with(
        "How many rentals happened in 2025?",
        "How many rentals happened in 2026?",
        swaps=(("2025", "2026"),),
    )


@pytest.mark.fast
def test_record_all_skips_mapped_reuse_pairs_without_flag(sandbox_corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    slot = sandbox_corpus.RecordingSlot(
        tier="questions",
        label="How many rentals happened in 2025?",
        kind="question",
        preset="owner_writer",
        mode="writer",
    )
    session = sandbox_corpus.RecordingSession.__new__(sandbox_corpus.RecordingSession)
    session.failed_slots = []
    session.max_attempts = 3
    session.questions = {}
    session.paraphrase_seeds_collected = []
    session.record_slot = MagicMock(return_value=(True, "", 1, MagicMock()))
    session.record_inline_reuse_param_fixtures = MagicMock(return_value=(True, ""))

    monkeypatch.setattr("aetherdialect.aetherdialect._init_log_sink", lambda _line: None)
    monkeypatch.setattr(sandbox_corpus, "_begin_eval_results", lambda *_a, **_k: None)
    monkeypatch.setattr(sandbox_corpus, "_BUILD_VERBOSE", False)

    ok = session.record_all(slots=[slot], record_reuse_pairs=False)

    assert ok is True
    session.record_inline_reuse_param_fixtures.assert_not_called()
