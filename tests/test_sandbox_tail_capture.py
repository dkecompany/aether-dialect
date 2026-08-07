"""Unit tests for sandbox corpus post-recording orchestration."""

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


def test_finalize_recording_tail_runs_reuse_without_paraphrases(
    sandbox_corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock()
    session.record_bundled_paraphrases.return_value = (True, "")
    session.record_reuse_param_fixtures.return_value = (True, "")
    session.record_migration_demo_fixtures.return_value = (True, "")
    corpus = MagicMock()
    corpus.fixtures = []
    pool = MagicMock()
    monkeypatch.setattr(sandbox_corpus, "write_build_fingerprint", lambda: None)
    monkeypatch.setattr(sandbox_corpus, "_finalize_fixture_corpus_repair", lambda _corpus: None)

    ok = sandbox_corpus.finalize_recording_tail(
        session,
        pool,
        corpus,
        capture_paraphrases=False,
        capture_reuse=True,
    )

    assert ok is True
    session.record_reuse_param_fixtures.assert_called_once()
    session.record_bundled_paraphrases.assert_not_called()


def test_finalize_recording_tail_runs_paraphrase_and_reuse_together(
    sandbox_corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.record_bundled_paraphrases.return_value = (True, "")
    session.record_reuse_param_fixtures.return_value = (True, "")
    session.record_migration_demo_fixtures.return_value = (True, "")
    corpus = MagicMock()
    corpus.fixtures = []
    pool = MagicMock()
    monkeypatch.setattr(sandbox_corpus, "write_build_fingerprint", lambda: None)
    monkeypatch.setattr(sandbox_corpus, "_finalize_fixture_corpus_repair", lambda _corpus: None)

    ok = sandbox_corpus.finalize_recording_tail(
        session,
        pool,
        corpus,
        paraphrase_seeds=[(MagicMock(), MagicMock())],
        capture_paraphrases=True,
        capture_reuse=True,
    )

    assert ok is True
    session.record_bundled_paraphrases.assert_called_once()
    session.record_reuse_param_fixtures.assert_called_once()


def test_paraphrase_seeds_for_missing_collects_live_when_not_in_collected(sandbox_corpus) -> None:
    session = MagicMock()
    session.collect_paraphrase_seeds.return_value = []
    missing = [sandbox_corpus.RecordingSlot(tier="questions", label="How many books do we have?")]

    seeds = sandbox_corpus._paraphrase_seeds_for_missing(session, missing, [])

    assert seeds == []
    session.collect_paraphrase_seeds.assert_called_once_with(missing)


def test_collect_paraphrase_seeds_uses_run_live_slot(
    sandbox_corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.corpus = MagicMock()
    step = MagicMock()
    session._run_live_slot.return_value = (step, "")
    monkeypatch.setattr(sandbox_corpus, "_check_slot_recording", lambda _step, _slot: (True, ""))

    slots = [sandbox_corpus.RecordingSlot(tier="questions", label="How many books do we have?")]
    seeds = sandbox_corpus.RecordingSession.collect_paraphrase_seeds(session, slots)

    assert len(seeds) == 1
    assert seeds[0][0].label == "How many books do we have?"
    session._run_live_slot.assert_called_once()


def test_record_inline_reuse_param_fixtures_short_circuits_when_reverse_ready(
    sandbox_corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = MagicMock()
    corpus.collapsed_slot_fixtures.return_value = [{"task": "param_extraction"}]
    pool = MagicMock()
    handle = MagicMock()
    pool.live_handle.return_value = handle
    engine_session = MagicMock()
    engine_session.accept_until_done.return_value = MagicMock(error=None)
    handle.engine.session.return_value = nullcontext(engine_session)

    session = MagicMock()
    session.corpus = corpus
    session.pool = pool
    session._llm_mod = MagicMock()
    session._template_match_enabled.return_value = nullcontext()

    monkeypatch.setattr(sandbox_corpus, "_is_param_extraction_fixture_row", lambda _row: True)
    monkeypatch.setattr(sandbox_corpus, "_reuse_reverse_param_rows_committed", lambda *args, **kwargs: True)

    ok, detail = sandbox_corpus.RecordingSession.record_inline_reuse_param_fixtures(
        session,
        "How many rentals happened in 2025?",
        "How many rentals happened in 2026?",
        swaps=(("2025", "2026"),),
    )

    assert ok is True
    assert detail == ""
    pool._ephemeral_handle.assert_not_called()
