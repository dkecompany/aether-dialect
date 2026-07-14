"""Unit tests for sandbox corpus post-recording orchestration."""

from __future__ import annotations

import importlib
import sys
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


def test_finalize_recording_tail_runs_reuse_without_paraphrases(sandbox_corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session.record_bundled_paraphrases.return_value = (True, "")
    session.record_reuse_param_fixtures.return_value = (True, "")
    session.record_migration_demo_fixtures.return_value = (True, "")
    corpus = MagicMock()
    corpus.fixtures = []
    pool = MagicMock()
    monkeypatch.setattr(sandbox_corpus, "write_build_fingerprint", lambda: None)

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


def test_paraphrase_seeds_for_missing_reports_unseeded(sandbox_corpus) -> None:
    session = MagicMock()
    session.collect_paraphrase_seeds.return_value = []
    missing = [sandbox_corpus.RecordingSlot(tier="questions", label="How many books do we have?")]

    seeds, detail = sandbox_corpus._paraphrase_seeds_for_missing(session, missing, [])

    assert seeds == []
    assert detail
    assert "How many books do we have?" in detail


def test_collect_paraphrase_seeds_uses_pool_run_mock(sandbox_corpus) -> None:
    pool = MagicMock()
    step = MagicMock()
    step.intent_summary = MagicMock(tables=["item"])
    pool.run_mock.return_value = (step, None)

    class _Session:
        def __init__(self, pool_obj: MagicMock) -> None:
            self.pool = pool_obj

    slots = [sandbox_corpus.RecordingSlot(tier="questions", label="How many books do we have?")]
    seeds = sandbox_corpus.RecordingSession.collect_paraphrase_seeds(_Session(pool), slots)

    assert len(seeds) == 1
    assert seeds[0][0].label == "How many books do we have?"
    pool.run_mock.assert_called_once()


def test_record_inline_reuse_param_fixtures_short_circuits_when_reverse_ready(
    sandbox_corpus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sandbox_corpus.FixtureCorpus(fixtures_path)
    pool = MagicMock()

    class _Session:
        def __init__(self, corpus_obj: object, pool_obj: MagicMock) -> None:
            self.corpus = corpus_obj
            self.pool = pool_obj
            self._llm_mod = MagicMock()

    monkeypatch.setattr(sandbox_corpus, "_reuse_reverse_param_rows_committed", lambda *args, **kwargs: True)

    ok, detail = sandbox_corpus.RecordingSession.record_inline_reuse_param_fixtures(
        _Session(corpus, pool),
        "How many rentals happened in 2025?",
        "How many rentals happened in 2026?",
        swaps=(("2025", "2026"),),
    )

    assert ok is True
    assert detail == ""
    pool._ephemeral_handle.assert_not_called()
