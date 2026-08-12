"""Tests for live/sandbox results and invoice path rotation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.fast
def test_allocate_run_artifact_path_rotates_existing(tmp_path: Path) -> None:
    from sandbox_recording import allocate_run_artifact_path

    base = tmp_path / "results.txt"
    assert allocate_run_artifact_path(base) == base
    base.write_text("keep", encoding="utf-8")
    first = allocate_run_artifact_path(base)
    assert first == tmp_path / "results1.txt"
    first.write_text("keep1", encoding="utf-8")
    second = allocate_run_artifact_path(base)
    assert second == tmp_path / "results2.txt"
    assert base.read_text(encoding="utf-8") == "keep"
    assert first.read_text(encoding="utf-8") == "keep1"


@pytest.mark.fast
def test_allocate_run_artifact_path_invoice_stem(tmp_path: Path) -> None:
    from sandbox_recording import allocate_run_artifact_path

    base = tmp_path / "invoice.txt"
    base.write_text("old", encoding="utf-8")
    assert allocate_run_artifact_path(base) == tmp_path / "invoice1.txt"


@pytest.mark.fast
def test_begin_eval_results_rotates_results_and_invoice(tmp_path: Path) -> None:
    corpus_mod = importlib.import_module("sandbox_corpus")
    import sandbox_recording as recording

    results_base = tmp_path / "sandbox_results.txt"
    invoice_base = tmp_path / "sandbox_invoice.txt"
    results_base.write_text("old-results", encoding="utf-8")
    invoice_base.write_text("old-invoice", encoding="utf-8")
    prev_results = recording.results_file()
    prev_invoice = recording.invoice_path()
    try:
        chosen = corpus_mod._begin_eval_results(results_base, invoice_path=invoice_base)
        assert chosen == tmp_path / "sandbox_results1.txt"
        assert recording.results_file() == chosen
        assert chosen.read_text(encoding="utf-8") == ""
        assert results_base.read_text(encoding="utf-8") == "old-results"
        assert recording.invoice_path() == tmp_path / "sandbox_invoice1.txt"
        assert invoice_base.read_text(encoding="utf-8") == "old-invoice"
    finally:
        recording.set_results_file(prev_results)
        recording.set_invoice_path(prev_invoice)
