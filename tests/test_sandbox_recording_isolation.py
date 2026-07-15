"""Regression tests for sandbox recording isolation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.skipif(not (_REPO / "env.env").is_file(), reason="requires live LLM credentials for recording simulation")
def test_recording_environment_forces_no_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recording environment must return reuse_type='none' even for repeated questions."""
    # This test simulates what scripts/sandbox_corpus.py does
    sc = importlib.import_module("sandbox_corpus")

    # Setup staging files in tmp_path
    seed_dst = tmp_path / "rental_shop_seed.sql"
    seed_dst.write_text("CREATE TABLE items (id INTEGER, type VARCHAR);", encoding="utf-8")

    env = sc.prepare_recording_environment()
    pool = sc.WarmRecordingPool(tmp_path)

    question = "How many items are in the catalog by item type?"

    try:
        # First ask
        pool.run_live(question)

        # Second ask of the same question in the same process
        step2 = pool.run_live(question)

        # If reuse was enabled, step2 would have reuse_type != 'none'
        # We check diagnostics for REUSE_HIT
        codes = {d.code for d in step2.diagnostics}
        assert "REUSE_HIT" not in codes

    finally:
        pool.close()
        sc.teardown_recording_environment(env)
