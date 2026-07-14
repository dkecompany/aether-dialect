"""Regression tests for sandbox recording isolation."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (_REPO / "env.env").is_file(), reason="requires live LLM credentials for recording simulation")
def test_recording_environment_forces_no_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recording environment must return reuse_type='none' even for repeated questions."""
    sc = importlib.import_module("sandbox_corpus")

    seed_dst = tmp_path / "rental_shop_seed.sql"
    seed_dst.write_text("CREATE TABLE items (id INTEGER, type VARCHAR);", encoding="utf-8")

    env = sc.prepare_recording_environment()
    pool = sc.WarmRecordingPool(tmp_path)

    question = "How many items are in the catalog by item type?"

    try:
        pool.run_live(question)
        step2 = pool.run_live(question)
        codes = {d.code for d in step2.diagnostics}
        assert "REUSE_HIT" not in codes

    finally:
        pool.close()
        sc.teardown_recording_environment(env)
