"""Artifact-path tests: results.csv must not depend on process cwd."""

from __future__ import annotations

import os

import pandas

import aetherdialect._pipeline
from aetherdialect._core_utils import pipeline_capture


def test_save_result_csv_writes_to_explicit_path_not_cwd(tmp_path, monkeypatch):
    """Explicit output_path wins; cwd is not used for the write location."""
    wrong_dir = tmp_path / "wrong_cwd"
    wrong_dir.mkdir()
    explicit_dir = tmp_path / "explicit_out"
    explicit_dir.mkdir()
    monkeypatch.chdir(wrong_dir)

    df = pandas.DataFrame({"col": [1, 2]})
    dest = explicit_dir / "results.csv"
    aetherdialect._pipeline.save_result_csv(df, output_path=dest)

    assert dest.exists()
    assert not (wrong_dir / "results.csv").exists()


def test_pipeline_capture_does_not_chdir(tmp_path, monkeypatch):
    """pipeline_capture(csv_dir=...) must not call os.chdir."""
    csv_dir = tmp_path / "csv_out"
    csv_dir.mkdir()
    wrong_dir = tmp_path / "wrong_cwd"
    wrong_dir.mkdir()
    monkeypatch.chdir(wrong_dir)

    chdir_calls: list[str] = []
    real_chdir = os.chdir

    def tracking_chdir(path: str | os.PathLike[str]) -> None:
        chdir_calls.append(os.fspath(path))
        real_chdir(path)

    monkeypatch.setattr(os, "chdir", tracking_chdir)

    with pipeline_capture(["y"], csv_dir=str(csv_dir)):
        aetherdialect._pipeline.save_result_csv(pandas.DataFrame({"a": [1]}))

    assert chdir_calls == []
    assert (csv_dir / "results.csv").exists()
    assert not (wrong_dir / "results.csv").exists()
