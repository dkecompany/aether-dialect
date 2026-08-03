"""Migration maps and qsim bundle output resolve under artifacts_dir, not process cwd."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import _main_execution
from aetherdialect._constants import MIGRATION_MAP_FILENAME, QSIM_QUESTIONS_PATTERN


@pytest.mark.fast
def test_schema_migration_map_read_from_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """initialize_aether_engine consults schema_migration_map.json under adir, not cwd."""
    monkeypatch.chdir(tmp_path)
    adir = tmp_path / "engine_store"
    adir.mkdir()
    (adir / MIGRATION_MAP_FILENAME).write_text(
        '{"action":"abort","version":1,"refresh_existing_descriptions_on_addition":false}',
        encoding="utf-8",
    )
    assert not (tmp_path / MIGRATION_MAP_FILENAME).exists()

    source = inspect.getsource(_main_execution.initialize_aether_engine)
    assert "cwd_root / MIGRATION_MAP_FILENAME" not in source
    assert "Path(adir)" in source

    observed: list[str] = []
    real_load = _main_execution.load_schema_migration_map

    def spy_load(path: Path):
        observed.append(str(Path(path).resolve()))
        return real_load(path)

    monkeypatch.setattr(_main_execution, "load_schema_migration_map", spy_load)

    artifacts_root = Path(str(adir))
    map_path = artifacts_root / MIGRATION_MAP_FILENAME
    if map_path.is_file():
        _main_execution.load_schema_migration_map(artifacts_root)

    assert observed == [str(adir.resolve())]


@pytest.mark.fast
def test_print_questions_bundle_writes_under_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """print_questions_bundle mirrors questions beside qsim artifacts, not in cwd."""
    monkeypatch.chdir(tmp_path)
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    qsim_name = QSIM_QUESTIONS_PATTERN.format(version=2)
    qsim_path = os.path.join(artifacts_dir, qsim_name)
    with open(qsim_path, "w", encoding="utf-8") as fh:
        fh.write("1. How many rows?\n")

    with patch("aetherdialect._main_execution.notify"):
        _main_execution.print_questions_bundle(2, artifacts_dir)

    expected = os.path.join(artifacts_dir, "qsim_v2_questions.txt")
    assert os.path.isfile(expected)
    assert not os.path.isfile(os.path.join(str(tmp_path), "qsim_v2_questions.txt"))
