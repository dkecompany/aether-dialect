"""Per-run QSim summary artifact layout."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_schema import QSimSummary
from aetherdialect._main_execution import MainExecutionOps


def _minimal_schema() -> MagicMock:
    schema = MagicMock()
    table = MagicMock()
    col = MagicMock()
    col.role = "dimension"
    table.columns = {"id": col}
    schema.tables = {"t": table}
    return schema


@pytest.mark.fast
def test_each_run_writes_its_own_summary(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    schema = _minimal_schema()
    qsim_dir = tmp_path / "qsim"

    with (
        patch("aetherdialect._main_interactive.generate_all_intents", return_value=[]),
        patch("aetherdialect._main_interactive.instantiate_all", return_value=[]),
        patch("aetherdialect._main_interactive.generate_all_questions", return_value=[]),
    ):
        MainExecutionOps.qsim_run_once(artifacts_dir=artifacts_dir, schema=schema, seed=1)
        MainExecutionOps.qsim_run_once(artifacts_dir=artifacts_dir, schema=schema, seed=2)

    assert (qsim_dir / "summary_1.json").is_file()
    assert (qsim_dir / "summary_2.json").is_file()
    assert not (tmp_path / "qsim_summary.json").exists()

    index_lines = (qsim_dir / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 2
    for line in index_lines:
        payload = json.loads(line)
        assert "run_id" in payload
        assert "timestamp" in payload

    summaries = MainExecutionOps.load_qsim_summaries(artifacts_dir)
    assert len(summaries) == 2
    assert all(isinstance(item, QSimSummary) for item in summaries)
    assert {item.version for item in summaries} == {1, 2}

    first_payload = json.loads((qsim_dir / "summary_1.json").read_text(encoding="utf-8"))
    assert isinstance(first_payload, dict)
    assert "version" in first_payload
