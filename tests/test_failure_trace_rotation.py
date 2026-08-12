"""Tests for failure-trace file rotation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_core import StepResult
from aetherdialect._utils_artifacts import append_failure_trace


def _fill_to_threshold(path: Path, *, threshold: int, pad_char: str = "x") -> str:
    content = pad_char * threshold
    path.write_text(content, encoding="utf-8")
    assert path.stat().st_size == threshold
    return content


def _failure_step(question: str, *, error: str = "boom") -> StepResult:
    return StepResult(
        scenario_id="slot-1",
        question=question,
        status="failed",
        error=error,
    )


@pytest.mark.fast
def test_trace_rotates_at_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    threshold = 64
    monkeypatch.setattr("aetherdialect._utils_artifacts.FAILURE_TRACE_ROTATE_BYTES", threshold)

    trace_path = tmp_path / "results.txt"
    rotated_path = tmp_path / "results.txt.1"

    old_content = _fill_to_threshold(trace_path, threshold=threshold)
    append_failure_trace(_failure_step("first rotation"), trace_path)

    assert rotated_path.is_file()
    assert rotated_path.read_text(encoding="utf-8") == old_content

    new_text = trace_path.read_text(encoding="utf-8")
    assert old_content not in new_text
    assert "question: first rotation" in new_text
    assert "boom" in new_text
    assert "=" * 80 not in new_text

    mid_content = trace_path.read_text(encoding="utf-8")
    assert trace_path.stat().st_size >= threshold

    append_failure_trace(_failure_step("second rotation", error="again"), trace_path)

    assert rotated_path.is_file()
    assert not (tmp_path / "results.txt.2").exists()
    assert rotated_path.read_text(encoding="utf-8") == mid_content

    latest = trace_path.read_text(encoding="utf-8")
    assert "question: second rotation" in latest
    assert "again" in latest
