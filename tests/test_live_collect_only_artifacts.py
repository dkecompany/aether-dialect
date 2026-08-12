"""Collect-only live collection must not allocate results/invoice files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "live_tests"


@pytest.mark.fast
def test_collect_only_does_not_allocate_live_artifacts(tmp_path: Path) -> None:
    before = {p.name for p in LIVE.glob("results*")} | {p.name for p in LIVE.glob("invoice*")}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(LIVE),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = {p.name for p in LIVE.glob("results*")} | {p.name for p in LIVE.glob("invoice*")}
    assert after == before
    _ = tmp_path
