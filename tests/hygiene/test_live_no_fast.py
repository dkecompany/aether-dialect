"""live_tests must never receive the fast marker after collection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIVE = _ROOT / "live_tests"
_HELPER = Path(__file__).resolve().parent / "_assert_live_no_fast.py"


@pytest.mark.hygiene
@pytest.mark.fast
def test_live_tests_never_receive_fast_after_collection() -> None:
    """Tests/conftest auto-marks non-live items; live items must stay unmarked."""
    if not _LIVE.is_dir():
        pytest.skip("live_tests directory not present")

    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    proc = subprocess.run(
        [sys.executable, str(_HELPER), str(_ROOT)],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
    assert "OK" in (proc.stdout or ""), (proc.stdout or "") + (proc.stderr or "")
