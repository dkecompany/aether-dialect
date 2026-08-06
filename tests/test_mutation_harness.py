"""Fast harness for the mutation-testing script and baseline schema."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_mutation_testing.py"
BASELINE = REPO_ROOT / "scripts" / "mutation_baseline.json"


@pytest.mark.fast
def test_mutation_script_help_and_baseline_schema() -> None:
    """``--help`` works and the baseline JSON matches the expected schema."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "mutation" in proc.stdout.lower()

    assert BASELINE.is_file(), "scripts/mutation_baseline.json must exist"
    doc = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert doc["version"] == 1
    assert doc["tool"] == "mutmut"
    assert doc["status"] in {"operator_run_required", "recorded"}
    targets = doc["targets"]
    assert isinstance(targets, dict)
    expected = {
        "_validation_execute.py",
        "_validation_schema.py",
        "_validation_semantic.py",
        "_intent_resolve.py",
        "_intent_repair.py",
        "_core_utils.py",
    }
    assert set(targets) == expected
    for key in expected:
        assert targets[key] is None or isinstance(targets[key], int)

    check_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--baseline", str(BASELINE)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert check_proc.returncode == 0

    dry_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert dry_proc.returncode == 0
    for module in expected:
        assert module in dry_proc.stdout
