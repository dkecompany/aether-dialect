"""Import cost: plain package import must not load heavy optional modules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_HEAVY_MODULES = (
    "aetherdialect._sandbox",
    "aetherdialect._seed_warmup",
    "aetherdialect._qsim",
    "aetherdialect._dialect_postgres",
    "aetherdialect._dialect_sqlglot_engines",
)


def _run_isolated_script(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.fast
def test_heavy_modules_not_loaded_by_plain_import() -> None:
    heavy_modules_repr = repr(_HEAVY_MODULES)
    script = f"""
import importlib
import sys

_HEAVY_MODULES = {heavy_modules_repr}

for name in list(sys.modules):
    if name == "aetherdialect" or name.startswith("aetherdialect."):
        del sys.modules[name]
importlib.invalidate_caches()
importlib.import_module("aetherdialect")

loaded = [name for name in _HEAVY_MODULES if name in sys.modules]
if loaded:
    print(f"plain import must not load heavy modules; loaded: {{loaded}}")
    sys.exit(1)
"""
    _run_isolated_script(script)
