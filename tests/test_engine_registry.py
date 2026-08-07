"""Engine registry must not depend on dialect module import order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aetherdialect._constants import ENGINE_MODULE_PATHS
from aetherdialect._dialect import DialectRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_engine_list_independent_of_import_order() -> None:
    static_engines = set(ENGINE_MODULE_PATHS)
    listed = set(DialectRegistry.list_engines())
    assert static_engines.issubset(listed)

    script = """
import importlib
import sys

from aetherdialect._constants import ENGINE_MODULE_PATHS

for name in list(sys.modules):
    if name == "aetherdialect" or name.startswith("aetherdialect."):
        del sys.modules[name]
importlib.invalidate_caches()
importlib.import_module("aetherdialect._dialect_postgres")
from aetherdialect._dialect import DialectRegistry
engines = set(DialectRegistry.list_engines())
static = set(ENGINE_MODULE_PATHS)
assert static.issubset(engines), sorted(static - engines)
print(",".join(sorted(static)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip().split(",") == sorted(ENGINE_MODULE_PATHS)
