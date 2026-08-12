"""Optional embedded federation coordinator driver imports."""

from __future__ import annotations

import ast
import builtins
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import ConfigError
from aetherdialect.aetherdialect import AetherFederation

REPO_ROOT = Path(__file__).resolve().parents[1]
COORDINATOR_MODULES = (
    REPO_ROOT / "src" / "aetherdialect" / "_federation_execute.py",
    REPO_ROOT / "src" / "aetherdialect" / "_dialect_sqlglot_engines.py",
)


def _module_has_top_level_duckdb_import(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] == "duckdb" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] == "duckdb":
            return True
    return False


@pytest.mark.fast
def test_import_without_coordinator_driver_succeeds() -> None:
    for path in COORDINATOR_MODULES:
        assert not _module_has_top_level_duckdb_import(path), f"top-level duckdb import remains in {path.name}"

    script = """
import importlib
import sys

real_import = __import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root == "duckdb":
        raise ImportError("blocked duckdb")
    return real_import(name, globals, locals, fromlist, level)

__builtins__.__import__ = blocked
importlib.invalidate_caches()
import aetherdialect
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.fast
def test_federation_construction_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb" or name.split(".", 1)[0] == "duckdb":
            raise ImportError("No module named 'duckdb'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        with pytest.raises(ConfigError, match=r"pip install aetherdialect\[(federation|duckdb)\]"):
            AetherFederation(
                "fed",
                members=(object(), object()),
                declaration="missing.json",
            )
