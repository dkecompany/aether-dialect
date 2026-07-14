"""CLI validation for scripts/source_rental_shop.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "source_rental_shop.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_enrich_llm_without_generate_errors() -> None:
    proc = _run_cli("--enrich-llm")
    assert proc.returncode != 0
    assert "--enrich-llm requires --generate" in proc.stderr


def test_download_and_generate_mutually_exclusive() -> None:
    proc = _run_cli("--download", "--generate")
    assert proc.returncode != 0
    assert "not allowed with" in proc.stderr.lower() or "mutually exclusive" in proc.stderr.lower()


def test_download_and_enrich_llm_errors() -> None:
    proc = _run_cli("--download", "--enrich-llm")
    assert proc.returncode != 0
    assert "--enrich-llm requires --generate" in proc.stderr


def test_pack_only_missing_tables_errors(tmp_path: Path) -> None:
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from source_rental_shop import pack_csv_bundle
    finally:
        sys.path.pop(0)
    empty = tmp_path / "csv"
    empty.mkdir()
    with pytest.raises(SystemExit, match="Cannot pack"):
        pack_csv_bundle(empty, tmp_path / "rental_shop.zip")
