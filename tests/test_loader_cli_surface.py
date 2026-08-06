"""Fast tests for load_rental_shop_engines CLI surface (ping, extract- csv)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_LOADER = _SCRIPTS / "load_rental_shop_engines.py"


def _load_loader_module():
    spec = importlib.util.spec_from_file_location("load_rental_shop_engines_cli_test", _LOADER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_rental_shop_engines_cli_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loader():
    return _load_loader_module()


@pytest.mark.fast
def test_loader_help_documents_ping_and_extract_csv_cli() -> None:
    proc = subprocess.run(
        [sys.executable, str(_LOADER), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    help_text = proc.stdout
    assert "--ping" in help_text
    assert "--extract-csv" in help_text
    assert "extract-csv" not in help_text.split("description:")[0] or "--extract-csv" in help_text


@pytest.mark.fast
def test_main_ping_dispatches_cmd_ping(loader) -> None:
    with (
        patch.object(loader, "_cmd_ping") as ping_mock,
        patch.object(loader, "load_env_file"),
        patch("sys.argv", ["load_rental_shop_engines.py", "--ping", "sqlite"]),
    ):
        loader.main()
    ping_mock.assert_called_once()
    assert ping_mock.call_args.args[0].engine == "sqlite"


@pytest.mark.fast
def test_main_extract_csv_dispatches_cmd_extract_csv(loader, tmp_path: Path) -> None:
    out_dir = tmp_path / "csv_out"
    with (
        patch.object(loader, "_cmd_extract_csv") as extract_mock,
        patch("sys.argv", ["load_rental_shop_engines.py", "--extract-csv", str(out_dir)]),
    ):
        loader.main()
    extract_mock.assert_called_once()
    assert extract_mock.call_args.args[0].out == out_dir
