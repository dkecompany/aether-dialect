"""CLI mode-collision validation for load_rental_shop_engines.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
_LOADER = _REPO / "scripts" / "load_rental_shop_engines.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_LOADER), *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_loader_module():
    spec = importlib.util.spec_from_file_location("load_rental_shop_engines_cli_collision_test", _LOADER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_rental_shop_engines_cli_collision_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loader():
    return _load_loader_module()


@pytest.mark.fast
@pytest.mark.parametrize(
    ("cli_args", "flag_a", "flag_b"),
    [
        (["--federation-verify", "all", "--federation-load", "all"], "--federation-verify", "--federation-load"),
        (["--federation-verify", "crm", "--all"], "--federation-verify", "--all"),
        (["--federation-load", "storefront", "--postgresql"], "--federation-load", "--postgresql"),
        (["--federation-verify", "all", "--exclude-mysql"], "--federation-verify", "--exclude-mysql"),
        (["--ping", "sqlite", "--federation-load", "all"], "--ping", "--federation-load"),
        (["--ping", "sqlite", "--all"], "--ping", "--all"),
        (["--extract-csv", "out", "--federation-verify", "all"], "--extract-csv", "--federation-verify"),
        (["--extract-csv", "out", "--mysql"], "--extract-csv", "--mysql"),
        (["--ping", "sqlite", "--extract-csv", "out"], "--ping", "--extract-csv"),
        (["--federation-load", "all", "--schema", "rental_shop"], "--federation-load", "--schema"),
        (["--federation-verify", "crm", "--recreate-schema"], "--federation-verify", "--recreate-schema"),
        (
            ["--federation-load", "all", "--allow-public-schema-recreate"],
            "--federation-load",
            "--allow-public-schema-recreate",
        ),
    ],
)
def test_colliding_mode_flags_exit_nonzero_and_name_both(cli_args: list[str], flag_a: str, flag_b: str) -> None:
    proc = _run_cli(*cli_args)
    assert proc.returncode != 0
    err = proc.stderr.lower()
    assert flag_a.lower() in err
    assert flag_b.lower() in err


@pytest.mark.fast
@pytest.mark.parametrize(
    "cli_args",
    [
        ["--federation-load", "all", "--drop-first"],
        ["--all", "--exclude-snowflake"],
        ["--federation-verify", "crm", "--verbose"],
    ],
)
def test_legal_mode_combos_parse_without_collision_error(loader, cli_args: list[str]) -> None:
    """Mode validation / dispatch only — loaders and CSV I/O never run."""
    argv = ["load_rental_shop_engines.py", *cli_args]
    with (
        patch.object(loader, "_cmd_federation_load") as fed_load,
        patch.object(loader, "_cmd_federation_verify") as fed_verify,
        patch.object(loader, "_cmd_ping"),
        patch.object(loader, "_cmd_extract_csv"),
        patch.object(loader, "load_env_file"),
        patch.object(loader, "_load_postgresql") as load_pg,
        patch.object(loader, "_load_databricks") as load_dbx,
        patch.object(loader, "_load_duckdb") as load_duck,
        patch.object(loader, "_load_sqlite") as load_sqlite,
        patch.object(loader, "_load_sqlalchemy_engine") as load_sa,
        patch.object(loader, "_cmd_verify"),
        # ``--all`` gates on csv_dir.is_dir() before calling loaders; pretend it exists
        # without creating files — mocked loaders never open CSVs.
        patch.object(Path, "is_dir", return_value=True),
        patch("sys.argv", argv),
    ):
        loader.main()
    if "--federation-load" in cli_args:
        fed_load.assert_called_once()
        assert load_pg.call_count == 0
    elif "--federation-verify" in cli_args:
        fed_verify.assert_called_once()
        assert load_pg.call_count == 0
    else:
        assert load_pg.called or load_dbx.called or load_duck.called or load_sqlite.called or load_sa.called
