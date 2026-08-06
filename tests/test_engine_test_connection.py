"""Fast tests for scripts/<engine>/test_connection.py import wiring."""

from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_ENGINES = [
    "bigquery",
    "databricks",
    "duckdb",
    "mariadb",
    "mysql",
    "postgresql",
    "redshift",
    "snowflake",
    "sqlite",
    "sqlserver",
]


@pytest.mark.fast
@pytest.mark.parametrize("engine", _ENGINES)
def test_engine_test_connection_py_compiles(engine: str) -> None:
    path = _SCRIPTS / engine / "test_connection.py"
    assert path.is_file(), f"missing {path}"
    py_compile.compile(str(path), doraise=True)


@pytest.mark.fast
@pytest.mark.parametrize("engine", _ENGINES)
def test_engine_test_connection_scripts_path_resolves(engine: str) -> None:
    script = _SCRIPTS / engine / "test_connection.py"
    text = script.read_text(encoding="utf-8")

    assert "_engine_entry" not in text, f"{script} still imports missing _engine_entry"
    assert 'parents[1] / "scripts"' not in text, f"{script} still appends /scripts to parents[1]"

    scripts_dir = script.resolve().parents[1]
    assert scripts_dir == _SCRIPTS
    assert (scripts_dir / "load_rental_shop_engines.py").is_file()

    spec = importlib.util.spec_from_file_location(
        "load_rental_shop_engines_entry_check",
        scripts_dir / "load_rental_shop_engines.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_rental_shop_engines_entry_check"] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_cmd_ping")
    assert hasattr(mod, "_cmd_verify")
    assert hasattr(mod, "DEFAULT_ENV_FILE")
