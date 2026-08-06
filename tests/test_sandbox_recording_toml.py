"""Tests for sandbox corpus recording config (in-memory DuckDB override)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from live_tests.conftest import _parse_live_env_file, write_sandbox_recording_toml

from aetherdialect._main_execution import MainExecutionOps


def test_write_sandbox_recording_toml_forces_memory_duckdb_and_keeps_llm_creds(tmp_path: Path) -> None:
    env_path = tmp_path / "test.env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test-key",
                "DUCKDB_PATH=scripts/duckdb/rental_shop.duckdb",
                "DUCKDB_SCHEMA=main",
            ]
        ),
        encoding="utf-8",
    )
    toml_path = write_sandbox_recording_toml(str(env_path))
    try:
        flat, _claimed, _named = MainExecutionOps._load_config_file(toml_path)
        assert flat.get("DUCKDB_PATH") == ":memory:"
        assert flat.get("DUCKDB_SCHEMA") == "main"
        assert flat.get("OPENAI_API_KEY") == "sk-test-key"
        assert flat.get("AETHERDIALECT_ENGINE") == "duckdb"
        assert flat.get("AETHERDIALECT_LLM_PROVIDER") == "openai"
    finally:
        os.unlink(toml_path)


def test_write_sandbox_recording_toml_overrides_file_duckdb_from_env() -> None:
    """Recording TOML must not inherit file-backed DuckDB from a typical env.env."""
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "env.env"
    if not env_path.is_file():
        pytest.skip("env.env not present")
    source = _parse_live_env_file(str(env_path))
    if not source.get("DUCKDB_PATH") or source.get("DUCKDB_PATH") == ":memory:":
        pytest.skip("env.env has no file-backed DUCKDB_PATH")
    toml_path = write_sandbox_recording_toml(str(env_path))
    try:
        flat, _claimed, _named = MainExecutionOps._load_config_file(toml_path)
        assert flat.get("DUCKDB_PATH") == ":memory:"
    finally:
        os.unlink(toml_path)
