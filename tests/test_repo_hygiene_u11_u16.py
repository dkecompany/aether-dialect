"""Repo hygiene evidence for federation loader, gitignore, dead scripts, and QSim merge (U11–U16)."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "aetherdialect"
_SCRIPTS = _REPO / "scripts"
_DATA = _SCRIPTS / "data"
_LOADER = _SCRIPTS / "load_rental_shop_engines.py"
_README = _SCRIPTS / "README.md"
_GITIGNORE = _REPO / ".gitignore"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_FEDERATION_COMMITTED_ARTIFACTS = (
    "federation_storefront_seed.sql",
    "federation_catalog_seed.sql",
    "federation_logistics_seed.sql",
    "federation_crm_seed.sql",
    "federation_partition.json",
)


def _load_loader_module():
    spec = importlib.util.spec_from_file_location("load_rental_shop_engines_u11", _LOADER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_rental_shop_engines_u11"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sandbox_corpus_function_names() -> set[str]:
    source = (_SCRIPTS / "sandbox_corpus.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _gitignore_lines() -> list[str]:
    return [line.strip() for line in _GITIGNORE.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def loader():
    return _load_loader_module()


@pytest.mark.fast
def test_u11_crm_mysql_load_projects_staff_columns(loader, tmp_path: Path) -> None:
    """Live CRM MariaDB partition load must project staff to declared columns like offline export."""
    from sandbox_corpus import federation_member_column_projections, federation_partition_tables

    partition = federation_partition_tables("crm")
    staff_projection = set(federation_member_column_projections("crm")["staff"])

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    ddl_path = tmp_path / "ddl.sql"
    ddl_blocks = [f"CREATE TABLE {table} (id INTEGER PRIMARY KEY);" for table in sorted(partition)]
    ddl_path.write_text("\n".join(ddl_blocks), encoding="utf-8")

    for table in partition:
        if table == "staff":
            pd.DataFrame(
                {
                    "staff_id": [1],
                    "first_name": ["Alice"],
                    "last_name": ["Smith"],
                    "store_id": [1],
                    "ssn": ["synthetic-secret"],
                    "email": ["alice@example.com"],
                }
            ).to_csv(csv_dir / f"{table}.csv", index=False)
        else:
            pd.DataFrame({"id": [1]}).to_csv(csv_dir / f"{table}.csv", index=False)

    loaded_columns: dict[str, list[str]] = {}

    def _tracking_to_sql(self, name, con, **kwargs):
        loaded_columns[name] = list(self.columns)

    mock_conn = MagicMock()
    mock_begin = MagicMock()
    mock_begin.__enter__ = MagicMock(return_value=mock_conn)
    mock_begin.__exit__ = MagicMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_begin

    args = argparse.Namespace(
        env_file=tmp_path / "env.env",
        drop_first=False,
        csv_dir=csv_dir,
        ddl=ddl_path,
    )

    with (
        patch.object(loader, "load_env_file"),
        patch.object(loader, "create_engine", return_value=mock_engine),
        patch.object(loader.MariaDBRuntimeConfig, "apply_environment"),
        patch.object(loader.MariaDBRuntimeConfig, "db_url", return_value="mysql://localhost/test"),
        patch.object(loader, "_ensure_mysql_database"),
        patch.object(pd.DataFrame, "to_sql", _tracking_to_sql),
    ):
        loader._load_federation_mysql_partition(
            args,
            source_id="crm",
            database="rental_shop_fed_crm",
            runtime_cls=loader.MariaDBRuntimeConfig,
        )

    assert set(loaded_columns["staff"]) == staff_projection
    assert "ssn" not in loaded_columns["staff"]
    assert "email" not in loaded_columns["staff"]


@pytest.mark.fast
def test_u12_federation_committed_artifacts_not_gitignored() -> None:
    """Committed federation seeds and partition roster must not be ignored."""
    lines = _gitignore_lines()
    for artifact in _FEDERATION_COMMITTED_ARTIFACTS:
        assert f"scripts/data/{artifact}" not in lines
        assert artifact not in lines


@pytest.mark.fast
def test_u12_sandbox_staging_build_output_is_gitignored() -> None:
    """Corpus staging workspace is generated output, not a committed artifact."""
    lines = _gitignore_lines()
    assert "scripts/sandbox_staging/" in lines
    assert "scripts/sandbox_staging.zip" in lines


@pytest.mark.fast
def test_u12_readme_documents_committed_federation_artifacts() -> None:
    """scripts/README.md must state federation seeds/partition json are committed artifacts."""
    text = _README.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "committed" in lowered
    for artifact in _FEDERATION_COMMITTED_ARTIFACTS:
        assert artifact in text
    assert "scripts/sandbox_staging/" in text
    assert "gitignored" in lowered


@pytest.mark.fast
def test_u15_dead_corpus_helpers_removed() -> None:
    """Uncalled sandbox_corpus helpers ordered deleted must not remain in source."""
    names = _sandbox_corpus_function_names()
    assert "build_expectations" not in names
    assert "_recording_handle_for_slot" not in names


@pytest.mark.fast
def test_u16_qsim_ops_module_merged() -> None:
    """LLM QSim ops must live in _qsim.py, not a standalone _qsim_ops module."""
    assert not (_SRC / "_qsim_ops.py").is_file()


@pytest.mark.fast
def test_u16_public_qsim_ops_symbols_importable_from_qsim() -> None:
    """Public QSim orchestration entry points remain importable after merge."""
    from aetherdialect._qsim import (
        append_advanced_skeleton_variants,
        generate_all_intents,
        generate_all_questions,
        greedy_cover_indices_by_atoms,
    )

    assert callable(generate_all_intents)
    assert callable(generate_all_questions)
    assert callable(greedy_cover_indices_by_atoms)
    assert callable(append_advanced_skeleton_variants)
