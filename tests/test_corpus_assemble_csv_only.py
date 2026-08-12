"""Corpus assemble uses CSV and DDL inputs without federation seed SQL."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_DATA = _SCRIPTS / "data"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.fast
def test_assemble_staging_source_does_not_reference_seed_sql_exports() -> None:
    sc = importlib.import_module("sandbox_corpus")
    source = inspect.getsource(sc.assemble_staging)
    for token in (
        "export_sandbox_seed_sql",
        "export_sandbox_federation_partition_seeds",
        "refresh_federation_member_schemas_from_seeds",
        "federation_storefront_seed.sql",
    ):
        assert token not in source


@pytest.mark.fast
def test_committed_federation_seed_sql_files_are_absent() -> None:
    for member in ("storefront", "catalog", "logistics", "crm"):
        seed = _DATA / f"federation_{member}_seed.sql"
        assert not seed.is_file(), f"unexpected federation seed SQL file present: {seed}"


@pytest.mark.fast
def test_pack_assertions_are_callable_without_zip() -> None:
    sc = importlib.import_module("sandbox_corpus")
    assert callable(sc.run_staging_pack_assertions)
    assert callable(sc.assert_staging_notes_parity)
    assert callable(sc.assert_staging_domain_knowledge_no_sensitive_columns)
    assert callable(sc.finalize_validate)
