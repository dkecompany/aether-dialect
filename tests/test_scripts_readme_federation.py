"""Fast tests that scripts/README.md documents the federation build pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_README = _REPO / "scripts" / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _README.read_text(encoding="utf-8")


@pytest.mark.fast
def test_scripts_readme_has_federation_build_section(readme_text: str) -> None:
    assert "## Federation build" in readme_text
    assert "source_rental_shop.py" in readme_text
    assert "sandbox_corpus.py" in readme_text
    assert "--federation-load all" in readme_text
    assert "--federation-verify all" in readme_text


@pytest.mark.fast
def test_scripts_readme_documents_federation_load_targets(readme_text: str) -> None:
    for target in ("storefront", "catalog", "logistics", "crm"):
        assert target in readme_text
    assert "--federation-load {storefront,catalog,logistics,crm,all}" in readme_text


@pytest.mark.fast
def test_scripts_readme_documents_partition_namespace_names(readme_text: str) -> None:
    assert "rental_shop_fed_storefront" in readme_text
    assert "rental_shop_fed_logistics" in readme_text
    assert "rental_shop_fed_catalog" in readme_text
    assert "rental_shop_fed_crm" in readme_text


@pytest.mark.fast
def test_scripts_readme_documents_drop_first_scope(readme_text: str) -> None:
    lowered = readme_text.lower()
    assert "--drop-first" in readme_text
    assert "scope:" in lowered
    assert "full `rental_shop`" in readme_text or "full rental_shop" in lowered
    assert "not" in lowered


@pytest.mark.fast
def test_scripts_readme_layout_lists_federation_artifacts(readme_text: str) -> None:
    assert "federation_declaration.json" in readme_text
    assert "federation_partition.json" in readme_text
    for schema in (
        "federation_storefront_schema.sql",
        "federation_catalog_schema.sql",
        "federation_logistics_schema.sql",
        "federation_crm_schema.sql",
    ):
        assert schema in readme_text
    assert "federation_storefront_seed.sql" not in readme_text


@pytest.mark.fast
def test_scripts_readme_documents_federation_export_pipeline(readme_text: str) -> None:
    lowered = readme_text.lower()
    assert "export_sandbox_federation_partition_schemas" in readme_text
    assert "export_sandbox_federation_partition_data_dirs" in readme_text
    assert "export_federation_member_data_dirs_from_existing_csvs" in readme_text
    assert "sandbox_staging" in lowered
    assert "csv" in lowered
    assert "seed sql" not in lowered or "no insert seed" in lowered or "ddl+csv" in lowered


@pytest.mark.fast
def test_scripts_readme_sandbox_metadata_table_is_current(readme_text: str) -> None:
    assert "## Sandbox data model (corpus metadata)" in readme_text
    assert "sandbox_paraphrase_pairs.json" not in readme_text
    for name in (
        "sandbox_expectations.json",
        "sandbox_scenarios.json",
        "sandbox_handcrafted_fixtures.json",
        "sandbox_migration_demo.json",
        "sandbox_structure_demo.json",
        "federation_storefront_notes.txt",
        "federation_catalog_notes.txt",
        "federation_logistics_notes.txt",
        "federation_crm_notes.txt",
    ):
        assert name in readme_text
