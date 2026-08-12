"""Fast tests: sandbox loaders use DDL schemas and CSV row data."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._sandbox import Sandbox
from tests._sandbox_csv_bundle import write_main_csv_ddl_bundle, write_member_csv_ddl_bundle

duckdb = pytest.importorskip("duckdb")


@pytest.mark.fast
def test_bundled_dataset_seed_has_no_insert_seed_filenames() -> None:
    with pytest.raises(KeyError):
        Sandbox.bundled_dataset_seed("main")


@pytest.mark.fast
def test_load_main_memory_connection_uses_csv_not_seed_sql(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_main_csv_ddl_bundle(bundle)
    (bundle / "rental_shop_seed.sql").write_text(
        "CREATE TABLE leaked (id INTEGER); INSERT INTO leaked VALUES (99);",
        encoding="utf-8",
    )
    connection = Sandbox._load_main_memory_connection(bundle)
    tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
    assert "customer" in tables
    assert "film" in tables
    assert "leaked" not in tables
    row = connection.execute("SELECT customer_id FROM customer").fetchone()
    assert row is not None and int(row[0]) == 1


@pytest.mark.fast
def test_load_member_memory_connection_uses_csv_not_seed_sql(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_member_csv_ddl_bundle(bundle, "storefront", tables=(("rental", "rental_id"),))
    (bundle / "federation_storefront_seed.sql").write_text(
        "CREATE TABLE leaked (id INTEGER); INSERT INTO leaked VALUES (99);",
        encoding="utf-8",
    )
    connection = Sandbox._load_member_memory_connection(bundle, "storefront")
    tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
    assert "rental" in tables
    assert "leaked" not in tables
    row = connection.execute("SELECT rental_id FROM rental").fetchone()
    assert row is not None and int(row[0]) == 1


@pytest.mark.fast
def test_load_dataset_default_path_is_csv_ddl(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_main_csv_ddl_bundle(bundle, tables=(("probe", "probe_id"),))
    with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
        sandbox.load_dataset("main")
        row = sandbox.connection("main").execute("SELECT probe_id FROM probe").fetchone()
        assert row is not None and int(row[0]) == 1
