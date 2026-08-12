"""Unit tests for sandbox subset and federation export callables in source_rental_shop."""

from __future__ import annotations

import importlib
import inspect
import sqlite3
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _source_rental_shop():
    return importlib.import_module("source_rental_shop")


def _sandbox_corpus():
    return importlib.import_module("sandbox_corpus")


def _build_minimal_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """CREATE TABLE customer (customer_id INTEGER PRIMARY KEY,
            store_id INTEGER, address_id INTEGER); CREATE TABLE film
            (item_id INTEGER PRIMARY KEY, original_language_id INTEGER);
            CREATE TABLE book (item_id INTEGER PRIMARY KEY); CREATE
            TABLE game (item_id INTEGER PRIMARY KEY); CREATE TABLE staff
            (staff_id INTEGER PRIMARY KEY, address_id INTEGER); CREATE
            TABLE address (address_id INTEGER PRIMARY KEY, city_id
            INTEGER); CREATE TABLE city (city_id INTEGER PRIMARY KEY,
            country_id INTEGER); CREATE TABLE country (country_id
            INTEGER PRIMARY KEY); CREATE TABLE language (language_id
            INTEGER PRIMARY KEY); CREATE TABLE inventory (inventory_id
            INTEGER PRIMARY KEY, item_id INTEGER, store_id INTEGER);
            CREATE TABLE rental (rental_id INTEGER PRIMARY KEY,
            customer_id INTEGER, inventory_id INTEGER); CREATE TABLE
            payment (payment_id INTEGER PRIMARY KEY, rental_id INTEGER);
            CREATE TABLE delivery (delivery_id INTEGER PRIMARY KEY,
            rental_id INTEGER); CREATE TABLE film_actor (actor_id
            INTEGER, film_item_id INTEGER); CREATE TABLE item_category
            (category_id INTEGER, item_id INTEGER); CREATE TABLE
            promotion_redemption (redemption_id INTEGER PRIMARY KEY,
            rental_id INTEGER); CREATE TABLE purchase_order (po_id
            INTEGER PRIMARY KEY, store_id INTEGER); CREATE TABLE
            purchase_line (line_id INTEGER PRIMARY KEY, po_id INTEGER);
            CREATE TABLE stock_transfer (transfer_id INTEGER PRIMARY
            KEY, item_id INTEGER); INSERT INTO customer VALUES (1, 1,
            1), (2, 1, 2), (3, 2, 3); INSERT INTO film VALUES (10, 1),
            (11, 1), (12, 2); INSERT INTO book VALUES (20), (21); INSERT
            INTO game VALUES (30); INSERT INTO staff VALUES (1, 1);
            INSERT INTO address VALUES (1, 1), (2, 1), (3, 2); INSERT
            INTO city VALUES (1, 1), (2, 2); INSERT INTO country VALUES
            (1), (2); INSERT INTO language VALUES (1), (2); INSERT INTO
            inventory VALUES (100, 10, 1), (101, 11, 7); INSERT INTO
            rental VALUES (1000, 1, 100), (1001, 2, 101); INSERT INTO
            payment VALUES (5000, 1000), (5001, 1001); INSERT INTO
            delivery VALUES (7000, 1000); INSERT INTO film_actor VALUES
            (1, 10); INSERT INTO item_category VALUES (1, 10); INSERT
            INTO promotion_redemption VALUES (8000, 1000); INSERT INTO
            purchase_order VALUES (9000, 1); INSERT INTO purchase_line
            VALUES (9100, 9000); INSERT INTO stock_transfer VALUES
            (9200, 10);"""
        )
    finally:
        conn.close()


@pytest.mark.fast
def test_compute_sandbox_subset_is_deterministic(tmp_path: Path) -> None:
    src = _source_rental_shop()
    db_path = tmp_path / "corpus.sqlite"
    _build_minimal_sqlite(db_path)
    conn = sqlite3.connect(db_path)
    try:
        first = src.compute_sandbox_subset(conn)
        second = src.compute_sandbox_subset(conn)
    finally:
        conn.close()
    assert first == second
    assert first["customer"]


@pytest.mark.fast
def test_sandbox_corpus_does_not_define_compute_sandbox_subset() -> None:
    sc = _sandbox_corpus()
    assert not hasattr(sc, "_compute_sandbox_subset")
    assert "def _compute_sandbox_subset" not in inspect.getsource(sc)


@pytest.mark.fast
def test_gitignore_ignores_federation_member_data_dirs() -> None:
    gitignore = (_REPO / ".gitignore").read_text(encoding="utf-8")
    for member in ("storefront", "catalog", "logistics", "crm"):
        assert f"scripts/data/federation_{member}_data/" in gitignore
    assert "scripts/sandbox_staging/" in gitignore


@pytest.mark.fast
def test_export_federation_member_data_dirs_from_existing_csvs(tmp_path: Path) -> None:
    src = _source_rental_shop()
    csv_dir = tmp_path / "rental_shop_csvs"
    csv_dir.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "federation_partition.json").write_text(
        '{"storefront": ["customer"], "catalog": ["payment"], "logistics": [], "crm": []}\n',
        encoding="utf-8",
    )
    sqlite_path = tmp_path / "corpus.sqlite"
    _build_minimal_sqlite(sqlite_path)
    (csv_dir / "customer.csv").write_text("customer_id,store_id,address_id\n1,1,1\n2,1,2\n", encoding="utf-8")
    (csv_dir / "payment.csv").write_text("payment_id,rental_id\n5000,1000\n5001,1001\n", encoding="utf-8")
    (csv_dir / "rental.csv").write_text("rental_id,inventory_id\n1000,100\n1001,101\n", encoding="utf-8")
    (csv_dir / "inventory.csv").write_text("inventory_id,store_id\n100,1\n101,7\n", encoding="utf-8")

    src.export_federation_member_data_dirs_from_existing_csvs(
        csv_dir=csv_dir,
        data_root=data_root,
        sqlite_path=sqlite_path,
    )

    storefront_customer = (data_root / "federation_storefront_data" / "customer.csv").read_text(encoding="utf-8")
    catalog_payment = (data_root / "federation_catalog_data" / "payment.csv").read_text(encoding="utf-8")
    assert "customer_id" in storefront_customer
    assert "5001" in catalog_payment
    assert "5000" not in catalog_payment
