"""Minimal CSV+DDL sandbox bundle helpers for fast tests."""

from __future__ import annotations

from pathlib import Path


def write_main_csv_ddl_bundle(
    root: Path,
    *,
    tables: tuple[tuple[str, str], ...] = (
        ("customer", "customer_id"),
        ("film", "film_id"),
    ),
) -> None:
    ddl = "\n".join(f"CREATE TABLE {table} ({col} INTEGER PRIMARY KEY);" for table, col in tables)
    (root / "rental_shop.sql").write_text(ddl, encoding="utf-8")
    data_dir = root / "rental_shop_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for table, col in tables:
        (data_dir / f"{table}.csv").write_text(f"{col}\n1\n", encoding="utf-8")
    if not (root / "rental_shop_notes.txt").is_file():
        (root / "rental_shop_notes.txt").write_text("catalog notes", encoding="utf-8")
    fixtures = root / "fixtures"
    fixtures.mkdir(exist_ok=True)
    if not (fixtures / "rental_shop_mock.json").is_file():
        (fixtures / "rental_shop_mock.json").write_text('{"fixtures": []}', encoding="utf-8")


def write_member_csv_ddl_bundle(
    root: Path,
    member: str,
    *,
    tables: tuple[tuple[str, str], ...] = (("rental", "rental_id"),),
) -> None:
    ddl = "\n".join(f"CREATE TABLE {table} ({col} INTEGER PRIMARY KEY);" for table, col in tables)
    (root / f"federation_{member}_schema.sql").write_text(ddl, encoding="utf-8")
    data_dir = root / f"federation_{member}_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for table, col in tables:
        (data_dir / f"{table}.csv").write_text(f"{col}\n1\n", encoding="utf-8")
