"""Sandbox subset selection and federation row export helpers for rental_shop."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from aetherdialect._sandbox import Sandbox

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
_DATA = _REPO_ROOT / "scripts" / "data"

SANDBOX_SAMPLE_SEED = 2202
PAYMENT_UNION_SPLIT_STORE_THRESHOLD = 6
SMALL_TABLES_WHOLE = frozenset(
    {
        "category",
        "country",
        "language",
        "staff",
        "store",
        "promotion",
        "courier",
        "supplier",
        "warehouse",
        "author",
        "publisher",
    }
)

CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS: tuple[int, ...] = (9999001, 9999002, 9999003)

CORPUS_REALISM_COUNTRY_CATALOG_DRIFT: dict[int, str] = {
    44: "Great Britain",
    1: "Australia (catalog replica)",
    62: "Japan",
}

CORPUS_REALISM_COUNTRY_STOREFRONT_ONLY: dict[int, str] = {
    15: "Brazil",
}

CORPUS_REALISM_COUNTRY_CATALOG_ONLY: dict[int, str] = {
    211: "Catalog-only Island Republic",
}

SUBSCRIPTION_RETAIL_RESKIN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Action", "Activewear & Gear"),
    ("Comedy", "Casual Essentials"),
    ("Documentary", "How-To Guides"),
    ("Drama", "Premium Lifestyle"),
    ("Family", "Family Subscription"),
    ("Games", "Equipment Rental"),
    ("Horror", "Limited Release"),
    ("Music", "Audio Subscriptions"),
    ("New", "New Arrivals"),
    ("Sci-Fi", "Tech & Innovation"),
    ("Sports", "Sports & Outdoors"),
    ("Travel", "Travel & Adventure"),
    ("Children", "Youth Programs"),
    ("Foreign", "International Plans"),
    ("Animation", "Digital Media"),
)

CRM_CUSTOMER_DESYNC_IDS: frozenset[int] = frozenset({1, 5, 12, 23, 37})
CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET: int = 1000
CRM_CUSTOMER_LOYALTY_TIERS: tuple[str, ...] = ("bronze", "silver", "gold", "platinum")

FEDERATION_PARTITION_TABLES: dict[str, frozenset[str]] = {
    "storefront": frozenset(
        {
            "country",
            "city",
            "address",
            "store",
            "staff",
            "customer",
            "rental",
            "reservation",
            "payment",
        },
    ),
    "catalog": frozenset(
        {
            "language",
            "item",
            "film",
            "book",
            "game",
            "actor",
            "film_actor",
            "category",
            "item_category",
            "item_feature",
            "game_supported_language",
            "author",
            "publisher",
            "inventory",
            "payment",
            "country",
            "city",
        },
    ),
    "logistics": frozenset(
        {
            "warehouse",
            "stock_transfer",
            "supplier",
            "purchase_order",
            "purchase_line",
            "courier",
            "delivery",
            "damage_report",
            "inventory_status_history",
            "receipts",
        },
    ),
    "crm": frozenset({"promotion", "promotion_redemption", "customer", "staff"}),
}

_CROSS_PARTITION_FK_RE = re.compile(
    r"\bREFERENCES\s+(\w+)\s*\(",
    re.IGNORECASE,
)
_SEED_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+\"?(\w+)\"?\s*\(([^)]+)\)\s*VALUES\s*\((.+?)\);",
    re.IGNORECASE | re.DOTALL,
)


def _noop_log(_message: str) -> None:
    return None


_log_callback: Callable[[str], None] = _noop_log


def set_export_log_callback(callback: Callable[[str], None] | None) -> None:
    global _log_callback
    _log_callback = callback or _noop_log


def _verbose(message: str) -> None:
    _log_callback(message)


def _digest(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def _remove_tree(path: Path) -> None:
    def _on_rm_error(func: Any, p: str, _exc_info: Any) -> None:
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_on_rm_error)


def load_federation_partition_map(data_root: Path | None = None) -> dict[str, frozenset[str]]:
    root = data_root or _DATA
    path = root / "federation_partition.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("federation_partition.json must be a JSON object")
        return {
            str(source_id): frozenset(str(table) for table in tables if str(table).strip())
            for source_id, tables in payload.items()
            if isinstance(tables, list)
        }
    return dict(FEDERATION_PARTITION_TABLES)


def payment_store_id_by_rental_id(csv_dir: Path) -> dict[int, int]:
    rental_path = csv_dir / "rental.csv"
    inventory_path = csv_dir / "inventory.csv"
    if not rental_path.is_file() or not inventory_path.is_file():
        return {}
    rentals = pd.read_csv(rental_path, usecols=["rental_id", "inventory_id"])
    inventory = pd.read_csv(inventory_path, usecols=["inventory_id", "store_id"])
    merged = rentals.merge(inventory, on="inventory_id", how="left")
    out: dict[int, int] = {}
    for rental_id, store_id in zip(merged["rental_id"], merged["store_id"], strict=False):
        if pd.isna(rental_id):
            continue
        out[int(rental_id)] = 0 if pd.isna(store_id) else int(store_id)
    return out


def payment_store_id_by_payment_id(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute(
        """
        SELECT p.payment_id, i.store_id
        FROM payment p
        JOIN rental r ON r.rental_id = p.rental_id
        JOIN inventory i ON i.inventory_id = r.inventory_id
        """,
    ).fetchall()
    return {int(payment_id): int(store_id) for payment_id, store_id in rows}


def compute_sandbox_subset(conn: sqlite3.Connection) -> dict[str, set[int]]:
    customers = [int(row[0]) for row in conn.execute("SELECT customer_id FROM customer ORDER BY customer_id")]
    films = [int(row[0]) for row in conn.execute("SELECT item_id FROM film ORDER BY item_id")]
    books = [int(row[0]) for row in conn.execute("SELECT item_id FROM book ORDER BY item_id")]
    games = [int(row[0]) for row in conn.execute("SELECT item_id FROM game ORDER BY item_id")]
    customer_ids = {
        customers[_digest(SANDBOX_SAMPLE_SEED, "cust", index) % len(customers)]
        for index in range(min(80, len(customers)))
    }
    customer_ids.update(cid for cid in CRM_CUSTOMER_DESYNC_IDS if cid in set(customers))
    if 2 in set(customers):
        customer_ids.add(2)
    film_ids = {
        films[_digest(SANDBOX_SAMPLE_SEED, "film", index) % len(films)] for index in range(min(100, len(films)))
    }
    book_ids = {books[_digest(SANDBOX_SAMPLE_SEED, "book", index) % len(books)] for index in range(min(40, len(books)))}
    game_ids = {games[_digest(SANDBOX_SAMPLE_SEED, "game", index) % len(games)] for index in range(min(20, len(games)))}
    item_ids = film_ids | book_ids | game_ids
    address_ids: set[int] = set()
    for customer_id in customer_ids:
        row = conn.execute("SELECT address_id FROM customer WHERE customer_id=?", (customer_id,)).fetchone()
        if row and row[0] is not None:
            address_ids.add(int(row[0]))
    for staff_row in conn.execute("SELECT staff_id FROM staff"):
        row = conn.execute("SELECT address_id FROM staff WHERE staff_id=?", (staff_row[0],)).fetchone()
        if row and row[0] is not None:
            address_ids.add(int(row[0]))
    city_ids: set[int] = set()
    for address_id in address_ids:
        row = conn.execute("SELECT city_id FROM address WHERE address_id=?", (address_id,)).fetchone()
        if row and row[0] is not None:
            city_ids.add(int(row[0]))
    country_ids: set[int] = set()
    for city_id in city_ids:
        row = conn.execute("SELECT country_id FROM city WHERE city_id=?", (city_id,)).fetchone()
        if row and row[0] is not None:
            country_ids.add(int(row[0]))
    language_ids: set[int] = set()
    for item_id in film_ids:
        row = conn.execute(
            "SELECT original_language_id FROM film WHERE item_id=?",
            (item_id,),
        ).fetchone()
        if row and row[0] is not None:
            language_ids.add(int(row[0]))
    inventory_ids: set[int] = set()
    store_ids: set[int] = set()
    for item_id in item_ids:
        for inv_id, store_id in conn.execute(
            "SELECT inventory_id, store_id FROM inventory WHERE item_id=?",
            (item_id,),
        ):
            inventory_ids.add(int(inv_id))
            store_ids.add(int(store_id))
    rental_ids: set[int] = set()
    for customer_id in customer_ids:
        for (rental_id,) in conn.execute("SELECT rental_id FROM rental WHERE customer_id=?", (customer_id,)):
            rental_ids.add(int(rental_id))
    payment_ids: set[int] = set()
    for rental_id in rental_ids:
        for (payment_id,) in conn.execute("SELECT payment_id FROM payment WHERE rental_id=?", (rental_id,)):
            payment_ids.add(int(payment_id))
    delivery_ids: set[int] = set()
    for rental_id in rental_ids:
        for (delivery_id,) in conn.execute("SELECT delivery_id FROM delivery WHERE rental_id=?", (rental_id,)):
            delivery_ids.add(int(delivery_id))
    actor_ids: set[int] = set()
    category_ids: set[int] = set()
    for item_id in film_ids:
        for (actor_id,) in conn.execute(
            "SELECT actor_id FROM film_actor WHERE film_item_id=?",
            (item_id,),
        ):
            actor_ids.add(int(actor_id))
        for (category_id,) in conn.execute(
            "SELECT category_id FROM item_category WHERE item_id=?",
            (item_id,),
        ):
            category_ids.add(int(category_id))
    redemption_ids: set[int] = set()
    for rental_id in rental_ids:
        for (redemption_id,) in conn.execute(
            "SELECT redemption_id FROM promotion_redemption WHERE rental_id=?",
            (rental_id,),
        ):
            redemption_ids.add(int(redemption_id))
    po_ids: set[int] = set()
    for store_id in store_ids:
        for (po_id,) in conn.execute(
            "SELECT po_id FROM purchase_order WHERE store_id=?",
            (store_id,),
        ):
            po_ids.add(int(po_id))
    line_ids: set[int] = set()
    for po_id in po_ids:
        for (line_id,) in conn.execute(
            "SELECT line_id FROM purchase_line WHERE po_id=?",
            (po_id,),
        ):
            line_ids.add(int(line_id))
    transfer_ids: set[int] = set()
    for item_id in item_ids:
        for (transfer_id,) in conn.execute(
            "SELECT transfer_id FROM stock_transfer WHERE item_id=?",
            (item_id,),
        ):
            transfer_ids.add(int(transfer_id))
    return {
        "customer": customer_ids,
        "film": film_ids,
        "book": book_ids,
        "game": game_ids,
        "item": item_ids,
        "address": address_ids,
        "city": city_ids,
        "country": country_ids,
        "language": language_ids,
        "inventory": inventory_ids,
        "store": store_ids,
        "rental": rental_ids,
        "payment": payment_ids,
        "delivery": delivery_ids,
        "actor": actor_ids,
        "category": category_ids,
        "film_actor": set(),
        "item_category": set(),
        "promotion_redemption": redemption_ids,
        "purchase_order": po_ids,
        "purchase_line": line_ids,
        "stock_transfer": transfer_ids,
    }


def row_allowed(
    table_name: str,
    cols: list[str],
    row: tuple[object, ...],
    subset: dict[str, set[int]],
) -> bool:
    if table_name in SMALL_TABLES_WHOLE:
        return True
    row_map = dict(zip(cols, row, strict=True))
    if table_name == "actor":
        return int(row_map["actor_id"]) in subset["actor"]
    if table_name == "address":
        return int(row_map["address_id"]) in subset["address"]
    if table_name == "city":
        return int(row_map["city_id"]) in subset["city"]
    if table_name == "country":
        return int(row_map["country_id"]) in subset["country"]
    if table_name == "film":
        return int(row_map["item_id"]) in subset["film"]
    if table_name == "item":
        return int(row_map["item_id"]) in subset["item"]
    if table_name == "book":
        return int(row_map["item_id"]) in subset["book"]
    if table_name == "game":
        return int(row_map["item_id"]) in subset["game"]
    if table_name == "language":
        return int(row_map["language_id"]) in subset["language"]
    if table_name == "film_actor":
        return int(row_map["film_item_id"]) in subset["film"]
    if table_name == "item_category":
        return int(row_map["item_id"]) in subset["item"]
    if table_name == "item_feature":
        return int(row_map["item_id"]) in subset["item"]
    if table_name == "game_supported_language":
        return int(row_map["item_id"]) in subset["game"]
    if table_name == "reservation":
        return int(row_map["customer_id"]) in subset["customer"]
    if table_name == "inventory_status_history":
        return int(row_map["inventory_id"]) in subset["inventory"]
    if table_name == "damage_report":
        return int(row_map["rental_id"]) in subset["rental"]
    if table_name == "inventory":
        return int(row_map["inventory_id"]) in subset["inventory"]
    if table_name == "customer":
        return int(row_map["customer_id"]) in subset["customer"]
    if table_name == "rental":
        return int(row_map["rental_id"]) in subset["rental"]
    if table_name == "payment":
        return int(row_map["payment_id"]) in subset["payment"]
    if table_name == "delivery":
        return int(row_map["delivery_id"]) in subset["delivery"]
    if table_name == "purchase_order":
        return int(row_map["po_id"]) in subset["purchase_order"]
    if table_name == "purchase_line":
        return int(row_map["line_id"]) in subset["purchase_line"]
    if table_name == "stock_transfer":
        return int(row_map["transfer_id"]) in subset["stock_transfer"]
    if table_name == "promotion_redemption":
        return int(row_map["redemption_id"]) in subset["promotion_redemption"]
    return False


def reskin_subscription_retail_text(text: str) -> str:
    out = text
    for old, new in SUBSCRIPTION_RETAIL_RESKIN_REPLACEMENTS:
        out = out.replace(old, new)
    out = out.replace("DVD", "subscription bundle")
    out = out.replace("VHS", "legacy equipment")
    return out


def storefront_rental_create_sql() -> str:
    return (
        "CREATE TABLE rental ("
        "rental_id INTEGER NOT NULL, "
        "rental_date TIMESTAMP NOT NULL, "
        "inventory_id INTEGER NOT NULL, "
        "customer_id INTEGER NOT NULL, "
        "return_date TIMESTAMP, "
        "staff_id INTEGER NOT NULL, "
        "last_update TIMESTAMP NOT NULL"
        ");"
    )


def crm_customer_desync_first_name(customer_id: int, first_name: str) -> str:
    if customer_id not in CRM_CUSTOMER_DESYNC_IDS:
        return first_name
    return f"{first_name} (crm)"


def crm_customer_desync_address_id(customer_id: int, address_id: int) -> int:
    if customer_id not in CRM_CUSTOMER_DESYNC_IDS:
        return address_id
    return int(address_id) + CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET


def parse_seed_insert_column_values(seed_sql: str, table_name: str, column_name: str) -> tuple[str, ...]:
    table_key = table_name.strip().lower()
    column_key = column_name.strip().lower()
    values: list[str] = []
    for match in _SEED_INSERT_RE.finditer(seed_sql):
        table = match.group(1).strip().lower()
        if table != table_key:
            continue
        columns = [part.strip().strip('"').lower() for part in match.group(2).split(",")]
        if column_key not in columns:
            continue
        column_index = columns.index(column_key)
        row_values = _split_sql_values(match.group(3))
        if column_index < len(row_values):
            values.append(row_values[column_index].strip().strip("'"))
    return tuple(values)


def _split_sql_values(values_clause: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_string = False
    index = 0
    text = values_clause.strip()
    while index < len(text):
        char = text[index]
        if in_string:
            current.append(char)
            if char == "'" and (index + 1 >= len(text) or text[index + 1] != "'"):
                in_string = False
            elif char == "'" and index + 1 < len(text) and text[index + 1] == "'":
                current.append(text[index + 1])
                index += 1
        elif char == "'":
            in_string = True
            current.append(char)
        elif char == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if current:
        parts.append("".join(current).strip())
    return parts


def _reskin_seed_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line.startswith("INSERT INTO"):
            out.append(reskin_subscription_retail_text(line))
        else:
            out.append(line)
    return out


def _format_timestamp_second_precision(value: object) -> str:
    text = str(value).strip().strip("'")
    if not text or text.upper() == "NULL":
        return "NULL"
    if " " in text:
        date_part, time_part = text.split(" ", 1)
        if "." in time_part:
            time_part = time_part.split(".", 1)[0]
        text = f"{date_part} {time_part}"
    return "'" + text.replace("'", "''") + "'"


def _loyalty_tier_for_customer(customer_id: int) -> str:
    return CRM_CUSTOMER_LOYALTY_TIERS[customer_id % len(CRM_CUSTOMER_LOYALTY_TIERS)]


def _logistics_receipts_create_sql() -> str:
    return (
        "CREATE TABLE receipts ("
        "rcpt_id INTEGER NOT NULL, "
        "rent_id INTEGER NOT NULL, "
        "amt REAL NOT NULL, "
        "dt TIMESTAMP NOT NULL"
        ");"
    )


def _logistics_purchase_order_create_sql() -> str:
    return (
        "CREATE TABLE purchase_order ("
        "ord_id INTEGER NOT NULL, "
        "sup_id INTEGER NOT NULL, "
        "store_id INTEGER NOT NULL, "
        "ord_dt TEXT NOT NULL, "
        "recv_dt TEXT, "
        "status VARCHAR(20) NOT NULL, "
        "last_update TIMESTAMP NOT NULL"
        ");"
    )


def _logistics_purchase_line_create_sql() -> str:
    return (
        "CREATE TABLE purchase_line ("
        "line_id INTEGER NOT NULL, "
        "ord_id INTEGER NOT NULL, "
        "item_id INTEGER NOT NULL, "
        "quantity SMALLINT NOT NULL, "
        "unit_cost NUMERIC(8,2) NOT NULL, "
        "last_update TIMESTAMP NOT NULL"
        ");"
    )


def _crm_customer_create_sql() -> str:
    return (
        "CREATE TABLE customer ("
        "customer_id INTEGER NOT NULL, "
        "store_id INTEGER NOT NULL, "
        "first_name VARCHAR(45) NOT NULL, "
        "last_name VARCHAR(45) NOT NULL, "
        "email_addr VARCHAR(50), "
        "address_id INTEGER NOT NULL, "
        "loyalty_tier VARCHAR(20), "
        "create_date TEXT NOT NULL, "
        "last_update TIMESTAMP"
        ");"
    )


def _build_logistics_receipt_lines(
    conn: sqlite3.Connection,
    subset: dict[str, set[int]],
) -> list[str]:
    lines: list[str] = []
    lines.append(_logistics_receipts_create_sql().rstrip(";") + ";;")
    for row in conn.execute("SELECT payment_id, rental_id, amount, payment_date FROM payment ORDER BY payment_id"):
        row_map = {
            "payment_id": row[0],
            "rental_id": row[1],
            "amount": row[2],
            "payment_date": row[3],
        }
        if not row_allowed("payment", ["payment_id", "rental_id", "amount", "payment_date"], row, subset):
            continue
        payment_id = int(row_map["payment_id"])
        if payment_id % 3 != 0:
            continue
        rental_id = int(row_map["rental_id"])
        amount = row_map["amount"]
        payment_date = str(row_map["payment_date"])
        if " " in payment_date and "." in payment_date.split(" ", 1)[1]:
            payment_date = payment_date.split(".", 1)[0]
        receipt_id = payment_id + 10_000_000
        lines.append(
            f"INSERT INTO receipts (rcpt_id, rent_id, amt, dt) VALUES "
            f"({receipt_id}, {rental_id}, {repr(float(amount))}, '{payment_date}');"
        )
    return lines


def _build_orphan_delivery_lines(
    conn: sqlite3.Connection,
    subset: dict[str, set[int]],
) -> list[str]:
    del subset
    lines: list[str] = []
    courier_ids = [int(row[0]) for row in conn.execute("SELECT courier_id FROM courier ORDER BY courier_id")]
    if not courier_ids:
        return lines
    max_delivery_id = int(
        conn.execute("SELECT COALESCE(MAX(delivery_id), 0) FROM delivery").fetchone()[0],
    )
    for index, rental_id in enumerate(CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS):
        delivery_id = max_delivery_id + index + 1
        courier_id = courier_ids[index % len(courier_ids)]
        lines.append(
            f"INSERT INTO delivery (delivery_id, rental_id, courier_id, address_id, dispatched_at, delivered_at, "
            f"status, delivery_fee, tracking_number, last_update) VALUES "
            f"({delivery_id}, {rental_id}, {courier_id}, {rental_id}, '2024-06-01 10:00:00', NULL, "
            f"'dispatched', 0.0, 'ORPHAN-{rental_id}', '2024-06-01 10:00:00');"
        )
    return lines


def _apply_catalog_country_drift_line(line: str) -> str:
    if not line.startswith("INSERT INTO country"):
        return line
    for country_id, drift_name in CORPUS_REALISM_COUNTRY_CATALOG_DRIFT.items():
        marker = f"({country_id},"
        if marker in line:
            parts = line.split("VALUES (", 1)
            if len(parts) != 2:
                return line
            prefix, rest = parts
            closing = rest.rfind(")")
            if closing < 0:
                return line
            values = rest[:closing]
            fields = [part.strip() for part in values.split(",")]
            if len(fields) >= 2:
                fields[1] = "'" + drift_name.replace("'", "''") + "'"
                return prefix + "VALUES (" + ", ".join(fields) + rest[closing:]
    return line


def _country_ids_in_seed_lines(lines: list[str]) -> set[int]:
    present: set[int] = set()
    for line in lines:
        if not line.startswith("INSERT INTO country"):
            continue
        parts = line.split("VALUES (", 1)
        if len(parts) != 2:
            continue
        values = parts[1].rsplit(")", 1)[0]
        fields = [part.strip() for part in values.split(",")]
        if fields:
            try:
                present.add(int(fields[0]))
            except ValueError:
                continue
    return present


def _append_missing_country_lines(lines: list[str], countries: dict[int, str]) -> list[str]:
    present = _country_ids_in_seed_lines(lines)
    for country_id, country_name in countries.items():
        if country_id in present:
            continue
        escaped = country_name.replace("'", "''")
        lines.append(
            f"INSERT INTO country (country_id, country, last_update) VALUES "
            f"({country_id}, '{escaped}', '2024-01-01 00:00:00');"
        )
        present.add(country_id)
    return lines


def _append_catalog_only_country_lines(lines: list[str]) -> list[str]:
    return _append_missing_country_lines(lines, CORPUS_REALISM_COUNTRY_CATALOG_ONLY)


def _append_catalog_realism_fixture_lines(lines: list[str]) -> list[str]:
    lines = _append_missing_country_lines(lines, CORPUS_REALISM_COUNTRY_CATALOG_DRIFT)
    lines = _append_catalog_only_country_lines(lines)
    if not any("Catalog City" in line for line in lines):
        lines.append(
            "INSERT INTO city (city_id, city, country_id, last_update) VALUES "
            "(300, 'Catalog City', 44, '2024-01-01 00:00:00');"
        )
    if not any("Premium Monthly Subscription" in line for line in lines):
        lines.append(
            "INSERT INTO item (item_id, item_type, title, description, release_year, language_id, "
            "rental_duration, rental_rate, replacement_cost, last_update) VALUES "
            "(1000, 'film', 'Premium Monthly Subscription', 'subscription bundle tier', 2024, 1, 30, "
            "9.99, 19.99, '2024-01-01 00:00:00');"
        )
    return lines


def _append_storefront_realism_fixture_lines(lines: list[str]) -> list[str]:
    lines = _append_missing_country_lines(lines, CORPUS_REALISM_COUNTRY_STOREFRONT_ONLY)
    if not any("'2024-03-22 08:04:33'" in line for line in lines):
        for index, line in enumerate(lines):
            if not line.startswith("INSERT INTO rental"):
                continue
            parts = line.split("VALUES (", 1)
            if len(parts) != 2:
                continue
            prefix, rest = parts
            values = rest.rsplit(")", 1)[0]
            fields = [part.strip() for part in values.split(",")]
            if len(fields) >= 2:
                fields[1] = "'2024-03-22 08:04:33'"
                lines[index] = prefix + "VALUES (" + ", ".join(fields) + ");"
                break
    return lines


def _export_crm_customer_lines(
    conn: sqlite3.Connection,
    subset: dict[str, set[int]],
) -> list[str]:
    lines: list[str] = []
    lines.append(_crm_customer_create_sql().rstrip(";") + ";;")
    full_cols = [
        "customer_id",
        "store_id",
        "first_name",
        "last_name",
        "email",
        "address_id",
        "activebool",
        "create_date",
        "last_update",
    ]
    for row in conn.execute("SELECT * FROM customer ORDER BY customer_id"):
        if not row_allowed("customer", full_cols, row, subset):
            continue
        row_map = dict(zip(full_cols, row, strict=True))
        customer_id = int(row_map["customer_id"])
        first_name = crm_customer_desync_first_name(customer_id, str(row_map["first_name"]))
        email = row_map["email"]
        email_sql = "NULL"
        if email is not None and str(email).strip():
            email_sql = "'" + str(email).replace("'", "''") + "'"
        create_date = row_map["create_date"]
        create_sql = "'" + str(create_date).replace("'", "''") + "'"
        last_update = row_map["last_update"]
        last_sql = "NULL"
        if last_update is not None and str(last_update).strip():
            last_sql = _format_timestamp_second_precision(last_update)
        loyalty = _loyalty_tier_for_customer(customer_id)
        address_id = crm_customer_desync_address_id(customer_id, int(row_map["address_id"]))
        first_sql = first_name.replace("'", "''")
        last_name_sql = str(row_map["last_name"]).replace("'", "''")
        lines.append(
            f"INSERT INTO customer (customer_id, store_id, first_name, last_name, email_addr, "
            f"address_id, loyalty_tier, create_date, last_update) VALUES "
            f"({customer_id}, {int(row_map['store_id'])}, '{first_sql}', "
            f"'{last_name_sql}', {email_sql}, "
            f"{address_id}, '{loyalty}', {create_sql}, {last_sql});"
        )
    return lines


def _export_logistics_purchase_lines(
    conn: sqlite3.Connection,
    subset: dict[str, set[int]],
) -> list[str]:
    lines: list[str] = []
    lines.append(_logistics_purchase_order_create_sql().rstrip(";") + ";;")
    lines.append(_logistics_purchase_line_create_sql().rstrip(";") + ";;")
    po_cols = ["po_id", "supplier_id", "store_id", "ordered_date", "received_date", "status", "last_update"]
    for row in conn.execute("SELECT * FROM purchase_order ORDER BY po_id"):
        if not row_allowed("purchase_order", po_cols, row, subset):
            continue
        row_map = dict(zip(po_cols, row, strict=True))
        ord_id = int(row_map["po_id"])
        sup_id = int(row_map["supplier_id"])
        store_id = int(row_map["store_id"])
        ord_dt = "'" + str(row_map["ordered_date"]).replace("'", "''") + "'"
        recv = row_map["received_date"]
        recv_dt = "NULL"
        if recv is not None and str(recv).strip():
            recv_dt = "'" + str(recv).replace("'", "''") + "'"
        status = "'" + str(row_map["status"]).replace("'", "''") + "'"
        last_update = _format_timestamp_second_precision(row_map["last_update"])
        lines.append(
            f"INSERT INTO purchase_order (ord_id, sup_id, store_id, ord_dt, recv_dt, status, last_update) "
            f"VALUES ({ord_id}, {sup_id}, {store_id}, {ord_dt}, {recv_dt}, {status}, {last_update});"
        )
    line_cols = ["line_id", "po_id", "item_id", "quantity", "unit_cost", "last_update"]
    for row in conn.execute("SELECT * FROM purchase_line ORDER BY line_id"):
        if not row_allowed("purchase_line", line_cols, row, subset):
            continue
        row_map = dict(zip(line_cols, row, strict=True))
        last_update = _format_timestamp_second_precision(row_map["last_update"])
        lines.append(
            f"INSERT INTO purchase_line (line_id, ord_id, item_id, quantity, unit_cost, last_update) "
            f"VALUES ({int(row_map['line_id'])}, {int(row_map['po_id'])}, {int(row_map['item_id'])}, "
            f"{int(row_map['quantity'])}, {repr(float(row_map['unit_cost']))}, {last_update});"
        )
    return lines


def _seed_line_targets_table(line: str, table: str) -> bool:
    return bool(
        re.match(
            rf'^\s*(CREATE\s+TABLE|INSERT\s+INTO)\s+"?{re.escape(table)}"?\b',
            line,
            flags=re.IGNORECASE,
        )
    )


def _apply_corpus_realism_post_export(
    source_id: str,
    lines: list[str],
    conn: sqlite3.Connection,
    subset: dict[str, set[int]],
    *,
    apply_realism: bool = True,
) -> list[str]:
    if not apply_realism:
        return _reskin_seed_lines(lines)
    if source_id == "storefront":
        out: list[str] = []
        for line in lines:
            if _seed_line_targets_table(line, "rental") and line.lstrip().upper().startswith("CREATE"):
                out.append(storefront_rental_create_sql().rstrip(";") + ";;")
                continue
            if _seed_line_targets_table(line, "rental") and line.lstrip().upper().startswith("INSERT"):
                parts = line.split("VALUES (", 1)
                if len(parts) != 2:
                    out.append(line)
                    continue
                prefix, rest = parts
                values = rest.rsplit(")", 1)[0]
                fields = [part.strip() for part in values.split(",")]
                if len(fields) >= 5:
                    fields[1] = _format_timestamp_second_precision(fields[1])
                    if fields[4].upper() != "NULL":
                        fields[4] = _format_timestamp_second_precision(fields[4])
                    out.append(prefix + "VALUES (" + ", ".join(fields) + ");")
                    continue
            out.append(line)
        lines = _append_storefront_realism_fixture_lines(out)
    if source_id == "catalog":
        lines = [_apply_catalog_country_drift_line(line) for line in lines]
        lines = _append_catalog_realism_fixture_lines(lines)
    if source_id == "logistics":
        filtered: list[str] = []
        for line in lines:
            if _seed_line_targets_table(line, "purchase_order") or _seed_line_targets_table(line, "purchase_line"):
                continue
            filtered.append(line)
        lines = filtered
        lines.extend(_export_logistics_purchase_lines(conn, subset))
        lines.extend(_build_logistics_receipt_lines(conn, subset))
        lines.extend(_build_orphan_delivery_lines(conn, subset))
    if source_id == "crm":
        filtered_crm: list[str] = []
        skipping_customer = False
        for line in lines:
            if _seed_line_targets_table(line, "customer") and line.lstrip().upper().startswith("CREATE"):
                skipping_customer = True
                continue
            if (
                skipping_customer
                and _seed_line_targets_table(line, "customer")
                and line.lstrip().upper().startswith("INSERT")
            ):
                continue
            if skipping_customer and line.lstrip().upper().startswith("CREATE"):
                skipping_customer = False
            if not skipping_customer:
                filtered_crm.append(line)
        lines = filtered_crm
        insert_idx = next(
            (index for index, line in enumerate(lines) if line.startswith("INSERT INTO")),
            len(lines),
        )
        lines[insert_idx:insert_idx] = _export_crm_customer_lines(conn, subset)
    return _reskin_seed_lines(lines)


def federation_foreign_key_allowed(table: str, fk_clause: str, partition_tables: frozenset[str]) -> bool:
    if table not in partition_tables:
        return False
    match = _CROSS_PARTITION_FK_RE.search(fk_clause)
    if match is None:
        return True
    return match.group(1) in partition_tables


def _strip_disallowed_create_table_foreign_keys(
    create_sql: str,
    table: str,
    partition_tables: frozenset[str],
) -> str:
    pattern = re.compile(r"\s+REFERENCES\s+\w+\s*\([^)]*\)", re.IGNORECASE)

    def _replace(match: re.Match[str]) -> str:
        clause = match.group(0).strip()
        if federation_foreign_key_allowed(table, clause, partition_tables):
            return match.group(0)
        return ""

    return pattern.sub(_replace, create_sql)


def _create_table_sql_for_projection(
    conn: sqlite3.Connection,
    table: str,
    columns: frozenset[str],
) -> str:
    info = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    col_defs: list[str] = []
    for _cid, name, ctype, _notnull, _dflt, pk in info:
        if str(name) not in columns:
            continue
        part = f"{name} {ctype or 'TEXT'}"
        if pk:
            part += " PRIMARY KEY"
        col_defs.append(part)
    if not col_defs:
        raise ValueError(f"no projected columns for table {table!r}")
    return f"CREATE TABLE {table} ({', '.join(col_defs)});"


def export_sandbox_main_data_dir(data_root: Path, conn: sqlite3.Connection) -> None:
    """Write downsampled ``rental_shop_data/`` CSV tables from a sqlite subset corpus."""
    subset = compute_sandbox_subset(conn)
    data_dir = data_root / "rental_shop_data"
    if data_dir.exists():
        _remove_tree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    ).fetchall()
    for (table_name,) in table_rows:
        table = str(table_name)
        if table.startswith("sqlite_"):
            continue
        cols = [desc[0] for desc in conn.execute(f"SELECT * FROM [{table}] LIMIT 0").description]
        rows: list[dict[str, object]] = []
        for row in conn.execute(f"SELECT * FROM [{table}]"):
            if not row_allowed(table, cols, row, subset):
                continue
            rows.append({col: value for col, value in zip(cols, row, strict=True)})
        if rows:
            _write_csv_rows(data_dir / f"{table}.csv", cols, rows)
    _verbose(f"Wrote {data_dir} ({len(list(data_dir.glob('*.csv')))} csv tables)")


def _payment_row_matches_filter(
    row_map: dict[str, object],
    payment_filter: str | None,
    *,
    payment_store_ids: dict[int, int] | None = None,
) -> bool:
    if not payment_filter:
        return True
    store_id = 0
    if payment_store_ids is not None:
        payment_id = int(row_map.get("payment_id", 0) or 0)
        store_id = int(payment_store_ids.get(payment_id, 0))
    threshold_text = payment_filter.split()[-1]
    threshold = int(threshold_text)
    if "<=" in payment_filter:
        return store_id <= threshold
    if ">" in payment_filter:
        return store_id > threshold
    return True


def _partition_export_lines(
    conn: sqlite3.Connection,
    subset: dict[str, set[int]],
    allowed_tables: set[str],
    *,
    source_id: str,
    payment_filter: str | None = None,
    column_projections: dict[str, frozenset[str]] | None = None,
    apply_realism: bool = True,
) -> list[str]:
    partition_tables = frozenset(allowed_tables)
    column_projections = column_projections or {}
    payment_store_ids = payment_store_id_by_payment_id(conn) if payment_filter else None
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    ).fetchall()
    lines: list[str] = []
    for (table_name,) in table_rows:
        table = str(table_name)
        if table.startswith("sqlite_"):
            continue
        include_payment = table == "payment" and payment_filter is not None and table in allowed_tables
        if table not in allowed_tables and not include_payment:
            continue
        projection = column_projections.get(table)
        if projection:
            lines.append(_create_table_sql_for_projection(conn, table, projection).rstrip(";") + ";;")
        else:
            create_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if create_row and create_row[0]:
                create_sql = _strip_disallowed_create_table_foreign_keys(
                    str(create_row[0]).rstrip(";"),
                    table,
                    partition_tables,
                )
                lines.append(create_sql + ";;")
        full_cols = [desc[0] for desc in conn.execute(f"SELECT * FROM [{table}] LIMIT 0").description]
        cols = full_cols
        if projection:
            cols = [col for col in full_cols if col in projection]
        col_list = ", ".join(cols)
        for row in conn.execute(f"SELECT * FROM [{table}]"):
            row_map = dict(zip(full_cols, row, strict=True))
            if table == "payment":
                if not _payment_row_matches_filter(
                    row_map,
                    payment_filter,
                    payment_store_ids=payment_store_ids,
                ):
                    continue
            if not row_allowed(table, full_cols, row, subset):
                continue
            vals = []
            for col_name in cols:
                value = row_map[col_name]
                if value is None or value == "":
                    vals.append("NULL")
                elif isinstance(value, bool):
                    vals.append("1" if value else "0")
                elif value in ("t", "f", "true", "false"):
                    vals.append("1" if str(value).lower() in ("t", "true") else "0")
                elif isinstance(value, str):
                    vals.append("'" + value.replace("'", "''") + "'")
                elif isinstance(value, bytes):
                    vals.append("X'" + value.hex() + "'")
                else:
                    vals.append(repr(value))
            lines.append(f"INSERT INTO {table} ({col_list}) VALUES ({', '.join(vals)});")
    return _apply_corpus_realism_post_export(
        source_id,
        lines,
        conn,
        subset,
        apply_realism=apply_realism,
    )


def _load_duckdb_from_seed_lines(lines: list[str]) -> Any:
    """Materialize in-memory DuckDB from CREATE/INSERT lines (not written to disk)."""
    from aetherdialect._utils import require_driver

    require_driver("duckdb")
    import duckdb

    connection = duckdb.connect(":memory:")
    text = "\n".join(lines) + "\n"
    for statement in Sandbox._split_sql_statements(text):
        if statement.strip():
            connection.execute(statement)
    return connection


def _federation_member_export_kwargs(
    conn: sqlite3.Connection,
    *,
    apply_realism: bool = True,
) -> dict[str, dict[str, Any]]:
    from sandbox_corpus import federation_member_column_projections, federation_partition_tables

    subset = compute_sandbox_subset(conn)
    member_tables = {
        "storefront": set(federation_partition_tables("storefront")),
        "catalog": set(federation_partition_tables("catalog")),
        "logistics": set(federation_partition_tables("logistics")),
        "crm": set(federation_partition_tables("crm")),
    }
    return {
        "storefront": {
            "subset": subset,
            "allowed_tables": member_tables["storefront"],
            "source_id": "storefront",
            "payment_filter": f"store_id <= {PAYMENT_UNION_SPLIT_STORE_THRESHOLD}",
            "column_projections": federation_member_column_projections("storefront"),
            "apply_realism": apply_realism,
        },
        "catalog": {
            "subset": subset,
            "allowed_tables": member_tables["catalog"],
            "source_id": "catalog",
            "payment_filter": f"store_id > {PAYMENT_UNION_SPLIT_STORE_THRESHOLD}",
            "column_projections": federation_member_column_projections("catalog"),
            "apply_realism": apply_realism,
        },
        "logistics": {
            "subset": subset,
            "allowed_tables": member_tables["logistics"],
            "source_id": "logistics",
            "payment_filter": None,
            "column_projections": federation_member_column_projections("logistics"),
            "apply_realism": apply_realism,
        },
        "crm": {
            "subset": subset,
            "allowed_tables": member_tables["crm"],
            "source_id": "crm",
            "payment_filter": None,
            "column_projections": federation_member_column_projections("crm"),
            "apply_realism": apply_realism,
        },
    }


def export_sandbox_federation_partition_schemas(
    data_root: Path,
    conn: sqlite3.Connection,
    *,
    apply_realism: bool = True,
) -> None:
    """Write CREATE-only ``federation_*_schema.sql`` from a sqlite subset corpus."""
    for member, kwargs in _federation_member_export_kwargs(conn, apply_realism=apply_realism).items():
        lines = _partition_export_lines(conn, **kwargs)
        body = _federation_seed_create_only_sql("\n".join(lines))
        schema_path = data_root / f"federation_{member}_schema.sql"
        header = f"-- Federation {member} partition schema (CREATE-only; rows load from federation_{member}_data).\n\n"
        schema_path.write_text(header + body, encoding="utf-8")
        _verbose(f"Wrote {schema_path} ({schema_path.stat().st_size} bytes)")


def _federation_seed_create_only_sql(seed_sql: str) -> str:
    kept: list[str] = []
    for statement in Sandbox._split_sql_statements(seed_sql):
        stripped = statement.strip()
        if not stripped:
            continue
        upper = stripped.lstrip().upper()
        if upper.startswith("INSERT ") or upper.startswith("COPY "):
            continue
        kept.append(stripped.rstrip(";") + ";")
    return ("\n\n".join(kept) + "\n") if kept else ""


def export_sandbox_federation_partition_data_dirs(
    data_root: Path,
    conn: sqlite3.Connection,
    *,
    apply_realism: bool = True,
) -> None:
    """Write downsampled ``federation_*_data/`` CSV trees from a sqlite subset corpus."""
    for member, kwargs in _federation_member_export_kwargs(conn, apply_realism=apply_realism).items():
        lines = _partition_export_lines(conn, **kwargs)
        data_dir = data_root / f"federation_{member}_data"
        if data_dir.exists():
            _remove_tree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        connection = _load_duckdb_from_seed_lines(lines)
        try:
            tables = [str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()]
            for table in sorted(tables):
                quoted = '"' + table.replace('"', '""') + '"'
                out_path = data_dir / f"{table}.csv"
                posix = out_path.resolve().as_posix().replace("'", "''")
                connection.execute(f"COPY {quoted} TO '{posix}' (HEADER, DELIMITER ',')")
        finally:
            connection.close()
        _verbose(f"Wrote {data_dir} ({len(list(data_dir.glob('*.csv')))} csv tables)")


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _payment_rows_for_member(
    conn: sqlite3.Connection,
    member: str,
    payment_store_ids: dict[int, int],
) -> list[dict[str, object]]:
    cols = [desc[0] for desc in conn.execute("SELECT * FROM payment LIMIT 0").description]
    rows: list[dict[str, object]] = []
    for row in conn.execute("SELECT * FROM payment ORDER BY payment_id"):
        row_map = dict(zip(cols, row, strict=True))
        payment_id = int(row_map["payment_id"])
        store_id = int(payment_store_ids.get(payment_id, 0))
        if member == "storefront" and store_id > PAYMENT_UNION_SPLIT_STORE_THRESHOLD:
            continue
        if member == "catalog" and store_id <= PAYMENT_UNION_SPLIT_STORE_THRESHOLD:
            continue
        rows.append({col: row_map[col] for col in cols})
    return rows


def _receipt_rows_from_payments(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payment_id, rental_id, amount, payment_date in conn.execute(
        "SELECT payment_id, rental_id, amount, payment_date FROM payment ORDER BY payment_id",
    ):
        if int(payment_id) % 3 != 0:
            continue
        payment_date_text = str(payment_date)
        if " " in payment_date_text and "." in payment_date_text.split(" ", 1)[1]:
            payment_date_text = payment_date_text.split(".", 1)[0]
        rows.append(
            {
                "rcpt_id": int(payment_id) + 10_000_000,
                "rent_id": int(rental_id),
                "amt": float(amount),
                "dt": payment_date_text,
            },
        )
    return rows


def _table_rows_from_sqlite(conn: sqlite3.Connection, table: str) -> tuple[list[str], list[dict[str, object]]]:
    cols = [desc[0] for desc in conn.execute(f"SELECT * FROM [{table}] LIMIT 0").description]
    rows = [dict(zip(cols, row, strict=True)) for row in conn.execute(f"SELECT * FROM [{table}]")]
    return cols, rows


def export_federation_member_data_dirs_from_existing_csvs(
    *,
    csv_dir: Path | None = None,
    data_root: Path | None = None,
    sqlite_path: Path | None = None,
) -> None:
    """Write full federation member CSV dirs from existing rental_shop CSVs or sqlite."""
    csv_dir = csv_dir or (_DATA / "rental_shop_csvs")
    data_root = data_root or _DATA
    if sqlite_path is None:
        sqlite_path = _SCRIPTS / "sqlite" / "rental_shop.sqlite"
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Missing sqlite corpus built from CSVs: {sqlite_path}")
    partition_map = load_federation_partition_map(data_root)
    conn = sqlite3.connect(sqlite_path)
    try:
        payment_store_ids = payment_store_id_by_payment_id(conn)
        for member, tables in partition_map.items():
            data_dir = data_root / f"federation_{member}_data"
            if data_dir.exists():
                _remove_tree(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            for table in sorted(tables):
                if table == "receipts":
                    receipt_rows = _receipt_rows_from_payments(conn)
                    _write_csv_rows(data_dir / "receipts.csv", ["rcpt_id", "rent_id", "amt", "dt"], receipt_rows)
                    continue
                if table == "payment":
                    cols, _ = _table_rows_from_sqlite(conn, "payment")
                    payment_rows = _payment_rows_for_member(conn, member, payment_store_ids)
                    _write_csv_rows(data_dir / "payment.csv", cols, payment_rows)
                    continue
                source_csv = csv_dir / f"{table}.csv"
                if source_csv.is_file():
                    shutil.copy2(source_csv, data_dir / f"{table}.csv")
                    continue
                cols, table_rows = _table_rows_from_sqlite(conn, table)
                if table_rows:
                    _write_csv_rows(data_dir / f"{table}.csv", cols, table_rows)
            _verbose(f"Wrote {data_dir} ({len(list(data_dir.glob('*.csv')))} csv tables)")
    finally:
        conn.close()
