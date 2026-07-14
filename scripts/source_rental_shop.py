"""Download or generate rental_shop CSV bundles for the 34-table schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import types
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from aetherdialect._config import DEFAULT_RANDOM_SEED
from load_rental_shop_engines import DEFAULT_ENV_FILE, load_env_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
_DATA = _REPO_ROOT / "scripts" / "data"

STORE_COUNT = 12
STAFF_COUNT = 24
WAREHOUSE_COUNT = 6
CITY_COUNT = 100
ADDRESSES_PER_CITY = 6
ADDRESS_COUNT = CITY_COUNT * ADDRESSES_PER_CITY

ACTIVITY_AS_OF = datetime(2026, 7, 1, 20, 0, 0)
ACTIVITY_END = ACTIVITY_AS_OF
ACTIVITY_START = ACTIVITY_AS_OF - timedelta(days=2920)
RECENT_WINDOW_DAYS = 90

_EMAIL_DOMAIN_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("gmail.com", 45),
    ("outlook.com", 20),
    ("hotmail.com", 15),
    ("yahoo.com", 10),
    ("icloud.com", 5),
    ("comcast.net", 5),
)

_FILM_LEXICON_COUNT = 1000
_ACTOR_LEXICON_COUNT = 200
_CUSTOMER_LEXICON_COUNT = 599
_STAFF_NAME_COUNT = 24

def _activity_as_of() -> datetime:
    override = os.environ.get("RENTAL_SHOP_AS_OF", "").strip()
    if override:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(override[: len(fmt.replace("%", "0"))], fmt)
            except ValueError:
                continue
        raise SystemExit(f"Invalid RENTAL_SHOP_AS_OF: {override!r}")
    return ACTIVITY_AS_OF


def _synth_fmt_ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _synth_fmt_date(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def _synth_activity_timestamp(*parts: object) -> str:
    span = int((ACTIVITY_END - ACTIVITY_START).total_seconds())
    offset = _digest(DEFAULT_RANDOM_SEED, *parts) % max(span, 1)
    return _synth_fmt_ts(ACTIVITY_START + timedelta(seconds=offset))


def _synth_rental_timestamp(rental_id: int | str, *parts: object) -> str:
    """Activity timestamp with mild Fri/Sat rental weighting."""

    for attempt in range(64):
        ts = _synth_activity_timestamp("rental", rental_id, *parts, attempt)
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        weekday = dt.weekday()
        weight = 3 if weekday in (4, 5) else 2 if weekday == 6 else 1
        if _digest(rental_id, *parts, attempt, "weekday") % weight == 0:
            return ts
    return _synth_activity_timestamp("rental", rental_id, *parts)


def _synth_business_hours_ts(base: datetime, entity_id: int | str, *parts: object) -> str:
    """Shift a timestamp into weekday dispatch hours (08:00-17:59)."""

    day_offset = _digest(entity_id, *parts, "biz_day") % 3
    dt = base + timedelta(days=day_offset)
    hour = 8 + (_digest(entity_id, *parts, "biz_hour") % 10)
    minute = _digest(entity_id, *parts, "biz_min") % 60
    second = _digest(entity_id, *parts, "biz_sec") % 60
    return _fmt_ts(dt.replace(hour=hour, minute=minute, second=second))


def _synth_spread_fk(entity_id: int | str, dim_count: int, salt: str = "fk") -> int:
    return 1 + (_digest(DEFAULT_RANDOM_SEED, salt, entity_id) % dim_count)


def _synth_read_jsonl(name: str) -> list[dict[str, object]]:
    path = _LEXICONS / name
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _synth_item_description(item_type: str, title: str, item_id: int | str) -> str:
    clean = str(title).strip().title()
    if item_type == "film":
        tones = (
            f"A popular rental title: {clean}.",
            f"{clean} remains a steady catalog favorite.",
            f"{clean} — available on DVD and Blu-ray at every store.",
        )
    elif item_type == "book":
        tones = (
            f"A well-loved rental book: {clean}.",
            f"Borrow {clean} from our fiction and nonfiction shelves.",
            f"{clean} — a steady rental on our library shelves.",
        )
    else:
        tones = (
            f"A rental game copy of {clean}.",
            f"{clean} — available for console and PC rental.",
            f"Game night favorite: {clean}.",
        )
    return tones[_digest(item_type, item_id, "desc") % len(tones)]


def _synth_phone_from_template(template: str, address_id: int) -> str:
    digits = "".join(ch for ch in str(template) if ch.isdigit())
    if len(digits) < 9:
        digits = f"555{address_id:07d}"
    base = digits[:12]
    suffix = str(_digest(address_id, "phone") % 10000).zfill(4)
    return (base + suffix)[:15]


def _synth_synthesize_geo_tables(
    countries_in: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    country_rows = [{k: v for k, v in row.items()} for row in countries_in]
    for row in country_rows:
        row["last_update"] = _synth_activity_timestamp("country", row.get("country_id"))

    city_lex = _require_lexicon("cities_sample.jsonl", min_rows=CITY_COUNT, fields=("city", "country_id"))
    city_rows: list[dict[str, str]] = []
    for idx, row in enumerate(city_lex[:CITY_COUNT], start=1):
        city_id = str(row.get("city_id") or idx)
        city_rows.append(
            {
                "city_id": city_id,
                "city": str(row["city"]),
                "country_id": str(row["country_id"]),
                "last_update": _synth_activity_timestamp("city", city_id),
            }
        )

    addr_lex = _require_lexicon(
        "addresses.jsonl",
        min_rows=ADDRESS_COUNT,
        fields=("address_line", "district", "city_id", "postal_code", "phone_template"),
    )
    address_rows: list[dict[str, str]] = []
    for address_id in range(1, ADDRESS_COUNT + 1):
        lex = addr_lex[address_id - 1]
        city_id = str(lex["city_id"])
        if int(city_id) > CITY_COUNT:
            raise SystemExit(
                f"addresses.jsonl row {address_id}: city_id {city_id} exceeds CITY_COUNT={CITY_COUNT}"
            )
        district = str(lex.get("district") or "").strip()
        if not district:
            raise SystemExit(f"addresses.jsonl row {address_id}: missing district")
        address_line = str(lex.get("address_line") or lex.get("address") or "").strip()
        if not address_line:
            raise SystemExit(f"addresses.jsonl row {address_id}: missing address_line")
        address_rows.append(
            {
                "address_id": str(address_id),
                "address": address_line,
                "district": district[:20],
                "city_id": city_id,
                "postal_code": str(lex["postal_code"]),
                "phone": _synth_phone_from_template(str(lex.get("phone_template") or ""), address_id),
                "last_update": _synth_activity_timestamp("address", address_id),
            }
        )
    return country_rows, city_rows, address_rows


def _synth_synthesize_stores(address_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    store_rows: list[dict[str, str]] = []
    for store_id in range(1, STORE_COUNT + 1):
        addr = address_rows[(store_id - 1) % len(address_rows)]
        store_rows.append(
            {
                "store_id": str(store_id),
                "manager_staff_id": "",
                "address_id": addr["address_id"],
                "last_update": _synth_activity_timestamp("store", store_id),
            }
        )
    return store_rows


def _email_domain_for(entity_id: int | str, *parts: object) -> str:
    roll = _digest(entity_id, *parts, "domain") % 100
    cumulative = 0
    for domain, weight in _EMAIL_DOMAIN_WEIGHTS:
        cumulative += weight
        if roll < cumulative:
            return domain
    return _EMAIL_DOMAIN_WEIGHTS[0][0]


def _synth_email_local(first: str, last: str, entity_id: int | str) -> str:
    local = f"{first.lower()}.{last.lower()}{entity_id}"
    return f"{local}@{_email_domain_for(entity_id, first, last)}"


def _synth_ssn(staff_id: int | str) -> str:
    area = 100 + (_digest(staff_id, "ssn_area") % 799)
    group = 10 + (_digest(staff_id, "ssn_grp") % 89)
    serial = 1000 + (_digest(staff_id, "ssn_serial") % 8999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def _synth_staff_password_hash(staff_id: int | str, username: str) -> str:
    payload = f"rental-shop:{staff_id}:{username}".encode()
    return hashlib.sha1(payload).hexdigest()


def _synth_synthesize_staff(
    store_rows: list[dict[str, str]],
    staff_names: list[dict[str, object]],
) -> list[dict[str, str]]:
    staff_rows: list[dict[str, str]] = []
    for staff_id in range(1, STAFF_COUNT + 1):
        store_id = str(1 + (staff_id - 1) // 2)
        if not staff_names:
            raise SystemExit("staff_names.jsonl is required and must contain at least one row")
        name_row = staff_names[(staff_id - 1) % len(staff_names)]
        first = str(name_row["first_name"])
        last = str(name_row["last_name"])
        email = (
            ""
            if staff_id in (23, 24)
            else _synth_email_local(first, last, staff_id)
        )
        username = f"staff{staff_id:02d}"
        staff_rows.append(
            {
                "staff_id": str(staff_id),
                "first_name": first,
                "last_name": last,
                "address_id": store_rows[int(store_id) - 1]["address_id"],
                "email": email,
                "store_id": store_id,
                "active": "1" if staff_id % 5 else "0",
                "username": username,
                "password": _synth_staff_password_hash(staff_id, username),
                "ssn": _synth_ssn(staff_id),
                "last_update": _synth_activity_timestamp("staff", staff_id),
            }
        )
    for store in store_rows:
        sid = int(store["store_id"])
        store["manager_staff_id"] = str(sid * 2 - 1)
    return staff_rows


def _synth_rebalance_customers(
    customers: list[dict[str, str]],
    store_count: int = STORE_COUNT,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in customers:
        cid = int(row["customer_id"])
        row = dict(row)
        row["store_id"] = str(_synth_spread_fk(cid, store_count, "customer_store"))
        row["create_date"] = _synth_fmt_date(
            ACTIVITY_START + timedelta(days=_digest(cid, "create") % 540)
        )
        row["last_update"] = _synth_activity_timestamp("customer", cid)
        out.append(row)
    return out


def _synth_rebalance_rentals(
    rentals: list[dict[str, str]],
    staff_count: int = STAFF_COUNT,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rentals:
        rid = int(row["rental_id"])
        row = dict(row)
        row["staff_id"] = str(_synth_spread_fk(rid, staff_count, "rental_staff"))
        row["rental_date"] = _synth_rental_timestamp(rid, "start")
        if row.get("return_date", "").strip():
            start = datetime.strptime(row["rental_date"], "%Y-%m-%d %H:%M:%S")
            dur = 1 + (_digest(rid, "dur") % 14)
            row["return_date"] = _synth_fmt_ts(start + timedelta(days=dur))
        row["last_update"] = _synth_activity_timestamp("rental", rid, "lu")
        out.append(row)
    return out


def _synth_rebalance_payments(
    payments: list[dict[str, str]],
    rentals: list[dict[str, str]],
) -> list[dict[str, str]]:
    rental_date = {r["rental_id"]: r["rental_date"] for r in rentals}
    out: list[dict[str, str]] = []
    for row in payments:
        row = dict(row)
        rid = row.get("rental_id", "")
        base = rental_date.get(rid)
        if base:
            start = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
            row["payment_date"] = _synth_fmt_ts(start + timedelta(hours=2))
        out.append(row)
    return out


def _synth_stagger_last_updates(
    rows: list[dict[str, object]],
    id_col: str,
    table: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        row = dict(row)
        key = row.get(id_col, 0)
        row["last_update"] = _synth_activity_timestamp(table, key)
        out.append(row)
    return out


def _synth_load_supplier_names() -> list[str]:
    path = _LEXICONS / "suppliers.csv"
    if not path.is_file():
        path = _SEED_LISTS / "suppliers.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["supplier_name"] for row in csv.DictReader(handle) if row.get("supplier_name")]


def _synth_load_publisher_names() -> list[str]:
    for rel in ("publishers_expanded.csv", "publishers.csv"):
        path = _LEXICONS / rel
        if not path.is_file():
            path = _SEED_LISTS / rel
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                names = [row.get("publisher_name") or row.get("publisher") or "" for row in csv.DictReader(handle)]
                return [n for n in names if n.strip()]
    return []


synth = types.SimpleNamespace(
    STORE_COUNT=STORE_COUNT,
    STAFF_COUNT=STAFF_COUNT,
    WAREHOUSE_COUNT=WAREHOUSE_COUNT,
    CITY_COUNT=CITY_COUNT,
    ADDRESSES_PER_CITY=ADDRESSES_PER_CITY,
    ADDRESS_COUNT=ADDRESS_COUNT,
    ACTIVITY_START=ACTIVITY_START,
    ACTIVITY_END=ACTIVITY_END,
    ACTIVITY_AS_OF=ACTIVITY_AS_OF,
    RECENT_WINDOW_DAYS=RECENT_WINDOW_DAYS,
    activity_as_of=_activity_as_of,
    activity_timestamp=_synth_activity_timestamp,
    spread_fk=_synth_spread_fk,
    item_description=_synth_item_description,
    synthesize_geo_tables=_synth_synthesize_geo_tables,
    synthesize_stores=_synth_synthesize_stores,
    synthesize_staff=_synth_synthesize_staff,
    rebalance_customers=_synth_rebalance_customers,
    rebalance_rentals=_synth_rebalance_rentals,
    rebalance_payments=_synth_rebalance_payments,
    stagger_last_updates=_synth_stagger_last_updates,
    load_supplier_names=_synth_load_supplier_names,
    load_publisher_names=_synth_load_publisher_names,
)
OUT_DIR = _DATA / "rental_shop_csvs"
ZIP_PATH = _DATA / "rental_shop.zip"
_DOWNLOADS = _DATA / "_downloads"
_FROZEN = _DOWNLOADS / "_frozen"
_SEED_LISTS = _DOWNLOADS / "_seed_lists"
_INPUTS_ZIP = _DATA / "inputs.zip"


def _log_progress(message: str) -> None:
    print(message, flush=True)


DEFAULT_BUNDLE_URL = (
    "https://stdialectsampledata.blob.core.windows.net/aether-dialect-sample-data/rental_shop.zip"
)
_TABLE_ORDER = (
    "actor",
    "category",
    "country",
    "language",
    "city",
    "address",
    "author",
    "publisher",
    "item",
    "film",
    "book",
    "game",
    "game_supported_language",
    "item_category",
    "film_actor",
    "item_feature",
    "store",
    "staff",
    "inventory",
    "inventory_status_history",
    "customer",
    "courier",
    "supplier",
    "warehouse",
    "promotion",
    "rental",
    "reservation",
    "damage_report",
    "payment",
    "delivery",
    "purchase_order",
    "purchase_line",
    "stock_transfer",
    "promotion_redemption",
)
_DDL_PATH = _DATA / "rental_shop.sql"

BOOK_COUNT = 300
GAME_COUNT = 150

_ITEM_ID_MAP: dict[tuple[str, int], int] | None = None

FILM_CATEGORY_IDS = tuple(range(1, 17))
BOOK_CATEGORY_IDS = tuple(range(17, 23))
GAME_CATEGORY_IDS = tuple(range(25, 31))

_EXTENSION_CATEGORY_NAMES: dict[str, str] = {
    "17": "Fiction",
    "18": "Nonfiction",
    "19": "Mystery",
    "20": "Biography",
    "21": "Young Adult",
    "22": "Reference",
    "25": "RPG",
    "26": "Platform",
    "27": "Strategy",
    "28": "Shooter",
    "29": "Puzzle",
    "30": "Simulation",
}

_LEXICONS = _DOWNLOADS / "_lexicons"
_GAMES_CSV_URL = (
    "https://zenodo.org/records/10262075/files/all_games_PC.csv?download=1"
)
_OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json?q=fiction&limit=500"

INVENTORY_TARGET = 5031
RENTAL_TARGET = 17516
PAYMENT_TARGET = 17521
FILM_ACTOR_TARGET = 5462

_LLM_CACHE_DIR = _DATA / "_llm_cache"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_LLM_BATCH_SIZE = 50

_SPINE_TABLES: dict[str, list[str]] = {
    "inventory": ["inventory_id", "film_id", "store_id", "last_update"],
    "rental": [
        "rental_id",
        "rental_date",
        "inventory_id",
        "customer_id",
        "return_date",
        "staff_id",
        "last_update",
    ],
    "payment": ["payment_id", "rental_id", "amount", "payment_date"],
    "film_actor": ["actor_id", "film_id", "last_update"],
    "film_category": ["film_id", "category_id", "last_update"],
}

OPEN_RENTAL_RATE = Decimal("0.18")
NO_RENTAL_INVENTORY_RATE = Decimal("0.10")
FAILED_DELIVERY_RATE = Decimal("0.12")
UNUSED_PROMO_IDS = frozenset({1, 2, 3, 4, 5, 7, 14, 21})

PLATFORMS = ("PS2", "Xbox", "GameCube", "PC", "GBA")
GAME_DEVELOPERS = (
    "Blizzard Entertainment",
    "Valve",
    "BioWare",
    "Ubisoft",
    "Electronic Arts",
    "Bethesda Game Studios",
    "Rockstar North",
    "Nintendo",
    "Capcom",
    "Square Enix",
    "FromSoftware",
    "CD Projekt Red",
    "id Software",
    "Infinity Ward",
    "Naughty Dog",
    "Insomniac Games",
    "Rare",
    "Bungie",
    "Epic Games",
    "Obsidian Entertainment",
    "Remedy Entertainment",
    "Guerrilla Games",
    "Sucker Punch Productions",
    "Arkane Studios",
    "Crystal Dynamics",
    "Monolith Productions",
    "Turn 10 Studios",
    "Playground Games",
    "Respawn Entertainment",
    "Treyarch",
)
ESRB_RATINGS = ("E", "E10+", "T", "M", "AO")
_GAME_TITLE_BLOCKLIST = frozenset(
    {
        "test",
        "demo",
        "sample",
        "placeholder",
        "unknown",
        "missing",
        "n/a",
        "not available",
        "undefined",
    }
)
DELIVERY_STATUSES = ("dispatched", "in_transit", "delivered", "returned")
PO_STATUSES = ("open", "received", "cancelled")
PROMO_TYPES = (
    "weekday_special",
    "bundle",
    "loyalty_reward",
    "clearance",
    "new_member",
)
INVENTORY_STATUSES = ("available", "rented", "damaged", "in_repair", "lost", "retired")
RESERVATION_STATUSES = ("pending", "fulfilled", "expired", "cancelled")
DAMAGE_SEVERITIES = ("minor", "moderate", "severe")

FEATURE_CANONICAL: dict[str, tuple[str, str]] = {
    "trailers": ("trailers", "video"),
    "commentaries": ("commentaries", "audio"),
    "deleted scenes": ("deleted_scenes", "video"),
    "behind the scenes": ("behind_the_scenes", "video"),
}


_INPUTS_MANIFEST: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("promotion_names.jsonl", 25, ("promo_name",)),
    ("courier_names.jsonl", 8, ("courier_name",)),
    ("warehouse_names.jsonl", 6, ("warehouse_name",)),
    ("cities_sample.jsonl", CITY_COUNT, ("city", "country_id")),
    ("addresses.jsonl", ADDRESS_COUNT, ("address_line", "district", "city_id", "postal_code", "phone_template")),
    ("country.jsonl", 6, ("country", "country_id")),
    ("language.jsonl", 6, ("name", "language_id")),
    ("category.jsonl", 16, ("name", "category_id")),
    ("staff_names.jsonl", STAFF_COUNT, ("first_name", "last_name")),
    ("customer.jsonl", _CUSTOMER_LEXICON_COUNT, ("customer_id", "first_name", "last_name")),
    ("actor.jsonl", _ACTOR_LEXICON_COUNT, ("actor_id", "first_name", "last_name")),
    ("film.jsonl", _FILM_LEXICON_COUNT, ("film_id", "title")),
)


def _lexicon_field(row: dict[str, str], field: str) -> str:
    val = str(row.get(field) or "").strip()
    if val:
        return val
    if field.endswith("_name"):
        return str(row.get("name") or "").strip()
    return ""


def _require_lexicon(name: str, *, min_rows: int = 1, fields: tuple[str, ...] = ()) -> list[dict[str, str]]:
    rows = _read_lexicon_jsonl(name)
    if not rows and name == "film.jsonl":
        rows = _read_lexicon_jsonl("films.jsonl")
    if len(rows) < min_rows:
        raise SystemExit(f"inputs lexicon {name}: need >={min_rows} rows, got {len(rows)}")
    for field in fields:
        for idx, row in enumerate(rows, start=1):
            if not _lexicon_field(row, field):
                raise SystemExit(f"inputs lexicon {name} row {idx}: missing {field}")
    return rows


def _validate_inputs_zip() -> None:
    for name, min_rows, fields in _INPUTS_MANIFEST:
        _require_lexicon(name, min_rows=min_rows, fields=fields)
    suppliers = _LEXICONS / "suppliers.csv"
    if not suppliers.is_file():
        raise SystemExit("inputs lexicon suppliers.csv: missing")
    with suppliers.open(newline="", encoding="utf-8") as handle:
        supplier_rows = list(csv.DictReader(handle))
    if len(supplier_rows) < 12:
        raise SystemExit(f"inputs lexicon suppliers.csv: need >=12 rows, got {len(supplier_rows)}")
    if not any(row.get("supplier_name") for row in supplier_rows):
        raise SystemExit("inputs lexicon suppliers.csv: missing supplier_name values")
    publishers = _LEXICONS / "publishers_expanded.csv"
    if not publishers.is_file():
        raise SystemExit("inputs lexicon publishers_expanded.csv: missing")


def _read_lexicon_jsonl(name: str) -> list[dict[str, str]]:
    path = _LEXICONS / name
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                rows.append({k: str(v) if v is not None else "" for k, v in payload.items()})
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_lexicon_alias(source: str, dest: str) -> None:
    src = _LEXICONS / source
    dst = _LEXICONS / dest
    if src.is_file() and not dst.is_file():
        shutil.copy2(src, dst)


def _materialize_frozen_lexicons() -> None:
    """Copy frozen lexicons from inputs.zip; fail if required members are missing."""

    _LEXICONS.mkdir(parents=True, exist_ok=True)
    for path in _FROZEN.glob("*.jsonl"):
        shutil.copy2(path, _LEXICONS / path.name)
    for name in (
        "addresses.jsonl",
        "suppliers.csv",
        "publishers_expanded.csv",
        "courier_names.jsonl",
        "warehouse_names.jsonl",
        "promotion_names.jsonl",
    ):
        src = _FROZEN / name
        if src.is_file():
            shutil.copy2(src, _LEXICONS / name)
    _copy_lexicon_alias("films.jsonl", "film.jsonl")
    _copy_lexicon_alias("actors.jsonl", "actor.jsonl")
    _copy_lexicon_alias("customers.jsonl", "customer.jsonl")
    _validate_inputs_zip()


def _bootstrap_lexicon_csv(name: str, fieldnames: list[str]) -> None:
    path = OUT_DIR / f"{name}.csv"
    if path.is_file() and _read_csv(name):
        return
    rows = _read_lexicon_jsonl(f"{name}.jsonl")
    if not rows:
        return
    if name in ("category", "language"):
        rows = [
            {k: str(v) for k, v in row.items()}
            for row in synth.stagger_last_updates(rows, fieldnames[0], name)
        ]
    _write_csv(name, fieldnames, rows)


def _bootstrap_film_csv_from_lexicon() -> None:
    path = OUT_DIR / "film.csv"
    if path.is_file() and _read_csv("film"):
        return
    features = {
        row["film_id"]: row.get("special_features") or ""
        for row in _read_lexicon_jsonl("film_features.jsonl")
    }
    films: list[dict[str, str]] = []
    for row in _read_lexicon_jsonl("film.jsonl"):
        film_id = int(row["film_id"])
        release_year = str(2006 + (_digest(film_id, "year") % 31) - 15)
        films.append(
            {
                "film_id": row["film_id"],
                "title": row.get("title") or "",
                "description": synth.item_description("film", row.get("title") or "", film_id),
                "release_year": release_year,
                "language_id": row.get("language_id") or "1",
                "original_language_id": row.get("original_language_id") or row.get("language_id") or "1",
                "rental_duration": row.get("rental_duration") or "3",
                "rental_rate": row.get("rental_rate") or "2.99",
                "length": row.get("length") or "90",
                "replacement_cost": row.get("replacement_cost") or "19.99",
                "rating": row.get("rating") or "PG",
                "last_update": row.get("last_update") or synth.activity_timestamp("film_src", film_id),
                "special_features": features.get(row["film_id"]) or row.get("special_features") or "",
                "fulltext": row.get("fulltext") or "",
            }
        )
    if not films:
        raise SystemExit("film.jsonl lexicon is required to bootstrap film.csv")
    _write_csv("film", list(films[0].keys()), films)


def _merge_lexicon_categories() -> list[dict[str, str]]:
    rows = _require_lexicon("category.jsonl", min_rows=16, fields=("name", "category_id"))
    out = [{k: str(v) for k, v in row.items()} for row in synth.stagger_last_updates(rows, "category_id", "category")]
    return out


def _film_duration_days(film_id: int, films_by_id: dict[int, dict[str, str]]) -> int:
    row = films_by_id.get(film_id, {})
    raw = str(row.get("rental_duration") or "3").strip()
    try:
        duration = int(raw)
    except ValueError:
        duration = 3
    return max(1, min(duration, 30))


def _rental_timestamp_for(rental_id: int) -> datetime:
    anchor = _activity_as_of()
    recent_start = anchor - timedelta(days=RECENT_WINDOW_DAYS)
    span_recent = max(int((anchor - recent_start).total_seconds()), 1)
    span_total = max(int((anchor - ACTIVITY_START).total_seconds()), 1)
    if _digest(rental_id, "recent") % 100 < 40:
        offset = _digest(rental_id, "ts", "recent") % span_recent
        dt = recent_start + timedelta(seconds=offset)
    else:
        offset = _digest(rental_id, "ts") % span_total
        dt = ACTIVITY_START + timedelta(seconds=offset)
    for attempt in range(64):
        weekday = dt.weekday()
        weight = 3 if weekday in (4, 5) else 2 if weekday == 6 else 1
        if _digest(rental_id, attempt, "weekday") % weight == 0:
            break
        dt = dt + timedelta(hours=1 + attempt)
    if dt > anchor:
        dt = anchor - timedelta(hours=1 + (_digest(rental_id) % 48))
    if dt < ACTIVITY_START:
        dt = ACTIVITY_START + timedelta(hours=_digest(rental_id) % 24)
    return dt


def _cap_at_anchor(value: datetime, anchor: datetime | None = None) -> datetime:
    limit = anchor or _activity_as_of()
    return value if value <= limit else limit - timedelta(hours=1 + (_digest(value.isoformat()) % 12))


def compute_return_date(
    rental_date: datetime,
    duration_days: int,
    rental_id: int,
    *,
    as_of: datetime | None = None,
) -> str:
    anchor = as_of or _activity_as_of()
    if _digest(rental_id, "open") % 5 == 0:
        return ""
    extra = _digest(rental_id, "late") % 2
    returned = rental_date + timedelta(days=duration_days + extra)
    if returned > anchor:
        returned = _cap_at_anchor(returned, anchor)
    if returned <= rental_date:
        returned = rental_date + timedelta(days=1)
    return _fmt_ts(returned)


def item_for_inventory(inventory_id: str, inventory_rows: list[dict[str, str]]) -> int | None:
    for row in inventory_rows:
        if row["inventory_id"] == inventory_id:
            raw = row.get("item_id") or row.get("film_id")
            return int(raw) if raw else None
    return None


def enforce_return_dates_at_anchor(rentals: list[dict[str, str]]) -> list[dict[str, str]]:
    anchor = _activity_as_of()
    for row in rentals:
        raw = str(row.get("return_date") or "").strip()
        if not raw:
            continue
        rental_date = _parse_ts(row["rental_date"])
        returned = _cap_at_anchor(_parse_ts(raw), anchor)
        if returned <= rental_date:
            returned = _cap_at_anchor(rental_date + timedelta(hours=2), anchor)
        row["return_date"] = _fmt_ts(returned)
    return rentals


def enforce_inventory_rental_sequence(rentals: list[dict[str, str]]) -> list[dict[str, str]]:
    by_inventory: dict[str, list[tuple[int, datetime, datetime | None]]] = {}
    for idx, row in enumerate(rentals):
        inv = row["inventory_id"]
        start = _parse_ts(row["rental_date"])
        end = _parse_ts(row["return_date"]) if str(row.get("return_date") or "").strip() else None
        by_inventory.setdefault(inv, []).append((idx, start, end))
    for _inv, spans in by_inventory.items():
        spans.sort(key=lambda t: t[1])
        prev_end: datetime | None = None
        for idx, start, end in spans:
            if prev_end is not None and start < prev_end:
                rentals[idx]["rental_date"] = _fmt_ts(prev_end + timedelta(hours=2))
                start = _parse_ts(rentals[idx]["rental_date"])
            if end is not None and end <= start:
                rentals[idx]["return_date"] = _fmt_ts(start + timedelta(days=1))
                end = _parse_ts(rentals[idx]["return_date"])
            prev_end = end or (_activity_as_of() + timedelta(days=365))
    return rentals


def synthesize_rental_spine() -> None:
    """Build inventory, rental, payment, film_actor, and film_category without Pagila."""
    _log_progress(
        f"[spine] synthesizing inventory/rental/payment/film_actor/film_category "
        f"(targets: inv={INVENTORY_TARGET}, rental={RENTAL_TARGET}, payment={PAYMENT_TARGET}, "
        f"film_actor={FILM_ACTOR_TARGET})"
    )

    films = _require_lexicon("film.jsonl", min_rows=1, fields=("film_id", "title"))
    if not films:
        raise SystemExit("film.jsonl lexicon is required to synthesize the rental spine")
    film_ids = [int(row["film_id"]) for row in films if row.get("film_id")]
    films_by_id = {int(row["film_id"]): row for row in films if row.get("film_id")}
    if not film_ids:
        raise SystemExit("film.jsonl must contain film_id rows")

    actors = _require_lexicon("actor.jsonl", min_rows=1, fields=("actor_id", "first_name", "last_name"))
    actor_ids = [int(row["actor_id"]) for row in actors if row.get("actor_id")]
    customers = _require_lexicon("customer.jsonl", min_rows=1, fields=("customer_id", "first_name", "last_name"))
    if not customers:
        customers = _read_csv("customer")
    customer_ids = [int(row["customer_id"]) for row in customers if row.get("customer_id")]
    if not customer_ids:
        raise SystemExit("customer lexicon is required to synthesize rentals")

    inventory_rows: list[dict[str, str]] = []
    inventory_id = 1
    per_film = max(1, INVENTORY_TARGET // max(len(film_ids), 1))
    for film_id in film_ids:
        copies = per_film + (1 if _digest(film_id, "inv_extra") % 3 == 0 else 0)
        for _copy_idx in range(copies):
            if inventory_id > INVENTORY_TARGET:
                break
            store_id = 1 + (_digest(inventory_id, "store") % STORE_COUNT)
            inventory_rows.append(
                {
                    "inventory_id": str(inventory_id),
                    "film_id": str(film_id),
                    "store_id": str(store_id),
                    "last_update": synth.activity_timestamp("inventory", inventory_id),
                }
            )
            inventory_id += 1
        if inventory_id > INVENTORY_TARGET:
            break
    while len(inventory_rows) < INVENTORY_TARGET:
        film_id = film_ids[len(inventory_rows) % len(film_ids)]
        store_id = 1 + (_digest(len(inventory_rows) + 1, "store") % STORE_COUNT)
        inventory_rows.append(
            {
                "inventory_id": str(len(inventory_rows) + 1),
                "film_id": str(film_id),
                "store_id": str(store_id),
                "last_update": synth.activity_timestamp("inventory", len(inventory_rows) + 1),
            }
        )
    inventory_rows = inventory_rows[:INVENTORY_TARGET]
    _write_csv(
        "inventory",
        ["inventory_id", "film_id", "store_id", "last_update"],
        inventory_rows,
    )
    _log_progress(f"[spine] inventory.csv written ({len(inventory_rows)} rows)")

    rental_rows: list[dict[str, str]] = []
    store_counts: dict[str, int] = {}
    for row in inventory_rows:
        sid = row["store_id"]
        store_counts[sid] = store_counts.get(sid, 0) + 1

    def _pick_inventory(rental_id: int) -> dict[str, str]:
        stores = sorted(store_counts)
        total = sum(store_counts[s] for s in stores)
        roll = _digest(rental_id, "inv_pick") % max(total, 1)
        cumulative = 0
        chosen_store = stores[0]
        for sid in stores:
            cumulative += store_counts[sid]
            if roll < cumulative:
                chosen_store = sid
                break
        pool = [row for row in inventory_rows if row["store_id"] == chosen_store]
        return pool[_digest(rental_id, "pool") % len(pool)]

    for rental_id in range(1, RENTAL_TARGET + 1):
        inv = _pick_inventory(rental_id)
        customer_id = customer_ids[(rental_id - 1) % len(customer_ids)]
        rental_dt = _rental_timestamp_for(rental_id)
        rental_date = _fmt_ts(rental_dt)
        film_id = int(inv["film_id"])
        duration = _film_duration_days(film_id, films_by_id)
        return_date = compute_return_date(rental_dt, duration, rental_id)
        rental_rows.append(
            {
                "rental_id": str(rental_id),
                "rental_date": rental_date,
                "inventory_id": inv["inventory_id"],
                "customer_id": str(customer_id),
                "return_date": return_date,
                "staff_id": str(1 + (_digest(rental_id, "staff") % STAFF_COUNT)),
                "last_update": synth.activity_timestamp("rental", rental_id, "lu"),
            }
        )
    rental_rows = enforce_inventory_rental_sequence(rental_rows)
    _write_csv(
        "rental",
        [
            "rental_id",
            "rental_date",
            "inventory_id",
            "customer_id",
            "return_date",
            "staff_id",
            "last_update",
        ],
        rental_rows,
    )
    _log_progress(f"[spine] rental.csv written ({len(rental_rows)} rows)")

    payment_rows: list[dict[str, str]] = []
    for payment_id in range(1, min(RENTAL_TARGET, PAYMENT_TARGET) + 1):
        rental = rental_rows[payment_id - 1]
        base = datetime.strptime(rental["rental_date"], "%Y-%m-%d %H:%M:%S")
        payment_rows.append(
            {
                "payment_id": str(payment_id),
                "rental_id": rental["rental_id"],
                "amount": _money(Decimal("2.99")),
                "payment_date": _fmt_ts(base + timedelta(hours=2)),
            }
        )
    while len(payment_rows) < PAYMENT_TARGET:
        rental = rental_rows[len(payment_rows) % len(rental_rows)]
        base = datetime.strptime(rental["rental_date"], "%Y-%m-%d %H:%M:%S")
        payment_rows.append(
            {
                "payment_id": str(len(payment_rows) + 1),
                "rental_id": rental["rental_id"],
                "amount": _money(
                    Decimal("1.50") + Decimal(_digest(len(payment_rows), "extra") % 300) / Decimal("100")
                ),
                "payment_date": _fmt_ts(base + timedelta(hours=2 + _digest(len(payment_rows), "extra_pay") % 72)),
            }
        )
    _write_csv("payment", ["payment_id", "rental_id", "amount", "payment_date"], payment_rows[:PAYMENT_TARGET])
    _log_progress(f"[spine] payment.csv written ({min(len(payment_rows), PAYMENT_TARGET)} rows)")

    film_actor_rows: list[dict[str, str]] = []
    if actor_ids:
        _log_progress(
            f"[spine] film_actor.csv building ({len(actor_ids)} actors x {len(film_ids)} films, "
            f"target={FILM_ACTOR_TARGET})"
        )
        film_order = sorted(film_ids, key=lambda fid: _digest("film_actor_order", fid))
        for actor_id in actor_ids:
            for film_id in film_order:
                if len(film_actor_rows) >= FILM_ACTOR_TARGET:
                    break
                film_actor_rows.append(
                    {
                        "actor_id": str(actor_id),
                        "film_id": str(film_id),
                        "last_update": synth.activity_timestamp("film_actor", actor_id, film_id),
                    }
                )
            if len(film_actor_rows) >= FILM_ACTOR_TARGET:
                break
        if len(film_actor_rows) < FILM_ACTOR_TARGET:
            raise SystemExit(
                f"film_actor: only {len(film_actor_rows)} pairs available; need {FILM_ACTOR_TARGET}"
            )
    _write_csv("film_actor", ["actor_id", "film_id", "last_update"], film_actor_rows[:FILM_ACTOR_TARGET])
    _log_progress(f"[spine] film_actor.csv written ({min(len(film_actor_rows), FILM_ACTOR_TARGET)} rows)")

    film_category_rows: list[dict[str, str]] = []
    for film_id in film_ids:
        cat_count = 1 + (_digest(film_id, "cats") % 2)
        for offset in range(cat_count):
            cat_id = FILM_CATEGORY_IDS[(_digest(film_id, "fcat", offset) % len(FILM_CATEGORY_IDS))]
            film_category_rows.append(
                {
                    "film_id": str(film_id),
                    "category_id": str(cat_id),
                    "last_update": synth.activity_timestamp("film_category", film_id, offset),
                }
            )
    _write_csv(
        "film_category",
        ["film_id", "category_id", "last_update"],
        film_category_rows,
    )
    _log_progress(f"[spine] film_category.csv written ({len(film_category_rows)} rows)")


def _bootstrap_spine_csvs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _bootstrap_film_csv_from_lexicon()
    _bootstrap_lexicon_csv("category", ["category_id", "name", "last_update"])
    _bootstrap_lexicon_csv("language", ["language_id", "name", "last_update"])
    _bootstrap_lexicon_csv("actor", ["actor_id", "first_name", "last_name", "last_update"])
    _bootstrap_lexicon_csv("country", ["country_id", "country", "last_update"])
    _bootstrap_lexicon_csv("customer", [
        "customer_id",
        "store_id",
        "first_name",
        "last_name",
        "email",
        "address_id",
        "activebool",
        "create_date",
        "last_update",
        "active",
    ])
    synthesize_rental_spine()


def _load_film_lexicon_source() -> list[dict[str, str]]:
    if (OUT_DIR / "film.csv").is_file():
        return _read_csv("film")
    _bootstrap_film_csv_from_lexicon()
    return _read_csv("film")


OBSOLETE_CSVS = (
    "film_category.csv",
    "customer_profile.csv",
    "customer_segment_history.csv",
    "customer_subscription.csv",
    "film_semantics.csv",
    "inventory_maintenance.csv",
    "promotion_campaign.csv",
    "refund.csv",
    "subscription_plan.csv",
    "support_ticket.csv",
)


def _digest(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def _build_interleaved_item_id_map() -> dict[tuple[str, int], int]:
    tagged: list[tuple[str, int]] = []
    for seq in range(_FILM_LEXICON_COUNT):
        tagged.append(("film", seq))
    for seq in range(BOOK_COUNT):
        tagged.append(("book", seq))
    for seq in range(GAME_COUNT):
        tagged.append(("game", seq))
    tagged.sort(key=lambda tag: _digest("interleave", tag[0], tag[1]))
    return {tag: idx + 1 for idx, tag in enumerate(tagged)}


def _item_id_for(item_type: str, seq: int) -> int:
    global _ITEM_ID_MAP
    if _ITEM_ID_MAP is None:
        _ITEM_ID_MAP = _build_interleaved_item_id_map()
    return _ITEM_ID_MAP[(item_type, seq)]


def _film_item_id(film_id: int) -> int:
    return _item_id_for("film", film_id - 1)


def _pick(options: tuple[str, ...], *parts: object) -> str:
    idx = _digest(DEFAULT_RANDOM_SEED, *parts) % len(options)
    return options[idx]


def _read_csv(name: str) -> list[dict[str, str]]:
    path = OUT_DIR / f"{name}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path = OUT_DIR / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _parse_ts(value: str) -> datetime:
    text = value.strip()
    if "+" in text:
        text = text.split("+", 1)[0]
    if text.endswith("Z"):
        text = text[:-1]
    if "." in text:
        text = text.split(".", 1)[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt.replace("%", "0"))], fmt)
        except ValueError:
            continue
    if len(text) >= 10:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    raise ValueError(f"unparseable timestamp: {value!r}")


def _fmt_ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_date(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _game_title_allowed(title: str) -> bool:
    clean = str(title or "").strip()
    if len(clean) < 2:
        return False
    lowered = clean.lower()
    if lowered in _GAME_TITLE_BLOCKLIST:
        return False
    return not any(token in lowered for token in _GAME_TITLE_BLOCKLIST)


def _load_seed_list(name: str) -> list[dict[str, str]]:
    path = _SEED_LISTS / f"{name}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if any(str(v or "").strip() for v in row.values())]
    if name == "games":
        rows = [row for row in rows if _game_title_allowed(str(row.get("title") or ""))]
    return rows


def _seed_row(seeds: list[dict[str, str]], index: int) -> dict[str, str]:
    return seeds[index % len(seeds)]


def _parse_special_features(raw: str) -> list[tuple[str, str]]:
    text = (raw or "").strip()
    if not text or text in ("{}", "[]"):
        return []
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1]
    elif text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            tokens = [str(x).strip() for x in parsed if str(x).strip()]
            return _map_feature_tokens(tokens)
        except json.JSONDecodeError:
            inner = text[1:-1]
    else:
        inner = text
    parts = re.findall(r'"([^"]+)"|([^,{}]+)', inner)
    tokens = [(left or right).strip() for left, right in parts if (left or right).strip()]
    return _map_feature_tokens(tokens)


def _map_feature_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower().strip()
        mapped = FEATURE_CANONICAL.get(key)
        if mapped and mapped[0] not in seen:
            out.append(mapped)
            seen.add(mapped[0])
    return out


def _sample_bounded_decimal(lo: Decimal, hi: Decimal, *parts: object) -> Decimal:
    cents = int((hi - lo) * 100)
    if cents <= 0:
        return lo
    return lo + Decimal(_digest(*parts) % (cents + 1)) / Decimal("100")


def _sample_bounded_int(lo: int, hi: int, *parts: object) -> int:
    if hi <= lo:
        return lo
    return lo + (_digest(*parts) % (hi - lo + 1))


def transform_catalog() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    films = _read_csv("film")
    if films and "film_id" not in films[0]:
        item_rows: list[dict[str, object]] = []
        for row in _read_csv("item"):
            item_id = int(row["item_id"])
            item_type = row.get("item_type") or "film"
            title = row.get("title") or ""
            if item_type == "film":
                release_year = str(2006 + (_digest(item_id, "year") % 31) - 15)
                description = synth.item_description("film", title, item_id)
            else:
                release_year = row.get("release_year") or ""
                description = row.get("description") or ""
            item_rows.append(
                {
                    "item_id": item_id,
                    "item_type": item_type,
                    "title": title,
                    "description": description,
                    "release_year": release_year,
                    "language_id": row["language_id"],
                    "rental_duration": row["rental_duration"],
                    "rental_rate": row["rental_rate"],
                    "replacement_cost": row["replacement_cost"],
                    "last_update": row["last_update"],
                }
            )
        [row for row in item_rows if row["item_type"] == "film"]
        _write_csv(
            "item",
            [
                "item_id",
                "item_type",
                "title",
                "description",
                "release_year",
                "language_id",
                "rental_duration",
                "rental_rate",
                "replacement_cost",
                "last_update",
            ],
            item_rows,
        )
        return films, item_rows
    item_rows: list[dict[str, object]] = []
    film_rows: list[dict[str, object]] = []
    for row in films:
        film_id = int(row["film_id"])
        item_id = _film_item_id(film_id)
        title = row.get("title") or ""
        release_year = str(1990 + (_digest(film_id, "year") % 35))
        item_rows.append(
            {
                "item_id": item_id,
                "item_type": "film",
                "title": title,
                "description": synth.item_description("film", title, item_id),
                "release_year": release_year,
                "language_id": 1 + (_digest(film_id, "flang") % 6),
                "rental_duration": row["rental_duration"],
                "rental_rate": row["rental_rate"],
                "replacement_cost": row["replacement_cost"],
                "last_update": synth.activity_timestamp("film_item", item_id),
            }
        )
        film_rows.append(
            {
                "item_id": item_id,
                "original_language_id": 1 + (_digest(film_id, "olang") % 6),
                "length": row.get("length") or "",
                "rating": row.get("rating") or "G",
                "last_update": synth.activity_timestamp("film", item_id),
            }
        )
    _write_csv(
        "item",
        [
            "item_id",
            "item_type",
            "title",
            "description",
            "release_year",
            "language_id",
            "rental_duration",
            "rental_rate",
            "replacement_cost",
            "last_update",
        ],
        item_rows,
    )
    _write_csv(
        "film",
        ["item_id", "original_language_id", "length", "rating", "last_update"],
        film_rows,
    )
    return films, item_rows


def generate_item_feature(source_films: list[dict[str, str]]) -> list[dict[str, object]]:
    existing = _read_csv("item_feature") if (OUT_DIR / "item_feature.csv").is_file() else []
    if existing:
        return existing
    films = source_films
    if not films or "special_features" not in films[0]:
        films = _load_film_lexicon_source()
    rows: list[dict[str, object]] = []
    for row in films:
        if "film_id" not in row:
            continue
        film_id = int(row["film_id"])
        item_id = _film_item_id(film_id)
        raw = row.get("special_features") or ""
        length = 0
        rating = "G"
        for film_row in films:
            if int(film_row.get("film_id") or film_row.get("item_id") or 0) == film_id:
                length = int(film_row.get("length") or 90)
                rating = str(film_row.get("rating") or "G")
                break
        rating_bonus = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4}.get(rating, 0)
        target_features = min(6, 1 + (length // 45) + rating_bonus)
        parsed = _parse_special_features(raw)
        while len(parsed) < target_features:
            extras = list(FEATURE_CANONICAL.values())
            pick = extras[len(parsed) % len(extras)]
            if pick not in parsed:
                parsed.append(pick)
            else:
                break
        for feature_name, feature_type in parsed[:target_features]:
            rows.append(
                {
                    "item_id": item_id,
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "last_update": row.get("last_update") or synth.activity_timestamp("feat", item_id),
                }
            )
    _write_csv(
        "item_feature",
        ["item_id", "feature_name", "feature_type", "last_update"],
        rows,
    )
    return rows


def _load_film_category_source() -> list[dict[str, str]]:
    film_cat_path = OUT_DIR / "film_category.csv"
    if film_cat_path.is_file():
        return _read_csv("film_category")
    films = _load_film_lexicon_source()
    out: list[dict[str, str]] = []
    for row in films:
        film_id = row.get("film_id") or row.get("item_id")
        if not film_id:
            continue
        fid = int(film_id)
        cat_id = FILM_CATEGORY_IDS[_digest(fid, "fcat") % len(FILM_CATEGORY_IDS)]
        out.append(
            {
                "film_id": str(fid),
                "category_id": str(cat_id),
                "last_update": synth.activity_timestamp("film_category", fid),
            }
        )
    return out


def transform_item_category() -> None:
    item_cat_path = OUT_DIR / "item_category.csv"
    if item_cat_path.is_file() and _read_csv("item_category"):
        return
    rows = _load_film_category_source()
    if not rows:
        return
    out = [
        {
            "item_id": str(_film_item_id(int(row["film_id"]))),
            "category_id": row["category_id"],
            "last_update": row["last_update"],
        }
        for row in rows
    ]
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in out:
        key = (row["item_id"], row["category_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    _write_csv("item_category", ["item_id", "category_id", "last_update"], deduped)


def transform_inventory() -> None:
    rows = _read_csv("inventory")
    if rows and "item_id" in rows[0]:
        return
    out = [
        {
            "inventory_id": row["inventory_id"],
            "item_id": str(_film_item_id(int(row["film_id"]))),
            "store_id": row["store_id"],
            "last_update": row["last_update"],
        }
        for row in rows
    ]
    _write_csv("inventory", ["inventory_id", "item_id", "store_id", "last_update"], out)


def transform_film_actor() -> None:
    rows = _read_csv("film_actor")
    if rows and "film_item_id" in rows[0]:
        return
    if rows and "item_id" in rows[0]:
        out = [
            {
                "actor_id": row["actor_id"],
                "film_item_id": row["item_id"],
                "last_update": row["last_update"],
            }
            for row in rows
        ]
    else:
        out = [
            {
                "actor_id": row["actor_id"],
                "film_item_id": str(_film_item_id(int(row["film_id"]))),
                "last_update": row["last_update"],
            }
            for row in rows
        ]
    _write_csv("film_actor", ["actor_id", "film_item_id", "last_update"], out)


def bootstrap_synthetic_spine() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Replace geo + retail spine with synthesized spread tables."""
    countries_in = _read_csv("country")
    country_rows, city_rows, address_rows = synth.synthesize_geo_tables(countries_in)
    _write_csv(
        "country",
        ["country_id", "country", "last_update"],
        country_rows,
    )
    _write_csv(
        "city",
        ["city_id", "city", "country_id", "last_update"],
        city_rows,
    )
    _write_csv(
        "address",
        [
            "address_id",
            "address",
            "district",
            "city_id",
            "postal_code",
            "phone",
            "last_update",
        ],
        address_rows,
    )
    actors = _read_csv("actor")
    for row in actors:
        row["last_update"] = synth.activity_timestamp("actor", row.get("actor_id"))
    _write_csv("actor", ["actor_id", "first_name", "last_name", "last_update"], actors)

    stores = synth.synthesize_stores(address_rows)
    staff = synth.synthesize_staff(
        stores,
        _require_lexicon("staff_names.jsonl", min_rows=STAFF_COUNT, fields=("first_name", "last_name")),
    )
    _write_csv(
        "store",
        ["store_id", "manager_staff_id", "address_id", "last_update"],
        stores,
    )
    _write_csv(
        "staff",
        [
            "staff_id",
            "first_name",
            "last_name",
            "address_id",
            "email",
            "store_id",
            "active",
            "username",
            "password",
            "ssn",
            "last_update",
        ],
        staff,
    )
    inventory = _read_csv("inventory")
    for row in inventory:
        iid = int(row["inventory_id"])
        row["store_id"] = str(synth.spread_fk(iid, synth.STORE_COUNT, "inv_store"))
        row["last_update"] = synth.activity_timestamp("inventory", iid)
    _write_csv("inventory", ["inventory_id", "item_id", "store_id", "last_update"], inventory)
    return country_rows, city_rows, address_rows


def transform_customer() -> list[dict[str, str]]:
    rows = _read_csv("customer")
    out = []
    for row in rows:
        cid = row["customer_id"]
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        email = _synth_email_local(first, last, cid)
        activebool = "true" if int(cid) % 5 else "false"
        out.append(
            {
                "customer_id": cid,
                "store_id": row["store_id"],
                "first_name": first,
                "last_name": last,
                "email": email,
                "address_id": row["address_id"],
                "activebool": activebool,
                "create_date": row.get("create_date") or "",
                "last_update": row.get("last_update") or "",
            }
        )
    out = synth.rebalance_customers(out, synth.STORE_COUNT)
    for row in out:
        cid = int(row["customer_id"])
        row["address_id"] = str(1 + ((cid - 1) % synth.ADDRESS_COUNT))
    _write_csv(
        "customer",
        [
            "customer_id",
            "store_id",
            "first_name",
            "last_name",
            "email",
            "address_id",
            "activebool",
            "create_date",
            "last_update",
        ],
        out,
    )
    return out


def transform_payment() -> list[dict[str, str]]:
    rows = _read_csv("payment")
    out = [
        {
            "payment_id": row["payment_id"],
            "rental_id": row["rental_id"],
            "amount": row["amount"],
            "payment_date": row["payment_date"],
        }
        for row in rows
    ]
    _write_csv("payment", ["payment_id", "rental_id", "amount", "payment_date"], out)
    return out


def _parse_author_name(raw: str) -> tuple[str, str] | None:
    text = str(raw or "").strip()
    if not text or text.lower() in ("unknown author", "unknown"):
        return None
    if "," in text:
        last, _, first = text.partition(",")
        first = first.strip()
        last = last.strip()
        if first and last:
            return first, last
    parts = text.split()
    if len(parts) < 2:
        return None
    particles = {"de", "del", "della", "van", "von", "le", "la", "du", "di"}
    if parts[0].lower() in particles and len(parts) >= 3:
        return " ".join(parts[:-1]), parts[-1]
    return parts[0], parts[-1]


def _isbn13_check_digit(body12: str) -> str:
    total = 0
    for idx, ch in enumerate(body12):
        digit = int(ch)
        total += digit * (1 if idx % 2 == 0 else 3)
    check = (10 - (total % 10)) % 10
    return str(check)


def _format_isbn13(item_id: int | str) -> str:
    body = f"978{_digest(item_id, 'isbn') % 1_000_000_000:09d}"
    check = _isbn13_check_digit(body)
    raw = body + check
    return f"{raw[0:3]}-{raw[3:4]}-{raw[4:10]}-{raw[10:12]}-{raw[12:]}"


def _collect_seed_authors(book_seeds: list[dict[str, str]]) -> list[tuple[str, str]]:
    seed_authors: list[tuple[str, str]] = []
    for seed in book_seeds:
        parsed = _parse_author_name(f"{seed.get('author_first', '')} {seed.get('author_last', '')}".strip())
        if parsed is None:
            parsed = _parse_author_name(str(seed.get("author_last") or ""))
        if parsed is None:
            continue
        if parsed not in seed_authors:
            seed_authors.append(parsed)
    return seed_authors


def _publisher_name_for_seed(seed: dict[str, str], publisher_names: list[str]) -> str:
    name = str(seed.get("publisher") or "").strip()
    if name and name.lower() != "independent":
        return name
    if not publisher_names:
        raise SystemExit("publisher name list is empty")
    return _pick(tuple(publisher_names), seed.get("title", ""), seed.get("author_first", ""), "publisher")


def _game_platform_for_item(item_id: int | str, seed: dict[str, str]) -> str:
    return _pick(PLATFORMS, item_id, "platform")


def _game_developer_for_item(item_id: int | str, seed: dict[str, str]) -> str:
    developer = str(seed.get("developer") or "").strip()
    if developer and developer.lower() != "independent":
        return developer
    return _pick(GAME_DEVELOPERS, item_id, "developer")


def _zenodo_developer_from_row(row: dict[str, str]) -> str:
    for key in ("developer", "publisher", "studio", "Developer", "Publisher", "Studio"):
        value = str(row.get(key) or "").strip()
        if value and value.lower() not in {"independent", "unknown", "n/a", ""}:
            return value
    return ""


def generate_authors_publishers(
    countries: list[dict[str, str]],
    book_seeds: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_authors = _collect_seed_authors(book_seeds)
    if not seed_authors:
        raise SystemExit("book seeds did not yield any parseable authors")
    author_rows: list[dict[str, object]] = []
    for author_id, (first, last) in enumerate(seed_authors, start=1):
        author_rows.append(
            {
                "author_id": author_id,
                "first_name": first,
                "last_name": last,
                "last_update": synth.activity_timestamp("author", author_id),
            }
        )
    publisher_names = synth.load_publisher_names()
    if len(publisher_names) < 40:
        raise SystemExit(
            f"publisher list needs at least 40 names, got {len(publisher_names)}"
        )
    publisher_rows: list[dict[str, object]] = []
    for publisher_id in range(1, 41):
        name = publisher_names[(publisher_id - 1) % len(publisher_names)]
        country_id = countries[_digest(publisher_id) % len(countries)]["country_id"]
        publisher_rows.append(
            {
                "publisher_id": publisher_id,
                "publisher_name": name,
                "country_id": country_id,
                "last_update": synth.activity_timestamp("publisher", publisher_id),
            }
        )
    _write_csv("author", ["author_id", "first_name", "last_name", "last_update"], author_rows)
    _write_csv(
        "publisher",
        ["publisher_id", "publisher_name", "country_id", "last_update"],
        publisher_rows,
    )
    return author_rows, publisher_rows


def _author_id_for_seed(
    seed: dict[str, str],
    author_rows: list[dict[str, object]],
) -> int:
    first = seed["author_first"]
    last = seed["author_last"]
    for row in author_rows:
        if row["first_name"] == first and row["last_name"] == last:
            return int(row["author_id"])
    return 1 + (_digest(first, last) % len(author_rows))


def _publisher_id_for_seed(
    seed: dict[str, str],
    publisher_rows: list[dict[str, object]],
    publisher_names: list[str],
) -> int:
    name = _publisher_name_for_seed(seed, publisher_names)
    for row in publisher_rows:
        if row["publisher_name"] == name:
            return int(row["publisher_id"])
    return 1 + (_digest(name) % len(publisher_rows))


def _books_games_current() -> bool:
    if not (OUT_DIR / "book.csv").is_file() or not _read_csv("book"):
        return False
    if not (OUT_DIR / "game.csv").is_file() or not _read_csv("game"):
        return False
    item_types = {int(row["item_id"]): row.get("item_type") for row in _read_csv("item")}
    book_ids = {i for i, t in item_types.items() if t == "book"}
    game_ids = {i for i, t in item_types.items() if t == "game"}
    return len(book_ids) >= BOOK_COUNT and len(game_ids) >= GAME_COUNT


def _sync_book_game_item_categories(categories: list[dict[str, str]]) -> None:
    book_category_ids = [str(c) for c in BOOK_CATEGORY_IDS]
    game_category_ids = [str(c) for c in GAME_CATEGORY_IDS]
    existing = _read_csv("item_category") if (OUT_DIR / "item_category.csv").is_file() else []
    seen = {(row["item_id"], row["category_id"]) for row in existing}
    rows = list(existing)
    for seq in range(BOOK_COUNT):
        item_id = _item_id_for("book", seq)
        row = {
            "item_id": str(item_id),
            "category_id": book_category_ids[_digest(item_id, "cat") % len(book_category_ids)],
            "last_update": synth.activity_timestamp("book_ic", item_id),
        }
        key = (row["item_id"], row["category_id"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    for seq in range(GAME_COUNT):
        item_id = _item_id_for("game", seq)
        row = {
            "item_id": str(item_id),
            "category_id": game_category_ids[_digest(item_id, "gcat") % len(game_category_ids)],
            "last_update": synth.activity_timestamp("game_ic", item_id),
        }
        key = (row["item_id"], row["category_id"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    _write_csv("item_category", ["item_id", "category_id", "last_update"], rows)


def generate_books_games(
    categories: list[dict[str, str]],
    languages: list[dict[str, str]],
    item_rows: list[dict[str, object]],
    author_rows: list[dict[str, object]],
    publisher_rows: list[dict[str, object]],
    book_seeds: list[dict[str, str]],
    game_seeds: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if _books_games_current():
        _sync_book_game_item_categories(categories)
        return (
            _read_csv("book"),
            _read_csv("game"),
            _read_csv("game_supported_language")
            if (OUT_DIR / "game_supported_language.csv").is_file()
            else [],
            [],
        )
    item_rows = [
        row for row in _read_csv("item") if row.get("item_type") == "film"
    ]
    book_category_ids = list(BOOK_CATEGORY_IDS)
    game_category_ids = list(GAME_CATEGORY_IDS)
    publisher_names = synth.load_publisher_names()
    language_ids = [int(lang["language_id"]) for lang in languages]
    book_subtype_rows: list[dict[str, object]] = []
    game_subtype_rows: list[dict[str, object]] = []
    game_lang_rows: list[dict[str, object]] = []
    item_category_rows: list[dict[str, object]] = []
    for i in range(BOOK_COUNT):
        item_id = _item_id_for("book", i)
        seed = _seed_row(book_seeds, i)
        author_id = _author_id_for_seed(seed, author_rows)
        publisher_id = _publisher_id_for_seed(seed, publisher_rows, publisher_names)
        year = 1995 + (_digest(item_id, "year") % 11)
        rate = Decimal("2.99") + Decimal(_digest(item_id, "rate") % 300) / Decimal("100")
        item_rows.append(
            {
                "item_id": item_id,
                "item_type": "book",
                "title": seed["title"],
                "description": synth.item_description("book", seed["title"], item_id),
                "release_year": year,
                "language_id": language_ids[_digest(item_id, "lang") % len(language_ids)],
                "rental_duration": 7,
                "rental_rate": _money(rate),
                "replacement_cost": _money(
                    Decimal("12.99") + (_digest(item_id, "rc") % 800) / Decimal("100")
                ),
                "last_update": synth.activity_timestamp("book_item", item_id),
            }
        )
        book_subtype_rows.append(
            {
                "item_id": item_id,
                "author_id": author_id,
                "publisher_id": publisher_id,
                "isbn": _format_isbn13(item_id),
                "page_count": 120 + (_digest(item_id, "pages") % 480),
                "last_update": synth.activity_timestamp("book", item_id),
            }
        )
        item_category_rows.append(
            {
                "item_id": item_id,
                "category_id": book_category_ids[_digest(item_id, "cat") % len(book_category_ids)],
                "last_update": synth.activity_timestamp("book_ic", item_id),
            }
        )
    for i in range(GAME_COUNT):
        item_id = _item_id_for("game", i)
        seed = _seed_row(game_seeds, i)
        year = 2000 + (_digest(item_id, "gyear") % 6)
        rate = Decimal("3.99") + Decimal(_digest(item_id, "grate") % 400) / Decimal("100")
        platform = _game_platform_for_item(item_id, seed)
        item_rows.append(
            {
                "item_id": item_id,
                "item_type": "game",
                "title": seed["title"],
                "description": synth.item_description("game", seed["title"], item_id),
                "release_year": year,
                "language_id": language_ids[_digest(item_id, "glang") % len(language_ids)],
                "rental_duration": 5,
                "rental_rate": _money(rate),
                "replacement_cost": _money(
                    Decimal("24.99") + (_digest(item_id, "grc") % 1500) / Decimal("100")
                ),
                "last_update": synth.activity_timestamp("game_item", item_id),
            }
        )
        game_subtype_rows.append(
            {
                "item_id": item_id,
                "platform": platform,
                "developer": _game_developer_for_item(item_id, seed),
                "esrb_rating": _pick(ESRB_RATINGS, item_id, "esrb"),
                "last_update": synth.activity_timestamp("game", item_id),
            }
        )
        lang_count = 1 + (_digest(item_id, "lc") % 2)
        chosen_langs: set[int] = set()
        for j in range(lang_count):
            lang_id = language_ids[_digest(item_id, j) % len(language_ids)]
            if lang_id in chosen_langs:
                continue
            chosen_langs.add(lang_id)
            game_lang_rows.append(
                {
                    "item_id": item_id,
                    "language_id": lang_id,
                    "last_update": synth.activity_timestamp("game_lang", item_id, j),
                }
            )
        item_category_rows.append(
            {
                "item_id": item_id,
                "category_id": game_category_ids[_digest(item_id, "gcat") % len(game_category_ids)],
                "last_update": synth.activity_timestamp("game_ic", item_id),
            }
        )
    _write_csv(
        "item",
        [
            "item_id",
            "item_type",
            "title",
            "description",
            "release_year",
            "language_id",
            "rental_duration",
            "rental_rate",
            "replacement_cost",
            "last_update",
        ],
        item_rows,
    )
    _write_csv(
        "book",
        ["item_id", "author_id", "publisher_id", "isbn", "page_count", "last_update"],
        book_subtype_rows,
    )
    _write_csv(
        "game",
        ["item_id", "platform", "developer", "esrb_rating", "last_update"],
        game_subtype_rows,
    )
    _write_csv(
        "game_supported_language",
        ["item_id", "language_id", "last_update"],
        game_lang_rows,
    )
    existing_ic = _read_csv("item_category") if (OUT_DIR / "item_category.csv").is_file() else []
    merged_ic = existing_ic + [{k: str(v) for k, v in row.items()} for row in item_category_rows]
    seen: set[tuple[str, str]] = set()
    all_ic: list[dict[str, str]] = []
    for row in merged_ic:
        key = (row["item_id"], row["category_id"])
        if key in seen:
            continue
        seen.add(key)
        all_ic.append(row)
    _write_csv("item_category", ["item_id", "category_id", "last_update"], all_ic)
    return book_subtype_rows, game_subtype_rows, game_lang_rows, item_category_rows


def append_book_game_inventory(
    book_rows: list[dict[str, object]],
    game_rows: list[dict[str, object]],
) -> tuple[list[dict[str, str]], dict[int, int]]:
    inventory = _read_csv("inventory")
    item_types = {int(row["item_id"]): row.get("item_type") for row in _read_csv("item")}
    if any(item_types.get(int(r["item_id"])) in ("book", "game") for r in inventory):
        item_to_inventory = {
            int(r["item_id"]): int(r["inventory_id"])
            for r in inventory
            if item_types.get(int(r["item_id"])) in ("book", "game")
        }
        return inventory, item_to_inventory
    max_inv = max(int(r["inventory_id"]) for r in inventory)
    item_to_inventory: dict[int, int] = {}
    next_id = max_inv + 1
    for row in book_rows + game_rows:
        item_id = int(row["item_id"])
        store_id = synth.spread_fk(item_id, synth.STORE_COUNT, "bg_store")
        inventory.append(
            {
                "inventory_id": str(next_id),
                "item_id": str(item_id),
                "store_id": str(store_id),
                "last_update": synth.activity_timestamp("bg_inv", next_id),
            }
        )
        item_to_inventory[item_id] = next_id
        next_id += 1
    _write_csv("inventory", ["inventory_id", "item_id", "store_id", "last_update"], inventory)
    return inventory, item_to_inventory


def append_book_game_rentals_payments(
    item_to_inventory: dict[int, int],
    customers: list[dict[str, str]],
    rentals: list[dict[str, str]],
    payments: list[dict[str, str]],
    skip_inventory_ids: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inv_rental_counts: dict[int, int] = {}
    for row in rentals:
        inv_id = int(row["inventory_id"])
        inv_rental_counts[inv_id] = inv_rental_counts.get(inv_id, 0) + 1
    max_rental = max(int(r["rental_id"]) for r in rentals)
    max_payment = max(int(r["payment_id"]) for r in payments)
    rental_id = max_rental + 1
    payment_id = max_payment + 1
    for item_id, inventory_id in sorted(item_to_inventory.items()):
        if str(inventory_id) in skip_inventory_ids:
            continue
        target_copies = 5 + (_digest(item_id, "copies") % 6)
        existing = inv_rental_counts.get(inventory_id, 0)
        for copy in range(existing, target_copies):
            customer = customers[_digest(item_id, copy, "cust") % len(customers)]
            rental_dt = _rental_timestamp_for(rental_id)
            item_row = _read_csv("item")
            items_map = {int(r["item_id"]): r for r in item_row}
            raw_dur = str(items_map.get(item_id, {}).get("rental_duration") or "3")
            try:
                duration = max(1, int(raw_dur))
            except ValueError:
                duration = 3
            return_date = compute_return_date(rental_dt, duration, rental_id)
            rentals.append(
                {
                    "rental_id": str(rental_id),
                    "rental_date": _fmt_ts(rental_dt),
                    "inventory_id": str(inventory_id),
                    "customer_id": customer["customer_id"],
                    "return_date": return_date,
                    "staff_id": "1" if _digest(item_id, copy) % 2 else "2",
                    "last_update": _fmt_ts(rental_dt),
                }
            )
            amount = Decimal("1.99") + Decimal(_digest(item_id, copy, "amt") % 800) / Decimal("100")
            pay_dt = _parse_ts(return_date) if return_date else rental_dt + timedelta(hours=2)
            if pay_dt > _activity_as_of():
                pay_dt = _activity_as_of() - timedelta(hours=1)
            payments.append(
                {
                    "payment_id": str(payment_id),
                    "rental_id": str(rental_id),
                    "amount": _money(amount),
                    "payment_date": _fmt_ts(pay_dt),
                }
            )
            rental_id += 1
            payment_id += 1
            inv_rental_counts[inventory_id] = inv_rental_counts.get(inventory_id, 0) + 1
    _write_csv(
        "rental",
        [
            "rental_id",
            "rental_date",
            "inventory_id",
            "customer_id",
            "return_date",
            "staff_id",
            "last_update",
        ],
        rentals,
    )
    _write_csv(
        "payment",
        ["payment_id", "rental_id", "amount", "payment_date"],
        payments,
    )
    return rentals, payments


def _no_rental_inventory_ids(inventory: list[dict[str, str]]) -> set[str]:
    target = int(round(len(inventory) * float(NO_RENTAL_INVENTORY_RATE)))
    ranked = sorted(
        inventory,
        key=lambda row: _digest("no_rental", row["inventory_id"]),
    )
    return {row["inventory_id"] for row in ranked[:target]}


def _sync_payments_to_rentals(
    rentals: list[dict[str, str]],
    payments: list[dict[str, str]],
) -> list[dict[str, str]]:
    rental_ids = {r["rental_id"] for r in rentals}
    synced = [p for p in payments if p["rental_id"] in rental_ids]
    _write_csv("payment", ["payment_id", "rental_id", "amount", "payment_date"], synced)
    return synced


def apply_benchmark_patterns(
    inventory: list[dict[str, str]],
    rentals: list[dict[str, str]],
    deliveries: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], set[str]]:
    no_rental_ids = _no_rental_inventory_ids(inventory)
    rentals = [
        r for r in rentals if r["inventory_id"] not in no_rental_ids
    ]
    target_open = int(round(len(rentals) * float(OPEN_RENTAL_RATE)))
    open_indices = [
        i
        for i, r in enumerate(rentals)
        if not str(r.get("return_date") or "").strip()
    ]
    closed_indices = [
        i
        for i, r in enumerate(rentals)
        if str(r.get("return_date") or "").strip()
    ]
    if len(open_indices) < target_open:
        need = target_open - len(open_indices)
        for idx in sorted(closed_indices, key=lambda i: _digest(i, "open"))[:need]:
            rentals[idx]["return_date"] = ""
    elif len(open_indices) > target_open:
        excess = len(open_indices) - target_open
        items_by_id: dict[int, dict[str, str]] = {}
        if (OUT_DIR / "item.csv").is_file():
            items_by_id = {int(r["item_id"]): r for r in _read_csv("item")}
        inv_to_item = {r["inventory_id"]: int(r["item_id"]) for r in _read_csv("inventory")}
        for idx in sorted(open_indices, key=lambda i: _digest(i, "close"))[:excess]:
            rental_date = _parse_ts(rentals[idx]["rental_date"])
            item_id = inv_to_item.get(rentals[idx]["inventory_id"])
            duration = 3
            if item_id is not None:
                item = items_by_id.get(item_id, {})
                raw = str(item.get("rental_duration") or "3")
                try:
                    duration = max(1, int(raw))
                except ValueError:
                    duration = 3
            rentals[idx]["return_date"] = _fmt_ts(
                min(rental_date + timedelta(days=duration), _activity_as_of())
            )
    _write_csv(
        "rental",
        [
            "rental_id",
            "rental_date",
            "inventory_id",
            "customer_id",
            "return_date",
            "staff_id",
            "last_update",
        ],
        rentals,
    )
    rentals = enforce_return_dates_at_anchor(rentals)
    if deliveries is not None:
        target_failed = int(round(len(deliveries) * float(FAILED_DELIVERY_RATE)))
        for _i, row in enumerate(
            sorted(range(len(deliveries)), key=lambda j: _digest(j, "fail"))[:target_failed]
        ):
            deliveries[row]["status"] = "failed"
            deliveries[row]["delivered_at"] = ""
        _write_csv(
            "delivery",
            [
                "delivery_id",
                "rental_id",
                "courier_id",
                "address_id",
                "dispatched_at",
                "delivered_at",
                "status",
                "delivery_fee",
                "tracking_number",
                "last_update",
            ],
            deliveries,
        )
    return rentals, inventory, no_rental_ids


def generate_couriers(countries: list[dict[str, str]]) -> list[dict[str, object]]:
    lexicon_rows = _require_lexicon("courier_names.jsonl", min_rows=8, fields=("courier_name",))
    names = [_lexicon_field(row, "courier_name") for row in lexicon_rows]
    rows: list[dict[str, object]] = []
    for courier_id in range(1, min(len(names), 15) + 1):
        rows.append(
            {
                "courier_id": courier_id,
                "courier_name": names[courier_id - 1],
                "phone": f"555-01{int(courier_id):02d}",
                "country_id": countries[_digest(courier_id) % len(countries)]["country_id"],
                "is_active": "t" if courier_id % 5 else "f",
                "last_update": synth.activity_timestamp("courier", courier_id),
            }
        )
    _write_csv(
        "courier",
        ["courier_id", "courier_name", "phone", "country_id", "is_active", "last_update"],
        rows,
    )
    return rows


def generate_suppliers(countries: list[dict[str, str]]) -> list[dict[str, object]]:
    names = synth.load_supplier_names()
    if len(names) < 12:
        raise SystemExit(f"suppliers.csv must contain at least 12 names, got {len(names)}")
    rows: list[dict[str, object]] = []
    for supplier_id in range(1, min(13, len(names) + 1)):
        rows.append(
            {
                "supplier_id": supplier_id,
                "supplier_name": names[(supplier_id - 1) % len(names)],
                "country_id": countries[_digest("sup", supplier_id) % len(countries)]["country_id"],
                "is_active": "t" if supplier_id % 5 else "f",
                "last_update": synth.activity_timestamp("supplier", supplier_id),
            }
        )
    _write_csv(
        "supplier",
        ["supplier_id", "supplier_name", "country_id", "is_active", "last_update"],
        rows,
    )
    return rows


def generate_warehouses(addresses: list[dict[str, str]]) -> list[dict[str, object]]:
    lexicon_rows = _require_lexicon("warehouse_names.jsonl", min_rows=WAREHOUSE_COUNT, fields=("warehouse_name",))
    labels = [_lexicon_field(row, "warehouse_name") for row in lexicon_rows]
    rows: list[dict[str, object]] = []
    for warehouse_id in range(1, synth.WAREHOUSE_COUNT + 1):
        addr = addresses[_digest("wh", warehouse_id) % len(addresses)]
        label = labels[(warehouse_id - 1) % len(labels)]
        rows.append(
            {
                "warehouse_id": warehouse_id,
                "warehouse_name": label,
                "address_id": addr["address_id"],
                "capacity": 5000 + warehouse_id * 1000,
                "last_update": synth.activity_timestamp("warehouse", warehouse_id),
            }
        )
    _write_csv(
        "warehouse",
        ["warehouse_id", "warehouse_name", "address_id", "capacity", "last_update"],
        rows,
    )
    return rows


def generate_promotions() -> list[dict[str, object]]:
    promo_names = [_lexicon_field(row, "promo_name") for row in _require_lexicon(
        "promotion_names.jsonl", min_rows=25, fields=("promo_name",)
    )]
    rows: list[dict[str, object]] = []
    for promotion_id in range(1, 26):
        start = synth.ACTIVITY_START + timedelta(
            days=_digest("promo", promotion_id) % 800
        )
        end = start + timedelta(days=30 + (_digest(promotion_id) % 60))
        promo_type = _pick(PROMO_TYPES, promotion_id)
        rows.append(
            {
                "promotion_id": promotion_id,
                "promo_name": promo_names[(promotion_id - 1) % len(promo_names)],
                "promo_type": promo_type,
                "discount_pct": _money(
                    _sample_bounded_decimal(Decimal("5"), Decimal("50"), promotion_id, "disc")
                ),
                "start_date": _fmt_date(start),
                "end_date": _fmt_date(end),
                "is_active": "t" if promotion_id % 7 else "f",
                "last_update": _fmt_ts(start),
            }
        )
    _write_csv(
        "promotion",
        [
            "promotion_id",
            "promo_name",
            "promo_type",
            "discount_pct",
            "start_date",
            "end_date",
            "is_active",
            "last_update",
        ],
        rows,
    )
    return rows


def generate_deliveries(
    rentals: list[dict[str, str]],
    couriers: list[dict[str, object]],
    customers: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    delivery_id = 1
    customer_by_id = {c["customer_id"]: c for c in customers}
    for rental in rentals:
        rental_id = int(rental["rental_id"])
        if _digest(rental_id, "del") % 7 != 0:
            continue
        customer = customer_by_id[rental["customer_id"]]
        rental_date = _parse_ts(rental["rental_date"])
        dispatch_base = rental_date + timedelta(days=_digest(rental_id, "disp") % 3)
        dispatched_at = _synth_business_hours_ts(dispatch_base, rental_id, "dispatch")
        dispatched = _parse_ts(dispatched_at)
        status = _pick(DELIVERY_STATUSES, rental_id, "status")
        delivered_at = ""
        if status == "delivered":
            delivered_at = _synth_business_hours_ts(
                dispatched + timedelta(days=1 + (_digest(rental_id) % 3)),
                rental_id,
                "delivered",
            )
        rows.append(
            {
                "delivery_id": delivery_id,
                "rental_id": rental_id,
                "courier_id": couriers[_digest(rental_id) % len(couriers)]["courier_id"],
                "address_id": customer["address_id"],
                "dispatched_at": dispatched_at,
                "delivered_at": delivered_at,
                "status": status,
                "delivery_fee": _money(
                    _sample_bounded_decimal(Decimal("2"), Decimal("25"), rental_id, "fee")
                ),
                "tracking_number": f"TRK{rental_id:08d}",
                "last_update": _fmt_ts(dispatched),
            }
        )
        delivery_id += 1
    _write_csv(
        "delivery",
        [
            "delivery_id",
            "rental_id",
            "courier_id",
            "address_id",
            "dispatched_at",
            "delivered_at",
            "status",
            "delivery_fee",
            "tracking_number",
            "last_update",
        ],
        rows,
    )
    return rows


def generate_purchase_orders(
    suppliers: list[dict[str, object]],
    items: list[dict[str, object]],
) -> None:
    po_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    line_id = 1
    for po_id in range(1, 41):
        status = _pick(PO_STATUSES, po_id)
        ordered = synth.ACTIVITY_START + timedelta(days=_digest(po_id, "ord") % 800)
        received = ""
        if status == "received":
            received = _fmt_date(ordered + timedelta(days=7 + (_digest(po_id) % 14)))
        po_rows.append(
            {
                "po_id": po_id,
                "supplier_id": suppliers[_digest(po_id) % len(suppliers)]["supplier_id"],
                "store_id": str(synth.spread_fk(po_id, synth.STORE_COUNT, "po_store")),
                "ordered_date": _fmt_date(ordered),
                "received_date": received,
                "status": status,
                "last_update": _fmt_ts(ordered),
            }
        )
        line_count = 2 + (_digest(po_id, "lines") % 4)
        for j in range(line_count):
            item = items[_digest(po_id, j, "item") % len(items)]
            line_rows.append(
                {
                    "line_id": line_id,
                    "po_id": po_id,
                    "item_id": item["item_id"],
                    "quantity": _sample_bounded_int(1, 50, line_id),
                    "unit_cost": _money(
                        _sample_bounded_decimal(Decimal("5"), Decimal("500"), line_id, "cost")
                    ),
                    "last_update": _fmt_ts(ordered),
                }
            )
            line_id += 1
    _write_csv(
        "purchase_order",
        ["po_id", "supplier_id", "store_id", "ordered_date", "received_date", "status", "last_update"],
        po_rows,
    )
    _write_csv(
        "purchase_line",
        ["line_id", "po_id", "item_id", "quantity", "unit_cost", "last_update"],
        line_rows,
    )


def generate_stock_transfers(
    warehouses: list[dict[str, object]],
    items: list[dict[str, object]],
) -> None:
    rows: list[dict[str, object]] = []
    for transfer_id in range(1, 201):
        transferred = synth.ACTIVITY_START + timedelta(
            seconds=_digest("xfer", transfer_id) % (4 * 365 * 86400)
        )
        item = items[_digest(transfer_id) % len(items)]
        rows.append(
            {
                "transfer_id": transfer_id,
                "item_id": item["item_id"],
                "from_warehouse_id": warehouses[_digest(transfer_id, "wh") % len(warehouses)][
                    "warehouse_id"
                ],
                "to_store_id": 1 + (_digest(transfer_id, "ts") % 2),
                "quantity": _sample_bounded_int(1, 100, transfer_id, "qty"),
                "transferred_at": _fmt_ts(transferred),
                "last_update": _fmt_ts(transferred),
            }
        )
    _write_csv(
        "stock_transfer",
        [
            "transfer_id",
            "item_id",
            "from_warehouse_id",
            "to_store_id",
            "quantity",
            "transferred_at",
            "last_update",
        ],
        rows,
    )


def generate_promotion_redemptions(
    promotions: list[dict[str, object]],
    rentals: list[dict[str, str]],
) -> None:
    rows: list[dict[str, object]] = []
    redemption_id = 1
    for rental in rentals:
        rental_id = int(rental["rental_id"])
        if _digest(rental_id, "red") % 8 != 0:
            continue
        promo = promotions[_digest(rental_id) % len(promotions)]
        promo_id = int(promo["promotion_id"])
        if promo_id in UNUSED_PROMO_IDS:
            continue
        rental_date = _parse_ts(rental["rental_date"])
        rows.append(
            {
                "redemption_id": redemption_id,
                "promotion_id": promo_id,
                "rental_id": rental_id,
                "discount_amount": _money(
                    _sample_bounded_decimal(Decimal("1"), Decimal("30"), rental_id, "disc")
                ),
                "redeemed_at": _fmt_ts(rental_date + timedelta(days=_digest(rental_id, "redeem") % 7)),
                "last_update": _fmt_ts(rental_date),
            }
        )
        redemption_id += 1
    _write_csv(
        "promotion_redemption",
        [
            "redemption_id",
            "promotion_id",
            "rental_id",
            "discount_amount",
            "redeemed_at",
            "last_update",
        ],
        rows,
    )


def generate_inventory_status_history(
    inventory: list[dict[str, str]],
    rentals: list[dict[str, str]],
) -> None:
    rented_inv: dict[str, list[dict[str, str]]] = {}
    for rental in rentals:
        rented_inv.setdefault(rental["inventory_id"], []).append(rental)
    rows: list[dict[str, object]] = []
    status_id = 1
    for inv in inventory:
        inv_id = inv["inventory_id"]
        history_rentals = rented_inv.get(inv_id, [])
        if not history_rentals and _digest(inv_id, "ish") % 4 != 0:
            continue
        changed = synth.ACTIVITY_START + timedelta(
            days=_digest(inv_id, "ish0") % 900
        )
        rows.append(
            {
                "status_id": status_id,
                "inventory_id": int(inv_id),
                "status": "available",
                "changed_at": _fmt_ts(changed),
                "staff_id": 1 + (_digest(inv_id, "st") % 2),
                "last_update": _fmt_ts(changed),
            }
        )
        status_id += 1
        if history_rentals:
            rental = sorted(history_rentals, key=lambda r: r["rental_date"])[0]
            changed = _parse_ts(rental["rental_date"])
            rows.append(
                {
                    "status_id": status_id,
                    "inventory_id": int(inv_id),
                    "status": "rented",
                    "changed_at": _fmt_ts(changed),
                    "staff_id": int(rental["staff_id"]),
                    "last_update": _fmt_ts(changed),
                }
            )
            status_id += 1
            if rental.get("return_date") and str(rental["return_date"]).strip():
                returned = _parse_ts(rental["return_date"])
                final_status = _pick(INVENTORY_STATUSES, inv_id, "final")
                if final_status in ("rented", "available"):
                    final_status = "available"
                rows.append(
                    {
                        "status_id": status_id,
                        "inventory_id": int(inv_id),
                        "status": final_status,
                        "changed_at": _fmt_ts(returned),
                        "staff_id": int(rental["staff_id"]),
                        "last_update": _fmt_ts(returned),
                    }
                )
                status_id += 1
    _write_csv(
        "inventory_status_history",
        ["status_id", "inventory_id", "status", "changed_at", "staff_id", "last_update"],
        rows,
    )


def generate_damage_reports(
    rentals: list[dict[str, str]],
) -> None:
    rows: list[dict[str, object]] = []
    damage_id = 1
    for rental in rentals:
        rental_id = int(rental["rental_id"])
        if not rental.get("return_date") or not str(rental["return_date"]).strip():
            continue
        if _digest(rental_id, "dmg") % 25 != 0:
            continue
        reported = _parse_ts(rental["return_date"]) + timedelta(hours=_digest(rental_id) % 48)
        rows.append(
            {
                "damage_id": damage_id,
                "rental_id": rental_id,
                "inventory_id": int(rental["inventory_id"]),
                "reported_by_staff_id": int(rental["staff_id"]),
                "severity": _pick(DAMAGE_SEVERITIES, rental_id, "sev"),
                "repair_cost": _money(
                    _sample_bounded_decimal(Decimal("10"), Decimal("500"), rental_id, "cost")
                ),
                "reported_at": _fmt_ts(reported),
                "last_update": _fmt_ts(reported),
            }
        )
        damage_id += 1
    _write_csv(
        "damage_report",
        [
            "damage_id",
            "rental_id",
            "inventory_id",
            "reported_by_staff_id",
            "severity",
            "repair_cost",
            "reported_at",
            "last_update",
        ],
        rows,
    )


def _csv_int(value: object) -> str:
    if value is None or str(value).strip() == "":
        return ""
    return str(int(float(str(value))))


def generate_reservations(
    customers: list[dict[str, str]],
    items: list[dict[str, object]],
    rentals: list[dict[str, str]],
) -> None:
    rental_by_customer: dict[str, list[dict[str, str]]] = {}
    for rental in rentals:
        rental_by_customer.setdefault(rental["customer_id"], []).append(rental)
    rows: list[dict[str, object]] = []
    reservation_id = 1
    for customer in customers:
        customer_id = int(customer["customer_id"])
        if _digest(customer_id, "res") % 5 != 0:
            continue
        item = items[_digest(customer_id, "item") % len(items)]
        item_id = int(item["item_id"])
        store_id = int(customer["store_id"])
        reserved = synth.ACTIVITY_START + timedelta(days=_digest(customer_id, "rsv") % 800)
        expires = reserved + timedelta(days=2 + (_digest(customer_id, "exp") % 5))
        status = _pick(RESERVATION_STATUSES, customer_id, "status")
        fulfilled_rental_id = ""
        if status == "fulfilled":
            cust_rentals = rental_by_customer.get(str(customer_id), [])
            if cust_rentals:
                fulfilled_rental_id = _csv_int(
                    sorted(cust_rentals, key=lambda r: r["rental_date"])[0]["rental_id"]
                )
            else:
                status = "expired"
        rows.append(
            {
                "reservation_id": _csv_int(reservation_id),
                "customer_id": _csv_int(customer_id),
                "item_id": _csv_int(item_id),
                "store_id": _csv_int(store_id),
                "reserved_at": _fmt_ts(reserved),
                "expires_at": _fmt_ts(expires),
                "fulfilled_rental_id": fulfilled_rental_id,
                "status": status,
                "last_update": _fmt_ts(reserved),
            }
        )
        reservation_id += 1
    _write_csv(
        "reservation",
        [
            "reservation_id",
            "customer_id",
            "item_id",
            "store_id",
            "reserved_at",
            "expires_at",
            "fulfilled_rental_id",
            "status",
            "last_update",
        ],
        rows,
    )


def remove_obsolete_csvs() -> None:
    for name in OBSOLETE_CSVS:
        path = OUT_DIR / name
        if path.is_file():
            path.unlink()


def _load_fk_rules() -> list[tuple[str, str, str, str]]:
    ddl = _DDL_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"ALTER TABLE (\w+) ADD CONSTRAINT \w+ FOREIGN KEY \((\w+)\) REFERENCES (\w+) \((\w+)\)"
    )
    matches = pattern.finditer(ddl)
    return [m.groups() for m in matches]


def verify_csv_integrity() -> list[str]:
    errors: list[str] = []
    cache: dict[str, list[dict[str, str]]] = {}

    def rows(table: str) -> list[dict[str, str]]:
        if table not in cache:
            path = OUT_DIR / f"{table}.csv"
            cache[table] = _read_csv(table) if path.is_file() else []
        return cache[table]

    for child, col, parent, parent_col in _load_fk_rules():
        child_rows = rows(child)
        parent_rows = rows(parent)
        if not child_rows or not parent_rows:
            continue
        parent_keys = {
            str(r[parent_col])
            for r in parent_rows
            if parent_col in r and str(r.get(parent_col, "")).strip() != ""
        }
        for idx, row in enumerate(child_rows, start=2):
            raw = row.get(col)
            if raw is None or str(raw).strip() == "":
                continue
            if str(raw) not in parent_keys:
                errors.append(
                    f"{child}.csv row {idx}: {col}={raw!r} missing in {parent}.{parent_col}"
                )
    return errors


def verify_csv_semantics(*, as_of: datetime | None = None) -> list[str]:
    errors: list[str] = []
    anchor = as_of or _activity_as_of()
    anchor_date = anchor.date()
    recent_start = anchor - timedelta(days=RECENT_WINDOW_DAYS)

    items = {int(r["item_id"]): r for r in _read_csv("item")} if (OUT_DIR / "item.csv").is_file() else {}
    inv_to_item = {r["inventory_id"]: int(r["item_id"]) for r in _read_csv("inventory")} if (OUT_DIR / "inventory.csv").is_file() else {}
    rentals = _read_csv("rental") if (OUT_DIR / "rental.csv").is_file() else []

    open_count = 0
    overdue_count = 0
    recent_count = 0
    for idx, row in enumerate(rentals, start=2):
        rental_date = _parse_ts(row["rental_date"])
        if rental_date > anchor:
            errors.append(f"rental.csv row {idx}: rental_date {row['rental_date']!r} after anchor")
        return_raw = str(row.get("return_date") or "").strip()
        if return_raw:
            return_date = _parse_ts(return_raw)
            if return_date < rental_date:
                errors.append(
                    f"rental.csv row {idx}: return_date {return_raw!r} before rental_date {row['rental_date']!r}"
                )
            if return_date > anchor:
                errors.append(f"rental.csv row {idx}: return_date {return_raw!r} after anchor")
        else:
            open_count += 1
            item_id = inv_to_item.get(row["inventory_id"])
            duration = 3
            if item_id is not None:
                raw = str(items.get(item_id, {}).get("rental_duration") or "3")
                try:
                    duration = max(1, int(raw))
                except ValueError:
                    duration = 3
            due = rental_date.date() + timedelta(days=duration)
            if due < anchor_date:
                overdue_count += 1
        if rental_date >= recent_start:
            recent_count += 1

    payment_rentals = {r["rental_id"] for r in _read_csv("payment")} if (OUT_DIR / "payment.csv").is_file() else set()
    for idx, row in enumerate(rentals, start=2):
        if row["rental_id"] not in payment_rentals:
            errors.append(f"rental.csv row {idx}: rental_id {row['rental_id']!r} has no payment row")

    if rentals and recent_count == 0:
        errors.append("rental.csv: no rentals in the recent window before anchor")

    print(
        f"Semantics: open={open_count} overdue={overdue_count} "
        f"recent_{RECENT_WINDOW_DAYS}d={recent_count} anchor={anchor.isoformat(sep=' ')}"
    )
    return errors


def snapshot_item_descriptions(csv_dir: Path) -> dict[int, str]:
    path = csv_dir / "item.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["item_id"]): str(row.get("description") or "")
        for row in rows
        if str(row.get("description") or "").strip()
    }


def restore_item_descriptions(items: list[dict[str, str]], snapshot: dict[int, str]) -> None:
    for row in items:
        item_id = int(row["item_id"])
        if item_id in snapshot and snapshot[item_id]:
            row["description"] = snapshot[item_id]


def realign_rental_lifecycle(rentals: list[dict[str, str]], inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    items = {int(r["item_id"]): r for r in _read_csv("item")}
    inv_to_item = {r["inventory_id"]: int(r["item_id"]) for r in inventory}
    films_by_id: dict[int, dict[str, str]] = {}
    if (OUT_DIR / "film.csv").is_file():
        for row in _read_csv("film"):
            films_by_id[int(row["item_id"])] = row
    out: list[dict[str, str]] = []
    for row in rentals:
        row = dict(row)
        rental_id = int(row["rental_id"])
        rental_dt = _rental_timestamp_for(rental_id)
        row["rental_date"] = _fmt_ts(rental_dt)
        item_id = inv_to_item.get(row["inventory_id"])
        duration = 3
        if item_id is not None:
            item = items.get(item_id, {})
            raw = str(item.get("rental_duration") or "3")
            try:
                duration = max(1, int(raw))
            except ValueError:
                duration = 3
        row["return_date"] = compute_return_date(rental_dt, duration, rental_id)
        row["last_update"] = synth.activity_timestamp("rental", rental_id, "lu")
        out.append(row)
    return enforce_inventory_rental_sequence(out)


def realign_dependent_tables(
    rentals: list[dict[str, str]],
    payments: list[dict[str, str]],
    customers: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rentals = realign_rental_lifecycle(rentals, _read_csv("inventory"))
    payments = _align_payment_amounts(rentals, _sync_payments_to_rentals(rentals, payments))
    customers = _finalize_customer_create_dates(customers, rentals)
    return rentals, payments, customers


def _print_row_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(OUT_DIR.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        counts[path.stem] = row_count
        print(f"  {path.stem}: {row_count}")
    return counts


def _finalize_customer_create_dates(
    customers: list[dict[str, str]],
    rentals: list[dict[str, str]],
) -> list[dict[str, str]]:
    first_rental: dict[str, datetime] = {}
    for rental in rentals:
        cid = rental["customer_id"]
        when = _parse_ts(rental["rental_date"])
        if cid not in first_rental or when < first_rental[cid]:
            first_rental[cid] = when
    out: list[dict[str, str]] = []
    for row in customers:
        row = dict(row)
        cid = row["customer_id"]
        if cid in first_rental:
            offset_days = 30 + (_digest(cid, "create_before") % 720)
            create = first_rental[cid] - timedelta(days=offset_days)
            if create < ACTIVITY_START:
                create = ACTIVITY_START + timedelta(days=_digest(cid, "create_floor") % 30)
            row["create_date"] = _fmt_date(create)
        out.append(row)
    return out


def _align_payment_amounts(
    rentals: list[dict[str, str]],
    payments: list[dict[str, str]],
) -> list[dict[str, str]]:
    items = {int(row["item_id"]): row for row in _read_csv("item")}
    inv_to_item = {row["inventory_id"]: int(row["item_id"]) for row in _read_csv("inventory")}
    rental_by_id = {row["rental_id"]: row for row in rentals}
    discount_by_rental: dict[str, Decimal] = {}
    if (OUT_DIR / "promotion_redemption.csv").is_file():
        for row in _read_csv("promotion_redemption"):
            discount_by_rental[row["rental_id"]] = discount_by_rental.get(row["rental_id"], Decimal("0")) + Decimal(
                str(row.get("discount_amount") or "0")
            )
    out: list[dict[str, str]] = []
    for payment in payments:
        row = dict(payment)
        rental = rental_by_id.get(row["rental_id"])
        if not rental:
            out.append(row)
            continue
        item_id = inv_to_item.get(rental["inventory_id"])
        item = items.get(item_id or -1, {})
        rate = Decimal(str(item.get("rental_rate") or "2.99"))
        jitter = Decimal("0.95") + Decimal(_digest(row["rental_id"], "pay_jitter") % 11) / Decimal("100")
        amount = rate * jitter
        amount -= discount_by_rental.get(row["rental_id"], Decimal("0"))
        if amount < Decimal("0.50"):
            amount = Decimal("0.50")
        row["amount"] = _money(amount)
        out.append(row)
    return out


def _llm_cache_path(model: str, payload: dict[str, object]) -> Path:
    digest = hashlib.sha256(
        json.dumps({"model": model, **payload}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return _LLM_CACHE_DIR / f"{digest}.json"


def _openai_chat_json(system_prompt: str, user_prompt: str, *, model: str) -> object:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for --enrich-llm")
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"OpenAI request failed: {exc}") from exc
    content = payload["choices"][0]["message"]["content"]
    return json.loads(content)


def _offline_item_description(item_type: str, title: str) -> str:
    clean = str(title).strip().rstrip(".")
    if item_type == "book":
        return f"A readable rental edition of {clean}."
    if item_type == "game":
        return f"A rental copy of {clean} suited for home play sessions."
    return f"A catalog title featuring {clean}."


def _run_llm_enrichment(*, offline: bool = False) -> None:
    model = os.environ.get("RENTAL_SHOP_LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    batch_size = int(os.environ.get("RENTAL_SHOP_LLM_BATCH_SIZE", str(DEFAULT_LLM_BATCH_SIZE)))
    _LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    api_configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    _log_progress(
        f"[llm] starting enrichment model={model} batch_size={batch_size} "
        f"offline={offline} api_key={'set' if api_configured else 'missing'} "
        f"cache={_LLM_CACHE_DIR}"
    )
    system_prompt = (
        "You write neutral retail catalog copy. Return strict JSON only. "
        "One or two sentences per item. No fake awards, review scores, ALL CAPS, "
        "Staff pick language, or Rental Shop boilerplate. Do not invent uncertain plot details."
    )

    items = _read_csv("item")
    books = {int(r["item_id"]): r for r in _read_csv("book")} if (OUT_DIR / "book.csv").is_file() else {}
    games = {int(r["item_id"]): r for r in _read_csv("game")} if (OUT_DIR / "game.csv").is_file() else {}
    films = {int(r["item_id"]): r for r in _read_csv("film")} if (OUT_DIR / "film.csv").is_file() else {}
    enriched: dict[int, str] = {}

    targets = [row for row in items if row.get("item_type") in ("film", "book", "game")]
    total_batches = max(1, (len(targets) + batch_size - 1) // batch_size)
    _log_progress(
        f"[llm] item.description: {len(targets)} rows across {total_batches} batches "
        f"(table=item column=description types=film,book,game)"
    )
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        batch_no = start // batch_size + 1
        id_range = f"{batch[0]['item_id']}-{batch[-1]['item_id']}" if batch else "empty"
        payload_items: list[dict[str, object]] = []
        for row in batch:
            item_id = int(row["item_id"])
            item_type = row["item_type"]
            entry: dict[str, object] = {
                "id": item_id,
                "type": item_type,
                "title": row.get("title") or "",
                "year": row.get("release_year") or "",
            }
            if item_type == "film":
                film = films.get(item_id, {})
                entry["rating"] = film.get("rating") or ""
            elif item_type == "book":
                book = books.get(item_id, {})
                entry["author"] = book.get("author_id") or ""
            else:
                game = games.get(item_id, {})
                entry["platform"] = game.get("platform") or ""
                entry["developer"] = game.get("developer") or ""
            payload_items.append(entry)

        if offline:
            for entry in payload_items:
                enriched[int(entry["id"])] = _offline_item_description(
                    str(entry["type"]), str(entry["title"])
                )
            _log_progress(f"[llm] item.description batch {batch_no}/{total_batches} ids={id_range} offline")
            continue

        cache_key = {"kind": "items", "items": payload_items}
        cache_path = _llm_cache_path(model, cache_key)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _log_progress(
                f"[llm] item.description batch {batch_no}/{total_batches} ids={id_range} cache=hit"
            )
        else:
            _log_progress(
                f"[llm] item.description batch {batch_no}/{total_batches} ids={id_range} cache=miss calling API"
            )
            t0 = time.monotonic()
            user_prompt = json.dumps({"items": payload_items}, ensure_ascii=False)
            cached = _openai_chat_json(system_prompt, user_prompt, model=model)
            cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _log_progress(
                f"[llm] item.description batch {batch_no}/{total_batches} ids={id_range} "
                f"done in {time.monotonic() - t0:.1f}s"
            )
        rows = cached.get("items") if isinstance(cached, dict) else cached
        if not isinstance(rows, list):
            raise SystemExit("LLM item enrichment returned unexpected JSON shape")
        for idx, entry in enumerate(rows):
            item_id = int(entry.get("id") or payload_items[idx]["id"])
            enriched[item_id] = str(entry.get("description") or "").strip()

    for row in items:
        item_id = int(row["item_id"])
        if item_id in enriched and enriched[item_id]:
            row["description"] = enriched[item_id]
    _write_csv(
        "item",
        [
            "item_id",
            "item_type",
            "title",
            "description",
            "release_year",
            "language_id",
            "rental_duration",
            "rental_rate",
            "replacement_cost",
            "last_update",
        ],
        items,
    )
    _log_progress(f"[llm] item.description applied to item.csv ({len(enriched)} descriptions)")
    print(f"LLM enrichment applied ({'offline fallback' if offline else model})")


def _run_post_gen_sanity_checks() -> None:
    customers = _read_csv("customer")
    bad_domains = ("rentalshop.org",)
    bad_emails = [r for r in customers if str(r.get("email", "")).split("@")[-1].lower() in bad_domains]
    if bad_emails:
        raise SystemExit(f"Sanity: {len(bad_emails)} customer emails still use legacy domains")

    film_items = [r for r in _read_csv("item") if r.get("item_type") == "film"]
    years = {r.get("release_year") for r in film_items if r.get("release_year")}
    if len(years) <= 5:
        raise SystemExit(f"Sanity: film release_year has only {len(years)} distinct values")

    rentals = _read_csv("rental")
    items = {int(r["item_id"]): r.get("item_type") for r in _read_csv("item")}
    inv_by_id = {r["inventory_id"]: int(r["item_id"]) for r in _read_csv("inventory")}
    bg_count = sum(
        1
        for r in rentals
        if items.get(inv_by_id.get(r["inventory_id"], 0), "film") in ("book", "game")
    )
    if rentals and bg_count / len(rentals) < 0.15:
        raise SystemExit(
            f"Sanity: book/game rentals {bg_count}/{len(rentals)} below 15% threshold"
        )
    print(
        f"Sanity OK: {len(customers)} customers, {len(years)} film years, "
        f"bg rentals {bg_count}/{len(rentals)}"
    )


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            dest.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed for {url}: {exc}") from exc


def _download_open_library_books() -> None:
    dest = _SEED_LISTS / "books.csv"
    try:
        with urllib.request.urlopen(_OPEN_LIBRARY_SEARCH, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Open Library download failed: {exc}") from exc
    rows: list[dict[str, str]] = []
    for doc in payload.get("docs", []):
        title = str(doc.get("title") or "").strip()
        if not title:
            continue
        authors = doc.get("author_name") or []
        author_name = str(authors[0]) if authors else ""
        parsed = _parse_author_name(author_name)
        if parsed is None:
            continue
        first, last = parsed
        publisher_field = doc.get("publisher")
        if isinstance(publisher_field, list) and publisher_field:
            publisher = str(publisher_field[0])
        else:
            publisher = ""
        rows.append(
            {
                "title": title,
                "author_first": first,
                "author_last": last,
                "publisher": publisher,
            }
        )
        if len(rows) >= 300:
            break
    if len(rows) < 50:
        raise SystemExit("Open Library book seed download returned too few rows")
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "author_first", "author_last", "publisher"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _download_zenodo_games() -> None:
    raw = _DOWNLOADS / "all_games_PC.csv"
    _download_file(_GAMES_CSV_URL, raw)
    rows: list[dict[str, str]] = []
    with raw.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = str(row.get("name") or "").strip()
            if not _game_title_allowed(title):
                continue
            developer = _zenodo_developer_from_row(row)
            entry: dict[str, str] = {"title": title}
            if developer:
                entry["developer"] = developer
            rows.append(entry)
            if len(rows) >= 200:
                break
    if len(rows) < 50:
        raise SystemExit("Zenodo game seed download returned too few rows")
    dest = _SEED_LISTS / "games.csv"
    fieldnames = ["title"]
    if any(row.get("developer") for row in rows):
        fieldnames.append("developer")
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_bootstrap_inputs() -> None:
    _log_progress("[generate] preparing bootstrap inputs (frozen bundle + Open Library + Zenodo seeds)")
    if _DOWNLOADS.is_dir():
        shutil.rmtree(_DOWNLOADS, ignore_errors=True)
    _FROZEN.mkdir(parents=True, exist_ok=True)
    _LEXICONS.mkdir(parents=True, exist_ok=True)
    _SEED_LISTS.mkdir(parents=True, exist_ok=True)
    if not _INPUTS_ZIP.is_file():
        raise SystemExit(f"Missing frozen inputs bundle: {_INPUTS_ZIP}")
    with zipfile.ZipFile(_INPUTS_ZIP) as archive:
        archive.extractall(_FROZEN)
    _materialize_frozen_lexicons()
    _log_progress("[generate] downloading Open Library book seeds")
    _download_open_library_books()
    _log_progress("[generate] downloading Zenodo game seeds")
    _download_zenodo_games()
    _log_progress("[generate] bootstrap inputs ready")


def _cleanup_bootstrap_downloads() -> None:
    if _DOWNLOADS.is_dir():
        shutil.rmtree(_DOWNLOADS, ignore_errors=True)


def _csv_bundle_complete(out_dir: Path) -> bool:
    return all((out_dir / f"{table}.csv").is_file() for table in _TABLE_ORDER)


def _verify_bundle_allowlist(out_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(out_dir.glob("*.csv")):
        stem = path.stem
        if stem.startswith("_"):
            errors.append(f"reject legacy CSV {path.name}")
        elif stem not in _TABLE_ORDER:
            errors.append(f"unexpected CSV {path.name}")
    for table in _TABLE_ORDER:
        if not (out_dir / f"{table}.csv").is_file():
            errors.append(f"missing required CSV {table}.csv")
    return errors


def _optional_bundle_sha256_check(zip_path: Path) -> None:
    expected = os.environ.get("RENTAL_SHOP_BUNDLE_SHA256", "").strip().lower()
    if not expected:
        return
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(
            f"Bundle SHA256 mismatch: expected {expected}, got {digest} ({zip_path})"
        )


def ensure_csv_bundle(out_dir: Path, zip_path: Path) -> str:
    """Populate *out_dir* from existing CSVs, a local zip, or HTTP download."""

    if _csv_bundle_complete(out_dir):
        return "existing"
    out_dir.mkdir(parents=True, exist_ok=True)
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(out_dir)
        return "local_zip"
    _download_file(DEFAULT_BUNDLE_URL, zip_path)
    _optional_bundle_sha256_check(zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    return "download"


def pack_csv_bundle(out_dir: Path, zip_path: Path) -> None:
    """Zip exactly the 34 canonical rental_shop tables (no ``_*.csv`` sidecars)."""

    for path in sorted(out_dir.glob("_*.csv")):
        path.unlink()
    missing = [table for table in _TABLE_ORDER if not (out_dir / f"{table}.csv").is_file()]
    if missing:
        raise SystemExit(f"Cannot pack: missing tables in {out_dir}: {', '.join(missing)}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for table in _TABLE_ORDER:
            path = out_dir / f"{table}.csv"
            archive.write(path, f"{table}.csv")
    print(f"Packed {len(_TABLE_ORDER)} tables into {zip_path}")


def _load_publisher_names_from_inputs() -> list[str]:
    with zipfile.ZipFile(_INPUTS_ZIP) as archive:
        raw = archive.read("publishers_expanded.csv").decode("utf-8")
    names = [
        row["publisher_name"]
        for row in csv.DictReader(raw.splitlines())
        if row.get("publisher_name")
    ]
    if len(names) < 40:
        raise SystemExit(f"publisher list needs at least 40 names, got {len(names)}")
    return names


def _distinct_values(rows: list[dict[str, str]], column: str) -> int:
    values = {str(row.get(column) or "").strip() for row in rows}
    values.discard("")
    return len(values)


def _resolve_book_seeds_path(explicit: Path | None) -> Path:
    if explicit is not None and explicit.is_file():
        return explicit
    for candidate in (
        _SEED_LISTS / "books.csv",
        _DATA / "_downloads" / "_seed_lists" / "books.csv",
    ):
        if candidate.is_file():
            return candidate
    _prepare_bootstrap_inputs()
    if (_SEED_LISTS / "books.csv").is_file():
        return _SEED_LISTS / "books.csv"
    raise SystemExit("book seed list not found; run bootstrap or pass --book-seeds")


def repair_csv_bundle(csv_dir: Path, *, book_seeds_path: Path | None = None) -> None:
    global OUT_DIR
    prior_out = OUT_DIR
    OUT_DIR = csv_dir
    try:
        address_rows = _read_csv("address")
        if address_rows and "address2" in address_rows[0]:
            for row in address_rows:
                row.pop("address2", None)
            _write_csv(
                "address",
                [
                    "address_id",
                    "address",
                    "district",
                    "city_id",
                    "postal_code",
                    "phone",
                    "last_update",
                ],
                address_rows,
            )

        staff_rows = _read_csv("staff")
        staff_fields = [
            "staff_id",
            "first_name",
            "last_name",
            "address_id",
            "email",
            "store_id",
            "active",
            "username",
            "password",
            "ssn",
            "last_update",
        ]
        for row in staff_rows:
            staff_id = int(row["staff_id"])
            row.pop("picture", None)
            username = str(row.get("username") or f"staff{staff_id:02d}")
            row["username"] = username
            row["active"] = "1" if staff_id % 5 else "0"
            row["password"] = _synth_staff_password_hash(staff_id, username)
        _write_csv("staff", staff_fields, staff_rows)

        customer_rows = _read_csv("customer")
        for row in customer_rows:
            cid = int(row["customer_id"])
            row["activebool"] = "true" if cid % 5 else "false"
        _write_csv(
            "customer",
            [
                "customer_id",
                "store_id",
                "first_name",
                "last_name",
                "email",
                "address_id",
                "activebool",
                "create_date",
                "last_update",
            ],
            customer_rows,
        )

        supplier_rows = _read_csv("supplier")
        for row in supplier_rows:
            sid = int(row["supplier_id"])
            row["is_active"] = "t" if sid % 5 else "f"
        _write_csv(
            "supplier",
            ["supplier_id", "supplier_name", "country_id", "is_active", "last_update"],
            supplier_rows,
        )

        items = {row["item_id"]: row for row in _read_csv("item")}
        game_rows = _read_csv("game")
        for row in game_rows:
            item_id = row["item_id"]
            item = items.get(item_id, {})
            seed = {"title": item.get("title", ""), "developer": row.get("developer", "")}
            row["platform"] = _game_platform_for_item(item_id, seed)
            row["developer"] = _game_developer_for_item(item_id, seed)
        _write_csv(
            "game",
            ["item_id", "platform", "developer", "esrb_rating", "last_update"],
            game_rows,
        )

        seeds_path = _resolve_book_seeds_path(book_seeds_path)
        with seeds_path.open(newline="", encoding="utf-8") as handle:
            book_seeds = list(csv.DictReader(handle))
        publisher_names = _load_publisher_names_from_inputs()
        seed_authors = _collect_seed_authors(book_seeds)
        author_rows: list[dict[str, object]] = []
        for author_id, (first, last) in enumerate(seed_authors, start=1):
            author_rows.append(
                {
                    "author_id": author_id,
                    "first_name": first,
                    "last_name": last,
                    "last_update": synth.activity_timestamp("author", author_id),
                }
            )
        _write_csv("author", ["author_id", "first_name", "last_name", "last_update"], author_rows)

        publisher_rows = _read_csv("publisher")
        for idx, row in enumerate(publisher_rows):
            row["publisher_name"] = publisher_names[idx % len(publisher_names)]
        _write_csv(
            "publisher",
            ["publisher_id", "publisher_name", "country_id", "last_update"],
            publisher_rows,
        )

        title_to_seed = {seed["title"]: seed for seed in book_seeds}
        book_rows = _read_csv("book")
        for row in book_rows:
            item = items.get(row["item_id"], {})
            seed = title_to_seed.get(item.get("title", ""))
            if seed is None:
                continue
            row["author_id"] = str(_author_id_for_seed(seed, author_rows))
            row["publisher_id"] = str(
                _publisher_id_for_seed(seed, publisher_rows, publisher_names)
            )
        _write_csv(
            "book",
            ["item_id", "author_id", "publisher_id", "isbn", "page_count", "last_update"],
            book_rows,
        )

        print(
            "distinct counts:",
            f"staff.active={_distinct_values(staff_rows, 'active')}",
            f"staff.password={_distinct_values(staff_rows, 'password')}",
            f"customer.activebool={_distinct_values(customer_rows, 'activebool')}",
            f"supplier.is_active={_distinct_values(supplier_rows, 'is_active')}",
            f"game.platform={_distinct_values(game_rows, 'platform')}",
            f"game.developer={_distinct_values(game_rows, 'developer')}",
        )
        errors = verify_csv_integrity()
        if errors:
            _report_fk_errors(errors, "FK verify")
    finally:
        OUT_DIR = prior_out


def _report_fk_errors(errors: list[str], label: str) -> None:
    for err in errors[:20]:
        print(f"{label}: {err}", file=sys.stderr)
    if len(errors) > 20:
        print(f"{label}: ... and {len(errors) - 20} more", file=sys.stderr)
    raise SystemExit(f"CSV FK integrity failed ({len(errors)} issues)")


def _run_download() -> None:
    source = ensure_csv_bundle(OUT_DIR, ZIP_PATH)
    print(f"CSV bundle source: {source}")
    allowlist_errors = _verify_bundle_allowlist(OUT_DIR)
    if allowlist_errors:
        for err in allowlist_errors[:20]:
            print(f"allowlist: {err}", file=sys.stderr)
        raise SystemExit(f"CSV bundle allowlist failed ({len(allowlist_errors)} issues)")
    if ZIP_PATH.is_file():
        _optional_bundle_sha256_check(ZIP_PATH)
    errors = verify_csv_integrity()
    if errors:
        _report_fk_errors(errors, "FK verify")
    print(f"rental_shop CSVs ready in {OUT_DIR} (FK integrity OK)")


def _purge_legacy_sidecar_csvs() -> None:
    for path in sorted(OUT_DIR.glob("_*.csv")):
        path.unlink()


def _run_generate(*, enrich_llm: bool = False) -> None:
    _prepare_bootstrap_inputs()
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _log_progress("[generate] bootstrapping spine CSVs")
        _bootstrap_spine_csvs()
        book_seeds = _load_seed_list("books")
        game_seeds = _load_seed_list("games")
        if len(book_seeds) < 50 or len(game_seeds) < 50:
            raise SystemExit("Seed lists must contain at least 50 rows each")

        _log_progress("[generate] transform_inventory + transform_film_actor")
        transform_inventory()
        transform_film_actor()
        _log_progress("[generate] bootstrap_synthetic_spine (geo/store/staff)")
        countries, _, addresses = bootstrap_synthetic_spine()
        categories = _merge_lexicon_categories()
        _write_csv("category", ["category_id", "name", "last_update"], categories)
        languages = [
            {k: str(v) for k, v in row.items()}
            for row in synth.stagger_last_updates(_read_csv("language"), "language_id", "language")
        ]
        _write_csv("language", ["language_id", "name", "last_update"], languages)

        _log_progress("[generate] transform_catalog + item features/categories")
        source_films, item_rows = transform_catalog()
        generate_item_feature(source_films)
        transform_item_category()
        customers = transform_customer()
        payments_raw = _read_csv("payment")
        rentals_raw = _read_csv("rental")
        rentals = synth.rebalance_rentals(rentals_raw)
        payments = synth.rebalance_payments(payments_raw, rentals)
        _write_csv(
            "rental",
            [
                "rental_id",
                "rental_date",
                "inventory_id",
                "customer_id",
                "return_date",
                "staff_id",
                "last_update",
            ],
            rentals,
        )
        _write_csv("payment", ["payment_id", "rental_id", "amount", "payment_date"], payments)

        _log_progress("[generate] books/games + inventory/rental extensions")
        author_rows, publisher_rows = generate_authors_publishers(countries, book_seeds)
        book_rows, game_rows, _, _ = generate_books_games(
            categories,
            languages,
            item_rows,
            author_rows,
            publisher_rows,
            book_seeds,
            game_seeds,
        )
        item_rows = _read_csv("item")
        inventory, item_to_inventory = append_book_game_inventory(book_rows, game_rows)
        no_rental_ids = _no_rental_inventory_ids(inventory)
        rentals, payments = append_book_game_rentals_payments(
            item_to_inventory, customers, rentals, payments, no_rental_ids
        )
        rentals, inventory, no_rental_ids = apply_benchmark_patterns(inventory, rentals)
        payments = _sync_payments_to_rentals(rentals, payments)

        _log_progress("[generate] logistics + promotions + operational tables")
        couriers = generate_couriers(countries)
        suppliers = generate_suppliers(countries)
        warehouses = generate_warehouses(addresses)
        promotions = generate_promotions()
        deliveries = generate_deliveries(rentals, couriers, customers)
        apply_benchmark_patterns(inventory, rentals, deliveries)
        generate_purchase_orders(suppliers, item_rows)
        generate_stock_transfers(warehouses, item_rows)
        generate_promotion_redemptions(promotions, rentals)
        generate_inventory_status_history(inventory, rentals)
        generate_damage_reports(rentals)
        generate_reservations(customers, item_rows, rentals)
        remove_obsolete_csvs()
        _purge_legacy_sidecar_csvs()

        _log_progress("[generate] finalizing customer create_date + payment amounts")
        customers = _finalize_customer_create_dates(_read_csv("customer"), _read_csv("rental"))
        _write_csv(
            "customer",
            [
                "customer_id",
                "store_id",
                "first_name",
                "last_name",
                "email",
                "address_id",
                "activebool",
                "create_date",
                "last_update",
            ],
            customers,
        )
        payments = _align_payment_amounts(_read_csv("rental"), _read_csv("payment"))
        _write_csv("payment", ["payment_id", "rental_id", "amount", "payment_date"], payments)

        rentals = enforce_return_dates_at_anchor(_read_csv("rental"))
        _write_csv(
            "rental",
            [
                "rental_id",
                "rental_date",
                "inventory_id",
                "customer_id",
                "return_date",
                "staff_id",
                "last_update",
            ],
            rentals,
        )

        if enrich_llm:
            offline = not os.environ.get("OPENAI_API_KEY", "").strip()
            _run_llm_enrichment(offline=offline)

        _run_post_gen_sanity_checks()

        sem_errors = verify_csv_semantics()
        if sem_errors:
            for err in sem_errors[:20]:
                print(f"semantics: {err}", file=sys.stderr)
            raise SystemExit(f"CSV semantic verify failed ({len(sem_errors)} issues)")

        errors = verify_csv_integrity()
        if errors:
            _report_fk_errors(errors, "FK verify")

        print("Row counts:")
        _print_row_counts()
        print(f"Generated rental_shop CSVs in {OUT_DIR} (FK integrity OK)")
    finally:
        _cleanup_bootstrap_downloads()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download",
        action="store_true",
        help="Ensure CSV bundle exists (existing dir, local zip, or HTTP download)",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Generate rental_shop CSVs from frozen inputs",
    )
    parser.add_argument(
        "--enrich-llm",
        action="store_true",
        help="Run LLM enrichment during --generate (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Zip rental_shop_csvs/ into scripts/data/rental_shop.zip",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Load KEY=VALUE pairs into the process environment (default: repo env.env)",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.is_file():
        load_env_file(env_path, override=bool(args.enrich_llm))
    elif args.enrich_llm:
        raise SystemExit(f"env file not found for --enrich-llm: {env_path}")

    if args.enrich_llm and not args.generate:
        parser.error("--enrich-llm requires --generate")
    if args.download and args.enrich_llm:
        parser.error("--enrich-llm cannot be used with --download")

    if args.pack and not args.generate and not args.download:
        pack_csv_bundle(OUT_DIR, ZIP_PATH)
        return

    if args.generate:
        _run_generate(enrich_llm=args.enrich_llm)
        if args.pack:
            pack_csv_bundle(OUT_DIR, ZIP_PATH)
        return

    if args.download:
        _run_download()
        return

    _run_download()


if __name__ == "__main__":
    main()
