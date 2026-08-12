"""Source-file markers for federation corpus realism; no database required."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_DATA = _SCRIPTS / "data"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _source_rental_shop():
    return importlib.import_module("source_rental_shop")


def _read_schema(name: str) -> str:
    return (_DATA / name).read_text(encoding="utf-8")


def _member_csv_text(member: str, table: str) -> str | None:
    path = _DATA / f"federation_{member}_data" / f"{table}.csv"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _require_realism_csv(member: str, table: str, *, marker: str) -> str:
    text = _require_member_csv(member, table)
    if marker not in text:
        pytest.skip(
            f"federation_{member}_data/{table}.csv lacks sandbox realism marker {marker!r}; "
            "run sandbox downsample export",
        )
    return text


def _require_member_csv(member: str, table: str) -> str:
    text = _member_csv_text(member, table)
    if text is None:
        pytest.skip(
            f"missing federation_{member}_data/{table}.csv (run export_federation_member_data_dirs_from_existing_csvs)"
        )
    return text


@pytest.mark.fast
def test_logistics_schema_reflects_full_partition() -> None:
    text = _read_schema("federation_logistics_schema.sql")
    assert "payment-only" not in text.lower()
    for table in (
        "warehouse",
        "supplier",
        "courier",
        "delivery",
        "damage_report",
        "inventory_status_history",
        "stock_transfer",
    ):
        assert table in text
    assert text.count("CREATE TABLE") >= 8


@pytest.mark.fast
def test_crm_schema_reflects_full_partition() -> None:
    text = _read_schema("federation_crm_schema.sql")
    assert "payment-only" not in text.lower()
    for table in ("promotion", "promotion_redemption", "customer", "staff"):
        assert table in text
    assert "CREATE TABLE payment" not in text


@pytest.mark.fast
def test_partition_json_lists_logistics_and_crm_tables() -> None:
    import sandbox_corpus as sc

    partition = json.loads((_DATA / "federation_partition.json").read_text(encoding="utf-8"))
    assert "receipts" in partition["logistics"]
    assert "country" in partition["catalog"]
    assert "city" in partition["catalog"]
    assert sc.federation_partition_tables("logistics") == sc.FEDERATION_PARTITION_TABLES["logistics"]
    assert sc.federation_partition_tables("crm") == sc.FEDERATION_PARTITION_TABLES["crm"]


@pytest.mark.fast
def test_catalog_country_drift_vs_storefront() -> None:
    catalog = _require_realism_csv("catalog", "country", marker="Great Britain")
    storefront = _require_realism_csv("storefront", "country", marker="Brazil")
    sc = _source_rental_shop()
    assert "Great Britain" in catalog
    assert "United Kingdom" in storefront
    assert "Catalog-only Island Republic" in catalog
    assert "Brazil" in storefront
    assert sc.CORPUS_REALISM_COUNTRY_CATALOG_DRIFT[44] in catalog
    assert sc.CORPUS_REALISM_COUNTRY_CATALOG_ONLY[211] in catalog


@pytest.mark.fast
def test_crm_customer_replica_column_drift() -> None:
    crm = _require_realism_csv("crm", "customer", marker="email_addr")
    sc = _source_rental_shop()
    assert "email_addr" in crm.splitlines()[0]
    assert "loyalty_tier" in crm.splitlines()[0]
    assert "activebool" not in crm.splitlines()[0]
    assert "(crm)" in crm
    assert sc.CRM_CUSTOMER_DESYNC_IDS
    assert ",1001," in crm.replace(" ", "")
    assert ",1005," in crm.replace(" ", "")
    assert sc.crm_customer_desync_address_id(1, 1) == 1 + sc.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
    assert sc.crm_customer_desync_address_id(2, 2) == 2


@pytest.mark.fast
def test_declaration_maps_crm_customer_columns() -> None:
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    customer = next(entry for entry in declaration["logical_tables"] if entry["logical"] == "customer")
    crm_member = next(member for member in customer["members"] if member["source"] == "crm")
    assert crm_member["columns"]["email"] == "email_addr"
    assert "loyalty_tier" not in crm_member["columns"]
    crm_csv = _require_realism_csv("crm", "customer", marker="loyalty_tier")
    assert "loyalty_tier" in crm_csv.splitlines()[0]


@pytest.mark.fast
def test_logistics_receipts_and_abbreviated_purchase_order() -> None:
    logistics = _read_schema("federation_logistics_schema.sql")
    assert "CREATE TABLE receipts" in logistics or 'CREATE TABLE "receipts"' in logistics
    assert "rcpt_id" in logistics
    assert "rent_id" in logistics
    assert "ord_id" in logistics
    assert "sup_id" in logistics
    assert "ord_dt TEXT" in logistics


@pytest.mark.fast
def test_declaration_maps_receipts_into_payment_union() -> None:
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    payment = next(entry for entry in declaration["logical_tables"] if entry["logical"] == "payment")
    logistics_member = next(member for member in payment["members"] if member["source"] == "logistics")
    assert logistics_member["table"] == "receipts"
    assert logistics_member["columns"]["payment_id"] == "rcpt_id"


@pytest.mark.fast
def test_storefront_rental_timestamps_second_precision_no_tz() -> None:
    storefront_schema = _read_schema("federation_storefront_schema.sql")
    assert "rental_date TIMESTAMP" in storefront_schema
    rental_section = storefront_schema.split("CREATE TABLE rental", 1)[1]
    assert "TIME ZONE" not in rental_section.split("CREATE TABLE", 1)[0]
    rental_csv = _require_realism_csv("storefront", "rental", marker="2024-03-22 08:04:33")
    assert "2024-03-22 08:04:33" in rental_csv
    assert ".000" not in rental_csv.splitlines()[1]


@pytest.mark.fast
def test_generator_storefront_rental_create_sql() -> None:
    sc = _source_rental_shop()
    ddl = sc.storefront_rental_create_sql()
    assert "rental_date TIMESTAMP" in ddl
    assert "TIME ZONE" not in ddl


@pytest.mark.fast
def test_orphan_delivery_rental_ids_in_logistics_data() -> None:
    logistics = _require_realism_csv("logistics", "delivery", marker="ORPHAN-9999001")
    sc = _source_rental_shop()
    for rental_id in sc.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS:
        assert str(rental_id) in logistics
    assert "ORPHAN-9999001" in logistics


@pytest.mark.fast
def test_subscription_retail_reskin_in_data_and_generator() -> None:
    catalog = _require_realism_csv("catalog", "item", marker="Premium Monthly Subscription")
    sc = _source_rental_shop()
    assert "Premium Monthly Subscription" in catalog
    assert "subscription bundle" in catalog
    assert "Family Subscription" in catalog
    assert sc.SUBSCRIPTION_RETAIL_RESKIN_REPLACEMENTS
    assert "subscription bundle" in sc.reskin_subscription_retail_text("DVD rental title")


@pytest.mark.fast
def test_reskin_replaces_category_vocabulary() -> None:
    sc = _source_rental_shop()
    out = sc.reskin_subscription_retail_text("Action Comedy DVD")
    assert "Activewear & Gear" in out
    assert "subscription bundle" in out
