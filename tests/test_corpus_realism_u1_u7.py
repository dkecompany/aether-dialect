"""Source-file markers for federation corpus realism (U1–U7); no database required."""

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


def _sandbox_corpus():
    return importlib.import_module("sandbox_corpus")


def _read_seed(name: str) -> str:
    return (_DATA / name).read_text(encoding="utf-8")


@pytest.mark.fast
def test_u1_logistics_seed_reflects_full_partition() -> None:
    text = _read_seed("federation_logistics_seed.sql")
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
def test_u1_crm_seed_reflects_full_partition() -> None:
    text = _read_seed("federation_crm_seed.sql")
    assert "payment-only" not in text.lower()
    for table in ("promotion", "promotion_redemption", "customer", "staff"):
        assert table in text
    assert "CREATE TABLE payment" not in text


@pytest.mark.fast
def test_u1_partition_json_lists_logistics_and_crm_tables() -> None:
    sc = _sandbox_corpus()
    partition = json.loads((_DATA / "federation_partition.json").read_text(encoding="utf-8"))
    assert "receipts" in partition["logistics"]
    assert "country" in partition["catalog"]
    assert "city" in partition["catalog"]
    assert sc.federation_partition_tables("logistics") == sc.FEDERATION_PARTITION_TABLES["logistics"]
    assert sc.federation_partition_tables("crm") == sc.FEDERATION_PARTITION_TABLES["crm"]


@pytest.mark.fast
def test_u2_catalog_country_drift_vs_storefront() -> None:
    catalog = _read_seed("federation_catalog_seed.sql")
    storefront = _read_seed("federation_storefront_seed.sql")
    sc = _sandbox_corpus()
    assert "Great Britain" in catalog
    assert "United Kingdom" in storefront
    assert "Catalog-only Island Republic" in catalog
    assert "Brazil" in storefront
    assert sc.CORPUS_REALISM_COUNTRY_CATALOG_DRIFT[44] in catalog
    assert sc.CORPUS_REALISM_COUNTRY_CATALOG_ONLY[211] in catalog


@pytest.mark.fast
def test_u3_crm_customer_replica_column_drift() -> None:
    crm = _read_seed("federation_crm_seed.sql")
    sc = _sandbox_corpus()
    assert "email_addr" in crm
    assert "loyalty_tier" in crm
    assert "activebool" not in crm
    assert "(crm)" in crm
    assert sc.CRM_CUSTOMER_DESYNC_IDS


@pytest.mark.fast
def test_u3_declaration_maps_crm_customer_columns() -> None:
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    customer = next(entry for entry in declaration["logical_tables"] if entry["logical"] == "customer")
    crm_member = next(member for member in customer["members"] if member["source"] == "crm")
    assert crm_member["columns"]["email"] == "email_addr"
    assert "loyalty_tier" in crm_member["columns"]


@pytest.mark.fast
def test_u4_logistics_legacy_receipts_and_abbreviated_purchase_order() -> None:
    logistics = _read_seed("federation_logistics_seed.sql")
    assert "CREATE TABLE receipts" in logistics
    assert "rcpt_id" in logistics
    assert "rent_id" in logistics
    assert "ord_id" in logistics
    assert "sup_id" in logistics
    assert "ord_dt TEXT" in logistics


@pytest.mark.fast
def test_u4_declaration_maps_receipts_into_payment_union() -> None:
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    payment = next(entry for entry in declaration["logical_tables"] if entry["logical"] == "payment")
    logistics_member = next(member for member in payment["members"] if member["source"] == "logistics")
    assert logistics_member["table"] == "receipts"
    assert logistics_member["columns"]["payment_id"] == "rcpt_id"


@pytest.mark.fast
def test_u5_storefront_rental_timestamps_second_precision_no_tz() -> None:
    storefront = _read_seed("federation_storefront_seed.sql")
    assert "rental_date TIMESTAMP" in storefront
    rental_section = storefront.split("CREATE TABLE rental", 1)[1]
    assert "TIME ZONE" not in rental_section.split("CREATE TABLE", 1)[0]
    assert "'2024-03-22 08:04:33'" in storefront
    assert ".000" not in storefront.split("INSERT INTO rental", 1)[1].split(";", 1)[0]


@pytest.mark.fast
def test_u5_generator_storefront_rental_create_sql() -> None:
    sc = _sandbox_corpus()
    ddl = sc._storefront_rental_create_sql()
    assert "rental_date TIMESTAMP" in ddl
    assert "TIME ZONE" not in ddl


@pytest.mark.fast
def test_u6_orphan_delivery_rental_ids_in_logistics_seed() -> None:
    logistics = _read_seed("federation_logistics_seed.sql")
    sc = _sandbox_corpus()
    for rental_id in sc.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS:
        assert str(rental_id) in logistics
    assert "ORPHAN-9999001" in logistics


@pytest.mark.fast
def test_u7_subscription_retail_reskin_in_seeds_and_generator() -> None:
    catalog = _read_seed("federation_catalog_seed.sql")
    sc = _sandbox_corpus()
    assert "Premium Monthly Subscription" in catalog
    assert "subscription bundle" in catalog
    assert "Family Subscription" in catalog
    assert sc.SUBSCRIPTION_RETAIL_RESKIN_REPLACEMENTS
    assert "subscription bundle" in sc._reskin_subscription_retail_text("DVD rental title")


@pytest.mark.fast
def test_u7_reskin_replaces_category_vocabulary() -> None:
    sc = _sandbox_corpus()
    out = sc._reskin_subscription_retail_text("Action Comedy DVD")
    assert "Activewear & Gear" in out
    assert "subscription bundle" in out
