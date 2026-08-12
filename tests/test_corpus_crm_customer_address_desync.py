"""CRM customer replica address desync from federation CSV fixtures (source-only)."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from io import StringIO
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_DATA = _SCRIPTS / "data"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _source_rental_shop():
    return importlib.import_module("source_rental_shop")


def _crm_customer_rows() -> dict[int, int]:
    path = _DATA / "federation_crm_data" / "customer.csv"
    if not path.is_file():
        pytest.skip("missing federation_crm_data/customer.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    if "email_addr" not in header:
        pytest.skip("full federation CSV dirs lack CRM replica column drift; run sandbox downsample export")
    rows: dict[int, int] = {}
    reader = csv.DictReader(StringIO(path.read_text(encoding="utf-8")))
    for row in reader:
        rows[int(row["customer_id"])] = int(row["address_id"])
    return rows


@pytest.mark.fast
def test_crm_csv_desyncs_address_ids_for_selected_customers() -> None:
    src = _source_rental_shop()
    rows = _crm_customer_rows()
    assert rows[1] == 1 + src.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
    assert rows[5] == 5 + src.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
    assert rows[2] == 2
    for customer_id in src.CRM_CUSTOMER_DESYNC_IDS:
        if customer_id in rows:
            assert rows[customer_id] != customer_id


@pytest.mark.fast
def test_generator_address_desync_matches_csv_policy() -> None:
    src = _source_rental_shop()
    for customer_id in (1, 5, 12, 23, 37):
        assert src.crm_customer_desync_address_id(customer_id, customer_id) == (
            customer_id + src.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
        )
    assert src.crm_customer_desync_address_id(2, 2) == 2


@pytest.mark.fast
def test_declaration_maps_address_id_for_crm_customer() -> None:
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    customer = next(entry for entry in declaration["logical_tables"] if entry["logical"] == "customer")
    crm_member = next(member for member in customer["members"] if member["source"] == "crm")
    assert crm_member["columns"]["address_id"] == "address_id"
    rows = _crm_customer_rows()
    assert rows[1] == 1001
    assert rows[5] == 1005
