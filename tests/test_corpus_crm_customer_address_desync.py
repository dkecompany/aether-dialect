"""CRM customer replica address desync from committed federation seeds (source-only)."""

from __future__ import annotations

import importlib
import json
import re
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


def _crm_customer_rows(crm_seed: str) -> dict[int, int]:
    sc = _sandbox_corpus()
    rows: dict[int, int] = {}
    for customer_id, address_id in zip(
        sc.parse_seed_insert_column_values(crm_seed, "customer", "customer_id"),
        sc.parse_seed_insert_column_values(crm_seed, "customer", "address_id"),
        strict=True,
    ):
        rows[int(customer_id)] = int(address_id)
    return rows


@pytest.mark.fast
def test_crm_seed_desyncs_address_ids_for_selected_customers() -> None:
    sc = _sandbox_corpus()
    crm = _read_seed("federation_crm_seed.sql")
    rows = _crm_customer_rows(crm)
    assert rows[1] == 1 + sc.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
    assert rows[5] == 5 + sc.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
    assert rows[2] == 2
    for customer_id in sc.CRM_CUSTOMER_DESYNC_IDS:
        if customer_id in rows:
            assert rows[customer_id] != customer_id


@pytest.mark.fast
def test_generator_address_desync_matches_seed_policy() -> None:
    sc = _sandbox_corpus()
    for customer_id in (1, 5, 12, 23, 37):
        assert sc._crm_customer_desync_address_id(customer_id, customer_id) == (
            customer_id + sc.CRM_CUSTOMER_ADDRESS_DESYNC_OFFSET
        )
    assert sc._crm_customer_desync_address_id(2, 2) == 2


@pytest.mark.fast
def test_declaration_maps_address_id_for_crm_customer() -> None:
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    customer = next(entry for entry in declaration["logical_tables"] if entry["logical"] == "customer")
    crm_member = next(member for member in customer["members"] if member["source"] == "crm")
    assert crm_member["columns"]["address_id"] == "address_id"
    crm = _read_seed("federation_crm_seed.sql")
    assert re.search(r",\s*1001,", crm)
    assert re.search(r",\s*1005,", crm)
