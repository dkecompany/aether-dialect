"""Fast tests for live federation topology wiring (collected without running live databases)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.fast
def test_live_schema_constants_name_four_member_targets() -> None:
    sc = importlib.import_module("sandbox_corpus")
    assert sc.FEDERATION_STOREFRONT_PG_SCHEMA == "rental_shop_fed_storefront"
    assert sc.FEDERATION_LOGISTICS_PG_SCHEMA == "rental_shop_fed_logistics"
    assert sc.FEDERATION_CATALOG_MYSQL_DATABASE == "rental_shop_fed_catalog"
    assert sc.FEDERATION_CRM_MARIADB_DATABASE == "rental_shop_fed_crm"


@pytest.mark.fast
def test_live_declaration_carries_four_cross_source_joins() -> None:
    path = _REPO / "live_tests" / "fixtures" / "federation_live_declaration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["federation_id"] == "live_rental_shop"
    assert len(payload["cross_source_joins"]) == 4
    assert any(row["logical"] == "customer" for row in payload["logical_tables"])


@pytest.mark.fast
def test_federation_load_cli_lists_logistics_and_crm_targets() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "load_rental_shop_engines.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    help_text = proc.stdout
    assert "logistics" in help_text
    assert "crm" in help_text


@pytest.mark.fast
def test_oracle_module_names_both_live_and_sandbox_corpora() -> None:
    oracles = importlib.import_module("live_tests.mydb_profile")
    assert oracles.LIVE_FULL_PAYMENT_TOTAL != oracles.SANDBOX_SUBSET_PAYMENT_TOTAL
    assert oracles.LIVE_FULL_JOIN_RENTAL_LINKED_TITLES != oracles.SANDBOX_SUBSET_JOIN_RENTAL_LINKED_TITLES


@pytest.mark.fast
def test_live_and_live_no_llm_markers_are_registered() -> None:
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"live:' in text
    assert '"live_no_llm:' in text


@pytest.mark.fast
def test_missing_partition_engine_probe_lists_three_families(monkeypatch: pytest.MonkeyPatch) -> None:
    live = importlib.import_module("live_tests.live_support")
    monkeypatch.setattr(live, "_ENV_FILE", Path("/nonexistent/env.env"))
    assert live.missing_federation_partition_engines() == ["postgresql", "mysql", "mariadb"]
