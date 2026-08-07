"""Tests for federation member partition ownership maps."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _sandbox_corpus():
    return importlib.import_module("sandbox_corpus")


@pytest.mark.fast
def test_federation_partition_map_includes_operational_tables() -> None:
    sc = _sandbox_corpus()
    storefront = sc.federation_partition_tables("storefront")
    catalog = sc.federation_partition_tables("catalog")
    assert "rental" in storefront
    assert "inventory" in catalog
    assert "film" in catalog
    assert storefront != frozenset({"payment"})
    assert catalog != frozenset({"payment"})


@pytest.mark.fast
def test_partition_map_splits_members_by_operational_ownership() -> None:
    sc = _sandbox_corpus()
    assert "rental" in sc.federation_partition_tables("storefront")
    assert "warehouse" in sc.federation_partition_tables("logistics")
    assert "promotion" in sc.federation_partition_tables("crm")
    assert "inventory" in sc.federation_partition_tables("catalog")


@pytest.mark.fast
def test_federation_partition_json_matches_constant() -> None:
    sc = _sandbox_corpus()
    loaded = sc.load_federation_partition_map()
    assert loaded == sc.FEDERATION_PARTITION_TABLES
