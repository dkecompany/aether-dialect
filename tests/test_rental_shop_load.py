"""Regression tests for rental_shop loader helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_RENTAL_SHOP_SCRIPT = _REPO / "scripts" / "load_rental_shop_engines.py"


def _load_rental_shop_module():
    spec = importlib.util.spec_from_file_location("rental_shop_dev", _RENTAL_SHOP_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rental_shop_dev"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rental_shop():
    return _load_rental_shop_module()


def test_trailers_sql_uses_item_feature(rental_shop) -> None:
    sql = rental_shop._trailers_sql("postgresql", "public")
    assert "item_feature" in sql
    assert "trailers" in sql.lower()
    assert "special_features" not in sql


def test_prepare_dataframe_nullable_int_empty_string(rental_shop) -> None:
    frame = pd.DataFrame(
        {
            "reservation_id": [1],
            "fulfilled_rental_id": [""],
        }
    )
    out = rental_shop._prepare_dataframe("mysql", "reservation", frame)
    assert pd.isna(out.loc[0, "fulfilled_rental_id"]) or out.loc[0, "fulfilled_rental_id"] is None


def test_prepare_dataframe_nullable_int_not_float_string(rental_shop) -> None:
    frame = pd.DataFrame(
        {
            "reservation_id": [1],
            "fulfilled_rental_id": [17710.0],
        }
    )
    out = rental_shop._prepare_dataframe("postgresql", "reservation", frame)
    assert out.loc[0, "fulfilled_rental_id"] == 17710
    assert isinstance(out.loc[0, "fulfilled_rental_id"], int)


def test_databricks_sql_literal_nullable_int(rental_shop) -> None:
    assert rental_shop._databricks_sql_literal(None) == "NULL"
    assert rental_shop._databricks_sql_literal(17710) == "17710"


def test_prepare_dataframe_address_defaults(rental_shop) -> None:
    frame = pd.DataFrame(
        {
            "address_id": [1],
            "district": [""],
            "phone": [""],
        }
    )
    out = rental_shop._prepare_dataframe("postgresql", "address", frame)
    assert out.loc[0, "district"] == "unknown"
    assert out.loc[0, "phone"] == "0000000000"
