"""Strict assertions for federation member live coverage tests."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP

from .federation_oracles import DELIVERY_RENTAL_LINKED_COUNT


def _scalar_value(step: Any) -> float | int | Decimal | None:
    data = getattr(step, "data", None)
    if data is not None and len(data) == 1 and len(data.columns) == 1:
        return data.iloc[0, 0]
    message = str(getattr(step, "message", "") or "").strip()
    if message:
        try:
            return Decimal(message)
        except Exception:
            return None
    return None


def assert_staff_email_from_storefront_authoritative(step: Any) -> None:
    assert step.done, getattr(step, "error", step)
    assert not step.error, getattr(step, "message", "")
    assert step.sql
    sql_lower = str(step.sql).lower()
    assert "staff" in sql_lower
    assert "email" in sql_lower
    assert "crm" not in sql_lower
    data = getattr(step, "data", None)
    assert data is not None and len(data) > 0, "expected staff email rows"
    email_col = next((col for col in data.columns if "email" in str(col).lower()), None)
    assert email_col is not None
    values = [str(value) for value in data[email_col].tolist() if value is not None and str(value).strip()]
    assert values, "expected non-empty email values"
    assert any("@" in value for value in values)


def assert_median_payment_refused_for_mysql_family(step: Any) -> None:
    assert step.done, "expected terminal step"
    assert step.error, "median must refuse on mysql-family federation members"
    code = getattr(step, "refusal_diagnostic_code", None) or getattr(step, "refusal_code", None)
    assert code == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP, code
    combined = f"{step.error} {getattr(step, 'message', '')}".lower()
    assert "median" in combined


def assert_delivery_rental_cross_source_linked_count(step: Any) -> None:
    assert step.done, getattr(step, "error", step)
    assert not step.error, getattr(step, "message", "")
    assert step.sql
    sql_lower = str(step.sql).lower()
    assert "delivery" in sql_lower
    assert "rental" in sql_lower
    succeeded = getattr(step, "federation_succeeded", ()) or ()
    if succeeded:
        sources = {str(entry[0]).lower() for entry in succeeded}
        assert {"logistics", "storefront"}.issubset(sources), sources
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == DELIVERY_RENTAL_LINKED_COUNT
