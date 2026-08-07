"""Live federation checks across PostgreSQL storefront and MySQL catalog."""

from __future__ import annotations

from decimal import Decimal

import pytest

from aetherdialect._core_utils import llm_usage_question_scope

from ._federation_live import (
    build_federation_live_engine,
    ensure_federation_partitions_loaded,
    federation_partitions_available,
)
from .federation_oracles import (
    LIVE_FULL_HORROR_DISTINCT_CUSTOMERS,
    LIVE_FULL_JOIN_RENTAL_LINKED_TITLES,
    LIVE_FULL_PAYMENT_TOTAL,
)

pytestmark = pytest.mark.live

_SKIP = not federation_partitions_available()


@pytest.fixture(scope="module")
def federation_engine():
    """Module-scoped federated engine with partition data loaded."""
    if _SKIP:
        pytest.skip("postgres/mysql federation partitions unavailable")
    ensure_federation_partitions_loaded()
    return build_federation_live_engine()


def _ask(engine, question: str):
    with llm_usage_question_scope():
        with engine.session() as session:
            return session.accept_until_done(question)


def _scalar_value(step) -> float | int | Decimal | None:
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


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_cross_source_join(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "how many rentals are linked to film titles",
    )
    assert step.done
    assert not step.error
    assert step.sql
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == LIVE_FULL_JOIN_RENTAL_LINKED_TITLES


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_payment_union(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "what is the total payment amount",
    )
    assert step.done
    assert not step.error
    assert step.sql
    value = _scalar_value(step)
    assert value is not None
    assert abs(Decimal(str(value)) - LIVE_FULL_PAYMENT_TOTAL) < Decimal("0.05")


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_horror_customers(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "how many distinct customers rented horror films",
    )
    assert step.done
    assert not step.error
    assert step.sql
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == LIVE_FULL_HORROR_DISTINCT_CUSTOMERS


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_ineligible_aggregate(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "what is the average payment amount grouped by film category",
    )
    assert step.done
    assert step.error
    combined = f"{step.error} {getattr(step, 'message', '')}".lower()
    assert "cross-source aggregate" in combined
