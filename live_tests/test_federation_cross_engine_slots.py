"""Ten numerically asserted live federation slots across PostgreSQL storefront and MySQL catalog."""

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
    ACTOR_COUNT,
    CATEGORY_COUNT,
    CUSTOMER_COUNT,
    FILM_CATALOG_ROW_COUNT,
    HORROR_DISTINCT_CUSTOMERS,
    INVENTORY_COUNT,
    JOIN_RENTAL_LINKED_TITLES,
    PAYMENT_TOTAL,
    RENTAL_COUNT,
)

pytestmark = pytest.mark.live

_SKIP = not federation_partitions_available()


@pytest.fixture(scope="module")
def federation_engine():
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


@pytest.mark.parametrize(
    ("slot_id", "question", "expected"),
    [
        ("join_rental_linked_titles", "how many rentals are linked to film titles", JOIN_RENTAL_LINKED_TITLES),
        ("payment_total", "what is the total payment amount", PAYMENT_TOTAL),
        ("horror_distinct_customers", "how many distinct customers rented horror films", HORROR_DISTINCT_CUSTOMERS),
        ("actor_count", "how many actors are in the database", ACTOR_COUNT),
        ("customer_count", "how many customers do we have", CUSTOMER_COUNT),
        ("rental_count", "how many rentals were made in total", RENTAL_COUNT),
        ("inventory_count", "how many inventory items are there", INVENTORY_COUNT),
        ("category_count", "how many film categories are there", CATEGORY_COUNT),
        ("horror_distinct_uppercase", "how many distinct customers rented HORROR films", HORROR_DISTINCT_CUSTOMERS),
    ],
)
@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federation_numeric_slot(federation_engine, slot_id: str, question: str, expected) -> None:
    del slot_id
    step = _ask(federation_engine, question)
    assert step.done
    assert not step.error, getattr(step, "message", "")
    value = _scalar_value(step)
    assert value is not None
    if isinstance(expected, Decimal):
        assert abs(Decimal(str(value)) - expected) < Decimal("0.05")
    else:
        assert int(value) == int(expected)


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federation_film_catalog_row_count_slot(federation_engine) -> None:
    step = _ask(federation_engine, "list all films in the catalog")
    assert step.done
    assert not step.error
    data = getattr(step, "data", None)
    assert data is not None
    assert len(data) == FILM_CATALOG_ROW_COUNT
