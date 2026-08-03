"""Live federation boundary checks across PostgreSQL storefront and MySQL catalog."""

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
    HORROR_DISTINCT_CUSTOMERS,
    HORROR_PARTIAL_SUM_WRONG,
    JOIN_RENTAL_LINKED_TITLES,
    PAYMENT_TOTAL,
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
def test_federated_horror_partial_sum_oracle(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "how many distinct customers rented horror films",
    )
    assert step.done
    assert not step.error
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == HORROR_DISTINCT_CUSTOMERS
    assert int(value) != HORROR_PARTIAL_SUM_WRONG


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_type_fidelity_payment_decimal(federation_engine) -> None:
    step = _ask(federation_engine, "what is the total payment amount")
    assert step.done
    assert not step.error
    value = _scalar_value(step)
    assert value is not None
    assert isinstance(value, (Decimal, float, int))
    assert abs(Decimal(str(value)) - PAYMENT_TOTAL) < Decimal("0.05")


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_type_fidelity_join_count_integer(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "how many rentals are linked to film titles",
    )
    assert step.done
    assert not step.error
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == JOIN_RENTAL_LINKED_TITLES


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_literal_casing_horror_filter(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "how many distinct customers rented HORROR films",
    )
    assert step.done
    assert not step.error
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == HORROR_DISTINCT_CUSTOMERS


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_composition_round_trip_identity(federation_engine) -> None:
    composite = federation_engine._schema_graph
    assert composite is not None
    first_id = composite.schema_graph_id
    first_hash = composite.effective_structural_hash
    second = build_federation_live_engine()
    second_graph = second._schema_graph
    assert second_graph is not None
    assert second_graph.schema_graph_id == first_id
    assert second_graph.effective_structural_hash == first_hash


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_semijoin_reduction_preserves_scalar(federation_engine) -> None:
    question = "how many rentals are linked to film titles"
    step = _ask(federation_engine, question)
    assert step.done
    assert not step.error
    baseline = _scalar_value(step)
    assert baseline is not None
    assert int(baseline) == 18758


@pytest.mark.skipif(_SKIP, reason="postgres/mysql federation partitions unavailable")
def test_federated_member_failure_reports_error(federation_engine) -> None:
    step = _ask(
        federation_engine,
        "how many rows are in the missing_federation_table",
    )
    assert step.done
    assert step.error
    combined = f"{step.error} {getattr(step, 'message', '')}".lower()
    assert "missing_federation_table" in combined or "not found" in combined or "does not exist" in combined
