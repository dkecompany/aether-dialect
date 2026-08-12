"""Live federation coverage for four-member topology (collected, not run in fast suite)."""

from __future__ import annotations

import pytest

from .live_support import (
    build_federation_live_engine,
    ensure_federation_partitions_loaded,
    federation_partitions_available,
)
from .mydb_profile import (
    assert_delivery_rental_cross_source_linked_count,
    assert_median_payment_refused_for_mysql_family,
    assert_staff_email_from_storefront_authoritative,
)

pytestmark = pytest.mark.live

_SKIP = not federation_partitions_available()


@pytest.fixture(scope="module")
def federation_engine():
    if _SKIP:
        pytest.skip("federation partition databases unavailable")
    ensure_federation_partitions_loaded()
    return build_federation_live_engine()


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_customer_query_prefers_storefront_authoritative_member(federation_engine) -> None:
    with federation_engine.session() as session:
        step = session.accept_until_done("how many customers are there")
    assert step.done
    assert not step.error
    assert step.sql
    assert "storefront" in str(step.sql).lower() or "customer" in str(step.sql).lower()


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_staff_sensitive_column_resolves_to_storefront_not_crm_mirror(federation_engine) -> None:
    with federation_engine.session() as session:
        step = session.accept_until_done("list staff email addresses")
    assert_staff_email_from_storefront_authoritative(step)


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_median_aggregate_refuses_on_mysql_family_members(federation_engine) -> None:
    with federation_engine.session() as session:
        step = session.accept_until_done("what is the median payment amount")
    assert_median_payment_refused_for_mysql_family(step)


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_logistics_to_storefront_join_plans_as_cross_source(federation_engine) -> None:
    with federation_engine.session() as session:
        step = session.accept_until_done("how many deliveries are linked to rentals")
    assert_delivery_rental_cross_source_linked_count(step)
