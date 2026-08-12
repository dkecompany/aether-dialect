"""Fast regressions: federation member live assertions reject tautology- passing steps."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
from aetherdialect._contracts_base import Diagnostic
from aetherdialect._contracts_core import SessionError, SessionOutcome, SessionStep


@pytest.mark.fast
def test_tautology_assertions_rejected() -> None:
    from live_tests.mydb_profile import (
        assert_delivery_rental_cross_source_linked_count,
        assert_median_payment_refused_for_mysql_family,
        assert_staff_email_from_storefront_authoritative,
    )

    wrong_success = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        sql="SELECT 1",
        error=None,
    )
    with pytest.raises(AssertionError):
        assert_staff_email_from_storefront_authoritative(wrong_success)
    with pytest.raises(AssertionError):
        assert_median_payment_refused_for_mysql_family(wrong_success)
    with pytest.raises(AssertionError):
        assert_delivery_rental_cross_source_linked_count(wrong_success)

    wrong_median_code = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        sql="SELECT AVG(amount) FROM payment",
        error=SessionError(
            code=SessionOutcome.EXECUTION_FAILED,
            detail_code="SOME_OTHER_CODE",
        ),
    )
    with pytest.raises(AssertionError):
        assert_median_payment_refused_for_mysql_family(wrong_median_code)

    tautology_median = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        sql="SELECT COUNT(*) FROM payment",
        error=None,
    )
    with pytest.raises(AssertionError):
        assert_median_payment_refused_for_mysql_family(tautology_median)

    plausible_median_refusal = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        error=SessionError(
            code=SessionOutcome.UNANSWERABLE,
            detail_code=DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
        ),
        diagnostics=(
            Diagnostic(
                stage="execute",
                level="error",
                code=DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
                message="median is not supported by all federation members",
            ),
        ),
    )
    assert_median_payment_refused_for_mysql_family(plausible_median_refusal)
