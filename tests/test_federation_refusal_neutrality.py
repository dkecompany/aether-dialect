"""Federation planner refusals must not leak member identity or cross- source vocabulary."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE
from aetherdialect._contracts_core import FederatedPlan
from aetherdialect._federation_execute import federation_user_facing_ineligible_message
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)

_CROSS_SOURCE_VOCAB = (
    "cross-source",
    "cross source",
    "federated",
    "members",
    "member_id",
    "member id",
)


@pytest.mark.parametrize(
    "raw_reason",
    [
        "member capability: where operator 'contains' is not supported by federation member 'secret_member'",
        "cross-source join path is not declared for referenced sources",
        "cross-source aggregate not supported: sum across ta and tb",
        "median is not supported by all federation members",
    ],
)
@pytest.mark.fast
def test_no_member_identity_in_user_text(raw_reason: str) -> None:
    user_text = federation_user_facing_ineligible_message(raw_reason)
    assert "secret_member" not in user_text
    assert "'secret_member'" not in user_text
    for token in ("member capability", "federation member"):
        assert token not in user_text.lower()


@pytest.mark.parametrize(
    "raw_reason",
    [
        "member capability: where operator 'contains' is not supported by federation member 'secret_member'",
        "cross-source join path is not declared for referenced sources",
        "cross-source aggregate not supported: sum across ta and tb",
        "median is not supported by all federation members",
    ],
)
@pytest.mark.fast
def test_no_cross_source_vocabulary_in_user_text(raw_reason: str) -> None:
    user_text = federation_user_facing_ineligible_message(raw_reason).lower()
    for term in _CROSS_SOURCE_VOCAB:
        assert term not in user_text


@pytest.mark.fast
def test_ineligible_handler_user_diagnostics_are_sanitized() -> None:
    raw_reason = "member capability: where operator 'ilike' is not supported by federation member 'secret_member'"
    plan = FederatedPlan(steps=(), ineligible_reason=raw_reason)
    token = set_diagnostic_collector([])
    try:
        with patch("aetherdialect._utils.print_rephrase_hint"):
            MainExecutionOps._handle_federation_ineligible_plan(
                plan,
                choice_port=None,
                store={},
                owner=None,
                persist_template_learning=False,
            )
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    user_messages = [d.message for d in diags if d.code != "engine_info"]
    combined = "\n".join(user_messages).lower()
    assert "secret_member" not in combined
    for term in _CROSS_SOURCE_VOCAB:
        assert term not in combined

    ineligible_diags = [d for d in diags if d.code == DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE]
    assert len(ineligible_diags) == 1
    detail_map = dict(ineligible_diags[0].details)
    assert detail_map["reason"] == raw_reason
