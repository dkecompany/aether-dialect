"""Refusal messages must not instruct end users to change configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._constants_runtime import REPHRASE_HINT_MESSAGES
from aetherdialect._contracts_core import QuestionRoute, QuestionValidationResult, RephraseHint
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)

_CONFIG_TERMS = ("deny_columns", "allow_columns", "deny columns", "allow columns")


@pytest.mark.fast
def test_no_configuration_advice_in_user_text() -> None:
    user_text = REPHRASE_HINT_MESSAGES["restricted_question"].lower()
    for term in _CONFIG_TERMS:
        assert term not in user_text


@pytest.mark.fast
def test_restricted_question_emits_operation_not_supported_message() -> None:
    token = set_diagnostic_collector([])
    try:
        with (
            patch(
                "aetherdialect._main_init.validate_question",
                return_value=QuestionValidationResult(accepted=False, route=QuestionRoute.RESTRICTED, corrected=""),
            ),
            patch("aetherdialect._main_init.load_pipeline_resources") as mock_load,
            patch("aetherdialect._main_init.match_question_level_template_reuse") as mock_match,
        ):
            mock_load.return_value = (None, None, {}, {}, {}, set())
            mock_match.return_value = type("Reuse", (), {"reuse_type": "none", "best_template": None})()
            MainExecutionOps.interactive_run_once(question="delete every table", pipeline_session=None)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    restricted_diags = [d for d in diags if d.stage == "rephrase_hint" or "restricted" in d.message.lower()]
    assert restricted_diags, "expected a restricted-question diagnostic"
    assert not any(d.remediation for d in restricted_diags)
    for diag in restricted_diags:
        lowered = diag.message.lower()
        assert "operation" in lowered or "not supported" in lowered
        for term in _CONFIG_TERMS:
            assert term not in lowered


@pytest.mark.fast
def test_restricted_question_remediation_has_no_scope_mechanism_wording() -> None:
    hint = REPHRASE_HINT_MESSAGES[RephraseHint.RESTRICTED_QUESTION.value].lower()
    assert "access scope" not in hint
    assert "rephrase" not in hint
    assert "schema" not in hint
    assert "operation" in hint or "not supported" in hint
    for term in _CONFIG_TERMS:
        assert term not in hint
