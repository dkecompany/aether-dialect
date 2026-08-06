"""Refusal messages must not instruct end users to change configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._constants import REMEDIATION_RESTRICTED_QUESTION, REPHRASE_HINT_MESSAGES
from aetherdialect._contracts_base import QuestionRoute, QuestionValidationResult
from aetherdialect._core_utils import (
    RephraseHint,
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._main_execution import MainExecutionOps

_CONFIG_TERMS = ("deny_columns", "allow_columns", "deny columns", "allow columns")


@pytest.mark.fast
def test_no_configuration_advice_in_user_text() -> None:
    user_text = REPHRASE_HINT_MESSAGES["restricted_question"].lower()
    for term in _CONFIG_TERMS:
        assert term not in user_text


@pytest.mark.fast
def test_restricted_question_emits_operator_remediation() -> None:
    token = set_diagnostic_collector([])
    try:
        with (
            patch(
                "aetherdialect._main_execution.validate_question",
                return_value=QuestionValidationResult(accepted=False, route=QuestionRoute.RESTRICTED, corrected=""),
            ),
            patch("aetherdialect._main_execution.load_pipeline_resources") as mock_load,
            patch("aetherdialect._main_execution.match_question_level_template_reuse") as mock_match,
        ):
            mock_load.return_value = (None, None, {}, {}, {}, set())
            mock_match.return_value = type("Reuse", (), {"reuse_type": "none", "best_template": None})()
            MainExecutionOps.interactive_run_once(question="show secret salary column", pipeline_session=None)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    restricted_diags = [d for d in diags if d.stage == "rephrase_hint" or "restricted" in d.message.lower()]
    assert restricted_diags, "expected a restricted-question diagnostic"
    assert any(d.remediation == REMEDIATION_RESTRICTED_QUESTION for d in restricted_diags)
    for diag in restricted_diags:
        lowered = diag.message.lower()
        for term in _CONFIG_TERMS:
            assert term not in lowered


@pytest.mark.fast
def test_restricted_rephrase_hint_constant_has_no_config_advice() -> None:
    hint = REPHRASE_HINT_MESSAGES[RephraseHint.RESTRICTED_QUESTION.value].lower()
    for term in _CONFIG_TERMS:
        assert term not in hint
