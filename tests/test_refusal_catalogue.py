"""Refusal catalogue completeness and single-source user text."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
    DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
    REFUSAL_CATALOGUE,
    REFUSAL_CONDITION_CODES,
    REFUSAL_DIAGNOSTIC_CODES,
    REPHRASE_HINT_MESSAGES,
    REPHRASE_HINT_REFUSAL_CODES,
)
from aetherdialect._contracts_base import (
    AggregateJoinFanOutError,
    AmbiguousDateLiteralError,
    ClauseWidenedRowsetError,
    ComparisonJoinScopeExceededError,
    FailureCategory,
    JoinPathTieCapExceededError,
    NoJoinPathError,
    NullInNegatedListError,
    ProbeCtePlacementError,
    RefusalCondition,
    SubdayDateWindowOnDateColumnError,
)
from aetherdialect._contracts_schema import IntentIssue
from aetherdialect._core_utils import (
    refusal_diagnostic_code_for_exception,
    refusal_diagnostic_code_for_federation_reason,
    refusal_diagnostic_code_for_intent_issue,
    refusal_diagnostic_code_for_outcome,
    refusal_diagnostic_code_for_rephrase_hint_key,
    refusal_reformulation_hint_for_code,
    refusal_user_text_for_code,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "aetherdialect"

_REFUSAL_REPHRASE_KEYS = frozenset(REPHRASE_HINT_REFUSAL_CODES.keys())

_CATALOGUE_HELPER_FUNCTIONS = frozenset(
    {
        "refusal_user_text_for_code",
        "refusal_reformulation_hint_for_code",
        "refusal_reformulation_hint_for_rephrase_hint_key",
        "refusal_message_for_exception",
    }
)

_SKIP_INLINE_REFUSAL_SCAN = frozenset(
    {
        "_federation.py",
        "_pipeline.py",
        "_intent_process.py",
        "_validation_semantic.py",
    }
)


@pytest.mark.fast
def test_every_refusal_condition_has_a_code() -> None:
    for condition in RefusalCondition:
        code = REFUSAL_CONDITION_CODES[condition.value]
        assert code in REFUSAL_CATALOGUE
        assert code in REFUSAL_DIAGNOSTIC_CODES

    assert refusal_diagnostic_code_for_outcome("permission_denied") == DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED
    assert refusal_diagnostic_code_for_outcome("restricted") == DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION
    assert refusal_diagnostic_code_for_outcome("invalid_question") == DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION
    assert refusal_diagnostic_code_for_outcome("parse_failed") == DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE
    assert refusal_diagnostic_code_for_outcome("schema_invalid_declined") == DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA

    assert (
        refusal_diagnostic_code_for_exception(NoJoinPathError("main query", ["a", "b"]))
        == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    )
    assert (
        refusal_diagnostic_code_for_exception(AggregateJoinFanOutError("main query", "fan-out"))
        == DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT
    )
    assert (
        refusal_diagnostic_code_for_exception(ComparisonJoinScopeExceededError("main query", "hop ceiling"))
        == DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING
    )
    assert (
        refusal_diagnostic_code_for_exception(JoinPathTieCapExceededError("alpha", "beta", 8, 4))
        == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP
    )
    assert (
        refusal_diagnostic_code_for_exception(ClauseWidenedRowsetError("main query", "widened"))
        == DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET
    )
    assert (
        refusal_diagnostic_code_for_exception(ProbeCtePlacementError("main query", "probe placement"))
        == DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT
    )
    assert (
        refusal_diagnostic_code_for_exception(NullInNegatedListError("t.c", "null in list"))
        == DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST
    )
    assert (
        refusal_diagnostic_code_for_exception(SubdayDateWindowOnDateColumnError("t.d", "subday"))
        == DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN
    )
    assert (
        refusal_diagnostic_code_for_exception(AmbiguousDateLiteralError("01/02/03", "ambiguous"))
        == DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL
    )

    denied_issue = IntentIssue(
        issue_id="denied_col",
        category=FailureCategory.DENIED_REFERENCE,
        severity="error",
        message="denied",
    )
    assert refusal_diagnostic_code_for_intent_issue(denied_issue) == DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
    assert (
        refusal_diagnostic_code_for_intent_issue(
            IntentIssue(
                issue_id="cte_step_count_exceeded",
                category=FailureCategory.CTE_STRUCTURE,
                severity="error",
                message="cap",
            )
        )
        == DIAGNOSTIC_CODE_REFUSAL_CTE_CAP
    )
    assert (
        refusal_diagnostic_code_for_intent_issue(
            IntentIssue(
                issue_id="clause_widened_rowset_limit_main",
                category=FailureCategory.WRONG_SORT_OR_LIMIT,
                severity="error",
                message="limit",
            )
        )
        == DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET
    )
    assert (
        refusal_diagnostic_code_for_intent_issue(
            IntentIssue(
                issue_id="probe_cte_left_operand_probe",
                category=FailureCategory.CTE_STRUCTURE,
                severity="error",
                message="probe",
            )
        )
        == DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT
    )
    assert (
        refusal_diagnostic_code_for_federation_reason("union logical column missing on member")
        == DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING
    )
    assert (
        refusal_diagnostic_code_for_federation_reason("member capability: ilike unsupported")
        == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    )

    for key in _REFUSAL_REPHRASE_KEYS:
        assert refusal_diagnostic_code_for_rephrase_hint_key(key) in REFUSAL_DIAGNOSTIC_CODES


@pytest.mark.fast
def test_every_entry_has_a_reformulation_hint() -> None:
    for code, entry in REFUSAL_CATALOGUE.items():
        assert "user_text" in entry and entry["user_text"].strip()
        assert "reformulation_hint" in entry and entry["reformulation_hint"].strip()
        assert refusal_reformulation_hint_for_code(code) == entry["reformulation_hint"]


@pytest.mark.fast
def test_no_inline_refusal_strings() -> None:
    for key, code in REPHRASE_HINT_REFUSAL_CODES.items():
        assert REPHRASE_HINT_MESSAGES[key] == REFUSAL_CATALOGUE[code]["reformulation_hint"]

    catalogue_user_texts = {entry["user_text"] for entry in REFUSAL_CATALOGUE.values()}
    for code in REFUSAL_DIAGNOSTIC_CODES:
        catalogue_user_texts.add(refusal_user_text_for_code(code))

    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if path.name in _SKIP_INLINE_REFUSAL_SCAN or path.name == "_constants.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _InlineRefusalStringVisitor(catalogue_user_texts, path)
        visitor.visit(tree)
        offenders.extend(visitor.offenders)

    assert not offenders, f"inline refusal user text outside catalogue: {offenders[:20]}"


class _InlineRefusalStringVisitor(ast.NodeVisitor):
    def __init__(self, catalogue_texts: set[str], path: Path) -> None:
        self._catalogue_texts = catalogue_texts
        self._path = path
        self.offenders: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _CATALOGUE_HELPER_FUNCTIONS:
            self.generic_visit(node)
            return
        if isinstance(func, ast.Attribute) and func.attr in _CATALOGUE_HELPER_FUNCTIONS:
            self.generic_visit(node)
            return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        text = node.value.strip()
        if len(text) < 24:
            return
        if text in self._catalogue_texts and self._path.name != "_core_utils.py":
            self.offenders.append(f"{self._path}:{node.lineno}:{text[:48]}...")
