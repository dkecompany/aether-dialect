"""Join-path failures surface a dedicated hint and record deterministic feedback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import REPHRASE_HINT_MESSAGES
from aetherdialect._contracts_base import FailureCategory, NoJoinPathError, SqlDiagnostic, SqlDiagnosticCode
from aetherdialect._contracts_core import FeedbackKind, GenerationPath, RejectionBucket, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import RephraseHint
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import _join_path_failure_outcome, _resolve_joins_fresh, generate_and_validate_sql
from aetherdialect._templates import lookup_join_feedback_for_question, record_deterministic_join_failure_feedback


def _disconnected_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "alpha": TableMetadata(
                name="alpha",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "beta": TableMetadata(
                name="beta",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff",
    )


def test_no_join_path_error_exposes_user_message_with_table_names() -> None:
    exc = NoJoinPathError("main query", ["alpha", "beta"])
    assert "alpha" in exc.user_message
    assert "beta" in exc.user_message
    assert "could not be connected" in exc.user_message


def test_join_path_failure_hint_is_distinct_from_sql_validation_hint() -> None:
    join_hint = REPHRASE_HINT_MESSAGES[RephraseHint.JOIN_PATH_UNAVAILABLE.value]
    sql_hint = REPHRASE_HINT_MESSAGES[RephraseHint.SQL_VALIDATION_FAILED.value]
    assert join_hint != sql_hint
    assert "relationship" in join_hint.lower()


@patch("aetherdialect._pipeline.print_rephrase_hint")
@patch("aetherdialect._pipeline.save_template_store")
def test_join_path_failure_outcome_records_feedback_and_user_message(
    _mock_save,
    mock_hint,
) -> None:
    schema = _disconnected_schema()
    intent = RuntimeIntent(
        tables=["alpha", "beta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    store: dict = {"question_feedback": {}}
    exc = NoJoinPathError("main query", ["alpha", "beta"])
    outcome = _join_path_failure_outcome(
        exc,
        q_norm="show alpha beta",
        intent=intent,
        schema=schema,
        store=store,
        generation_path=GenerationPath.FRESH,
        matched_template=None,
        structural_match_templates=(),
        persist_template_learning=False,
    )
    assert outcome.success is False
    assert outcome.sql_validation_error == exc.user_message
    mock_hint.assert_called_once_with(RephraseHint.JOIN_PATH_UNAVAILABLE)
    feedback = lookup_join_feedback_for_question(store, "show alpha beta")
    assert feedback and exc.user_message in feedback[0]


def test_deterministic_join_failure_feedback_is_wrong_tables_bucket() -> None:
    schema = _disconnected_schema()
    intent = RuntimeIntent(
        tables=["alpha", "beta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    store: dict = {"question_feedback": {}}
    exc = NoJoinPathError("main query", ["alpha", "beta"])
    record_deterministic_join_failure_feedback(store, "q", exc, intent=intent, schema=schema)
    rows = store["question_feedback"]["q"]
    assert len(rows) == 1
    assert rows[0]["kind"] == FeedbackKind.INTENT_REJECTED.value
    assert RejectionBucket.WRONG_TABLES_OR_JOINS.value in rows[0]["buckets"]


def test_join_edges_from_signature_raises_when_path_is_disconnected() -> None:
    from aetherdialect._sql_gen import _join_edges_from_signature

    sig = ["a.id->b.id", "c.id->d.id"]
    with pytest.raises(NoJoinPathError) as exc_info:
        _join_edges_from_signature(sig, ["catalog_fk", "catalog_fk"], "a", None)
    assert "could not be connected" in exc_info.value.user_message


@patch("aetherdialect._pipeline.print_rephrase_hint")
@patch("aetherdialect._pipeline.save_template_store")
@patch("aetherdialect._pipeline.finalize_substitute_sql", return_value="SELECT alpha.id FROM alpha")
@patch("aetherdialect._pipeline._run_sql_validation_cascade")
@patch("aetherdialect._pipeline.apply_diagnostic_repairs")
@patch("aetherdialect._pipeline._resolve_joins_fresh")
@patch("aetherdialect._pipeline.build_deterministic_sql", return_value="SELECT alpha.id FROM alpha, beta")
@patch(
    "aetherdialect._pipeline._join_signatures_for_deterministic_from_anchor",
    return_value=([], {}),
)
def test_b3_diagnostic_retry_routes_no_join_path_to_join_failure_outcome(
    _mock_anchor,
    _mock_build_sql,
    mock_resolve,
    mock_apply_repair,
    mock_validate,
    _mock_finalize,
    _mock_save,
    mock_hint,
) -> None:
    schema = _disconnected_schema()
    intent = RuntimeIntent(
        tables=["alpha", "beta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    store: dict = {"question_feedback": {}}
    diag = SqlDiagnostic(code=SqlDiagnosticCode.UNKNOWN_COLUMN, message="bad column", offending_identifier="alpha.id")
    mock_validate.return_value = (False, "validation failed", FailureCategory.EXECUTION_SCHEMA_ERROR, [diag])
    mock_apply_repair.return_value = (intent, True)
    mock_resolve.side_effect = [
        ("SELECT alpha.id FROM alpha JOIN beta ON alpha.id = beta.id", {}),
        NoJoinPathError("main query", ["alpha", "beta"]),
    ]
    out = generate_and_validate_sql(
        "show alpha beta",
        intent,
        schema,
        {"candidates": []},
        {},
        MagicMock(),
        store,
    )
    assert out.success is False
    assert out.sql_validation_error is not None
    assert "could not be connected" in out.sql_validation_error
    mock_hint.assert_called_once_with(RephraseHint.JOIN_PATH_UNAVAILABLE)


def test_resolve_joins_fresh_pass_two_does_not_mutate_shared_join_candidates() -> None:
    from unittest.mock import MagicMock

    schema = _disconnected_schema()
    intent = RuntimeIntent(
        tables=["alpha", "beta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    original_candidates = [
        {"candidate_id": "J01", "join_path_signature": [], "edge_kinds": [], "edge_count": 0},
    ]
    join_candidates = {"candidates": list(original_candidates)}
    cmap = {"J01": []}
    det_sql = "SELECT alpha.id\nFROM alpha, beta"

    def _mutate_candidates(jc, *_args, **_kwargs):
        jc["candidates"].append({"candidate_id": "J99", "join_path_signature": [], "edge_kinds": [], "edge_count": 0})
        return {"candidates": jc["candidates"]}, {}

    with (
        patch("aetherdialect._pipeline.merge_join_hints_for_na_scopes", side_effect=_mutate_candidates),
        patch(
            "aetherdialect._pipeline.get_join_choice_from_llm",
            side_effect=[
                {"join_choice:main": "NA"},
                {"join_choice:main": "J01"},
            ],
        ),
        patch(
            "aetherdialect._pipeline.join_scope_pass1_plan",
            return_value=({}, [{"scope": "join_choice:main"}], {"join_choice:main": True}, {}),
        ),
        patch(
            "aetherdialect._pipeline.join_scope_pass2_llm_scopes",
            return_value=[{"scope": "join_choice:main"}],
        ),
        pytest.raises(NoJoinPathError),
    ):
        _resolve_joins_fresh(
            det_sql,
            intent,
            cmap,
            {},
            "q",
            join_candidates,
            schema=schema,
            dialect=MagicMock(),
        )
    assert join_candidates["candidates"] == original_candidates
