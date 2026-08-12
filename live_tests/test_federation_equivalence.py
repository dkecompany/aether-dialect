"""Live federation vs single-engine result-frame equivalence (operator sequence)."""

from __future__ import annotations

import pytest

from aetherdialect import Sandbox
from aetherdialect._utils import llm_usage_question_scope

from .conftest import _build_live_aether_engine
from .live_support import (
    FederationEquivalenceQuestion,
    build_federation_live_engine,
    ensure_federation_partitions_loaded,
    federation_partitions_available,
    generate_federation_equivalence_questions,
)

pytestmark = [pytest.mark.needs_corpus, pytest.mark.live]

_SKIP_PARTITIONS = not federation_partitions_available()
_CORPUS_SKIP_REASON = "needs_corpus: bundled sandbox data.zip absent"


def _require_corpus() -> None:
    if not Sandbox.data_zip_path().is_file():
        pytest.skip(_CORPUS_SKIP_REASON)
    try:
        if Sandbox.sandbox_doctor():
            pytest.skip(_CORPUS_SKIP_REASON)
    except Exception:
        pytest.skip(_CORPUS_SKIP_REASON)


def _canonical_sort_frame(frame):
    import pandas as pd

    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    if frame.empty:
        return frame.copy()
    normalized = frame.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: value.isoformat() if hasattr(value, "isoformat") else value
        )
    sort_columns = sorted(normalized.columns.tolist())
    normalized = normalized[sort_columns]
    return normalized.sort_values(by=sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)


def _first_frame_difference(left, right):
    left_sorted = _canonical_sort_frame(left)
    right_sorted = _canonical_sort_frame(right)
    if left_sorted.equals(right_sorted):
        return None
    max_rows = max(len(left_sorted), len(right_sorted))
    for row_index in range(max_rows):
        left_row = left_sorted.iloc[row_index].to_dict() if row_index < len(left_sorted) else None
        right_row = right_sorted.iloc[row_index].to_dict() if row_index < len(right_sorted) else None
        if left_row != right_row:
            return {
                "row_index": row_index,
                "single_engine_row": left_row,
                "federation_row": right_row,
            }
    return {
        "row_index": 0,
        "single_engine_row": left_sorted.head(1).to_dict(orient="records"),
        "federation_row": right_sorted.head(1).to_dict(orient="records"),
    }


def _ask(engine, question: str):
    with llm_usage_question_scope():
        with engine.session() as session:
            return session.accept_until_done(question)


@pytest.fixture(scope="module")
def equivalence_engines():
    """Module-scoped single-engine and four-member federation engines."""
    _require_corpus()
    if _SKIP_PARTITIONS:
        pytest.skip("federation partition databases unavailable")
    ensure_federation_partitions_loaded()
    single_engine = _build_live_aether_engine(relax_sensitivity=True)
    federation_engine = build_federation_live_engine()
    return single_engine, federation_engine


@pytest.fixture(scope="module")
def equivalence_questions(equivalence_engines) -> list[FederationEquivalenceQuestion]:
    del equivalence_engines
    _require_corpus()
    schema = _build_live_aether_engine(relax_sensitivity=True)._schema_graph
    questions = generate_federation_equivalence_questions(schema)
    if len(questions) < 100:
        pytest.fail(f"expected several hundred equivalence questions, got {len(questions)}")
    return questions


@pytest.mark.skipif(_SKIP_PARTITIONS, reason="federation partition databases unavailable")
def test_federation_matches_single_engine_on_generated_questions(
    equivalence_engines,
    equivalence_questions: list[FederationEquivalenceQuestion],
) -> None:
    """Each schema-derived question should return the same frame from both engines."""
    _require_corpus()
    single_engine, federation_engine = equivalence_engines
    for item in equivalence_questions:
        single_step = _ask(single_engine, item.question)
        federated_step = _ask(federation_engine, item.question)
        assert single_step.done, (
            f"{item.question_id}: single-engine turn did not complete: "
            f"{getattr(single_step, 'error', '')} {getattr(single_step, 'message', '')}"
        )
        assert federated_step.done, (
            f"{item.question_id}: federation turn did not complete: "
            f"{getattr(federated_step, 'error', '')} {getattr(federated_step, 'message', '')}"
        )
        if single_step.error or federated_step.error:
            continue
        single_frame = getattr(single_step, "data", None)
        federated_frame = getattr(federated_step, "data", None)
        if _canonical_sort_frame(single_frame).equals(_canonical_sort_frame(federated_frame)):
            continue
        difference = _first_frame_difference(single_frame, federated_frame)
        pytest.fail(
            "\n".join(
                [
                    f"equivalence mismatch for {item.question_id}",
                    f"question: {item.question}",
                    f"single-engine sql: {getattr(single_step, 'sql', '')}",
                    f"federation sql: {getattr(federated_step, 'sql', '')}",
                    f"first differing row: {difference}",
                ]
            )
        )
