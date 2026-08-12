"""Hermetic tests for zero-LLM sandbox fixture fan-out."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_CORPUS = importlib.import_module("sandbox_corpus")

_STUB_LITERALS = {
    "owner": (
        '{"rental":{"columns":{"rental_id":"int"}},"film":{"columns":{"film_id":"int"}},'
        '"customer":{"columns":{"customer_id":"int"}}}'
    ),
    "consumer": '{"book":{"columns":{"book_id":"int"}}}',
}


def _owner_schema_body() -> dict[str, object]:
    return {
        "rental": {"columns": {"rental_id": "int"}},
        "film": {"columns": {"film_id": "int"}},
        "customer": {"columns": {"customer_id": "int"}},
    }


@pytest.fixture(autouse=True)
def _pin_stub_literals() -> None:
    from aetherdialect._llm_provider import MockProvider

    MockProvider._pin_mock_schema_literals(_STUB_LITERALS)


def test_question_home_member_and_full_only() -> None:
    assert _CORPUS.question_home_member("How many books do we have?") == "catalog"
    assert _CORPUS.is_full_only_question("How many customers are there?") is True
    assert _CORPUS.question_home_member("How many customers are there?") is None


def test_adapt_fixture_user_swaps_schema_literal_stub() -> None:
    owner_user = _CORPUS.MockProvider.mock_fixture_user_key(
        json.dumps(
            {
                "question": "How many books do we have?",
                "schema_literal_json": _owner_schema_body(),
            },
        ),
        literals=_STUB_LITERALS,
    )
    consumer_user = _CORPUS.adapt_fixture_user_for_surface(
        owner_user,
        source_slot="owner",
        target_slot="consumer",
        literals=_STUB_LITERALS,
    )
    assert consumer_user != owner_user
    assert "rental" in owner_user or "film" in owner_user or "customer" in owner_user
    assert "book" in consumer_user


def test_plan_fixture_fan_out_derives_consumer_reader_keys() -> None:
    question = "How many books do we have?"
    owner_user = _CORPUS.MockProvider.mock_fixture_user_key(
        json.dumps({"question": question, "schema_literal_json": _owner_schema_body()}),
        literals=_STUB_LITERALS,
    )
    canonical = [
        {
            "task": "intent",
            "system": "interpret",
            "user": owner_user,
            "output_text": '{"interpret_plan":{"approach":"count books","tables":["book"],"grounding":[]}}',
        },
        {
            "task": "default",
            "system": f"gatekeeper {_CORPUS._GATEKEEPER_MARKER}",
            "user": question,
            "output_text": '{"valid_database_question":"yes","query_type":"analytical","corrected":""}',
        },
    ]
    slot = _CORPUS.RecordingSlot(tier="questions", label=question)
    seen = {_CORPUS.fixture_key(row) for row in canonical}
    rows, notes = _CORPUS.plan_fixture_fan_out(
        canonical,
        [slot],
        seen,
        construction_for_slot=_CORPUS._construction_for_slot,
        recipe_for_slot=_CORPUS._recipe_for_slot,
        federation_ineligible=_CORPUS.FEDERATION_INELIGIBLE_QUESTIONS,
        fixture_key=_CORPUS.fixture_key,
        literals=_STUB_LITERALS,
        include_scope_refusals=False,
    )
    assert rows
    assert any("fan-out" in note for note in notes)
    intent_rows = [row for row in rows if row.get("task") == "intent"]
    assert intent_rows
    assert intent_rows[0]["user"] != owner_user


def test_scope_refusal_rows_for_out_of_scope_member_question() -> None:
    question = "How many customers are there?"
    rows = _CORPUS.scope_refusal_rows_for_question(
        question,
        member="catalog",
        literals=_STUB_LITERALS,
        gatekeeper_system=_CORPUS._GATEKEEPER_MARKER,
    )
    assert len(rows) >= 1
    gatekeeper = rows[0]
    assert gatekeeper["task"] == "default"
    payload = json.loads(gatekeeper["output_text"])
    assert payload["valid_database_question"] == "no"


def test_plan_fixture_fan_out_adds_refusals_for_full_only_question() -> None:
    question = "How many customers are there?"
    slot = _CORPUS.RecordingSlot(tier="questions", label=question)
    canonical = [
        {
            "task": "default",
            "system": _CORPUS._GATEKEEPER_MARKER,
            "user": question,
            "output_text": '{"valid_database_question":"yes","query_type":"analytical","corrected":""}',
        },
    ]
    seen = {_CORPUS.fixture_key(row) for row in canonical}
    rows, notes = _CORPUS.plan_fixture_fan_out(
        canonical,
        [slot],
        seen,
        construction_for_slot=_CORPUS._construction_for_slot,
        recipe_for_slot=_CORPUS._recipe_for_slot,
        federation_ineligible=_CORPUS.FEDERATION_INELIGIBLE_QUESTIONS,
        fixture_key=_CORPUS.fixture_key,
        literals=_STUB_LITERALS,
        include_scope_refusals=True,
    )
    refusal_rows = [row for row in rows if "out of scope" in row.get("output_text", "")]
    assert refusal_rows
    assert any("scope refusal" in note for note in notes)


def test_fixture_rows_for_question_matches_intent_payload() -> None:
    question = "How many games are in the catalog?"
    row = {
        "task": "intent",
        "system": "x",
        "user": json.dumps({"question": question, "schema_literal_json": {}}),
        "output_text": "{}",
    }
    matched = _CORPUS.fixture_rows_for_question([row], question)
    assert matched == [row]
