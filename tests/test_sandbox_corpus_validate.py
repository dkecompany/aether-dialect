"""Tests for shared sandbox corpus validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._sandbox import check_sandbox_faithfulness, question_ok


def test_question_ok_success_step_has_no_status() -> None:
    step = SimpleNamespace(done=True, sql="SELECT 1", status=None, error=None, message=None, intent=None)
    assert question_ok(step, "How many items are in the catalog by item type?")
    ok_step = SimpleNamespace(done=True, sql="SELECT 1", status="ok", error=None, message=None)
    bad_step = SimpleNamespace(done=True, sql=None, status="ok", error=None, message=None)
    assert question_ok(ok_step, "How many customers are there?")
    assert not question_ok(bad_step, "How many customers are there?")


def test_question_ok_allows_no_sql_for_weather_tour_question() -> None:
    step = SimpleNamespace(done=True, sql=None, error="rejected", status="invalid_question", message=None)
    assert question_ok(step, "What's the weather today?")


def test_question_ok_json_validation_failure() -> None:
    step = SimpleNamespace(
        done=True,
        sql=None,
        status="permission_denied",
        error="permission denied",
        message=None,
    )
    assert question_ok(step, "How many items are there?")


def test_faithfulness_gate_rejects_missing_bridge_table() -> None:
    intent = SimpleNamespace(tables=["item", "language"])
    step = SimpleNamespace(
        done=True,
        sql='SELECT "language"."name" FROM "item" JOIN "language" ON "item"."language_id" = "language"."language_id"',
        status="ok",
        error=None,
        message=None,
        intent=intent,
    )
    detail = check_sandbox_faithfulness(step, "Which games support English?")
    assert detail is not None
    assert "game_supported_language" in detail


def test_faithfulness_passes_when_required_tables_in_cte_or_sql() -> None:
    intent = RuntimeIntent(
        tables=["ranked_cities"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("city.name"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        cte_steps=[
            RuntimeCteStep(
                cte_name="ranked_cities",
                tables=["city", "customer"],
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("city.name"))],
                output_columns=["name"],
            )
        ],
        planner_cte_names=["ranked_cities"],
    )
    step = SimpleNamespace(
        done=True,
        sql='SELECT "city"."city" FROM "city" JOIN "customer" ON "customer"."city_id" = "city"."city_id"',
        status="ok",
        error=None,
        message=None,
        intent=intent,
    )
    assert check_sandbox_faithfulness(step, "Which city has the most customers?") is None


def test_question_ok_consumer_staff_expects_ok() -> None:
    step = SimpleNamespace(
        done=True,
        sql='SELECT "staff"."first_name" FROM "staff"',
        status="ok",
        error=None,
        message=None,
        intent=None,
    )
    assert question_ok(
        step,
        "Show active staff at each store.",
        profile="consumer_reader",
        tier="consumer_reader",
    )


def test_question_ok_owner_staff_expects_ok() -> None:
    step = SimpleNamespace(
        done=True,
        sql='SELECT "staff"."first_name" FROM "staff"',
        status="ok",
        error=None,
        message=None,
        intent=None,
    )
    assert question_ok(
        step,
        "Show active staff at each store.",
        profile="owner_writer",
        tier="questions",
    )


def test_expectation_index_resolves_profile_tier() -> None:
    from aetherdialect._sandbox import _expectation_payload_for_context

    question = "Show active staff at each store."
    owner = _expectation_payload_for_context(
        question,
        profile="owner_writer",
        tier="questions",
    )
    consumer = _expectation_payload_for_context(
        question,
        profile="consumer_reader",
        tier="consumer_reader",
    )
    assert owner is not None
    assert consumer is not None
    assert owner.get("terminal_status") == "ok"
    assert consumer.get("terminal_status") == "ok"


def test_expectations_json_loads_from_scripts_data() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "data" / "sandbox_expectations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots = payload["slots"]
    assert len(slots) >= 90
    reuse_rows = [row for row in slots if row.get("question") == "How many rentals happened in 2026?"]
    assert not reuse_rows
