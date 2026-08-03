"""Unit tests for sandbox_corpus helper functions (no live LLM)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_feedback_scenario_reads_anchor_from_scenarios_json() -> None:
    sc = importlib.import_module("sandbox_corpus")
    scenario = sc._feedback_scenario()
    assert scenario.get("anchor_question") == "How many books do we have?"
    assert scenario.get("allowed_rejection_text") == "The intent is wrong, count distinct films only."


def test_handcrafted_entries_loaded_for_fail_everywhere_question() -> None:
    sc = importlib.import_module("sandbox_corpus")
    rows = sc._handcrafted_entries_for_question("How many rentals happened on 2025-01-01?")
    assert {str(row.get("stage")) for row in rows} == {"interpret", "ground", "compose"}


def test_handcrafted_stage_mapping_uses_intent_prompts() -> None:
    sc = importlib.import_module("sandbox_corpus")

    assert sc._handcrafted_stage_for_system(sc.INTENT_INTERPRET_SYSTEM) == "interpret"
    assert sc._handcrafted_stage_for_system(sc.INTENT_GROUND_SYSTEM) == "ground"
    assert sc._handcrafted_stage_for_system(sc.INTENT_COMPOSE_SYSTEM) == "compose"


def test_build_sandbox_catalog_uses_generated_pairs() -> None:
    sc = importlib.import_module("sandbox_corpus")
    catalog = sc.build_sandbox_catalog(
        paraphrase_pairs=[
            {
                "canonical": "How many rentals happened in 2025?",
                "paraphrases": ["How many rentals happened in 2026?"],
            }
        ]
    )
    assert catalog["paraphrase_pairs"] == [
        {
            "canonical": "How many rentals happened in 2025?",
            "paraphrases": ["How many rentals happened in 2026?"],
        }
    ]
    assert any(
        row["question"] == "How many rentals happened on 2025-01-01?" for row in catalog["validation_failure_demo"]
    )


def test_slot_id_for_feedback_kind() -> None:
    sc = importlib.import_module("sandbox_corpus")
    slot = sc.RecordingSlot(tier="feedback", label="reject text", kind="feedback")
    assert sc.slot_id_for(slot) == "feedback:reject text"


def test_recording_slots_exclude_consumer_reader() -> None:
    sc = importlib.import_module("sandbox_corpus")
    questions = sc.load_staging_questions()
    recording = sc.iter_recording_slots(questions)
    consumer = sc.iter_consumer_validation_slots(questions)
    assert not any(slot.tier == "consumer_reader" for slot in recording)
    assert len(consumer) == len(questions["questions"])
    assert all(slot.preset == "consumer_reader" for slot in consumer)
    assert len(recording) + len(consumer) >= 90


def test_paraphrase_catalog_ready_requires_nonempty_pairs(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    catalog_path = tmp_path / "sandbox_catalog.json"
    catalog_path.write_text('{"version": 1, "paraphrase_pairs": []}\n', encoding="utf-8")
    assert sc._paraphrase_catalog_ready(staging_dir=tmp_path) is False
    catalog_path.write_text(
        '{"version": 1, "paraphrase_pairs": [{"canonical": "q", "paraphrases": ["p"]}]}\n',
        encoding="utf-8",
    )
    assert sc._paraphrase_catalog_ready(staging_dir=tmp_path) is True


def test_reuse_fixtures_ready_detects_reuse_fixture(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    assert sc._reuse_fixtures_ready(corpus) is False
    corpus.fixtures.append(
        {
            "task": "default",
            "system": "x",
            "user": "How many rentals happened in 2026?",
            "output_text": "{}",
        },
    )
    assert sc._reuse_fixtures_ready(corpus) is True


def test_migration_fixtures_ready_detects_schema_base(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    assert sc._migration_fixtures_ready(corpus) is False
    corpus.fixtures.append(
        {
            "task": "schema_base",
            "system": "classify",
            "user": '{"tables":[{"name":"item","columns":[{"name":"item_title"}]}]}',
            "output_text": "{}",
        },
    )
    assert sc._migration_fixtures_ready(corpus) is True


def test_missing_paraphrase_canonicals(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    questions = {
        "questions": ["How many books do we have?", "How many games are in the catalog?"],
        "validation_failures": [],
        "feedback_samples": [],
    }
    manifest = {
        "slots": [
            {
                "slot_id": sc.slot_id_for(
                    sc.RecordingSlot(tier="questions", label="How many books do we have?"),
                ),
                "committed": True,
            },
            {
                "slot_id": sc.slot_id_for(
                    sc.RecordingSlot(tier="questions", label="How many games are in the catalog?"),
                ),
                "committed": True,
            },
        ],
    }
    catalog_path = tmp_path / "sandbox_catalog.json"
    catalog_path.write_text(
        '{"version": 1, "paraphrase_pairs": [{"canonical": "How many books do we have?", "paraphrases": ["x"]}]}\n',
        encoding="utf-8",
    )
    missing = sc._missing_paraphrase_canonicals(questions, manifest, staging_dir=tmp_path)
    assert [slot.label for slot in missing] == ["How many games are in the catalog?"]


def test_recording_pipeline_ready_flags_incomplete(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    questions = {
        "questions": ["How many books do we have?"],
        "validation_failures": [],
        "feedback_samples": [],
    }
    manifest = {"slots": []}
    fixtures_path = tmp_path / "fixtures" / "rental_shop_mock.json"
    fixtures_path.parent.mkdir(parents=True)
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    ready, reasons = sc._recording_pipeline_ready(
        questions=questions,
        manifest=manifest,
        corpus=corpus,
        staging_dir=tmp_path,
    )
    assert ready is False
    assert reasons


def test_corpus_snapshot_restore_roundtrip(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    corpus.fixtures.append(
        {
            "task": "default",
            "system": "gatekeeper",
            "user": "before",
            "output_text": "{}",
        },
    )
    corpus.seen = {sc.fixture_key(row) for row in corpus.fixtures}
    snap = sc._corpus_snapshot(corpus)
    corpus.fixtures.append(
        {
            "task": "default",
            "system": "gatekeeper",
            "user": "after",
            "output_text": "{}",
        },
    )
    corpus.seen.add(sc.fixture_key(corpus.fixtures[-1]))
    sc._restore_corpus_snapshot(corpus, snap)
    assert len(corpus.fixtures) == 1
    assert corpus.fixtures[0]["user"] == "before"


def test_mock_verify_targets_include_consumer_reader_for_practice_questions() -> None:
    sc = importlib.import_module("sandbox_corpus")
    slot = sc.RecordingSlot(tier="questions", label="How many books do we have?")
    targets = sc._mock_verify_targets_for_slot(slot)
    tiers = {tier for *_rest, tier in targets}
    assert tiers == {"questions", "consumer_reader"}
    sc = importlib.import_module("sandbox_corpus")

    assert sc._paraphrase_eligible_question("What's the weather today?") is False
    assert sc._paraphrase_eligible_question("What is the best pizza topping?") is False
    assert sc._paraphrase_eligible_question("How many books do we have?") is True
    assert sc._paraphrase_eligible_question("Show payroll deductions by employee SSN.", kind="question") is False


def test_prune_orphan_intent_rows_keeps_active_chain(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    question = "Which actors have the most film credits?"
    interpret_user = '{"interpret_plan_schema":{},"question":"Which actors have the most film credits?"}'
    active_interpret = {
        "task": "intent",
        "system": sc.INTENT_INTERPRET_SYSTEM,
        "user": interpret_user,
        "output_text": ('{"interpret_plan":{"approach":"active plan","tables":["actor"],"grounding":[]}}'),
    }
    active_ground = {
        "task": "intent",
        "system": sc.INTENT_GROUND_SYSTEM,
        "user": (
            '{"interpret_plan":{"approach":"active plan","tables":["actor"],"grounding":[]},"schema_literal_json":{}}'
        ),
        "output_text": '{"tables":["actor"]}',
    }
    stale_interpret = {
        "task": "intent",
        "system": sc.INTENT_INTERPRET_SYSTEM,
        "user": (
            '{"interpret_plan_schema":{},"question":"Which actors have the most film credits?",'
            '"recording_slot":"stale"}'
        ),
        "output_text": ('{"interpret_plan":{"approach":"stale plan","tables":["actor"],"grounding":[]}}'),
    }
    stale_ground = {
        "task": "intent",
        "system": sc.INTENT_GROUND_SYSTEM,
        "user": (
            '{"interpret_plan":{"approach":"stale plan","tables":["actor"],"grounding":[]},"schema_literal_json":{}}'
        ),
        "output_text": '{"tables":["actor","film_actor"]}',
    }
    unrelated = {
        "task": "default",
        "system": "gatekeeper",
        "user": "other question",
        "output_text": "{}",
    }
    fixtures = [
        dict(stale_interpret),
        dict(stale_ground),
        dict(active_interpret),
        dict(active_ground),
        dict(unrelated),
    ]
    keep_keys = sc._active_intent_keep_keys_for_question(fixtures, question)
    pruned, removed = sc._prune_orphan_intent_rows(fixtures, question, keep_keys)
    assert removed == 2
    intent_rows = [row for row in pruned if row.get("task") == "intent"]
    assert len(intent_rows) == 2
    assert all(
        "active plan" in row.get("user", "") or "active plan" in row.get("output_text", "") for row in intent_rows
    )
    assert any(row.get("task") == "default" for row in pruned)


def test_commit_slot_prunes_orphan_intent_for_question(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    question = "How many rentals were made in total?"
    stale = {
        "task": "intent",
        "system": sc.INTENT_GROUND_SYSTEM,
        "user": (
            '{"interpret_plan":{"approach":"old rentals total"},'
            '"question":"How many rentals were made in total?",'
            '"schema_literal_json":{}}'
        ),
        "output_text": "{}",
    }
    corpus.fixtures.append(dict(stale))
    corpus.seen.add(sc.fixture_key(stale))
    corpus.start_slot()
    fresh_interpret = {
        "task": "intent",
        "system": sc.INTENT_INTERPRET_SYSTEM,
        "user": f'{{"question":"{question}"}}',
        "output_text": '{"interpret_plan":{"approach":"fresh rentals total"}}',
    }
    fresh_ground = {
        "task": "intent",
        "system": sc.INTENT_GROUND_SYSTEM,
        "user": ('{"interpret_plan":{"approach":"fresh rentals total"},"schema_literal_json":{}}'),
        "output_text": '{"tables":["rental"]}',
    }
    corpus.record(
        task=fresh_interpret["task"],
        system=fresh_interpret["system"],
        user_key=fresh_interpret["user"],
        output_text=fresh_interpret["output_text"],
    )
    corpus.record(
        task=fresh_ground["task"],
        system=fresh_ground["system"],
        user_key=fresh_ground["user"],
        output_text=fresh_ground["output_text"],
    )
    removed = corpus.commit_slot(prune_question=question)
    assert removed == 1
    intent_rows = [row for row in corpus.fixtures if row.get("task") == "intent"]
    assert len(intent_rows) == 2
    assert not any("old rentals total" in row.get("user", "") for row in intent_rows)
