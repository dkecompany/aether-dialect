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


def test_smoke_questions_include_flaky_practice_pair() -> None:
    sc = importlib.import_module("sandbox_corpus")
    smoke = sc.smoke_questions()
    assert smoke["questions"] == [
        sc.SMOKE_TOUR_QUESTION,
        sc.SMOKE_FLAKY_QUESTION,
    ]
    assert smoke["validation_failures"]
    assert smoke["feedback_samples"]


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


def test_paraphrase_gatekeeper_ready_detects_gatekeeper_row(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    paraphrase = "How many rentals happened in 2026?"
    assert sc._paraphrase_gatekeeper_ready(corpus, paraphrase) is False
    corpus.fixtures.append(
        {
            "task": "default",
            "system": "You decide if user input is a database query request or not.",
            "user": paraphrase,
            "output_text": "{}",
        },
    )
    assert sc._paraphrase_gatekeeper_ready(corpus, paraphrase) is True


def test_reuse_fixtures_ready_requires_gatekeeper_and_reverse_rows(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    assert sc._reuse_fixtures_ready(corpus) is False
    corpus.fixtures.append(
        {
            "task": "default",
            "system": "You decide if user input is a database query request or not.",
            "user": "How many rentals happened in 2026?",
            "output_text": "{}",
        },
    )
    assert sc._reuse_fixtures_ready(corpus) is False
    corpus.fixtures.append(
        {
            "task": "default",
            "system": sc.PARAM_EXTRACTION_SYSTEM_MARKER,
            "user": '{"question":"how many rentals happened in 2025?"}',
            "output_text": "{}",
        },
    )
    assert sc._reuse_fixtures_ready(corpus) is False


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


def test_committed_keys_seeded_from_loaded_corpus(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        '{"version": 1, "fixtures": [{"task": "intent", "system": "s", "user": "u", "output_text": "o"}]}\n',
        encoding="utf-8",
    )
    corpus = sc.FixtureCorpus(fixtures_path)
    assert ("intent", "s", "u") in corpus.committed_keys


def test_commit_slot_adds_new_keys_and_is_idempotent(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = sc.FixtureCorpus(fixtures_path)
    corpus.start_slot()
    corpus.record(task="intent", system="s", user_key="u", output_text="o")
    corpus.commit_slot()
    assert len(corpus.fixtures) == 1
    assert ("intent", "s", "u") in corpus.committed_keys
    corpus.start_slot()
    corpus.record(task="intent", system="s", user_key="u", output_text="o")
    corpus.commit_slot()
    assert len(corpus.fixtures) == 1


def test_commit_slot_frozen_shared_subcall_keeps_frozen_output(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        '{"version": 1, "fixtures": [{"task": "intent", "system": "s", "user": "u", "output_text": "frozen"}]}\n',
        encoding="utf-8",
    )
    corpus = sc.FixtureCorpus(fixtures_path)
    corpus.start_slot()
    corpus.record(task="intent", system="s", user_key="u", output_text="different")
    corpus.commit_slot()
    assert len(corpus.fixtures) == 1
    assert corpus.fixtures[0]["output_text"] == "frozen"


def test_commit_slot_allows_same_output_shared_subcall(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        '{"version": 1, "fixtures": [{"task": "default", "system": "s", "user": "shared", "output_text": "o"}]}\n',
        encoding="utf-8",
    )
    corpus = sc.FixtureCorpus(fixtures_path)
    corpus.start_slot()
    corpus.record(task="default", system="s", user_key="shared", output_text="o")
    corpus.record(task="intent", system="s2", user_key="u2", output_text="o2")
    corpus.commit_slot()
    assert len(corpus.fixtures) == 2


def test_snapshot_restore_undoes_tentative_merge(tmp_path: Path) -> None:
    sc = importlib.import_module("sandbox_corpus")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        '{"version": 1, "fixtures": [{"task": "intent", "system": "s", "user": "u", "output_text": "o"}]}\n',
        encoding="utf-8",
    )
    corpus = sc.FixtureCorpus(fixtures_path)
    snap = corpus.snapshot()
    corpus.start_slot()
    corpus.record(task="intent", system="s2", user_key="u2", output_text="o2")
    corpus.commit_slot(freeze=False)
    assert len(corpus.fixtures) == 2
    assert ("intent", "s2", "u2") not in corpus.committed_keys
    corpus.restore(snap)
    assert len(corpus.fixtures) == 1
    assert ("intent", "s2", "u2") not in corpus.seen


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


def test_intent_system_label_maps_stages() -> None:
    sc = importlib.import_module("sandbox_corpus")
    assert sc._intent_system_label(sc.INTENT_INTERPRET_SYSTEM) == "interpret"
    assert sc._intent_system_label(sc.INTENT_GROUND_SYSTEM) == "ground"
    assert sc._intent_system_label(sc.INTENT_COMPOSE_SYSTEM) == "compose"
    assert sc._intent_system_label("some other system prompt") == "default"


def test_append_replay_trace_flags_mock_miss(tmp_path: Path, monkeypatch) -> None:
    sc = importlib.import_module("sandbox_corpus")
    trace_path = tmp_path / "replay_trace.txt"
    monkeypatch.setattr(sc, "SANDBOX_REPLAY_TRACE_PATH", trace_path)
    sc._reset_replay_trace()
    live_rows = [{"task": "intent", "system_id": "interpret", "user_sha": "abc", "user_head": "u", "result": "live:1"}]
    mock_rows = [
        {"task": "intent", "system_id": "interpret", "user_sha": "zzz", "user_head": "u", "result": "MISS:not found"},
    ]
    sc._append_replay_trace(
        slot_id="questions:q",
        label="q",
        tier="questions",
        attempt=1,
        outcome="DETERMINISM-DIVERGENCE",
        live_rows=live_rows,
        mock_rows=mock_rows,
        detail="boom",
    )
    text = trace_path.read_text(encoding="utf-8")
    assert "DETERMINISM-DIVERGENCE" in text
    assert "LIVE record trace" in text
    assert "MOCK replay trace" in text
    assert "mock MISS" in text
    assert "never produced" in text
