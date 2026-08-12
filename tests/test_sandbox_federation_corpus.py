"""Tests for sandbox federation corpus routing, slot ids, and seed export helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _sandbox_corpus():
    return importlib.import_module("sandbox_corpus")


@pytest.mark.fast
def test_slot_id_carries_role_mode_and_question_only() -> None:
    sc = _sandbox_corpus()
    slot = sc.RecordingSlot(tier="questions", label="How many books do we have?", preset="owner_writer")
    assert sc.slot_id_for(slot) == "owner:writer:How many books do we have?"
    assert "federation" not in sc.slot_id_for(slot)
    assert "questions" not in sc.slot_id_for(slot)


@pytest.mark.fast
def test_recipe_routes_federation_from_mechanism_without_tier() -> None:
    sc = _sandbox_corpus()
    slot = sc.RecordingSlot(
        tier="questions",
        label="How many rentals are linked to film titles?",
        preset="owner_writer",
    )
    scenario = {
        "mechanism": "federation_cross_source_join",
        "recipe": "federation",
    }
    assert sc._recipe_for_slot(slot, scenario) == "federation"
    assert sc._construction_for_slot(slot).surface == "federation"


@pytest.mark.fast
def test_owner_question_mock_verify_targets_include_federation_recipe() -> None:
    sc = _sandbox_corpus()
    slot = sc.RecordingSlot(
        tier="questions",
        label="How many books do we have?",
        preset="owner_writer",
    )
    presets = [target[0].surface for target in sc._mock_verify_targets_for_slot(slot)]
    assert presets == ["single", "single"]


@pytest.mark.fast
def test_smoke_questions_include_cross_source_practice_question() -> None:
    sc = _sandbox_corpus()
    smoke = sc.smoke_questions()
    assert sc.FEDERATION_SMOKE_QUESTION in smoke["questions"]


@pytest.mark.fast
def test_crm_staff_column_projection_limits_exported_columns() -> None:
    sc = _sandbox_corpus()
    projections = sc.federation_member_column_projections("crm")
    assert set(projections["staff"]) == {"staff_id", "first_name", "last_name", "store_id"}


@pytest.mark.fast
def test_partition_export_helpers_exist_without_running_corpus_build() -> None:
    src = importlib.import_module("source_rental_shop")
    sc = _sandbox_corpus()
    assert callable(src.export_sandbox_federation_partition_schemas)
    assert callable(src.export_sandbox_federation_partition_data_dirs)
    assert callable(src.export_federation_member_data_dirs_from_existing_csvs)
    assert callable(sc.assemble_staging)
    assert callable(sc.run_staging_pack_assertions)


@pytest.mark.fast
def test_partition_seed_export_strips_cross_partition_foreign_keys() -> None:
    sc = _sandbox_corpus()
    ddl = (
        "CREATE TABLE rental (rental_id INTEGER PRIMARY KEY, inventory_id INTEGER REFERENCES inventory(inventory_id));"
    )
    sanitized = sc._strip_disallowed_create_table_foreign_keys(
        ddl,
        "rental",
        frozenset({"rental"}),
    )
    assert "REFERENCES" not in sanitized


@pytest.mark.fast
def test_default_federation_member_specs_list_four_partition_schemas(tmp_path: Path) -> None:
    from aetherdialect._constants_runtime import SANDBOX_BUNDLED_MEMBER_SCHEMAS
    from aetherdialect._sandbox import Sandbox

    for _member_name, schema_name in SANDBOX_BUNDLED_MEMBER_SCHEMAS:
        (tmp_path / schema_name).write_text("-- stub", encoding="utf-8")
    specs = Sandbox._default_federation_member_specs(tmp_path)
    assert [name for name, _path in specs] == [name for name, _schema in SANDBOX_BUNDLED_MEMBER_SCHEMAS]


@pytest.mark.fast
def test_member_space_questions_are_subsets_of_practice_corpus() -> None:
    from aetherdialect import Sandbox
    from aetherdialect._constants_runtime import SANDBOX_MEMBER_SPACE_QUESTIONS

    sc = _sandbox_corpus()
    practice = set(sc.parse_questions_file(_REPO / "scripts" / "data" / "sandbox_questions.txt")["questions"])
    mapping = Sandbox._sandbox_member_space_questions()
    assert isinstance(mapping, dict)
    assert set(mapping) == set(SANDBOX_MEMBER_SPACE_QUESTIONS)
    for space_name, questions in mapping.items():
        assert questions
        assert set(questions) <= practice
        assert Sandbox._sandbox_member_space_questions(space_name) == questions


@pytest.mark.fast
def test_space_packing_slot_is_single_member_spaces_row() -> None:
    sc = _sandbox_corpus()
    questions = sc.parse_questions_file(_REPO / "scripts" / "data" / "sandbox_questions.txt")
    space_slots = [slot for slot in sc.iter_recording_slots(questions) if slot.kind == "space"]
    assert len(space_slots) == 1
    assert space_slots[0].label == "member_spaces"
    assert sc.slot_id_for(space_slots[0]) == "owner:writer:member_spaces"
