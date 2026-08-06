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
    assert sc._mock_preset_for_slot(slot)[0] == "federation"


@pytest.mark.fast
def test_owner_question_mock_verify_targets_include_federation_recipe() -> None:
    sc = _sandbox_corpus()
    slot = sc.RecordingSlot(
        tier="questions",
        label="How many books do we have?",
        preset="owner_writer",
    )
    presets = [target[0] for target in sc._mock_verify_targets_for_slot(slot)]
    assert presets == ["owner_writer", "consumer_reader", "federation"]


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
    sc = _sandbox_corpus()
    assert callable(sc.export_federation_partition_seeds)
    assert callable(sc.assemble_staging)


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
def test_default_federation_member_specs_list_four_partition_seeds(tmp_path: Path) -> None:
    from aetherdialect._constants import SANDBOX_BUNDLED_MEMBER_SEEDS
    from aetherdialect._sandbox import Sandbox

    for _member_name, seed_name in SANDBOX_BUNDLED_MEMBER_SEEDS:
        (tmp_path / seed_name).write_text("-- stub", encoding="utf-8")
    specs = Sandbox._default_federation_member_specs(tmp_path)
    assert [name for name, _path in specs] == [name for name, _seed in SANDBOX_BUNDLED_MEMBER_SEEDS]
