"""T5: additive-only schema diffs must not default skeleton action to destructive."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import MIGRATION_MAP_ACTION_DESTRUCTIVE, MIGRATION_MAP_ACTION_REMAP
from aetherdialect._contracts_base import MigrationTier
from aetherdialect._contracts_schema import SchemaDiff, TableDiff
from aetherdialect._templates import export_schema_migration_map_skeleton


@pytest.mark.fast
def test_destructive_tier_additive_column_defaults_to_remap(tmp_path) -> None:
    diff = SchemaDiff(per_table={"orders": TableDiff(added_columns=("note",))})
    path = export_schema_migration_map_skeleton(
        tmp_path,
        tier=MigrationTier.DESTRUCTIVE,
        schema_diff=diff,
        rename_plan=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["action"] == MIGRATION_MAP_ACTION_REMAP
    assert payload["action"] != MIGRATION_MAP_ACTION_DESTRUCTIVE


@pytest.mark.fast
def test_destructive_tier_additive_table_defaults_to_remap(tmp_path) -> None:
    diff = SchemaDiff(added_tables=("customers",))
    path = export_schema_migration_map_skeleton(
        tmp_path,
        tier=MigrationTier.DESTRUCTIVE,
        schema_diff=diff,
        rename_plan=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["action"] == MIGRATION_MAP_ACTION_REMAP


@pytest.mark.fast
def test_destructive_tier_drop_still_defaults_to_destructive(tmp_path) -> None:
    diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})
    path = export_schema_migration_map_skeleton(
        tmp_path,
        tier=MigrationTier.DESTRUCTIVE,
        schema_diff=diff,
        rename_plan=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["action"] == MIGRATION_MAP_ACTION_DESTRUCTIVE
