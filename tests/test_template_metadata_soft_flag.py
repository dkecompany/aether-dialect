"""Metadata-only schema diffs soft-flag without deleting templates."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import SchemaDiff, TableDiff
from aetherdialect._templates_ops import TemplateOps


@pytest.mark.fast
def test_nullability_only_keeps_templates() -> None:
    diff = SchemaDiff(
        per_table={
            "orders": TableDiff(
                nullability_changed_columns=frozenset({"amount"}),
            )
        }
    )
    tables, cols = TemplateOps._surgical_invalidation_targets(diff)
    soft_tables, soft_cols = TemplateOps._surgical_soft_flag_targets(diff)
    assert not tables
    assert not cols
    assert ("orders", "amount") in soft_cols
    assert not soft_tables


@pytest.mark.fast
def test_drop_column_invalidates() -> None:
    diff = SchemaDiff(
        per_table={
            "orders": TableDiff(
                dropped_columns=frozenset({"amount"}),
            )
        }
    )
    tables, cols = TemplateOps._surgical_invalidation_targets(diff)
    assert ("orders", "amount") in cols
    assert not tables
