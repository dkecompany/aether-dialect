"""Profiling failures surface diagnostics instead of silent no-op behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED,
    DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED,
)
from aetherdialect._contracts_schema import ColumnMetadata, TableMetadata
from aetherdialect._core_utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._schema_catalog import _profile_composite_descriptive
from aetherdialect._schema_graph import _profile_table_clone


def _name_pair_table() -> TableMetadata:
    return TableMetadata(
        name="contacts",
        columns={
            "first_name": ColumnMetadata(
                name="first_name",
                data_type="varchar",
                sensitivity="none",
                value_type="string",
                distinct_count=2,
                distinct_ratio=1.0,
                frequent_values=["Ada", "Bob"],
            ),
            "last_name": ColumnMetadata(
                name="last_name",
                data_type="varchar",
                sensitivity="none",
                value_type="string",
                distinct_count=2,
                distinct_ratio=1.0,
                frequent_values=["Lee", "Ng"],
            ),
        },
        primary_key=[],
        foreign_keys=[],
        row_count=2,
        composite_descriptive_ratios={("first_name", "last_name"): 0.5},
    )


def test_composite_descriptive_failure_emits_diagnostic_and_clears_name_column_profiles() -> None:
    table = _name_pair_table()
    dialect = MagicMock()
    dialect.qualified_table_ref.return_value = '"contacts"'
    dialect.quote_identifier.side_effect = lambda name: f'"{name}"'

    class _Conn:
        def execute(self, _statement, *_args, **_kwargs):
            raise RuntimeError("probe unavailable")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _Engine:
        def connect(self):
            return _Conn()

    token = set_diagnostic_collector([])
    try:
        _profile_composite_descriptive(dialect, _Engine(), table)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED
    assert diags[0].details == (("table", "contacts"),)
    assert table.composite_descriptive_ratios == {}
    assert table.columns["first_name"].profile_failed is True
    assert table.columns["last_name"].profile_failed is True
    assert table.columns["first_name"].frequent_values == []
    assert table.columns["last_name"].value_overlap_sample == []


def test_profile_table_clone_failure_emits_diagnostic_and_returns_none() -> None:
    table = TableMetadata(
        name="orders",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
    )
    dialect = MagicMock()
    dialect.profile_schema.side_effect = RuntimeError("clone probe failed")

    token = set_diagnostic_collector([])
    try:
        clone = _profile_table_clone(dialect, table, notes_content=None)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert clone is None
    assert table.profile_failed is True
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED
    assert diags[0].details == (("table", "orders"),)
