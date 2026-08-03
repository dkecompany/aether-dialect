"""Profiling failure markers, query bounds, and catalog description wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import (
    DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED,
    DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED,
    DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED,
)
from aetherdialect._contracts_base import DescriptionOwner
from aetherdialect._contracts_schema import ColumnMetadata, TableMetadata
from aetherdialect._core_utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._schema_build import tables_meta_to_schema_graph
from aetherdialect._schema_catalog import (
    _build_frequent_values_sql,
    _build_minmax_sql,
    _build_mode_sql,
    _maybe_set_profile_statement_timeout,
    _profile_column,
    _profile_composite_descriptive,
    _profile_table,
    apply_catalog_descriptions_from_tables_meta,
)
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


def test_composite_descriptive_failure_sets_profile_failed_on_name_columns() -> None:
    table = _name_pair_table()
    dialect = MagicMock()
    dialect.qualified_table_ref.return_value = '"contacts"'
    dialect.quote_identifier.side_effect = lambda name: f'"{name}"'
    dialect.profiling_stats_sample_suffix.return_value = ""
    dialect.profiling_stats_use_subquery_when_sampling.return_value = False

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
    assert table.columns["first_name"].profile_failed is True
    assert table.columns["last_name"].profile_failed is True


def test_profile_table_clone_failure_sets_profile_failed_on_table() -> None:
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


def test_column_profile_failure_marks_column_and_continues_siblings() -> None:
    table = TableMetadata(
        name="items",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", value_type="integer"),
            "label": ColumnMetadata(name="label", data_type="varchar", sensitivity="none", value_type="string"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    dialect = MagicMock()
    dialect.qualified_table_ref.return_value = '"items"'
    dialect.quote_identifier.side_effect = lambda name: f'"{name}"'
    dialect.profiling_stats_sample_suffix.return_value = ""
    dialect.profiling_stats_use_subquery_when_sampling.return_value = False
    dialect.profile_statement_timeout_sql.return_value = None

    calls: list[str] = []

    class _Conn:
        def execute(self, statement, *_args, **_kwargs):
            sql = str(statement)
            calls.append(sql)
            if "COUNT(*)" in sql and "FROM" in sql and "DISTINCT" not in sql:
                return _ScalarResult(3)
            if '"label"' in sql:
                raise RuntimeError("label probe failed")
            return _ScalarResult(0)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _ScalarResult:
        def __init__(self, value: int) -> None:
            self._value = value

        def fetchone(self):
            return (self._value, self._value, 0)

        def fetchall(self):
            return []

        def scalar(self):
            return self._value

    class _Engine:
        def connect(self):
            return _Conn()

    token = set_diagnostic_collector([])
    try:
        _profile_table(dialect, _Engine(), table)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert table.columns["label"].profile_failed is True
    assert table.columns["id"].profile_failed is False
    assert any(d.code == DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED for d in diags)
    assert any('"id"' in sql for sql in calls)


def test_bounded_profiling_sql_uses_sample_clause() -> None:
    sample = "ORDER BY 1 LIMIT 50000"
    qcol = '"amount"'
    qtbl = '"sales"'
    freq_sql = _build_frequent_values_sql(qcol, qtbl, 10, sample_clause=sample, use_subquery=True)
    assert "LIMIT 50000" in freq_sql
    assert qcol in freq_sql
    minmax_sql = _build_minmax_sql(qcol, qtbl, sample_clause=sample, use_subquery=True)
    assert "LIMIT 50000" in minmax_sql
    mode_sql = _build_mode_sql(qcol, qtbl, sample_clause=sample, use_subquery=True)
    assert "LIMIT 50000" in mode_sql


def test_maybe_set_profile_timeout_uses_dialect_override() -> None:
    executed: list[str] = []

    class _Conn:
        def execute(self, stmt, *_args, **_kwargs):
            executed.append(str(stmt))

    dialect = MagicMock()
    dialect.profile_timeout_ms = 42
    dialect.profile_statement_timeout_sql.return_value = "SET statement_timeout = 42"

    _maybe_set_profile_statement_timeout(_Conn(), dialect)
    assert executed == ["SET statement_timeout = 42"]


def test_tables_meta_table_comment_becomes_catalog_description() -> None:
    meta = {
        "orders": {
            "column_names_original": ["id"],
            "column_types": ["integer"],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "table_comment": "Customer purchase records",
            "column_comments": {"id": "Surrogate primary key"},
        }
    }
    sg = tables_meta_to_schema_graph(meta)
    assert sg.tables["orders"].description == "Customer purchase records"
    assert sg.tables["orders"].description_owner == DescriptionOwner.CATALOG
    assert sg.tables["orders"].columns["id"].description == "Surrogate primary key"
    assert sg.tables["orders"].columns["id"].description_owner == DescriptionOwner.CATALOG


def test_apply_catalog_descriptions_respects_higher_owner() -> None:
    meta = {
        "orders": {
            "column_names_original": ["id"],
            "column_types": ["integer"],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "table_comment": "Catalog text",
        }
    }
    sg = tables_meta_to_schema_graph({k: {**v, "table_comment": None} for k, v in meta.items()})
    sg.tables["orders"].description = "User text"
    sg.tables["orders"].description_owner = DescriptionOwner.USER_OVERRIDE
    apply_catalog_descriptions_from_tables_meta(sg, meta)
    assert sg.tables["orders"].description == "User text"
    assert sg.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE


def test_apply_column_roles_llm_with_notes_uses_notes_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from aetherdialect._schema_catalog import apply_column_roles_llm

    table = TableMetadata(
        name="orders",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", value_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
    )
    sg = __import__("aetherdialect._contracts_schema", fromlist=["SchemaGraph"]).SchemaGraph(
        tables={"orders": table}, join_paths_multi={}
    )

    def _fake_classify(_schema, notes_content=None, *, column_scope=None):
        _ = notes_content, column_scope
        return {
            "orders": (
                "fact",
                "Orders from notes",
                {"id": ("identifier", "Order id from notes", "none")},
            )
        }

    monkeypatch.setattr(
        "aetherdialect._schema_catalog.llm_classify_schema",
        _fake_classify,
    )
    apply_column_roles_llm(sg, notes_content="Domain notes about orders.")
    assert sg.tables["orders"].description_owner == DescriptionOwner.NOTES
    assert sg.tables["orders"].columns["id"].description_owner == DescriptionOwner.NOTES
