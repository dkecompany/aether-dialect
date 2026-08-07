"""Tests for ephemeral session scope intersection with aetherspace scope."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError, SpaceContext
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_ephemeral_narrows_table_allow_list() -> None:
    base_tables = frozenset({"a", "b", "c"})
    ephemeral = SpaceContext(tables=frozenset({"a", "b"}))
    tables, columns, deny_objects, deny_columns = MainExecutionOps.intersect_space_scope(
        base_tables,
        frozenset(),
        frozenset(),
        frozenset(),
        ephemeral,
    )
    assert tables == frozenset({"a", "b"})
    assert columns == frozenset()
    assert deny_objects == frozenset()
    assert deny_columns == frozenset()


@pytest.mark.fast
def test_ephemeral_cannot_widen_beyond_base() -> None:
    base_tables = frozenset({"a", "b"})
    ephemeral = SpaceContext(tables=frozenset({"a", "b", "c"}))
    tables, columns, deny_objects, deny_columns = MainExecutionOps.intersect_space_scope(
        base_tables,
        frozenset(),
        frozenset(),
        frozenset(),
        ephemeral,
    )
    assert tables == frozenset({"a", "b"})
    assert columns == frozenset()
    assert deny_objects == frozenset()
    assert deny_columns == frozenset()


@pytest.mark.fast
def test_deny_lists_union() -> None:
    base_deny_objects = frozenset({"x"})
    base_deny_columns = frozenset({"t.secret"})
    ephemeral = SpaceContext(
        deny_objects=frozenset({"y"}),
        deny_columns=frozenset({"u.token"}),
    )
    tables, columns, deny_objects, deny_columns = MainExecutionOps.intersect_space_scope(
        frozenset({"a", "b"}),
        frozenset(),
        base_deny_objects,
        base_deny_columns,
        ephemeral,
    )
    assert tables == frozenset({"a", "b"})
    assert columns == frozenset()
    assert deny_objects == frozenset({"x", "y"})
    assert deny_columns == frozenset({"t.secret", "u.token"})


@pytest.mark.fast
def test_empty_ephemeral_is_identity() -> None:
    base_tables = frozenset({"a", "b"})
    base_columns = frozenset({"a.id"})
    base_deny_objects = frozenset({"x"})
    base_deny_columns = frozenset({"a.secret"})

    for ephemeral in (None, SpaceContext()):
        tables, columns, deny_objects, deny_columns = MainExecutionOps.intersect_space_scope(
            base_tables,
            base_columns,
            base_deny_objects,
            base_deny_columns,
            ephemeral,
        )
        assert tables == base_tables
        assert columns == base_columns
        assert deny_objects == base_deny_objects
        assert deny_columns == base_deny_columns


@pytest.mark.fast
def test_column_allow_intersection() -> None:
    base_columns = frozenset({"a.id", "b.id"})
    ephemeral = SpaceContext(columns=frozenset({"a.id"}))
    tables, columns, deny_objects, deny_columns = MainExecutionOps.intersect_space_scope(
        frozenset(),
        base_columns,
        frozenset(),
        frozenset(),
        ephemeral,
    )
    assert tables == frozenset()
    assert columns == frozenset({"a.id"})
    assert deny_objects == frozenset()
    assert deny_columns == frozenset()


@pytest.mark.fast
def test_allow_deny_overlap_raises() -> None:
    ephemeral = SpaceContext(tables=frozenset({"a"}))
    with pytest.raises(ConfigError, match="tables and deny_objects overlap"):
        MainExecutionOps.intersect_space_scope(
            frozenset({"a", "b"}),
            frozenset(),
            frozenset({"a"}),
            frozenset(),
            ephemeral,
        )
