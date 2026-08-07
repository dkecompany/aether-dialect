"""Tests for MySQL column character set and collation reflection."""

from __future__ import annotations

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_COLUMN_CHARSET_MISMATCH,
    MYSQL_CONNECTION_CHARSET,
)
from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._core_utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._dialect import DialectRegistry
from aetherdialect._schema_build import tables_meta_to_schema_graph


@pytest.mark.fast
def test_column_charset_recorded_and_mismatch_reported() -> None:
    """Reflection records per-column charset/collation and warns on charset mismatch."""
    meta = {
        "labels": {
            "column_names_original": ["code", "legacy_label"],
            "column_types": ["varchar(50)", "varchar(50)"],
            "column_character_sets": ["utf8mb4", "latin1"],
            "column_collations": ["utf8mb4_bin", "latin1_swedish_ci"],
            "primary_keys": ["code"],
            "foreign_keys": [],
        },
    }
    token = set_diagnostic_collector([])
    try:
        sg = tables_meta_to_schema_graph(meta, engine="mysql")
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    utf8_col = sg.tables["labels"].columns["code"]
    latin_col = sg.tables["labels"].columns["legacy_label"]

    assert utf8_col.character_set == "utf8mb4"
    assert utf8_col.collation == "utf8mb4_bin"
    assert latin_col.character_set == "latin1"
    assert latin_col.collation == "latin1_swedish_ci"

    mismatch_diags = [d for d in diags if d.code == DIAGNOSTIC_CODE_COLUMN_CHARSET_MISMATCH]
    assert len(mismatch_diags) == 1
    assert "legacy_label" in mismatch_diags[0].message
    assert MYSQL_CONNECTION_CHARSET == "utf8mb4"

    dialect = DialectRegistry.get_class("mysql").__new__(DialectRegistry.get_class("mysql"))
    assert dialect.column_is_case_insensitive_collation(utf8_col) is False
    assert dialect.column_is_case_insensitive_collation(latin_col) is True
    assert utf8_col.is_case_insensitive_collation is False
    assert latin_col.is_case_insensitive_collation is True
    assert utf8_col.overlap_comparison == "exact"
    assert latin_col.overlap_comparison == "case_folded"

    restored = ColumnMetadata.from_dict(utf8_col.to_dict())
    assert restored.character_set == "utf8mb4"
    assert restored.collation == "utf8mb4_bin"
