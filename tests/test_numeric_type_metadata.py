"""Tests for numeric precision, scale and exactness on column metadata."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import DatabaseFeatureCapability, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_manifest import federation_ir_capability_reason
from aetherdialect._intent_normalize import coerce_element
from aetherdialect._schema_reflect import tables_meta_to_schema_graph
from aetherdialect._utils import column_metadata_requires_exact_comparison


@pytest.mark.fast
def test_precision_scale_and_exactness_survive_reflection() -> None:
    """Column metadata carries precision, scale and exactness through graph build and serialization."""
    meta = {
        "amounts": {
            "column_names_original": ["exact_amt", "approx_amt", "big_cnt", "small_cnt"],
            "column_types": ["DECIMAL(19,4)", "DOUBLE PRECISION", "BIGINT", "INTEGER"],
            "primary_keys": ["exact_amt"],
            "foreign_keys": [],
        },
    }
    sg = tables_meta_to_schema_graph(meta)
    cols = sg.tables["amounts"].columns

    exact = cols["exact_amt"]
    assert exact.numeric_precision == 19
    assert exact.numeric_scale == 4
    assert exact.is_exact_numeric is True
    assert exact.value_type == "number"

    approx = cols["approx_amt"]
    assert approx.is_exact_numeric is False
    assert approx.value_type == "number"

    big = cols["big_cnt"]
    assert big.is_exact_numeric is True
    assert big.value_type == "integer"

    small = cols["small_cnt"]
    assert small.is_exact_numeric is True
    assert small.value_type == "integer"

    for col in (exact, approx, big, small):
        restored = ColumnMetadata.from_dict(col.to_dict())
        assert restored.numeric_precision == col.numeric_precision
        assert restored.numeric_scale == col.numeric_scale
        assert restored.is_exact_numeric == col.is_exact_numeric


@pytest.mark.fast
def test_unsigned_flag_reflected_and_used() -> None:
    """Unsigned columns expose is_unsigned, force exact comparison near max, and gate federation by metadata."""
    meta = {
        "counts": {
            "column_names_original": ["big_id", "plain_id"],
            "column_types": ["BIGINT UNSIGNED", "BIGINT"],
            "primary_keys": ["big_id"],
            "foreign_keys": [],
        },
    }
    sg = tables_meta_to_schema_graph(meta)
    unsigned_col = sg.tables["counts"].columns["big_id"]
    signed_col = sg.tables["counts"].columns["plain_id"]
    assert unsigned_col.is_unsigned is True
    assert signed_col.is_unsigned is False

    restored = ColumnMetadata.from_dict(unsigned_col.to_dict())
    assert restored.is_unsigned is True

    near_max = ColumnMetadata(
        name="big_id",
        data_type="BIGINT UNSIGNED",
        is_unsigned=True,
        max_val="18446744073709551600",
    )
    assert column_metadata_requires_exact_comparison(near_max) is True

    coerced = coerce_element("18446744073709551615", "bigint unsigned", col_meta=near_max)
    assert isinstance(coerced, int)
    assert coerced == 18446744073709551615

    stealth_unsigned = ColumnMetadata(
        name="id",
        data_type="bigint",
        is_unsigned=True,
        sensitivity="none",
    )
    schema = SchemaGraph(
        tables={
            "t": TableMetadata(
                name="t",
                columns={"id": stealth_unsigned},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    cap = DatabaseFeatureCapability(
        table_count=1,
        fk_edge_count=0,
        has_numeric_measures=True,
        has_date_columns=False,
        has_array_columns=False,
        has_categorical_columns=False,
        max_tables_on_any_join_path=1,
        max_fk_chain_depth=0,
        has_self_referential_fk=False,
        tables_supporting_self_join=frozenset(),
        has_window_capable_table_sets=False,
        aggregatable_columns_by_table={},
        date_columns_by_table={},
        array_columns_by_table={},
        supports_unsigned_semantics=False,
    )
    reason = federation_ir_capability_reason(intent, cap, schema=schema)
    assert reason is not None
    assert "unsigned" in reason.lower()
