"""Profiling value sample fields on ColumnMetadata."""

from __future__ import annotations

from aetherdialect._contracts_schema import ColumnMetadata


def test_column_metadata_from_dict_reads_sample_fields() -> None:
    """``frequent_values`` and ``value_overlap_sample`` round-trip through ``from_dict``."""
    col = ColumnMetadata.from_dict(
        {
            "name": "status",
            "data_type": "varchar",
            "frequent_values": ["active", "inactive"],
            "value_overlap_sample": ["alpha", "beta", "gamma"],
        }
    )
    assert col.frequent_values == ["active", "inactive"]
    assert col.value_overlap_sample == ["alpha", "beta", "gamma"]


def test_column_metadata_to_dict_writes_sample_fields() -> None:
    """Serialized column metadata includes ``frequent_values`` and ``value_overlap_sample``."""
    col = ColumnMetadata(
        name="status",
        data_type="varchar",
        frequent_values=["yes", "no"],
        value_overlap_sample=["a", "b"],
    )
    payload = col.to_dict()
    assert payload["frequent_values"] == ["yes", "no"]
    assert payload["value_overlap_sample"] == ["a", "b"]
