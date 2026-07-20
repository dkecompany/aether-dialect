"""Tests for profiling value sample consolidation and K8 cache backward compatibility."""

from __future__ import annotations

from aetherdialect._contracts_schema import ColumnMetadata


def test_column_metadata_from_dict_legacy_top_k_and_semantic_keys() -> None:
    """Legacy ``top_k_values`` and ``semantic_distinct_values`` map to consolidated fields."""
    col = ColumnMetadata.from_dict(
        {
            "name": "status",
            "data_type": "varchar",
            "top_k_values": ["active", "inactive"],
            "semantic_distinct_values": ["alpha", "beta", "gamma"],
        }
    )
    assert col.frequent_values == ["active", "inactive"]
    assert col.value_overlap_sample == ["alpha", "beta", "gamma"]
    assert not hasattr(col, "top_k_values") or col.to_dict().get("top_k_values") is None


def test_column_metadata_to_dict_writes_only_new_keys() -> None:
    """Serialized column metadata uses ``frequent_values`` and ``value_overlap_sample`` only."""
    col = ColumnMetadata(
        name="status",
        data_type="varchar",
        frequent_values=["yes", "no"],
        value_overlap_sample=["a", "b"],
    )
    payload = col.to_dict()
    assert payload["frequent_values"] == ["yes", "no"]
    assert payload["value_overlap_sample"] == ["a", "b"]
    assert "top_k_values" not in payload
    assert "semantic_distinct_values" not in payload
