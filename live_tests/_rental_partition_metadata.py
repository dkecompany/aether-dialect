"""Synthetic partition-column tagging for rental_shop DuckDB/SQLite test fixtures."""

from __future__ import annotations

from aetherdialect._contracts_schema import SchemaGraph


def apply_synthetic_rental_partition_metadata(sg: SchemaGraph) -> None:
    """Tag local DuckDB/SQLite rental_shop graphs with synthetic partition columns for pruning tests."""
    rental = sg.tables.get("rental")
    if rental is None:
        return
    if "rental_date" in rental.columns and not rental.partition_columns:
        rental.partition_columns = ["rental_date"]
