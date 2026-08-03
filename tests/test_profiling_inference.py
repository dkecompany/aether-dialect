"""Profiling inference guards for thin FK samples and table iteration order."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import InferenceTag
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect, DuckDBDialect
from aetherdialect._schema_catalog import (
    _profile_column,
    profile_schema,
    profile_schema_spark,
    profile_schema_sql_connector,
)
from aetherdialect._schema_graph import (
    _infer_missing_fks_composite,
    fk_overlap_validates,
    infer_missing_fks,
)


def _col(name: str, sample: list[str], *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="varchar",
        value_type="string",
        is_primary_key=pk,
        value_overlap_sample=list(sample),
    )


def _table(name: str, *, columns: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    cols = columns or {"id": _col("id", ["1"], pk=True)}
    return TableMetadata(
        name=name,
        columns=cols,
        primary_key=["id"],
        foreign_keys=[],
    )


def _schema(*table_names: str) -> SchemaGraph:
    return SchemaGraph(
        tables={name: _table(name) for name in table_names},
        join_paths_multi={},
        created_at="",
    )


_SHARED_COMPOSITE_SAMPLE = ["1", "2", "3", "4", "5"]
_DISJOINT_CHILD_SAMPLE = ["9", "8", "7", "6", "5"]
_DISJOINT_PARENT_SAMPLE = ["1", "2", "3", "4", "0"]


def _composite_overlap_tables(
    *,
    child_order_sample: list[str],
    child_line_sample: list[str],
    parent_order_sample: list[str],
    parent_line_sample: list[str],
) -> dict[str, TableMetadata]:
    return {
        "order_lines": TableMetadata(
            name="order_lines",
            columns={
                "order_id": _col("order_id", parent_order_sample, pk=True),
                "line_no": _col("line_no", parent_line_sample, pk=True),
            },
            primary_key=["order_id", "line_no"],
            foreign_keys=[],
        ),
        "shipments": TableMetadata(
            name="shipments",
            columns={
                "shipment_id": _col("shipment_id", ["s1"], pk=True),
                "order_id": _col("order_id", child_order_sample),
                "line_no": _col("line_no", child_line_sample),
            },
            primary_key=["shipment_id"],
            foreign_keys=[],
        ),
    }


@pytest.mark.fast
def test_fk_overlap_rejects_thin_samples_below_minimum() -> None:
    src = _col("customer_id", ["1", "2", "3", "4"])
    dst = _col("id", ["1", "2", "3", "4", "5"], pk=True)
    assert len(src.value_overlap_sample) < PolicyConfig.FK_INFER_OVERLAP_MIN_SAMPLE
    assert fk_overlap_validates(src, dst) is False


@pytest.mark.fast
def test_composite_fk_inference_rejects_disjoint_overlap_samples() -> None:
    tables = _composite_overlap_tables(
        child_order_sample=_DISJOINT_CHILD_SAMPLE,
        child_line_sample=_SHARED_COMPOSITE_SAMPLE,
        parent_order_sample=_DISJOINT_PARENT_SAMPLE,
        parent_line_sample=_SHARED_COMPOSITE_SAMPLE,
    )
    edges = infer_missing_fks(tables)
    composite = [edge for edge in edges if edge.inference_tag == InferenceTag.COMPOSITE]
    assert composite == []


@pytest.mark.fast
def test_infer_missing_fks_composite_applies_overlap_gate() -> None:
    tables = _composite_overlap_tables(
        child_order_sample=_DISJOINT_CHILD_SAMPLE,
        child_line_sample=_SHARED_COMPOSITE_SAMPLE,
        parent_order_sample=_DISJOINT_PARENT_SAMPLE,
        parent_line_sample=_SHARED_COMPOSITE_SAMPLE,
    )
    inferred = _infer_missing_fks_composite(tables, {name.lower(): name for name in tables}, existing=[])
    assert inferred == []


@pytest.mark.fast
def test_profile_column_zero_row_count_stays_zero_for_ratios() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: str) -> MagicMock:
        stmt = str(sql)
        result = MagicMock()
        if "COUNT(*)" in stmt and "COUNT(DISTINCT" in stmt:
            result.fetchone.return_value = (0, 0, 0)
        elif "MAX(freq)" in stmt:
            result.fetchone.return_value = (3,)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    col = ColumnMetadata(name="status", data_type="varchar", value_type="string")
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    _profile_column(dialect, engine, col, "empty_table", row_count=0)
    assert col.distinct_ratio == 0.0
    assert col.null_ratio == 0.0
    assert col.mode_frequency_ratio == 0.0


@pytest.mark.fast
def test_profile_schema_visits_tables_in_sorted_order() -> None:
    schema = _schema("zebra", "Alpha", "middle")
    engine = MagicMock()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    visited: list[str] = []

    with patch("aetherdialect._schema_catalog._profile_table", side_effect=lambda _d, _e, table, **_k: visited.append(table.name)):
        profile_schema(engine, schema, dialect)
    assert visited == ["Alpha", "middle", "zebra"]


@pytest.mark.fast
def test_profile_schema_spark_visits_tables_in_sorted_order() -> None:
    schema = _schema("zebra", "Alpha", "middle")
    spark = MagicMock()
    dialect = DatabricksDialect.__new__(DatabricksDialect)
    visited: list[str] = []

    with patch(
        "aetherdialect._schema_catalog._profile_table_spark",
        side_effect=lambda _spark, _cat, _sch, table, **_k: visited.append(table.name),
    ):
        profile_schema_spark(spark, "cat", "sch", schema, dialect=dialect)
    assert visited == ["Alpha", "middle", "zebra"]


@pytest.mark.fast
def test_profile_schema_sql_connector_visits_tables_in_sorted_order() -> None:
    schema = _schema("zebra", "Alpha", "middle")
    connection = MagicMock()
    dialect = DatabricksDialect.__new__(DatabricksDialect)
    visited: list[str] = []

    with patch(
        "aetherdialect._schema_catalog._profile_table_sql_connector",
        side_effect=lambda _conn, _cat, _sch, table, **_k: visited.append(table.name),
    ):
        profile_schema_sql_connector(connection, "cat", "sch", schema, dialect=dialect)
    assert visited == ["Alpha", "middle", "zebra"]
