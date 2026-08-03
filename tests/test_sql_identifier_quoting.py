"""SQL identifiers are quoted before they reach rendered SQL."""

from __future__ import annotations

import pytest

from aetherdialect._constants import SELF_JOIN_CTE_NAME_PREFIX, anti_join_presence_column
from aetherdialect._contracts_base import JoinInjectionFailedError, SchemaInvariantError
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import get_dialect
from aetherdialect._federation import _apply_coordinator_probe_joins, _quote_ident
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._intent_resolve import encode_inline_self_join_as_cte, normalize_cte_names
from aetherdialect._sql_gen import _join_edges_from_signature, inject_join_into_deterministic_sql
from tests.test_join_kind_preservation import catalog_edge_kinds_for_signatures


def _reserved_order_schema() -> SchemaGraph:
    edge = {
        "src_table": "line",
        "src_cols": ["order_id"],
        "dst_table": "order",
        "dst_cols": ["id"],
    }
    return SchemaGraph(
        tables={
            "order": TableMetadata(
                name="order",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "line": TableMetadata(
                name="line",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "order_id": ColumnMetadata(name="order_id", data_type="integer", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={"line": {"order": [[edge]]}},
        effective_structural_hash="h",
    )


def test_reserved_word_table_in_anti_join_predicate_is_quoted() -> None:
    det = "SELECT line.id\nFROM line"
    sig = [["line.order_id->order.id"]]
    dialect = get_dialect("duckdb")
    out = inject_join_into_deterministic_sql(
        det,
        sig,
        schema=_reserved_order_schema(),
        edge_kinds_ordered=catalog_edge_kinds_for_signatures(sig),
        dialect=dialect,
        cte_emissions={"order": "anti_join"},
    )
    marker = anti_join_presence_column("order")
    quoted_tbl = dialect.quote_table_column("order", marker)
    assert quoted_tbl in out
    assert f"{quoted_tbl} IS NULL" in out


def test_federation_quote_ident_quotes_sql_reserved_words() -> None:
    assert _quote_ident("order") == '"order"'
    assert _quote_ident("group") == '"group"'
    assert _quote_ident("plain_table") == '"plain_table"'


def test_coordinator_probe_join_quotes_reserved_join_keys() -> None:
    probe = RuntimeCteStep(
        cte_name="probe_a",
        tables=["line"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("line.order"))],
        output_columns=["order"],
        emission="semi_join",
    )
    sql = _apply_coordinator_probe_joins(
        'SELECT 1 AS "order"',
        [probe],
        {"a": "src_a"},
        {"line": "a"},
    )
    assert 'drv."order" = probe_a."order"' in sql


def test_join_signature_with_missing_column_is_rejected_at_parse() -> None:
    schema = _reserved_order_schema()
    with pytest.raises(JoinInjectionFailedError, match="missing column"):
        _join_edges_from_signature(
            ["line.order_id->order.missing_col"],
            ["catalog_fk"],
            "line",
            schema,
        )


def test_self_join_cte_name_collision_raises() -> None:
    schema = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            f"{SELF_JOIN_CTE_NAME_PREFIX}orders": TableMetadata(
                name=f"{SELF_JOIN_CTE_NAME_PREFIX}orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )
    intent = RuntimeIntent(
        tables=["orders", "orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(SchemaInvariantError, match="collides with physical table"):
        encode_inline_self_join_as_cte(intent, schema)


def test_normalized_cte_name_collision_with_physical_table_raises() -> None:
    cte = RuntimeCteStep(
        cte_name="rollup",
        tables=["cte1"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.id"))],
        output_columns=["id"],
    )
    intent = RuntimeIntent(
        tables=["cte1", "rollup"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    with pytest.raises(SchemaInvariantError, match="normalized CTE name 'cte1'"):
        normalize_cte_names(intent)
