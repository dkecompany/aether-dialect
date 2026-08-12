"""Single-engine schema graphs stamp dialect capability flags without federation membership."""

import pytest

from aetherdialect._contracts_base import EngineContext, MulGroup, NormalizedExpr
from aetherdialect._contracts_core import SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_sqlglot_engines import MySQLDialect
from aetherdialect._schema_graph import assign_schema_graph_hashes, compute_dialect_probe
from aetherdialect._validation_shape import validate_select_cols_schema


def _mysql_graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "t": TableMetadata(
                name="t",
                columns={"n": ColumnMetadata(name="n", data_type="integer")},
                primary_key=["n"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )


@pytest.mark.fast
def test_mysql_median_refused_without_federation() -> None:
    graph = _mysql_graph()
    mysql = MySQLDialect.__new__(MySQLDialect)
    compute_dialect_probe(mysql, EngineContext())
    assign_schema_graph_hashes(graph, EngineContext(), "")

    membership = graph.federation_membership if isinstance(graph.federation_membership, dict) else {}
    assert not membership.get("federation_id")
    assert membership.get("engine") == "mysql"
    assert graph.database_feature_capability.supports_median is False

    sc = SelectCol(
        expr=NormalizedExpr(add_groups=[MulGroup(multiply=[NormalizedExpr.from_column("t.n")], agg_func="median")])
    )
    issues = validate_select_cols_schema([sc], graph, set(graph.tables), context="main")
    assert any("median is not supported" in i.message for i in issues)
