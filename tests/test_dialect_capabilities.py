"""Dialect capability surfaces, BigQuery FK warning, and dialect-aware param escaping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
    substitute_params,
)
from aetherdialect._dialect import Dialect, engine_supports_median, engine_supports_ordered_string_agg
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import BigQueryDialect, MySQLDialect
from aetherdialect._federation import stamp_federation_member_graph
from aetherdialect._schema_graph import compute_database_feature_capability, recompute_join_paths_multi
from aetherdialect._schema_overrides import _add_profiling_data


def _member_graph(engine: str) -> SchemaGraph:
    tables = {
        "t": TableMetadata(
            name="t",
            columns={"n": ColumnMetadata(name="n", data_type="integer")},
            primary_key=["n"],
            foreign_keys=[],
        )
    }
    graph = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    stamp_federation_member_graph(graph, federation_id="fed", source_id="local", engine=engine)
    return graph


def _two_table_graph(engine: str) -> SchemaGraph:
    tables = {
        "a": TableMetadata(
            name="a",
            columns={"id": ColumnMetadata(name="id", data_type="integer")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "b": TableMetadata(
            name="b",
            columns={"id": ColumnMetadata(name="id", data_type="integer")},
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    graph = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    stamp_federation_member_graph(graph, federation_id="fed", source_id="local", engine=engine)
    return graph


@pytest.mark.fast
def test_capability_flags_read_dialect_not_engine_denylist() -> None:
    assert engine_supports_ordered_string_agg("databricks") is False
    assert engine_supports_median("mysql") is False
    assert engine_supports_median("postgresql") is True


@pytest.mark.fast
def test_compute_database_feature_capability_uses_dialect_surface() -> None:
    mysql_cap = compute_database_feature_capability(_member_graph("mysql"))
    databricks_cap = compute_database_feature_capability(_member_graph("databricks"))
    assert mysql_cap.supports_median is False
    assert databricks_cap.supports_ordered_string_agg is False


@pytest.mark.fast
def test_semi_join_capability_comes_from_dialect_not_table_count_alone() -> None:
    class _NoSemiDialect(Dialect):
        name = "stub_nosemi"

        @property
        def supports_semi_join(self) -> bool:
            return False

    graph = _two_table_graph("postgresql")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "aetherdialect._schema_graph._graph_supports_semi_join",
            lambda _sg: _NoSemiDialect.__new__(_NoSemiDialect).supports_semi_join,
        )
        cap = compute_database_feature_capability(graph)
    assert cap.supports_semi_join is False


@pytest.mark.fast
def test_bigquery_empty_catalog_fk_emits_join_inference_warning() -> None:
    graph = _two_table_graph("bigquery")
    dialect = BigQueryDialect.__new__(BigQueryDialect)
    dialect.name = "bigquery"
    dialect.profile_schema = MagicMock()

    token = set_diagnostic_collector([])
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("aetherdialect._schema_overrides.apply_column_roles_llm", lambda *args, **kwargs: None)
            monkeypatch.setattr("aetherdialect._schema_overrides.run_fk_inference_if_disconnected", lambda *args, **kwargs: 0)
            monkeypatch.setattr("aetherdialect._schema_overrides.infer_missing_pks_from_profile", lambda *args, **kwargs: None)
            monkeypatch.setattr("aetherdialect._schema_overrides.load_inference_block_lists", lambda *args, **kwargs: (frozenset(), frozenset()))
            _add_profiling_data(dialect, graph, notes_content=None)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert any(d.code == DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT for d in diags)
    assert any("join graph" in d.message.lower() or "inferred" in d.message.lower() for d in diags)


@pytest.mark.fast
def test_substitute_params_uses_mysql_backslash_escaping() -> None:
    mysql = MySQLDialect.__new__(MySQLDialect)
    sql = substitute_params("WHERE path = :p1", {"p1": r"a\b"}, dialect=mysql)
    assert sql == "WHERE path = 'a\\\\b'"


@pytest.mark.fast
def test_substitute_params_uses_postgres_estring_for_backslashes() -> None:
    pg = PostgresDialect.__new__(PostgresDialect)
    sql = substitute_params("WHERE path = :p1", {"p1": r"a\b"}, dialect=pg)
    assert sql == r"WHERE path = E'a\\b'"
