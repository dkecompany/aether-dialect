"""Sandbox consumer allow_objects: one construction path for every owner subset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect import EngineContext, Sandbox
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import assert_consumer_sql_in_scope, recompute_join_paths_multi
from tests._sandbox_csv_bundle import write_main_csv_ddl_bundle

_NARROW_ALLOW_OBJECTS = frozenset({"customer", "payment", "rental", "address", "city", "country"})


def _write_consumer_test_bundle(root: Path) -> None:
    write_main_csv_ddl_bundle(root, tables=(("customer", "customer_id"),))


def _minimal_schema_graph(table_names: tuple[str, ...]) -> SchemaGraph:
    tables = {
        name: TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
        for name in table_names
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="s2_consumer_sg",
        effective_structural_hash="s2_consumer_sg",
    )


@pytest.mark.fast
def test_any_subset_consumer_no_special_path(tmp_path: Path) -> None:
    """Distinct allow subsets share one consumer construction path (scope only differs)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_consumer_test_bundle(bundle)
    narrow = frozenset({"customer", "rental"})
    wider = frozenset({"customer", "film", "payment", "rental"})
    contexts_at_init: list[frozenset[str] | None] = []

    class FakeEngine:
        def __init__(self, schema_context: EngineContext, **kwargs: object) -> None:
            del kwargs
            contexts_at_init.append(
                frozenset(schema_context.allow_objects) if schema_context.allow_objects else None,
            )
            from aetherdialect._contracts_core import RuntimeConfig
            from aetherdialect._utils_artifacts import load_runtime_config

            llm_exec = load_runtime_config(merged_env={})
            self._schema_graph = _minimal_schema_graph(tuple(_NARROW_ALLOW_OBJECTS | wider))
            self._runtime_config = RuntimeConfig(
                engine="duckdb",
                artifacts_dir="/tmp/sandbox_consumer_allow",
                engine_context=schema_context,
                llm_execution=llm_exec,
                execution_context=schema_context,
            )
            self._schema_role = "consumer"
            self._dialect = MagicMock()
            self._sandbox_mode = True

    import aetherdialect._sandbox

    original = aetherdialect._sandbox.Sandbox._aether_engine_cls
    aetherdialect._sandbox.Sandbox._aether_engine_cls = lambda: FakeEngine
    try:
        with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
            sandbox.load_dataset("main")
            engine_narrow = sandbox.engine(EngineContext(allow_objects=narrow), role="consumer")
            engine_wide = sandbox.engine(EngineContext(allow_objects=wider), role="consumer")
    finally:
        aetherdialect._sandbox.Sandbox._aether_engine_cls = original

    assert contexts_at_init == [narrow, wider]
    narrow_exec = engine_narrow._runtime_config.execution_context
    wide_exec = engine_wide._runtime_config.execution_context
    assert narrow_exec.allow_objects == narrow
    assert wide_exec.allow_objects == wider
    assert engine_narrow._consumer_visible_objects == narrow
    assert engine_wide._consumer_visible_objects == wider


@pytest.mark.fast
def test_example_narrow_allow_denies_out_of_scope_question(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_consumer_test_bundle(bundle)

    class FakeEngine:
        def __init__(self, schema_context: EngineContext, **kwargs: object) -> None:
            del kwargs
            from aetherdialect._contracts_core import RuntimeConfig
            from aetherdialect._utils_artifacts import load_runtime_config

            llm_exec = load_runtime_config(merged_env={})
            self._schema_graph = _minimal_schema_graph(("customer", "staff"))
            self._runtime_config = RuntimeConfig(
                engine="duckdb",
                artifacts_dir="/tmp/sandbox_consumer_narrow",
                engine_context=schema_context,
                llm_execution=llm_exec,
                execution_context=schema_context,
            )
            self._schema_role = "consumer"
            self._dialect = MagicMock()

    import aetherdialect._sandbox

    original = aetherdialect._sandbox.Sandbox._aether_engine_cls
    aetherdialect._sandbox.Sandbox._aether_engine_cls = lambda: FakeEngine
    try:
        with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
            sandbox.load_dataset("main")
            engine = sandbox.engine(
                EngineContext(allow_objects=_NARROW_ALLOW_OBJECTS),
                role="consumer",
            )
    finally:
        aetherdialect._sandbox.Sandbox._aether_engine_cls = original

    assert "staff" not in engine._consumer_visible_objects
    allowed = assert_consumer_sql_in_scope(
        "SELECT salary FROM staff",
        engine._dialect,
        engine._runtime_config.execution_context,
        engine._schema_graph,
        engine._consumer_visible_objects,
    )
    assert allowed is False
