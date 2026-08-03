"""schema_graph_id must be content-addressed, not random per build."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from aetherdialect._constants import SCHEMA_GRAPH_ID_PREFIX
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import SchemaGraph, TableMetadata
from aetherdialect._core_utils import schema_prompt_cache_id
from aetherdialect._schema_graph import (
    assign_schema_graph_hashes,
    derive_deterministic_schema_graph_id,
)


def _table(name: str) -> TableMetadata:
    return TableMetadata(name=name, columns={}, primary_key=[], foreign_keys=[], row_count=1)


def _owner_build(
    *,
    table_name: str = "orders",
    ctx: EngineContext | None = None,
) -> SchemaGraph:
    context = ctx or EngineContext()
    sg = SchemaGraph(join_paths_multi={}, tables={table_name: _table(table_name)})
    assign_schema_graph_hashes(sg, context, "", schema_role="owner")
    return sg


_SUBPROCESS_SNIPPET = textwrap.dedent(
    """
    from aetherdialect._contracts_base import EngineContext
    from aetherdialect._contracts_schema import SchemaGraph, TableMetadata
    from aetherdialect._schema_graph import assign_schema_graph_hashes

    ctx = EngineContext()
    sg = SchemaGraph(
        join_paths_multi={},
        tables={"orders": TableMetadata(name="orders", columns={}, primary_key=[], foreign_keys=[], row_count=1)},
    )
    assign_schema_graph_hashes(sg, ctx, "", schema_role="owner")
    print(sg.schema_graph_id)
    """
)


@pytest.mark.fast
def test_same_schema_inputs_mint_identical_id_across_calls() -> None:
    first = _owner_build()
    second = _owner_build()
    expected = derive_deterministic_schema_graph_id(first.effective_structural_hash, first.structural_hash)
    assert first.schema_graph_id == second.schema_graph_id
    assert first.schema_graph_id == expected
    assert first.schema_graph_id.startswith(SCHEMA_GRAPH_ID_PREFIX)


@pytest.mark.fast
def test_schema_graph_id_stable_across_subprocess_builds() -> None:
    in_process = _owner_build().schema_graph_id
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SNIPPET],
        check=True,
        capture_output=True,
        text=True,
    )
    child_id = proc.stdout.strip()
    assert child_id
    assert child_id == in_process


@pytest.mark.fast
def test_scope_change_mints_different_schema_graph_id() -> None:
    default_ctx = _owner_build()
    scoped = _owner_build(ctx=EngineContext(allow_objects=frozenset({"orders"})))
    assert default_ctx.effective_structural_hash != scoped.effective_structural_hash
    assert default_ctx.schema_graph_id != scoped.schema_graph_id


@pytest.mark.fast
def test_prompt_cache_id_uses_content_addressed_schema_graph_id() -> None:
    graph = _owner_build()
    cache_id = schema_prompt_cache_id(graph)
    assert cache_id is not None
    assert cache_id.startswith(graph.schema_graph_id)
