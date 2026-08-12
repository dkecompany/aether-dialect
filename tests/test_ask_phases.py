"""Ask-phase progress events during non-federation intent parsing."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._constants_runtime import ASK_PHASE_D, ASK_PHASE_F
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, LogicalIntent, SchemaGraph, TableMetadata
from aetherdialect._intent_loop import full_intent_parse, logical_intent_to_serialisable
from aetherdialect._utils import pop_ask_phase_callback, push_ask_phase_callback


def _minimal_schema() -> SchemaGraph:
    col = ColumnMetadata(
        name="id",
        data_type="integer",
        is_primary_key=True,
        distinct_count=1,
        distinct_ratio=1.0,
        null_ratio=0.0,
        is_canonical_duplicate=False,
    )
    return SchemaGraph(
        tables={"t": TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])},
        join_paths_multi={},
        effective_structural_hash="h",
    )


@pytest.mark.fast
def test_non_federation_ask_emits_compose_and_validate_phases() -> None:
    """Intent parse emits compose (D) and schema validate (F) ask phases."""
    sg = _minimal_schema()
    li = logical_intent_to_serialisable(LogicalIntent(tables=("t",), select="id column"))
    interpret_ok = '{"approach":"count rows on t","tables":["t"]}'
    good_ground = json.dumps(li, separators=(",", ":"), sort_keys=True)
    stub_intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        natural_language="nl",
    )
    llm_seq = [interpret_ok, good_ground, "{}"]
    emitted: list[str] = []
    token = push_ask_phase_callback(lambda ev: emitted.append(ev.phase))
    try:
        with (
            patch("aetherdialect._intent_loop.LLMProvider.chat", side_effect=list(llm_seq)),
            patch(
                "aetherdialect._intent_loop._format_repair_loop",
                return_value=(stub_intent, 0),
            ),
        ):
            out, _warns, _calls, _plan = full_intent_parse("count rows", sg, store=None)
    finally:
        pop_ask_phase_callback(token)

    assert out is not None
    assert ASK_PHASE_D in emitted
    assert ASK_PHASE_F in emitted
