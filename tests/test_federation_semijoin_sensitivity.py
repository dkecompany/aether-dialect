"""Runtime semi-join sensitivity guard tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationDeclarationError
from aetherdialect._contracts_core import RuntimeIntent, SourceStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import (
    plan_federated_intent,
    semijoin_key_is_allowed,
)
from aetherdialect._pipeline_execute import _execute_federation_source_step
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(left: str, right: str) -> dict[str, SchemaGraph]:
    left_table = TableMetadata(
        name=left,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "secret": ColumnMetadata(name="secret", data_type="text", sensitivity="restricted"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="a",
    )
    right_table = TableMetadata(
        name=right,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="b",
    )
    return {
        "a": SchemaGraph(
            tables={left: left_table},
            join_paths_multi=recompute_join_paths_multi({left: left_table}),
            profiling_hash="test-profiled",
        ),
        "b": SchemaGraph(
            tables={right: right_table},
            join_paths_multi=recompute_join_paths_multi({right: right_table}),
            profiling_hash="test-profiled",
        ),
    }


_MANIFEST = {
    "federation_id": "fed_guard",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}

_RESTRICTED_JOIN_MANIFEST = {
    **_MANIFEST,
    "cross_source_joins": [
        {"left": "left_t.secret", "right": "right_t.id", "kind": "inner", "logical_key": "secret"},
    ],
}


def test_semijoin_key_is_allowed_rejects_restricted() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(_graph("left_t", "right_t"), manifest)
    assert semijoin_key_is_allowed(composite, "left_t", "secret") is False
    assert semijoin_key_is_allowed(composite, "right_t", "id") is True


def test_restricted_cross_source_join_rejected_at_compose() -> None:
    manifest = parse_federation_manifest(_RESTRICTED_JOIN_MANIFEST, include_derived_roster=True)
    with pytest.raises(FederationDeclarationError, match="sensitivity none"):
        compose_composite_graph(_graph("left_t", "right_t"), manifest)


def test_semijoin_skips_restricted_driving_key(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(_graph("left_t", "right_t"), manifest)
    intent = RuntimeIntent(
        tables=["right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    prep = type(
        "Prep",
        (),
        {
            "sub_intent": intent,
            "sql": "SELECT id FROM right_t",
            "structural_defaults": None,
        },
    )()
    executed = {"a": pd.DataFrame({"secret": ["x"]})}
    calls: list[str] = []

    def _fake_execute_guarded_sql(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
        calls.append("execute")
        return [(1,)]

    monkeypatch.setattr("aetherdialect._pipeline_execute.execute_guarded_sql", _fake_execute_guarded_sql)
    monkeypatch.setattr(
        "aetherdialect._pipeline_execute.generate_and_validate_sql",
        lambda *a, **k: type("Out", (), {"success": True, "sql": "SELECT id FROM right_t"})(),
    )
    monkeypatch.setattr(
        "aetherdialect._pipeline_execute.build_result_dataframe",
        lambda *a, **k: pd.DataFrame({"id": [1]}),
    )
    monkeypatch.setattr(
        "aetherdialect._federation_execute.validate_sql",
        lambda *a, **k: (True, None, None, None),
    )
    mock_dialect = MagicMock()
    mock_dialect.finalize_render.return_value = "SELECT id FROM right_t"
    step = SourceStep(source_id="b", sub_intent=intent)
    frame = _execute_federation_source_step(
        step,
        prepared_by_source={"b": prep},
        composite_schema=composite,
        dialect_map={},
        dialect=mock_dialect,
        manifest=manifest,
        executed=executed,
        plan=plan,
        semijoin_cap=100,
        q_norm="q",
        join_candidates={},
        cmap={},
        store={},
        gate_kwargs={},
    )
    assert frame is not None
    assert calls
