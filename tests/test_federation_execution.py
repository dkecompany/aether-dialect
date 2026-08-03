"""Federation decomposition dialect capability checks and member cost- cap enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import explain_cost_gate_violation
from aetherdialect._federation import (
    compose_composite_graph,
    federation_plan_step_fingerprints,
    parse_federation_manifest,
    plan_federated_intent,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import _run_sql_validation_cascade
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import intent_key


def _graph(table: str, *, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"name": ColumnMetadata(name="name", data_type="text", sensitivity="none")},
            primary_key=["name"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def _two_source_manifest() -> dict:
    return {
        "federation_id": "fed_exec_t55_t56",
        "sources": [
            {"source_id": "a", "engine": "postgresql", "role": "owner"},
            {"source_id": "b", "engine": "postgresql", "role": "owner"},
        ],
        "table_namespace": {"left_t": "a", "right_t": "b"},
        "cross_source_joins": [
            {"left": "left_t.name", "right": "right_t.name", "kind": "inner", "logical_key": "name"}
        ],
    }


def _ilike_intent(*, table: str) -> RuntimeIntent:
    return RuntimeIntent(
        tables=[table],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column(f"{table}.name"),
                    op="ilike",
                    value_type="string",
                    raw_value="%x%",
                ),
            ]
        ),
    )


@pytest.mark.fast
def test_plan_federated_intent_uses_member_dialects_for_operator_intersection() -> None:
    manifest = parse_federation_manifest(_two_source_manifest(), include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    blocked = MagicMock()
    blocked.supports_ilike = False
    blocked.supports_case_insensitive_wrap = False
    blocked.extra_where_ops = MagicMock(return_value=frozenset({"=", "!=", "<", ">", "<=", ">="}))
    ok = MagicMock()
    ok.supports_ilike = True
    ok.supports_case_insensitive_wrap = True
    ok.extra_where_ops = MagicMock(return_value=frozenset({"=", "ilike", "not ilike"}))
    intent = _ilike_intent(table="right_t")
    plan_without = plan_federated_intent(intent, composite, manifest)
    plan_with = plan_federated_intent(
        intent,
        composite,
        manifest,
        dialects_by_source={"a": ok, "b": blocked},
    )
    assert plan_without.ineligible_reason is None
    assert plan_with.ineligible_reason is not None
    assert "ilike" in plan_with.ineligible_reason.lower()
    assert "'b'" in plan_with.ineligible_reason or "b" in plan_with.ineligible_reason


@pytest.mark.fast
def test_explain_cost_gate_prefers_dialect_member_limit_over_global_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyConfig, "MAX_QUERY_COST_ROWS", 50_000_000)
    dialect = MagicMock()
    dialect.max_query_cost_rows = 100.0
    failed, msg = explain_cost_gate_violation(500.0, None, dialect=dialect)
    assert failed
    assert "100" in msg


@pytest.mark.fast
def test_federation_validation_cascade_applies_fingerprinted_member_cost_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyConfig, "MAX_QUERY_COST_ROWS", 50_000_000)
    dialect = MagicMock()
    dialect.can_explain.return_value = True
    dialect.parse_select.return_value = "SELECT"
    dialect.ast_validate_full.return_value = []
    dialect.explain_validation_sql.return_value = "SELECT 1"

    def _explain_with_gate(
        _sql: str,
        _params: dict[str, object] | None = None,
        *,
        schema: object | None = None,
        intent: object | None = None,
    ) -> tuple[bool, list[object], str]:
        failed, why = explain_cost_gate_violation(500.0, None, dialect=dialect)
        if failed:
            return False, [], why
        return True, [], ""

    dialect.explain_diagnose.side_effect = _explain_with_gate
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    ok, err, _cat, _diags = _run_sql_validation_cascade(
        "SELECT 1",
        intent,
        dialect,
        max_query_cost_rows=100.0,
    )
    assert not ok
    assert err is not None
    assert "cost gate" in err.lower()


@pytest.mark.fast
def test_federation_plan_fingerprint_embeds_member_cost_cap() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cost_cap_t56",
            "sources": [
                {
                    "source_id": "a",
                    "engine": "duckdb",
                    "role": "owner",
                    "limits": {"max_query_cost_rows": 100.0},
                },
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.name", "right": "right_t.name", "kind": "inner", "logical_key": "name"}
            ],
            "coordinator": {"default_source_row_cap": 2000, "default_source_timeout_ms": 9000},
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    fingerprints = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key, manifest=manifest)
    assert '"max_query_cost_rows":100.0' in fingerprints[0][1]
