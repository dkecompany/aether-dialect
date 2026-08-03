"""Structural validation for unscoped owners when the execution scope gate is inactive."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_base import EngineContext, NormalizedExpr, WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._pipeline import _execution_scope_gate_active, _run_sql_validation_cascade
from aetherdialect._schema_graph import assert_consumer_intent_in_scope
from aetherdialect._validation_execute import validate_semantics


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _bridge_schema() -> SchemaGraph:
    bridge_edge = {
        "src_table": "a",
        "src_cols": ["id"],
        "dst_table": "bridge",
        "dst_cols": ["aid"],
    }
    second = {
        "src_table": "bridge",
        "src_cols": ["cid"],
        "dst_table": "c",
        "dst_cols": ["id"],
    }
    path = [bridge_edge, second]
    return SchemaGraph(
        tables={
            "a": TableMetadata(name="a", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
            "c": TableMetadata(name="c", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
            "bridge": TableMetadata(
                name="bridge",
                columns={"aid": _col("aid"), "cid": _col("cid")},
                primary_key=["aid"],
                foreign_keys=[],
            ),
            "island": TableMetadata(name="island", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        },
        join_paths_multi={"a": {"c": [path]}},
        effective_structural_hash="h",
    )


def _schema_with_denied_secret() -> SchemaGraph:
    t = TableMetadata(
        name="t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "secret": ColumnMetadata(name="secret", data_type="text"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"t": t},
        join_paths_multi={},
        effective_structural_hash="x",
        deny_columns={"t": {"secret"}},
    )


class TestUnscopedOwnerGateInactive:
    def test_execution_scope_gate_inactive_for_unscoped_owner(self) -> None:
        gate = _execution_scope_gate_active(EngineContext(), None, "owner")
        assert gate is False

    def test_consumer_sql_scope_gate_skipped_for_unscoped_owner(self) -> None:
        """assert_consumer_intent_in_scope is not the structural backstop when gate is off."""
        schema = _bridge_schema()
        intent = RuntimeIntent(
            tables=["a", "c"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            resolved_join_tables=["a", "bridge", "c", "island"],
        )
        gate = _execution_scope_gate_active(EngineContext(), None, "owner")
        assert gate is False
        assert assert_consumer_intent_in_scope(intent, EngineContext(), schema, None) is True


class TestUnscopedOwnerJoinBridgeValidation:
    def test_validate_semantics_unreachable_bridge_when_gate_inactive(self) -> None:
        schema = _bridge_schema()
        intent = RuntimeIntent(
            tables=["a", "c"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            resolved_join_tables=["a", "bridge", "c", "island"],
        )
        assert _execution_scope_gate_active(EngineContext(), None, "owner") is False
        result = validate_semantics(intent, schema)
        errors = [i for i in result.issues if i.severity == "error"]
        assert any("island" in i.message for i in errors)

    @patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
    def test_cascade_unreachable_bridge_when_gate_inactive(self, _mock_validate_sql: object) -> None:
        schema = _bridge_schema()
        intent = RuntimeIntent(
            tables=["a", "c"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            resolved_join_tables=["a", "bridge", "c", "island"],
        )
        assert _execution_scope_gate_active(EngineContext(), None, "owner") is False
        ok, err, _cat, _diags = _run_sql_validation_cascade(
            "SELECT a.id FROM a JOIN bridge ON a.id = bridge.aid JOIN c ON bridge.cid = c.id",
            intent,
            None,
            schema=schema,
        )
        assert ok is False
        assert "island" in err


class TestUnscopedOwnerNonTerminalCteDenyValidation:
    def test_validate_semantics_non_terminal_cte_denied_col_when_gate_inactive(self) -> None:
        schema = _schema_with_denied_secret()
        inner = RuntimeCteStep(
            cte_name="inner_x",
            tables=["t"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.secret"))],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            cte_steps=[inner],
        )
        assert _execution_scope_gate_active(EngineContext(), None, "owner") is False
        result = validate_semantics(intent, schema)
        errors = [i for i in result.issues if i.severity == "error"]
        assert any(i.category.value == "deny_bare_select" for i in errors)

    def test_validate_semantics_non_terminal_cte_denied_filter_when_gate_inactive(self) -> None:
        schema = _schema_with_denied_secret()
        inner = RuntimeCteStep(
            cte_name="inner_x",
            tables=["t"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t.secret"),
                        op="=",
                        value_type="text",
                        raw_value="x",
                    ),
                ]
            ),
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            cte_steps=[inner],
        )
        assert _execution_scope_gate_active(EngineContext(), None, "owner") is False
        result = validate_semantics(intent, schema)
        errors = [i for i in result.issues if i.severity == "error"]
        assert any(i.category.value == "denied_reference" for i in errors)
