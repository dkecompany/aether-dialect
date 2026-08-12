"""CTE count and reference depth are bounded."""

from __future__ import annotations

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import SELF_JOIN_CTE_NAME_PREFIX
from aetherdialect._contracts_base import NormalizedExpr, SchemaInvariantError
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_bind import encode_inline_self_join_as_cte
from aetherdialect._validation_shape import max_cte_reference_depth, validate_cte_limits


def _schema() -> SchemaGraph:
    tables = {
        "staff": TableMetadata(
            name="staff",
            columns={"staff_id": ColumnMetadata(name="staff_id", data_type="integer", is_primary_key=True)},
            primary_key=["staff_id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi={}, effective_structural_hash="staff-only")


def _cte_step(name: str, *, tables: list[str]) -> RuntimeCteStep:
    return RuntimeCteStep(
        cte_name=name,
        tables=tables,
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("staff.staff_id"))],
        output_columns=["staff_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )


def _chain_intent(length: int) -> RuntimeIntent:
    steps: list[RuntimeCteStep] = []
    for idx in range(length):
        name = f"cte{idx + 1}"
        tables = ["staff"] if idx == 0 else [f"cte{idx}"]
        steps.append(_cte_step(name, tables=tables))
    return RuntimeIntent(
        tables=["staff", f"cte{length}"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"cte{length}.staff_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=steps,
    )


@pytest.mark.fast
def test_exceeding_cte_count_refuses_with_observed_and_permitted_values(monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "MAX_CTE_STEPS", 2)
    issues = validate_cte_limits(_chain_intent(3))
    assert len(issues) == 1
    assert "3 CTE steps" in issues[0].message
    assert "at most 2" in issues[0].message


@pytest.mark.fast
def test_exceeding_cte_reference_depth_refuses_with_observed_and_permitted_values(monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "MAX_CTE_REFERENCE_DEPTH", 2)
    issues = validate_cte_limits(_chain_intent(3))
    assert any("reference depth is 3" in issue.message for issue in issues)
    assert any("at most 2" in issue.message for issue in issues)


@pytest.mark.fast
def test_cte_reference_depth_counts_cte_to_cte_chain() -> None:
    assert max_cte_reference_depth(_chain_intent(4).cte_steps or []) == 4


@pytest.mark.fast
def test_self_join_encoding_refuses_when_cte_step_cap_would_be_exceeded(monkeypatch, simple_schema) -> None:
    monkeypatch.setattr(PolicyConfig, "MAX_CTE_STEPS", 1)
    at_cap = RuntimeCteStep(
        cte_name="existing",
        tables=["customers"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
        output_columns=["customer_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["customers", "customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[at_cap],
    )
    with pytest.raises(SchemaInvariantError, match="at most 1"):
        encode_inline_self_join_as_cte(intent, simple_schema)


@pytest.mark.fast
def test_self_join_encoding_refuses_when_cte_reference_depth_cap_would_be_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "MAX_CTE_REFERENCE_DEPTH", 1)
    schema = _schema()
    sj_name = f"{SELF_JOIN_CTE_NAME_PREFIX}staff"
    inner = _cte_step("inner", tables=[sj_name])
    intent = RuntimeIntent(
        tables=["staff", "staff"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("staff.staff_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[inner],
    )
    with pytest.raises(SchemaInvariantError, match="reference depth"):
        encode_inline_self_join_as_cte(intent, schema)
