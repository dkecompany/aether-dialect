"""Schema path health notices must report foreign-key and semantic connectivity separately."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._schema_overrides import (
    _format_disconnected_components_message,
    notify_schema_path_health,
)


def _col(name: str, **overrides) -> ColumnMetadata:
    defaults = dict(
        name=name,
        data_type="varchar",
        value_type="string",
        is_primary_key=False,
        is_foreign_key=False,
        fk_target=None,
    )
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _table(name: str, columns: dict[str, ColumnMetadata], **overrides) -> TableMetadata:
    defaults = dict(
        name=name,
        columns=columns,
        primary_key=[],
        foreign_keys=[],
    )
    defaults.update(overrides)
    return TableMetadata(**defaults)


@pytest.mark.fast
def test_health_notice_reports_foreign_key_and_semantic_component_counts() -> None:
    msg = _format_disconnected_components_message(
        [{"orders"}, {"audit"}],
        [{"orders"}, {"audit"}],
    )
    assert "2 component(s) over foreign keys" in msg
    assert "2 component(s) after semantic neighbours" in msg


@pytest.mark.fast
def test_health_notice_emits_when_semantic_overlap_bridges_foreign_key_islands(monkeypatch) -> None:
    captured: list[str] = []

    def capture_notify(msg: str, **kwargs: object) -> None:
        captured.append(msg)

    monkeypatch.setattr("aetherdialect._schema_overrides.notify", capture_notify)

    orders = _table(
        "orders",
        {
            "id": _col("id", is_primary_key=True),
            "status_code": _col("status_code"),
        },
    )
    statuses = _table(
        "statuses",
        {
            "code": _col("code", is_primary_key=True),
        },
    )
    audit = _table("audit", {"id": _col("id", is_primary_key=True)})
    orders.columns["status_code"].semantic_join_neighbors = [("statuses", "code")]
    statuses.columns["code"].semantic_join_neighbors = [("orders", "status_code")]

    sg = SchemaGraph(
        tables={"orders": orders, "statuses": statuses, "audit": audit},
        join_paths_multi={},
        effective_structural_hash="health",
    )
    notify_schema_path_health(sg)
    assert len(captured) == 1
    assert "3 component(s) over foreign keys" in captured[0]
    assert "2 component(s) after semantic neighbours" in captured[0]


@pytest.mark.fast
def test_health_notice_emits_when_only_semantic_overlap_connects_tables(monkeypatch) -> None:
    captured: list[str] = []

    def capture_notify(msg: str, **kwargs: object) -> None:
        captured.append(msg)

    monkeypatch.setattr("aetherdialect._schema_overrides.notify", capture_notify)

    orders = _table(
        "orders",
        {
            "id": _col("id", is_primary_key=True),
            "status_code": _col("status_code"),
        },
    )
    statuses = _table(
        "statuses",
        {
            "code": _col("code", is_primary_key=True),
        },
    )
    orders.columns["status_code"].semantic_join_neighbors = [("statuses", "code")]
    statuses.columns["code"].semantic_join_neighbors = [("orders", "status_code")]

    sg = SchemaGraph(
        tables={"orders": orders, "statuses": statuses},
        join_paths_multi={},
        effective_structural_hash="health",
    )
    notify_schema_path_health(sg)
    assert len(captured) == 1
    assert "2 component(s) over foreign keys" in captured[0]
    assert "1 component(s) after semantic neighbours" in captured[0]


@pytest.mark.fast
def test_health_notice_silent_when_fully_connected(monkeypatch) -> None:
    captured: list[str] = []

    def capture_notify(msg: str, **kwargs: object) -> None:
        captured.append(msg)

    monkeypatch.setattr("aetherdialect._schema_overrides.notify", capture_notify)

    customers = _table("customers", {"id": _col("id", is_primary_key=True)}, primary_key=["id"])
    orders = _table(
        "orders",
        {"id": _col("id", is_primary_key=True), "customer_id": _col("customer_id")},
        primary_key=["id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
            )
        ],
    )
    sg = SchemaGraph(
        tables={"customers": customers, "orders": orders},
        join_paths_multi={},
        effective_structural_hash="health",
    )
    notify_schema_path_health(sg)
    assert captured == []
