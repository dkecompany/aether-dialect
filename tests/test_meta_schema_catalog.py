"""schema_catalog metadata answers from filtered schema dump."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import META_DEFAULT_SOURCE_ID, SESSION_KIND_META
from aetherdialect._contracts_base import Diagnostic, QuestionRoute, SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._core_utils import drain_diagnostic_collector, set_diagnostic_collector
from aetherdialect._main_execution import MainExecutionOps


def _col(
    name: str,
    *,
    data_type: str = "integer",
    role: str = "identifier",
    sensitivity: SensitivityClassification = SensitivityClassification.NONE,
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    fk_target: tuple[str, str] | None = None,
    description: str = "",
    is_denied: bool = False,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type=data_type,
        value_type="integer" if data_type == "integer" else "string",
        role=role,
        sensitivity=sensitivity,
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
        fk_target=fk_target,
        description=description,
        is_denied=is_denied,
        distinct_count=10,
        distinct_ratio=1.0,
        row_count=10,
        null_ratio=0.0,
    )


def _engine_graph() -> SchemaGraph:
    customers = TableMetadata(
        name="customers",
        columns={
            "customer_id": _col("customer_id", is_primary_key=True, description="pk"),
            "email": _col("email", data_type="varchar", role="free_text", description="email"),
            "ssn": _col(
                "ssn",
                data_type="varchar",
                role="free_text",
                sensitivity=SensitivityClassification.HIDDEN,
                description="secret",
            ),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
        description="Customer rows",
        source_id="",
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "order_id": _col("order_id", is_primary_key=True),
            "customer_id": _col(
                "customer_id",
                is_foreign_key=True,
                fk_target=("customers", "customer_id"),
            ),
            "amount": _col("amount", data_type="numeric", role="numeric_measure"),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["customer_id"],
            )
        ],
        description="Order rows",
        source_id="",
    )
    return SchemaGraph(
        tables={"customers": customers, "orders": orders},
        join_paths_multi={},
        schema_graph_id="sg-meta-engine",
        effective_structural_hash="hash-meta",
    )


def _federation_graph() -> SchemaGraph:
    a = TableMetadata(
        name="crm.accounts",
        columns={"id": _col("id", is_primary_key=True), "name": _col("name", data_type="varchar", role="free_text")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="crm",
        description="Accounts",
    )
    b = TableMetadata(
        name="log.shipments",
        columns={"id": _col("id", is_primary_key=True), "account_id": _col("account_id")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="logistics",
        description="Shipments",
    )
    c = TableMetadata(
        name="store.orders",
        columns={"id": _col("id", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
        source_id="storefront",
        description="Store orders",
    )
    return SchemaGraph(
        tables={"crm.accounts": a, "log.shipments": b, "store.orders": c},
        join_paths_multi={},
        schema_graph_id="sg-meta-fed",
        effective_structural_hash="hash-fed",
        federation_membership={"federation_id": "fed1"},
    )


def _count_answer(
    *, tables: int | None = None, columns: int | None = None, members: int | None = None
) -> dict[str, Any]:
    return {
        "response_kind": "schema_catalog",
        "headline": "Inventory summary.",
        "counts": {
            "tables": tables,
            "columns": columns,
            "members": members,
            "columns_in_table": None,
            "tables_in_member": None,
        },
        "tables": [],
        "relationships": [],
        "notes": [],
    }


@pytest.mark.fast
def test_answer_matches_schema() -> None:
    schema = _engine_graph()
    dump = MainExecutionOps.build_meta_schema_dump(schema)
    assert dump["inventory"]["table_count"] == 2
    answer = {
        "response_kind": "schema_catalog",
        "headline": "Customers and orders are available.",
        "counts": {
            "tables": None,
            "columns": None,
            "members": None,
            "columns_in_table": None,
            "tables_in_member": None,
        },
        "tables": [
            {
                "name": "customers",
                "source_id": META_DEFAULT_SOURCE_ID,
                "description": "Customer rows",
                "columns": [
                    {"name": "customer_id", "data_type": "integer", "role": "identifier", "description": "pk"},
                    {"name": "email", "data_type": "varchar", "role": "free_text", "description": "email"},
                ],
            }
        ],
        "relationships": [
            {
                "left": "orders.customer_id",
                "right": "customers.customer_id",
                "kind": "fk",
            }
        ],
        "notes": [],
    }
    MainExecutionOps.validate_meta_schema_answer(answer, dump)
    owner = MagicMock()
    with patch("aetherdialect._main_execution.LLMProvider.json", return_value=answer):
        step = MainExecutionOps.answer_metadata_question(
            owner, "what tables exist", QuestionRoute.SCHEMA_CATALOG, schema, None, None
        )
    assert step.kind == SESSION_KIND_META
    assert step.meta_payload is not None
    assert step.meta_payload["response_kind"] == "schema_catalog"
    assert "Customers and orders are available." in (step.message or "")


@pytest.mark.fast
def test_count_tables_uses_inventory() -> None:
    schema = _engine_graph()
    dump = MainExecutionOps.build_meta_schema_dump(schema)
    answer = _count_answer(tables=dump["inventory"]["table_count"])
    MainExecutionOps.validate_meta_schema_answer(answer, dump)
    with patch("aetherdialect._main_execution.LLMProvider.json", return_value=answer):
        step = MainExecutionOps.answer_metadata_question(
            MagicMock(), "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, None, None
        )
    assert step.meta_payload is not None
    assert step.meta_payload["counts"]["tables"] == dump["inventory"]["table_count"]
    assert str(dump["inventory"]["table_count"]) in (step.message or "")


@pytest.mark.fast
def test_count_columns_in_table() -> None:
    schema = _engine_graph()
    dump = MainExecutionOps.build_meta_schema_dump(schema)
    cols = dump["inventory"]["columns_per_table"]["customers"]
    answer = {
        "response_kind": "schema_catalog",
        "headline": "Customer column count.",
        "counts": {
            "tables": None,
            "columns": None,
            "members": None,
            "columns_in_table": {"table": "customers", "columns": cols},
            "tables_in_member": None,
        },
        "tables": [],
        "relationships": [],
        "notes": [],
    }
    MainExecutionOps.validate_meta_schema_answer(answer, dump)
    with patch("aetherdialect._main_execution.LLMProvider.json", return_value=answer):
        step = MainExecutionOps.answer_metadata_question(
            MagicMock(), "how many columns in customers", QuestionRoute.SCHEMA_CATALOG, schema, None, None
        )
    assert step.meta_payload is not None
    assert step.meta_payload["counts"]["columns_in_table"]["columns"] == cols


@pytest.mark.fast
def test_federation_member_count() -> None:
    schema = _federation_graph()
    dump = MainExecutionOps.build_meta_schema_dump(schema)
    assert dump["inventory"]["member_count"] == 3
    assert set(dump["inventory"]["tables_per_member"]) == {"crm", "logistics", "storefront"}
    answer = _count_answer(members=3)
    MainExecutionOps.validate_meta_schema_answer(answer, dump)
    with patch("aetherdialect._main_execution.LLMProvider.json", return_value=answer):
        step = MainExecutionOps.answer_metadata_question(
            MagicMock(), "how many members", QuestionRoute.SCHEMA_CATALOG, schema, None, None
        )
    assert step.meta_payload is not None
    assert step.meta_payload["counts"]["members"] == 3


@pytest.mark.fast
def test_hidden_columns_absent_from_dump() -> None:
    dump = MainExecutionOps.build_meta_schema_dump(_engine_graph())
    customers = next(t for t in dump["tables"] if t["name"] == "customers")
    names = {c["name"] for c in customers["columns"]}
    assert "ssn" not in names
    assert "email" in names
    assert dump["inventory"]["columns_per_table"]["customers"] == 2


@pytest.mark.fast
def test_invented_table_fails_validation() -> None:
    schema = _engine_graph()
    dump = MainExecutionOps.build_meta_schema_dump(schema)
    answer = {
        "response_kind": "schema_catalog",
        "headline": "Invented.",
        "counts": {
            "tables": None,
            "columns": None,
            "members": None,
            "columns_in_table": None,
            "tables_in_member": None,
        },
        "tables": [
            {
                "name": "not_a_real_table",
                "source_id": META_DEFAULT_SOURCE_ID,
                "description": "",
                "columns": [],
            }
        ],
        "relationships": [],
        "notes": [],
    }
    with pytest.raises(ValueError, match="not_a_real_table"):
        MainExecutionOps.validate_meta_schema_answer(answer, dump)


@pytest.mark.fast
def test_step_kind_meta_sql_none() -> None:
    schema = _engine_graph()
    answer = _count_answer(tables=2)
    with patch("aetherdialect._main_execution.LLMProvider.json", return_value=answer):
        step = MainExecutionOps.answer_metadata_question(
            MagicMock(), "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, None, None
        )
    assert step.kind == SESSION_KIND_META
    assert step.done is True
    assert step.sql is None
    assert step.data is None
    assert step.error is None
    assert step.status is None


@pytest.mark.fast
def test_diagnostics_include_route_and_validated() -> None:
    schema = _engine_graph()
    answer = _count_answer(tables=2)
    buf: list[Diagnostic] = []
    tok = set_diagnostic_collector(buf)
    try:
        with patch("aetherdialect._main_execution.LLMProvider.json", return_value=answer):
            # Emulate route notify then answer path (same order as interactive_run_once).
            from aetherdialect._core_utils import notify

            notify("Metadata route: schema_catalog", stage="meta", code="meta.route.schema_catalog", level="info")
            step = MainExecutionOps.answer_metadata_question(
                MagicMock(), "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, None, None
            )
        drained = drain_diagnostic_collector()
        codes = {d.code for d in step.diagnostics} | {d.code for d in drained} | {d.code for d in buf}
    finally:
        from aetherdialect._core_utils import reset_diagnostic_collector

        reset_diagnostic_collector(tok)
    assert "meta.route.schema_catalog" in codes
    assert "meta.answer.validated" in codes
    assert "meta.cache.miss" in codes
