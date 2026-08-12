"""Federation union logical-table refusal messaging."""

from __future__ import annotations

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FederationMappings,
    LogicalTableMapping,
    LogicalTableMember,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._federation_execute import (
    federation_ineligible_answerable_hint,
    federation_ineligible_reason_code,
)
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import refusal_diagnostic_code_for_federation_reason


def _union_lines_schema() -> SchemaGraph:
    table = TableMetadata(
        name="lines",
        columns={
            "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
            "region": ColumnMetadata(name="region", data_type="string", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="",
        member_source_ids=["north", "south"],
        column_member_sources={"amount": ["north"], "region": ["south"]},
    )
    tables = {"lines": table}
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def test_missing_union_column_refusal_names_column() -> None:
    schema = _union_lines_schema()
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_union_column",
            "sources": [
                {"source_id": "north", "engine": "duckdb", "role": "owner"},
                {"source_id": "south", "engine": "duckdb", "role": "owner"},
            ],
        },
        include_derived_roster=True,
    )
    mappings = FederationMappings(
        version="0.2.3",
        logical_tables=(
            LogicalTableMapping(
                logical="lines",
                semantics="union",
                members=(
                    LogicalTableMember(source="north", table="lines", columns={"amount": "amount"}),
                    LogicalTableMember(source="south", table="lines", columns={"region": "region"}),
                ),
            ),
        ),
    )
    intent = RuntimeIntent(
        tables=["lines"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("lines.amount")),
            SelectCol(expr=NormalizedExpr.from_column("lines.region")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, schema, manifest, mappings)
    reason = plan.ineligible_reason
    assert reason is not None
    assert "amount" in reason
    assert "lines" in reason
    assert "south" in reason
    assert refusal_diagnostic_code_for_federation_reason(reason) == DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING
    assert federation_ineligible_reason_code(reason) == "union_column_missing"
    hint = federation_ineligible_answerable_hint(reason)
    assert hint is not None
    assert "members that do have the column" in hint
