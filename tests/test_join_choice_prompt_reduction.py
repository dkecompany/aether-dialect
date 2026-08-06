"""Join-choice prompts expose candidate ids and path signatures only."""

from __future__ import annotations

import json

from aetherdialect._contracts_base import ColumnRole
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import (
    JOIN_CHOICE_SCOPE_MAIN,
    _serialize_join_candidate_row,
    build_join_choice_prompt,
    join_hints_multi,
)


def test_serialized_join_candidate_omits_engine_provenance_fields() -> None:
    row = {
        "candidate_id": "J01",
        "join_path_signature": ["orders.customer_id->customers.id"],
        "edge_kinds": ["catalog_fk"],
        "candidate_tier": "base",
    }
    out = _serialize_join_candidate_row(row)
    assert set(out.keys()) == {"candidate_id", "join_path_signature"}
    assert "edge_kinds" not in out
    assert "candidate_tier" not in out


def test_join_choice_prompt_payload_omits_engine_provenance_fields() -> None:
    scopes = [
        {
            "scope": JOIN_CHOICE_SCOPE_MAIN,
            "tables": ["orders", "customers"],
            "candidates": [
                {
                    "candidate_id": "J01",
                    "join_path_signature": ["orders.customer_id->customers.id"],
                    "edge_kinds": ["catalog_fk"],
                    "candidate_tier": "base",
                }
            ],
        }
    ]
    _, user = build_join_choice_prompt("how many orders?", "SELECT 1", scopes)
    payload = json.loads(user)
    serialized = payload["scopes"][0]["candidates"][0]
    assert serialized["candidate_id"] == "J01"
    assert "edge_kinds" not in serialized
    assert "candidate_tier" not in serialized


def test_declared_key_paths_remain_first_in_enumeration_order() -> None:
    left_cols = {
        "right_tbl_id": ColumnMetadata(
            name="right_tbl_id",
            data_type="varchar",
            value_type="string",
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=100,
            row_count=200,
            null_ratio=0.0,
            semantic_join_neighbors=[("right_tbl", "other")],
        ),
    }
    right_cols = {
        "id": ColumnMetadata(
            name="id",
            data_type="varchar",
            value_type="string",
            is_primary_key=True,
            distinct_count=100,
            row_count=100,
            null_ratio=0.0,
            semantic_join_neighbors=[],
        ),
        "other": ColumnMetadata(
            name="other",
            data_type="varchar",
            value_type="string",
            distinct_count=50,
            row_count=100,
            null_ratio=0.0,
            semantic_join_neighbors=[("left_tbl", "right_tbl_id")],
        ),
    }
    tables = {
        "left_tbl": TableMetadata(name="left_tbl", columns=left_cols, primary_key=[], foreign_keys=[]),
        "right_tbl": TableMetadata(
            name="right_tbl",
            columns=right_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="join_prompt_order",
    )
    hints = join_hints_multi(
        schema,
        ["left_tbl", "right_tbl"],
        None,
        virtual_specs={},
        include_semantic=True,
    )
    non_j00 = [c for c in hints["candidates"] if c.get("candidate_id") != "J00"]
    assert len(non_j00) >= 1
    tiers = [c.get("candidate_tier") for c in non_j00]
    if any(t == "extended" for t in tiers):
        first_extended = next(i for i, t in enumerate(tiers) if t == "extended")
        assert all(t == "base" for t in tiers[:first_extended])
    _, user = build_join_choice_prompt(
        "list pairs",
        "SELECT 1",
        [{"scope": JOIN_CHOICE_SCOPE_MAIN, "tables": ["left_tbl", "right_tbl"], "candidates": non_j00}],
    )
    payload = json.loads(user)
    for cand in payload["scopes"][0]["candidates"]:
        assert "edge_kinds" not in cand
        assert "candidate_tier" not in cand
