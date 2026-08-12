"""Join enumeration policy: cap refusal, probe collapse, kind integrity, ordering, preset."""

from __future__ import annotations

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import (
    DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP,
    JOIN_CHOICE_SCOPE_MAIN,
    JOIN_PATH_TIE_REFUSAL_CEILING,
)
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    JoinCandidateCapExceededError,
    JoinProbeEdgeKindMismatchError,
    RuntimeIntent,
    ScopeClass,
    SelectCol,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._federation_manifest import federation_scaled_join_candidate_cap
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import (
    _candidate_join_paths_for_tables,
    _order_join_candidates_stable,
    classify_scope_candidates,
    collapse_probe_edge_candidate_variation,
    join_hints_multi,
    join_scope_pass1_plan,
    normalize_probe_edges_in_join_path_signature,
)
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from tests.test_join_path_tie_ceiling import _cross_product_ambiguity_schema


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _parallel_mid_tables(mid_count: int) -> SchemaGraph:
    tables: dict[str, TableMetadata] = {
        "src": TableMetadata(
            name="src",
            columns={"id": _col("id"), "amount": _col("amount")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "dst": TableMetadata(name="dst", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
    }
    for index in range(mid_count):
        mid_name = f"mid{index}"
        tables[mid_name] = TableMetadata(
            name=mid_name,
            columns={"id": _col("id"), "src_id": _col("src_id"), "dst_id": _col("dst_id")},
            primary_key=["id"],
            foreign_keys=[
                FKEdge(src_table=mid_name, src_cols=["src_id"], dst_table="src", dst_cols=["id"]),
                FKEdge(src_table=mid_name, src_cols=["dst_id"], dst_table="dst", dst_cols=["id"]),
            ],
        )
    join_paths_multi = recompute_join_paths_multi(tables)
    return SchemaGraph(tables=tables, join_paths_multi=join_paths_multi, effective_structural_hash="anchor_order")


def test_nested_merge_uses_federation_scaled_cross_product_cap() -> None:
    """Inner cartesian merges must honor the passed cap, not the static policy default."""
    schema = _cross_product_ambiguity_schema(3)
    default_cap = PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP
    scaled_cap = federation_scaled_join_candidate_cap(2)
    assert scaled_cap > default_cap
    with pytest.raises(JoinCandidateCapExceededError):
        _candidate_join_paths_for_tables(
            schema,
            ["root", "t2", "t3"],
            cross_product_cap=default_cap,
            tie_cap=JOIN_PATH_TIE_REFUSAL_CEILING,
        )
    paths = _candidate_join_paths_for_tables(
        schema,
        ["root", "t2", "t3"],
        cross_product_cap=scaled_cap,
        tie_cap=JOIN_PATH_TIE_REFUSAL_CEILING,
    )
    assert paths


def test_cross_product_cap_exceeded_refuses_with_session_diagnostic() -> None:
    schema = _cross_product_ambiguity_schema(5)
    cap = 16
    token = set_diagnostic_collector([])
    try:
        with pytest.raises(JoinCandidateCapExceededError) as exc_info:
            join_hints_multi(
                schema,
                ["root", "t2", "t3"],
                cross_product_cap=cap,
                tie_cap=JOIN_PATH_TIE_REFUSAL_CEILING,
            )
        err = exc_info.value
        assert err.enumerated == 25
        assert err.cap == cap
        diags = drain_diagnostic_collector()
        assert any(d.code == DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP for d in diags)
    finally:
        reset_diagnostic_collector(token)


def test_probe_fk_attachment_variants_still_collapse() -> None:
    probe = frozenset({"probe"})
    candidates = [
        {
            "candidate_id": "J01",
            "join_path_signature": ["parent.id->probe.k", "parent.id->child.parent_id"],
            "edge_kinds": ["virtual_fk_bridge", "catalog_fk"],
            "candidate_tier": "base",
        },
        {
            "candidate_id": "J02",
            "join_path_signature": ["parent.id->probe.other_k", "parent.id->child.parent_id"],
            "edge_kinds": ["virtual_fk_bridge", "catalog_fk"],
            "candidate_tier": "base",
        },
    ]
    collapsed = collapse_probe_edge_candidate_variation(candidates, probe)
    assert len(collapsed) == 1


def test_probe_semantic_edge_variants_not_collapsed() -> None:
    probe = frozenset({"probe"})
    candidates = [
        {
            "candidate_id": "J01",
            "join_path_signature": ["parent.other->probe.name", "parent.id->child.parent_id"],
            "edge_kinds": ["semantic_profile", "catalog_fk"],
            "candidate_tier": "extended",
        },
        {
            "candidate_id": "J02",
            "join_path_signature": ["parent.id->probe.id", "parent.id->child.parent_id"],
            "edge_kinds": ["semantic_profile", "catalog_fk"],
            "candidate_tier": "extended",
        },
    ]
    collapsed = collapse_probe_edge_candidate_variation(candidates, probe)
    assert len(collapsed) == 2


def test_normalize_probe_edges_raises_on_kind_length_mismatch() -> None:
    with pytest.raises(JoinProbeEdgeKindMismatchError) as exc_info:
        normalize_probe_edges_in_join_path_signature(
            ["parent.id->probe.k", "parent.id->child.parent_id"],
            ["catalog_fk"],
            frozenset({"probe"}),
            ["parent.id->probe.resolved"],
            ["virtual_fk_bridge"],
        )
    assert exc_info.value.signature_len == 2
    assert exc_info.value.kinds_len == 1


def test_join_candidate_ordering_independent_of_intent_table_order() -> None:
    schema = _parallel_mid_tables(2)
    sum_expr = NormalizedExpr.from_agg("SUM", "src.amount")
    intent_a = RuntimeIntent(
        tables=["src", "dst"],
        grain="row_level",
        select_cols=[SelectCol(expr=sum_expr)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent_b = RuntimeIntent(
        tables=["dst", "src"],
        grain="row_level",
        select_cols=[SelectCol(expr=sum_expr)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    hints_a = join_hints_multi(schema, ["src", "dst"], intent_a, include_semantic=False)
    hints_b = join_hints_multi(schema, ["src", "dst"], intent_b, include_semantic=False)
    keyed_a = [(c["candidate_id"], tuple(c["join_path_signature"])) for c in hints_a["candidates"]]
    keyed_b = [(c["candidate_id"], tuple(c["join_path_signature"])) for c in hints_b["candidates"]]
    assert len(keyed_a) >= 2
    assert keyed_a == keyed_b


def test_order_join_candidates_stable_uses_sorted_anchor() -> None:
    schema = _parallel_mid_tables(2)
    paths = [list(p) for p in schema.join_paths_multi["src"]["dst"]]
    sum_expr = NormalizedExpr.from_agg("SUM", "src.amount")
    intent_a = RuntimeIntent(
        tables=["src", "dst"],
        grain="row_level",
        select_cols=[SelectCol(expr=sum_expr)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent_b = RuntimeIntent(
        tables=["dst", "src"],
        grain="row_level",
        select_cols=[SelectCol(expr=sum_expr)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    ordered_a = _order_join_candidates_stable(paths, schema, intent_a)
    ordered_b = _order_join_candidates_stable(paths, schema, intent_b)
    assert ordered_a != ordered_b or intent_a.tables[0] != intent_b.tables[0]
    assert _order_join_candidates_stable(paths, schema, intent_a) == _order_join_candidates_stable(
        paths, schema, intent_b
    )


def test_single_semantic_candidate_presets_join_choice() -> None:
    candidate = {
        "candidate_id": "J01",
        "join_path_signature": ["left.other->right.id"],
        "edge_kinds": ["semantic_profile"],
        "candidate_tier": "extended",
        "edge_count": 1,
    }
    assert classify_scope_candidates([candidate]) == ScopeClass.semantic_only
    preset, llm_scopes, _, _ = join_scope_pass1_plan(
        main_multi_table=True,
        main_tables=["left", "right"],
        main_candidates=[candidate],
        cte_scopes=[],
        forbid_na=False,
    )
    assert preset[JOIN_CHOICE_SCOPE_MAIN] == "J01"
    assert llm_scopes == []


def test_multiple_semantic_candidates_still_request_join_choice() -> None:
    candidates = [
        {
            "candidate_id": "J01",
            "join_path_signature": ["left_tbl.right_tbl_id->right_tbl.other"],
            "edge_kinds": ["semantic_profile"],
            "candidate_tier": "extended",
            "edge_count": 1,
        },
        {
            "candidate_id": "J02",
            "join_path_signature": ["left_tbl.alt_id->right_tbl.id"],
            "edge_kinds": ["semantic_profile"],
            "candidate_tier": "extended",
            "edge_count": 1,
        },
    ]
    assert classify_scope_candidates(candidates) == ScopeClass.semantic_only
    preset, llm_scopes, _, _ = join_scope_pass1_plan(
        main_multi_table=True,
        main_tables=["left_tbl", "right_tbl"],
        main_candidates=candidates,
        cte_scopes=[],
        forbid_na=False,
    )
    assert preset == {}
    assert llm_scopes
