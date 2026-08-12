"""Join-path tie ceilings and deterministic candidate cross-product capping."""

from __future__ import annotations

import pytest

from aetherdialect import Sandbox
from aetherdialect._config import PolicyConfig
from aetherdialect._constants import (
    DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP,
    DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED,
    JOIN_PATH_TIE_REFUSAL_CEILING,
)
from aetherdialect._contracts_base import StructuralKnowledgeFact, StructuralKnowledgeKind
from aetherdialect._contracts_core import JoinCandidateCapExceededError, JoinPathTieCapExceededError
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import join_path_pair_tie_count, recompute_join_paths_multi
from aetherdialect._sql_gen import (
    _candidate_join_paths_for_tables,
    _join_path_signature_for_path,
    join_hints_multi,
    pin_join_paths_multi,
)
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _parallel_mid_tables(mid_count: int) -> dict[str, TableMetadata]:
    tables: dict[str, TableMetadata] = {
        "src": TableMetadata(name="src", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
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
    return tables


def test_equal_length_paths_beyond_old_tie_cap_are_all_kept() -> None:
    tables = _parallel_mid_tables(5)
    paths = recompute_join_paths_multi(tables)["src"]["dst"]
    assert len(paths) == 5
    assert len(paths) > PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP


def test_pair_beyond_refusal_ceiling_builds_and_refuses_at_query_time() -> None:
    tables = _parallel_mid_tables(JOIN_PATH_TIE_REFUSAL_CEILING + 1)
    join_paths_multi = recompute_join_paths_multi(tables)
    assert join_path_pair_tie_count(join_paths_multi, "src", "dst") == JOIN_PATH_TIE_REFUSAL_CEILING + 1
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=join_paths_multi,
        effective_structural_hash="tie_overflow",
    )
    token = set_diagnostic_collector([])
    try:
        with pytest.raises(JoinPathTieCapExceededError) as exc_info:
            _candidate_join_paths_for_tables(schema, ["src", "dst"])
        err = exc_info.value
        assert {err.source_table, err.target_table} == {"src", "dst"}
        assert err.path_count == JOIN_PATH_TIE_REFUSAL_CEILING + 1
        assert err.ceiling == PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP
        diags = drain_diagnostic_collector()
        assert any(d.code == DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED for d in diags)
    finally:
        reset_diagnostic_collector(token)


def test_pair_within_tie_cap_survives_query_time_enumeration() -> None:
    tables = _parallel_mid_tables(PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP)
    join_paths_multi = recompute_join_paths_multi(tables)
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=join_paths_multi,
        effective_structural_hash="tie_ok",
    )
    paths = _candidate_join_paths_for_tables(schema, ["src", "dst"])
    assert paths


@pytest.mark.fast
def test_declared_pinning_prunes_before_tie_cap_refusal() -> None:
    """A multi-tie pair above the tie cap serves once a declared path is pinned."""
    tables = _parallel_mid_tables(5)
    join_paths_multi = recompute_join_paths_multi(tables)
    assert len(join_paths_multi["src"]["dst"]) > PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=join_paths_multi,
        effective_structural_hash="tie_pin",
    )
    pinned_path = join_paths_multi["src"]["dst"][0]
    declared_sig = _join_path_signature_for_path(pinned_path, schema)
    referenced: set[str] = set()
    for segment in declared_sig:
        left, right = segment.split("->", 1)
        referenced.add(left.strip())
        referenced.add(right.strip())
    fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.JOIN.value,
            text="prefer one bridge",
            referenced_entities=frozenset(referenced),
            payload={"path_signature": declared_sig},
        )
    )
    pin_join_paths_multi(schema, (fact,))
    paths = _candidate_join_paths_for_tables(schema, ["src", "dst"])
    assert len(paths) == 1
    assert _join_path_signature_for_path(paths[0], schema) == declared_sig


def _cross_product_ambiguity_schema(variants_per_leg: int) -> SchemaGraph:
    tables: dict[str, TableMetadata] = {
        "root": TableMetadata(name="root", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        "t2": TableMetadata(name="t2", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        "t3": TableMetadata(name="t3", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
    }
    for index in range(variants_per_leg):
        for target in ("t2", "t3"):
            bridge = f"link_{target}_{index}"
            tables[bridge] = TableMetadata(
                name=bridge,
                columns={"id": _col("id"), "root_id": _col("root_id"), "target_id": _col("target_id")},
                primary_key=["id"],
                foreign_keys=[
                    FKEdge(src_table=bridge, src_cols=["root_id"], dst_table="root", dst_cols=["id"]),
                    FKEdge(src_table=bridge, src_cols=["target_id"], dst_table=target, dst_cols=["id"]),
                ],
            )
    join_paths_multi = recompute_join_paths_multi(tables)
    return SchemaGraph(tables=tables, join_paths_multi=join_paths_multi, effective_structural_hash="cross_product")


def test_merge_paths_cartesian_cap_exceeded_refuses_with_diagnostic() -> None:
    schema = _cross_product_ambiguity_schema(5)
    cap = PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP
    token = set_diagnostic_collector([])
    try:
        with pytest.raises(JoinCandidateCapExceededError) as exc_info:
            _candidate_join_paths_for_tables(
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


def test_cross_product_cap_survivors_are_independent_of_table_insertion_order() -> None:
    schema = _cross_product_ambiguity_schema(2)
    cap = PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP
    ordered_a = _candidate_join_paths_for_tables(
        schema, ["root", "t2", "t3"], cross_product_cap=cap, tie_cap=JOIN_PATH_TIE_REFUSAL_CEILING
    )
    ordered_b = _candidate_join_paths_for_tables(
        schema, ["t3", "root", "t2"], cross_product_cap=cap, tie_cap=JOIN_PATH_TIE_REFUSAL_CEILING
    )
    sigs_a = [tuple(_join_path_signature_for_path(path, schema)) for path in ordered_a]
    sigs_b = [tuple(_join_path_signature_for_path(path, schema)) for path in ordered_b]
    assert sigs_a == sigs_b
    assert len(sigs_a) >= 2


@pytest.mark.needs_corpus
def test_rental_game_supported_language_staff_emits_sixteen_candidates() -> None:
    pytest.importorskip("duckdb")
    from aetherdialect import AetherEngine

    with Sandbox.create_offline_sandbox(AetherEngine) as sb:
        schema = sb.engine._schema_graph
    hints = join_hints_multi(schema, ["game_supported_language", "staff"])
    substantive = [candidate for candidate in hints.get("candidates", []) if candidate.get("join_path_signature")]
    assert len(substantive) == 16
