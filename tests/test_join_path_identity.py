"""Join path identity is order-insensitive within a layer and matches emission order."""

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, RuntimeIntent, SeedWarmupIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._intent_bind import (
    join_path_key_concrete,
    join_path_key_runtime,
    join_path_segments_fingerprint_concrete,
    join_path_segments_fingerprint_runtime,
)
from aetherdialect._pipeline_generate import _resolve_joins_fresh
from aetherdialect._sql_gen import (
    _canonicalize_join_sig_segments,
    _orient_join_sig_for_from,
    canonicalize_stored_join_path_signature,
)


def _film_star_schema() -> SchemaGraph:
    category_fk = FKEdge(src_table="category", src_cols=["category_id"], dst_table="film", dst_cols=["category_id"])
    language_fk = FKEdge(src_table="language", src_cols=["language_id"], dst_table="film", dst_cols=["language_id"])
    category_edge = {
        "src_table": "category",
        "src_cols": ["category_id"],
        "dst_table": "film",
        "dst_cols": ["category_id"],
    }
    language_edge = {
        "src_table": "language",
        "src_cols": ["language_id"],
        "dst_table": "film",
        "dst_cols": ["language_id"],
    }
    tables = {
        "film": TableMetadata(
            name="film",
            columns={
                "film_id": ColumnMetadata(name="film_id", data_type="integer", sensitivity="none"),
                "category_id": ColumnMetadata(name="category_id", data_type="integer", sensitivity="none"),
                "language_id": ColumnMetadata(name="language_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["film_id"],
            foreign_keys=[],
        ),
        "category": TableMetadata(
            name="category",
            columns={
                "category_id": ColumnMetadata(name="category_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["category_id"],
            foreign_keys=[category_fk],
        ),
        "language": TableMetadata(
            name="language",
            columns={
                "language_id": ColumnMetadata(name="language_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["language_id"],
            foreign_keys=[language_fk],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi={
            "category": {"film": [[category_edge]]},
            "language": {"film": [[language_edge]]},
            "film": {
                "category": [[category_edge]],
                "language": [[language_edge]],
            },
        },
        effective_structural_hash="h",
    )


@pytest.mark.fast
def test_star_topology_signature_is_canonicalized_for_storage() -> None:
    signature = ["category.category_id->film.category_id", "language.language_id->film.language_id"]
    stored = canonicalize_stored_join_path_signature(signature)
    assert stored == sorted(signature, key=lambda s: s.strip().lower())


@pytest.mark.fast
def test_linear_topology_signature_order_is_preserved() -> None:
    signature = ["a.x->b.x", "b.y->c.y"]
    stored = canonicalize_stored_join_path_signature(signature)
    assert stored == signature


@pytest.mark.fast
def test_stored_signature_matches_oriented_emission_for_hub_on_right_star() -> None:
    """When FROM anchor is the hub, stored signature must match oriented emission form."""
    raw_sig = [
        "category.category_id->film.category_id",
        "language.language_id->film.language_id",
    ]
    from_anchor = "film"
    expected_emitted = _canonicalize_join_sig_segments(_orient_join_sig_for_from(raw_sig, from_anchor))
    stored = canonicalize_stored_join_path_signature(raw_sig, from_anchor=from_anchor)
    assert stored == expected_emitted
    assert stored == [
        "film.category_id->category.category_id",
        "film.language_id->language.language_id",
    ]


@pytest.mark.fast
def test_resolve_joins_fresh_stores_oriented_signature_for_hub_on_right_star() -> None:
    """Pipeline storage must match the oriented form used during SQL emission."""
    from unittest.mock import patch

    from aetherdialect._dialect_postgres import PostgresDialect

    schema = _film_star_schema()
    raw_sig = [
        "category.category_id->film.category_id",
        "language.language_id->film.language_id",
    ]
    intent = RuntimeIntent(
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    join_candidates = {
        "candidates": [
            {"candidate_id": "J00", "join_path_signature": []},
            {"candidate_id": "J01", "join_path_signature": raw_sig, "edge_kinds": ["catalog_fk", "catalog_fk"]},
        ],
    }
    cmap = {"J00": [], "J01": raw_sig}
    det = "SELECT film.film_id FROM film\nWHERE 1=1"
    dialect = PostgresDialect.__new__(PostgresDialect)

    with patch("aetherdialect._sql_gen.get_join_choice_from_llm", return_value={"main": "J01"}):
        _resolve_joins_fresh(
            det,
            intent,
            cmap,
            None,
            "list films with category and language",
            join_candidates,
            schema=schema,
            dialect=dialect,
        )

    expected = canonicalize_stored_join_path_signature(raw_sig, from_anchor="film")
    assert intent.chosen_join_path_signature == expected


@pytest.mark.fast
def test_join_path_key_orients_hub_on_right_from_anchor() -> None:
    """Fingerprint keys must orient segments from the FROM anchor, not sort hub-on-right."""
    raw_sig = [
        "category.category_id->film.category_id",
        "language.language_id->film.language_id",
    ]
    oriented_sig = canonicalize_stored_join_path_signature(raw_sig, from_anchor="film")
    rt_raw = RuntimeIntent(
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=raw_sig,
    )
    rt_oriented = RuntimeIntent(
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=oriented_sig,
    )
    assert join_path_key_runtime(rt_raw) == join_path_key_runtime(rt_oriented)


@pytest.mark.fast
def test_join_path_segments_fingerprint_orients_hub_on_right_from_anchor() -> None:
    raw_sig = [
        "category.category_id->film.category_id",
        "language.language_id->film.language_id",
    ]
    oriented_sig = canonicalize_stored_join_path_signature(raw_sig, from_anchor="film")
    rt_raw = RuntimeIntent(
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=raw_sig,
    )
    rt_oriented = RuntimeIntent(
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=oriented_sig,
    )
    assert join_path_segments_fingerprint_runtime(rt_raw) == join_path_segments_fingerprint_runtime(rt_oriented)

    conc_raw = ConcreteIntent(
        intent_id="raw",
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=raw_sig,
    )
    conc_oriented = ConcreteIntent(
        intent_id="oriented",
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=oriented_sig,
    )
    assert join_path_segments_fingerprint_concrete(conc_raw) == join_path_segments_fingerprint_concrete(conc_oriented)
    assert join_path_key_concrete(conc_raw) == join_path_key_concrete(conc_oriented)


@pytest.mark.fast
def test_warmup_stores_oriented_signature_for_hub_on_right_star(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Seed warmup must store join signatures oriented from the runtime FROM anchor."""
    from unittest.mock import MagicMock

    from aetherdialect._dialect_postgres import PostgresDialect
    from aetherdialect._seed_warmup import SeedWarmupCacheSession

    schema = _film_star_schema()
    raw_sig = [
        "category.category_id->film.category_id",
        "language.language_id->film.language_id",
    ]

    def fake_resolve(_tables, _schema, _iid, _cache, **_kw):
        return "J01", raw_sig, {"candidates": []}

    monkeypatch.setattr("aetherdialect._seed_warmup.SeedWarmupCacheSession.resolve_joins_for_table_set", fake_resolve)
    monkeypatch.setattr("aetherdialect._seed_warmup.validate_sql", lambda *_a, **_kw: (True, None, None, []))
    monkeypatch.setattr(
        "aetherdialect._seed_warmup.SeedWarmupCacheSession.instantiate_intent",
        lambda _i, _vd: MagicMock(param_values={}),
    )
    monkeypatch.setattr(
        "aetherdialect._seed_warmup.apply_runtime_post_processing_lite",
        lambda rt, _schema, **_kw: (rt, []),
    )
    monkeypatch.setattr("aetherdialect._seed_warmup.curated_warmup_semantic_issues", lambda *_a, **_kw: [])
    monkeypatch.setattr("aetherdialect._seed_warmup.prune_unused_cte_steps", lambda rt: rt)
    monkeypatch.setattr(
        "aetherdialect._seed_warmup.build_deterministic_sql",
        lambda *_a, **_kw: "SELECT film.film_id FROM film",
    )
    monkeypatch.setattr(
        "aetherdialect._seed_warmup.inject_join_into_deterministic_sql",
        lambda sql, *_a, **_kw: sql,
    )
    monkeypatch.setattr(
        "aetherdialect._seed_warmup.finalize_substitute_sql",
        lambda *_a, **_kw: "SELECT film.film_id FROM film",
    )

    dialect = PostgresDialect.__new__(PostgresDialect)
    dialect.finalize_render = lambda sql, _params, **_kw: sql
    dialect.execute = lambda _sql, _params=None: []
    dialect.can_explain = lambda: False

    si = SeedWarmupIntent(
        intent_id="orient_join",
        tables=["film", "category", "language"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    sess = SeedWarmupCacheSession(manifest={}, work_units={})
    results, _, _, _ = SeedWarmupCacheSession.run_seed_warmup_execution(
        [si],
        schema,
        dialect,
        1,
        join_cache={},
        warmup_cache=sess,
        warmup_report_version=1,
        persist_template_learning=False,
        warmup_lattice_root=str(tmp_path),
    )
    assert results[0].execute_ok
    expected = canonicalize_stored_join_path_signature(raw_sig, from_anchor="film")
    stored = results[0].intent.chosen_join_path_signature
    assert stored == expected


@pytest.mark.fast
def test_join_path_key_differs_for_edge_kinds_with_same_segments() -> None:
    """FK and value-overlap joins with identical segments must not share a join_path_key."""
    from dataclasses import replace

    base = ConcreteIntent(
        intent_id="id",
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["child.parent_id->parent.id"],
        chosen_join_candidate_id="J01",
    )
    fk = replace(base)
    object.__setattr__(fk, "chosen_join_edge_kinds", ["catalog_fk"])
    overlap = replace(base)
    object.__setattr__(overlap, "chosen_join_edge_kinds", ["semantic_profile"])
    assert join_path_key_concrete(fk) != join_path_key_concrete(overlap)

    rt_fk = RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["child.parent_id->parent.id"],
        chosen_join_candidate_id="J01",
    )
    object.__setattr__(rt_fk, "chosen_join_edge_kinds", ["catalog_fk"])
    rt_overlap = replace(
        rt_fk,
        chosen_join_path_signature=["child.parent_id->parent.id"],
    )
    object.__setattr__(rt_overlap, "chosen_join_edge_kinds", ["semantic_profile"])
    assert join_path_key_runtime(rt_fk) != join_path_key_runtime(rt_overlap)
