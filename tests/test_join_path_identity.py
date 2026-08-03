"""Join path identity is order-insensitive within a layer and matches emission order."""

import pytest

from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import _resolve_joins_fresh
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

    with patch("aetherdialect._pipeline.get_join_choice_from_llm", return_value={"main": "J01"}):
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
