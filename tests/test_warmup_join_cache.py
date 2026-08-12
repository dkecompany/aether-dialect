"""Warmup join cache keys must include normalized question hints, not just table sets."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._seed_warmup import SeedWarmupCacheSession
from aetherdialect._utils import normalize_question


def _ambiguous_join_schema() -> SchemaGraph:
    film_fk = FKEdge(src_table="category", src_cols=["category_id"], dst_table="film", dst_cols=["category_id"])
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
            columns={"category_id": ColumnMetadata(name="category_id", data_type="integer", sensitivity="none")},
            primary_key=["category_id"],
            foreign_keys=[film_fk],
        ),
        "language": TableMetadata(
            name="language",
            columns={"language_id": ColumnMetadata(name="language_id", data_type="integer", sensitivity="none")},
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
def test_same_tables_different_hints_not_shared() -> None:
    """Natural-language hints for the same table set must not share one join-cache entry."""
    schema = _ambiguous_join_schema()
    cache: dict = {}
    tables = ["category", "film", "language"]
    hint_a = "films by category name"
    hint_b = "films by spoken language"
    cat_sig = ["category.category_id->film.category_id"]
    lang_sig = ["language.language_id->film.language_id"]
    candidates = {
        "candidates": [
            {"candidate_id": "J00", "join_path_signature": []},
            {"candidate_id": "J01", "join_path_signature": cat_sig},
            {"candidate_id": "J02", "join_path_signature": lang_sig},
        ]
    }
    cmap = {"J00": [], "J01": cat_sig, "J02": lang_sig}

    with (
        patch("aetherdialect._seed_warmup.join_hints_multi", return_value=candidates),
        patch("aetherdialect._seed_warmup.join_candidate_map", return_value=cmap),
        patch("aetherdialect._seed_warmup.get_join_choice_from_llm") as mock_llm,
    ):
        mock_llm.side_effect = [
            {"main": "J01"},
            {"main": "J02"},
        ]
        entry_a = SeedWarmupCacheSession.resolve_joins_for_table_set(
            tables,
            schema,
            hint_a,
            cache,
            hint_is_natural_language=True,
        )
        entry_b = SeedWarmupCacheSession.resolve_joins_for_table_set(
            tables,
            schema,
            hint_b,
            cache,
            hint_is_natural_language=True,
        )

    assert mock_llm.call_count == 2
    assert entry_a[0] == "J01"
    assert entry_b[0] == "J02"
    key_a = (frozenset(tables), normalize_question(hint_a))
    key_b = (frozenset(tables), normalize_question(hint_b))
    assert key_a in cache
    assert key_b in cache
    assert cache[key_a] is not cache[key_b]
