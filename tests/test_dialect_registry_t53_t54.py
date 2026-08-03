"""Snowflake unnest semantics and dialect registry surface derivation."""

from __future__ import annotations

import pytest

from aetherdialect._constants import (
    ARRAY_CONTAINS_EXCLUDED_ENGINES,
    CANONICAL_ENGINE_ORDER,
    EMBEDDED_ENGINE_NAMES,
    NATIVE_BACKEND_ENGINES,
    QUALIFIED_TABLE_REF_ENGINES,
    STATISTICAL_AGG_EXCLUDED_ENGINES,
    STRUCTURAL_INDEX_ENGINES,
    TOML_ENGINE_FIELD_MAPS,
    WINDOW_FRAMES_EXCLUDED_ENGINES,
)
from aetherdialect._contracts_core import SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import derive_dialect_registry_surfaces, get_registered_engines, get_runtime_config_class
from aetherdialect._dialect_sqlglot_engines import SnowflakeDialect
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._sql_gen import _maybe_render_array_unnest_select


def _snowflake_uninit() -> SnowflakeDialect:
    return SnowflakeDialect.__new__(SnowflakeDialect)


@pytest.mark.fast
def test_snowflake_rejects_unnest_in_select_list() -> None:
    dialect = _snowflake_uninit()
    assert dialect.supports_unnest_select_item is False
    assert dialect.import_unnest_policy() == "from_only"


@pytest.mark.fast
def test_snowflake_skips_select_list_unnest_expansion_for_cte_arrays() -> None:
    meta = ColumnMetadata(name="tags", data_type="array", element_type="text", sensitivity="none")
    sc = SelectCol(expr=NormalizedExpr.from_column("orders.tags"))
    schema = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"tags": meta},
                primary_key=[],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )
    out = _maybe_render_array_unnest_select(
        sc,
        schema,
        {},
        _snowflake_uninit(),
        ["tag"],
        0,
        for_cte=True,
    )
    assert out is None


@pytest.mark.fast
def test_derived_registry_surfaces_match_runtime_constants() -> None:
    derived = derive_dialect_registry_surfaces()
    assert derived["canonical_engine_order"] == CANONICAL_ENGINE_ORDER
    assert derived["native_backend_engines"] == NATIVE_BACKEND_ENGINES
    assert derived["embedded_engine_names"] == EMBEDDED_ENGINE_NAMES
    assert derived["structural_index_engines"] == STRUCTURAL_INDEX_ENGINES
    assert derived["qualified_table_ref_engines"] == QUALIFIED_TABLE_REF_ENGINES
    assert derived["statistical_agg_excluded_engines"] == STATISTICAL_AGG_EXCLUDED_ENGINES
    assert derived["window_frames_excluded_engines"] == WINDOW_FRAMES_EXCLUDED_ENGINES
    assert derived["array_contains_excluded_engines"] == ARRAY_CONTAINS_EXCLUDED_ENGINES


@pytest.mark.fast
def test_every_registered_dialect_has_runtime_config_and_toml_fields() -> None:
    derived = derive_dialect_registry_surfaces()
    for engine in get_registered_engines():
        get_runtime_config_class(engine)
        toml_section = next(
            (section for section in derived["toml_field_map_engines"] if TOML_ENGINE_FIELD_MAPS.get(section)),
            None,
        )
        assert toml_section is not None, f"registered engine {engine!r} missing TOML field map"
