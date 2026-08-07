"""JSON array containment must use native dialect operators, not substring search."""

from __future__ import annotations

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import (
    EngineIdentity,
    MulGroup,
    NormalizedExpr,
    WhereParam,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import pop_engine_identity, push_engine_identity
from aetherdialect._dialect import DialectRegistry
from aetherdialect._validation_schema import validate_contains_array_filters


def _uninit(engine: str):
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


_JSON_META = ColumnMetadata(name="special_features", data_type="JSON", element_type="string")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("engine", "marker"),
    [
        ("postgresql", "@>"),
        ("mysql", "JSON_CONTAINS"),
        ("mariadb", "JSON_CONTAINS"),
        ("duckdb", "list_contains"),
        ("databricks", "array_contains"),
    ],
)
def test_containment_uses_native_operator_per_dialect(engine: str, marker: str) -> None:
    dialect = _uninit(engine)
    sql = dialect.render_containment("film.special_features", ":p1", "string")
    assert sql is not None
    assert marker in sql
    assert "INSTR" not in sql.upper()
    assert "STRPOS" not in sql.upper()
    assert "CHARINDEX" not in sql.upper()
    assert "LOCATE" not in sql.upper()

    rendered = dialect.render_array_contains(
        "film.special_features",
        "p1",
        column_meta=_JSON_META,
        value_type="string",
    )
    rendered_upper = rendered.upper()
    assert marker.upper() in rendered_upper or (engine == "duckdb" and "ARRAY_CONTAINS" in rendered_upper)
    assert "INSTR" not in rendered_upper
    assert "STRPOS" not in rendered_upper
    assert "CHARINDEX" not in rendered_upper
    assert "LOCATE" not in rendered_upper


@pytest.mark.fast
@pytest.mark.parametrize("engine", ["sqlite", "sqlserver", "redshift", "snowflake", "bigquery"])
def test_containment_refused_where_unsupported(engine: str) -> None:
    dialect = _uninit(engine)
    assert dialect.render_containment("film.special_features", ":p1", "string") is None
    assert DialectRegistry.engine_supports_json_containment(engine) is False

    with pytest.raises(ValueError, match="json containment is not supported"):
        dialect.render_array_contains(
            "film.special_features",
            "p1",
            column_meta=_JSON_META,
            value_type="string",
        )

    schema = SchemaGraph(
        tables={
            "film": TableMetadata(
                name="film",
                columns={"special_features": _JSON_META},
                foreign_keys=[],
                primary_key="",
            )
        },
        join_paths_multi={},
        effective_structural_hash="",
    )
    fp = WhereParam(
        left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["film.special_features"])], sub_groups=[]),
        op="contains",
        value_type="string",
        param_key="p1",
    )
    token = push_engine_identity(EngineIdentity(engine, EngineConfig.RUNTIME))
    try:
        issues = validate_contains_array_filters([fp], schema, {}, "main")
    finally:
        pop_engine_identity(token)
    assert len(issues) == 1
    assert "not supported" in issues[0].message
