"""View description projection helpers for sandbox graph reuse."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import TableKind
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_reflect import project_base_descriptions_onto_views


def _graph_with_film() -> SchemaGraph:
    film = TableMetadata(
        name="film",
        kind=TableKind.TABLE,
        columns={
            "film_id": ColumnMetadata(name="film_id", data_type="INTEGER"),
            "title": ColumnMetadata(name="title", data_type="TEXT"),
        },
        primary_key=["film_id"],
        foreign_keys=[],
    )
    film.base_description = "Catalog film row"
    film.description = "Catalog film row"
    film.columns["film_id"].base_description = "Primary key for film"
    film.columns["film_id"].description = "Primary key for film"
    film.columns["title"].base_description = "Film title"
    film.columns["title"].description = "Film title"
    return SchemaGraph(tables={"film": film}, join_paths_multi={})


@pytest.mark.fast
def test_project_base_descriptions_onto_views_by_column_name() -> None:
    tables = _graph_with_film()
    view = TableMetadata(
        name="film_catalog_v",
        kind=TableKind.VIEW,
        columns={
            "film_id": ColumnMetadata(name="film_id", data_type="INTEGER"),
            "title": ColumnMetadata(name="title", data_type="TEXT"),
        },
        primary_key=[],
        foreign_keys=[],
        view_definition="SELECT film_id, title FROM film",
    )
    views = SchemaGraph(tables={"film_catalog_v": view}, join_paths_multi={})
    projected = project_base_descriptions_onto_views(tables, views)
    assert projected >= 2
    assert views.tables["film_catalog_v"].columns["title"].base_description == "Film title"
    assert views.tables["film_catalog_v"].columns["film_id"].base_description == "Primary key for film"
