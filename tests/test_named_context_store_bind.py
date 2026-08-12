"""Named-context and aetherspace session bind the active template-store partition."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import NormalizedExpr, SpaceContext
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect.aetherdialect import AetherEngine


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "film": TableMetadata(
                name="film",
                columns={"film_id": ColumnMetadata(name="film_id", data_type="integer")},
                primary_key=["film_id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff_bind",
        schema_graph_id="graph_bind_test",
    )


def _scalar_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["film"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_session_open_binds_store_to_active_space_partition(tmp_path: Path) -> None:
    schema = _schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    snap = MainSpaceOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainSpaceOps.save_aetherspace_snapshot(str(artifacts_dir), "films_only", snap)
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="films_only",
    )
    templates = dict(TemplateOps.store_to_templates(store))
    TemplateOps.insert_template(
        store,
        templates,
        schema,
        "how many films in space",
        _scalar_intent(),
        "SELECT film_id FROM film",
        dialect=MagicMock(),
    )
    TemplateOps.save_template_store(store)

    engine = AetherEngine.__new__(AetherEngine)
    engine._schema_graph = schema
    engine._artifacts_dir = artifacts_dir
    engine._schema_role = MagicMock()
    engine._schema_role = type("Role", (), {"value": "owner"})()
    engine._consumer_visible_objects = None
    engine._closed = False
    engine._sandbox_closed = False
    engine._context_name = "master"
    engine._templates = {}
    engine._store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    engine._templates_by_space = {}
    engine._store_by_space = {}
    engine._rejected = {}
    engine._pipeline_writer_lock = MagicMock()

    with patch.object(AetherEngine, "_resolve_aetherspace") as resolve:
        space_desc = MagicMock()
        space_desc.uid = "films_only"
        resolve.return_value = (space_desc, frozenset({"film"}), frozenset(), frozenset(), frozenset())
        with patch("aetherdialect.aetherdialect.PipelineSession"):
            engine.session(space="films_only")

    cached = MainSpaceOps.owner_template_store_for_space(engine, "films_only")
    assert cached is not None
    bound_templates = TemplateOps.store_to_templates(cached)
    assert bound_templates
