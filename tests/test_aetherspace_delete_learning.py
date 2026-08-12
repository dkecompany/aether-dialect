"""delete_aetherspace learning merge/purge behaviour."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, AetherFederation
from aetherdialect._constants import MASTER_AETHERSPACE_NAME
from aetherdialect._contracts_base import ConfigError, EngineContext, NormalizedExpr, SpaceContext
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps


def _column(name: str, *, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _table(name: str, *, columns: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    cols = columns or {"id": _column("id")}
    return TableMetadata(name=name, columns=cols, primary_key=["id"], foreign_keys=[])


def _sample_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "film": _table(
                "film",
                columns={
                    "film_id": _column("film_id"),
                    "title": _column("title", data_type="text"),
                },
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff_del_learn",
        schema_graph_id="graph_del_learn",
    )


def _scalar_intent(table: str, col: str) -> RuntimeIntent:
    return RuntimeIntent(
        tables=[table],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{col}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _persist_space_template(
    artifacts_dir: Path,
    schema: SchemaGraph,
    space: str,
    question: str,
    *,
    col: str,
    sql: str,
) -> None:
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name=space,
    )
    templates = dict(TemplateOps.store_to_templates(store))
    TemplateOps.insert_template(
        store,
        templates,
        schema,
        question,
        _scalar_intent("film", col),
        sql,
        dialect=MagicMock(),
    )
    TemplateOps.save_template_store(store)


def _load_templates(artifacts_dir: Path, schema: SchemaGraph, space: str = MASTER_AETHERSPACE_NAME) -> dict:
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name=space,
    )
    return dict(TemplateOps.store_to_templates(store))


def _save_named_space(artifacts_dir: Path, schema: SchemaGraph, name: str) -> None:
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainExecutionOps.save_aetherspace_snapshot(str(artifacts_dir), name, snap)


def _space_store_dir(artifacts_dir: Path, space: str) -> Path:
    return artifacts_dir / "intent_templates" / "spaces" / space


@pytest.mark.fast
def test_delete_persist_learning_promotes_space_only_template(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    schema = _sample_schema()
    _save_named_space(artifacts_dir, schema, "films_only")
    _persist_space_template(
        artifacts_dir,
        schema,
        "films_only",
        "how many films in space",
        col="film_id",
        sql="SELECT film_id FROM film",
    )
    assert _load_templates(artifacts_dir, schema, "master") == {}

    result = MainExecutionOps.delete_aetherspace(
        str(artifacts_dir),
        "films_only",
        persist_learning=True,
        schema_graph=schema,
    )
    assert result.deleted is True
    assert result.merge_counts.get("carried_new_id", 0) >= 1

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    assert TemplateOps.resolve_template_for_question("how many films in space", master_templates) is not None
    assert not _space_store_dir(artifacts_dir, "films_only").exists()
    assert MainExecutionOps.load_aetherspace_snapshot(str(artifacts_dir), "films_only") is None


@pytest.mark.fast
def test_delete_persist_learning_master_wins_on_duplicate_question(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    schema = _sample_schema()
    _save_named_space(artifacts_dir, schema, "films_only")
    _persist_space_template(
        artifacts_dir,
        schema,
        "master",
        "how many films",
        col="film_id",
        sql="SELECT film_id FROM film",
    )
    _persist_space_template(
        artifacts_dir,
        schema,
        "films_only",
        "how many films",
        col="title",
        sql="SELECT title FROM film",
    )

    result = MainExecutionOps.delete_aetherspace(
        str(artifacts_dir),
        "films_only",
        persist_learning=True,
        schema_graph=schema,
    )
    assert result.merge_counts.get("discarded_same_q_diff_intent", 0) == 1

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    resolved = TemplateOps.resolve_template_for_question("how many films", master_templates)
    assert resolved is not None
    tmpl, _idx = resolved
    assert "film_id" in (tmpl.sql_param or "")


@pytest.mark.fast
def test_delete_returns_empty_merge_counts_when_persist_learning_false(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    schema = _sample_schema()
    _save_named_space(artifacts_dir, schema, "films_only")
    result = MainExecutionOps.delete_aetherspace(
        str(artifacts_dir),
        "films_only",
        persist_learning=False,
        schema_graph=schema,
    )
    assert result.deleted is True
    assert result.merge_counts == {}


@pytest.mark.fast
def test_delete_persist_learning_false_discards_space_templates(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    schema = _sample_schema()
    _save_named_space(artifacts_dir, schema, "films_only")
    _persist_space_template(
        artifacts_dir,
        schema,
        "films_only",
        "space only question",
        col="film_id",
        sql="SELECT film_id FROM film",
    )

    MainExecutionOps.delete_aetherspace(
        str(artifacts_dir),
        "films_only",
        persist_learning=False,
        schema_graph=schema,
    )

    assert _load_templates(artifacts_dir, schema, "master") == {}
    assert not _space_store_dir(artifacts_dir, "films_only").exists()
    assert MainExecutionOps.load_aetherspace_snapshot(str(artifacts_dir), "films_only") is None


@pytest.mark.fast
def test_delete_unknown_and_master_raise(tmp_path: Path) -> None:
    schema = _sample_schema()
    with pytest.raises(ConfigError, match="unknown aetherspace"):
        MainExecutionOps.delete_aetherspace(
            str(tmp_path),
            "missing",
            schema_graph=schema,
        )
    with pytest.raises(ConfigError, match="cannot be deleted"):
        MainExecutionOps.delete_aetherspace(
            str(tmp_path),
            "master",
            schema_graph=schema,
        )


@pytest.mark.fast
def test_engine_and_federation_delete_signatures_match() -> None:
    engine_sig = inspect.signature(AetherEngine.delete_aetherspace)
    fed_sig = inspect.signature(AetherFederation.delete_aetherspace)
    assert engine_sig == fed_sig
    assert "persist_learning" in engine_sig.parameters
    assert engine_sig.parameters["persist_learning"].default is True


@pytest.mark.fast
def test_engine_delete_delegates_to_helper(tmp_path: Path) -> None:
    schema = _sample_schema()
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "films_only", snap)
    with patch("aetherdialect.aetherdialect.delete_aetherspace", return_value=True) as delete_mock:
        engine = AetherEngine.__new__(AetherEngine)
        engine._artifacts_dir = tmp_path
        engine._schema_graph = schema
        engine._schema_role = "owner"
        engine._context_name = "master"
        engine._runtime_config = SimpleNamespace(execution_context=None, engine_context=EngineContext())
        engine._consumer_visible_objects = None
        engine._pipeline_writer_lock = __import__("threading").Lock()
        assert engine.delete_aetherspace("films_only", persist_learning=False) is True
        delete_mock.assert_called_once()
        _args, kwargs = delete_mock.call_args
        assert kwargs["persist_learning"] is False
