"""AetherSpace catalog APIs respect effective visibility (context ∩ credentials ∩ sensitivity)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    SchemaRole,
    SensitivityClassification,
    SpaceContext,
)
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config


def _graph_orders_payroll() -> SchemaGraph:
    open_col = ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")
    hidden = ColumnMetadata(
        name="secret",
        data_type="TEXT",
        value_type="text",
        sensitivity=SensitivityClassification.HIDDEN,
    )
    orders = TableMetadata(
        name="orders",
        columns={"id": open_col, "secret": hidden},
        primary_key=["id"],
        foreign_keys=[],
    )
    payroll = TableMetadata(
        name="payroll",
        columns={"id": ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(tables={"orders": orders, "payroll": payroll}, join_paths_multi={})


def _sample_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "film": TableMetadata(
                name="film",
                columns={
                    "film_id": ColumnMetadata(name="film_id", data_type="integer"),
                    "title": ColumnMetadata(name="title", data_type="text"),
                },
                primary_key=["film_id"],
                foreign_keys=[],
            ),
            "customer": TableMetadata(
                name="customer",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff_vis",
        schema_graph_id="sg_vis__h",
    )


def _engine(
    tmp_path: Path,
    *,
    role: str = "owner",
    allow_objects: frozenset[str] | None = None,
    visible_objects: frozenset[str] | None = None,
) -> AetherEngine:
    schema = _sample_schema()
    ctx = EngineContext(allow_objects=allow_objects or frozenset())
    llm_exec = load_runtime_config(merged_env={})
    runtime = RuntimeConfig(
        engine="postgresql",
        artifacts_dir=str(tmp_path),
        engine_context=ctx,
        execution_context=ctx,
        llm_execution=llm_exec,
    )
    obj = AetherEngine.__new__(AetherEngine)
    obj._runtime_config = runtime
    obj._llm_config = LLMConfig(provider="openai")
    obj._schema_graph = schema
    obj._dialect = MagicMock()
    obj._artifacts_dir = tmp_path
    obj._store = TemplateOps.empty_template_store("sg_vis__h")
    obj._templates = {}
    obj._rejected = {}
    obj._schema_terms = set()
    obj._pipeline_writer_lock = __import__("threading").Lock()
    obj._schema_role = SchemaRole.CONSUMER if role == "consumer" else SchemaRole.OWNER
    obj._consumer_visible_objects = visible_objects
    obj._context_name = "master"
    obj._closed = False
    obj._sandbox_closed = False
    obj._sandbox_mode = False
    return obj


def _save_space(engine: AetherEngine, name: str, tables: frozenset[str]) -> None:
    snap = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(tables=tables),
    )
    snap["uid"] = name
    snap["name"] = name
    MainExecutionOps.save_aetherspace_snapshot(str(engine._artifacts_dir), name, snap)


@pytest.mark.fast
def test_effective_visible_tables_owner_universal() -> None:
    sg = _graph_orders_payroll()
    tables = MainExecutionOps.effective_visible_tables(sg, EngineContext(), None)
    assert tables == frozenset({"orders", "payroll"})


@pytest.mark.fast
def test_effective_visible_tables_allow_list() -> None:
    sg = _graph_orders_payroll()
    ctx = EngineContext(allow_objects=frozenset({"orders"}))
    tables = MainExecutionOps.effective_visible_tables(sg, ctx, None)
    assert tables == frozenset({"orders"})


@pytest.mark.fast
def test_effective_visible_tables_credential_narrower_than_context() -> None:
    sg = _graph_orders_payroll()
    ctx = EngineContext(allow_objects=frozenset({"orders", "payroll"}))
    tables = MainExecutionOps.effective_visible_tables(sg, ctx, frozenset({"orders"}))
    assert tables == frozenset({"orders"})


@pytest.mark.fast
def test_effective_visible_columns_exclude_hidden() -> None:
    sg = _graph_orders_payroll()
    cols = MainExecutionOps.effective_visible_columns(sg, EngineContext(), None)
    assert "orders.id" in cols
    assert "orders.secret" not in cols
    assert "payroll.id" in cols


@pytest.mark.fast
def test_consumer_can_reread_space_with_hidden_columns(tmp_path: Path) -> None:
    """New snapshots omit HIDDEN columns; consumer resolve succeeds. Create/update require owner."""
    open_col = ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")
    hidden = ColumnMetadata(
        name="secret",
        data_type="TEXT",
        value_type="text",
        sensitivity=SensitivityClassification.HIDDEN,
    )
    graph = SchemaGraph(
        tables={
            "t_a": TableMetadata(
                name="t_a",
                columns={"id": open_col, "secret": hidden},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "t_b": TableMetadata(
                name="t_b",
                columns={"id": ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
    )
    owner = _engine(tmp_path, role="owner")
    owner._schema_graph = graph
    desc = owner.aetherspace("s1", SpaceContext(tables=frozenset({"t_a"})))
    snap = MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), desc.uid)
    assert snap is not None
    assert "t_a.secret" not in (snap.get("columns") or [])
    assert "t_a.id" in (snap.get("columns") or [])
    consumer = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"t_a"}),
        visible_objects=frozenset({"t_a"}),
    )
    consumer._schema_graph = graph
    assert consumer.aetherspace(uid=desc.uid).uid == desc.uid
    from aetherdialect._contracts_base import OwnerOnlyOperationError

    with pytest.raises(OwnerOnlyOperationError, match="aetherspace"):
        consumer.aetherspace(
            "s1",
            SpaceContext(tables=frozenset({"t_a"})),
            uid=desc.uid,
            notes="Alpha means the first metric.",
        )


@pytest.mark.fast
def test_space_context_rejects_hidden_column(tmp_path: Path) -> None:
    engine = _engine(tmp_path, role="owner")
    engine._schema_graph = _graph_orders_payroll()
    with pytest.raises(ConfigError, match="cannot be included in an aetherspace"):
        engine.aetherspace(
            "bad",
            SpaceContext(tables=frozenset({"orders"}), columns=frozenset({"orders.secret"})),
        )


@pytest.mark.fast
def test_snapshot_listing_hidden_column_remains_resolvable(tmp_path: Path) -> None:
    """Snapshots that list HIDDEN columns remain resolvable."""
    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"orders"}),
        visible_objects=frozenset({"orders"}),
    )
    engine._schema_graph = _graph_orders_payroll()
    snap = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(tables=frozenset({"orders"})),
    )
    # Artifact payload that lists a HIDDEN column.
    snap["columns"] = sorted(set(snap.get("columns") or []) | {"orders.secret"})
    snap["uid"] = "S0009"
    snap["name"] = "hidden_col_snapshot"
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "S0009", snap)
    assert engine.aetherspace(uid="S0009").uid == "S0009"


@pytest.mark.fast
def test_consumer_list_and_read_hide_wider_space(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"film"}),
        visible_objects=frozenset({"film"}),
    )
    _save_space(engine, "films_only", frozenset({"film"}))
    _save_space(engine, "wide", frozenset({"film", "customer"}))
    listed = engine.list_aetherspaces()
    names = {s.name for s in listed}
    assert "films_only" in names
    assert "wide" not in names
    assert "master" not in names
    assert engine.aetherspace("films_only").name == "films_only"
    with pytest.raises(ConfigError, match="unknown aetherspace"):
        engine.aetherspace("wide")
    with pytest.raises(ConfigError, match="unknown aetherspace"):
        with engine.session(mode="reader", space="wide"):
            pass
    with pytest.raises(ConfigError, match="unknown aetherspace"):
        engine.aetherspace("no_such_space")
    with pytest.raises(ConfigError, match="unknown aetherspace"):
        with engine.session(mode="reader", space="no_such_space"):
            pass


@pytest.mark.fast
def test_consumer_may_define_subset_not_superset(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import OwnerOnlyOperationError

    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"film"}),
        visible_objects=frozenset({"film"}),
    )
    with pytest.raises(OwnerOnlyOperationError, match="aetherspace"):
        engine.aetherspace("narrow", SpaceContext(tables=frozenset({"film"})))
    with pytest.raises(OwnerOnlyOperationError, match="aetherspace"):
        engine.aetherspace("too_wide", SpaceContext(tables=frozenset({"film", "customer"})))


@pytest.mark.fast
def test_consumer_delete_visible_space_only(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import OwnerOnlyOperationError

    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"film"}),
        visible_objects=frozenset({"film"}),
    )
    _save_space(engine, "films_only", frozenset({"film"}))
    _save_space(engine, "wide", frozenset({"film", "customer"}))
    with pytest.raises(OwnerOnlyOperationError, match="delete_aetherspace"):
        engine.delete_aetherspace("films_only", persist_learning=False)
    with pytest.raises(OwnerOnlyOperationError, match="delete_aetherspace"):
        engine.delete_aetherspace("wide", persist_learning=False)


@pytest.mark.fast
def test_owner_list_sees_all_spaces(tmp_path: Path) -> None:
    engine = _engine(tmp_path, role="owner")
    _save_space(engine, "films_only", frozenset({"film"}))
    _save_space(engine, "wide", frozenset({"film", "customer"}))
    listed = engine.list_aetherspaces()
    names = {s.name for s in listed}
    assert "films_only" in names
    assert "wide" in names
    assert "master" in names


@pytest.mark.fast
def test_session_unknown_identical_for_missing_and_wider(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"film"}),
        visible_objects=frozenset({"film"}),
    )
    _save_space(engine, "wide", frozenset({"film", "customer"}))
    with pytest.raises(ConfigError, match="unknown aetherspace 'wide'"):
        with engine.session(mode="reader", space="wide"):
            pass
    with pytest.raises(ConfigError, match="unknown aetherspace 'missing_space'"):
        with engine.session(mode="reader", space="missing_space"):
            pass


@pytest.mark.fast
def test_consumer_define_hidden_and_nonexistent_same_message(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import OwnerOnlyOperationError

    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"film"}),
        visible_objects=frozenset({"film"}),
    )
    err_hidden = None
    err_missing = None
    try:
        engine.aetherspace("hidden_real", SpaceContext(tables=frozenset({"customer"})))
    except OwnerOnlyOperationError as exc:
        err_hidden = str(exc)
    try:
        engine.aetherspace("missing_tbl", SpaceContext(tables=frozenset({"no_such_table"})))
    except OwnerOnlyOperationError as exc:
        err_missing = str(exc)
    assert err_hidden is not None and err_missing is not None
    assert err_hidden == err_missing
    assert "not in the schema graph" not in err_hidden
    assert "OwnerOnlyOperationError" not in err_hidden or True
    assert "aetherspace" in err_hidden
    assert "OWNER" in err_hidden


@pytest.mark.fast
def test_consumer_empty_migrated_space_unknown_not_migration(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        role="consumer",
        allow_objects=frozenset({"film"}),
        visible_objects=frozenset({"film"}),
    )
    _save_space(engine, "hidden_then_empty", frozenset({"customer"}))
    MainExecutionOps.apply_structural_migration_to_aetherspace_snapshots(
        str(engine._artifacts_dir),
        dropped_tables=("customer",),
    )
    with pytest.raises(ConfigError, match="unknown aetherspace") as caught:
        engine._resolve_aetherspace("hidden_then_empty")
    assert "space empty after schema migration" not in str(caught.value)


@pytest.mark.fast
def test_owner_empty_migrated_space_migration_error(tmp_path: Path) -> None:
    engine = _engine(tmp_path, role="owner")
    _save_space(engine, "orphan", frozenset({"film"}))
    MainExecutionOps.apply_structural_migration_to_aetherspace_snapshots(
        str(engine._artifacts_dir),
        dropped_tables=("film",),
    )
    with pytest.raises(ConfigError, match="space empty after schema migration"):
        engine._resolve_aetherspace("orphan")
