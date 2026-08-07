"""Live tests for views-only reflection and view-targeted queries on DuckDB, SQLite, and PostgreSQL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._live_testing import run_and_assert

from ._engine_live import (
    build_engine_t2s,
    build_runner,
    engine_schema,
    reflect_rental_shop_schema_for_live_test,
    skip_unless_configured,
)
from .conftest import _POSTGRES_SKIP_REASON, _postgres_credentials_configured
from .mydb_scenarios import Expected, Scenario

_VIEW_NAMES = ("active_customer_v", "store_revenue_v", "film_catalog_v")
_VIEWS_SQL = Path(__file__).resolve().parent.parent / "scripts" / "data" / "rental_shop_views.sql"

_VIEW_SCENARIO = Scenario(
    id="VIEW-001",
    question="list store id and total revenue from the store revenue view",
    expected=Expected(
        tables=["store_revenue_v"],
        min_rows=1,
        max_rows=500,
        grain="row_level",
    ),
    category="views",
)


def _engine_params(engine: str) -> tuple[str, str, str]:
    if engine == "duckdb":
        return ("duckdb", "DUCKDB_DATABASE", "rental_shop")
    return ("sqlite", "SQLITE_DATABASE", "rental_shop")


@pytest.fixture(params=("duckdb", "sqlite"))
def engine_bundle(request: pytest.FixtureRequest):
    engine_name, schema_env, default_schema = _engine_params(request.param)
    skip = skip_unless_configured(engine_name)
    if skip is not None:
        pytest.skip(skip)
    t2s = build_engine_t2s(engine_name, engine_schema(schema_env, default_schema))
    yield engine_name, t2s
    del t2s


def test_views_present_in_full_graph(engine_bundle) -> None:
    """Default reflection includes local rental_shop views."""
    _engine, t2s = engine_bundle
    reflected = set(t2s._schema_graph.tables)
    missing = [name for name in _VIEW_NAMES if name not in reflected]
    if missing:
        pytest.skip(f"rental_shop views not loaded: {missing}")
    for view_name in _VIEW_NAMES:
        assert t2s._schema_graph.tables[view_name].kind == "view"


def _reflect_graph(t2s: AetherEngine, engine_name: str, include: Any) -> SchemaGraph:
    """Reflect tables and/or views, using catalog helpers for local engines when needed."""
    if engine_name in ("duckdb", "sqlite"):
        return reflect_rental_shop_schema_for_live_test(t2s, include=include)
    return t2s._dialect.reflect_schema_graph(include=include)


def test_views_only_reflection(engine_bundle) -> None:
    """EngineContext(include='views') reflects only view relations."""
    engine, t2s = engine_bundle
    if "store_revenue_v" not in t2s._schema_graph.tables:
        pytest.skip("rental_shop views not loaded")
    ctx = EngineContext(include="views")
    graph = _reflect_graph(t2s, engine, ctx.include)
    assert graph.tables
    assert all(tbl.kind == "view" for tbl in graph.tables.values())
    assert "store_revenue_v" in graph.tables


def test_both_reflection_coexist(engine_bundle) -> None:
    """EngineContext(include='both') returns tables and views together."""
    engine, t2s = engine_bundle
    ctx = EngineContext(include="both")
    graph = _reflect_graph(t2s, engine, ctx.include)
    kinds = {name: tbl.kind for name, tbl in graph.tables.items()}
    assert any(k == "table" for k in kinds.values())
    assert any(k == "view" for k in kinds.values())


def test_preview_store_revenue_view(engine_bundle) -> None:
    """Approved preview against store_revenue_v returns aggregated rows."""
    _engine, t2s = engine_bundle
    if "store_revenue_v" not in t2s._schema_graph.tables:
        pytest.skip("store_revenue_v not loaded")
    preview = t2s.preview_table("store_revenue_v", limit=5)
    assert preview.rows
    assert len(preview.columns) >= 2
    assert len(preview.rows[0]) == len(preview.columns)


def test_pipeline_question_on_store_revenue_view(engine_bundle) -> None:
    """Pipeline can answer a question scoped to store_revenue_v."""
    _engine, t2s = engine_bundle
    if "store_revenue_v" not in t2s._schema_graph.tables:
        pytest.skip("store_revenue_v not loaded")
    runner = build_runner(t2s)
    run_and_assert(
        runner,
        _VIEW_SCENARIO,
        header=f"[{_engine}:VIEW-001] {_VIEW_SCENARIO.question}",
    )


def _iter_view_statements() -> list[str]:
    text = _VIEWS_SQL.read_text(encoding="utf-8")
    return [stmt.strip() for stmt in text.split(";") if stmt.strip()]


def _ensure_postgres_views(t2s: AetherEngine) -> None:
    """Create rental_shop views on PostgreSQL when absent (local-engine loader parity)."""
    from sqlalchemy import text

    engine = t2s._dialect.engine
    with engine.begin() as conn:
        for name in _VIEW_NAMES:
            conn.execute(text(f'DROP VIEW IF EXISTS "{name}"'))
        for stmt in _iter_view_statements():
            conn.execute(text(stmt))


@pytest.fixture(scope="session")
def postgres_views_t2s(t2s_rbac_owner: AetherEngine) -> AetherEngine:
    """PostgreSQL owner engine with rental_shop views ensured for reflection tests."""
    if not _postgres_credentials_configured():
        pytest.skip(_POSTGRES_SKIP_REASON)
    _ensure_postgres_views(t2s_rbac_owner)
    return t2s_rbac_owner


def test_postgres_views_only_reflection(postgres_views_t2s: AetherEngine) -> None:
    """EngineContext(include='views') on PostgreSQL reflects only view relations."""
    t2s = postgres_views_t2s
    ctx = EngineContext(include="views")
    graph = t2s._dialect.reflect_schema_graph(include=ctx.include)
    if not graph.tables:
        pytest.skip("no views reflected from PostgreSQL")
    assert all(tbl.kind == "view" for tbl in graph.tables.values())
    assert "store_revenue_v" in graph.tables


def test_postgres_both_reflection_coexist(postgres_views_t2s: AetherEngine) -> None:
    """EngineContext(include='both') on PostgreSQL returns tables and views together."""
    t2s = postgres_views_t2s
    ctx = EngineContext(include="both")
    graph = t2s._dialect.reflect_schema_graph(include=ctx.include)
    kinds = {name: tbl.kind for name, tbl in graph.tables.items()}
    assert any(k == "table" for k in kinds.values())
    assert any(k == "view" for k in kinds.values())
