"""Tests for SQL-history and query-log seed warmup entrypoint kwargs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import SeedWarmupConfig
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    SeedWarmupIntent,
    SelectCol,
)
from aetherdialect._main_execution import (
    _run_seed_warmup_sql_history_pipeline,
    run_seed_warmup_from_history_execution,
    run_seed_warmup_from_query_log_execution,
)


def _intent(**overrides: object) -> SeedWarmupIntent:
    defaults: dict[str, object] = dict(
        intent_id="sqlhist_0",
        tables=["t1"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        param_values={},
        source="sql_history",
    )
    defaults.update(overrides)
    return SeedWarmupIntent(**defaults)


def _minimal_engine_shell() -> AetherEngine:
    obj = AetherEngine.__new__(AetherEngine)
    obj._schema_graph = MagicMock()
    obj._dialect = MagicMock()
    obj._artifacts_dir = "/tmp/sqlhist_warmup"
    obj._store = {"next_id": 1}
    obj._templates = {}
    return obj


@pytest.mark.parametrize(
    ("entry_fn", "extra_kwargs"),
    [
        (run_seed_warmup_from_history_execution, {"sql_history_filepath": "/tmp/hist.sql"}),
        (
            run_seed_warmup_from_query_log_execution,
            {"lookback_days": 30, "max_queries": 10},
        ),
    ],
)
def test_sql_warmup_entrypoints_thread_expand_and_max_kept_intents(
    entry_fn,
    extra_kwargs: dict[str, object],
    tmp_path,
) -> None:
    """``expand`` and ``max_kept_intents`` reach the SQL-history pipeline."""
    engine = _minimal_engine_shell()
    engine._artifacts_dir = str(tmp_path)
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch(
            "aetherdialect._main_execution.load_sql_history_statements",
            return_value=["SELECT 1"],
        ),
        patch(
            "aetherdialect._main_execution.compute_sql_history_content_hash",
            return_value="abc",
        ),
        patch(
            "aetherdialect._main_execution.fetch_query_log",
            return_value=["SELECT 1"],
        ),
        patch(
            "aetherdialect._main_execution._raw_db_connection_for_query_log",
            return_value=MagicMock(),
        ),
        patch(
            "aetherdialect._main_execution._dialect_name_for_query_log",
            return_value="postgresql",
        ),
        patch(
            "aetherdialect._main_execution._run_seed_warmup_sql_history_pipeline",
            side_effect=_capture,
        ),
    ):
        if entry_fn is run_seed_warmup_from_history_execution:
            entry_fn(engine, str(tmp_path / "hist.sql"), expand=True, max_kept_intents=None)
        else:
            entry_fn(engine, expand=True, max_kept_intents=None, **extra_kwargs)

    assert captured.get("expand") is True
    assert captured.get("max_kept_intents") is None


def test_sql_history_pipeline_expand_builds_larger_queue(monkeypatch, tmp_path) -> None:
    """``expand=True`` runs expansion and increases the warmup queue beyond converted rows."""
    schema = MagicMock()
    schema.ensure_schema_stats.return_value = MagicMock()
    dialect = MagicMock()
    base_intent = _intent()
    synthetic = _intent(intent_id="syn_1", source="synthetic")

    monkeypatch.setattr(
        "aetherdialect._main_execution.convert_sql_to_intent",
        lambda *_a, **_k: MagicMock(intent=MagicMock(), failure_code=None, sql_hash="h1"),
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.dedup_runtime_intents",
        lambda runtimes: runtimes,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.seed_warmup_intent_from_runtime_intent",
        lambda *_a, **_k: base_intent,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.compute_schema_limits",
        lambda _s: MagicMock(max_filters=3, max_groupby=2, max_tables=4),
    )
    expand_called: list[bool] = []

    def _fake_expand(*_a, **_k):
        expand_called.append(True)
        return [synthetic]

    monkeypatch.setattr(
        "aetherdialect._main_execution.expand_gold_intents",
        _fake_expand,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.resolve_joins_for_table_set",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.open_seed_warmup_cache_session",
        lambda *_a, **_k: MagicMock(
            manifest={},
            work_units=[],
            touched_work_unit_ids=[],
        ),
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.save_seed_warmup_cache_zip",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.save_seed_warmup_report",
        lambda *_a, **_k: None,
    )

    seen_queues: list[list[SeedWarmupIntent]] = []

    def _fake_exec(queue, *_a, **_k):
        seen_queues.append(list(queue))
        return [], [], 1, {"execute_ok_count": 0}

    monkeypatch.setattr(
        "aetherdialect._main_execution.run_seed_warmup_execution",
        _fake_exec,
    )

    _run_seed_warmup_sql_history_pipeline(
        schema=schema,
        dialect=dialect,
        output_dir=str(tmp_path),
        store=None,
        templates={},
        sql_texts=["SELECT 1"],
        sql_history_content_hash="hash",
        seed=SeedWarmupConfig.RANDOM_SEED,
        expand=True,
        max_kept_intents=None,
    )

    assert len(seen_queues) == 1
    assert expand_called
    assert len(seen_queues[0]) >= 1


def test_sql_history_pipeline_passes_max_kept_intents_to_execution(monkeypatch, tmp_path) -> None:
    """``max_kept_intents=None`` is forwarded to ``run_seed_warmup_execution``."""
    schema = MagicMock()
    schema.ensure_schema_stats.return_value = MagicMock()
    dialect = MagicMock()
    base_intent = _intent()

    monkeypatch.setattr(
        "aetherdialect._main_execution.convert_sql_to_intent",
        lambda *_a, **_k: MagicMock(intent=MagicMock(), failure_code=None, sql_hash="h1"),
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.dedup_runtime_intents",
        lambda runtimes: runtimes,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.seed_warmup_intent_from_runtime_intent",
        lambda *_a, **_k: base_intent,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.compute_schema_limits",
        lambda _s: MagicMock(max_filters=3, max_groupby=2, max_tables=4),
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.resolve_joins_for_table_set",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.open_seed_warmup_cache_session",
        lambda *_a, **_k: MagicMock(
            manifest={},
            work_units=[],
            touched_work_unit_ids=[],
        ),
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.save_seed_warmup_cache_zip",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "aetherdialect._main_execution.save_seed_warmup_report",
        lambda *_a, **_k: None,
    )

    captured: dict[str, object] = {}

    def _fake_exec(*_a, **kwargs):
        captured.update(kwargs)
        return [], [], 1, {"execute_ok_count": 0}

    monkeypatch.setattr(
        "aetherdialect._main_execution.run_seed_warmup_execution",
        _fake_exec,
    )

    _run_seed_warmup_sql_history_pipeline(
        schema=schema,
        dialect=dialect,
        output_dir=str(tmp_path),
        store=None,
        templates={},
        sql_texts=["SELECT 1"],
        sql_history_content_hash="hash",
        seed=SeedWarmupConfig.RANDOM_SEED,
        expand=False,
        max_kept_intents=None,
    )

    assert captured.get("max_kept_intents") is None
