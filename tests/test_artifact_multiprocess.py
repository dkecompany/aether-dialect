"""Multiprocess artifact safety for template store saves, write-queue drain, and plan RMW."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import (
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_PARTITION_PREFIX,
    WRITE_QUEUE_FILENAME,
)
from aetherdialect._contracts_base import EngineIdentity, NormalizedExpr
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FeedbackKind,
    QuestionFeedbackEntry,
    RejectionBucket,
    SelectCol,
    Template,
    ValueHistory,
    WriteQueueEvent,
)
from aetherdialect._contracts_schema import FederationPlanTemplate, SQLShape, TemplateStats
from aetherdialect._federation_execute import (
    load_federation_plan_templates,
    save_federation_plan_template,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates import TemplateStoreView
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    normalize_question,
    pop_engine_identity,
    push_engine_identity,
)
from aetherdialect._utils_artifacts import emit_write_queue_event, read_gzip_json

_GRAPH_ID = "sg_test000000000001__abcd1234"
_ROUNDS = 8
_TRANSIENT_ATTEMPTS = 30
_TRANSIENT_DELAY_SECONDS = 0.05

_T = TypeVar("_T")


def _retry_transient(call: Callable[[], _T]) -> _T:
    last_exc: BaseException | None = None
    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            return call()
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt + 1 >= _TRANSIENT_ATTEMPTS:
                raise
            time.sleep(_TRANSIENT_DELAY_SECONDS)
    raise AssertionError(f"retry loop exhausted: {last_exc!r}")


def _store_dir(artifacts_dir: str) -> str:
    return TemplateOps.template_store_dir_for_space(artifacts_dir)


def _configure_artifacts(artifacts_dir: str) -> None:
    PolicyConfig.REGENERATE_TEMPLATE_STORE = False
    EngineConfig.TEMPLATE_STORE_DIR = os.path.join(artifacts_dir, "intent_templates")


def _typed_template(*, tid: str) -> Template:
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return Template(
        id=tid,
        effective_structural_hash="h",
        schema_graph_id=_GRAPH_ID,
        intent_signature=concrete,
        intent_key=f"ik_{tid}",
        tables_used=["t"],
        sql_param="SELECT t.id FROM t",
        sql_fp=f"fp_{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
        schema_column_types={"t.id": "integer"},
    )


def _distinct_partition_template_ids(n: int) -> list[str]:
    ids: list[str] = []
    seen: set[int] = set()
    for i in range(500_000):
        tid = f"T{i:06d}"
        part = TemplateStoreView.template_partition_number(tid)
        if part in seen:
            continue
        seen.add(part)
        ids.append(tid)
        if len(ids) >= n:
            break
    assert len(ids) == n
    return ids


def _plan_template(plan_id: str) -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=plan_id,
        composite_schema_graph_id="cg_full",
        intent_key=f"ik_{plan_id}",
        step_fingerprints=(("alpha", "fp_a"), ("beta", "fp_b")),
        combine_hash="combine_hash",
        question="show joined entities",
        accepted_questions=("accepted q",),
        format_version="0.2.3",
        member_template_ids=(("alpha", "T0001"), ("beta", "T0002")),
        residual_hash="residual_hash",
        join_feedback=("join hint",),
        manifest_hash="manifest_hash",
        member_tuple_hash="member_tuple_hash",
    )


def _assert_store_header_and_shards(store_dir: str, expected_tids: set[str]) -> None:
    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    assert os.path.isfile(header_path), "template store header missing"
    header = read_gzip_json(header_path)
    assert isinstance(header, dict)
    partition_map = header.get("partition_map", {})
    assert isinstance(partition_map, dict)
    for tid in expected_tids:
        assert tid in partition_map, f"{tid} lost from header partition_map: {sorted(partition_map)}"
        part = int(partition_map[tid])
        shard_path = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}{part:02x}.json.gz")
        assert os.path.isfile(shard_path), f"shard missing for {tid} at partition {part}"
        shard = read_gzip_json(shard_path)
        assert isinstance(shard, dict)
        assert tid in shard, f"{tid} missing from shard payload at partition {part}"


def _worker_engine_identity():
    return push_engine_identity(EngineIdentity(EngineConfig.TYPE, EngineConfig.RUNTIME))


def _template_save_worker(artifacts_dir: str, template_id: str, errors: Any) -> None:
    token = _worker_engine_identity()
    try:
        _configure_artifacts(artifacts_dir)
        os.makedirs(_store_dir(artifacts_dir), exist_ok=True)
        template = replace(_typed_template(tid=template_id), trust_level=1)
        for _ in range(_ROUNDS):

            def _round() -> None:
                store = TemplateOps.load_template_store(_GRAPH_ID, schema=None, artifacts_dir=artifacts_dir)
                store.set_template_raw_dict(
                    template_id,
                    TemplateOps._convert_to_json_serializable(template.to_dict()),
                )
                TemplateStoreView.refresh_template_store_indexes(store, template_objs=[template])
                TemplateOps.save_template_store(store)

            _retry_transient(_round)
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        pop_engine_identity(token)


def _write_queue_drain_worker(artifacts_dir: str, errors: Any) -> None:
    token = _worker_engine_identity()
    try:
        _configure_artifacts(artifacts_dir)
        for _ in range(_ROUNDS):

            def _round() -> None:
                store = TemplateOps.load_template_store(_GRAPH_ID, schema=None, artifacts_dir=artifacts_dir)
                owner = MagicMock()
                owner._schema_graph = MagicMock()
                owner._schema_graph.schema_graph_id = _GRAPH_ID
                owner._store = store
                owner._templates = TemplateOps.store_to_templates(store)
                owner._rejected = {}
                owner._dialect = None
                MainExecutionOps.drain_write_queue(owner, artifacts_dir)

            _retry_transient(_round)
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        pop_engine_identity(token)


def _plan_template_save_worker(federation_dir: str, plan_id: str, errors: Any) -> None:
    try:
        template = _plan_template(plan_id)
        for _ in range(_ROUNDS):
            _retry_transient(lambda: save_federation_plan_template(federation_dir, template))
    except BaseException as exc:
        errors.append(repr(exc))


def _spawn_workers(target: Any, arg_sets: list[tuple[Any, ...]]) -> list[Any]:
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    errors = manager.list()
    processes = [ctx.Process(target=target, args=(*args, errors)) for args in arg_sets]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"worker exited with code {proc.exitcode}"
    return list(errors)


@pytest.mark.fast
def test_concurrent_template_store_saves_survive_multiprocess(tmp_path) -> None:
    """Two processes saving different templates must not lose ids or strip header/shards."""
    artifacts_dir = str(tmp_path)
    tid_a, tid_b = _distinct_partition_template_ids(2)
    errors = _spawn_workers(
        _template_save_worker,
        [(artifacts_dir, tid_a), (artifacts_dir, tid_b)],
    )
    assert not errors, f"template save workers failed: {errors!r}"

    _configure_artifacts(artifacts_dir)
    loaded = TemplateOps.load_template_store(_GRAPH_ID, schema=None, artifacts_dir=artifacts_dir)
    assert tid_a in loaded.partition_map, f"{tid_a} lost to concurrent save: {sorted(loaded.partition_map)}"
    assert tid_b in loaded.partition_map, f"{tid_b} lost to concurrent save: {sorted(loaded.partition_map)}"
    _assert_store_header_and_shards(_store_dir(artifacts_dir), {tid_a, tid_b})


@pytest.mark.fast
def test_concurrent_write_queue_drains_persist_multiprocess(tmp_path) -> None:
    """Two processes draining the same queue must apply every event once without truncation races."""
    artifacts_dir = str(tmp_path)
    _configure_artifacts(artifacts_dir)
    os.makedirs(_store_dir(artifacts_dir), exist_ok=True)
    store = TemplateOps.empty_template_store_for_space(_GRAPH_ID, artifacts_dir=artifacts_dir)
    TemplateOps.save_template_store(store)

    ts = datetime.now(UTC).isoformat()
    q_norms = [normalize_question(f"multiprocess drain question {label}") for label in ("alpha", "beta")]
    for idx, q_norm in enumerate(q_norms):
        entry = QuestionFeedbackEntry(
            summary=f"summary_{idx}",
            buckets=(RejectionBucket.OTHER,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash=_GRAPH_ID,
            intent_structural_hash=f"ik_{idx}",
            intent_payload="{}",
            created_at=ts,
            updated_at=ts,
        )
        emit_write_queue_event(
            artifacts_dir,
            WriteQueueEvent(
                kind="feedback_record",
                schema_graph_id=_GRAPH_ID,
                schema_hash=_GRAPH_ID,
                produced_at=ts,
                payload=(("q_norm", q_norm), ("entry_json", json.dumps(entry.to_dict()))),
            ),
            space_name="master",
        )
    queue_path = tmp_path / WRITE_QUEUE_FILENAME
    assert queue_path.stat().st_size > 0

    errors = _spawn_workers(_write_queue_drain_worker, [(artifacts_dir,), (artifacts_dir,)])
    assert not errors, f"write queue drain workers failed: {errors!r}"

    loaded = TemplateOps.load_template_store(_GRAPH_ID, schema=None, artifacts_dir=artifacts_dir)
    for q_norm in q_norms:
        assert q_norm in loaded.question_feedback, f"feedback for {q_norm!r} lost after concurrent drain"
        assert len(loaded.question_feedback[q_norm]) == 1
    assert not queue_path.read_bytes(), "write queue was truncated before all events persisted"

    expected_tids = set(loaded.partition_map)
    if expected_tids:
        _assert_store_header_and_shards(_store_dir(artifacts_dir), expected_tids)


@pytest.mark.fast
def test_concurrent_plan_template_rmw_survives_multiprocess(tmp_path) -> None:
    """Two processes saving different federation plans must not clobber each other's RMW."""
    federation_dir = str(tmp_path / "federation")
    os.makedirs(federation_dir, exist_ok=True)
    errors = _spawn_workers(
        _plan_template_save_worker,
        [(federation_dir, "plan_alpha"), (federation_dir, "plan_beta")],
    )
    assert not errors, f"plan template workers failed: {errors!r}"

    loaded = load_federation_plan_templates(federation_dir)
    assert "plan_alpha" in loaded, f"plan_alpha lost to concurrent write: {sorted(loaded)}"
    assert "plan_beta" in loaded, f"plan_beta lost to concurrent write: {sorted(loaded)}"
