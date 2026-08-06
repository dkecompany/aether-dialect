"""Profiling refresh must prune stale warmup units; work_unit_id content-addressed."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import pytest

from aetherdialect._config import SeedWarmupConfig
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import SeedWarmupIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._seed_warmup import SeedWarmupCacheSession

open_seed_warmup_cache_session = SeedWarmupCacheSession.open_seed_warmup_cache_session
warmup_intent_fingerprint = SeedWarmupCacheSession.warmup_intent_fingerprint


def _orders_intent() -> SeedWarmupIntent:
    return SeedWarmupIntent(
        intent_id="warm_orders",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        param_values={},
        expansion_metadata=None,
        limit=None,
    )


def _orders_schema(*, profiling_hash: str = "p0", graph_id: str = "same") -> SchemaGraph:
    tbl = TableMetadata(
        name="orders",
        columns={"order_id": ColumnMetadata(name="order_id", data_type="integer", sensitivity="none")},
        primary_key=["order_id"],
        foreign_keys=[],
        row_count=1,
    )
    return SchemaGraph(
        tables={"orders": tbl},
        join_paths_multi={},
        effective_structural_hash="same",
        structural_hash="same",
        schema_graph_id=graph_id,
        profiling_hash=profiling_hash,
    )


def _work_unit_record(*, fingerprint: str, work_unit_id: str) -> dict:
    intent = _orders_intent()
    return {
        "work_unit_id": work_unit_id,
        "intent_fingerprint": fingerprint,
        "serialized_intent": intent.to_dict(),
        "execute_result": {"ok": True, "runtime": intent.to_runtime_intent().to_dict()},
        "lifecycle_state": "execute_recorded",
    }


def _seed_cache_zip(
    td: str,
    *,
    profiling_hash: str = "p0",
    work_units: dict[str, dict],
) -> None:
    path = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            SeedWarmupConfig.WARMUP_CACHE_MANIFEST,
            json.dumps(
                {
                    "effective_structural_hash": "same",
                    "schema_hash": "same",
                    "schema_graph_id": "same",
                    "seed_content_hash": "same",
                    "profiling_hash": profiling_hash,
                },
            ),
        )
        for wid, rec in work_units.items():
            zf.writestr(f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}{wid}.json", json.dumps(rec))


@pytest.mark.fast
def test_profiling_hash_drift_prunes_stale_work_units() -> None:
    fp = SeedWarmupCacheSession.warmup_intent_fingerprint(_orders_intent())
    with tempfile.TemporaryDirectory() as td:
        _seed_cache_zip(td, work_units={"stale-wid": _work_unit_record(fingerprint=fp, work_unit_id="stale-wid")})
        schema = SchemaGraph(
            tables={},
            join_paths_multi={},
            effective_structural_hash="same",
            structural_hash="same",
            schema_graph_id="same",
            profiling_hash="p1",
        )
        sess = SeedWarmupCacheSession.open_seed_warmup_cache_session(td, schema, "same")
        assert sess.work_units == {}
        assert sess.fp_to_wid == {}


@pytest.mark.fast
def test_profiling_hash_drift_retains_live_work_units() -> None:
    fp = SeedWarmupCacheSession.warmup_intent_fingerprint(_orders_intent())
    with tempfile.TemporaryDirectory() as td:
        _seed_cache_zip(td, work_units={"live-wid": _work_unit_record(fingerprint=fp, work_unit_id="live-wid")})
        schema = _orders_schema(profiling_hash="p1")
        sess = SeedWarmupCacheSession.open_seed_warmup_cache_session(td, schema, "same")
        assert "live-wid" in sess.work_units
        assert sess.fp_to_wid[fp] == "live-wid"


@pytest.mark.fast
def test_ensure_work_unit_id_is_content_addressed_from_fingerprint() -> None:
    fp = SeedWarmupCacheSession.warmup_intent_fingerprint(_orders_intent())
    sess = SeedWarmupCacheSession(manifest={}, work_units={})
    first = sess.ensure_work_unit_id(fp)
    second = sess.ensure_work_unit_id(fp)
    assert first == fp
    assert second == fp
    assert sess.ensure_work_unit_id(SeedWarmupCacheSession.warmup_intent_fingerprint(_orders_intent())) == fp


@pytest.mark.fast
def test_reingest_same_fingerprint_reuses_content_addressed_work_unit_id() -> None:
    fp = SeedWarmupCacheSession.warmup_intent_fingerprint(_orders_intent())
    first = SeedWarmupCacheSession(manifest={}, work_units={})
    wid_a = first.ensure_work_unit_id(fp)
    second = SeedWarmupCacheSession(manifest={}, work_units={})
    wid_b = second.ensure_work_unit_id(fp)
    assert wid_a == wid_b == fp
