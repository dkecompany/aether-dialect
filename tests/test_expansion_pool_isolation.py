"""Expansion subtree pool must be keyed per artifacts directory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import SeedWarmupIntent, SelectCol
from aetherdialect._expansion_ops import (
    _EXPANSION_SUBTREE_POOL,
    _record_expansion_subtree_pool,
    clear_expansion_subtree_pool,
)


def _intent(intent_id: str, table: str) -> SeedWarmupIntent:
    return SeedWarmupIntent(
        intent_id=intent_id,
        natural_language=intent_id,
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column(f"{table}.name"),
                    op="=",
                    raw_value="x",
                )
            ]
        ),
        having=None,
    )


def _engine_bundle() -> MagicMock:
    bundle = MagicMock()
    bundle.dialect = MagicMock()
    bundle.data_quality_report = None
    return bundle


@pytest.mark.fast
def test_concurrent_warmups_isolated(tmp_path: Path) -> None:
    clear_expansion_subtree_pool()
    key_a = str(tmp_path / "artifacts-a")
    key_b = str(tmp_path / "artifacts-b")
    _record_expansion_subtree_pool(_intent("a1", "t_a"), pool_key=key_a)
    _record_expansion_subtree_pool(_intent("b1", "t_b"), pool_key=key_b)

    assert isinstance(_EXPANSION_SUBTREE_POOL, dict)
    assert len(_EXPANSION_SUBTREE_POOL.get(key_a, [])) == 1
    assert len(_EXPANSION_SUBTREE_POOL.get(key_b, [])) == 1
    assert _EXPANSION_SUBTREE_POOL[key_a][0].intent_id == "a1"
    assert _EXPANSION_SUBTREE_POOL[key_b][0].intent_id == "b1"

    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()),
        patch("aetherdialect.aetherdialect.drain_write_queue"),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect"),
        patch("aetherdialect.aetherdialect.LLMProvider.clear_llm_clients"),
        patch("aetherdialect.aetherdialect.drop_engine_skeleton_cache_owner"),
        patch("aetherdialect.aetherdialect.release_close_resources"),
        patch.object(AetherEngine, "_audit_emit"),
    ):
        engine = AetherEngine(MagicMock(), artifacts_dir=key_a)
        engine._artifacts_dir = Path(key_a)
        engine.close()

    assert not _EXPANSION_SUBTREE_POOL.get(key_a)
    assert len(_EXPANSION_SUBTREE_POOL.get(key_b, [])) == 1
    clear_expansion_subtree_pool(key_b)
    assert not _EXPANSION_SUBTREE_POOL.get(key_b)
    clear_expansion_subtree_pool()
