"""Tests for Unicode-aware question normalisation."""

from __future__ import annotations

import os

import pytest

from aetherdialect._core_utils import normalize_question


@pytest.mark.fast
def test_accented_letters_survive() -> None:
    """Accented letters remain after normalisation (NFKC + lowercase)."""
    assert normalize_question("Show café revenue") == "show café revenue"
    decomposed = "caf\u00e9"  # precomposed é
    assert normalize_question(f"Total {decomposed} sales") == f"total {decomposed} sales"
    composed = "caf\u0065\u0301"  # e + combining acute
    assert normalize_question(f"Total {composed} sales") == f"total {decomposed} sales"


@pytest.mark.fast
def test_two_different_questions_do_not_collide() -> None:
    """Questions that differ only by accent do not normalise to the same string."""
    accented = normalize_question("café revenue by region")
    plain = normalize_question("cafe revenue by region")
    assert accented != plain
    assert "é" in accented
    assert "é" not in plain


@pytest.mark.fast
def test_stale_question_index_version_triggers_rebuild(tmp_path) -> None:
    """An older question-normalisation version rebuilds indexes instead of failing lookup."""
    from aetherdialect._config import EngineConfig
    from aetherdialect._constants import (
        QUESTION_NORMALIZATION_VERSION,
        QUESTION_NORMALIZATION_VERSION_KEY,
        TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
        TEMPLATE_STORE_HEADER_FILENAME,
    )
    from aetherdialect._contracts_base import NormalizedExpr
    from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
    from aetherdialect._contracts_schema import SQLShape, TemplateStats
    from aetherdialect._core_utils import read_gzip_json, write_gzip_json_atomic
    from aetherdialect._templates import (
        TemplateOps,
        TemplateStoreView,
    )
    from aetherdialect._utils import question_token_fingerprint_from_raw

    artifacts_dir = str(tmp_path)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = TemplateOps.empty_template_store("graph_v1")
        intent = ConcreteIntent(
            intent_id="i1",
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.c"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            column_map={"c": "t"},
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="graph_v1",
            intent_signature=intent,
            intent_key="k1",
            tables_used=["t"],
            sql_param="SELECT c FROM t",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="sig1",
            value_history=ValueHistory(param_values=[{}], questions=["café sales"], natural_language=["café sales"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        TemplateOps.templates_to_store(store, {"T0001": tmpl})
        TemplateOps.save_template_store(store)
        hdr_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
        hdr = read_gzip_json(hdr_path)
        assert isinstance(hdr, dict)
        hdr[QUESTION_NORMALIZATION_VERSION_KEY] = "0.0.0"
        stale_fp = question_token_fingerprint_from_raw("cafe sales")
        hdr[TEMPLATE_QUESTION_TOKEN_INDEX_KEY] = {stale_fp: [["T0001", "0"]]}
        write_gzip_json_atomic(hdr_path, hdr, sort_keys=True)

        loaded = TemplateOps.load_template_store("graph_v1", schema=None, artifacts_dir=artifacts_dir)
        assert isinstance(loaded, TemplateStoreView)
        fresh_fp = question_token_fingerprint_from_raw("café sales")
        idx = loaded.get(TEMPLATE_QUESTION_TOKEN_INDEX_KEY) or {}
        assert stale_fp not in idx
        assert fresh_fp in idx
        assert ["T0001", "0"] in idx[fresh_fp]
        assert loaded.get(QUESTION_NORMALIZATION_VERSION_KEY) == QUESTION_NORMALIZATION_VERSION
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev
