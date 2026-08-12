"""Pipeline and warmup exports must not create files in the process working directory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas
import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    ConcreteIntent,
    DirectReuseSuspendContext,
    GenerationPath,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._pipeline_execute import (
    complete_direct_sql_reuse_user_choice,
    results_csv_output_path,
    save_result_csv,
)
from aetherdialect._seed_warmup import SeedWarmupCacheSession
from aetherdialect._templates_ops import TemplateOps


def _cwd_snapshot(cwd: Path) -> set[str]:
    return {p.name for p in cwd.iterdir()}


def _make_pipeline_template() -> Template:
    return Template(
        id="T0001",
        effective_structural_hash="test_hash",
        intent_signature=ConcreteIntent(
            intent_id="test",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="test_key",
        tables_used=["orders"],
        sql_param="SELECT order_id FROM orders",
        sql_fp="test_fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="test_sig",
        value_history=ValueHistory(param_values=[{}], questions=["test"], natural_language=["test"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
    )


@pytest.mark.fast
def test_no_file_written_to_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline turn and warmup export must leave the watched cwd untouched."""
    cwd = tmp_path / "watched_cwd"
    cwd.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    store_dir = artifacts_dir / "intent_templates" / "spaces" / "master"
    store_dir.mkdir(parents=True)

    monkeypatch.chdir(cwd)
    before = _cwd_snapshot(cwd)

    df = pandas.DataFrame({"col": [1, 2]})
    csv_dest = results_csv_output_path(artifacts_dir=str(artifacts_dir))
    save_result_csv(df, output_path=csv_dest)

    report_path = artifacts_dir / "seed_warmup_report_v1.json"
    SeedWarmupCacheSession.save_seed_warmup_report([], str(report_path))

    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    tmpl = _make_pipeline_template()
    store = {"_store_dir": str(store_dir)}
    ctx = DirectReuseSuspendContext(
        q_norm="q",
        ref_tmpl=tmpl,
        dialect=MagicMock(),
        store=store,
        templates={"T0001": tmpl},
        rejected={},
        schema=MagicMock(),
        intent=intent,
        sql="SELECT 1",
        rows=((1,),),
        display_sql="SELECT 1",
        headers=None,
        is_exact=True,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        sd_reuse=None,
    )
    with (
        patch("aetherdialect._llm_provider.LLMProvider.chat", return_value='{"aliases":{}}'),
        patch.object(TemplateOps, "save_template_store"),
        patch.object(TemplateOps, "templates_to_store", side_effect=lambda s, t: s),
        patch.object(TemplateOps, "delete_rejected_templates_matching_question"),
        patch.object(TemplateOps, "promote_trust"),
    ):
        complete_direct_sql_reuse_user_choice(ctx, "y")

    after = _cwd_snapshot(cwd)
    assert before == after
    assert (artifacts_dir / "results.csv").exists()
    assert report_path.is_file()
    assert not (cwd / "results.csv").exists()
    assert not (cwd / "live_tests").exists()


@pytest.mark.fast
def test_results_csv_skipped_without_store_artifacts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline turns must not write CSV when no artifacts destination can be resolved."""
    cwd = tmp_path / "watched_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before = _cwd_snapshot(cwd)

    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    tmpl = _make_pipeline_template()
    ctx = DirectReuseSuspendContext(
        q_norm="q",
        ref_tmpl=tmpl,
        dialect=MagicMock(),
        store={},
        templates={"T0001": tmpl},
        rejected={},
        schema=MagicMock(),
        intent=intent,
        sql="SELECT 1",
        rows=((1,),),
        display_sql="SELECT 1",
        headers=None,
        is_exact=True,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        sd_reuse=None,
    )
    with (
        patch("aetherdialect._llm_provider.LLMProvider.chat", return_value='{"aliases":{}}'),
        patch.object(TemplateOps, "save_template_store"),
        patch.object(TemplateOps, "templates_to_store", side_effect=lambda s, t: s),
        patch.object(TemplateOps, "delete_rejected_templates_matching_question"),
        patch.object(TemplateOps, "promote_trust"),
        patch("aetherdialect._pipeline_execute.save_result_csv") as save_csv,
    ):
        complete_direct_sql_reuse_user_choice(ctx, "y")
        save_csv.assert_not_called()

    assert _cwd_snapshot(cwd) == before
