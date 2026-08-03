"""Tests for caller-visible template enumeration and execution on AetherEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import ConfigError, ParameterBinding, StoredTemplateDetail, TemplateExecutionResult
from aetherdialect._contracts_core import Template
from aetherdialect._contracts_schema import TemplateStats
from tests.test_aetherdialect import _make_aether_stub
from tests.test_reuse_saved_question import _minimal_template


def _engine_with_templates(*templates: Template) -> AetherEngine:
    tmpl_map = {t.id: t for t in templates}
    dialect = MagicMock()
    dialect.sqlglot_dialect = "duckdb"
    return _make_aether_stub(_templates=tmpl_map, _dialect=dialect, _context_name="master")


@pytest.mark.fast
def test_list_templates_empty() -> None:
    engine = _engine_with_templates()
    assert engine.list_templates() == ()


@pytest.mark.fast
def test_list_templates_returns_summary_metadata() -> None:
    tmpl = _minimal_template(question="count of item in category x")
    tmpl.stats = TemplateStats(accept=3, reject=0)
    engine = _engine_with_templates(tmpl)
    rows = engine.list_templates()
    assert len(rows) == 1
    row = rows[0]
    assert row.id == "T0001"
    assert row.q_norm == "count of item in category x"
    assert row.sql_param == tmpl.sql_param
    assert row.space == "master"
    assert row.use_count == 3


@pytest.mark.fast
def test_list_templates_excludes_federation_plan_only() -> None:
    visible = _minimal_template()
    hidden = _minimal_template(question="hidden federation row")
    hidden.id = "T0002"
    hidden.federation_plan_only = True
    engine = _engine_with_templates(visible, hidden)
    ids = {row.id for row in engine.list_templates()}
    assert ids == {"T0001"}


@pytest.mark.fast
def test_fetch_template_by_id() -> None:
    tmpl = _minimal_template()
    engine = _engine_with_templates(tmpl)
    detail = engine.fetch_template("T0001")
    assert isinstance(detail, StoredTemplateDetail)
    assert detail.summary.id == "T0001"
    assert detail.sql_fp == "fp"
    assert detail.parameters == (ParameterBinding(handle="p1", current_value="x", display_name="category"),)


@pytest.mark.fast
def test_fetch_template_by_sql_fp() -> None:
    tmpl = _minimal_template()
    engine = _engine_with_templates(tmpl)
    detail = engine.fetch_template("fp")
    assert detail.summary.id == "T0001"


@pytest.mark.fast
def test_fetch_template_unknown_raises() -> None:
    engine = _engine_with_templates()
    with pytest.raises(ConfigError, match="unknown template ref"):
        engine.fetch_template("missing")


@pytest.mark.fast
def test_execute_template_returns_execution_result() -> None:
    tmpl = _minimal_template()
    engine = _engine_with_templates(tmpl)
    expected = TemplateExecutionResult(
        rows=((1,),),
        sql="SELECT 1",
        display_sql="SELECT 1",
        columns=("count",),
    )
    with patch("aetherdialect.aetherdialect.execute_stored_template_by_ref", return_value=expected) as run:
        result = engine.execute_template("T0001", {"p1": "y"})
    run.assert_called_once()
    assert result is expected


@pytest.mark.fast
def test_execute_template_supports_dataframe_output() -> None:
    tmpl = _minimal_template()
    engine = _engine_with_templates(tmpl)
    expected = TemplateExecutionResult(
        rows=((1,), (2,)),
        sql="SELECT 1",
        display_sql="SELECT 1",
        columns=("n",),
    )
    with patch("aetherdialect.aetherdialect.execute_stored_template_by_ref", return_value=expected):
        frame = engine.execute_template("T0001", {"p1": "y"}, as_dataframe=True)
    assert list(frame.columns) == ["n"]
    assert frame.iloc[0, 0] == 1
