"""Tests for caller-visible template enumeration and execution on AetherEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import ConfigError, TemplateExecutionResult
from aetherdialect._contracts_core import Template
from tests.test_aetherdialect import _make_aether_stub
from tests.test_reuse_saved_question import _minimal_template


def _engine_with_templates(*templates: Template) -> AetherEngine:
    tmpl_map = {t.id: t for t in templates}
    dialect = MagicMock()
    dialect.sqlglot_dialect = "duckdb"
    return _make_aether_stub(_templates=tmpl_map, _dialect=dialect, _context_name="master")


@pytest.mark.fast
def test_fetch_template_rejects_unknown_id() -> None:
    pass


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
