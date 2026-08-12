"""Slim StoredTemplateSummary / StoredTemplateDetail field contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import ApprovalState
from aetherdialect._templates_ops import TemplateOps
from tests.template_fixtures import _minimal_template


@pytest.mark.fast
def test_summary_has_id_and_approval_only() -> None:
    tmpl = _minimal_template()
    tmpl.approval_state = ApprovalState.APPROVED
    tmpl.footprint_tables = ("t1",)
    tmpl.footprint_columns = ("t1.category",)
    dialect = MagicMock()
    dialect.sqlglot_dialect = "postgresql"
    summary = TemplateOps.summarize_stored_template(tmpl, space="master", dialect=dialect)
    assert summary.id == "T0001"
    assert summary.approval_state == "approved"
    assert not hasattr(summary, "sql_fp")
    assert not hasattr(summary, "footprint_tables")
    assert not hasattr(summary, "q_norm")


@pytest.mark.fast
def test_detail_has_pparam_slots() -> None:
    tmpl = _minimal_template()
    dialect = MagicMock()
    dialect.sqlglot_dialect = "postgresql"
    schema = MagicMock()
    schema.tables = {}
    detail = TemplateOps.build_stored_template_detail(
        tmpl, space="analytics", schema=schema, dialect=dialect, history_index=0
    )
    assert detail.summary.id == "T0001"
    assert detail.approval_state == "approved"
    assert not hasattr(detail, "sql_fp")
    assert detail.parameters
    slot = detail.parameters[0]
    assert slot.handle == "p1"
    assert slot.display_name == "category"
    assert slot.column_expr == "t1.category"
    assert all(p.handle.startswith("p") and p.handle[1:].isdigit() for p in detail.parameters)
