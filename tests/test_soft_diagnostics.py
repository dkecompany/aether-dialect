"""EXPLAIN soft diagnostics emit one structured row per finding."""

from __future__ import annotations

import pytest

from aetherdialect._constants import SOFT_DIAGNOSTIC_CODES
from aetherdialect._contracts_base import SqlDiagnostic, SqlDiagnosticCode
from aetherdialect._core_utils import drain_diagnostic_collector, reset_diagnostic_collector, set_diagnostic_collector
from aetherdialect._pipeline import emit_explain_soft_diagnostics


@pytest.mark.fast
def test_each_finding_emits_its_own_code() -> None:
    findings = (
        SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED, message="seq"),
        SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE, message="zero"),
    )
    token = set_diagnostic_collector([])
    try:
        emit_explain_soft_diagnostics(findings)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    codes = [d.code if isinstance(d.code, str) else getattr(d.code, "value", d.code) for d in diags]
    assert codes == [
        SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED.value,
        SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE.value,
    ]
    assert set(codes) <= set(SOFT_DIAGNOSTIC_CODES)
