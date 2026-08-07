"""BigQuery execute-time bytes billing must honour dialect member limit attrs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._dialect_sqlglot_engines import BigQueryDialect


@pytest.mark.fast
def test_member_override_sets_maximum_bytes_billed() -> None:
    dialect = object.__new__(BigQueryDialect)
    dialect.max_query_cost_bytes = 100.0
    job_config = MagicMock()

    with patch("aetherdialect._dialect_sqlglot_engines.PolicyConfig.MAX_QUERY_COST_BYTES", 50_000_000_000):
        with patch("aetherdialect._dialect_sqlglot_engines.cost_cap_active", return_value=True):
            BigQueryDialect.apply_execute_cost_limits(dialect, job_config)

    assert job_config.maximum_bytes_billed == 100

    with patch("aetherdialect._dialect_sqlglot_engines.PolicyConfig.MAX_QUERY_COST_BYTES", 50_000_000_000):
        with patch("aetherdialect._dialect_sqlglot_engines.cost_cap_active", return_value=True):
            max_bytes, _timeout = BigQueryDialect._bq_job_limits(dialect)

    assert max_bytes == 100
    assert PolicyConfig.MAX_QUERY_COST_BYTES != 100
