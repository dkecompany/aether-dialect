"""Federation SessionStep.sql is a source_id → member SQL mapping."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_core import (
    FederatedSqlBundle,
    FederatedStatementRecord,
    GenerationPath,
    SqlGenerationOutcome,
)
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_multi_member_sql_is_source_mapping() -> None:
    bundle = FederatedSqlBundle(
        statements=(
            FederatedStatementRecord(source_id="crm", engine="postgresql", statement="SELECT 1 FROM crm.t"),
            FederatedStatementRecord(source_id="ops", engine="duckdb", statement="SELECT 2 FROM ops.t"),
        ),
        display_sql="/* federated */",
    )
    gen_out = SqlGenerationOutcome(
        sql="",
        success=True,
        generation_path=GenerationPath.FEDERATION_PLAN,
        matched_template=None,
    )
    sql = MainExecutionOps.resolved_session_step_sql(
        None,
        gen_out=gen_out,
        federated_bundle=bundle,
        generation_path=GenerationPath.FEDERATION_PLAN,
    )
    assert isinstance(sql, dict)
    assert list(sql.keys()) == ["crm", "ops"]
    assert sql["crm"] == "SELECT 1 FROM crm.t"
    assert sql["ops"] == "SELECT 2 FROM ops.t"


@pytest.mark.fast
def test_mapping_values_dialect_specific() -> None:
    bundle = FederatedSqlBundle(
        statements=(
            FederatedStatementRecord(
                source_id="pg",
                engine="postgresql",
                statement="SELECT * FROM t WHERE d = :p1",
            ),
            FederatedStatementRecord(
                source_id="bq",
                engine="bigquery",
                statement="SELECT * FROM `t` WHERE d = @p1",
            ),
        )
    )
    sql = MainExecutionOps.resolved_session_step_sql(
        None,
        federated_bundle=bundle,
        generation_path=GenerationPath.FEDERATION_PLAN,
    )
    assert isinstance(sql, dict)
    assert ":p1" in sql["pg"]
    assert "@p1" in sql["bq"]


@pytest.mark.fast
def test_one_member_degenerate_sql_str() -> None:
    bundle = FederatedSqlBundle(
        statements=(FederatedStatementRecord(source_id="crm", engine="postgresql", statement="SELECT 1 FROM crm.t"),)
    )
    sql = MainExecutionOps.resolved_session_step_sql(
        None,
        federated_bundle=bundle,
        generation_path=GenerationPath.FEDERATION_PLAN,
    )
    assert isinstance(sql, str)
    assert sql == "SELECT 1 FROM crm.t"
