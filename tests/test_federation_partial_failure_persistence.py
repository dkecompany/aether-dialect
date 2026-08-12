"""Federation partial failure must not mutate on-disk artifacts or template stores."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationPartialFailureError
from aetherdialect._contracts_core import SourceStep
from aetherdialect._contracts_schema import FederationMappings
from aetherdialect._federation_execute import (
    compute_federation_storage_dir,
    federation_source_artifacts_dir,
    persist_federation_tree,
)
from aetherdialect._pipeline_execute import (
    execute_federated_prepare,
    execute_federated_sql_plan,
)
from tests.federation_helpers import (
    TwoMemberFederation,
    build_staged_two_member_prepare_outcome,
    hash_directory_tree,
    seed_member_template_stores,
    template_store_partition_count,
)


def _seed_artifact_tree(tmp_path: Any, fed: TwoMemberFederation) -> dict[str, int]:
    federation_dir = compute_federation_storage_dir(str(tmp_path), fed.manifest.federation_id)
    persist_federation_tree(
        federation_dir,
        manifest=fed.manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=fed.composite,
        member_graphs=fed.member_graphs,
    )
    seed_member_template_stores(str(tmp_path), fed.manifest, fed.member_graphs)
    counts: dict[str, int] = {}
    for binding in fed.manifest.sources:
        artifacts_dir = federation_source_artifacts_dir(str(tmp_path), binding)
        counts[binding.source_id] = template_store_partition_count(
            artifacts_dir,
            fed.member_graphs[binding.source_id],
        )
    return counts


def _member_executor_that_fails_on_second(
    fed: TwoMemberFederation,
) -> Any:
    def _execute(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        if step.source_id == fed.left_source:
            return pd.DataFrame({"id": [1]})
        raise RuntimeError("member statement failed mid-execution")

    return _execute


@pytest.mark.fast
def test_mid_execution_member_failure_leaves_artifact_tree_unchanged(
    two_member_federation: TwoMemberFederation,
    tmp_path: Any,
) -> None:
    fed = two_member_federation
    _seed_artifact_tree(tmp_path, fed)
    before_hash = hash_directory_tree(tmp_path)
    prepared, composite, manifest = build_staged_two_member_prepare_outcome(fed)

    with (
        patch("aetherdialect._federation_execute.revalidate_prepared_federation_plan"),
        patch(
            "aetherdialect._pipeline_execute._execute_federation_source_step",
            side_effect=_member_executor_that_fails_on_second(fed),
        ),
    ):
        with pytest.raises(FederationPartialFailureError) as exc_info:
            execute_federated_prepare(
                prepared,
                composite,
                dialect=MagicMock(),
                dialects_by_source={fed.left_source: MagicMock(), fed.right_source: MagicMock()},
                manifest=manifest,
            )

    assert exc_info.value.source_id == fed.right_source
    assert exc_info.value.phase == "member"
    assert hash_directory_tree(tmp_path) == before_hash


@pytest.mark.fast
def test_mid_execution_member_failure_leaves_template_store_row_count_unchanged(
    two_member_federation: TwoMemberFederation,
    tmp_path: Any,
) -> None:
    fed = two_member_federation
    before_counts = _seed_artifact_tree(tmp_path, fed)
    prepared, composite, manifest = build_staged_two_member_prepare_outcome(fed)

    with (
        patch("aetherdialect._federation_execute.revalidate_prepared_federation_plan"),
        patch(
            "aetherdialect._pipeline_execute._execute_federation_source_step",
            side_effect=_member_executor_that_fails_on_second(fed),
        ),
    ):
        with pytest.raises(FederationPartialFailureError):
            execute_federated_prepare(
                prepared,
                composite,
                dialect=MagicMock(),
                dialects_by_source={fed.left_source: MagicMock(), fed.right_source: MagicMock()},
                manifest=manifest,
            )

    for binding in fed.manifest.sources:
        artifacts_dir = federation_source_artifacts_dir(str(tmp_path), binding)
        after_count = template_store_partition_count(
            artifacts_dir,
            fed.member_graphs[binding.source_id],
        )
        assert after_count == before_counts[binding.source_id]


@pytest.mark.fast
def test_execution_failure_stamps_prepared_outcome_with_member_phase_and_error(
    two_member_federation: TwoMemberFederation,
) -> None:
    fed = two_member_federation
    prepared, composite, manifest = build_staged_two_member_prepare_outcome(fed)

    with (
        patch("aetherdialect._pipeline_execute.prepare_federated_sql_plan", return_value=prepared),
        patch("aetherdialect._federation_execute.revalidate_prepared_federation_plan"),
        patch(
            "aetherdialect._pipeline_execute._execute_federation_source_step",
            side_effect=_member_executor_that_fails_on_second(fed),
        ),
        patch("aetherdialect._pipeline_execute.persist_federated_member_stores") as persist_stores,
    ):
        outcome = execute_federated_sql_plan(
            "join left and right",
            prepared.plan,
            composite,
            dialect=MagicMock(),
            dialects_by_source={fed.left_source: MagicMock(), fed.right_source: MagicMock()},
            join_candidates={},
            cmap={},
            store={},
            manifest=manifest,
        )

    persist_stores.assert_not_called()
    assert outcome.success is False
    assert outcome.prepared is not None
    assert outcome.prepared.success is False
    assert outcome.prepared.source_id == fed.right_source
    assert outcome.prepared.phase == "member"
    assert outcome.prepared.sql_validation_error
    assert outcome.sql_validation_error == outcome.prepared.sql_validation_error
    assert outcome.error_kind == "federation_execution_failed"
