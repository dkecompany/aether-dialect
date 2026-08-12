"""Federation plan template prune must preserve survivors and write atomically."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_schema import FederationPlanTemplate
from aetherdialect._federation_execute import (
    load_federation_plan_templates,
    prune_federation_plan_templates_for_sources,
    save_federation_plan_template,
)
from aetherdialect._federation_manifest import federation_artifact_paths


def _full_template(
    plan_id: str,
    *,
    sources: tuple[tuple[str, str], ...] = (("alpha", "fp_a"), ("beta", "fp_b")),
    member_template_ids: tuple[tuple[str, str], ...] = (("alpha", "T0001"), ("beta", "T0002")),
) -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=plan_id,
        composite_schema_graph_id="cg_full",
        intent_key=f"ik_{plan_id}",
        step_fingerprints=sources,
        combine_hash="combine_hash",
        question="show joined entities",
        accepted_questions=("accepted q",),
        format_version="0.2.3",
        member_template_ids=member_template_ids,
        residual_hash="residual_hash",
        join_feedback=("join hint",),
        manifest_hash="manifest_hash",
        member_tuple_hash="member_tuple_hash",
    )


@pytest.mark.fast
def test_prune_preserves_all_fields_on_survivors() -> None:
    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(fed_dir, _full_template("plan_alpha"))
        save_federation_plan_template(
            fed_dir,
            _full_template(
                "plan_beta",
                sources=(("beta", "fp_b"),),
                member_template_ids=(("beta", "T0002"),),
            ),
        )

        prune_federation_plan_templates_for_sources(fed_dir, {"alpha"})

        loaded = load_federation_plan_templates(fed_dir)
        assert "plan_alpha" not in loaded
        survivor = loaded["plan_beta"]
        assert survivor.accepted_questions == ("accepted q",)
        assert survivor.member_template_ids == (("beta", "T0002"),)
        assert survivor.residual_hash == "residual_hash"
        assert survivor.manifest_hash == "manifest_hash"
        assert survivor.member_tuple_hash == "member_tuple_hash"
        assert survivor.format_version == "0.2.3"

        path = federation_artifact_paths(fed_dir)["plan_templates"]
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        row = raw["plan_beta"]
        assert row["accepted_questions"] == ["accepted q"]
        assert row["member_template_ids"] == [["beta", "T0002"]]
        assert row["residual_hash"] == "residual_hash"
        assert row["manifest_hash"] == "manifest_hash"
        assert row["member_tuple_hash"] == "member_tuple_hash"
        assert row["format_version"] == "0.2.3"


@pytest.mark.fast
def test_prune_drops_templates_referencing_removed_member_template_ids() -> None:
    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(fed_dir, _full_template("plan_alpha"))
        save_federation_plan_template(
            fed_dir,
            _full_template("plan_beta", sources=(("beta", "fp_b"),)),
        )

        prune_federation_plan_templates_for_sources(fed_dir, {"alpha"})

        loaded = load_federation_plan_templates(fed_dir)
        assert "plan_alpha" not in loaded
        assert "plan_beta" in loaded
        assert loaded["plan_beta"].member_template_ids == (("beta", "T0002"),)


@pytest.mark.fast
def test_prune_writes_under_artifact_lock_atomically() -> None:
    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(fed_dir, _full_template("plan_alpha"))
        save_federation_plan_template(
            fed_dir,
            _full_template(
                "plan_beta",
                sources=(("beta", "fp_b"),),
                member_template_ids=(("beta", "T0002"),),
            ),
        )

        with (
            patch("aetherdialect._federation_execute.artifact_lock") as lock_cm,
            patch("aetherdialect._federation_execute._write_federation_json_atomic") as write_atomic,
        ):
            lock_cm.return_value.__enter__ = MagicMock(return_value=None)
            lock_cm.return_value.__exit__ = MagicMock(return_value=False)
            prune_federation_plan_templates_for_sources(fed_dir, {"alpha"})

        lock_cm.assert_called_once_with(fed_dir)
        write_atomic.assert_called_once()
        path, payload = write_atomic.call_args[0]
        assert path == federation_artifact_paths(fed_dir)["plan_templates"]
        assert "plan_beta" in payload
        assert "plan_alpha" not in payload
        assert payload["plan_beta"]["accepted_questions"] == ["accepted q"]


@pytest.mark.fast
def test_prune_on_member_drift_drops_affected_templates() -> None:
    from aetherdialect._federation_execute import prune_federation_plan_templates_on_drift

    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(fed_dir, _full_template("plan_alpha"))
        save_federation_plan_template(
            fed_dir,
            _full_template(
                "plan_beta",
                sources=(("beta", "fp_b"),),
                member_template_ids=(("beta", "T0002"),),
            ),
        )

        member_graphs = {
            "alpha": MagicMock(
                schema_graph_id="sg_alpha_drift",
                effective_structural_hash="eff_alpha_drift",
                profiling_hash="profile_alpha",
                notes_sha256="",
                notes_hash="",
            ),
            "beta": MagicMock(
                schema_graph_id="sg_beta",
                effective_structural_hash="eff_beta",
                profiling_hash="profile_beta",
                notes_sha256="",
                notes_hash="",
            ),
        }
        manifest = MagicMock()
        manifest.sources = [
            MagicMock(source_id="alpha"),
            MagicMock(source_id="beta"),
        ]

        with (
            patch(
                "aetherdialect._federation_manifest.federation_member_hash_tuple",
                return_value=(
                    ("alpha", "sg_alpha_drift", "eff_alpha_drift", "profile_alpha", "", ""),
                    ("beta", "sg_beta", "eff_beta", "profile_beta", "", ""),
                ),
            ),
            patch(
                "aetherdialect._federation_manifest.load_federation_artifact_manifest_dict",
                return_value={
                    "federation_members": [
                        ["alpha", "sg_alpha", "eff_alpha", "profile_alpha", "", ""],
                        ["beta", "sg_beta", "eff_beta", "profile_beta", "", ""],
                    ],
                },
            ),
        ):
            prune_federation_plan_templates_on_drift(fed_dir, member_graphs, manifest)

        loaded = load_federation_plan_templates(fed_dir)
        assert "plan_alpha" not in loaded
        assert "plan_beta" in loaded
        assert loaded["plan_beta"].accepted_questions == ("accepted q",)
