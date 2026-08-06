"""Federation plan templates must reject stale format_version rows."""

from __future__ import annotations

import json
import tempfile

import pytest

from aetherdialect._constants import FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION
from aetherdialect._contracts_base import FederationPlanTemplate
from aetherdialect._federation import (
    federation_artifact_paths,
    load_federation_plan_templates,
    save_federation_plan_template,
)


def _plan_template(plan_id: str) -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=plan_id,
        composite_schema_graph_id="cg",
        intent_key=plan_id,
        step_fingerprints=(),
        combine_hash="h",
        question="show orders",
    )


@pytest.mark.fast
def test_wrong_plan_format_ignored_or_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_federation_plan_template(tmp, _plan_template("good"))
        path = federation_artifact_paths(tmp)["plan_templates"]
        payload = json.loads(open(path, encoding="utf-8").read())
        payload["stale"] = {
            **payload["good"],
            "format_version": "0.0.0",
        }
        payload["future"] = {
            **payload["good"],
            "format_version": "9.9.9",
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        loaded = load_federation_plan_templates(tmp)
        assert set(loaded) == {"good"}
        assert loaded["good"].format_version == FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION
