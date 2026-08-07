"""Template store header format_version mismatches fail closed with a named error."""

from __future__ import annotations

import os

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import TEMPLATE_STORE_FORMAT_VERSION, TEMPLATE_STORE_HEADER_FILENAME
from aetherdialect._contracts_base import ConfigError
from aetherdialect._core_utils import write_gzip_json_atomic
from aetherdialect._templates import TemplateOps


@pytest.mark.fast
@pytest.mark.parametrize("wrong_version", [2, 99])
def test_wrong_format_version_raises(tmp_path, monkeypatch, wrong_version: int) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    artifacts_dir = str(tmp_path)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir)
    os.makedirs(store_dir, exist_ok=True)
    graph_id = "sg_test000000000001__abcd1234"
    write_gzip_json_atomic(
        os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME),
        {
            "format_version": wrong_version,
            "schema_graph_id": graph_id,
            "next_id": 1,
            "partition_map": {},
        },
        sort_keys=True,
    )

    with pytest.raises(ConfigError, match=r"format_version") as exc_info:
        TemplateOps.load_template_store(graph_id, schema=None, artifacts_dir=artifacts_dir)

    msg = str(exc_info.value)
    assert str(wrong_version) in msg
    assert str(TEMPLATE_STORE_FORMAT_VERSION) in msg
