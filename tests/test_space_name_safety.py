"""Tests for aetherspace and template-store space name path safety."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps


@pytest.mark.fast
def test_dotdot_space_name_refused(tmp_path) -> None:
    artifacts_dir = str(tmp_path)
    for unsafe in ("..", ".", "../evil", "foo/bar", "foo\\bar", "has space"):
        with pytest.raises(ValueError, match="invalid template space name"):
            TemplateOps.template_store_dir_for_space(artifacts_dir, unsafe)
    # Allocator uids and mixed-case display names normalize to a lowercase path segment.
    assert TemplateOps.template_store_dir_for_space(artifacts_dir, "S0001").endswith("s0001")
    assert TemplateOps.template_store_dir_for_space(artifacts_dir, "UPPER").endswith("upper")
    for unsafe in ("..", ".", "../evil", "foo/bar", "foo\\bar", "has space"):
        with pytest.raises(ConfigError, match="invalid aetherspace uid"):
            MainExecutionOps._aetherspace_path(artifacts_dir, unsafe)
    # Slug uids are lowercased; display-name case is not path-unsafe for uid identity.
    assert MainExecutionOps._aetherspace_path(artifacts_dir, "UPPER").endswith("upper.json")
