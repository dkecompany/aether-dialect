"""Tests for aetherspace and template-store space name path safety."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates import TemplateOps


@pytest.mark.fast
def test_dotdot_space_name_refused(tmp_path) -> None:
    artifacts_dir = str(tmp_path)
    for unsafe in ("..", ".", "../evil", "foo/bar", "foo\\bar", "has space", "UPPER"):
        with pytest.raises(ValueError, match="invalid template space name"):
            TemplateOps.template_store_dir_for_space(artifacts_dir, unsafe)
        with pytest.raises(ConfigError, match="invalid aetherspace name"):
            MainExecutionOps._aetherspace_path(artifacts_dir, unsafe)
