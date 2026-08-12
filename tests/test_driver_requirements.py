"""Driver import requirements per engine."""

from __future__ import annotations

import builtins
import re
from unittest.mock import patch

import pytest

from aetherdialect import ConfigError
from aetherdialect._constants import ENGINE_DRIVER_REQUIREMENTS
from aetherdialect._utils import require_driver


def _driver_import_names(engine_name: str) -> tuple[str, ...]:
    import_names, _, _extra_name = ENGINE_DRIVER_REQUIREMENTS[engine_name]
    if isinstance(import_names, str):
        return (import_names,)
    return tuple(import_names)


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine_name",
    sorted(ENGINE_DRIVER_REQUIREMENTS),
)
def test_missing_driver_names_package(engine_name: str) -> None:
    import_names = _driver_import_names(engine_name)
    _, distribution, _extra_name = ENGINE_DRIVER_REQUIREMENTS[engine_name]
    dist_label = distribution if isinstance(distribution, str) else distribution[0]
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        roots = {item.split(".", 1)[0] for item in import_names}
        if name in import_names or name.split(".", 1)[0] in roots:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        with pytest.raises(ConfigError, match=re.escape(dist_label)):
            require_driver(engine_name)


@pytest.mark.fast
def test_missing_driver_error_has_no_pip_install_recipe() -> None:
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "psycopg" or name.split(".", 1)[0] == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        with pytest.raises(ConfigError) as exc_info:
            require_driver("postgresql")
    assert "pip install" not in str(exc_info.value).lower()


@pytest.mark.fast
def test_postgresql_requires_psycopg_v3() -> None:
    real_import = builtins.__import__
    calls: list[str] = []

    def _tracked_import(name: str, *args: object, **kwargs: object) -> object:
        root = name.split(".", 1)[0]
        if root == "psycopg":
            calls.append(root)
            return object()
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_tracked_import):
        require_driver("postgresql")
    assert calls == ["psycopg"]

    def _block_psycopg(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] == "psycopg":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_block_psycopg):
        with pytest.raises(ConfigError) as excinfo:
            require_driver("postgresql")
    message = str(excinfo.value)
    assert "psycopg" in message
    assert "pip install" not in message.lower()
    assert "psycopg2" not in message
