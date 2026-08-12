"""Runtime db_url builders percent-encode special-character passwords."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import quote

import pytest

from aetherdialect._config import (
    MariaDBRuntimeConfig,
    MySQLRuntimeConfig,
    OracleRuntimeConfig,
    PostgresRuntimeConfig,
    RedshiftRuntimeConfig,
    SnowflakeRuntimeConfig,
    SQLServerRuntimeConfig,
)

_SPECIAL_PASSWORD = "P@ssw0rd!#with$pecials"
_ENCODED_PASSWORD = quote(_SPECIAL_PASSWORD, safe="")


@contextmanager
def _patched_attrs(target: Any, **values: object) -> Iterator[None]:
    saved = {name: getattr(target, name) for name in values}
    try:
        for name, value in values.items():
            setattr(target, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(target, name, value)


def _userinfo_authority(url: str) -> str:
    match = re.search(r"://([^/]+?)(?:/|\?|$)", url)
    assert match is not None, f"no authority in url: {url!r}"
    return match.group(1)


def _assert_password_quoted(url: str) -> None:
    authority = _userinfo_authority(url)
    assert ":" in authority and "@" in authority, f"expected user:password@host authority: {authority!r}"
    userinfo, _, _host = authority.rpartition("@")
    _, _, password = userinfo.partition(":")
    assert password == _ENCODED_PASSWORD, f"password segment {password!r} in {url!r}"
    assert _SPECIAL_PASSWORD not in password
    assert _SPECIAL_PASSWORD not in authority


def _postgres_url() -> str:
    with _patched_attrs(
        PostgresRuntimeConfig,
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        DATABASE="shop",
        HOST="localhost",
        PORT=5432,
    ):
        return PostgresRuntimeConfig.db_url()


def _mysql_url() -> str:
    with _patched_attrs(
        MySQLRuntimeConfig,
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        DATABASE="shop",
        HOST="localhost",
        PORT=3306,
    ):
        return MySQLRuntimeConfig.db_url()


def _mariadb_url() -> str:
    with _patched_attrs(
        MariaDBRuntimeConfig,
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        DATABASE="shop",
        HOST="localhost",
        PORT=3306,
    ):
        return MariaDBRuntimeConfig.db_url()


def _redshift_url() -> str:
    with _patched_attrs(
        RedshiftRuntimeConfig,
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        DATABASE="dev",
        HOST="example.redshift.amazonaws.com",
        PORT=5439,
        USE_IAM=False,
        CLUSTER_IDENTIFIER=None,
        WORKGROUP=None,
    ):
        return RedshiftRuntimeConfig.db_url()


def _sqlserver_url() -> str:
    with _patched_attrs(
        SQLServerRuntimeConfig,
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        DATABASE="shop",
        HOST="localhost",
        PORT=1433,
        AUTH_MODE="sql",
        DRIVER="ODBC Driver 18 for SQL Server",
    ):
        return SQLServerRuntimeConfig.db_url()


def _oracle_url() -> str:
    with _patched_attrs(
        OracleRuntimeConfig,
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        SERVICE_NAME="FREEPDB1",
        SID=None,
        HOST="localhost",
        PORT=1521,
        AUTH_MODE="password",
    ):
        return OracleRuntimeConfig.db_url()


def _snowflake_url() -> str:
    with _patched_attrs(
        SnowflakeRuntimeConfig,
        ACCOUNT="xy12345",
        USER="app",
        PASSWORD=_SPECIAL_PASSWORD,
        DATABASE="SHOP",
        SCHEMA="PUBLIC",
        WAREHOUSE=None,
        ROLE=None,
        AUTHENTICATOR=None,
        PRIVATE_KEY_PATH=None,
        OAUTH_TOKEN=None,
    ):
        return SnowflakeRuntimeConfig.db_url()


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine_name,build_url",
    [
        ("postgresql", _postgres_url),
        ("mysql", _mysql_url),
        ("mariadb", _mariadb_url),
        ("redshift", _redshift_url),
        ("sqlserver", _sqlserver_url),
        ("oracle", _oracle_url),
        ("snowflake", _snowflake_url),
    ],
)
def test_db_url_percent_encodes_special_password(engine_name: str, build_url: Callable[[], str]) -> None:
    url = build_url()
    assert isinstance(url, str) and url, engine_name
    _assert_password_quoted(url)
