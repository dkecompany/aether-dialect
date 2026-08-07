"""Fast unit tests for federation partition load verification (mocked DB)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_LOADER = _SCRIPTS / "load_rental_shop_engines.py"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load_loader_module():
    spec = importlib.util.spec_from_file_location("load_rental_shop_engines_test", _LOADER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_rental_shop_engines_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loader():
    return _load_loader_module()


@pytest.mark.fast
def test_cmd_federation_verify_exists(loader) -> None:
    assert hasattr(loader, "_cmd_federation_verify")


@pytest.mark.fast
def test_federation_load_validates_env_before_connect(loader) -> None:
    args = argparse.Namespace(
        env_file=Path("/nonexistent/env.env"),
        federation_load="storefront",
        drop_first=False,
        csv_dir=_SCRIPTS / "data" / "rental_shop_csvs",
        ddl=_SCRIPTS / "data" / "rental_shop.sql",
        database=None,
        schema=None,
    )
    with (
        patch.object(loader, "load_env_file"),
        patch.dict(os.environ, {}, clear=True),
        patch.object(loader, "_load_federation_postgresql_partition") as load_pg,
        patch.object(loader, "_cmd_federation_verify"),
    ):
        with pytest.raises(SystemExit, match="Federation load/verify requires complete database credentials"):
            loader._cmd_federation_load(args)
    load_pg.assert_not_called()


@pytest.mark.fast
def test_federation_verify_validates_env_before_connect(loader) -> None:
    args = argparse.Namespace(
        env_file=Path("/nonexistent/env.env"),
        federation_verify="catalog",
        federation_load=None,
        schema=None,
        database=None,
    )
    with (
        patch.object(loader, "load_env_file"),
        patch.dict(os.environ, {}, clear=True),
        patch.object(loader, "_verify_federation_mysql_partition") as verify_mysql,
    ):
        with pytest.raises(SystemExit, match="Federation load/verify requires complete database credentials"):
            loader._cmd_federation_verify(args)
    verify_mysql.assert_not_called()


@pytest.mark.fast
def test_env_example_grant_script_path_exists() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "env.example.env").read_text(encoding="utf-8")
    assert "scripts/sql/grant_pguser2_rental_shop.sql" in text
    assert "grant_pguser2_consumer" not in text
    assert (repo / "scripts" / "sql" / "grant_pguser2_rental_shop.sql").is_file()


@pytest.mark.fast
def test_federation_load_invokes_verify(loader) -> None:
    args = argparse.Namespace(
        env_file=Path("/nonexistent/env.env"),
        federation_load="all",
        drop_first=False,
        csv_dir=_SCRIPTS / "data" / "rental_shop_csvs",
        ddl=_SCRIPTS / "data" / "rental_shop.sql",
        database=None,
        schema=None,
    )
    with (
        patch.object(loader, "load_env_file"),
        patch.object(loader, "_validate_federation_env"),
        patch.object(loader, "_load_federation_postgresql_partition"),
        patch.object(loader, "_load_federation_mysql_partition"),
        patch.object(loader, "_cmd_federation_verify") as verify_mock,
    ):
        loader._cmd_federation_load(args)
    verify_mock.assert_called_once_with(args)


@pytest.mark.fast
def test_federation_verify_postgresql_partition_counts_partition_tables(loader) -> None:
    from sandbox_corpus import federation_partition_tables

    partition = federation_partition_tables("storefront")
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar_one.side_effect = [10] * len(partition)
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    args = argparse.Namespace(env_file=Path("/nonexistent/env.env"), schema=None, database=None)
    pg_cfg = loader.PostgresRuntimeConfig
    with (
        patch.object(loader, "load_env_file"),
        patch.object(pg_cfg, "apply_environment"),
        patch.object(pg_cfg, "db_url", return_value="postgresql+psycopg://localhost/test"),
        patch.object(loader, "create_engine", return_value=mock_engine),
    ):
        loader._verify_federation_postgresql_partition(
            args,
            source_id="storefront",
            schema="rental_shop_fed_storefront",
        )

    assert mock_conn.execute.call_count == len(partition)


@pytest.mark.fast
def test_federation_verify_mysql_partition_raises_on_empty_table(loader) -> None:
    from sandbox_corpus import federation_partition_tables

    partition = federation_partition_tables("catalog")
    empty_table = sorted(partition)[0]
    counts = {table: 0 if table == empty_table else 5 for table in partition}
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar_one.side_effect = [counts[t] for t in sorted(partition)]

    mock_engine = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_conn
    mock_engine.connect.return_value = mock_ctx

    args = argparse.Namespace(env_file=Path("/nonexistent/env.env"))
    runtime_cls = loader.MySQLRuntimeConfig
    with (
        patch.object(loader, "load_env_file"),
        patch.object(runtime_cls, "apply_environment"),
        patch.object(runtime_cls, "db_url", return_value="mysql+pymysql://localhost/test"),
        patch.object(runtime_cls, "connect_args", return_value={}),
        patch.object(loader, "create_engine", return_value=mock_engine),
    ):
        with pytest.raises(SystemExit, match="missing or empty tables"):
            loader._verify_federation_mysql_partition(
                args,
                source_id="catalog",
                database="rental_shop_fed_catalog",
                runtime_cls=runtime_cls,
            )


@pytest.mark.fast
def test_main_federation_verify_flag_runs_verify_only(loader) -> None:
    with (
        patch.object(loader, "_cmd_federation_verify") as verify_mock,
        patch.object(loader, "_cmd_federation_load") as load_mock,
        patch.object(loader, "load_env_file"),
        patch("sys.argv", ["load_rental_shop_engines.py", "--federation-verify", "storefront"]),
    ):
        loader.main()
    verify_mock.assert_called_once()
    load_mock.assert_not_called()
