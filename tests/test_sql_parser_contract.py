"""Behavioural contract pinning the sqlglot surface the library depends on."""

from __future__ import annotations

import inspect

import pytest
import sqlglot
from sqlglot import exp, parse_one

DIALECT_TOKENS = (
    "bigquery",
    "databricks",
    "duckdb",
    "mysql",
    "postgres",
    "redshift",
    "snowflake",
    "sqlite",
    "tsql",
)

EXPRESSION_CLASSES = (
    exp.Add,
    exp.Alias,
    exp.And,
    exp.Anonymous,
    exp.Column,
    exp.Count,
    exp.CTE,
    exp.From,
    exp.Func,
    exp.Is,
    exp.Join,
    exp.Literal,
    exp.Mul,
    exp.Not,
    exp.Null,
    exp.Or,
    exp.Placeholder,
    exp.Select,
    exp.Sub,
    exp.Subquery,
    exp.Sum,
    exp.Table,
    exp.TableAlias,
    exp.Where,
    exp.With,
)


@pytest.mark.fast
def test_parser_surface_the_library_depends_on() -> None:
    """Pin sqlglot dialect tokens, expression nodes, and parse/generate APIs. This contract justifies the ``sqlglot>=29.0,<30`` ceiling in pyproject.toml: a major bump may rename dialect tokens, reshape expression constructors, or change ``parse_one`` / ``Expression.sql`` keyword behaviour used throughout the dialect and SQL-generation layers."""
    assert hasattr(sqlglot, "parse_one")
    assert callable(sqlglot.parse_one)
    assert "dialect" in inspect.signature(sqlglot.parse_one).parameters
    assert "read" in inspect.signature(sqlglot.parse_one).parameters

    for token in DIALECT_TOKENS:
        tree = parse_one("SELECT 1", dialect=token)
        assert isinstance(tree, exp.Select)
        rendered = tree.sql(dialect=token)
        assert "1" in rendered

    for cls in EXPRESSION_CLASSES:
        assert issubclass(cls, exp.Expression)

    join = exp.Join(
        this=exp.Table(this=exp.to_identifier("t")),
        on=exp.EQ(this=exp.Column(this=exp.to_identifier("a")), expression=exp.Column(this=exp.to_identifier("b"))),
        kind="INNER",
    )
    assert join.args.get("kind") == "INNER"

    chained = exp.And(this=exp.Column(this=exp.to_identifier("x")), expression=exp.Column(this=exp.to_identifier("y")))
    assert isinstance(chained, exp.And)

    placeholder_tree = parse_one("SELECT 'x'", dialect="duckdb")
    for literal in list(placeholder_tree.find_all(exp.Literal)):
        literal.replace(exp.Placeholder(this="p"))
    assert placeholder_tree.sql(dialect="duckdb")
