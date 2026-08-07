# AetherDialect — The **deterministic** Text-to-SQL engine

<p align="center">
  <img src="https://github.com/dkecompany/aether-dialect/raw/main/docs/aether-logo.svg" alt="AetherDialect logo" width="120" />
</p>

`aetherdialect` turns analytical questions into read-only `SELECT` pipelines: a structured intent representation, multi-stage validation (including dialect `EXPLAIN`), template reuse from accepted answers, and bounded learning from rejections. The language model fills bounded slots in that intent; it does not author unconstrained SQL.

## Why this exists

Teams need answers from relational data without shipping opaque generated SQL. AetherDialect targets analysts and integrators who want a **repeatable** path from question to result: the same question can return cached SQL with no model round-trip, schema drift surfaces as an explicit migration stop instead of silent breakage, and every generated statement is checked against the catalog and engine before it runs.

## Install

```bash
pip install aetherdialect
pip install "aetherdialect[mysql]"        # MySQL
pip install "aetherdialect[mariadb]"      # MariaDB
pip install "aetherdialect[sqlserver]"    # SQL Server (pyodbc)
pip install "aetherdialect[postgresql]"   # PostgreSQL
pip install "aetherdialect[redshift]"     # Amazon Redshift
pip install "aetherdialect[databricks]"   # Databricks
pip install "aetherdialect[snowflake]"    # Snowflake
pip install "aetherdialect[bigquery]"     # Google BigQuery
pip install "aetherdialect[mysql,postgresql]"  # pick any subset
```

Requires Python 3.11 or newer. SQLite and DuckDB need no extra install. Configure the LLM and database via a TOML `config_file` (recommended) and/or process environment; the full key list lives in the [API reference](https://github.com/dkecompany/aether-dialect/blob/main/docs/API_REFERENCE.md).

## Quick start

**New here?** Follow the [Getting started guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/GETTING_STARTED.md): try the offline sandbox first, then wire any supported database with inlined TOML examples, first-run profiling expectations, and `run_interactive` vs `session()`.

**No database yet?** Try offline practice:

```python
from aetherdialect import AetherEngine

with AetherEngine.offline_sandbox() as sb:
    with sb.session() as session:
        step = session.accept_until_done("How many films are there?")
    print(step.sql)
```

See the [Sandbox guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/SANDBOX.md).

## What makes this different

- Constant-learning cache: exact `q_norm` reuse returns SQL with zero LLM calls; near-paraphrases (token Levenshtein at most 2) reuse the same template with one bounded LLM call that only extracts parameters. ([How it works](https://github.com/dkecompany/aether-dialect/blob/main/docs/HOW_IT_WORKS.md))

- Schema overrides are a JSON file you read, edit, and version. Every override (descriptions, roles, sensitivity, added or suppressed foreign keys, primary key endorsements) is replayed on every cache invalidation. ([API reference](https://github.com/dkecompany/aether-dialect/blob/main/docs/API_REFERENCE.md))

- Migration is never silent. When the catalog changes structurally, the engine writes a `schema_migration_map.json` skeleton and stops. You decide the action; it resumes. ([User guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/USER_GUIDE.md))

- Generated SQL passes through four validation layers (intent JSON, dialect AST, schema/catalog alignment, dialect EXPLAIN). The LLM never emits raw SQL; it fills bounded slots in a structured intent IR. ([Security](https://github.com/dkecompany/aether-dialect/blob/main/docs/SECURITY.md))

- Reader / writer split is built in. Many readers can ask questions; the engine drains `write_queue.jsonl` at the **start of every writer-mode turn** under the artifacts lock so learning persists without readers touching the partitioned template store files. ([Integrator guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/INTEGRATOR_GUIDE.md))

## At a glance

**Supported engine selections (11):** SQLite, DuckDB, CSV (`.csv` and `.xlsx`), MySQL, MariaDB, SQL Server, PostgreSQL, Amazon Redshift, Databricks, Snowflake, Google BigQuery. Install `aetherdialect[csv]` for file uploads. Set `[engine] selected` or `AETHERDIALECT_ENGINE` when more than one block is configured.

**SQL scope:** DML/DDL, set operators inside one statement (`UNION`/`EXCEPT`/`INTERSECT`), lateral joins, recursive CTEs, correlated subqueries beyond CTE shapes, row-skipping `OFFSET`/`FETCH`, and window/JSON outside the IR whitelist are refused. The engine never emits literal `EXISTS`, `NOT EXISTS`, or bare `DISTINCT ON` SQL — but existence (semi-join), absence and set-difference over keys (anti-join), nested boolean filters, and per-partition top-one are **supported** via first-class intent IR and compiled deterministically. Details: [Supported intent constructs](https://github.com/dkecompany/aether-dialect/blob/main/docs/SUPPORT_MATRIX.md#supported-intent-constructs) · [Refused constructs and reformulations](https://github.com/dkecompany/aether-dialect/blob/main/docs/SUPPORT_MATRIX.md#refused-constructs-and-reformulations).

**What reaches the LLM:** Prompt-safe schema metadata (visible table/column names, types, roles, descriptions, capped enum heads), the user question, bounded intent JSON, join-choice candidates, optional **notes file** and **DDL file** content when you configure them on `EngineContext`, and summarised failure feedback — not raw warehouse row dumps. Inventory: [Security — LLM context](https://github.com/dkecompany/aether-dialect/blob/main/docs/SECURITY.md#5-llm-context-inventory).

**What “safe” means here:** `SELECT`-only enforcement, forbidden-SQL regex, dialect AST validation, schema alignment, and `EXPLAIN` before execution; sensitivity tiers and deny lists gate what appears in prompts. This complements your database IAM and network controls; it does not replace them. Detail: [Security — Threat model](https://github.com/dkecompany/aether-dialect/blob/main/docs/SECURITY.md#1-threat-model).

**Production checklist:** least-privilege DB role; explicit stable `artifacts_dir` on durable storage; reviewed `notes_file` / `EngineContext.sql_file` content; `config_file` or env secrets not committed; one writer process per `artifacts_dir`; plan for `schema_migration_map.json` when the catalog changes.

No warehouse or LLM keys yet? [Sandbox guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/SANDBOX.md) → [Getting started](https://github.com/dkecompany/aether-dialect/blob/main/docs/GETTING_STARTED.md) → [User guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/USER_GUIDE.md).

## Reading order

| Doc | When to read it |
| --- | --- |
| [Getting started](https://github.com/dkecompany/aether-dialect/blob/main/docs/GETTING_STARTED.md) | First run: offline sandbox or warehouse TOML, construction wait, `run_interactive` vs `session()`. |
| [User guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/USER_GUIDE.md) | Operator manual: scope, notes, overrides, asking questions, migration, warmup, CSV upload validation, pitfalls — minimal code. |
| [Integrator guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/INTEGRATOR_GUIDE.md) | Embedding: suspend/terminal steps, reader/writer queue, multi-user deployment, observability. |
| [Sandbox guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/SANDBOX.md) | Offline practice with mock LLM and rental shop; production-shaped session API. |
| [API reference](https://github.com/dkecompany/aether-dialect/blob/main/docs/API_REFERENCE.md) | Exported types, TOML schema, methods, overrides JSON, diagnostics, exceptions. |
| [How it works](https://github.com/dkecompany/aether-dialect/blob/main/docs/HOW_IT_WORKS.md) | Conceptual pipeline: schema build, storage, question phases, learning, write queue. |
| [Security](https://github.com/dkecompany/aether-dialect/blob/main/docs/SECURITY.md) | Threat model, LLM disclosure inventory, sensitivity tiers, deny lists. |
| [Support matrix](https://github.com/dkecompany/aether-dialect/blob/main/docs/SUPPORT_MATRIX.md) | Per-engine capabilities, IR-unsupported constructs, dialect notes. |

## License

See [LICENSE](https://github.com/dkecompany/aether-dialect/blob/main/LICENSE).
