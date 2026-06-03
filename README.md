# AetherDialect — The **deterministic** Text-to-SQL engine

`aetherdialect` turns analytical questions into read-only `SELECT` pipelines: a structured intent representation, multi-stage validation (including dialect `EXPLAIN`), template reuse from accepted answers, and bounded learning from rejections. The language model fills bounded slots in that intent; it does not author unconstrained SQL.

## Why this exists

Teams need answers from relational data without shipping opaque generated SQL. AetherDialect targets analysts and integrators who want a **repeatable** path from question to result: the same question can return cached SQL with no model round-trip, schema drift surfaces as an explicit migration stop instead of silent breakage, and every generated statement is checked against the catalog and engine before it runs.

## Install

```bash
pip install aetherdialect
pip install "aetherdialect[postgresql]"
pip install "aetherdialect[databricks]"
pip install "aetherdialect[postgresql,databricks]"
```

Requires Python 3.10 or newer. Configure the LLM and database via a TOML `config_file` (recommended) and/or process environment; the full key list lives in the [API reference](https://github.com/dkecompany/aether-dialect/blob/main/docs/API_REFERENCE.md).

## Quick start

```python
from aetherdialect import SchemaContext, Text2SQL

t2s = Text2SQL(
    SchemaContext(),
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
)
t2s.run_interactive()
```

`run_interactive` prompts once per invocation; call it again for another question. For programmatic UIs, use `Text2SQL.session()` or `Text2SQL.asession()` and drive `SessionStep` objects — see the [Integrator guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/INTEGRATOR_GUIDE.md).

`dry_run_warmup` exercises a newline-delimited seed question file through validation and execution without persisting templates; see the [User guide — Seed warmup](https://github.com/dkecompany/aether-dialect/blob/main/docs/USER_GUIDE.md#seed-warmup).

## What makes this different

- Constant-learning cache: exact `q_norm` reuse returns SQL with zero LLM calls; near-paraphrases (token Levenshtein at most 2) reuse the same template with one bounded LLM call that only extracts parameters. ([How it works](https://github.com/dkecompany/aether-dialect/blob/main/docs/HOW_IT_WORKS.md))

- Schema overrides are a JSON file you read, edit, and version. Every override (descriptions, roles, sensitivity, added or suppressed foreign keys, primary key endorsements) is replayed on every cache invalidation. ([API reference](https://github.com/dkecompany/aether-dialect/blob/main/docs/API_REFERENCE.md))

- Migration is never silent. When the catalog changes structurally, the engine writes a `schema_migration_map.json` skeleton and stops. You decide the action; it resumes. ([User guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/USER_GUIDE.md))

- Generated SQL passes through four validation layers (intent JSON, dialect AST, schema/catalog alignment, dialect EXPLAIN). The LLM never emits raw SQL; it fills bounded slots in a structured intent IR. ([Security](https://github.com/dkecompany/aether-dialect/blob/main/docs/SECURITY.md))

- Reader / writer split is built in. Many readers can ask questions; the engine drains `write_queue.jsonl` at the **start of every writer-mode turn** under the artifacts lock so learning persists without readers touching the partitioned template store files. ([Integrator guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/INTEGRATOR_GUIDE.md))

## Documentation

| Doc                                                                                                                          | When to read it                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [User guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/USER_GUIDE.md)                                      | Install, first run, asking questions, notes, overrides, migration, seed warmup and dry-run warmup, pitfalls.                                                                                      |
| [Integrator guide](https://github.com/dkecompany/aether-dialect/blob/main/docs/INTEGRATOR_GUIDE.md)                          | Embedding patterns, sessions, multi-turn relay, threading, reader/writer split and queue, audit and diagnostics, cache reset.                                                                     |
| [API reference](https://github.com/dkecompany/aether-dialect/blob/main/docs/API_REFERENCE.md)                                | Types, `config_file` TOML schema, methods, schema overrides JSON, diagnostic codes, exceptions.                                                                                                   |
| [How it works](https://github.com/dkecompany/aether-dialect/blob/main/docs/HOW_IT_WORKS.md)                                  | Architecture diagrams, schema build, engine storage, question pipeline, migration, overrides, validation, learning model, configuration, observability, warmup/QSim, offline-mock design pointer. |
| [Offline testing and mock LLM (design)](https://github.com/dkecompany/aether-dialect/blob/main/docs/OFFLINE_AND_MOCK_LLM.md) | Planned mock LLM provider and fixture workflow for hermetic tests; links to `dev_workspace/mock.txt` (not on PyPI).                                                                               |
| [Security](https://github.com/dkecompany/aether-dialect/blob/main/docs/SECURITY.md)                                          | Threat model, LLM context inventory, on-disk inventory, sensitivity model, deny lists, raw-SQL impossibility, EXPLAIN gate, network.                                                              |
| [Support matrix](https://github.com/dkecompany/aether-dialect/blob/main/docs/SUPPORT_MATRIX.md)                              | Per-engine table, IR-unsupported constructs and reformulations.                                                                                                                                   |

## License

See [LICENSE](https://github.com/dkecompany/aether-dialect/blob/main/LICENSE).
