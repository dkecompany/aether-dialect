# How it works

Conceptual end-to-end narrative: what happens between a user's question and a returned table. Public types and signatures: [API reference](API_REFERENCE.md). Operator semantics: [User guide](USER_GUIDE.md).

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → this file → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

The core idea is unchanged across every entry point: prefer reuse and rules over open-ended generation. Each turn is a sequence of cheap checks; the language model does small, bounded jobs under deterministic constraints from the schema graph and validators.

## Sections

| Section | Contents |
| --- | --- |
| [1. Configuration](#1-configuration) | Effective settings merge |
| [2. Schema build pipeline](#2-schema-build-pipeline) | Reflect, profile, classify |
| [3. Engine storage and artifact lifecycle](#3-engine-storage-and-artifact-lifecycle) | On-disk layout |
| [4. Migration](#4-migration) | Drift handling |
| [5. AetherEngine, roles, and modes](#5-aetherengine-roles-and-modes) | Instances and deployment shape |
| [6. AetherSpace](#6-aetherspace) | Knowledge narrowing |
| [7. Question pipeline](#7-question-pipeline) | Interpret through execute |
| [8. Learning model](#8-learning-model) | Templates and feedback |
| [9. Concurrent sessions](#9-concurrent-sessions-write-queue) | Reader/writer queue |

---

## 1. Configuration

Effective settings are built for each process without mutating `os.environ`. When **config_file** is provided, each flattened field that appears in the file is authoritative for its mapped environment key. This merged map drives database connection, LLM provider selection, timeouts, and execution limits.

## 2. Schema build pipeline

When an engine is constructed, it compiles a schema graph from the live database. The graph is built once per fingerprint and cached on disk.

- **Reflect catalog** — live warehouse introspection or optional DDL from `EngineContext.sql_file`.
- **Profiling** — read queries against live data to compute counts, roles, and samples.
- **Usability gating** — columns that fail statistical gates (for example extreme null rate, single-value, or sentinel-dominated distributions) are marked unusable and omitted from LLM-facing schema literals. Overrides may mark additional columns unusable with `"usable": false` but cannot re-enable profiler-omitted columns.
- **Classification** — LLM identifies table/column roles and sensitivity tiers from profiles and optional domain notes.
- **FK inference** — connects the graph using name matches and value-containment checks when catalog FKs are missing.

## 3. Engine storage and artifact lifecycle

All persisted state lives under an **engine storage directory**:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

The connection slug is derived from **database identity** (engine type, host, database name, and similar) — not from user, role, or session paths. Each distinct database connection gets its own directory and its own master engine context snapshot.

- **`artifact_manifest.json`** — tracks fingerprints for structural, profiling, and scope state.
- **`schema_graph.json.gz`** — compiled graph snapshot.
- **`intent_templates/`** — partitioned template store for learning from accepted questions.
- **`applied_overrides.json`** — user layer for correcting descriptions, roles, tiers, and usability.

## 4. Migration

Structural catalog drift stops initialization with `MigrationPendingError`. Operators edit `schema_migration_map.json` to choose **remap**, **destructive**, or **abort** ([User guide — Migration](USER_GUIDE.md#migration)).

## 5. AetherEngine, roles, and modes

- **AetherEngine** — root facade binding one database connection, one artifacts tree, and one compiled master graph (plus optional named engine contexts and AetherSpaces).
- **role** — `owner` builds and mutates shared artifacts; `consumer` pins to the owner's published schema snapshot and visible scope.
- **session mode** — `writer` (default) persists learning directly; `reader` enqueues learning events to `write_queue.jsonl` for a writer on the same artifacts directory to drain at the start of its next turn.

## 6. AetherSpace

An **AetherSpace** is a named knowledge subset over the master graph. It improves model focus and partitions template learning by space name. Named spaces inherit master graph construction; they do not rebuild or re-hash the catalog. `SpaceContext` supplies table/column allow and deny lists at question time only — not a substitute for database permissions ([Security — Threat model](SECURITY.md#1-threat-model)).

## 7. Question pipeline

Turn order for a single question:

1. **Intake** — validate and normalize question text.
2. **Reuse** — check the template store; direct reuse (≤2 token edit distance) skips parse LLM calls; larger changes run the full pipeline.
3. **Interpret** — LLM produces a natural-language analytical plan.
4. **Ground** — LLM binds the plan to schema identifiers.
5. **Compose** — LLM lowers logical intent into the intermediate representation (IR).
6. **Funnel** — deterministic repair and multi-tier validation (schema and semantic).
7. **Finalize** — join discovery and parameter binding.
8. **Execute** — generated SQL passes AST validation, schema alignment, EXPLAIN, and cost gates.

Programmatic callers observe **suspend** steps (`step.done == false`) and **terminal** steps (`step.done == true`) via `SessionStep.kind` ([Integrator guide — Suspend and terminal steps](INTEGRATOR_GUIDE.md#suspend-and-terminal-steps)).

## 8. Learning model

Accepted turns fold into the partitioned template store (keyed by normalized question text and AetherSpace). Future similar questions hit reuse before invoking parse LLM calls. Negative feedback is summarized and steers retry and template selection away from past mistakes.

## 9. Concurrent sessions (write queue)

Reader sessions defer durable learning by appending structured events to `write_queue.jsonl`. A writer session on the same artifacts directory drains the queue automatically at the start of each turn under the artifacts lock.

---

**See also:** [User guide](USER_GUIDE.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [Sandbox guide](SANDBOX.md) · [API reference](API_REFERENCE.md) · [Security](SECURITY.md) · [Support matrix](SUPPORT_MATRIX.md) · [README](../README.md#documentation)
