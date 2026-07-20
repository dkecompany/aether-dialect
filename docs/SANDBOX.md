# Sandbox guide

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → this guide → [API reference](API_REFERENCE.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

The **sandbox** is an offline practice environment: bundled **rental_shop** seed data in in-memory DuckDB, a mock LLM, and no warehouse or API credentials. The session API matches production — `ask`, `step`, suspend loops, reader/writer modes, schema overrides, and named AetherSpaces. Production setup: [Getting started](GETTING_STARTED.md). Suspend semantics: [Integrator guide — Suspend and terminal steps](INTEGRATOR_GUIDE.md#suspend-and-terminal-steps).

## Sections

| Section | Contents |
| --- | --- |
| [About the data](#about-the-data) | Bundled schema and licenses |
| [Enter the sandbox](#enter-the-sandbox) | Handle lifecycle and cleanup |
| [Ask questions](#ask-questions) | Session patterns |
| [Question sets](#question-sets) | Practice questions and catalog helpers |
| [Owner vs consumer presets](#owner-vs-consumer-presets) | Roles in offline mode |
| [Reader/writer queue](#readerwriter-queue) | Shared artifacts demo |
| [Rejections and feedback](#rejections-and-feedback) | Bundled feedback samples |
| [Template reuse](#template-reuse) | Direct vs full parse |
| [Schema overrides demo](#schema-overrides-demo) | Bundled override file |
| [Migration demo](#migration-demo) | Toy rename walkthrough |
| [Sensitivity](#sensitivity) | Hidden-column exercises |
| [Column security](#column-security) | deny_columns practice |
| [Named AetherSpaces](#named-aetherspaces) | Define spaces offline |
| [Views and partition pruning](#views-and-partition-pruning) | DuckDB-only demos |
| [What you cannot do](#what-you-cannot-do-in-the-sandbox) | Sandbox limits |
| [Exit checklist](#exit-checklist) | Checklist |

---

## About the data

The bundled **rental_shop** dataset is a 34-table multi-category rental schema (films, books, games, logistics, promotions). Activity dates are synthetic, predominantly **2022–2025**.

| Content | Source | License |
| ------- | ------ | ------- |
| Film titles, actor/customer/staff names | Wikidata + name-dataset lexicons (frozen in bundle) | CC0 / MIT |
| Countries, cities | GeoNames city sample + synthetic addresses | GeoNames CC-BY |
| Book / game catalog names | [Open Library](https://openlibrary.org/) + Zenodo game sample | CC0 |
| Publishers, suppliers, courier/warehouse names | Frozen synthetic lists | bundled |
| Staff SSN, passwords | Synthetic sample credentials | bundled |

## Enter the sandbox

Install DuckDB support:

```bash
pip install "aetherdialect[duckdb]"
```

```python
from aetherdialect import AetherEngine

with AetherEngine.offline_sandbox() as sb:
    with sb.session() as session:
        step = session.accept_until_done("How many films are there?")
    print(step.sql)
```

### Handle lifecycle

Each `offline_sandbox()` call:

1. **Wipes** any prior temp artifacts owned by a previous handle (when using the default temp `artifacts_dir`).
2. Unpacks the bundled corpus, loads seed SQL into `:memory:` DuckDB, seeds baseline schema artifacts, and wires the mock LLM.
3. Returns a `SandboxHandle` where **`sb.engine`** is the underlying `AetherEngine` and **`sb.session(...)`** is a shortcut for `sb.engine.session(...)`.

Always use `with AetherEngine.offline_sandbox() as sb:` or call **`sb.close()`** when finished. Closing:

- Ends open session state on the sandbox engine.
- Disposes the in-memory DuckDB connection.
- Deletes the temporary extract directory and, when the handle owns it, the temporary `artifacts_dir`.
- Removes working-directory sidecars registered through the handle (for example `schema_overrides.json` from `apply_bundled_schema_overrides()`).

The next entry rebuilds everything from the bundled corpus. Nothing persists between handles unless you pass an explicit shared `artifacts_dir`.

Optional parameters:

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `preset` | `"owner_writer"` | `"consumer_reader"` for narrowed scope and consumer role |
| `artifacts_dir` | temp directory owned by the handle | Share learning between sandbox instances |
| `restricted_consumer=True` | off | Smaller allow-list for permission-denied exercises |
| `deny_columns=frozenset({...})` | none | Column-security practice ([Column security](#column-security)) |

Health check: `AetherEngine.sandbox_doctor()` before shipping; `AetherEngine.assert_sandbox_complete()` in CI.

## Ask questions

Use the same session patterns as production ([Integrator guide](INTEGRATOR_GUIDE.md)).

**Auto-confirm suspends:**

```python
step = session.accept_until_done("How many films are there?")
```

**Manual loop:**

```python
step = session.ask("How many films are there?")
while not step.done:
    if step.reply_shape == "yes_no":
        step = session.step("y")
    elif step.reply_shape == "free_text":
        step = session.step("looks good")
print(step.sql)
```

## Question sets

Curated practice strings ship inside the installed package. Load them through `AetherEngine.sandbox_questions()`; do not hard-code long question lists in your scripts. Paraphrase/reuse demos, validation-failure examples, and feedback rejection text live in `sandbox_catalog.json` (`sandbox_paraphrase_pairs()`, `sandbox_validation_failure_demo()`, `sandbox_feedback_demo()`).

```python
questions = AetherEngine.sandbox_questions()
pairs = AetherEngine.sandbox_paraphrase_pairs()
failures = AetherEngine.sandbox_validation_failure_demo()
feedback = AetherEngine.sandbox_feedback_demo()
```

| API | Use |
| --- | --- |
| `sandbox_questions()` | All bundled practice questions |
| `sandbox_paraphrase_pairs()` | Canonical→paraphrase wordings recorded during corpus build |
| `sandbox_validation_failure_demo()` | Questions expected to end in terminal validation errors |
| `sandbox_feedback_demo()` | Anchor question + allowed rejection text for the feedback recipe |

```python
with AetherEngine.offline_sandbox() as sb:
    with sb.session() as session:
        for question in AetherEngine.sandbox_questions():
            step = session.accept_until_done(question)
            session.reset()
```

## Owner vs consumer presets

**Owner writer** (default): full schema visibility; writer mode and local schema overrides.

**Consumer reader** (`preset="consumer_reader"`): narrowed `allow_objects`; reader mode enqueues learning; writer mode raises `OwnerOnlyOperationError`.

```python
with AetherEngine.offline_sandbox(preset="consumer_reader") as sb:
    with sb.session(mode="reader") as session:
        session.accept_until_done("How many films are there?")
```

Pass `restricted_consumer=True` for a smaller allow-list.

## Reader/writer queue

Readers append events to `write_queue.jsonl`; an owner writer on the same `artifacts_dir` drains the queue. Resolve the path with `sb.engine.write_queue_path`.

```python
import tempfile
from aetherdialect import AetherEngine

shared = tempfile.mkdtemp(prefix="sandbox_rw_")
q_text = AetherEngine.sandbox_questions()[0]

with AetherEngine.offline_sandbox(preset="consumer_reader", artifacts_dir=shared) as reader:
    with reader.session(mode="reader") as session:
        session.accept_until_done(q_text)

with AetherEngine.offline_sandbox(preset="owner_writer", artifacts_dir=shared) as writer:
    with writer.session(mode="writer") as session:
        session.accept_until_done(q_text)
```

Both sides must use the same `artifacts_dir` string.

## Rejections and feedback

Reject with `step("n")`, then supply free-text feedback when `reply_shape == "free_text"`. The bundled corpus includes **one** feedback demo slot: the anchor question accepts only the exact rejection string from `AetherEngine.sandbox_feedback_demo()` — arbitrary feedback text fails at runtime by design.

## Template reuse

After you accept a question, ask again with different wording in the same session. Reuse is automatic — there is no separate API.

1. **Direct reuse** — wording within ≤2 normalized token edit distance replays the stored SQL path with no full parse LLM calls. `sandbox_paraphrase_pairs()` includes year variants (for example 2025 vs 2026 rental counts) recorded during corpus build to demonstrate parameter substitution on an accepted template.
2. **Full parse** — larger wording changes run through the mock LLM and the normal suspend loop.

Ask normally; compare diagnostics (`REUSE_HIT` vs `REUSE_MISS`) between small and large wording changes ([User guide — Asking a question](USER_GUIDE.md#asking-a-question)).

## Schema overrides demo

```python
with AetherEngine.offline_sandbox() as sb:
    sb.apply_bundled_schema_overrides()
```

Copies bundled demo JSON to the working directory and applies overrides (owners) or enqueues proposals (consumers). Overrides are manual — there is no list/discovery API. The bundled demo marks **`staff.password`** and **`staff.ssn`** as **hidden** for sensitivity exercises ([Security — Sensitivity tags](SECURITY.md#3-sensitivity-tags)). Usability flips are **off-only** (`"usable": false`).

## Migration demo

The bundled corpus includes a toy `item.title` → `item_title` rename with pre-migration artifacts:

1. Open a sandbox configured for the migration demo seed.
2. Construction detects drift, writes `schema_migration_map.json`, and stops with `MigrationPendingError`.
3. Complete the map with `"action": "remap"` and the column rename entry, then reconstruct.

Production workflow: [User guide — Migration](USER_GUIDE.md#migration).

## Sensitivity

Canonical tier definitions: [Security — Sensitivity tags](SECURITY.md#3-sensitivity-tags). After `apply_bundled_schema_overrides()`, try questions from `AetherEngine.sandbox_validation_failure_demo()` to see terminal errors when targeting hidden columns such as `staff.ssn`.

## Column security

`EngineContext.deny_columns` removes columns from the in-memory graph before profiling — stronger than **hidden** sensitivity.

```python
with AetherEngine.offline_sandbox(deny_columns=frozenset({"staff.ssn"})) as sb:
    with sb.session() as session:
        step = session.ask("Show payroll deductions by employee SSN.")
```

## Named AetherSpaces

Offline sessions default to `space="master"`. Define named spaces on the owner engine the same way as production:

```python
from aetherdialect import AetherEngine, SpaceContext

with AetherEngine.offline_sandbox() as sb:
    sb.engine.aetherspace(
        "catalog",
        space_context=SpaceContext(
            tables=frozenset({"item", "film", "category", "item_category"}),
        ),
        notes_file="path/to/catalog_notes.txt",  # bundled second-space demo notes in corpus
    )
    with sb.session(space="catalog") as session:
        session.accept_until_done("How many films are in the Horror category?")
    with sb.session(space="master") as session:
        session.accept_until_done("How many films are in the Horror category?")
```

Compare answers to see how knowledge narrowing changes results. Template learning is **per-space**. Conceptual guide: [User guide — AetherSpace](USER_GUIDE.md#aetherspace).

## Views and partition pruning

The bundled DuckDB seed includes analytical views (`active_customer_v`, `store_revenue_v`, `film_catalog_v`) and partition metadata on `rental.rental_date`. Ask questions that filter on rental dates or query revenue views to exercise view scope and partition predicate injection.

For production warehouses, set `EngineContext(include="views")` or `"both"` with your own view DDL — [User guide — EngineContext](USER_GUIDE.md#enginecontext).

## What you cannot do in the sandbox

- **Seed warmup** and **QSim** — `ConfigError` on sandbox instances.
- Author arbitrary migration maps or override files beyond the bundled demos.
- Expect answers for questions outside the recorded mock corpus.

## Exit checklist

| Sandbox | Production |
| ------- | ---------- |
| `AetherEngine.offline_sandbox()` | `AetherEngine(EngineContext(...), artifacts_dir=…, config_file=…)` |
| Mock LLM | `[llm] provider = "openai"` or `"azure"` in TOML |
| In-memory DuckDB | Your warehouse connection |
| `preset="consumer_reader"` | `role="consumer"` + `allow_objects` |
| Temp extract dir | Durable `artifacts_dir` on shared storage |

1. Point `AetherEngine` at your warehouse and durable artifacts directory.
2. Configure real LLM credentials ([User guide — Configure](USER_GUIDE.md#configure)).
3. One writer per `artifacts_dir`; readers share the path ([Integrator guide — Reader and writer split](INTEGRATOR_GUIDE.md#reader-and-writer-split)).
4. Plan `schema_migration_map.json` when the catalog changes.
5. Run seed warmup / QSim only outside the sandbox.

Corpus rebuild (maintainers): [scripts/README — Sandbox corpus](../scripts/README.md#sandbox-corpus).

---

**See also:** [Getting started](GETTING_STARTED.md) · [User guide](USER_GUIDE.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [API reference](API_REFERENCE.md) · [How it works](HOW_IT_WORKS.md) · [Security](SECURITY.md) · [Support matrix](SUPPORT_MATRIX.md) · [README](../README.md#documentation)
