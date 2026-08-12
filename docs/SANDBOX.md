# Sandbox guide

The **sandbox** is a zero-config offline environment. `Sandbox()` supplies the bundled **rental_shop** seed database in in-memory DuckDB, a sandbox LLM wired to recorded fixtures, a temporary artifacts directory, and a throwaway TOML it writes for itself. You do **not** provide `aetherdialect.toml`, warehouse credentials, an LLM API key, or a provider choice. Production wiring: [Getting started — Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse). Suspend semantics: [Integrator guide — The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → this document → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [About the environment](#about-the-environment) | What the sandbox supplies and the bundled data |
| [Enter and exit the sandbox](#enter-and-exit-the-sandbox) | Handle lifecycle, cleanup, options |
| [Ask questions](#ask-questions) | Session patterns |
| [Question sets](#question-sets) | Bundled questions and catalog helpers |
| [Owner vs consumer roles](#owner-vs-consumer-roles) | Engine role and session mode offline |
| [Reader/writer sessions](#readerwriter-sessions) | Shared artifacts demo |
| [Rejections and feedback](#rejections-and-feedback) | Bundled feedback samples |
| [Template reuse](#template-reuse) | Direct vs full parse |
| [Structure document format](#structure-document-format) | Engine-level structural document |
| [Federation declaration format](#federation-declaration-format) | Authoring cross-source joins and mappings |
| [Federation walkthrough](#federation-walkthrough) | Offline multi-member demo |
| [Structure document demo](#structure-document-demo) | Bundled structure demo file |
| [Migration demo](#migration-demo) | Toy rename walkthrough |
| [Column security](#column-security) | deny_columns and hidden sensitivity |
| [Named AetherSpaces](#named-aetherspaces) | Define spaces offline |
| [Views and partition pruning](#views-and-partition-pruning) | DuckDB-only demos |
| [Sandbox limits](#sandbox-limits) | Limits and unsupported paths |

---

## About the environment

`Sandbox()` automatically:

1. Extracts the bundled rental_shop corpus shipped inside the installed package.
2. Loads the in-memory DuckDB dataset from that corpus.
3. Seeds baseline schema artifacts when a bundled baseline is present.
4. Writes a temporary TOML that selects DuckDB `:memory:` and the **sandbox** LLM with the bundled fixtures file.
5. Pins mock fixture keys and schema literals from the bundle.
6. Returns a `Sandbox` handle whose `engine()` builds a ready owner `AetherEngine` on the default dataset.

No warehouse DSN, no API key, and no operator TOML are required for that path.

The bundled **rental_shop** dataset is a 34-table multi-category rental schema (films, books, games, logistics, promotions). Activity dates in the generated corpus are synthetic (predominantly **2022–2025**). Staff SSN and password columns are synthetic sample credentials used for sensitivity exercises. The sandbox is closed-world: every table, column, federation member, and bundled question is listed in [Sandbox data reference](SANDBOX_DATA_REFERENCE.md).

## Enter and exit the sandbox

```bash
pip install aetherdialect
```

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    engine = sandbox.engine()
    with engine.session(mode="writer") as session:
        step = session.accept_until_done("How many films are there?")
    print(step.sql)
```

The public entry points are **engine role** (`owner` vs `consumer` on the constructed engine) and **session mode** (`writer` vs `reader` on `session()`).

### Handle lifecycle

Always use `with Sandbox() as sandbox:` or call **`sandbox.close()`** when finished. Closing ends open session state, disposes the in-memory DuckDB connection(s), deletes the temporary extract directory, and removes the temporary artifacts directory.

Each `Sandbox()` handle is self-contained inside the package. Learning does not persist between separate handles.

### Production-shaped authoring

**`Sandbox()`** is the authoring entry point: you supply `EngineContext`, pass `sandbox.connection()` as `native_connection`, and call **`sandbox.adopt(engine)`** before opening a session when building engines manually. Adoption applies sandbox mode (sandbox fixtures, warmup suppression, and pinned schema literals). An engine built on a sandbox connection without adoption raises on `session()`.

```python
from aetherdialect import AetherEngine, EngineContext, EngineLimits, Sandbox

with Sandbox() as sandbox:
    engine = AetherEngine(
        EngineContext(allow_objects=frozenset({"customer", "rental"})),
        native_connection=sandbox.connection(),
        artifacts_dir=sandbox.artifacts_dir,
        config_file=sandbox.config_file,
        limits=EngineLimits(),
    )
    sandbox.adopt(engine)
    with engine.session(mode="writer") as session:
        step = session.accept_until_done("How many rentals are there?")
```

Behavioural caps use constructor `limits=` the same way as production. `[limits]` in a TOML file applies only when you load it explicitly with `EngineLimits.from_config_file(path)` and pass the result as `limits=`.

Federation authoring uses the same environment:

```python
from aetherdialect import FederationContext, Sandbox

with Sandbox() as sandbox:
    federation = sandbox.federation(
        "sandbox_rental_shop",
        context=FederationContext(allow_objects=frozenset({"payment"})),
    )
    with federation.session(mode="writer") as session:
        session.accept_until_done("How many payments are there?")
```

### Scoped engines

`Sandbox().engine(engine_context=...)` honours narrowed scope. A scope smaller than the full owner catalog derives the graph as a subset of the bundled seed.

```python
from aetherdialect import EngineContext, Sandbox

scope = EngineContext(allow_objects=frozenset({"customer", "rental"}))

with Sandbox() as sandbox:
    engine = sandbox.engine(engine_context=scope)
    inventory = engine.export_structure()
    assert "film" not in {t["name"] for t in inventory["tables"]}
```

### Multiple engines from one sandbox

Declare a different owner or consumer context on each `sandbox.engine(...)` call inside one `Sandbox` handle. The bundle is extracted once; each build re-derives its graph from the shared seed.

### Fixture alias resolution

When `notes_file` or `sql_file` names a path that does not exist on disk, the sandbox resolves common aliases to bundled fixtures and emits a diagnostic on `diagnostic_sink` (for example: `'notes.txt' resolved to bundled fixture 'rental_shop_notes.txt'`).

| User path | Bundled fixture |
| --- | --- |
| `notes.txt`, `catalog_notes.txt` | `rental_shop_notes.txt` |
| `schema.sql` | `rental_shop.sql` |

### Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `include` | `"tables"` | `"views"` applies bundled view definitions and reflects views only |
| `deny_columns` | none | Strip qualified columns from the graph before profiling ([Column security](#column-security)) |
| `engine_context` | full owner scope | Narrow tables, notes, and DDL paths the production way |
| `notes_file` / `sql_file` | from bundle | Override paths; aliases resolve as above |
| `llm_config` | bundled sandbox TOML | Non-sandbox provider selects live LLM mode |
| `maintainer_access` | `False` | Required for maintainer overrides |

Typical sandbox use loads bundled **CSV data and DDL only**. `artifacts_dir` exists for integrator-level tests that mimic production shared storage; **omit it for ordinary sandbox sessions**.

### Maintainer overrides

Maintainer overrides require `maintainer_access=True` on `Sandbox(...)`:

| Hook | Effect |
| --- | --- |
| `bundle_dir=` | Use an unpacked bundle directory instead of the shipped `data.zip` |
| `AETHERDIALECT_SANDBOX_DATA_ZIP` | Point at a custom zip before construction |
| `federation(..., members={...})` with filesystem paths | Supply member seeds by path instead of bundled dataset names |
| `seed_sql=` / `load_dataset(..., seed_sql=...)` | Alternate seed SQL instead of bundled CSV+DDL |

### LLM provider mode

Pass **`llm_config=`** to use your own TOML instead of the bundled sandbox provider. Inspect `sandbox.llm_mode` (`"sandbox"` or `"network"`) and `sandbox.uses_network` at construction. Sandbox mode replays bundled fixtures; network mode calls the configured provider. Unrecorded questions raise `MockFixtureMissingError`.

## Ask questions

Use the same session patterns as production ([Integrator guide — The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps)).

```python
step = session.accept_until_done("How many films are there?")
```

## Question sets

Curated question strings ship inside the installed package and are listed in [Sandbox data reference — Question corpus](SANDBOX_DATA_REFERENCE.md#question-corpus). Copy the strings you want into `session.ask(...)` / `session.accept_until_done(...)`.

| Tier | Use |
| --- | --- |
| `questions` | General questions on the owner/consumer engine |
| `validation_failures` | Questions expected to end in terminal validation errors |
| `feedback_samples` | Anchor question for the [rejections and feedback](#rejections-and-feedback) exercise |
| `views_questions` | Questions for `include="views"` |

## Owner vs consumer roles

The sandbox uses the same **owner/consumer** split as production.

| Role | Typical session mode | What loads | Learning |
| --- | --- | --- | --- |
| owner (default) | `writer` | Full **rental_shop** seed (~34 tables) and owner schema baseline from the bundled corpus | Templates and structure persist locally for that handle |
| consumer | `reader` | Same seed, narrowed by your `EngineContext.allow_objects` | Shared learning is session-local; consumers cannot call owner-only structural APIs |

```python
from aetherdialect import EngineContext, Sandbox

with Sandbox() as sandbox:
    engine = sandbox.engine(role="owner")
    with engine.session(mode="writer") as session:
        session.accept_until_done("How many films are there?")

consumer_scope = EngineContext(allow_objects=frozenset({"customer", "film", "payment", "rental"}))
with Sandbox() as sandbox:
    engine = sandbox.engine(consumer_scope, role="consumer")
    with engine.session(mode="reader") as session:
        session.accept_until_done("How many films are there?")
```

For a **permission-denied exercise**, pass a six-table `allow_objects` set so questions about `staff` fail at runtime:

```python
restricted = EngineContext(
    allow_objects=frozenset({"customer", "payment", "rental", "address", "city", "country"}),
)
with Sandbox() as sandbox:
    engine = sandbox.engine(restricted, role="consumer")
    with engine.session(mode="reader") as session:
        step = session.accept_until_done("Show me all staff salaries.")
    assert step.error is not None and step.error.code.value == "forbidden"
```

Inspect a narrowed graph before your first question:

```python
with Sandbox() as sandbox:
    engine = sandbox.engine(role="consumer")
    inventory = engine.export_structure()
    print(inventory["table_count"], len(inventory["tables"]))
```

## Reader/writer sessions

Production reader/writer sharing uses a durable `artifacts_dir` on disk. The sandbox does not require a durable artifacts directory for ordinary sessions.

Reader sessions keep learning session-local. An owner writer on the same shared artifacts root persists templates and feedback under the artifacts lock and drains `write_queue.jsonl` at writer turn start. Readers do **not** enqueue durable write-queue events.

```python
import tempfile
from aetherdialect import Sandbox

shared = tempfile.mkdtemp(prefix="sandbox_rw_")
q_text = "How many films are there?"

with Sandbox(artifacts_dir=shared, cleanup=False) as reader_sandbox:
    reader = reader_sandbox.engine(role="consumer")
    with reader.session(mode="reader") as session:
        session.accept_until_done(q_text)

with Sandbox(artifacts_dir=shared, cleanup=False) as writer_sandbox:
    writer = writer_sandbox.engine(role="owner")
    with writer.session(mode="writer") as session:
        session.accept_until_done(q_text)
```

Both sides must use the same `artifacts_dir` string. Mechanism: [How it works — Concurrent sessions and durability](HOW_IT_WORKS.md#8-concurrent-sessions-and-durability).

## Rejections and feedback

Reject with `step("n")`, then supply free-text feedback when `reply_shape == "free_text"`. The bundled corpus includes **one** feedback demo slot. See [Sandbox data reference — Question corpus](SANDBOX_DATA_REFERENCE.md#question-corpus).

## Template reuse

After you accept a question, ask again with different wording in the same session. Reuse is automatic.

1. **Direct reuse** — wording within the fuzzy token-edit budget (default ≤2) can replay the stored SQL path without a full interpret/ground/compose cycle.
2. **Full parse** — larger wording changes run through the mock LLM and the normal suspend loop.

Compare diagnostics (`REUSE_HIT` vs `REUSE_MISS`) between small and large wording changes.

## Structure document format

The structure document is the single read/write surface for tables, columns, types, keys, roles, sensitivity, and usability. Export returns a dict; apply accepts a dict. The library performs no disk I/O — callers own persistence (suggested file name: `schema_structure.json`). After `apply_structure`, the library persists `applied_structure.json` under the engine artifact tree.

Top-level keys: `table_count`, `tables`, `relationships`, `foreign_keys_add`, `foreign_keys_remove`, `primary_keys_add`, `primary_keys_remove`.

Each `tables[]` entry: `name`, `columns` (each with `name`, `data_type`, and optional `role`, `sensitivity`, `usable`), optional `primary_key`, `foreign_keys`, optional table `role`.

Annotated sandbox example (abridged):

```json
{
  "table_count": 1,
  "tables": [
    {
      "name": "staff",
      "columns": [
        {"name": "staff_id", "data_type": "INTEGER", "role": "identifier", "sensitivity": "none", "usable": true},
        {"name": "ssn", "data_type": "VARCHAR", "sensitivity": "hidden", "usable": true},
        {"name": "password", "data_type": "VARCHAR", "sensitivity": "hidden", "usable": true}
      ],
      "primary_key": ["staff_id"]
    }
  ],
  "relationships": [],
  "foreign_keys_add": [],
  "foreign_keys_remove": [],
  "primary_keys_add": [],
  "primary_keys_remove": []
}
```

Round trip:

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    engine = sandbox.engine()
    doc = engine.export_structure()
    # edit doc in memory
    engine.apply_structure(doc)
```

Full field reference: [API reference — Structure document](API_REFERENCE.md#structure-document).

## Federation declaration format

Author a federation declaration dict per federation. Pass its path or the parsed dict to `AetherFederation(..., declaration=...)`. Fields `sources` and `table_namespace` are composed from registered engines at construction — do not author them.

| Field | Meaning |
| --- | --- |
| `federation_id` | Stable federation name; must match the `AetherFederation` constructor `name` |
| `cross_source_joins[]` | Cross-member joins: `left` / `right` (`table.column`), `kind`, `logical_key` |
| `aliases` | Optional map of alias → `{source, table}` |
| `coordinator` | Coordinator caps (`row_cap`, `default_source_row_cap`, `default_source_timeout_ms`, `semijoin_key_cap`, `spill_row_threshold`, ...) |
| `logical_columns[]` | Optional unified column keys across members |
| `logical_tables[]` | Optional union/replica logical tables across members |

Annotated sandbox example (bundled as `federation_declaration.json`):

```json
{
  "federation_id": "sandbox_rental_shop",
  "aliases": {},
  "coordinator": {
    "row_cap": 500000,
    "default_source_row_cap": 500000,
    "default_source_timeout_ms": 30000,
    "semijoin_key_cap": 50000,
    "spill_row_threshold": 50000
  },
  "cross_source_joins": [
    {
      "kind": "inner",
      "left": "rental.inventory_id",
      "logical_key": "inventory_id",
      "right": "inventory.inventory_id"
    }
  ],
  "logical_columns": [
    {
      "logical": "inventory_id",
      "members": ["rental.inventory_id", "inventory.inventory_id"],
      "role": "join_key",
      "unify_in_graph": true
    }
  ],
  "logical_tables": [
    {
      "logical": "payment",
      "semantics": "union",
      "members": [
        {"source": "storefront", "table": "payment", "columns": {}},
        {"source": "catalog", "table": "payment", "columns": {}}
      ]
    }
  ]
}
```

```python
with Sandbox() as sandbox:
    federation = sandbox.federation("sandbox_rental_shop")
    doc = federation.export_federation()
    federation.apply_federation(doc)
```

Full nested key set: [API reference — Federation documents](API_REFERENCE.md#federation-documents).

## Federation walkthrough

Call `Sandbox().federation("sandbox_rental_shop")` to load the four partition seeds (`storefront`, `catalog`, `logistics`, and `crm`) as DuckDB members on separate in-memory connections with per-member artifact trees, then construct `AetherFederation` from the bundled declaration. The declaration wires:

- A row-partitioned `payment` **union** across `storefront` and `catalog`.
- `customer` and `staff` **replicas** with `storefront` authoritative.
- Four declared cross-source joins linking rentals, inventory, purchase lines, deliveries, and promotion redemptions across members.

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    federation = sandbox.federation("sandbox_rental_shop")
    with federation.session(mode="writer") as session:
        step = session.accept_until_done("How many rentals are linked to film titles?")
    print(step.sql)  # dict[str, str] on multi-member turns
```

Typical scenarios covered by the bundled federation declaration:

1. A cross-source question answered with the same `SessionStep` shape as single-source.
2. Operator-confirmed mapping on the boundary join key.
3. A `union` logical table without double counting.
4. A `replica` query that resolves to the authoritative member.
5. Configuration rejection when a cross-source key is not `sensitivity == none`.
6. Scope-style rejection for unsupported cross-source aggregates.
7. A federation AetherSpace spanning all four catalogs.

### Annotated declaration excerpt

```json
{
  "federation_id": "sandbox_rental_shop",
  "cross_source_joins": [
    { "left": "rental.inventory_id", "right": "inventory.inventory_id", "kind": "inner", "logical_key": "inventory_id" },
    { "left": "delivery.rental_id", "right": "rental.rental_id", "kind": "inner", "logical_key": "rental_id" }
  ],
  "logical_tables": [
    {
      "logical": "payment",
      "semantics": "union",
      "members": [
        { "source": "storefront", "table": "payment", "columns": {} },
        { "source": "catalog", "table": "payment", "columns": {} }
      ]
    },
    {
      "logical": "customer",
      "semantics": "replica",
      "authoritative_source": "storefront",
      "members": [
        { "source": "storefront", "table": "customer", "columns": {} },
        { "source": "crm", "table": "customer", "columns": {} }
      ]
    }
  ]
}
```

## Structure document demo

Structural edits use `export_structure()` / `apply_structure(document)` — the same production path. Demo content in [Sandbox data reference — Structure and sensitivity fixtures](SANDBOX_DATA_REFERENCE.md#structure-and-sensitivity-fixtures):

```python
from aetherdialect import Sandbox

demo_doc = {...}  # from schema_structure_demo.json in data reference

with Sandbox() as sandbox:
    engine = sandbox.engine()
    engine.apply_structure(demo_doc)
```

| Target | Effect |
| --- | --- |
| `staff.ssn`, `staff.password` | sensitivity → **hidden** |
| `film` | analyst description via knowledge layer (`export_knowledge` / `apply_knowledge`) |

After applying, try a `validation_failures` question targeting `staff.ssn`. Compare owner vs consumer roles — consumers cannot call owner-only structural APIs. Sensitivity tiers: [User guide — Sensitivity classification](USER_GUIDE.md#sensitivity-classification).

## Migration demo

The shipped corpus includes a toy `item.title` → `item_title` rename under `migration_demo/`:

```python
with Sandbox() as sandbox:
    preview = sandbox.preview_migration_corpus_variant()
    print(preview.tier, preview.affected_columns)
    sandbox.apply_migration_corpus_variant()
    engine = sandbox.engine()
    with engine.session() as session:
        session.accept_until_done("How many books do we have?")
```

Production workflow: [User guide — Migration](USER_GUIDE.md#migration).

## Column security

Two mechanisms exercise column security in the sandbox. Sensitivity tier definitions: [User guide — Sensitivity classification](USER_GUIDE.md#sensitivity-classification). Enforcement: [Security — Sensitivity tags](SECURITY.md#6-sensitivity-tags).

| Mechanism | Where set | Effect |
| --- | --- | --- |
| `deny_columns` on `Sandbox().engine(deny_columns=...)` or `EngineContext` | constructor / `engine()` | Column is **removed from the schema graph** before profiling |
| `sensitivity: hidden` in structure document | [Structure document demo](#structure-document-demo) | Column remains in the graph but is blocked from prompts and validation |

```python
with Sandbox() as sandbox:
    engine = sandbox.engine(EngineContext(deny_columns=frozenset({"staff.ssn"})))
    with engine.session() as session:
        step = session.ask("Show payroll deductions by employee SSN.")
```

After applying the structure demo, try a `validation_failures` question targeting `staff.ssn` to see the terminal refusal.

## Named AetherSpaces

Offline sessions default to the default space (`space=None` or `engine.default_space_uid`). The sandbox ships four named spaces aligned to federation partition members — `storefront`, `catalog`, `logistics`, and `crm`.

```python
from aetherdialect import Sandbox, SpaceContext

with Sandbox() as sandbox:
    engine = sandbox.engine()
    space = engine.aetherspace(
        "catalog",
        space_context=SpaceContext(
            tables=frozenset({"item", "film", "category", "item_category"}),
        ),
    )
    with engine.session(space=space.uid) as session:
        session.accept_until_done("How many films are in the Horror category?")
```

Compare answers across spaces to see how knowledge narrowing changes results. Bundled questions: [Sandbox data reference — Member-space question subsets](SANDBOX_DATA_REFERENCE.md#member-space-question-subsets). Conceptual guide: [User guide — AetherSpace](USER_GUIDE.md#aetherspace).

## Views and partition pruning

The bundled DuckDB seed includes analytical views and partition metadata on `rental.rental_date`. Use `Sandbox().engine(include="views")` to run against those views. Production: `EngineContext(include="views")` — [User guide — EngineContext](USER_GUIDE.md#enginecontext).

## Sandbox limits

- **Seed warmup** and **QSim** raise `ConfigError` on sandbox instances.
- Questions outside the recorded mock corpus raise `MockFixtureMissingError` — coverage is limited to the bundled question list.
- Arbitrary migration maps or structure documents beyond the bundled demos are not supported in the sandbox.

### Production stages not exercised offline

| Stage | What production does | Sandbox behaviour |
| --- | --- | --- |
| `live_reflection_and_profiling` | Live catalog reflection and column profiling | Replays a frozen `schema_graph.json.gz` baseline |
| `probe_mismatch_partial_rebuild` | DDL probe mismatch triggers structural diff | Baseline fingerprints match the bundle |
| `cold_build_descriptions_and_classification` | LLM description generation on first build | Descriptions and roles come from the recorded baseline |
| `member_cold_reflect_profile_and_member_drift_migration_pending` | Member cold reflect/profile and member-drift migration | Sandbox seeds frozen member baselines |
| `warmup_and_question_simulation` | Seed warmup and QSim | Blocked via production-API guard |
| `model_turns_outside_recorded_fixtures` | Arbitrary LLM turns for unseen questions | Sandbox provider replays recorded fixtures only |

When you move to production, follow [Getting started — Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse).

---

**See also:** [Sandbox data reference](SANDBOX_DATA_REFERENCE.md) | [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [API reference](API_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
