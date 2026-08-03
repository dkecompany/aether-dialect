# Sandbox guide

The **sandbox** is an offline practice environment that needs **no configuration from you**. `AetherEngine.offline_sandbox()` supplies the bundled **rental_shop** seed database in in-memory DuckDB, a mock LLM wired to recorded fixtures, a temporary artifacts directory, and a throwaway TOML it writes for itself. You do **not** provide `aetherdialect.toml`, warehouse credentials, an LLM API key, or a provider choice. Production wiring: [Getting started - Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse). Suspend semantics: [Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).

**Reading order:** [README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [User guide](USER_GUIDE.md) -> [Integrator guide](INTEGRATOR_GUIDE.md) -> this document -> [API reference](API_REFERENCE.md) -> [How it works](HOW_IT_WORKS.md) -> [Security](SECURITY.md) -> [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Sandbox data reference](SANDBOX_DATA_REFERENCE.md) | Full table, view, federation, and question inventory |
| [About the environment](#about-the-environment) | What the sandbox supplies and the bundled data |
| [Enter and exit the sandbox](#enter-and-exit-the-sandbox) | Handle lifecycle, cleanup, options |
| [Ask questions](#ask-questions) | Session patterns |
| [Question sets](#question-sets) | Practice questions and catalog helpers |
| [Owner vs consumer roles](#owner-vs-consumer-roles) | Engine role and session mode in offline mode |
| [Reader/writer queue](#readerwriter-queue) | Shared artifacts demo |
| [Rejections and feedback](#rejections-and-feedback) | Bundled feedback samples |
| [Template reuse](#template-reuse) | Direct vs full parse |
| [Schema overrides demo](#schema-overrides-demo) | Bundled override file |
| [Migration demo](#migration-demo) | Toy rename walkthrough |
| [Sensitivity](#sensitivity) | Hidden-column exercises |
| [Column security](#column-security) | deny_columns practice |
| [Named AetherSpaces](#named-aetherspaces) | Define spaces offline |
| [Views and partition pruning](#views-and-partition-pruning) | DuckDB-only demos |
| [Federation recipe](#federation-recipe) | Offline multi-member demo |
| [What you cannot do in the sandbox](#what-you-cannot-do-in-the-sandbox) | Honest limits |

---

## About the environment

`create_offline_sandbox` / `AetherEngine.offline_sandbox()` automatically:

1. Unpacks the shipped `data.zip` corpus.
2. Loads seed SQL into an in-memory DuckDB connection.
3. Seeds baseline schema artifacts when a bundled baseline is present.
4. Writes a temporary TOML that selects DuckDB `:memory:` and the **mock** LLM with the bundled fixtures file.
5. Pins mock fixture keys and schema literals from the bundle.
6. Returns a `SandboxHandle` whose `engine` is a ready owner `AetherEngine` on the default dataset.

No warehouse DSN, no API key, and no operator TOML are required for that path.

The bundled **rental_shop** dataset is a 34-table multi-category rental schema (films, books, games, logistics, promotions). Activity dates in the generated corpus are synthetic (predominantly **2022-2025**). Staff SSN and password columns are synthetic sample credentials used for sensitivity exercises. Every table, column, federation member, and practice question is listed in [Sandbox data reference](SANDBOX_DATA_REFERENCE.md).

## Enter and exit the sandbox

```bash
pip install aetherdialect
```

```python
from aetherdialect import AetherEngine

with AetherEngine.offline_sandbox() as sb:
    with sb.engine.session(mode="writer") as session:
        step = session.accept_until_done("How many films are there?")
    print(step.sql)
```

`sb.session()` is a convenience alias for `sb.engine.session(mode="writer")` on the default owner engine. The public entry points are **engine role** (`owner` vs `consumer` on the constructed engine) and **session mode** (`writer` vs `reader` on `session()`).

### Handle lifecycle

Always use `with AetherEngine.offline_sandbox() as sb:` or call **`sb.close()`** when finished. Closing ends open session state, disposes the in-memory DuckDB connection(s), deletes the temporary extract directory, and removes the temporary artifacts directory and working-directory sidecars registered through the handle (for example `schema_overrides.json` from `apply_bundled_schema_overrides()`).

The next entry rebuilds from the bundled corpus. Learning does **not** persist between separate `offline_sandbox()` calls - each handle is self-contained inside the package.

### Production-shaped authoring

- **`AetherEngine.offline_sandbox()`** - returns a working engine (or federation) so you can practice immediately.
- **`Sandbox()`** - returns a seeded environment so you build engines the same way you would in production.

`AetherEngine.offline_sandbox()` returns a ready-made engine for quick practice. **`Sandbox()`** is the authoring entry point when you want the same constructor shape as production: you supply `EngineContext`, pass `sandbox.connection()` as `native_connection`, and call **`sandbox.adopt(engine)`** before opening a session. Adoption is required - it applies sandbox mode (mock fixtures, warmup suppression, and pinned schema literals). An engine built on a sandbox connection without adoption raises on `session()`.

```python
from aetherdialect import AetherEngine, EngineContext, Sandbox

with Sandbox() as sandbox:
    engine = AetherEngine(
        EngineContext(allow_objects=frozenset({"customer", "rental"})),
        native_connection=sandbox.connection(),
        artifacts_dir=sandbox.artifacts_dir,
        config_file=sandbox.config_file,
    )
    sandbox.adopt(engine)
    with engine.session(mode="writer") as session:
        step = session.accept_until_done("How many rentals are there?")
```

Federation authoring uses the same environment: bundled member datasets load automatically when present. Call `sandbox.federation(...)` with the bundled declaration or your own:

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

### Scoped engines on both entry points

`AetherEngine.offline_sandbox(engine_context=...)` and `Sandbox().engine(engine_context=...)` honour the same narrowed scope. A scope smaller than the full owner catalog derives the graph as a subset of the bundled seed; both paths apply identical baseline-trust rules for the same `EngineContext`.

```python
from aetherdialect import AetherEngine, EngineContext, Sandbox

scope = EngineContext(allow_objects=frozenset({"customer", "rental"}))

with AetherEngine.offline_sandbox(engine_context=scope) as sb:
    assert "film" not in sb.engine._schema_graph.tables

with Sandbox() as sandbox:
    engine = sandbox.engine(scope)
    assert engine._runtime_config.engine_context.allow_objects == frozenset({"customer", "rental"})
```

### Multiple engines from one sandbox

Declare a different owner or consumer context on each `sandbox.engine(...)` call inside one `Sandbox` handle. The bundle is extracted once; each build re-derives its graph from the shared seed without re-profiling the full catalog.

```python
from aetherdialect import EngineContext, Sandbox

with Sandbox() as sandbox:
    customer_engine = sandbox.engine(EngineContext(allow_objects=frozenset({"customer"})))
    rental_engine = sandbox.engine(EngineContext(allow_objects=frozenset({"rental"})))
    assert "customer" in customer_engine._schema_graph.tables
    assert "rental" in rental_engine._schema_graph.tables
    assert "rental" not in customer_engine._schema_graph.tables
```

Use `with Sandbox() as sandbox:` (or `with AetherEngine.offline_sandbox() as sb:`) so DuckDB connections, the extracted bundle directory, and temp artifacts are disposed on exit.

### Fixture alias resolution

When `notes_file` or `sql_file` names a path that does not exist on disk, the sandbox resolves common aliases to bundled fixtures and records a notice on `engine.init_notices`. An existing file on disk is used as written.

| User path | Bundled fixture |
| --- | --- |
| `notes.txt`, `catalog_notes.txt` | `rental_shop_notes.txt` |
| `schema.sql` | `rental_shop.sql` |

```python
from aetherdialect import EngineContext, Sandbox

with Sandbox() as sandbox:
    engine = sandbox.engine(EngineContext(notes_file="notes.txt", sql_file="schema.sql"))
    for notice in engine.init_notices:
        print(notice)  # 'notes.txt' resolved to bundled fixture 'rental_shop_notes.txt'
```

The same resolution applies to `notes_file=` / `sql_file=` on `AetherEngine.offline_sandbox(...)`.

### Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `include` | `"tables"` | `"views"` applies bundled view definitions and reflects views only (`"both"` is rejected) |
| `deny_columns` | none | Strip qualified columns from the graph before profiling ([Column security](#column-security)) |
| `engine_context` | full owner scope | Narrow tables, notes, and DDL paths the production way |
| `notes_file` / `sql_file` | from bundle | Override paths on `offline_sandbox(...)`; aliases resolve as above |
| `llm_config` | bundled mock TOML | Non-mock provider selects live LLM mode (see below) |
| `maintainer_access` | `False` | Required for closed-world escape hooks (see below) |

`artifacts_dir` exists for integrator-level tests that mimic production shared storage; **do not pass it for normal sandbox practice** - the corpus and temp artifacts are managed inside the package.

### Maintainer hooks (closed-world escapes)

The sandbox is closed-world: users name bundled datasets, fixtures, and tables only. Maintainer hooks reach outside that boundary and require `maintainer_access=True` on `Sandbox(...)` or `create_offline_sandbox` / `AetherEngine.offline_sandbox(...)`:

| Hook | Effect |
| --- | --- |
| `bundle_dir=` | Use an unpacked bundle directory instead of the shipped `data.zip` |
| `seed_sql=` / `connection=` on `offline_sandbox(...)` | Seed or attach a custom DuckDB database |
| `load_dataset(..., seed_sql=...)` / `sql_file=` | Load arbitrary on-disk seed SQL |
| `AETHERDIALECT_SANDBOX_DATA_ZIP` | Point at a custom zip before construction |
| `federation(..., members={...})` with filesystem paths | Supply member seeds by path instead of bundled dataset names |

Without `maintainer_access=True`, these raise `ConfigError` naming the hook.

### LLM provider mode

Pass **`llm_config=`** on `Sandbox(...)` or `offline_sandbox(...)` to use your own TOML instead of the bundled mock provider. Inspect `sandbox.llm_mode` (`"mock"` or `"network"`) and `sandbox.uses_network` at construction. Mock mode replays bundled fixtures; network mode calls the configured provider and may make live API requests. When mock mode is active, unrecorded questions raise `MockFixtureMissingError` (see `sandbox_questions()`).

Health check: `AetherEngine.sandbox_doctor()` before shipping; `AetherEngine.assert_sandbox_complete()` in CI.

## Ask questions

Use the same session patterns as production ([Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps)).

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

Curated practice strings ship inside the installed package. Load them through `AetherEngine.sandbox_questions()`; do not hard-code long question lists in your scripts. Paraphrase/reuse demos, validation-failure examples, and feedback rejection text live in the bundled `sandbox_catalog.json` (`sandbox_paraphrase_pairs()`, `sandbox_validation_failure_demo()`, `sandbox_feedback_demo()`).

```python
questions = AetherEngine.sandbox_questions()
pairs = AetherEngine.sandbox_paraphrase_pairs()
failures = AetherEngine.sandbox_validation_failure_demo()
feedback = AetherEngine.sandbox_feedback_demo()
```

| API | Use |
| --- | --- |
| `sandbox_questions()` | All bundled practice questions |
| `sandbox_paraphrase_pairs()` | Canonical->paraphrase wordings recorded during corpus build |
| `sandbox_validation_failure_demo()` | Questions expected to end in terminal validation errors |
| `sandbox_feedback_demo()` | Anchor question + allowed rejection text for the feedback recipe |

```python
with AetherEngine.offline_sandbox() as sb:
    with sb.session() as session:
        for question in AetherEngine.sandbox_questions():
            step = session.accept_until_done(question)
            session.reset()
```

## Owner vs consumer roles

Offline practice uses the same **owner/consumer** split as production. Choose the role on the engine (`Sandbox().engine(role=...)` or `offline_sandbox(engine_context=...)` with a consumer scope) and pass **`mode="writer"`** or **`mode="reader"`** on `session()`.

| Role | Typical session mode | What loads | Learning |
| --- | --- | --- | --- |
| owner (default) | `writer` | Full **rental_shop** seed (~34 tables) and owner schema baseline from `data.zip` | Templates, overrides, and feedback persist locally for that handle |
| consumer | `reader` | Same seed, narrowed by your `EngineContext.allow_objects` (must be a subset of the owner scope) | Learning events enqueue to `write_queue.jsonl`; `apply_schema_overrides` enqueues proposals instead of applying |

```python
# Owner - full catalog, writer learning
with AetherEngine.offline_sandbox() as sb:
    with sb.engine.session(mode="writer") as session:
        session.accept_until_done("How many films are there?")

# Consumer - supply a narrowed EngineContext, then open a reader session
from aetherdialect import EngineContext, Sandbox

consumer_scope = EngineContext(allow_objects=frozenset({"customer", "film", "payment", "rental"}))
with Sandbox() as sandbox:
    engine = sandbox.engine(consumer_scope, role="consumer")
    with engine.session(mode="reader") as session:
        session.accept_until_done("How many films are there?")
```

For a **permission-denied exercise**, pass a six-table `allow_objects` set (`customer`, `payment`, `rental`, `address`, `city`, `country`) so questions about tables such as `staff` fail at runtime:

```python
from aetherdialect import EngineContext, Sandbox

restricted = EngineContext(
    allow_objects=frozenset({"customer", "payment", "rental", "address", "city", "country"}),
)
with Sandbox() as sandbox:
    engine = sandbox.engine(restricted, role="consumer")
    with engine.session(mode="reader") as session:
        step = session.accept_until_done("Show me all staff salaries.")
    assert step.status == "permission_denied" or step.error
```

Inspect a narrowed graph before your first question:

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    engine = sandbox.engine(role="consumer")
    stats = engine.get_schema_stats()
    print(stats)  # table/column counts for the narrowed graph
```

Consumer questions that reference tables outside the allow list (for example `staff`) should fail with permission or schema errors - see `sandbox_validation_failure_demo()`.

## Federation declaration format

Author a single `federation_declaration.json` per federation. Pass its path to `AetherFederation(..., declaration_file=...)`. The document merges manifest fields (joins, aliases, coordinator caps) with optional logical mapping sections (`logical_tables`, `logical_columns`). Member roster fields (`sources`, `table_namespace`) are **derived** from registered engines at compose time - do not author them in the declaration.

| Field | Meaning |
| --- | --- |
| `version` | Declaration format version (currently `1`) |
| `federation_id` | Stable federation name; must match the `AetherFederation` constructor `name` |
| `cross_source_joins[]` | Cross-member joins: `left` / `right` (`table.column`), `kind`, `logical_key` |
| `aliases` | Optional map of alias -> `{source, table}` |
| `coordinator` | Coordinator caps (`row_cap`, `default_source_row_cap`, `default_source_timeout_ms`, `semijoin_key_cap`, `spill_row_threshold`, ...) |
| `logical_columns[]` | Optional unified column keys across members |
| `logical_tables[]` | Optional union/replica logical tables across members |

Annotated sandbox example (bundled as `federation_declaration.json`):

```json
{
  "version": 1,
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

`export_federation_declaration()` writes this authored shape to the working directory. `export_federation_manifest()` and `export_federation_mappings()` write **persisted** sidecars (including derived roster fields) for review under the federation tree. Field reference: [API reference - Federation and migration JSON](API_REFERENCE.md#federation-and-migration-json).

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    federation = sandbox.federation("sandbox_rental_shop")
    with federation.session(mode="writer") as session:
        step = session.accept_until_done("How many rentals are linked to film titles?")
    print(step.federated_bundle.display_sql)
```

## Reader/writer queue

Advanced: production reader/writer sharing uses a durable `artifacts_dir` on disk. The sandbox **does not require this** for normal practice. The pattern below is documented for integrators mirroring production; it is not the default offline path.

Readers append events to `write_queue.jsonl`; an owner writer on the same shared artifacts root drains the queue. Resolve the path with `engine.write_queue_path`.

```python
import tempfile
from aetherdialect import AetherEngine, Sandbox

shared = tempfile.mkdtemp(prefix="sandbox_rw_")
q_text = AetherEngine.sandbox_questions()[0]

with Sandbox(artifacts_dir=shared, cleanup=False) as reader_sandbox:
    reader = reader_sandbox.engine(role="consumer")
    with reader.session(mode="reader") as session:
        session.accept_until_done(q_text)

with Sandbox(artifacts_dir=shared, cleanup=False) as writer_sandbox:
    writer = writer_sandbox.engine(role="owner")
    with writer.session(mode="writer") as session:
        session.accept_until_done(q_text)
```

Both sides must use the same `artifacts_dir` string.

## Rejections and feedback

Reject with `step("n")`, then supply free-text feedback when `reply_shape == "free_text"`. The bundled corpus includes **one** feedback demo slot: the anchor question accepts only the exact rejection string from `AetherEngine.sandbox_feedback_demo()` - arbitrary feedback text fails at runtime by design.

## Template reuse

After you accept a question, ask again with different wording in the same session. Reuse is automatic - there is no separate API.

1. **Direct reuse** - wording within the fuzzy token-edit budget (default <=2) can replay the stored SQL path without a full interpret/ground/compose cycle. `sandbox_paraphrase_pairs()` includes year variants recorded during corpus build to demonstrate parameter substitution on an accepted template - for example accepting a question about rentals in **2025**, then asking the same question with **2026** replays the template and swaps only the year parameter.
2. **Full parse** - larger wording changes run through the mock LLM and the normal suspend loop.

Ask normally; compare diagnostics (`REUSE_HIT` vs `REUSE_MISS`) between small and large wording changes ([User guide - Asking a question](USER_GUIDE.md#asking-a-question)).

## Schema overrides demo

```python
with AetherEngine.offline_sandbox() as sb:
    sb.apply_bundled_schema_overrides()
```

Copies bundled `sandbox_overrides_demo.json` to the working directory and applies overrides (owners) or enqueues proposals (consumers).

**What the demo changes** (read the file after apply, or inspect the bundled JSON in the corpus):

| Target | Effect |
| --- | --- |
| `staff.ssn`, `staff.password` | sensitivity -> **hidden** (column stays in the graph; prompts and validation treat it as off-limits) |
| `film` | analyst description on the table |

**How to see allowed vs denied behavior without a discovery API:**

1. `AetherEngine.sandbox_validation_failure_demo()` - questions that should fail after overrides (for example targeting `staff.ssn`).
2. `export_schema_overrides()` on a production owner engine writes the current override editor JSON; in the sandbox, read `./schema_overrides.json` after `apply_bundled_schema_overrides()`.
3. Compare owner vs consumer roles - consumer cannot call owner-only override APIs.

Overrides are manual - there is no runtime "list overrides" method beyond export/read of the JSON file. Full schema: [API reference - Schema overrides JSON](API_REFERENCE.md#schema-overrides-json-schema_overridesjson). Sensitivity tiers: [Security - Sensitivity tags](SECURITY.md#6-sensitivity-tags).

## Migration demo

The shipped corpus includes a toy `item.title` -> `item_title` rename under `migration_demo/` (pre-migration artifacts plus a remap map). The internal sandbox tour loads those assets, constructs an engine against the post-rename seed, expects `MigrationPendingError`, then applies the map with `AetherEngine.apply_migration_map(...)`. Production workflow: [User guide - Migration](USER_GUIDE.md#migration).

## Sensitivity

Canonical tier definitions: [Security - Sensitivity tags](SECURITY.md#6-sensitivity-tags). After `apply_bundled_schema_overrides()`, try questions from `AetherEngine.sandbox_validation_failure_demo()` to see terminal errors when targeting hidden columns such as `staff.ssn`.

## Column security

`deny_columns` is a **constructor parameter on `offline_sandbox()`**, separate from **hidden** sensitivity in overrides.

| Mechanism | Where set | Effect |
| --- | --- | --- |
| `deny_columns` on `offline_sandbox(...)` | `EngineContext.deny_columns` at sandbox entry | Column is **removed from the schema graph** before profiling - the engine does not know the column exists |
| `sensitivity: hidden` in overrides | `apply_bundled_schema_overrides()` demo | Column remains in the graph but is blocked from prompts and validation |

Use `deny_columns` when you want column-security exercises where the column should not appear in schema metadata at all:

```python
with AetherEngine.offline_sandbox(deny_columns=frozenset({"staff.ssn"})) as sb:
    with sb.session() as session:
        step = session.ask("Show payroll deductions by employee SSN.")
```

Production equivalent: set `deny_columns` on `EngineContext` at construction ([User guide - EngineContext](USER_GUIDE.md#enginecontext)).

## Named AetherSpaces

Offline sessions default to `space="master"`. Define named spaces on the owner engine the same way as production:

```python
from aetherdialect import AetherEngine, SpaceContext

with AetherEngine.offline_sandbox() as sb:
    sb.engine.aetherspace(
        "catalog",
        space_context=SpaceContext(
            tables=frozenset({"item", "film", "category", "item_category"}),
            notes_file="sandbox_space_catalog_notes.txt",  # bundled second-space demo notes
        ),
    )
    with sb.session(space="catalog") as session:
        session.accept_until_done("How many films are in the Horror category?")
    with sb.session(space="master") as session:
        session.accept_until_done("How many films are in the Horror category?")
```

Compare answers to see how knowledge narrowing changes results. Template learning is **per-space**. Conceptual guide: [User guide - AetherSpace](USER_GUIDE.md#aetherspace).

## Views and partition pruning

The bundled DuckDB seed includes analytical views (`active_customer_v`, `store_revenue_v`, `film_catalog_v`) and partition metadata on `rental.rental_date`. Ask questions that filter on rental dates or query revenue views to exercise view scope and partition predicate injection.

For production warehouses, set `EngineContext(include="views")` with your own view DDL - [User guide - EngineContext](USER_GUIDE.md#enginecontext). `include` is mutually exclusive per kind: `"tables"` reflects base tables only (default) and `"views"` reflects views only. The offline sandbox accepts the same selector via `AetherEngine.offline_sandbox(include="views")` when the views corpus variant is bundled.

## Federation recipe

When the shipped bundle includes all four partition seeds (`storefront`, `catalog`, `logistics`, and `crm`), call `Sandbox().federation("sandbox_rental_shop")`. That loads four DuckDB members on **separate in-memory connections** with per-member artifact trees, then constructs `AetherFederation` from the bundled `federation_declaration.json`. The declaration wires:

- A row-partitioned `payment` **union** across `storefront` and `catalog`.
- `customer` and `staff` **replicas** with `storefront` authoritative and a column subset on `crm` (`staff_id`, `first_name`, `last_name`, `store_id` only - credentials never enter the CRM seed).
- Four declared cross-source joins linking rentals, inventory, purchase lines, deliveries, and promotion redemptions across members.

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    federation = sandbox.federation("sandbox_rental_shop")
    with federation.session(mode="writer") as session:
        step = session.accept_until_done("How many rentals are linked to film titles?")
    print(step.federated_bundle.display_sql)
```

Federated turns return a structured `FederatedSqlBundle` on `SessionStep.federated_bundle` (per-source statements plus coordinator glue); `SessionStep.sql` remains display-only.

Typical scenarios covered by the bundled federation declaration:

1. A cross-source question answered with the same `SessionStep` shape as single-source.
2. Operator-confirmed mapping on the boundary join key.
3. A `union` logical table without double counting.
4. A `replica` query that resolves to the authoritative member.
5. Configuration rejection when a cross-source key is not `sensitivity == none`.
6. Scope-style rejection for unsupported cross-source aggregates.
7. A federation AetherSpace spanning all four catalogs.

If any partition seed is absent from the active bundle, `sandbox.federation(...)` raises `ConfigError` naming the missing files rather than inventing members.

**Partition seed status:** The four bundled member seeds (`federation_storefront_seed.sql`, `federation_catalog_seed.sql`, `federation_logistics_seed.sql`, `federation_crm_seed.sql`) in `scripts/data/` are **payment-only NON-CANONICAL placeholders**. They parse in DuckDB and satisfy `SANDBOX_BUNDLED_MEMBER_SEEDS`. The operator corpus build runs `export_federation_partition_seeds` to replace them from `rental_shop.sqlite` before packing `data.zip`.

### Annotated declaration excerpt

The bundled declaration (abridged) shows how union, replica, and join declarations fit together:

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

Authoring with your own declaration, member seeds, and `FederationContext` follows the same shape - see [Production-shaped authoring](#enter-and-exit-the-sandbox).

## What you cannot do in the sandbox

- **Seed warmup** and **QSim** raise `ConfigError` on sandbox instances.
- Questions outside the recorded mock corpus raise `MockFixtureMissingError` (or fail the turn) - coverage is finite by design.
- Arbitrary migration maps or override files beyond the bundled demos are outside the practice path.

### Production stages not exercised offline

The sandbox replays bundled baselines instead of running several production construction paths. The following stages are catalogued in `SANDBOX_UNEXERCISED_PRODUCTION_STAGES` and are intentionally not exercised offline:

| Stage | What production does | Sandbox behaviour |
| --- | --- | --- |
| `live_reflection_and_profiling` | Live catalog reflection and column profiling against your warehouse | Replays a frozen `schema_graph.json.gz` baseline |
| `probe_mismatch_partial_rebuild` | DDL probe mismatch triggers structural diff and partial rebuild | Baseline fingerprints match the bundle; no probe mismatch path |
| `cold_build_descriptions_and_classification` | LLM description generation and column classification on first build | Descriptions and roles come from the recorded baseline |
| `composite_composition_replay_skip` | Full federation composite composition when member baselines differ | Composite graph is replayed when seeded baselines satisfy replay |
| `warmup_and_question_simulation` | Seed warmup and QSim enumeration against live schema | Blocked via production-API guard |
| `model_turns_outside_recorded_fixtures` | Arbitrary LLM turns for unseen questions | Mock provider replays recorded fixtures only |

Bundled malformed mock fixtures exercise compose repair paths for questions listed in `SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS`.

When you move to production, follow [Getting started - Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse) for TOML, LLM credentials, and durable artifacts.

---

**See also:** [Sandbox data reference](SANDBOX_DATA_REFERENCE.md) | [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [API reference](API_REFERENCE.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
