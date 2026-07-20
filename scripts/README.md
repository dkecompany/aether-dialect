# Rental shop data pipeline

Self-contained scripts to build the **rental_shop** CSV bundle (34-table multi-category schema: films, books, games, logistics, promotions), load it into dev engines, and publish the offline sandbox corpus. Configure credentials in root `env.env` (copy from `env.example.env`).

## Sections

| Section | Contents |
| --- | --- |
| [Quick start](#quick-start) | CSV bundle and engine load |
| [source_rental_shop.py](#source_rental_shoppypy) | Bundle generation |
| [load_rental_shop_engines.py](#load_rental_shop_enginespy) | Multi-engine load |
| [Data sources](#data-sources) | Lexicons and external feeds |
| [Layout](#layout) | Paths and roles |
| [Sandbox data model](#sandbox-data-model-four-files) | Four JSON/notes files |
| [Sandbox corpus](#sandbox-corpus) | Recording and packaging |
| [Maintainer publish](#maintainer-publish) | Azure bundle upload |

## Quick start

```text
# 1 — Ensure CSV bundle (default: download cascade)
.venv\Scripts\python.exe scripts\source_rental_shop.py

# 2 — Load all engines (add --drop-first to replace existing data)
.venv\Scripts\python.exe scripts\load_rental_shop_engines.py --all --drop-first
```

For a from-scratch maintainer run with LLM enrichment, see [Maintainer publish](#maintainer-publish) below.

## `source_rental_shop.py`

Ensures `scripts/data/rental_shop_csvs/` exists. One mode flag per invocation (except `--pack`, which combines with `--generate`).

| Invocation | Behaviour |
| --- | --- |
| *(no flags)* / `--download` | **Download cascade:** use existing `rental_shop_csvs/` if present; else extract `scripts/data/rental_shop.zip`; else fetch `DEFAULT_BUNDLE_URL` (Azure blob constant in the script). |
| `--generate` | Synthesize CSVs from frozen `inputs.zip` (required manifest below) plus Open Library books and Zenodo games. Fail-fast if any lexicon member is missing or too short. Runs FK and semantic verification. |
| `--generate --enrich-llm` | Same as `--generate`, then LLM-enrich **`item.description` only** (requires `OPENAI_API_KEY`; disk cache under `scripts/data/_llm_cache/`). |
| `--pack` | Zip `rental_shop_csvs/` → `scripts/data/rental_shop.zip` (alone, or after `--generate`). |

**Invalid combinations** (script exits with an error):

- `--enrich-llm` without `--generate`
- `--enrich-llm` with `--download`

Output directory: `scripts/data/rental_shop_csvs/` (gitignored). Optional env: `RENTAL_SHOP_BUNDLE_SHA256` (verify downloaded zip), `RENTAL_SHOP_LLM_MODEL`, `RENTAL_SHOP_LLM_BATCH_SIZE`.

## `load_rental_shop_engines.py`

Loads CSVs + canonical DDL into supported engines from `scripts/data/rental_shop.sql`.

| Flag | Behaviour |
| --- | --- |
| `--all` | Start from all supported engines; subtract any `--exclude-*` flags. |
| `--postgres`, `--mysql`, … | Include only listed engines (cannot mix with excludes unless `--all` is set). |
| `--exclude-*` | Skip engines when used with `--all`. |
| `--drop-first` | Drop/recreate target schema or tables before load. |
| `--csv-dir` | CSV directory (default `scripts/data/rental_shop_csvs/`). |
| `--ddl` | DDL path (default `scripts/data/rental_shop.sql`). |

With **no arguments**, all ten engines load. Include and exclude flags cannot be combined unless `--all` is present.

**BigQuery:** set `BIGQUERY_LOCATION` to the dataset data region (e.g. `australia-southeast1`), not the project default.

## Data sources

| Content | Source |
| --- | --- |
| Film titles, actors, customers, staff, categories, countries, languages, cities, addresses, promotions, couriers, warehouses, suppliers | Frozen `scripts/data/inputs.zip` (generator exits if any required member is missing) |
| Books | Open Library search seeds (download at generate time) |
| Games | Zenodo PlayMyData slice (download at generate time) |

**`inputs.zip` required members:** `promotion_names.jsonl`, `courier_names.jsonl`, `warehouse_names.jsonl`, `suppliers.csv`, `publishers_expanded.csv`, `addresses.jsonl`, `cities_sample.jsonl`, `country.jsonl`, `language.jsonl`, `category.jsonl`, `staff_names.jsonl`, `customer.jsonl`, `actor.jsonl`, `film.jsonl` (or `films.jsonl`). No runtime template fallbacks except pre-enrich `item.description` placeholders.

Activity anchor for generated dates: `2026-07-01` (`RENTAL_SHOP_AS_OF` env override for tests).

## Layout

| Path | Role |
| --- | --- |
| `scripts/data/rental_shop_csvs/` | Generated or downloaded CSV exports (gitignored) |
| `scripts/data/rental_shop.zip` | Packed CSV bundle for publish/download (gitignored) |
| `scripts/data/rental_shop.sql` | Canonical DDL (hand-maintained) |
| `scripts/data/rental_shop_notes.txt` | Domain notes for schema context |
| `scripts/data/rental_shop_overrides.json` | Shipped sensitivity overrides (`version` **1**) |
| `scripts/data/sandbox_questions.txt` | Offline sandbox question corpus (`# questions`, `# validation_failures`, `# feedback_samples`) |
| `scripts/data/inputs.zip` | Frozen lexicons and name lists |
| `scripts/sandbox_staging/` | Sandbox bundle staging (gitignored) |
| `scripts/<engine>/` | Per-engine load wrappers |

## Sandbox data model (four files)

All four files under `scripts/data/` serve distinct roles — do not merge them with `sandbox_questions.txt`.

| File | Role | Consumed by |
| --- | --- | --- |
| `sandbox_expectations.json` | Deterministic per-slot terminal outcomes (`terminal_status`, `must_tables`, `sql_contains`, faithfulness) for recording validation and `AetherEngine.assert_sandbox_complete()` | `sandbox_corpus.py`, package sandbox gate |
| `sandbox_scenarios.json` | Non-standard recording flows: validation-failure mechanisms, feedback-sample anchor questions, allowed rejection text | `sandbox_corpus.py` recording router |
| `sandbox_paraphrase_pairs.json` | **Maintainer-only** canonical↔paraphrase mapping for doc linking, shared expectation rules, optional `--record-reuse-pairs` | `sandbox_corpus.py` — **not** a public user API |
| `sandbox_space_catalog_notes.txt` | Second AetherSpace notes file for catalog-space demo (inherited-then-refined descriptions) | Bundled into sandbox zip |

Metadata (`sandbox_expectations.json`, `sandbox_scenarios.json`) is **hand-maintained** under `scripts/data/` and copied into staging unchanged during builds. User-facing discovery ships as `sandbox_catalog.json` inside `data.zip` (built from scenarios + recorded paraphrase pairs).

## Maintainer publish

1. `source_rental_shop.py --generate --enrich-llm`
2. Confirm FK integrity passes (`verify_csv_integrity()` in the script)
3. `source_rental_shop.py --pack`
4. Upload `rental_shop.zip` to Azure; update `DEFAULT_BUNDLE_URL` in `source_rental_shop.py`

## Sandbox corpus

Rebuild the shipped offline bundle after DDL, notes, overrides, or question-corpus changes:

```text
.venv\Scripts\python.exe scripts\sandbox_corpus.py
.venv\Scripts\python.exe scripts\sandbox_corpus.py --repair
```

### Two modes

| Flag | Behaviour |
| --- | --- |
| *(no flags)* | Full build: assemble staging, build baseline, record fixtures, validate, and pack. |
| `--repair` | Re-record failing fixture slots when staging fingerprint matches last build. |
| `--smoke` | End-to-end pipeline with one practice question plus validation/feedback/scenario slots; writes `sandbox_staging.zip` only. |
| `--record-reuse-pairs` | Record paraphrase pairs in the same warm session to capture reuse traces. |

### What full rebuild does

1. Stages seed SQL, notes, overrides, baseline artifacts, and question lists under `scripts/sandbox_staging/`.
2. Records mock LLM fixtures by driving `AetherEngine.offline_sandbox()` sessions against staged seed (in-memory DuckDB — not file-backed dev DuckDB).
3. Validates staging (practice questions, consumer reader paths, feedback flows, schema overrides demo).
4. Packs `src/aetherdialect/sandbox/data.zip` only after validation passes.

### Recording policy

| Control | Corpus build (`sandbox_corpus.py`) | User sandbox (`offline_sandbox()`) |
| --- | --- | --- |
| Template direct reuse | Off — every slot gets a full LLM fixture trace | On — in-session fuzzy/direct reuse |
| Template learning persistence | Fixtures only in zip; no learned templates shipped | Learning in temp `artifacts_dir`; deleted on handle close |
| Runtime paraphrase generation on accept | N/A at record time | Skipped under mock (no live LLM credentials) |

Default recording runs **owner writer slots only** (~50 live LLM traces). Pack-time validation replays **both** owner and consumer reader paths against committed fixtures — consumer expectations stay in `sandbox_expectations.json` but are not re-recorded. Optional `--record-reuse-pairs` records mapped paraphrase pairs in the same warm session to capture reuse-specific traces. Maintainer-only `fixture_copy_replacements` in `sandbox_paraphrase_pairs.json` is a last-resort hack for near-identical fixtures (for example year literal swaps).

**DuckDB config split:** root `env.env` drives live tests and dev tooling with file-backed DuckDB (`DUCKDB_PATH=scripts/duckdb/rental_shop.duckdb`). Corpus build forces `:memory:` DuckDB seeded from staged `rental_shop_seed.sql`. LLM credentials come from `env.env`.

### Reset generated artifacts (preserve source inputs)

From the repo root (PowerShell):

```powershell
Remove-Item -Recurse -Force scripts/sandbox_staging, scripts/sqlite/*.sqlite, .pytest_cache -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Force src/aetherdialect/sandbox/data.zip -ErrorAction SilentlyContinue
Remove-Item -Force live_tests/results.txt, live_tests/evidence_*.txt -ErrorAction SilentlyContinue
$userData = python -c "from platformdirs import user_data_dir; print(user_data_dir('aetherdialect'))"
Remove-Item -Recurse -Force $userData -ErrorAction SilentlyContinue
```

Keeps source inputs (`scripts/data/*`, staging SQL/notes, question lists, manifests).

## Synthetic data notes

- ISBNs are hashed/synthetic; activity dates are synthetic (2022–2025 window).
- Permanent quality gate: **FK integrity only** (`verify_csv_integrity()` in `source_rental_shop.py`).
- `staff.ssn` is synthetic HR data (hidden tier in overrides), not real sensitive data.
