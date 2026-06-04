# Bitcoin Agent Privacy Experiments

This directory contains the experiment workflow used to evaluate whether an xpub-only AI agent can construct Bitcoin PSBTs with better structural privacy.

The code is research tooling, not a production wallet. It never handles private keys, never signs transactions, and never broadcasts transactions. The agent can inspect wallet state from an extended public key, construct unsigned BIP-174 PSBTs, and pass those PSBTs to an offline privacy scorer.

Canonical TFM context lives in `/home/jaume/feina/TFM/ResumTFM.md`; chronological progress and decisions live in `/home/jaume/feina/TFM/PROGRESS.md`. Keep this README focused on how the experiment system runs and how the 2026 prompt-first charts are produced.

## What This Experiment System Does

The experiment pipeline compares LLM providers, models, temperatures, prompt strategies, transaction amounts, repetitions, and follow-up prompts. Each experiment asks the Bitcoin agent to produce a PSBT. The runner saves the generated artifacts and, when the scorer is available, evaluates each PSBT with TxPrivScore.

The core flow is:

```text
experiments.csv
  -> experiment_runner.py
  -> BitcoinAIAgent
  -> generated PSBT
  -> privacy_scorer_v2.py
  -> results/experiments_*.csv + results/experiments_*.json + results/psbts/
```

The Streamlit UI is a local helper on top of the same CSV and runner workflow. It exists to make experiment setup, execution, and comparison faster without changing the underlying pipeline.

## Security Model

- The agent is xpub-only.
- Private keys are never loaded, requested, stored, signed with, or broadcast from this workflow.
- Outputs are unsigned PSBTs intended for later human review and external signing.
- The privacy scorer is offline and evaluates PSBT structure, not live blockchain behavior.
- A high privacy score does not mean a PSBT is safe to sign; fee sanity and human review remain mandatory.

## Directory Structure

```text
experiments/
├── experiments.csv          # Experiment definitions
├── experiment_runner.py     # CLI runner and result writer
├── wallet_baseline.csv      # Real-wallet baseline definitions
├── wallet_baseline_runner.py # Wallet baseline runner, no LLM imports
├── wallet_baseline_results.py # Wallet baseline result normalization and comparison helpers
├── wallet_imports/          # Manual Sparrow/Wasabi PSBT drop folders
├── web_ui.py                # Local Streamlit interface
├── experiment_manager.py    # Backward-compatible CSV read/write helpers
├── prompt_templates.py      # Prompt generation from amount + strategy
├── result_utils.py          # CSV/JSON result loading and normalization
├── phase12_results.py       # Shared 2026 prompt-corpus normalization and cost estimates
├── model_costs_phase12.csv  # Versioned local model-price table for chart estimates
├── paper_charts.py          # Optional chart helpers for result comparison and 2026 prompt UI charts
└── README.md
```

Generated files are intentionally not part of the repository:

```text
experiments/results/
experiments/results/psbts/
experiments/*.psbt
experiments/*.base64
experiments/__pycache__/
```

These paths are ignored by `.gitignore` so local results and PSBT artifacts are not uploaded accidentally.

## Requirements

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure API keys and Bitcoin settings in `.env` at the project root. Do not commit `.env`.

Common variables:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
BITCOIN_XPUB=...
BITCOIN_NETWORK=mainnet
```

The runner loads `.env` from the project root before importing the agent.

## Running Experiments from the CLI

Run commands from `bitcoin-agent-project/experiments`:

```bash
cd experiments

# Validate parsing without running LLM calls
python experiment_runner.py experiments.csv --dry-run

# Run all enabled experiments
python experiment_runner.py experiments.csv

# Run a single experiment
python experiment_runner.py experiments.csv --filter id:exp_openai_gpt54_basic_pct10_t03

# Run several selected experiments into one result file
python experiment_runner.py experiments.csv --filter ids:exp_openai_gpt54_basic_pct10_t03,exp_openrouter_gemini31pro_basic_pct10_t03

# Run an explicitly selected disabled row without changing the CSV enabled flag
python experiment_runner.py experiments.csv --filter id:exp_anthropic_opus47_basic_pct10_t03 --include-disabled

# Filter by provider, model, tag, or name
python experiment_runner.py experiments.csv --filter provider:openai
python experiment_runner.py experiments.csv --filter model:gpt-5.4
python experiment_runner.py experiments.csv --filter tag:prompt-privacy-simple
python experiment_runner.py experiments.csv --filter name:basic

# Reduce rate-limit pressure
python experiment_runner.py experiments.csv --interleave --delay 3

# Run provider lanes concurrently
python experiment_runner.py experiments.csv \
  --parallel-profile provider \
  --max-concurrency 3 \
  --provider-limits openai=1,anthropic=1,openrouter=1

# Aggressive current-batch profile: one lane per model, up to 3 OpenRouter lanes
python experiment_runner.py experiments.csv \
  --parallel-profile model \
  --max-concurrency 5 \
  --provider-limits openai=1,anthropic=1,openrouter=3 \
  --model-limit 1
```

Use `--verbose` when debugging:

```bash
python experiment_runner.py experiments.csv --filter id:exp_openai_gpt54_basic_pct10_t03 --verbose
```

The default remains sequential. `--parallel-profile provider` and `--parallel-profile model` are opt-in local batch modes; they do not change the agent security model, and the agent still never signs or broadcasts. Keep limits conservative if you see 429/503/529 responses, especially on OpenRouter where upstream providers can throttle independently.

## Wallet Baseline Experiments

`wallet_baseline_runner.py` adds a separate real-wallet baseline lane for comparing wallet coin selection against the agent corpus. It does not import `BitcoinAIAgent`, does not read `experiments.csv`, and does not modify the LLM runner contract.

The baseline flow is:

```text
wallet_baseline.csv
  -> wallet_baseline_runner.py
  -> Electrum/Core unsigned PSBT or imported external PSBT
  -> privacy_scorer_v2.py
  -> results/wallet_baseline_YYYYMMDD_HHMMSS/
```

The default `wallet_baseline.csv` contains enabled Electrum and Bitcoin Core rows for the five controlled amount pressures used by the 2026 agent corpus: 10%, 30%, 50%, 80%, and 95% of the 1 BTC controlled wallet balance. The nominal output targets are therefore 10,000,000; 30,000,000; 50,000,000; 80,000,000; and 95,000,000 sats.

It also contains disabled `adapter=import` rows for Sparrow and Wasabi. Enable or run those rows only after placing unsigned BIP-174 exports in:

```text
wallet_imports/sparrow/sparrow_pct10.psbt
wallet_imports/sparrow/sparrow_pct30.psbt
wallet_imports/sparrow/sparrow_pct50.psbt
wallet_imports/sparrow/sparrow_pct80.psbt
wallet_imports/sparrow/sparrow_pct95.psbt
wallet_imports/wasabi/wasabi_pct10.psbt
wallet_imports/wasabi/wasabi_pct30.psbt
wallet_imports/wasabi/wasabi_pct50.psbt
wallet_imports/wasabi/wasabi_pct80.psbt
wallet_imports/wasabi/wasabi_pct95.psbt
```

Supported adapters:

| Adapter | Purpose |
| --- | --- |
| `electrum` | Calls Electrum `payto --unsigned` with a watch-only wallet and native wallet fee selection. |
| `bitcoin_core` | Calls Bitcoin Core `walletcreatefundedpsbt` from a descriptor watch-only wallet with native wallet fee selection. |
| `import` | Reads a PSBT exported manually from Sparrow, Wasabi, Core, or another wallet and scores it with the same offline scorer. |

Local verified wallet binaries used for manual export, when available:

```text
/home/jaume/feina/tools/wallet-baselines/sparrow-2.4.2/bin/Sparrow
HOME=/home/jaume/feina/tools/wallet-baselines/wasabi-home \
  /home/jaume/feina/tools/wallet-baselines/wasabi-2.7.2/wassabee
/home/jaume/feina/tools/wallet-baselines/wasabi-2.7.2/wassabeed
```

The Wasabi `HOME=...` prefix keeps its data directory inside the writable
research workspace instead of `~/.walletwasabi`. The local daemon help does not
expose a `--datadir` flag, so the runner uses a controlled HOME-style datadir
under `experiments/wallets/wasabi_baseline/`.

Safety rules enforced by the runner:

- Electrum rows create unsigned PSBTs only; the command path uses `payto --unsigned`.
- Bitcoin Core rows create unsigned PSBTs only; the command path uses `walletcreatefundedpsbt`.
- The runner never calls signing or broadcast commands.
- Electrum automation requires an explicit watch-only wallet path through `wallet_path`, `--electrum-wallet`, or `WALLET_BASELINE_ELECTRUM_WALLET`.
- Bitcoin Core automation creates/loads a descriptor wallet with `disable_private_keys=true` and refuses wallets where `private_keys_enabled` is not false.
- Wasabi safe automation creates a watch-only wallet JSON only (`EncryptedSecret=null`, `ExtPubKey=...`) and may load/check it through JSON-RPC.
- Wasabi RPC PSBT generation is marked unsupported by safe RPC: the runner never calls `build`, never sends wallet passwords, and never accepts transaction hex as a PSBT.
- Sparrow is classified as manual-import-only unless its local help exposes a real headless transaction-builder command.
- The runner performs a conservative local wallet-file check and refuses files that appear to contain seed/private-key material.
- Imported PSBTs must be BIP-174 payloads and are rejected if they contain partial signatures or finalized input scripts.
- If Electrum or Core cannot provide native fee selection, the row fails explicitly; no fixed-fee fallback is injected.

Dry-run and dependency checks:

```bash
cd experiments

# Print the default Electrum/Core rows and exact nominal targets
python wallet_baseline_runner.py wallet_baseline.csv --dry-run

# Check that the configured Electrum command can expose payto --unsigned
python wallet_baseline_runner.py wallet_baseline.csv --preflight

# Classify the local Sparrow CLI. Current verified Sparrow is manual-import-only.
python wallet_baseline_runner.py wallet_baseline.csv --sparrow-preflight

# Create/update the safe Wasabi watch-only wallet JSON and status marker.
python wallet_baseline_runner.py wallet_baseline.csv --wasabi-prepare

# Generate a public-only Wasabi/Coldcard-style skeleton for the GUI PSBT workflow.
python wallet_baseline_runner.py wallet_baseline.csv --wasabi-generate-skeleton

# Check an already-running Wasabi daemon RPC and the controlled UTXO set.
python wallet_baseline_runner.py wallet_baseline.csv --wasabi-prepare --wasabi-rpc-check

# Optionally start wassabeed in the controlled datadir, then run the RPC check.
python wallet_baseline_runner.py wallet_baseline.csv --wasabi-start-daemon --timeout-seconds 60
```

Wasabi status is written to:

```text
experiments/wallets/wasabi_baseline/wasabi_preflight_status.json
```

The public-only skeleton trial writes:

```text
experiments/wallets/wasabi_baseline/coldcard_public_only_skeleton.json
```

This skeleton is generated only from public material. It contains the normalized
public `ExtPubKey`, `AccountKeyPath=84'/0'/0'`, `TaprootAccountKeyPath=86'/0'/0'`,
`EncryptedSecret=null`, `PasswordVerified=true`, empty `HdPubKeys`, and an
explicit placeholder master fingerprint `00000000` when the real fingerprint is
unknown. The Web UI labels this state as **Public-only skeleton generated** until
Wasabi exports real unsigned PSBTs.

To test Wasabi's GUI PSBT workflow with the controlled datadir:

```bash
HOME=/home/jaume/feina/bitcoin-agent-project/experiments/wallets/wasabi_baseline \
  /home/jaume/feina/tools/wallet-baselines/wasabi-2.7.2/wassabee --network=main --usetor=false
```

Import the skeleton through Wasabi's wallet import flow. Count Wasabi as a real
wallet baseline only if Wasabi itself enables the PSBT workflow and saves the
five unsigned PSBT files under `wallet_imports/wasabi/`. Do not sign, import a
final transaction, or broadcast from this workflow.

The Web UI reads this marker and shows Wasabi as prepared via RPC only when the
daemon loads the watch-only wallet and exposes the controlled 1 BTC UTXO set.
It also shows the public-only skeleton state separately. Until unsigned PSBT
files exist under `wallet_imports/wasabi/`, the scoring path still remains
`adapter=import`.

Run Electrum baselines with explicit watch-only state:

```bash
python wallet_baseline_runner.py wallet_baseline.csv \
  --electrum-wallet /path/to/electrum/watch-only-wallet \
  --xpub "$BITCOIN_XPUB" \
  --compare-agents
```

Run Bitcoin Core baselines with the RPC settings from `.env`:

```bash
python wallet_baseline_runner.py wallet_baseline.csv \
  --filter adapter:bitcoin_core \
  --compare-agents
```

Run imported PSBT rows:

```bash
python wallet_baseline_runner.py wallet_baseline.csv \
  --filter adapter:import \
  --include-disabled \
  --compare-agents
```

Result files are written to `experiments/results/wallet_baseline_YYYYMMDD_HHMMSS/` with:

- `wallet_baseline_YYYYMMDD_HHMMSS.csv`
- `wallet_baseline_YYYYMMDD_HHMMSS.json`
- `psbts/*.psbt` and `psbts/*.base64`
- optional `wallet_vs_agent_amount_summary.csv` when `--compare-agents` is used

Wallet baseline rows are not merged into `phase-1` or `phase-2`. The optional comparison uses `phase12_results.py` only to summarize the canonical agent corpus by amount percentage.

`wallet_baseline_results.py` loads all `wallet_baseline_*` result folders, keeps failed attempts for reliability, and selects the latest successful PSBT per `(wallet, amount_pct)` for score/structure charts.

## Running the Local Web UI

The UI is a local Streamlit app. It uses the same `experiments.csv` and calls `experiment_runner.py` through a subprocess.

```bash
cd experiments
streamlit run web_ui.py
```

Main UI capabilities:

- inspect existing experiments from `experiments.csv`,
- create, edit, or clone experiments,
- control provider, model, temperature, repetitions, timeout, network, tags, and enabled state,
- generate prompts from amount + strategy or preserve fully custom prompts,
- run selected experiments through the CLI runner,
- choose sequential, provider-lane, or model-lane execution,
- run selected disabled rows explicitly; the UI passes `--include-disabled` only when needed,
- monitor any active `experiment_runner.py` batch from the Results page, including terminal-launched runs,
- inspect result tables, scores, fee sanity, PSBT paths, and comparison charts,
- inspect prompt-first research charts from the primary 2026 corpus.
- inspect real-wallet baseline charts from Electrum/Core/Sparrow/Wasabi, including coverage, reliability, wallet-vs-agent deltas, and fee/structure tradeoffs.

The UI is optional. Any experiment created by the UI is still represented as a CSV row and can be run from the CLI.

## 2026 Prompt Academic Charts

The prompt-first research chart workflow uses the current 2026 result files as the primary corpus:

```text
experiments/results/phase1/experiments_20260422_114240.csv
experiments/results/phase2_all_phase1_models_20260427_rerun/experiments_20260427_093606.csv
```

The `phase1` and `phase2` directory names are retained as execution-lot traceability labels. They are not treated as separate experimental systems in the charts or in the TFM narrative. The main analytical axis is prompt strategy: `basic`, `privacy-simple`, and `multiturn-detailed`.

Legacy February result files are not mixed into these figures. Generate paper-ready summaries and figures from the workspace root:

```bash
python analysis/charts/generate_phase12_academic_charts.py
```

Outputs are written to:

```text
analysis/results/phase12_*.csv
paper/TFM/figures/phase12/*.pdf
paper/TFM/figures/phase12/*.png
```

The Streamlit sidebar also includes **2026 Prompt Charts**, which uses the same `phase12_results.py` normalization layer for interactive Altair charts.

Historical runs did not store token usage, so the 2026 prompt cost charts use transparent local estimates from `model_costs_phase12.csv`. Future runner outputs include optional trailing token and cost columns when provider metadata exposes usage.

## Active 2026 Matrix

The active `experiments.csv` is now the cleaned 2026 batch:

- `100` enabled Phase 1 execution-lot rows:
  5 active models x 2 prompt strategies (`basic`, `privacy-simple`) x 5 amount targets (`10%`, `30%`, `50%`, `80%`, `95%`) x 2 temperatures (`0.3`, `1.0`)
- `40` disabled deferred Phase 1 frontier rows:
  2 high-cost frontier models kept in place for later reactivation, with the same prompts, IDs, and tags preserved
- `40` disabled Phase 2 execution-lot rows:
  4 pre-created finalist lanes x `multiturn-detailed` x the same 5 amount targets x the same 2 temperatures
- total CSV rows remain `180`
- `repetitions=1` for the `basic`/`privacy-simple` rows and `repetitions=2` for the `multiturn-detailed` rows

The historical paper matrix is preserved at:

```text
experiments_paper_legacy.csv
```

Important batch-specific notes:

- Google-family models in the active matrix run through `openrouter`, not the direct `google` provider lane.
- `OPENROUTER_API_KEY` is therefore required for both `google/gemini-3.1-pro-preview` and `google/gemma-4-31b-it`.
- Amount targets are encoded in prompt text and normalized tags such as `amt-pct-50`; the runner contract is unchanged.
- The UI shows these rows as prompt-defined amount targets instead of pretending they are fixed BTC amounts.

## CSV Format

The legacy runner fields remain the stable execution contract:

| Column | Meaning |
| --- | --- |
| `id` | Unique experiment identifier |
| `name` | Human-readable name |
| `provider` | LLM provider: `openai`, `anthropic`, `google`, `openrouter` |
| `model` | Provider model name |
| `temperature` | LLM generation temperature |
| `user_prompt` | Main request sent to the agent |
| `followup_prompts` | Pipe-separated follow-up prompts |
| `repetitions` | Number of repetitions |
| `timeout_seconds` | Timeout per prompt call |
| `network` | Bitcoin network, usually `mainnet` for the research setup |
| `tags` | Pipe-separated tags for filtering and analysis |
| `enabled` | `true` or `false` |

The UI may add optional columns while preserving old rows:

| Column | Meaning |
| --- | --- |
| `description` | Research notes |
| `amount_btc` | Structured transaction amount used for template prompts |
| `strategy` | Prompt strategy |
| `prompt_mode` | `template` or `custom` |
| `system_prompt` | Optional system prompt override |
| `priority` | Optional ordering field |
| `xpub` | Optional xpub override; empty means use `.env` |

Legacy rows without `amount_btc`, `strategy`, or `prompt_mode` still work. The active 2026 matrix intentionally uses only the stable runner columns plus normalized tags, so the UI infers amount targets and strategy from prompt text and tags while preserving manual prompt text.

Future result CSV files may include optional trailing accounting columns. Older CSV readers can ignore them:

| Column | Meaning |
| --- | --- |
| `input_tokens` | Prompt/input tokens captured from provider metadata when available |
| `output_tokens` | Completion/output tokens captured from provider metadata when available |
| `total_tokens` | Total captured tokens |
| `estimated_cost_usd` | Actual-cost calculation when tokens exist, otherwise local estimate |
| `cost_source` | `actual`, `estimated`, or `missing-price` |

## Prompt Strategies

`prompt_templates.py` defines the currently supported Catalan prompt strategies:

| Strategy | Behavior |
| --- | --- |
| `basic` | Functional request only: create a PSBT for the amount |
| `privacy-simple` | One-shot request with a short privacy cue |
| `multiturn-simple` | Basic request followed by a simple privacy-improvement follow-up |
| `multiturn-detailed` | Basic request followed by detailed privacy instructions |
| `privacy-detailed` | One-shot request with detailed privacy instructions |

The active 2026 matrix uses only three of those strategies:

- `basic`
- `privacy-simple`
- `multiturn-detailed`

Historically, amount and privacy wording were embedded directly inside `user_prompt`. The template helpers still support structured ad hoc experiments in the UI, but the active 2026 batch keeps amount semantics in prompt text plus tags to avoid changing the runner contract.

## Scoring

The runner imports the scorer lazily from:

```text
/home/jaume/feina/analysis/scoring/privacy_scorer_v2.py
```

In the full research workspace, this scorer is TxPrivScore v2 and evaluates PSBT structure offline. If the scorer cannot be imported, the runner keeps CLI compatibility and records results without privacy scores.

Important result fields:

| Field | Meaning |
| --- | --- |
| `privacy_score` | Overall structural privacy score, 0-100 |
| `privacy_grade` | Letter grade derived from the score |
| `fee_sanity_ok` | `1` when fees look sane, `0` when fees look astronomically wrong |
| `sanity_status` | `ok`, `suspicious`, or `broken` |
| `fee_rate_sat_vb` | Estimated fee rate, when available |
| `fee_sats` | Estimated fee in satoshis, when available |
| `psbt_file` | Path to the saved PSBT artifact |

Treat `privacy_score` and fee sanity as separate dimensions. A PSBT can look structurally private while still being operationally unusable because of an absurd fee.

## Results

Each runner invocation writes a timestamped pair:

```text
results/experiments_YYYYMMDD_HHMMSS.csv
results/experiments_YYYYMMDD_HHMMSS.json
```

The runner creates that pair for the batch and rewrites it incrementally after each completed run, so the Streamlit Results view can inspect a live batch while it is still executing. The Results page also detects active `experiment_runner.py` processes, reconstructs the expected run count from the command's CSV/filter when possible, and auto-refreshes a live progress panel. The CSV is the compact summary. The JSON keeps detailed scorer breakdowns, agent responses, and PSBT metadata when available. Binary and Base64 PSBTs are saved under:

```text
results/psbts/YYYYMMDD_HHMMSS/
```

These files are local research artifacts and are intentionally ignored by Git.
The timestamped subdirectory matches the result CSV/JSON pair, so rerunning the same experiment ID does not overwrite older PSBT artifacts.

## Basic Checks

From the project root:

```bash
# CSV parsing and runner wiring
cd experiments
python experiment_runner.py experiments.csv --dry-run --filter id:exp_openai_gpt54_basic_pct10_t03
python experiment_runner.py experiments.csv --dry-run --filter id:exp_anthropic_opus47_basic_pct10_t03
python experiment_runner.py experiments.csv --dry-run --filter id:exp_openrouter_gemini31pro_basic_pct10_t03

# Syntax check
cd ..
python -m py_compile experiments/*.py src/bitcoin_ai_agent.py

# UI/helper tests
pytest -q tests/test_experiment_web_integration.py
pytest -q tests/test_llm_factory_compatibility.py
```

The dry-run does not call LLM APIs. Running real experiments may use API credits and can take several minutes per experiment.

Before launching the full enabled `basic`/`privacy-simple` batch, the intended workflow is:

1. Dry-run one representative row per provider lane.
2. Run one paid smoke test per selected model, preferably the `basic`, `10%`, `T0.3` row.
3. Run a small provider-lane smoke batch with `--parallel-profile provider`.
4. Only then launch the full enabled prompt-screening matrix with model lanes if provider limits look healthy.

## Research Notes

This experiment system is designed to preserve backward compatibility with the CSV + runner workflow while adding structured controls for new research variables. The important current limitation is that some research factors are still encoded in natural language prompts. The UI and prompt templates make amount and prompt strategy first-class without forcing a breaking schema migration.

The intended use is iterative local experimentation: define rows, run selected experiments, inspect generated PSBTs and scoring output, and compare how model and prompt choices affect privacy and fee sanity.

One known risk in the current 2026 batch is the `google/gemini-3.1-pro-preview` OpenRouter lane under `multiturn-detailed`: OpenRouter documents extra care around multi-turn reasoning/tool state, and the current runner does not explicitly preserve Gemini `reasoning_details`. Treat the disabled `multiturn-detailed` Gemini rows as opt-in until smoke-tested successfully.
