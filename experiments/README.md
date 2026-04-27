# Bitcoin Agent Privacy Experiments

This directory contains the experiment workflow used to evaluate whether an xpub-only AI agent can construct Bitcoin PSBTs with better structural privacy.

The code is research tooling, not a production wallet. It never handles private keys, never signs transactions, and never broadcasts transactions. The agent can inspect wallet state from an extended public key, construct unsigned BIP-174 PSBTs, and pass those PSBTs to an offline privacy scorer.

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
├── web_ui.py                # Local Streamlit interface
├── experiment_manager.py    # Backward-compatible CSV read/write helpers
├── prompt_templates.py      # Prompt generation from amount + strategy
├── result_utils.py          # CSV/JSON result loading and normalization
├── paper_charts.py          # Optional chart helpers for result comparison
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
- monitor any active `experiment_runner.py` batch from the Results page, including terminal-launched runs,
- inspect result tables, scores, fee sanity, PSBT paths, and comparison charts.

The UI is optional. Any experiment created by the UI is still represented as a CSV row and can be run from the CLI.

## Active 2026 Matrix

The active `experiments.csv` is now the cleaned 2026 batch:

- `100` enabled Phase 1 screening rows:
  5 active models x 2 prompt strategies (`basic`, `privacy-simple`) x 5 amount targets (`10%`, `30%`, `50%`, `80%`, `95%`) x 2 temperatures (`0.3`, `1.0`)
- `40` disabled deferred Phase 1 frontier rows:
  2 high-cost frontier models kept in place for later reactivation, with the same prompts, IDs, and tags preserved
- `40` disabled Phase 2 challenge rows:
  4 pre-created finalist lanes x `multiturn-detailed` x the same 5 amount targets x the same 2 temperatures
- total CSV rows remain `180`
- `repetitions=1` for Phase 1 and `repetitions=2` for Phase 2

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
results/psbts/
```

These files are local research artifacts and are intentionally ignored by Git.

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

Before launching the full Phase 1 batch, the intended workflow is:

1. Dry-run one representative row per provider lane.
2. Run one paid smoke test per selected model, preferably the `basic`, `10%`, `T0.3` row.
3. Run a small provider-lane smoke batch with `--parallel-profile provider`.
4. Only then launch the full enabled Phase 1 screening matrix with model lanes if provider limits look healthy.

## Research Notes

This experiment system is designed to preserve backward compatibility with the CSV + runner workflow while adding structured controls for new research variables. The important current limitation is that some research factors are still encoded in natural language prompts. The UI and prompt templates make amount and prompt strategy first-class without forcing a breaking schema migration.

The intended use is iterative local experimentation: define rows, run selected experiments, inspect generated PSBTs and scoring output, and compare how model and prompt choices affect privacy and fee sanity.

One known risk in the current 2026 batch is the `google/gemini-3.1-pro-preview` OpenRouter lane under `multiturn-detailed`: OpenRouter documents extra care around multi-turn reasoning/tool state, and the current runner does not explicitly preserve Gemini `reasoning_details`. Treat the disabled Phase 2 Gemini rows as opt-in until smoke-tested successfully.
