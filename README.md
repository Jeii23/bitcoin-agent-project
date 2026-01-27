# Bitcoin Agent Project

HD wallet address derivation, PSBT (BIP-174) construction / decoding, UTXO listing and a natural‑language agent interface – hardened for safety (no synthetic addresses, strict checksum validation, explicit change handling, Taproot output support).

## Key Features
- BIP32/84 HD receive & change address derivation (via custom logic + `hdwallet` fallback).
- Strict Bech32 / Bech32m checksum validation (BIP‑173 / BIP‑350). Mixed‑case rejected.
- PSBT v0 builder (BIP‑174) with:
  - Deterministic greedy UTXO selection (value‑descending) with realistic vbytes fee estimation.
  - Output script generation for P2PKH, P2SH, P2WPKH, P2WSH, and Taproot P2TR (witness v1).
  - Dust change suppression (dust < 546 sats is added to fee).
  - Fail‑fast if non‑dust change is required and no `change_address` supplied (no fake change placeholders).
  - Witness UTXO fallback builds correct scriptPubKey by address type when full prev tx not fetched.
- Decode helper for PSBT (basic field extraction + integrity checks).
- No synthetic / placeholder addresses anywhere (eliminates fund loss risk from fake Bech32 strings).
- Minimal dependency footprint (trimmed `requirements.txt`).
- Taproot (v1) output script support (OP_1 0x51 + push32 + program). Tests include BIP‑350 vectors.
- Comprehensive pytest suite (offline‑safe; HTTP calls monkey‑patched during tests).
- Optional conversational agent (LangChain + OpenAI) that can:
  - Generate unused receive addresses
  - List UTXOs (receive + change chains)
  - Show balance & fee rates
  - Create PSBT transactions (standard BIP‑174) and return full Base64 PSBT.

## Safety & Security Principles
1. Fail Fast: Invalid addresses, derivation errors, missing change, or checksum issues raise explicit errors.
2. No Silent Fallbacks: Removed all synthetic/fake address generation previously used as placeholders.
3. Explicit Change: If change > dust and you don't provide `change_address`, creation fails (prevents misrouting funds to a placeholder or unintended burn output).
4. Dust Handling: Change below dust threshold is treated as additional fee; no meaningless dust outputs.
5. Taproot Correctness: Witness v1 (32‑byte) programs encoded as `OP_1 0x20 <32 bytes>`.
6. Strict Validation: Bech32 vs Bech32m checksum constants chosen per witness version (v0 → Bech32, v1+ → Bech32m) per BIP‑350.
7. Offline Tests: Unit tests never hit the network directly (use monkeypatch). Library code keeps network requests isolated and timeout bounded.

## Quickstart
```bash
# (Recommended) Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install minimal runtime deps
pip install -r requirements.txt

# (Optional) Install pytest for tests
pip install pytest

# Run tests
pytest -q
```

## Core APIs
### Address Derivation
```
from address_derivation import derive_address_and_path
info = derive_address_and_path(xpub, network="testnet", index=0, change=False)
print(info["address"], info["path"])  # e.g. tb1q...
```
- Returns dict: `{address, path}` or raises on failure.
- Change chain: `change=True` → m/84'/1'/0'/1/i (testnet).

### UTXO Listing (agent tool)
Lists first N receive + change addresses; aggregates UTXOs.
```
from bitcoin_ai_agent import list_utxos
res = list_utxos.invoke({"xpub": XPUB, "network": "testnet"})
```

### PSBT Creation
```
from psbt_creator import create_transaction_psbt
res = create_transaction_psbt(
    xpub=XPUB,
    recipient_address="tb1q...dest",
  amount_sats=100_000,
    utxos=your_utxo_list,          # each: {txid, vout, value_satoshis, address}
    change_address="tb1q...change",
    network="testnet",
    fee_rate=10  # sat/vB (if fee_satoshis omitted)
)
if res["success"]:
    print("PSBT Base64:", res["psbt"])  # ready for signing
else:
    print("Error:", res["error"])       # includes missing change message if needed
```
Notes:
- Either provide `fee_rate` (dynamic estimation) OR explicit `fee_satoshis`.
- If selection leaves non‑dust change and `change_address` is None → error (intentional).

### PSBT Decoding
```
from psbt_creator import PSBTCreator
c = PSBTCreator(network="testnet")
info = c.decode_psbt(psbt_base64)
print(info)
```
Returns `{valid, version, tx, num_inputs, num_outputs}`.

### Taproot Example
```
# Create output to Taproot (testnet vector)
res = create_transaction_psbt(
    xpub=XPUB,
    recipient_address="tb1pqqqqp399et2xygdj5xreqhjjvcmzhxw4aywxecjdzew6hylgvsesf3hn0c",
  amount_sats=20_000,
    utxos=utxos,
    change_address="tb1q...change",
    fee_satoshis=1000,
    network="testnet"
)
```

## Conversational Agent (Optional)
Set environment variables in `.env`:
```
# Required: API key for your chosen LLM provider
OPENAI_API_KEY=sk-...          # For OpenAI GPT models
ANTHROPIC_API_KEY=sk-ant-...   # For Anthropic Claude models
GOOGLE_API_KEY=AI...           # For Google Gemini models
OPENROUTER_API_KEY=sk-or-...   # For OpenRouter (200+ models via single API)

# LLM provider selection (default: openai)
LLM_PROVIDER=openai            # Options: openai, anthropic, google, openrouter
LLM_MODEL=gpt-4o               # Override default model for provider

# Bitcoin configuration
BITCOIN_XPUB=your_testnet_xpub
BITCOIN_NETWORK=testnet
```

### Supported LLM Providers
| Provider | Default Model | Example Models |
| -------- | ------------- | -------------- |
| `openai` | gpt-4o | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3 |
| `anthropic` | claude-sonnet-4 | claude-sonnet-4, claude-opus-4, claude-3.5-sonnet |
| `google` | gemini-1.5-pro | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp |
| `openrouter` | anthropic/claude-sonnet-4 | meta-llama/llama-3.3-70b-instruct, deepseek/deepseek-r1, mistralai/mistral-large, qwen/qwen-2.5-72b-instruct |

**OpenRouter** provides access to 200+ models (Llama, DeepSeek, Qwen, Mistral, and more) through a single API. Get your key at [openrouter.ai](https://openrouter.ai/).

Run an interaction (example sketch):
```python
from bitcoin_ai_agent import BitcoinAIAgent
import asyncio, os

# Option 1: Use default provider from .env
agent = BitcoinAIAgent()

# Option 2: Explicitly select provider
agent = BitcoinAIAgent(llm_provider="openrouter", llm_model="meta-llama/llama-3.3-70b-instruct")

agent.setup(os.getenv("BITCOIN_XPUB"), "testnet")
response = asyncio.run(agent.chat("Crea una transacció de 0.001 BTC a tb1q..."))
print(response)
```
The agent responds in Catalan (current localization) and appends the full Base64 PSBT.

## Testing
Key test files:
- `tests/test_psbt_creator_tx.py` – creation paths incl. change / dust / insufficient funds / mixed address types.
- `tests/test_psbt_witness_utxo_fallback.py` – script correctness when full prev tx not available.
- `tests/test_taproot_output.py` – Taproot scriptPubKey vector validation.
- `tests/test_address_validation_failfast.py` – strict checksum enforcement.
- `tests/test_no_fake_addresses*.py` – ensures no synthetic fallback addresses.

Run selectively:
```
pytest -q tests/test_taproot_output.py
```

## Migration Notes (Recent Hardenings)
| Change | Impact |
| ------ | ------ |
| Removed synthetic fake addresses | Errors now raised; callers must handle exceptions. |
| `create_psbt` now returns dict (was Base64 string) | Access PSBT at `result['psbt']`; helper `create_psbt_base64()` provided. |
| Added Taproot (P2TR) script support | New validation tests; no API change. |
| Explicit change required | If change > dust and no `change_address`, function returns `success=False`. |
| Fail-fast address validation | Invalid Bech32/Bech32m rejected early. |

## Dependency Footprint
Minimal `requirements.txt` (only modules actually imported):
- hdwallet, base58
- requests, python-dotenv
- langchain-openai, langchain-core, langgraph, openai
- rich
Pytest is optional for running the test suite.

## Limitations / Non‑Goals
- No transaction signing or broadcasting (intentionally out of scope).
- No coin selection optimization beyond simple greedy + change/no‑change evaluation.
- No BIP32 hardened path derivation beyond standard m/84' scope for now.
- Limited PSBT decoding (not a full parser of all key-value pairs).

## Security Recommendations
- Always derive change addresses from the same XPUB that owns the inputs.
- Inspect PSBT before signing (validate outputs + fee).
- Never hard‑code a placeholder change address; treat error messages seriously.
- Use hardware wallets for signing real funds (this toolkit focuses on construction & analysis).

## Contributing
1. Fork & branch (`feat/taproot-upgrade`, etc.).
2. Add/adjust tests for any behavior change.
3. Keep public function signatures stable or provide shims.
4. Run `pytest -q` before opening a PR.

## License
MIT (see repository if license file added; otherwise treat code as provided under permissive terms – add a LICENSE file before production use).

## Disclaimer
Educational / testing purposes on testnet by default. Use with real funds only after thorough review and at your own risk.

---
If you need an additional example (e.g., multi‑input Taproot + legacy mix) or a deeper PSBT decoder, open an issue or request the enhancement.
