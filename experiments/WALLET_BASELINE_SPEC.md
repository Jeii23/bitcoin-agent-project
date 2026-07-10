# Wallet Baseline Specification — Historical Implemented Plan

Status: `IMPLEMENTED/HISTORICAL`. This specification records the design used to add wallet baselines. Current usage and behavior are documented in `README.md` and current code/tests. New changes require a new project-local plan rather than editing this completed specification as if it were pending.

# Wallet Baseline Experiments Spec

## Goal

Add a reproducible baseline for real wallet coin selection so wallet-generated
unsigned PSBTs can be scored with the same offline TxPrivScore pipeline as the
agent corpus.

## Non-Goals

- Do not modify `experiments.csv` or the existing LLM `experiment_runner.py`.
- Do not introduce signing, private-key handling, or broadcasting.
- Do not merge wallet baseline rows into `phase-1` or `phase-2`.
- Do not automate GUI wallet signing flows or require wallet seeds/passphrases.

## Decisions Made

- Use a separate CLI runner: `bitcoin-agent-project/experiments/wallet_baseline_runner.py`.
- Use a separate definition CSV: `bitcoin-agent-project/experiments/wallet_baseline.csv`.
- Use a separate output namespace: `experiments/results/wallet_baseline_YYYYMMDD_HHMMSS/`.
- Use the controlled 1 BTC wallet balance with nominal targets at 10%, 30%, 50%, 80%, and 95%.
- Keep wallet baselines comparable to the agent prompts by sending a nominal output target equal to `pct * 100,000,000 sats`.
- v1 adapters:
  - `electrum`: automatic unsigned PSBT creation with Electrum `payto --unsigned`.
  - `bitcoin_core`: automatic unsigned PSBT creation with Bitcoin Core `walletcreatefundedpsbt` from a descriptor watch-only wallet.
  - `import`: manually exported PSBTs from external wallets, scored through the same scorer.
- The comparison set is Electrum, Bitcoin Core, Sparrow, and Wasabi. Sparrow
  and Wasabi use disabled `adapter=import` rows until unsigned manual PSBT
  exports are placed under `experiments/wallet_imports/`.
- Wasabi has a safe preparation path, not an automatic PSBT builder: the
  runner can create a watch-only wallet JSON from the project xpub, run/check
  JSON-RPC, load the wallet, and verify controlled UTXOs. It records
  `wasabi_rpc_psbt_generation_supported=false` because the safe RPC surface
  does not provide a password-free BIP-174 PSBT builder.
- Wasabi also has a public-only skeleton preparation path for testing the
  documented hardware/cold-wallet PSBT workflow without hardware. The runner
  can generate `coldcard_public_only_skeleton.json` from the normalized public
  xpub, with `EncryptedSecret=null`, `PasswordVerified=true`, empty
  `HdPubKeys`, no seed/xprv/password, and an explicit placeholder
  `MasterFingerprint=00000000` when the real fingerprint is unavailable.
  This is only counted as a Wasabi score if Wasabi itself imports the skeleton
  and exports the unsigned BIP-174 PSBTs through its PSBT workflow.
- Sparrow remains `manual-import-only`: the local/documented CLI exposes app
  launch, network, directory, terminal, and version/help options, but no
  headless coin-selection or PSBT export subcommand.
- Fee policy for automatic wallet adapters is `wallet-native`; there is no silent fixed-fee fallback.
- The optional comparison uses `phase12_results.py` to summarize the canonical agent corpus by amount percentage.
- The Streamlit UI includes a Wallet Baselines page backed by a separate
  normalizer. It displays coverage, reliability, wallet-vs-agent deltas, and
  wallet-only charts without altering the agent corpus.

## Open Questions

- Which watch-only Electrum wallet path should become the local default, if any.
- Whether future analysis should compare by amount only or by amount plus
  prompt strategy/model category.

## Implementation Plan

1. Parse `wallet_baseline.csv` into typed experiment rows.
2. Implement PSBT import handling for binary and base64 BIP-174 payloads.
3. Reject non-PSBT payloads and PSBTs containing input signatures or finalized input scripts.
4. Implement Electrum preflight and automatic `payto --unsigned` execution.
5. Implement Bitcoin Core watch-only descriptor setup and automatic `walletcreatefundedpsbt` execution.
6. Reuse `privacy_scorer_v2.score_psbt_privacy` for every generated/imported PSBT.
7. Write normalized CSV/JSON result files with comparable fields: score, grade, fee sanity, fee rate, fee sats, input/output counts, artifact paths, tags.
8. Add an optional amount-level comparison CSV against the canonical agent corpus.
9. Add `wallet_baseline_results.py` and wallet chart builders for Web UI analysis.
10. Add Wasabi safe preparation/status commands and Sparrow CLI classification.
11. Add Wasabi public-only skeleton generation for the safe GUI PSBT workflow trial.
12. Document commands, safety constraints, blocked manual imports, and outputs.

## Tests To Add

- CSV parsing and target-satoshi normalization.
- Result-row normalization for CSV output.
- Import adapter scoring with an offline PSBT fixture generated in the test.
- Rejection of non-BIP-174 payloads.
- Rejection of PSBTs containing signature/finalization input keys.
- Dry-run CLI for the five default Electrum rows.
- Electrum preflight reporting for the known `aiohttp_socks` dependency failure.
- Wallet baseline result normalization across multiple run folders.
- Latest successful wallet/amount selection while keeping failed attempts for
  reliability.
- Wallet-vs-agent amount comparison.
- Wallet chart builders with partial four-wallet coverage.
- Wasabi watch-only JSON generation without seed/xprv/password material.
- Wasabi public-only skeleton generation without seed/xprv/password material,
  including zpub-to-xpub normalization and skeleton status markers.
- Wasabi JSON-RPC payloads restricted to safe methods and no wallet password.
- Wasabi RPC preflight with a mock daemon response.
- Rejection of Wasabi RPC transaction hex as non-BIP-174.
- Sparrow CLI classification as `manual-import-only`.

## Risks And Mitigations

- Risk: accidentally automating a wallet that has private keys.
  Mitigation: require an explicit Electrum wallet path and conservatively refuse wallet files that appear to contain seed/private-key markers.
- Risk: imported PSBTs are already signed or finalized.
  Mitigation: parse PSBT maps and reject partial signature, final scriptSig, and final scriptWitness input keys before scoring.
- Risk: Electrum dependencies fail at runtime.
  Mitigation: provide `--preflight` and return a specific hint for missing `aiohttp_socks`.
- Risk: Bitcoin Core baseline accidentally uses a private-key wallet or a broader UTXO pool.
  Mitigation: create/load a dedicated descriptor wallet with private keys disabled and import only the project xpub receive/change descriptors.
- Risk: wallet baselines are confused with agent phases.
  Mitigation: use a separate CSV, runner, result directory prefix, and optional comparison file.
- Risk: Sparrow/Wasabi manual exports are missing or finalized.
  Mitigation: keep import rows disabled until files exist and reject signed or
  finalized PSBTs before scoring.
- Risk: Wasabi RPC automation silently crosses into signing semantics.
  Mitigation: the runner refuses unsafe RPC methods such as `build`, never
  sends wallet passwords, and records Wasabi as unsupported by safe RPC for
  PSBT generation.
