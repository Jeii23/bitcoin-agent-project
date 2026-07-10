# Wasabi Manual PSBT Imports

Place unsigned BIP-174 PSBT exports here for the wallet baseline import adapter:

- `wasabi_pct10.psbt`
- `wasabi_pct30.psbt`
- `wasabi_pct50.psbt`
- `wasabi_pct80.psbt`
- `wasabi_pct95.psbt`

The corresponding `wallet_baseline.csv` rows are disabled until these files
exist. If Wasabi cannot create unsigned PSBTs from a watch-only or hardware-style
wallet without a seed/passphrase in the automation path, leave the rows disabled
and record the limitation in `bitcoin-agent-project/PROGRESS.md` and in the progress log of any paper that depends on the result.

For the public-only skeleton trial, generate:

```bash
python ../wallet_baseline_runner.py ../wallet_baseline.csv --wasabi-generate-skeleton
```

Then start Wasabi with the controlled datadir:

```bash
HOME=/home/jaume/feina/bitcoin-agent-project/experiments/wallets/wasabi_baseline \
  /home/jaume/feina/tools/wallet-baselines/wasabi-2.7.2/wassabee --network=main --usetor=false
```

Import `experiments/wallets/wasabi_baseline/coldcard_public_only_skeleton.json`
only as a public/cold skeleton. Count the files in this directory as Wasabi
baseline data only if Wasabi itself exports them as unsigned PSBTs through its
PSBT workflow.
