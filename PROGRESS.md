# Bitcoin Agent Project Progress

Status: `MAINTENANCE`

This file records future implementation-level decisions that affect the xpub-only agent, experiment runner, wallet baselines, PSBT artifacts, or integration with the offline scorer. Historical TFM execution details remain in `../TFM/PROGRESS.md` and should not be duplicated here.

## Stable current constraints

- xpub-only observation and address derivation;
- unsigned BIP-174 PSBT construction;
- no private keys, signing, or broadcasting;
- CSV/CLI experiment compatibility;
- offline structural scoring;
- local researcher UI only;
- historical result files remain immutable unless an explicit migration is approved.

## Current state

The project is shared infrastructure rather than the sole active deliverable. Paper-specific changes should be recorded in the relevant paper progress log; update this file only when code, experiment contracts, wallet adapters, or shared behavior changes.

## Entry template

```md
## YYYY-MM-DD — Implementation decision

### Behavior changed
...

### Compatibility impact
...

### Files and tests
...

### Dependent papers
...
```
