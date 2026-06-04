#!/usr/bin/env python3
"""Configure the local Wasabi baseline datadir to use the project Bitcoin Core RPC.

This script reads the project .env and edits only the controlled Wasabi
baseline Config.json. It does not print RPC credentials.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_DIR / ".env"
CONFIG_PATH = (
    PROJECT_DIR
    / "experiments"
    / "wallets"
    / "wasabi_baseline"
    / ".walletwasabi"
    / "client"
    / "Config.json"
)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip().strip('"').strip("'")
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        values[key.strip()] = value
    return values


def main() -> int:
    env = load_env(ENV_PATH)
    required = ["BITCOIN_RPC_HOST", "BITCOIN_RPC_PORT", "BITCOIN_RPC_USER", "BITCOIN_RPC_PASSWORD"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing required .env keys: {', '.join(missing)}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    config["UseTor"] = "Disabled"
    config["UseBitcoinRpc"] = True
    config["BitcoinRpcEndPoint"] = f"http://{env['BITCOIN_RPC_HOST']}:{env['BITCOIN_RPC_PORT']}"
    config["BitcoinRpcCredentialString"] = (
        f"{env['BITCOIN_RPC_USER']}:{env['BITCOIN_RPC_PASSWORD']}"
    )
    config["FeeRateEstimationProvider"] = "None"
    config["ExchangeRateProvider"] = "None"
    # Wasabi validates this at startup and does not accept "None".
    # Keeping a supported value here does not broadcast anything by itself.
    config["ExternalTransactionBroadcaster"] = "MempoolSpace"
    config["DownloadNewVersion"] = False

    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Wasabi baseline RPC config updated.")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  BitcoinRpcEndPoint: {config['BitcoinRpcEndPoint']}")
    print("  BitcoinRpcCredentialString: <redacted>")
    print(f"  UseBitcoinRpc: {config['UseBitcoinRpc']}")
    print(f"  UseTor: {config['UseTor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
