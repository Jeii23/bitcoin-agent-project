#!/usr/bin/env python3
"""Configure local Wasabi backend/client for the controlled private main chain.

The project uses a Bitcoin Core node with a private main-network chain. Wasabi
needs both Bitcoin Core RPC and a Wasabi backend/indexer that serves filters for
that chain. This script configures the local Wasabi backend to index the project
node, and configures the local Wasabi client to use that backend instead of the
public api.wasabiwallet.io service.

RPC credentials are read from .env and are never printed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_DIR / ".env"
WALLETS_DIR = PROJECT_DIR / "experiments" / "wallets"

BACKEND_HOME = WALLETS_DIR / "wasabi_backend"
BACKEND_CONFIG = BACKEND_HOME / ".walletwasabi" / "backend" / "Config.json"

CLIENT_HOME = WALLETS_DIR / "wasabi_baseline"
CLIENT_CONFIG = CLIENT_HOME / ".walletwasabi" / "client" / "Config.json"
CLIENT_INDEXSTORE = (
    CLIENT_HOME
    / ".walletwasabi"
    / "client"
    / "BitcoinStore"
    / "Main"
    / "IndexStore"
)
PRIVATE_BACKEND_URI = "http://localhost:37127/"


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


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def move_public_client_index_aside() -> Path | None:
    if not CLIENT_INDEXSTORE.exists():
        return None
    backup = CLIENT_INDEXSTORE.with_name("IndexStore.public-mainnet-backup")
    if backup.exists():
        return backup
    shutil.move(str(CLIENT_INDEXSTORE), str(backup))
    return backup


def main() -> int:
    env = load_env(ENV_PATH)
    required = ["BITCOIN_RPC_HOST", "BITCOIN_RPC_PORT", "BITCOIN_RPC_USER", "BITCOIN_RPC_PASSWORD"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing required .env keys: {', '.join(missing)}")

    rpc_endpoint = f"http://{env['BITCOIN_RPC_HOST']}:{env['BITCOIN_RPC_PORT']}"
    rpc_credentials = f"{env['BITCOIN_RPC_USER']}:{env['BITCOIN_RPC_PASSWORD']}"

    backend_config = {
        "Network": "Main",
        "BitcoinRpcConnectionString": rpc_credentials,
        "MainNetBitcoinCoreRpcEndPoint": rpc_endpoint,
        "TestNetBitcoinCoreRpcEndPoint": "http://localhost:48332",
        "RegTestBitcoinCoreRpcEndPoint": "http://localhost:18443",
        "FilterType": "legacy",
    }
    atomic_write_json(BACKEND_CONFIG, backend_config)

    client_config = json.loads(CLIENT_CONFIG.read_text(encoding="utf-8-sig"))
    client_config["BackendUri"] = PRIVATE_BACKEND_URI
    client_config["CoordinatorUri"] = ""
    client_config["UseTor"] = "Disabled"
    client_config["DownloadNewVersion"] = False
    client_config["UseBitcoinRpc"] = True
    client_config["BitcoinRpcEndPoint"] = rpc_endpoint
    client_config["BitcoinRpcCredentialString"] = rpc_credentials
    client_config["ExternalTransactionBroadcaster"] = "MempoolSpace"
    atomic_write_json(CLIENT_CONFIG, client_config)

    backup = move_public_client_index_aside()

    print("Wasabi private backend/client config updated.")
    print(f"  Backend config: {BACKEND_CONFIG}")
    print(f"  Client config:  {CLIENT_CONFIG}")
    print(f"  Private BackendUri: {PRIVATE_BACKEND_URI}")
    print(f"  BitcoinRpcEndPoint: {rpc_endpoint}")
    print("  BitcoinRpcConnectionString: <redacted>")
    if backup:
        print(f"  Public client IndexStore moved aside: {backup}")
    else:
        print("  Public client IndexStore: no move needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
