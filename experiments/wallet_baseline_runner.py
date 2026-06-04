#!/usr/bin/env python3
"""Wallet coin-selection baseline runner.

This runner is intentionally separate from experiment_runner.py. It never
imports the LLM agent, never signs, and never broadcasts. It either asks a
watch-only wallet to create an unsigned PSBT or imports an already exported
PSBT, then sends the PSBT through the same offline TxPrivScore scorer used by
the agent corpus.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
SCORING_DIR = WORKSPACE_ROOT / "analysis" / "scoring"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCORING_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from address_derivation import derive_bitcoin_address, _normalize_to_x_or_t_pub
except ImportError:  # pragma: no cover - surfaced at runtime with a clear error
    derive_bitcoin_address = None
    _normalize_to_x_or_t_pub = None

try:
    from privacy_scorer_v2 import score_psbt_privacy
except ImportError:  # pragma: no cover - surfaced at runtime with a clear error
    score_psbt_privacy = None


logger = logging.getLogger(__name__)

PSBT_MAGIC = b"psbt\xff"
CONTROLLED_BALANCE_SATS = 100_000_000
DEFAULT_AMOUNT_PCTS = ("10", "30", "50", "80", "95")
DEFAULT_RECIPIENT_INDEX = 50
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "results"

VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
ELECTRUM_SCRIPT = WORKSPACE_ROOT / "tools" / "electrum" / "run_electrum"
if VENV_PYTHON.exists() and ELECTRUM_SCRIPT.exists():
    DEFAULT_ELECTRUM_CMD = f"{VENV_PYTHON} {ELECTRUM_SCRIPT}"
elif ELECTRUM_SCRIPT.exists():
    DEFAULT_ELECTRUM_CMD = str(ELECTRUM_SCRIPT)
else:
    DEFAULT_ELECTRUM_CMD = "electrum"

LOCAL_WASABI_DAEMON = WORKSPACE_ROOT / "tools" / "wallet-baselines" / "wasabi-2.7.2" / "wassabeed"
DEFAULT_WASABI_CMD = str(LOCAL_WASABI_DAEMON) if LOCAL_WASABI_DAEMON.exists() else "wassabeed"
DEFAULT_WASABI_DATADIR = SCRIPT_DIR / "wallets" / "wasabi_baseline"
DEFAULT_WASABI_WALLET = "wallet_baseline_wasabi_watchonly"
DEFAULT_WASABI_RPC_URL = "http://127.0.0.1:37128"
DEFAULT_WASABI_STATUS_PATH = DEFAULT_WASABI_DATADIR / "wasabi_preflight_status.json"
DEFAULT_WASABI_PUBLIC_ONLY_SKELETON = DEFAULT_WASABI_DATADIR / "coldcard_public_only_skeleton.json"
DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER = "00000000"

LOCAL_SPARROW_CMD = WORKSPACE_ROOT / "tools" / "wallet-baselines" / "sparrow-2.4.2" / "bin" / "Sparrow"
DEFAULT_SPARROW_CMD = str(LOCAL_SPARROW_CMD) if LOCAL_SPARROW_CMD.exists() else "sparrow"

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}
SUPPORTED_ADAPTERS = {"electrum", "import", "bitcoin_core", "core"}
WASABI_SAFE_RPC_METHODS = {
    "getstatus",
    "listwallets",
    "loadwallet",
    "getwalletinfo",
    "listcoins",
    "listunspentcoins",
    "listkeys",
    "getfeerates",
}
SIGNED_INPUT_KEY_TYPES = {
    0x02: "partial signature",
    0x07: "final scriptSig",
    0x08: "final scriptWitness",
}

RESULT_CSV_COLUMNS = [
    "experiment_id",
    "wallet",
    "adapter",
    "amount_pct",
    "success",
    "error_message",
    "psbt_generated",
    "privacy_score",
    "privacy_grade",
    "fee_sanity_ok",
    "sanity_status",
    "fee_rate_sat_vb",
    "fee_sats",
    "num_inputs",
    "num_outputs",
    "psbt_file",
    "tags",
    "target_sats",
    "recipient_address",
    "timestamp",
]


class BaselineError(Exception):
    """Expected, user-facing baseline failure."""


class CoreRPCError(BaselineError):
    """Bitcoin Core RPC error with the RPC code preserved."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


@dataclass
class WalletBaselineExperiment:
    """One wallet-baseline experiment row."""

    id: str
    wallet: str
    adapter: str
    amount_pct: str
    network: str
    recipient_policy: str
    fee_policy: str
    psbt_file: Optional[str]
    tags: List[str]
    enabled: bool = True
    xpub: Optional[str] = None
    wallet_path: Optional[str] = None
    electrum_cmd: Optional[str] = None
    core_wallet: Optional[str] = None
    descriptor_range: int = 200
    wasabi_cmd: Optional[str] = None
    wasabi_datadir: Optional[str] = None
    wasabi_wallet: Optional[str] = None
    wasabi_rpc_url: Optional[str] = None
    wasabi_rpc_user: Optional[str] = None
    wasabi_rpc_password: Optional[str] = None
    sparrow_cmd: Optional[str] = None
    recipient_address: Optional[str] = None
    recipient_index: int = DEFAULT_RECIPIENT_INDEX
    target_sats: int = 0
    notes: str = ""


@dataclass
class WalletBaselineResult:
    """Normalized result row for wallet baselines."""

    experiment_id: str
    wallet: str
    adapter: str
    amount_pct: str
    timestamp: str
    success: bool = False
    error_message: Optional[str] = None
    psbt_generated: bool = False
    psbt_base64: Optional[str] = None
    psbt_file: Optional[str] = None
    privacy_score: Optional[int] = None
    privacy_grade: Optional[str] = None
    privacy_breakdown: Optional[Dict[str, Any]] = None
    fee_sanity_ok: Optional[int] = None
    sanity_status: str = ""
    fee_rate_sat_vb: Optional[float] = None
    fee_sats: Optional[int] = None
    num_inputs: Optional[int] = None
    num_outputs: Optional[int] = None
    target_sats: Optional[int] = None
    recipient_address: Optional[str] = None
    wallet_command: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AdapterOutput:
    """PSBT returned by a baseline adapter."""

    psbt_base64: str
    psbt_file: str
    recipient_address: Optional[str]
    command: Optional[List[str]] = None


def load_project_env() -> None:
    """Load project .env without making python-dotenv a hard dependency."""
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip().strip('"').strip("'")
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


def parse_bool(value: Any, default: bool = True) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def split_tags(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value or "").replace(",", "|").replace(";", "|")
    return [part.strip() for part in text.split("|") if part.strip()]


def normalize_decimal_text(value: Any) -> str:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        raise ValueError("amount_pct is required")
    try:
        dec = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid amount_pct: {value!r}") from exc
    if dec <= 0 or dec > 100:
        raise ValueError(f"amount_pct must be within (0, 100], got {value!r}")
    normalized = format(dec.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def target_sats_for_pct(amount_pct: str, initial_balance_sats: int = CONTROLLED_BALANCE_SATS) -> int:
    return int((Decimal(amount_pct) * Decimal(initial_balance_sats)) / Decimal(100))


def sats_to_btc_text(sats: int) -> str:
    return f"{Decimal(sats) / Decimal(100_000_000):.8f}"


def result_grade(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    if score >= 30:
        return "E"
    return "F"


class WalletBaselineCSVParser:
    """Parser for wallet_baseline.csv definitions."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)

    def parse(self) -> List[WalletBaselineExperiment]:
        experiments: List[WalletBaselineExperiment] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                try:
                    experiments.append(self._parse_row(row))
                except Exception as exc:
                    row_id = row.get("id") or f"line {row_number}"
                    raise ValueError(f"Error parsing wallet baseline row {row_id}: {exc}") from exc
        return experiments

    def _parse_row(self, row: Dict[str, str]) -> WalletBaselineExperiment:
        exp_id = (row.get("id") or "").strip()
        if not exp_id:
            raise ValueError("id is required")

        adapter = (row.get("adapter") or "").strip().lower().replace("-", "_")
        if adapter not in SUPPORTED_ADAPTERS:
            raise ValueError(f"adapter must be one of {sorted(SUPPORTED_ADAPTERS)}, got {adapter!r}")
        if adapter == "core":
            adapter = "bitcoin_core"

        amount_pct = normalize_decimal_text(row.get("amount_pct"))
        target_sats_raw = (row.get("target_sats") or "").strip()
        target_sats = int(target_sats_raw) if target_sats_raw else target_sats_for_pct(amount_pct)

        recipient_index_raw = (row.get("recipient_index") or "").strip()
        recipient_index = int(recipient_index_raw) if recipient_index_raw else DEFAULT_RECIPIENT_INDEX
        descriptor_range_raw = (row.get("descriptor_range") or "").strip()
        descriptor_range = int(descriptor_range_raw) if descriptor_range_raw else int(
            os.getenv("WALLET_BASELINE_DESCRIPTOR_RANGE", "200")
        )

        return WalletBaselineExperiment(
            id=exp_id,
            wallet=(row.get("wallet") or adapter).strip(),
            adapter=adapter,
            amount_pct=amount_pct,
            network=(row.get("network") or os.getenv("BITCOIN_NETWORK") or "mainnet").strip().lower(),
            recipient_policy=(row.get("recipient_policy") or "same-wallet-fresh").strip(),
            fee_policy=(row.get("fee_policy") or "wallet-native").strip(),
            psbt_file=(row.get("psbt_file") or "").strip() or None,
            xpub=(row.get("xpub") or "").strip() or os.getenv("BITCOIN_XPUB") or os.getenv("XPUB") or None,
            wallet_path=(row.get("wallet_path") or "").strip()
            or os.getenv("WALLET_BASELINE_ELECTRUM_WALLET")
            or os.getenv("ELECTRUM_WALLET_PATH")
            or None,
            electrum_cmd=(row.get("electrum_cmd") or "").strip()
            or os.getenv("WALLET_BASELINE_ELECTRUM_CMD")
            or os.getenv("ELECTRUM_CMD")
            or None,
            core_wallet=(row.get("core_wallet") or "").strip()
            or os.getenv("WALLET_BASELINE_CORE_WALLET")
            or "wallet_baseline_core_watchonly",
            descriptor_range=descriptor_range,
            wasabi_cmd=(row.get("wasabi_cmd") or "").strip()
            or os.getenv("WALLET_BASELINE_WASABI_CMD")
            or DEFAULT_WASABI_CMD,
            wasabi_datadir=(row.get("wasabi_datadir") or "").strip()
            or os.getenv("WALLET_BASELINE_WASABI_DATADIR")
            or str(DEFAULT_WASABI_DATADIR),
            wasabi_wallet=(row.get("wasabi_wallet") or "").strip()
            or os.getenv("WALLET_BASELINE_WASABI_WALLET")
            or DEFAULT_WASABI_WALLET,
            wasabi_rpc_url=(row.get("wasabi_rpc_url") or "").strip()
            or os.getenv("WALLET_BASELINE_WASABI_RPC_URL")
            or DEFAULT_WASABI_RPC_URL,
            wasabi_rpc_user=(row.get("wasabi_rpc_user") or "").strip()
            or os.getenv("WALLET_BASELINE_WASABI_RPC_USER")
            or None,
            wasabi_rpc_password=(row.get("wasabi_rpc_password") or "").strip()
            or os.getenv("WALLET_BASELINE_WASABI_RPC_PASSWORD")
            or None,
            sparrow_cmd=(row.get("sparrow_cmd") or "").strip()
            or os.getenv("WALLET_BASELINE_SPARROW_CMD")
            or DEFAULT_SPARROW_CMD,
            recipient_address=(row.get("recipient_address") or "").strip() or None,
            recipient_index=recipient_index,
            target_sats=target_sats,
            tags=split_tags(row.get("tags")),
            enabled=parse_bool(row.get("enabled"), default=True),
            notes=(row.get("notes") or "").strip(),
        )


def _compact_size_decode(data: bytes, offset: int) -> Tuple[int, int]:
    if offset >= len(data):
        raise ValueError("Truncated CompactSize")
    first = data[offset]
    if first < 0xFD:
        return first, 1
    if first == 0xFD:
        if offset + 3 > len(data):
            raise ValueError("Truncated CompactSize 0xfd")
        return int.from_bytes(data[offset + 1 : offset + 3], "little"), 3
    if first == 0xFE:
        if offset + 5 > len(data):
            raise ValueError("Truncated CompactSize 0xfe")
        return int.from_bytes(data[offset + 1 : offset + 5], "little"), 5
    if offset + 9 > len(data):
        raise ValueError("Truncated CompactSize 0xff")
    return int.from_bytes(data[offset + 1 : offset + 9], "little"), 9


def _read_psbt_map(data: bytes, offset: int) -> Tuple[List[Tuple[bytes, bytes]], int]:
    entries: List[Tuple[bytes, bytes]] = []
    while True:
        key_len, used = _compact_size_decode(data, offset)
        offset += used
        if key_len == 0:
            return entries, offset
        if offset + key_len > len(data):
            raise ValueError("Truncated PSBT key")
        key = data[offset : offset + key_len]
        offset += key_len

        value_len, used = _compact_size_decode(data, offset)
        offset += used
        if offset + value_len > len(data):
            raise ValueError("Truncated PSBT value")
        value = data[offset : offset + value_len]
        offset += value_len
        entries.append((key, value))


def _count_unsigned_tx_parts(tx_bytes: bytes) -> Tuple[int, int]:
    if len(tx_bytes) < 10:
        raise ValueError("Unsigned transaction too short")
    offset = 4
    if offset + 2 <= len(tx_bytes) and tx_bytes[offset : offset + 2] == b"\x00\x01":
        offset += 2

    n_inputs, used = _compact_size_decode(tx_bytes, offset)
    offset += used
    for _ in range(n_inputs):
        if offset + 36 > len(tx_bytes):
            raise ValueError("Truncated unsigned transaction input")
        offset += 36
        script_len, used = _compact_size_decode(tx_bytes, offset)
        offset += used + script_len
        if offset + 4 > len(tx_bytes):
            raise ValueError("Truncated unsigned transaction sequence")
        offset += 4

    n_outputs, used = _compact_size_decode(tx_bytes, offset)
    offset += used
    for _ in range(n_outputs):
        if offset + 8 > len(tx_bytes):
            raise ValueError("Truncated unsigned transaction output")
        offset += 8
        script_len, used = _compact_size_decode(tx_bytes, offset)
        offset += used + script_len
        if offset > len(tx_bytes):
            raise ValueError("Truncated unsigned transaction script")
    return n_inputs, n_outputs


def validate_bip174_unsigned_psbt(psbt_bytes: bytes) -> None:
    """Reject non-PSBT payloads and PSBTs that already contain signatures."""
    if not psbt_bytes.startswith(PSBT_MAGIC):
        raise BaselineError("Input is not a BIP-174 PSBT: missing psbt magic bytes")

    offset = len(PSBT_MAGIC)
    global_entries, offset = _read_psbt_map(psbt_bytes, offset)
    unsigned_tx = None
    for key, value in global_entries:
        if key and key[0] == 0x00:
            unsigned_tx = value
            break
    if unsigned_tx is None:
        raise BaselineError("PSBT is missing the global unsigned transaction")

    n_inputs, n_outputs = _count_unsigned_tx_parts(unsigned_tx)
    for input_index in range(n_inputs):
        input_entries, offset = _read_psbt_map(psbt_bytes, offset)
        for key, _value in input_entries:
            if key and key[0] in SIGNED_INPUT_KEY_TYPES:
                kind = SIGNED_INPUT_KEY_TYPES[key[0]]
                raise BaselineError(
                    f"Refusing signed/finalized PSBT: input {input_index} contains {kind}"
                )

    for _ in range(n_outputs):
        _output_entries, offset = _read_psbt_map(psbt_bytes, offset)


def decode_psbt_payload(payload: bytes | str) -> Tuple[str, bytes]:
    """Return normalized base64 and raw bytes for binary or base64 PSBT data."""
    if isinstance(payload, bytes) and payload.startswith(PSBT_MAGIC):
        psbt_bytes = payload
        psbt_base64 = base64.b64encode(psbt_bytes).decode("ascii")
        return psbt_base64, psbt_bytes

    text = payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else str(payload)
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            text = text.strip('"')

    match = re.search(r"cHNidP[A-Za-z0-9+/=\s]+", text)
    candidate = match.group(0) if match else text
    candidate = re.sub(r"\s+", "", candidate)
    try:
        psbt_bytes = base64.b64decode(candidate, validate=True)
    except Exception as exc:
        raise BaselineError("Could not decode PSBT as binary BIP-174 or base64") from exc
    return base64.b64encode(psbt_bytes).decode("ascii"), psbt_bytes


def read_psbt_file(psbt_path: Path) -> Tuple[str, bytes]:
    if not psbt_path.exists():
        raise BaselineError(f"PSBT file does not exist: {psbt_path}")
    psbt_base64, psbt_bytes = decode_psbt_payload(psbt_path.read_bytes())
    validate_bip174_unsigned_psbt(psbt_bytes)
    return psbt_base64, psbt_bytes


def resolve_recipient_address(exp: WalletBaselineExperiment) -> str:
    if exp.recipient_address:
        return exp.recipient_address
    if not exp.xpub:
        raise BaselineError(
            "Electrum baseline needs recipient_address or xpub/BITCOIN_XPUB "
            "to derive a fresh same-wallet destination"
        )
    if derive_bitcoin_address is None:
        raise BaselineError("address_derivation could not be imported")
    derived = derive_bitcoin_address(
        exp.xpub,
        index=exp.recipient_index,
        change=False,
        network=exp.network,
    )
    if not derived.get("success"):
        raise BaselineError(f"Could not derive recipient address: {derived.get('error')}")
    return str(derived["address"])


def command_parts(command: Optional[str]) -> List[str]:
    raw = command or DEFAULT_ELECTRUM_CMD
    parts = shlex.split(raw)
    if not parts:
        raise BaselineError("Electrum command is empty")
    return parts


def generic_command_parts(command: Optional[str], default_command: str, label: str) -> List[str]:
    raw = command or default_command
    parts = shlex.split(raw)
    if not parts:
        raise BaselineError(f"{label} command is empty")
    return parts


def electrum_dependency_hint(output: str) -> str:
    if "aiohttp_socks" in output:
        return (
            "Missing Electrum Python dependency 'aiohttp_socks'. Install it in the "
            "same Python environment used by the Electrum command."
        )
    if "No module named" in output:
        return output.strip().splitlines()[-1]
    return output.strip().splitlines()[-1] if output.strip() else "unknown error"


def check_electrum_preflight(command: Optional[str], timeout_seconds: int = 20) -> Tuple[bool, str]:
    parts = command_parts(command)
    executable = parts[0]
    if "/" not in executable and shutil.which(executable) is None:
        return False, f"Electrum command not found on PATH: {executable}"
    if "/" in executable and not Path(executable).exists():
        return False, f"Electrum command does not exist: {executable}"

    probe = parts + ["help", "payto"]
    try:
        completed = subprocess.run(
            probe,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Electrum preflight timed out after {timeout_seconds}s"
    except OSError as exc:
        return False, f"Electrum preflight failed to start: {exc}"

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        return False, electrum_dependency_hint(output)
    if "--unsigned" not in output and "unsigned" not in output.lower():
        return False, "Electrum payto help did not expose an unsigned PSBT option"
    return True, "ok"


def assert_watch_only_wallet_file(wallet_path: str) -> None:
    """Conservative local file check to avoid automating private-key wallets."""
    path = Path(wallet_path).expanduser()
    if not path.exists():
        raise BaselineError(f"Electrum wallet path does not exist: {path}")
    try:
        raw = path.read_bytes()[:2_000_000]
    except OSError as exc:
        raise BaselineError(f"Could not inspect Electrum wallet file for safety: {exc}") from exc

    text = raw.decode("utf-8", errors="ignore")
    private_fields = {"seed", "private_key", "prvkey", "keypairs"}

    def has_private_material(value: Any, parent_key: str = "") -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key).lower()
                if key_text == "xprv" and child:
                    return True
                if key_text in private_fields and child:
                    return True
                if has_private_material(child, key_text):
                    return True
            return False
        if isinstance(value, list):
            return any(has_private_material(item, parent_key) for item in value)
        return False

    try:
        wallet_data, _end = json.JSONDecoder().raw_decode(text.lstrip())
        has_private_keys = has_private_material(wallet_data)
    except json.JSONDecodeError:
        lowered = text.lower()
        private_markers = ('"xprv"', '"seed"', '"private_key"', '"prvkey"', '"keypairs"')
        has_private_keys = any(marker in lowered for marker in private_markers)

    if has_private_keys:
        raise BaselineError(
            "Refusing to automate this Electrum wallet because the wallet file "
            "appears to contain private-key or seed material. Use an xpub-only "
            "watch-only Electrum wallet for wallet baselines."
        )


def save_psbt_artifacts(psbt_base64: str, psbt_bytes: bytes, psbt_dir: Path, stem: str) -> str:
    psbt_dir.mkdir(parents=True, exist_ok=True)
    psbt_path = psbt_dir / f"{stem}.psbt"
    psbt_path.write_bytes(psbt_bytes)
    (psbt_dir / f"{stem}.base64").write_text(psbt_base64, encoding="ascii")
    return str(psbt_path)


def run_import_adapter(exp: WalletBaselineExperiment, psbt_dir: Path) -> AdapterOutput:
    if not exp.psbt_file:
        raise BaselineError("Import adapter requires psbt_file")
    source = Path(exp.psbt_file).expanduser()
    if not source.is_absolute():
        source = (SCRIPT_DIR / source).resolve()
    psbt_base64, psbt_bytes = read_psbt_file(source)
    psbt_path = save_psbt_artifacts(psbt_base64, psbt_bytes, psbt_dir, exp.id)
    return AdapterOutput(
        psbt_base64=psbt_base64,
        psbt_file=psbt_path,
        recipient_address=exp.recipient_address,
    )


def run_electrum_adapter(
    exp: WalletBaselineExperiment,
    psbt_dir: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> AdapterOutput:
    if exp.fee_policy != "wallet-native":
        raise BaselineError(
            f"Electrum adapter only supports fee_policy=wallet-native, got {exp.fee_policy!r}"
        )
    if not exp.wallet_path:
        raise BaselineError(
            "Electrum adapter requires wallet_path or WALLET_BASELINE_ELECTRUM_WALLET. "
            "The wallet must be xpub-only/watch-only."
        )
    assert_watch_only_wallet_file(exp.wallet_path)

    ok, message = check_electrum_preflight(exp.electrum_cmd)
    if not ok:
        raise BaselineError(f"Electrum preflight failed: {message}")

    recipient = resolve_recipient_address(exp)
    amount_btc = sats_to_btc_text(exp.target_sats)
    command = command_parts(exp.electrum_cmd) + [
        "payto",
        "--unsigned",
        recipient,
        amount_btc,
        "-w",
        str(Path(exp.wallet_path).expanduser()),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise BaselineError(f"Electrum payto failed: {electrum_dependency_hint(output)}")

    psbt_base64, psbt_bytes = decode_psbt_payload(output)
    validate_bip174_unsigned_psbt(psbt_bytes)
    psbt_path = save_psbt_artifacts(psbt_base64, psbt_bytes, psbt_dir, exp.id)
    return AdapterOutput(
        psbt_base64=psbt_base64,
        psbt_file=psbt_path,
        recipient_address=recipient,
        command=command,
    )


def core_rpc_settings() -> Dict[str, str]:
    required = {
        "host": os.getenv("BITCOIN_RPC_HOST", ""),
        "port": os.getenv("BITCOIN_RPC_PORT", ""),
        "user": os.getenv("BITCOIN_RPC_USER", ""),
        "password": os.getenv("BITCOIN_RPC_PASSWORD", ""),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise BaselineError(
            "Bitcoin Core adapter requires RPC settings in .env: "
            + ", ".join(f"BITCOIN_RPC_{key.upper()}" for key in missing)
        )
    return required


def core_rpc_call(
    method: str,
    params: Optional[List[Any]] = None,
    wallet: Optional[str] = None,
    timeout_seconds: int = 60,
) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise BaselineError("Bitcoin Core adapter requires the requests package") from exc

    settings = core_rpc_settings()
    base_url = f"http://{settings['host']}:{settings['port']}"
    url = base_url if wallet is None else f"{base_url}/wallet/{wallet}"
    response = requests.post(
        url,
        json={"jsonrpc": "1.0", "id": "wallet-baseline", "method": method, "params": params or []},
        auth=(settings["user"], settings["password"]),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        error = payload["error"]
        raise CoreRPCError(error.get("message", str(error)), error.get("code"))
    return payload.get("result")


def core_watchonly_descriptors(exp: WalletBaselineExperiment) -> List[Dict[str, Any]]:
    if not exp.xpub:
        raise BaselineError("Bitcoin Core adapter requires xpub/BITCOIN_XPUB")
    if _normalize_to_x_or_t_pub is None:
        raise BaselineError("address_derivation._normalize_to_x_or_t_pub could not be imported")

    normalized_xpub = _normalize_to_x_or_t_pub(exp.xpub)
    receive = core_rpc_call("getdescriptorinfo", [f"wpkh({normalized_xpub}/0/*)"])
    change = core_rpc_call("getdescriptorinfo", [f"wpkh({normalized_xpub}/1/*)"])
    descriptor_range = [0, exp.descriptor_range]
    return [
        {
            "desc": receive["descriptor"],
            "timestamp": 0,
            "active": True,
            "internal": False,
            "range": descriptor_range,
        },
        {
            "desc": change["descriptor"],
            "timestamp": 0,
            "active": True,
            "internal": True,
            "range": descriptor_range,
        },
    ]


def ensure_core_watchonly_wallet(exp: WalletBaselineExperiment) -> str:
    wallet_name = exp.core_wallet or "wallet_baseline_core_watchonly"
    loaded_wallets = core_rpc_call("listwallets")
    if wallet_name not in loaded_wallets:
        wallet_dir = core_rpc_call("listwalletdir")
        known_wallets = {
            wallet.get("name")
            for wallet in (wallet_dir.get("wallets") or [])
            if isinstance(wallet, dict)
        }
        if wallet_name in known_wallets:
            core_rpc_call("loadwallet", [wallet_name])
        else:
            core_rpc_call("createwallet", [wallet_name, True, True, "", False, True, False])

    info = core_rpc_call("getwalletinfo", wallet=wallet_name)
    if info.get("private_keys_enabled") is not False:
        raise BaselineError(
            f"Refusing Bitcoin Core wallet {wallet_name!r}: private_keys_enabled is not false"
        )
    if info.get("descriptors") is not True:
        raise BaselineError(f"Refusing Bitcoin Core wallet {wallet_name!r}: descriptors are not enabled")

    descriptors = core_watchonly_descriptors(exp)
    def descriptor_already_present(error: str) -> bool:
        lowered = error.lower()
        return (
            "already exists" in lowered
            or "new range must include current range" in lowered
        )

    try:
        import_result = core_rpc_call("importdescriptors", [descriptors], wallet=wallet_name, timeout_seconds=300)
    except CoreRPCError as exc:
        if not descriptor_already_present(str(exc)):
            raise
    else:
        errors = [
            item.get("error", {}).get("message", "")
            for item in import_result
            if isinstance(item, dict) and not item.get("success")
        ]
        non_duplicate_errors = [
            error for error in errors if not descriptor_already_present(error)
        ]
        if non_duplicate_errors:
            raise BaselineError(f"Bitcoin Core descriptor import failed: {non_duplicate_errors[0]}")

    return wallet_name


def run_bitcoin_core_adapter(exp: WalletBaselineExperiment, psbt_dir: Path) -> AdapterOutput:
    if exp.fee_policy != "wallet-native":
        raise BaselineError(
            f"Bitcoin Core adapter only supports fee_policy=wallet-native, got {exp.fee_policy!r}"
        )

    wallet_name = ensure_core_watchonly_wallet(exp)
    recipient = resolve_recipient_address(exp)
    amount_btc = sats_to_btc_text(exp.target_sats)

    options = {
        "replaceable": True,
    }
    result = core_rpc_call(
        "walletcreatefundedpsbt",
        [[], [{recipient: amount_btc}], 0, options, True],
        wallet=wallet_name,
        timeout_seconds=120,
    )
    psbt_base64 = result.get("psbt")
    if not psbt_base64:
        raise BaselineError("Bitcoin Core did not return a PSBT")
    psbt_base64, psbt_bytes = decode_psbt_payload(psbt_base64)
    validate_bip174_unsigned_psbt(psbt_bytes)
    psbt_path = save_psbt_artifacts(psbt_base64, psbt_bytes, psbt_dir, exp.id)
    return AdapterOutput(
        psbt_base64=psbt_base64,
        psbt_file=psbt_path,
        recipient_address=recipient,
        command=["bitcoin-core-rpc", f"wallet={wallet_name}", "walletcreatefundedpsbt"],
    )


def wasabi_wallet_name(exp: WalletBaselineExperiment) -> str:
    return (exp.wasabi_wallet or DEFAULT_WASABI_WALLET).strip() or DEFAULT_WASABI_WALLET


def wasabi_datadir(exp: WalletBaselineExperiment) -> Path:
    return Path(exp.wasabi_datadir or DEFAULT_WASABI_DATADIR).expanduser()


def wasabi_client_dir(exp: WalletBaselineExperiment) -> Path:
    return wasabi_datadir(exp) / ".walletwasabi" / "client"


def wasabi_wallet_file(exp: WalletBaselineExperiment) -> Path:
    return wasabi_client_dir(exp) / "Wallets" / f"{wasabi_wallet_name(exp)}.json"


def wasabi_public_only_skeleton_file(exp: WalletBaselineExperiment) -> Path:
    return wasabi_datadir(exp) / "coldcard_public_only_skeleton.json"


def wasabi_network_name(network: str) -> str:
    value = (network or "mainnet").strip().lower()
    if value in {"main", "mainnet"}:
        return "Main"
    if value in {"test", "testnet", "testnet4"}:
        return "TestNet"
    if value == "regtest":
        return "RegTest"
    raise BaselineError(f"Unsupported Wasabi network: {network!r}")


def wasabi_cli_network(network: str) -> str:
    value = (network or "mainnet").strip().lower()
    if value in {"main", "mainnet"}:
        return "main"
    if value in {"test", "testnet", "testnet4"}:
        return "testnet"
    if value == "regtest":
        return "regtest"
    raise BaselineError(f"Unsupported Wasabi network: {network!r}")


def wasabi_account_key_path(network: str) -> str:
    coin_type = "0" if wasabi_network_name(network) == "Main" else "1"
    return f"84'/{coin_type}'/0'"


def wasabi_taproot_account_key_path(network: str) -> str:
    return wasabi_account_key_path(network).replace("84'", "86'", 1)


def wasabi_normalized_xpub(exp: WalletBaselineExperiment) -> str:
    if not exp.xpub:
        raise BaselineError("Wasabi watch-only preparation requires xpub/BITCOIN_XPUB")
    if _normalize_to_x_or_t_pub is None:
        raise BaselineError("address_derivation._normalize_to_x_or_t_pub could not be imported")
    return str(_normalize_to_x_or_t_pub(exp.xpub))


def build_wasabi_watchonly_wallet_json(exp: WalletBaselineExperiment) -> Dict[str, Any]:
    """Build a Wasabi watch-only wallet file from public key material."""
    min_gap_limit = max(100, int(exp.descriptor_range or 100))
    wallet = {
        "WalletName": wasabi_wallet_name(exp),
        "EncryptedSecret": None,
        "ChainCode": None,
        "MasterFingerprint": None,
        "ExtPubKey": wasabi_normalized_xpub(exp),
        "TaprootExtPubKey": "",
        "PasswordVerified": True,
        "MinGapLimit": min_gap_limit,
        "AccountKeyPath": wasabi_account_key_path(exp.network),
        "TaprootAccountKeyPath": wasabi_taproot_account_key_path(exp.network),
        "BlockchainState": {
            "Network": wasabi_network_name(exp.network),
            "Height": "0",
        },
        "HdPubKeys": [],
    }
    assert_wasabi_watchonly_wallet_json_safe(wallet)
    return wallet


def build_wasabi_public_only_skeleton_json(
    exp: WalletBaselineExperiment,
    *,
    master_fingerprint: str = DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER,
) -> Dict[str, Any]:
    """Build a Wasabi/Coldcard-style skeleton using only public material."""
    fingerprint = (master_fingerprint or DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{8}", fingerprint):
        raise BaselineError(
            "Wasabi public-only skeleton master fingerprint must be 8 hex characters"
        )
    min_gap_limit = max(100, int(exp.descriptor_range or 100))
    skeleton = {
        "EncryptedSecret": None,
        "ChainCode": None,
        "MasterFingerprint": fingerprint.lower(),
        "ExtPubKey": wasabi_normalized_xpub(exp),
        "TaprootExtPubKey": "",
        "PasswordVerified": True,
        "MinGapLimit": min_gap_limit,
        "AccountKeyPath": wasabi_account_key_path(exp.network),
        "TaprootAccountKeyPath": wasabi_taproot_account_key_path(exp.network),
        "BlockchainState": {
            "Network": wasabi_network_name(exp.network),
            "Height": "0",
        },
        "HdPubKeys": [],
    }
    assert_wasabi_watchonly_wallet_json_safe(skeleton)
    return skeleton


def assert_wasabi_watchonly_wallet_json_safe(wallet_json: Dict[str, Any]) -> None:
    """Reject Wasabi wallet JSON that contains signing secrets or passwords."""

    private_key_names = {
        "mnemonic",
        "seed",
        "xprv",
        "xpriv",
        "privatekey",
        "private_key",
        "key",
        "password",
        "passphrase",
    }

    def walk(value: Any, parent_key: str = "") -> None:
        parent_lower = parent_key.lower()
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).replace("-", "").replace("_", "").lower()
                if key_lower == "encryptedsecret":
                    if child not in (None, "", [], {}):
                        raise BaselineError("Wasabi watch-only wallet must have EncryptedSecret=null")
                    continue
                if key_lower == "passwordverified":
                    continue
                if key_lower in private_key_names and child:
                    raise BaselineError(f"Refusing Wasabi wallet JSON with private field {key!r}")
                walk(child, str(key))
            return
        if isinstance(value, list):
            for item in value:
                walk(item, parent_key)
            return
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered.startswith(("xprv", "yprv", "zprv", "tprv", "uprv", "vprv")):
                raise BaselineError(f"Refusing Wasabi wallet JSON with private key in {parent_key!r}")
            if parent_lower in private_key_names and lowered:
                raise BaselineError(f"Refusing Wasabi wallet JSON with private field {parent_key!r}")

    walk(wallet_json)
    if wallet_json.get("EncryptedSecret") is not None:
        raise BaselineError("Wasabi watch-only wallet must not contain an encrypted secret")
    if not wallet_json.get("ExtPubKey"):
        raise BaselineError("Wasabi watch-only wallet must contain ExtPubKey")


def write_wasabi_watchonly_wallet_file(exp: WalletBaselineExperiment) -> Path:
    wallet_json = build_wasabi_watchonly_wallet_json(exp)
    wallet_path = wasabi_wallet_file(exp)
    wallet_path.parent.mkdir(parents=True, exist_ok=True)
    if wallet_path.exists():
        try:
            existing = json.loads(wallet_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise BaselineError(f"Existing Wasabi wallet file is not valid JSON: {wallet_path}") from exc
        assert_wasabi_watchonly_wallet_json_safe(existing)
        if existing.get("ExtPubKey") and existing.get("ExtPubKey") != wallet_json["ExtPubKey"]:
            raise BaselineError(
                f"Existing Wasabi watch-only wallet {wallet_path} uses a different ExtPubKey"
            )
    wallet_path.write_text(json.dumps(wallet_json, indent=2) + "\n", encoding="utf-8")
    return wallet_path


def _read_wasabi_status(status_path: Path) -> Dict[str, Any]:
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}


def _preserve_wasabi_skeleton_status(status: Dict[str, Any], previous: Dict[str, Any]) -> None:
    for key in (
        "skeleton_file",
        "skeleton_source",
        "synthetic_skeleton_generated",
        "synthetic_skeleton_imported",
        "psbt_workflow_available",
        "master_fingerprint",
        "master_fingerprint_source",
        "master_fingerprint_placeholder",
    ):
        if key in previous and key not in status:
            status[key] = previous[key]
    if previous.get("skeleton_source") == "generated-public-only":
        status["psbt_generation_route"] = previous.get(
            "psbt_generation_route",
            "wasabi-gui-psbt-workflow-export",
        )


def generate_wasabi_public_only_skeleton(
    exp: WalletBaselineExperiment,
    *,
    master_fingerprint: str = DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER,
    status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    skeleton_path = wasabi_public_only_skeleton_file(exp)
    skeleton = build_wasabi_public_only_skeleton_json(
        exp,
        master_fingerprint=master_fingerprint,
    )
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_path.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")

    marker_path = Path(status_path) if status_path is not None else wasabi_datadir(exp) / "wasabi_preflight_status.json"
    status = {
        "wallet": "wasabi",
        "wallet_name": wasabi_wallet_name(exp),
        "network": exp.network,
        "datadir": str(wasabi_datadir(exp)),
        "skeleton_file": str(skeleton_path),
        "skeleton_source": "generated-public-only",
        "synthetic_skeleton_generated": True,
        "synthetic_skeleton_imported": False,
        "psbt_workflow_available": None,
        "master_fingerprint": skeleton["MasterFingerprint"],
        "master_fingerprint_source": (
            "unknown-placeholder"
            if skeleton["MasterFingerprint"] == DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER
            else "user-provided"
        ),
        "master_fingerprint_placeholder": (
            skeleton["MasterFingerprint"] == DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER
        ),
        "watch_only_wallet_json_safe": True,
        "wasabi_rpc_psbt_generation_supported": False,
        "safe_rpc_psbt_generation_supported": False,
        "psbt_generation_route": "wasabi-gui-psbt-workflow-export",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_wasabi_status(marker_path, status)
    return status


def _assert_wasabi_rpc_params_safe(value: Any, parent_key: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).replace("-", "").replace("_", "").lower()
            if key_lower in {"password", "passphrase", "secret", "seed", "mnemonic"}:
                raise BaselineError(f"Refusing Wasabi RPC payload with sensitive field {key!r}")
            _assert_wasabi_rpc_params_safe(child, str(key))
        return
    if isinstance(value, list):
        for item in value:
            _assert_wasabi_rpc_params_safe(item, parent_key)


def wasabi_rpc_payload(method: str, params: Optional[Any] = None) -> Dict[str, Any]:
    method_name = method.strip().lower()
    if method_name not in WASABI_SAFE_RPC_METHODS:
        raise BaselineError(
            f"Refusing Wasabi RPC method {method!r}; safe automation only allows "
            f"{sorted(WASABI_SAFE_RPC_METHODS)}"
        )
    _assert_wasabi_rpc_params_safe(params)
    return {
        "jsonrpc": "2.0",
        "id": "wallet-baseline",
        "method": method_name,
        "params": [] if params is None else params,
    }


def wasabi_rpc_call(
    exp: WalletBaselineExperiment,
    method: str,
    params: Optional[Any] = None,
    *,
    wallet: Optional[str] = None,
    timeout_seconds: int = 20,
) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise BaselineError("Wasabi RPC preflight requires the requests package") from exc

    base_url = (exp.wasabi_rpc_url or DEFAULT_WASABI_RPC_URL).rstrip("/")
    url = base_url if wallet is None else f"{base_url}/wallet/{wallet}"
    auth = None
    if exp.wasabi_rpc_user or exp.wasabi_rpc_password:
        auth = (exp.wasabi_rpc_user or "", exp.wasabi_rpc_password or "")
    response = requests.post(
        url,
        json=wasabi_rpc_payload(method, params),
        auth=auth,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise BaselineError(f"Wasabi RPC {method} failed: {message}")
    return payload.get("result")


def wasabi_daemon_command(exp: WalletBaselineExperiment) -> List[str]:
    command = generic_command_parts(exp.wasabi_cmd, DEFAULT_WASABI_CMD, "Wasabi")
    rpc_url = (exp.wasabi_rpc_url or DEFAULT_WASABI_RPC_URL).rstrip("/") + "/"
    return command + [
        "--jsonrpcserverenabled=true",
        f"--jsonrpcserverprefixes={rpc_url}",
        "--usetor=false",
        f"--network={wasabi_cli_network(exp.network)}",
    ]


def start_wasabi_daemon(
    exp: WalletBaselineExperiment,
    *,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Start wassabeed in the controlled datadir if RPC is not already up."""
    try:
        wasabi_rpc_call(exp, "listwallets", timeout_seconds=3)
        return {
            "daemon_started": False,
            "daemon_already_running": True,
            "daemon_pid": None,
            "daemon_command": ["existing-wasabi-rpc", exp.wasabi_rpc_url or DEFAULT_WASABI_RPC_URL],
        }
    except Exception:
        pass

    datadir = wasabi_datadir(exp)
    datadir.mkdir(parents=True, exist_ok=True)
    log_path = datadir / "wassabeed.log"
    command = wasabi_daemon_command(exp)
    executable = command[0]
    if "/" not in executable and shutil.which(executable) is None:
        raise BaselineError(f"Wasabi daemon command not found on PATH: {executable}")
    if "/" in executable and not Path(executable).exists():
        raise BaselineError(f"Wasabi daemon command does not exist: {executable}")

    env = os.environ.copy()
    env["HOME"] = str(datadir)
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(datadir),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    deadline = datetime.now().timestamp() + timeout_seconds
    last_error = ""
    while datetime.now().timestamp() < deadline:
        if process.poll() is not None:
            raise BaselineError(
                f"Wasabi daemon exited early with code {process.returncode}. See log: {log_path}"
            )
        try:
            wasabi_rpc_call(exp, "listwallets", timeout_seconds=3)
            return {
                "daemon_started": True,
                "daemon_already_running": False,
                "daemon_pid": process.pid,
                "daemon_command": command,
                "daemon_log": str(log_path),
            }
        except Exception as exc:
            last_error = str(exc)
            import time

            time.sleep(1)
    raise BaselineError(
        f"Wasabi daemon did not expose RPC within {timeout_seconds}s. "
        f"Last error: {last_error}. See log: {log_path}"
    )


def _wasabi_wallet_is_listed(wallets: Any, wallet_name: str) -> bool:
    if not isinstance(wallets, list):
        return False
    for wallet in wallets:
        if isinstance(wallet, str) and wallet == wallet_name:
            return True
        if isinstance(wallet, dict):
            names = {wallet.get("walletName"), wallet.get("WalletName"), wallet.get("name"), wallet.get("Name")}
            if wallet_name in names:
                return True
    return False


def _coin_amount_to_sats(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        dec = Decimal(str(value))
    except InvalidOperation:
        return 0
    if dec <= 21_000_000 and "." in str(value):
        return int(dec * Decimal(100_000_000))
    return int(dec)


def _summarize_wasabi_coins(coins: Any) -> Tuple[int, int]:
    if isinstance(coins, dict):
        iterable = coins.get("coins") or coins.get("Coins") or coins.get("unspentCoins") or []
    else:
        iterable = coins if isinstance(coins, list) else []
    count = 0
    total_sats = 0
    for coin in iterable:
        if not isinstance(coin, dict):
            continue
        count += 1
        total_sats += _coin_amount_to_sats(
            coin.get("amount")
            or coin.get("Amount")
            or coin.get("amountSats")
            or coin.get("amount_sats")
            or coin.get("value")
            or coin.get("Value")
        )
    return count, total_sats


def run_wasabi_rpc_preflight(
    exp: WalletBaselineExperiment,
    *,
    rpc_call=wasabi_rpc_call,
    timeout_seconds: int = 20,
) -> Dict[str, Any]:
    wallet_name = wasabi_wallet_name(exp)
    status: Dict[str, Any] = {
        "rpc_ok": False,
        "wallet_loaded": False,
        "utxo_check_ok": False,
        "coin_count": 0,
        "coin_total_sats": 0,
        "wasabi_rpc_psbt_generation_supported": False,
        "safe_rpc_psbt_generation_supported": False,
    }
    wallets = rpc_call(exp, "listwallets", timeout_seconds=timeout_seconds)
    if not _wasabi_wallet_is_listed(wallets, wallet_name):
        rpc_call(exp, "loadwallet", [wallet_name], timeout_seconds=timeout_seconds)
    status["wallet_loaded"] = True

    try:
        status["wallet_info"] = rpc_call(exp, "getwalletinfo", wallet=wallet_name, timeout_seconds=timeout_seconds)
    except BaselineError as exc:
        status["wallet_info_error"] = str(exc)

    try:
        coins = rpc_call(exp, "listunspentcoins", wallet=wallet_name, timeout_seconds=timeout_seconds)
    except Exception as exc:
        status["listunspentcoins_error"] = str(exc)
        coins = rpc_call(exp, "listcoins", wallet=wallet_name, timeout_seconds=timeout_seconds)
    coin_count, coin_total_sats = _summarize_wasabi_coins(coins)
    status["coin_count"] = coin_count
    status["coin_total_sats"] = coin_total_sats
    status["utxo_check_ok"] = coin_count > 0 and abs(coin_total_sats - CONTROLLED_BALANCE_SATS) <= 100_000

    try:
        keys = rpc_call(exp, "listkeys", wallet=wallet_name, timeout_seconds=timeout_seconds)
        status["listkeys_available"] = True
        if isinstance(keys, list):
            status["key_count"] = len(keys)
    except Exception as exc:
        status["listkeys_available"] = False
        status["listkeys_error"] = str(exc)

    status["rpc_ok"] = True
    status["prepared_via_rpc"] = bool(status["wallet_loaded"] and status["utxo_check_ok"])
    return status


def prepare_wasabi_watchonly(
    exp: WalletBaselineExperiment,
    *,
    rpc_check: bool = False,
    start_daemon: bool = False,
    status_path: Optional[Path] = None,
    timeout_seconds: int = 20,
    rpc_call=wasabi_rpc_call,
) -> Dict[str, Any]:
    wallet_path = write_wasabi_watchonly_wallet_file(exp)
    marker_path = Path(status_path) if status_path is not None else wasabi_datadir(exp) / "wasabi_preflight_status.json"
    previous_status = _read_wasabi_status(marker_path)
    status: Dict[str, Any] = {
        "wallet": "wasabi",
        "wallet_name": wasabi_wallet_name(exp),
        "network": exp.network,
        "datadir": str(wasabi_datadir(exp)),
        "wallet_file": str(wallet_path),
        "wasabi_cmd": exp.wasabi_cmd or DEFAULT_WASABI_CMD,
        "wasabi_rpc_url": exp.wasabi_rpc_url or DEFAULT_WASABI_RPC_URL,
        "watch_only_wallet_json_safe": True,
        "watch_only_prepared": True,
        "rpc_checked": bool(rpc_check),
        "rpc_ok": False,
        "utxo_check_ok": False,
        "prepared_via_rpc": False,
        "wasabi_rpc_psbt_generation_supported": False,
        "safe_rpc_psbt_generation_supported": False,
        "psbt_generation_route": "manual-psbt-workflow-export",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _preserve_wasabi_skeleton_status(status, previous_status)
    if start_daemon:
        try:
            status.update(start_wasabi_daemon(exp, timeout_seconds=timeout_seconds))
            rpc_check = True
        except Exception as exc:
            status["error_message"] = str(exc)
            _write_wasabi_status(marker_path, status)
            raise
    if rpc_check:
        try:
            status.update(run_wasabi_rpc_preflight(exp, rpc_call=rpc_call, timeout_seconds=timeout_seconds))
            if not status.get("utxo_check_ok"):
                raise BaselineError(
                    "Wasabi RPC responded, but the watch-only wallet did not expose the controlled "
                    f"1 BTC UTXO set (coins={status.get('coin_count')}, "
                    f"total_sats={status.get('coin_total_sats')})."
                )
        except Exception as exc:
            status["error_message"] = str(exc)
            _write_wasabi_status(marker_path, status)
            raise
    _write_wasabi_status(marker_path, status)
    return status


def _write_wasabi_status(status_path: Path, status: Dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def validate_wasabi_rpc_psbt_result(value: Any) -> str:
    """Accept only BIP-174 PSBTs; Wasabi RPC build hex is rejected here."""
    candidate = value
    if isinstance(value, dict):
        candidate = value.get("psbt") or value.get("PSBT") or value.get("transaction") or value.get("hex")
    try:
        psbt_base64, psbt_bytes = decode_psbt_payload(candidate)
        validate_bip174_unsigned_psbt(psbt_bytes)
        return psbt_base64
    except Exception as exc:
        raise BaselineError("Wasabi RPC result is not a BIP-174 unsigned PSBT") from exc


def classify_sparrow_cli(
    command: Optional[str] = None,
    *,
    timeout_seconds: int = 20,
) -> Dict[str, Any]:
    parts = generic_command_parts(command, DEFAULT_SPARROW_CMD, "Sparrow")
    executable = parts[0]
    if "/" not in executable and shutil.which(executable) is None:
        return {
            "wallet": "sparrow",
            "status": "unavailable",
            "manual_import_only": True,
            "reason": f"Sparrow command not found on PATH: {executable}",
            "command": parts,
        }
    if "/" in executable and not Path(executable).exists():
        return {
            "wallet": "sparrow",
            "status": "unavailable",
            "manual_import_only": True,
            "reason": f"Sparrow command does not exist: {executable}",
            "command": parts,
        }

    try:
        completed = subprocess.run(
            parts + ["--help"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "wallet": "sparrow",
            "status": "unavailable",
            "manual_import_only": True,
            "reason": f"Sparrow help timed out after {timeout_seconds}s",
            "command": parts,
        }
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    output_lower = output.lower()
    psbt_commands = ("createpsbt", "walletcreatefundedpsbt", "payto", "sendmany", "sendtoaddress")
    has_headless_builder = any(marker in output_lower for marker in psbt_commands)
    status = "headless-psbt-supported" if has_headless_builder else "manual-import-only"
    return {
        "wallet": "sparrow",
        "status": status,
        "manual_import_only": not has_headless_builder,
        "returncode": completed.returncode,
        "command": parts + ["--help"],
        "reason": (
            "Sparrow help exposes a transaction-builder command"
            if has_headless_builder
            else "Sparrow CLI exposes app/terminal options only; use manual unsigned PSBT export"
        ),
        "help_excerpt": "\n".join(output.splitlines()[:20]),
    }


def extract_score_fields(report: Any) -> Dict[str, Any]:
    data = report.to_dict()
    fee_analysis = data.get("fee_analysis") or {}
    metadata = data.get("metadata") or {}
    score = int(report.scores["overall"])
    return {
        "privacy_score": score,
        "privacy_grade": result_grade(score),
        "privacy_breakdown": data,
        "fee_sanity_ok": data.get("fee_sanity_ok"),
        "sanity_status": data.get("sanity_status", ""),
        "fee_rate_sat_vb": fee_analysis.get("fee_rate_sat_vb"),
        "fee_sats": fee_analysis.get("fee_sats"),
        "num_inputs": metadata.get("num_inputs"),
        "num_outputs": metadata.get("num_outputs"),
    }


def result_to_csv_row(result: WalletBaselineResult) -> Dict[str, Any]:
    return {
        "experiment_id": result.experiment_id,
        "wallet": result.wallet,
        "adapter": result.adapter,
        "amount_pct": result.amount_pct,
        "success": int(bool(result.success)),
        "error_message": result.error_message or "",
        "psbt_generated": int(bool(result.psbt_generated)),
        "privacy_score": result.privacy_score if result.privacy_score is not None else "",
        "privacy_grade": result.privacy_grade or "",
        "fee_sanity_ok": result.fee_sanity_ok if result.fee_sanity_ok is not None else "",
        "sanity_status": result.sanity_status or "",
        "fee_rate_sat_vb": result.fee_rate_sat_vb if result.fee_rate_sat_vb is not None else "",
        "fee_sats": result.fee_sats if result.fee_sats is not None else "",
        "num_inputs": result.num_inputs if result.num_inputs is not None else "",
        "num_outputs": result.num_outputs if result.num_outputs is not None else "",
        "psbt_file": result.psbt_file or "",
        "tags": "|".join(result.tags),
        "target_sats": result.target_sats if result.target_sats is not None else "",
        "recipient_address": result.recipient_address or "",
        "timestamp": result.timestamp,
    }


class WalletBaselineRunner:
    """Execute wallet baseline rows and write CSV/JSON results."""

    def __init__(
        self,
        output_root: Path = DEFAULT_RESULT_ROOT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        run_timestamp: Optional[str] = None,
    ):
        self.run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_root) / f"wallet_baseline_{self.run_timestamp}"
        self.psbt_dir = self.output_dir / "psbts"
        self.timeout_seconds = timeout_seconds
        self.results: List[WalletBaselineResult] = []

    def run(self, experiments: Sequence[WalletBaselineExperiment]) -> List[WalletBaselineResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for exp in experiments:
            result = self.run_one(exp)
            self.results.append(result)
            self.write_results()
        return self.results

    def run_one(self, exp: WalletBaselineExperiment) -> WalletBaselineResult:
        result = WalletBaselineResult(
            experiment_id=exp.id,
            wallet=exp.wallet,
            adapter=exp.adapter,
            amount_pct=exp.amount_pct,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            target_sats=exp.target_sats,
            recipient_address=exp.recipient_address,
            tags=exp.tags,
        )

        try:
            if exp.adapter == "import":
                adapter_output = run_import_adapter(exp, self.psbt_dir)
            elif exp.adapter == "electrum":
                adapter_output = run_electrum_adapter(exp, self.psbt_dir, self.timeout_seconds)
            elif exp.adapter == "bitcoin_core":
                adapter_output = run_bitcoin_core_adapter(exp, self.psbt_dir)
            else:  # pragma: no cover - parser prevents this
                raise BaselineError(f"Unsupported adapter: {exp.adapter}")

            result.psbt_generated = True
            result.psbt_base64 = adapter_output.psbt_base64
            result.psbt_file = adapter_output.psbt_file
            result.recipient_address = adapter_output.recipient_address or result.recipient_address
            result.wallet_command = adapter_output.command

            if score_psbt_privacy is None:
                raise BaselineError("privacy_scorer_v2 could not be imported")
            report = score_psbt_privacy(adapter_output.psbt_base64, network=exp.network)
            for key, value in extract_score_fields(report).items():
                setattr(result, key, value)
            result.success = True
        except Exception as exc:
            result.success = False
            result.error_message = str(exc)
            logger.warning("%s failed: %s", exp.id, exc)
        return result

    def write_results(self) -> Tuple[Path, Path]:
        csv_path = self.output_dir / f"wallet_baseline_{self.run_timestamp}.csv"
        json_path = self.output_dir / f"wallet_baseline_{self.run_timestamp}.json"

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_CSV_COLUMNS)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result_to_csv_row(result))

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(result) for result in self.results], handle, indent=2)

        return csv_path, json_path


def apply_overrides(
    experiments: Iterable[WalletBaselineExperiment],
    *,
    xpub: Optional[str] = None,
    electrum_wallet: Optional[str] = None,
    electrum_cmd: Optional[str] = None,
    wasabi_cmd: Optional[str] = None,
    wasabi_datadir: Optional[str] = None,
    wasabi_wallet: Optional[str] = None,
    wasabi_rpc_url: Optional[str] = None,
    sparrow_cmd: Optional[str] = None,
    recipient_index: Optional[int] = None,
) -> List[WalletBaselineExperiment]:
    result = []
    for exp in experiments:
        if xpub:
            exp.xpub = xpub
        if electrum_wallet and exp.adapter == "electrum":
            exp.wallet_path = electrum_wallet
        if electrum_cmd and exp.adapter == "electrum":
            exp.electrum_cmd = electrum_cmd
        if wasabi_cmd and exp.wallet.lower() == "wasabi":
            exp.wasabi_cmd = wasabi_cmd
        if wasabi_datadir and exp.wallet.lower() == "wasabi":
            exp.wasabi_datadir = wasabi_datadir
        if wasabi_wallet and exp.wallet.lower() == "wasabi":
            exp.wasabi_wallet = wasabi_wallet
        if wasabi_rpc_url and exp.wallet.lower() == "wasabi":
            exp.wasabi_rpc_url = wasabi_rpc_url
        if sparrow_cmd and exp.wallet.lower().startswith("sparrow"):
            exp.sparrow_cmd = sparrow_cmd
        if recipient_index is not None:
            exp.recipient_index = recipient_index
        result.append(exp)
    return result


def create_filter(raw_filter: Optional[str]):
    if not raw_filter:
        return lambda _exp: True
    if ":" not in raw_filter:
        raise ValueError("Filter must use key:value syntax")
    key, value = raw_filter.split(":", 1)
    key = key.strip().lower()
    value = value.strip()
    if key == "adapter":
        value = value.lower().replace("-", "_")
        if value == "core":
            value = "bitcoin_core"
    values = {part.strip() for part in value.split(",") if part.strip()}

    def matches(exp: WalletBaselineExperiment) -> bool:
        if key == "id":
            return exp.id == value
        if key == "ids":
            return exp.id in values
        if key == "wallet":
            return exp.wallet == value
        if key == "adapter":
            return exp.adapter == value
        if key == "tag":
            return value in exp.tags
        if key == "amount_pct":
            return exp.amount_pct == normalize_decimal_text(value)
        raise ValueError(f"Unsupported filter key: {key}")

    return matches


def selected_experiments(
    experiments: List[WalletBaselineExperiment],
    raw_filter: Optional[str],
    include_disabled: bool,
) -> List[WalletBaselineExperiment]:
    matcher = create_filter(raw_filter)
    return [exp for exp in experiments if matcher(exp) and (exp.enabled or include_disabled)]


def print_dry_run(experiments: Sequence[WalletBaselineExperiment]) -> None:
    print(f"Wallet baseline dry-run: {len(experiments)} experiment(s)")
    for exp in experiments:
        print(
            f"- {exp.id}: wallet={exp.wallet} adapter={exp.adapter} "
            f"amount_pct={exp.amount_pct}% target_sats={exp.target_sats} "
            f"network={exp.network} fee_policy={exp.fee_policy}"
        )


def run_preflight(experiments: Sequence[WalletBaselineExperiment]) -> bool:
    electrum_commands = sorted(
        {
            exp.electrum_cmd or DEFAULT_ELECTRUM_CMD
            for exp in experiments
            if exp.adapter == "electrum"
        }
    )
    if not electrum_commands:
        print("No Electrum rows selected; preflight OK.")
        return True

    ok_all = True
    for command in electrum_commands:
        ok, message = check_electrum_preflight(command)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {command}: {message}")
        ok_all = ok_all and ok
    return ok_all


def first_wasabi_experiment(
    selected: Sequence[WalletBaselineExperiment],
    all_experiments: Sequence[WalletBaselineExperiment],
) -> Optional[WalletBaselineExperiment]:
    for exp in selected:
        if exp.wallet.strip().lower() == "wasabi":
            return exp
    for exp in all_experiments:
        if exp.wallet.strip().lower() == "wasabi":
            return exp
    return None


def first_sparrow_experiment(
    selected: Sequence[WalletBaselineExperiment],
    all_experiments: Sequence[WalletBaselineExperiment],
) -> Optional[WalletBaselineExperiment]:
    for exp in selected:
        if exp.wallet.strip().lower().startswith("sparrow"):
            return exp
    for exp in all_experiments:
        if exp.wallet.strip().lower().startswith("sparrow"):
            return exp
    return None


def write_agent_comparison(results: Sequence[WalletBaselineResult], output_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Create a lightweight amount-level comparison against the canonical agent corpus."""
    try:
        from phase12_results import normalize_phase12_results
    except Exception as exc:
        return None, f"Could not import phase12_results: {exc}"

    try:
        df = normalize_phase12_results()
        if df.empty:
            return None, "Canonical phase12 corpus is empty"
        agent_fee_ok = df[(df["fee_ok_bool"]) & (df["amount_pct"].notna())].copy()
        if agent_fee_ok.empty:
            return None, "Canonical phase12 corpus has no fee-sane rows"
        grouped = agent_fee_ok.groupby("amount_pct").agg(
            agent_fee_sane_count=("usable_score", "count"),
            agent_mean_privacy_score=("usable_score", "mean"),
            agent_median_fee_sats=("fee_sats", "median"),
            agent_median_fee_rate_sat_vb=("fee_rate_sat_vb", "median"),
        )
    except Exception as exc:
        return None, f"Could not normalize phase12 corpus: {exc}"

    comparison_path = output_dir / "wallet_vs_agent_amount_summary.csv"
    columns = [
        "experiment_id",
        "wallet",
        "adapter",
        "amount_pct",
        "wallet_success",
        "wallet_fee_sanity_ok",
        "wallet_privacy_score",
        "wallet_fee_sats",
        "wallet_fee_rate_sat_vb",
        "agent_fee_sane_count",
        "agent_mean_privacy_score",
        "agent_median_fee_sats",
        "agent_median_fee_rate_sat_vb",
        "wallet_minus_agent_mean",
    ]
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            try:
                amount_key = float(result.amount_pct)
            except ValueError:
                amount_key = None
            agent_row = grouped.loc[amount_key] if amount_key in grouped.index else None
            agent_mean = None if agent_row is None else float(agent_row["agent_mean_privacy_score"])
            wallet_score = result.privacy_score
            writer.writerow(
                {
                    "experiment_id": result.experiment_id,
                    "wallet": result.wallet,
                    "adapter": result.adapter,
                    "amount_pct": result.amount_pct,
                    "wallet_success": int(bool(result.success)),
                    "wallet_fee_sanity_ok": result.fee_sanity_ok if result.fee_sanity_ok is not None else "",
                    "wallet_privacy_score": wallet_score if wallet_score is not None else "",
                    "wallet_fee_sats": result.fee_sats if result.fee_sats is not None else "",
                    "wallet_fee_rate_sat_vb": result.fee_rate_sat_vb if result.fee_rate_sat_vb is not None else "",
                    "agent_fee_sane_count": "" if agent_row is None else int(agent_row["agent_fee_sane_count"]),
                    "agent_mean_privacy_score": "" if agent_mean is None else round(agent_mean, 2),
                    "agent_median_fee_sats": "" if agent_row is None else agent_row["agent_median_fee_sats"],
                    "agent_median_fee_rate_sat_vb": ""
                    if agent_row is None
                    else agent_row["agent_median_fee_rate_sat_vb"],
                    "wallet_minus_agent_mean": ""
                    if wallet_score is None or agent_mean is None
                    else round(float(wallet_score) - agent_mean, 2),
                }
            )
    return comparison_path, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run wallet coin-selection baselines.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(SCRIPT_DIR / "wallet_baseline.csv"),
        help="Wallet baseline definition CSV",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected experiments without running wallets")
    parser.add_argument("--preflight", action="store_true", help="Check Electrum dependencies and exit")
    parser.add_argument("--filter", help="Filter rows, e.g. id:wallet_electrum_pct10 or adapter:import")
    parser.add_argument("--include-disabled", action="store_true", help="Allow explicitly selected disabled rows")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULT_ROOT), help="Directory where result folders are created")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Wallet command timeout")
    parser.add_argument("--xpub", help="Override xpub for same-wallet recipient derivation")
    parser.add_argument("--electrum-wallet", help="Override Electrum watch-only wallet path")
    parser.add_argument("--electrum-cmd", help="Override Electrum command")
    parser.add_argument("--wasabi-prepare", action="store_true", help="Create/update safe Wasabi watch-only wallet JSON")
    parser.add_argument("--wasabi-generate-skeleton", action="store_true", help="Generate a public-only Wasabi/Coldcard skeleton JSON")
    parser.add_argument(
        "--wasabi-master-fingerprint",
        default=DEFAULT_WASABI_MASTER_FINGERPRINT_PLACEHOLDER,
        help="8-hex master fingerprint for the generated public-only skeleton; default is an explicit unknown placeholder",
    )
    parser.add_argument("--wasabi-rpc-check", action="store_true", help="Check Wasabi daemon RPC and controlled UTXOs")
    parser.add_argument("--wasabi-start-daemon", action="store_true", help="Start wassabeed in the controlled datadir before RPC check")
    parser.add_argument("--wasabi-cmd", help="Override Wasabi daemon command path for status output")
    parser.add_argument("--wasabi-datadir", help="Override Wasabi HOME-style data directory")
    parser.add_argument("--wasabi-wallet", help="Override Wasabi watch-only wallet name")
    parser.add_argument("--wasabi-rpc-url", help="Override Wasabi JSON-RPC URL")
    parser.add_argument("--sparrow-preflight", action="store_true", help="Classify local Sparrow CLI support")
    parser.add_argument("--sparrow-cmd", help="Override Sparrow command")
    parser.add_argument("--recipient-index", type=int, help="Override same-wallet receive index")
    parser.add_argument("--compare-agents", action="store_true", help="Write amount-level wallet vs agent comparison")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_project_env()

    parser = WalletBaselineCSVParser(Path(args.csv_path))
    experiments = parser.parse()
    experiments = apply_overrides(
        experiments,
        xpub=args.xpub,
        electrum_wallet=args.electrum_wallet,
        electrum_cmd=args.electrum_cmd,
        wasabi_cmd=args.wasabi_cmd,
        wasabi_datadir=args.wasabi_datadir,
        wasabi_wallet=args.wasabi_wallet,
        wasabi_rpc_url=args.wasabi_rpc_url,
        sparrow_cmd=args.sparrow_cmd,
        recipient_index=args.recipient_index,
    )
    selected = selected_experiments(experiments, args.filter, args.include_disabled)

    if args.dry_run:
        print_dry_run(selected)
        return 0
    if args.preflight:
        return 0 if run_preflight(selected) else 1
    if args.sparrow_preflight:
        exp = first_sparrow_experiment(selected, experiments)
        command = exp.sparrow_cmd if exp else args.sparrow_cmd
        status = classify_sparrow_cli(command, timeout_seconds=args.timeout_seconds)
        print(json.dumps(status, indent=2))
        return 1 if status.get("status") == "unavailable" else 0
    if args.wasabi_generate_skeleton:
        exp = first_wasabi_experiment(selected, experiments)
        if exp is None:
            print("No Wasabi wallet baseline row found.")
            return 1
        try:
            status = generate_wasabi_public_only_skeleton(
                exp,
                master_fingerprint=args.wasabi_master_fingerprint,
            )
        except Exception as exc:
            print(f"Wasabi skeleton generation failed: {exc}")
            return 1
        print(json.dumps(status, indent=2))
        return 0
    if args.wasabi_prepare or args.wasabi_start_daemon:
        exp = first_wasabi_experiment(selected, experiments)
        if exp is None:
            print("No Wasabi wallet baseline row found.")
            return 1
        try:
            status = prepare_wasabi_watchonly(
                exp,
                rpc_check=args.wasabi_rpc_check,
                start_daemon=args.wasabi_start_daemon,
                timeout_seconds=args.timeout_seconds,
            )
        except Exception as exc:
            print(f"Wasabi preparation failed: {exc}")
            return 1
        print(json.dumps(status, indent=2))
        return 0
    if not selected:
        print("No wallet baseline experiments selected.")
        return 1

    runner = WalletBaselineRunner(
        output_root=Path(args.output_dir),
        timeout_seconds=args.timeout_seconds,
    )
    results = runner.run(selected)
    csv_path, json_path = runner.write_results()
    print(f"Results CSV: {csv_path}")
    print(f"Results JSON: {json_path}")

    if args.compare_agents:
        comparison_path, error = write_agent_comparison(results, runner.output_dir)
        if comparison_path:
            print(f"Agent comparison CSV: {comparison_path}")
        else:
            print(f"Agent comparison not written: {error}")

    failures = sum(1 for result in results if not result.success)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
