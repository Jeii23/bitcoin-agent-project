#!/usr/bin/env python3
"""Normalize and compare real-wallet baseline result lots.

Wallet baselines intentionally live outside the LLM experiment corpus.  This
module loads those separate result folders, keeps failed attempts for
reliability analysis, and selects the latest successful row per wallet/amount
for score and structure comparisons.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from phase12_results import AMOUNT_ORDER, flatten_privacy_breakdown, normalize_phase12_results


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_DEFINITION_CSV = SCRIPT_DIR / "wallet_baseline.csv"
DEFAULT_WASABI_STATUS_PATH = SCRIPT_DIR / "wallets" / "wasabi_baseline" / "wasabi_preflight_status.json"

WALLET_ORDER = [
    "electrum",
    "bitcoin-core",
    "sparrow-efficiency",
    "sparrow-privacy",
    "wasabi",
]

WALLET_LABELS = {
    "electrum": "Electrum",
    "bitcoin-core": "Bitcoin Core",
    "bitcoin_core": "Bitcoin Core",
    "core": "Bitcoin Core",
    "sparrow": "Sparrow",
    "sparrow-efficiency": "Sparrow Efficiency",
    "sparrow-privacy": "Sparrow Privacy",
    "wasabi": "Wasabi",
}

EXPECTED_AMOUNT_PCTS = [10.0, 30.0, 50.0, 80.0, 95.0]
EXPECTED_AMOUNT_LABELS = {value: f"{value:g}%" for value in EXPECTED_AMOUNT_PCTS}

BASE_COLUMNS = [
    "experiment_id",
    "wallet",
    "wallet_label",
    "adapter",
    "amount_pct",
    "amount_label",
    "success",
    "success_bool",
    "error_message",
    "psbt_generated",
    "psbt_generated_bool",
    "privacy_score",
    "usable_score",
    "privacy_grade",
    "fee_sanity_ok",
    "fee_ok_bool",
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
    "run_id",
    "source_csv",
    "source_rank",
]


def canonical_wallet(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "bitcoin-core": "bitcoin-core",
        "bitcoincore": "bitcoin-core",
        "core": "bitcoin-core",
    }
    return aliases.get(text, text)


def wallet_label(value: object) -> str:
    canonical = canonical_wallet(value)
    return WALLET_LABELS.get(canonical, canonical.replace("-", " ").title())


def _to_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y", "ok"}:
        return True
    if text in {"false", "0", "0.0", "no", "n"}:
        return False
    if text == "":
        return None
    return None


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _wallet_sort_key(value: object) -> int:
    wallet = canonical_wallet(value)
    try:
        return WALLET_ORDER.index(wallet)
    except ValueError:
        return len(WALLET_ORDER)


def _amount_sort_key(value: object) -> int:
    amount = _to_float(value)
    if amount in EXPECTED_AMOUNT_PCTS:
        return EXPECTED_AMOUNT_PCTS.index(amount)
    if amount is None:
        return len(EXPECTED_AMOUNT_PCTS)
    return len(EXPECTED_AMOUNT_PCTS) + int(amount)


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=BASE_COLUMNS)


def wallet_baseline_csv_files(results_root: Path = DEFAULT_RESULTS_ROOT) -> List[Path]:
    root = Path(results_root)
    if not root.exists():
        return []
    return sorted(root.glob("wallet_baseline_*/wallet_baseline_*.csv"))


def _json_rows_for_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                rows[str(row.get("experiment_id", ""))] = row
    return rows


def _read_csv_rows(csv_path: Path, source_rank: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    json_index = _json_rows_for_csv(csv_path)
    flat_rows: List[Dict[str, Any]] = []

    run_id = csv_path.parent.name
    for row_index, row in df.iterrows():
        experiment_id = str(row.get("experiment_id", ""))
        json_row = json_index.get(experiment_id, {})
        flat = flatten_privacy_breakdown(json_row.get("privacy_breakdown") if json_row else None)
        flat_rows.append(
            {
                **flat,
                "run_id": run_id,
                "source_csv": str(csv_path),
                "source_rank": source_rank,
                "row_rank": row_index,
            }
        )

    extras = pd.DataFrame(flat_rows)
    merged = df.reset_index(drop=True).copy()
    for column in extras.columns:
        merged[column] = extras[column].reset_index(drop=True)
    return merged


def normalize_wallet_baseline_results(
    csv_paths: Optional[Iterable[Path]] = None,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> pd.DataFrame:
    paths = [Path(path) for path in csv_paths] if csv_paths is not None else wallet_baseline_csv_files(results_root)
    frames = [
        _read_csv_rows(path, source_rank=index)
        for index, path in enumerate(paths)
        if Path(path).exists()
    ]
    if not frames:
        return _empty_results()

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["wallet"] = df.get("wallet", "").map(canonical_wallet)
    df["wallet_label"] = df["wallet"].map(wallet_label)
    df["amount_pct"] = pd.to_numeric(df.get("amount_pct"), errors="coerce")
    df["amount_label"] = df["amount_pct"].map(lambda value: f"{value:g}%" if pd.notna(value) else "")
    df["success_bool"] = df.get("success", "").map(_to_bool).fillna(False)
    df["psbt_generated_bool"] = df.get("psbt_generated", "").map(_to_bool).fillna(False)
    df["fee_ok_bool"] = df.get("fee_sanity_ok", "").map(_to_bool).fillna(False)
    df["usable_score"] = df["privacy_score"].where(df["success_bool"] & df["fee_ok_bool"])

    numeric_columns = [
        "privacy_score",
        "usable_score",
        "fee_sanity_ok",
        "fee_rate_sat_vb",
        "fee_sats",
        "num_inputs",
        "num_outputs",
        "target_sats",
        "score_overall",
        "score_clustering",
        "score_change_detection",
        "score_fingerprinting",
        "score_metadata_leakage",
        "confidence_numeric",
        "total_input_sats",
        "total_output_sats",
        "change_probability",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["wallet_sort"] = df["wallet"].map(_wallet_sort_key)
    df["amount_sort"] = df["amount_pct"].map(_amount_sort_key)
    df = df.sort_values(["source_rank", "row_rank"], kind="stable").reset_index(drop=True)
    return df


def load_wallet_baseline_definitions(csv_path: Path = DEFAULT_DEFINITION_CSV) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["wallet"] = df.get("wallet", "").map(canonical_wallet)
    df["wallet_label"] = df["wallet"].map(wallet_label)
    df["amount_pct"] = pd.to_numeric(df.get("amount_pct"), errors="coerce")
    df["amount_label"] = df["amount_pct"].map(lambda value: f"{value:g}%" if pd.notna(value) else "")
    df["enabled_bool"] = df.get("enabled", "").map(_to_bool).fillna(True)
    return df


def load_wasabi_preflight_status(status_path: Path = DEFAULT_WASABI_STATUS_PATH) -> Dict[str, Any]:
    path = Path(status_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def select_latest_successful_by_wallet_amount(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_results()
    candidates = df[df["success_bool"] & df["psbt_generated_bool"]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=df.columns)
    candidates = candidates.sort_values(["source_rank", "row_rank"], kind="stable")
    latest = candidates.groupby(["wallet", "amount_pct"], dropna=False).tail(1)
    return latest.sort_values(["wallet_sort", "amount_sort"], kind="stable").reset_index(drop=True)


def build_wallet_coverage_matrix(
    results_df: pd.DataFrame,
    definitions_df: Optional[pd.DataFrame] = None,
    *,
    wallets: Sequence[str] = WALLET_ORDER,
    amounts: Sequence[float] = EXPECTED_AMOUNT_PCTS,
    wasabi_status: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    definitions = definitions_df if definitions_df is not None else load_wallet_baseline_definitions()
    wasabi_status = wasabi_status if wasabi_status is not None else load_wasabi_preflight_status()
    latest = select_latest_successful_by_wallet_amount(results_df)
    latest_by_key = {
        (row["wallet"], float(row["amount_pct"])): row
        for _, row in latest.iterrows()
        if pd.notna(row.get("amount_pct"))
    }
    def_by_key: Dict[Tuple[str, float], pd.Series] = {}
    if definitions is not None and not definitions.empty:
        for _, row in definitions.iterrows():
            if pd.notna(row.get("amount_pct")):
                def_by_key[(row["wallet"], float(row["amount_pct"]))] = row

    rows: List[Dict[str, Any]] = []
    for wallet in wallets:
        canonical = canonical_wallet(wallet)
        for amount in amounts:
            key = (canonical, float(amount))
            latest_row = latest_by_key.get(key)
            def_row = def_by_key.get(key)
            row: Dict[str, Any] = {
                "wallet": canonical,
                "wallet_label": wallet_label(canonical),
                "amount_pct": float(amount),
                "amount_label": EXPECTED_AMOUNT_LABELS.get(float(amount), f"{amount:g}%"),
                "expected": True,
                "privacy_score": pd.NA,
                "fee_ok_bool": False,
                "psbt_generated_bool": False,
                "success_bool": False,
                "coverage_status": "missing-result",
                "status_label": "Missing result",
            }
            if def_row is not None:
                row["adapter"] = def_row.get("adapter", "")
                row["enabled_bool"] = bool(def_row.get("enabled_bool", True))
                row["psbt_file"] = def_row.get("psbt_file", "")
                psbt_file = str(def_row.get("psbt_file", "") or "").strip()
                psbt_path = Path(psbt_file)
                if psbt_file and not psbt_path.is_absolute():
                    psbt_path = SCRIPT_DIR / psbt_path
                if def_row.get("adapter") == "import" and (not psbt_file or not psbt_path.exists()):
                    row["coverage_status"] = "waiting-for-manual-psbt"
                    row["status_label"] = "Waiting for manual PSBT"
                    if canonical == "wasabi" and wasabi_status:
                        row["wasabi_rpc_ok"] = bool(wasabi_status.get("rpc_ok"))
                        row["wasabi_utxo_check_ok"] = bool(wasabi_status.get("utxo_check_ok"))
                        row["wasabi_status_file"] = str(DEFAULT_WASABI_STATUS_PATH)
                        row["wasabi_skeleton_source"] = wasabi_status.get("skeleton_source", "")
                        row["wasabi_skeleton_file"] = wasabi_status.get("skeleton_file", "")
                        if wasabi_status.get("prepared_via_rpc"):
                            row["coverage_status"] = "prepared-via-rpc"
                            row["status_label"] = "Prepared via RPC; waiting for PSBT workflow export"
                        elif wasabi_status.get("skeleton_source") == "generated-public-only":
                            row["coverage_status"] = "public-only-skeleton-generated"
                            row["status_label"] = "Public-only skeleton generated; waiting for Wasabi PSBT export"
                        elif wasabi_status.get("safe_rpc_psbt_generation_supported") is False:
                            row["coverage_status"] = "unsupported-by-safe-rpc"
                            row["status_label"] = "Unsupported by safe RPC; waiting for PSBT workflow export"
                elif not bool(def_row.get("enabled_bool", True)):
                    row["coverage_status"] = "defined-disabled"
                    row["status_label"] = "Defined, disabled"
            else:
                row["coverage_status"] = "not-defined"
                row["status_label"] = "Not defined"
            if latest_row is not None:
                for column in latest_row.index:
                    row[column] = latest_row[column]
                row["coverage_status"] = "ok" if bool(latest_row.get("fee_ok_bool", False)) else "fee-bad"
                row["status_label"] = "Fee-sane PSBT" if row["coverage_status"] == "ok" else "PSBT, fee-bad"
                if (
                    canonical == "wasabi"
                    and row["coverage_status"] == "ok"
                    and wasabi_status.get("skeleton_source") == "generated-public-only"
                ):
                    row["status_label"] = "Scored from public-only skeleton"
            rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage["wallet_sort"] = coverage["wallet"].map(_wallet_sort_key)
    coverage["amount_sort"] = coverage["amount_pct"].map(_amount_sort_key)
    return coverage.sort_values(["wallet_sort", "amount_sort"], kind="stable").reset_index(drop=True)


def build_wallet_reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["wallet", "wallet_label"], dropna=False).agg(
        attempts=("experiment_id", "count"),
        psbt_generated=("psbt_generated_bool", "sum"),
        fee_ok=("fee_ok_bool", "sum"),
        failed=("success_bool", lambda values: int((~values.astype(bool)).sum())),
    ).reset_index()
    grouped["fee_bad"] = grouped["psbt_generated"] - grouped["fee_ok"]
    grouped["fee_ok_rate"] = grouped["fee_ok"] / grouped["attempts"]
    grouped["failed_rate"] = grouped["failed"] / grouped["attempts"]
    grouped["wallet_sort"] = grouped["wallet"].map(_wallet_sort_key)
    return grouped.sort_values("wallet_sort").reset_index(drop=True)


def build_wallet_agent_amount_comparison(
    wallet_df: pd.DataFrame,
    *,
    agent_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    latest = select_latest_successful_by_wallet_amount(wallet_df)
    if latest.empty:
        return pd.DataFrame()
    if agent_df is None:
        agent_df = normalize_phase12_results()
    if agent_df.empty:
        return pd.DataFrame()

    agents = agent_df[(agent_df["fee_ok_bool"]) & (agent_df["amount_pct"].notna())].copy()
    if agents.empty:
        return pd.DataFrame()
    agent_summary = agents.groupby("amount_pct", dropna=False).agg(
        agent_fee_sane_count=("usable_score", "count"),
        agent_mean_privacy_score=("usable_score", "mean"),
        agent_median_fee_sats=("fee_sats", "median"),
        agent_median_fee_rate_sat_vb=("fee_rate_sat_vb", "median"),
    ).reset_index()
    merged = latest.merge(agent_summary, on="amount_pct", how="left")
    merged["wallet_minus_agent_mean"] = merged["privacy_score"] - merged["agent_mean_privacy_score"]
    return merged.sort_values(["wallet_sort", "amount_sort"], kind="stable").reset_index(drop=True)
