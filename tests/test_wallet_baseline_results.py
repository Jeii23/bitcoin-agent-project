import csv
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import paper_charts
from wallet_baseline_results import (
    build_wallet_agent_amount_comparison,
    build_wallet_coverage_matrix,
    build_wallet_reliability_summary,
    load_wallet_baseline_definitions,
    normalize_wallet_baseline_results,
    select_latest_successful_by_wallet_amount,
)


WALLET_COLUMNS = [
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


def write_wallet_lot(root: Path, run_id: str, rows):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    csv_path = run_dir / f"{run_id}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WALLET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_rows = []
    for row in rows:
        breakdown = None
        if row.get("privacy_score"):
            breakdown = {
                "scores": {
                    "overall": int(row["privacy_score"]),
                    "clustering": 82,
                    "change_detection": 76,
                    "fingerprinting": 91,
                    "metadata_leakage": 100,
                },
                "fee_sanity_ok": int(row.get("fee_sanity_ok") or 0),
                "sanity_status": row.get("sanity_status") or "",
                "fee_analysis": {
                    "fee_sats": int(row.get("fee_sats") or 0),
                    "fee_rate_sat_vb": float(row.get("fee_rate_sat_vb") or 0),
                },
                "metadata": {
                    "num_inputs": int(row.get("num_inputs") or 0),
                    "num_outputs": int(row.get("num_outputs") or 0),
                },
            }
        json_rows.append({"experiment_id": row["experiment_id"], "privacy_breakdown": breakdown})
    csv_path.with_suffix(".json").write_text(json.dumps(json_rows), encoding="utf-8")
    return csv_path


def wallet_row(**overrides):
    row = {
        "experiment_id": "wallet_electrum_pct10",
        "wallet": "electrum",
        "adapter": "electrum",
        "amount_pct": "10",
        "success": "1",
        "error_message": "",
        "psbt_generated": "1",
        "privacy_score": "73",
        "privacy_grade": "B",
        "fee_sanity_ok": "1",
        "sanity_status": "ok",
        "fee_rate_sat_vb": "10.4",
        "fee_sats": "2172",
        "num_inputs": "2",
        "num_outputs": "2",
        "psbt_file": "/tmp/example.psbt",
        "tags": "wallet-baseline|electrum|amt-pct-10",
        "target_sats": "10000000",
        "recipient_address": "bc1qexample",
        "timestamp": "2026-05-20 12:00:00",
    }
    row.update(overrides)
    return row


def test_wallet_baseline_normalization_keeps_failures_and_selects_latest_success(tmp_path):
    failed = wallet_row(success="0", error_message="preflight failed", psbt_generated="0", privacy_score="")
    success = wallet_row(privacy_score="80", fee_sats="1463", num_inputs="1")
    paths = [
        write_wallet_lot(tmp_path, "wallet_baseline_20260519_000000", [failed]),
        write_wallet_lot(tmp_path, "wallet_baseline_20260520_000000", [success]),
    ]

    df = normalize_wallet_baseline_results(paths)
    assert len(df) == 2

    latest = select_latest_successful_by_wallet_amount(df)
    assert len(latest) == 1
    assert latest.iloc[0]["privacy_score"] == 80
    assert latest.iloc[0]["run_id"] == "wallet_baseline_20260520_000000"

    reliability = build_wallet_reliability_summary(df)
    assert reliability.iloc[0]["attempts"] == 2
    assert reliability.iloc[0]["failed"] == 1
    assert reliability.iloc[0]["fee_ok"] == 1


def test_wallet_coverage_marks_pending_manual_import_rows(tmp_path):
    definition_csv = tmp_path / "wallet_baseline.csv"
    definition_csv.write_text(
        "id,wallet,adapter,amount_pct,network,recipient_policy,fee_policy,psbt_file,tags,enabled\n"
        "wallet_sparrow_pct10,sparrow,import,10,mainnet,same-wallet-fresh,wallet-native,"
        "wallet_imports/sparrow/sparrow_pct10.psbt,wallet-baseline|sparrow,false\n",
        encoding="utf-8",
    )
    definitions = load_wallet_baseline_definitions(definition_csv)
    coverage = build_wallet_coverage_matrix(pd.DataFrame(), definitions)

    sparrow_10 = coverage[(coverage["wallet"] == "sparrow") & (coverage["amount_pct"] == 10.0)].iloc[0]
    assert sparrow_10["coverage_status"] == "waiting-for-manual-psbt"
    assert len(coverage) == 20


def test_wallet_coverage_marks_wasabi_safe_rpc_status(tmp_path):
    definition_csv = tmp_path / "wallet_baseline.csv"
    definition_csv.write_text(
        "id,wallet,adapter,amount_pct,network,recipient_policy,fee_policy,psbt_file,tags,enabled\n"
        "wallet_wasabi_pct10,wasabi,import,10,mainnet,same-wallet-fresh,wallet-native,"
        "wallet_imports/wasabi/wasabi_pct10.psbt,wallet-baseline|wasabi,false\n",
        encoding="utf-8",
    )
    definitions = load_wallet_baseline_definitions(definition_csv)
    coverage = build_wallet_coverage_matrix(
        pd.DataFrame(),
        definitions,
        wasabi_status={
            "prepared_via_rpc": True,
            "rpc_ok": True,
            "utxo_check_ok": True,
            "safe_rpc_psbt_generation_supported": False,
        },
    )

    wasabi_10 = coverage[(coverage["wallet"] == "wasabi") & (coverage["amount_pct"] == 10.0)].iloc[0]
    assert wasabi_10["coverage_status"] == "prepared-via-rpc"
    assert "waiting for PSBT workflow export" in wasabi_10["status_label"]


def test_wallet_coverage_marks_wasabi_public_only_skeleton(tmp_path):
    definition_csv = tmp_path / "wallet_baseline.csv"
    definition_csv.write_text(
        "id,wallet,adapter,amount_pct,network,recipient_policy,fee_policy,psbt_file,tags,enabled\n"
        "wallet_wasabi_pct10,wasabi,import,10,mainnet,same-wallet-fresh,wallet-native,"
        "wallet_imports/wasabi/wasabi_pct10.psbt,wallet-baseline|wasabi,false\n",
        encoding="utf-8",
    )
    definitions = load_wallet_baseline_definitions(definition_csv)
    coverage = build_wallet_coverage_matrix(
        pd.DataFrame(),
        definitions,
        wasabi_status={
            "skeleton_source": "generated-public-only",
            "skeleton_file": "/tmp/coldcard_public_only_skeleton.json",
            "safe_rpc_psbt_generation_supported": False,
        },
    )

    wasabi_10 = coverage[(coverage["wallet"] == "wasabi") & (coverage["amount_pct"] == 10.0)].iloc[0]
    assert wasabi_10["coverage_status"] == "public-only-skeleton-generated"
    assert wasabi_10["wasabi_skeleton_source"] == "generated-public-only"


def test_wallet_agent_amount_comparison_uses_agent_means(tmp_path):
    path = write_wallet_lot(tmp_path, "wallet_baseline_20260520_000000", [wallet_row(privacy_score="73")])
    wallet_df = normalize_wallet_baseline_results([path])
    agent_df = pd.DataFrame(
        [
            {"amount_pct": 10.0, "fee_ok_bool": True, "usable_score": 80.0, "fee_sats": 500, "fee_rate_sat_vb": 3.0},
            {"amount_pct": 10.0, "fee_ok_bool": True, "usable_score": 84.0, "fee_sats": 600, "fee_rate_sat_vb": 4.0},
        ]
    )

    comparison = build_wallet_agent_amount_comparison(wallet_df, agent_df=agent_df)
    assert comparison.iloc[0]["agent_mean_privacy_score"] == 82.0
    assert comparison.iloc[0]["wallet_minus_agent_mean"] == -9.0


def test_wallet_charts_handle_partial_four_wallet_coverage(tmp_path, monkeypatch):
    path = write_wallet_lot(tmp_path, "wallet_baseline_20260520_000000", [wallet_row(privacy_score="73")])
    wallet_df = normalize_wallet_baseline_results([path])
    comparison = pd.DataFrame(
        [
            {
                "wallet_label": "Electrum",
                "amount_pct": 10.0,
                "privacy_score": 73.0,
                "agent_mean_privacy_score": 82.0,
                "wallet_minus_agent_mean": -9.0,
            }
        ]
    )
    monkeypatch.setattr(paper_charts, "build_wallet_agent_amount_comparison", lambda _df: comparison)

    for chart_name in paper_charts.WALLET_BASELINE_CHART_OPTIONS:
        chart = paper_charts.build_wallet_baseline_chart(chart_name, wallet_df)
        assert chart is not None, chart_name
