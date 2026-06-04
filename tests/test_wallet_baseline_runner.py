import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SRC_DIR = PROJECT_ROOT / "src"
for path in (EXPERIMENTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from psbt_creator import create_transaction_psbt
import wallet_baseline_runner as wbr
from wallet_baseline_runner import (
    BaselineError,
    WalletBaselineCSVParser,
    WalletBaselineExperiment,
    WalletBaselineRunner,
    _read_psbt_map,
    assert_watch_only_wallet_file,
    build_wasabi_public_only_skeleton_json,
    build_wasabi_watchonly_wallet_json,
    classify_sparrow_cli,
    generate_wasabi_public_only_skeleton,
    check_electrum_preflight,
    prepare_wasabi_watchonly,
    read_psbt_file,
    result_to_csv_row,
    target_sats_for_pct,
    validate_wasabi_rpc_psbt_result,
    wasabi_rpc_payload,
)


INPUT_ADDR = "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"
RECIPIENT = "tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx"
CHANGE = "tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"


def make_fixture_psbt() -> str:
    result = create_transaction_psbt(
        xpub="",
        utxos=[],
        manual_selected_utxos=[
            {
                "txid": "11" * 32,
                "vout": 0,
                "value_satoshis": 100_000,
                "address": INPUT_ADDR,
            }
        ],
        recipient_address=RECIPIENT,
        amount_sats=50_000,
        change_address=CHANGE,
        fee_satoshis=500,
        network="testnet",
    )
    assert result["success"], result.get("error")
    return result["psbt"]


def test_wallet_baseline_csv_parser_computes_targets(tmp_path):
    csv_path = tmp_path / "wallet_baseline.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "wallet",
                "adapter",
                "amount_pct",
                "network",
                "recipient_policy",
                "fee_policy",
                "psbt_file",
                "tags",
                "enabled",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "wallet_import_pct30",
                "wallet": "sparrow",
                "adapter": "import",
                "amount_pct": "30",
                "network": "mainnet",
                "recipient_policy": "same-wallet-fresh",
                "fee_policy": "wallet-native",
                "psbt_file": "fixtures/sparrow.psbt",
                "tags": "wallet-baseline|sparrow|amt-pct-30",
                "enabled": "true",
            }
        )

    parsed = WalletBaselineCSVParser(csv_path).parse()
    assert len(parsed) == 1
    assert parsed[0].amount_pct == "30"
    assert parsed[0].target_sats == 30_000_000
    assert parsed[0].tags == ["wallet-baseline", "sparrow", "amt-pct-30"]
    assert target_sats_for_pct("95") == 95_000_000


def test_import_adapter_scores_psbt_fixture_and_normalizes_csv(tmp_path):
    psbt_path = tmp_path / "fixture.base64"
    psbt_path.write_text(make_fixture_psbt(), encoding="ascii")
    exp = WalletBaselineExperiment(
        id="wallet_import_fixture_pct50",
        wallet="fixture-wallet",
        adapter="import",
        amount_pct="50",
        network="testnet",
        recipient_policy="same-wallet-fresh",
        fee_policy="wallet-native",
        psbt_file=str(psbt_path),
        tags=["wallet-baseline", "fixture", "amt-pct-50"],
        target_sats=50_000_000,
    )

    runner = WalletBaselineRunner(output_root=tmp_path, run_timestamp="test")
    result = runner.run_one(exp)

    assert result.success
    assert result.psbt_generated
    assert result.privacy_score is not None
    assert result.fee_sanity_ok == 1
    assert result.num_inputs == 1
    assert result.num_outputs == 2

    row = result_to_csv_row(result)
    assert row["experiment_id"] == "wallet_import_fixture_pct50"
    assert row["tags"] == "wallet-baseline|fixture|amt-pct-50"
    assert row["privacy_grade"]


def test_import_rejects_non_bip174_payload(tmp_path):
    bad_path = tmp_path / "finalized.hex"
    bad_path.write_text("02000000000100", encoding="ascii")

    with pytest.raises(BaselineError, match="BIP-174"):
        read_psbt_file(bad_path)


def test_import_rejects_signed_or_finalized_psbt(tmp_path):
    import base64

    psbt_bytes = base64.b64decode(make_fixture_psbt())
    _global_entries, input_offset = _read_psbt_map(psbt_bytes, len(b"psbt\xff"))
    signed_like = psbt_bytes[:input_offset] + b"\x01\x02\x00" + psbt_bytes[input_offset:]
    signed_path = tmp_path / "signed.psbt"
    signed_path.write_bytes(signed_like)

    with pytest.raises(BaselineError, match="signed/finalized"):
        read_psbt_file(signed_path)


def test_watch_only_wallet_check_allows_null_xprv(tmp_path):
    wallet_path = tmp_path / "watch_only"
    wallet_path.write_text(
        '{"wallet_type":"standard","keystore":{"xpub":"zpub-public","xprv":null}}',
        encoding="utf-8",
    )

    assert_watch_only_wallet_file(str(wallet_path))


def test_watch_only_wallet_check_allows_electrum_append_log_after_json(tmp_path):
    wallet_path = tmp_path / "watch_only_with_append_log"
    wallet_path.write_text(
        '{"wallet_type":"standard","keystore":{"xpub":"zpub-public","xprv":null}}\n'
        '{"op":"add","path":"/addresses/receiving/0","value":"bc1qpublic"}\n',
        encoding="utf-8",
    )

    assert_watch_only_wallet_file(str(wallet_path))


def test_watch_only_wallet_check_rejects_real_private_material(tmp_path):
    wallet_path = tmp_path / "private_wallet"
    wallet_path.write_text(
        '{"wallet_type":"standard","keystore":{"xpub":"zpub-public","xprv":"xprv-secret"}}',
        encoding="utf-8",
    )

    with pytest.raises(BaselineError, match="private-key"):
        assert_watch_only_wallet_file(str(wallet_path))


def test_dry_run_cli_lists_default_wallet_baseline_cases():
    completed = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS_DIR / "wallet_baseline_runner.py"),
            str(EXPERIMENTS_DIR / "wallet_baseline.csv"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wallet baseline dry-run: 10 experiment(s)" in completed.stdout
    assert "wallet_electrum_pct10" in completed.stdout
    assert "wallet_bitcoincore_pct10" in completed.stdout
    assert "target_sats=95000000" in completed.stdout


def test_electrum_preflight_reports_missing_aiohttp_socks(tmp_path):
    fake = tmp_path / "fake_electrum.py"
    fake.write_text(
        "import sys\n"
        "print(\"ModuleNotFoundError: No module named 'aiohttp_socks'\", file=sys.stderr)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    ok, message = check_electrum_preflight(f"{sys.executable} {fake}")
    assert not ok
    assert "aiohttp_socks" in message


def test_wasabi_watchonly_wallet_json_contains_no_signing_material(monkeypatch, tmp_path):
    monkeypatch.setattr(wbr, "_normalize_to_x_or_t_pub", lambda value: "xpub-normalized")
    exp = WalletBaselineExperiment(
        id="wallet_wasabi_pct10",
        wallet="wasabi",
        adapter="import",
        amount_pct="10",
        network="mainnet",
        recipient_policy="same-wallet-fresh",
        fee_policy="wallet-native",
        psbt_file="wallet_imports/wasabi/wasabi_pct10.psbt",
        tags=["wallet-baseline", "wasabi"],
        xpub="zpub-public",
        wasabi_datadir=str(tmp_path),
        target_sats=10_000_000,
    )

    wallet_json = build_wasabi_watchonly_wallet_json(exp)
    serialized = str(wallet_json).lower()
    assert wallet_json["EncryptedSecret"] is None
    assert wallet_json["ExtPubKey"] == "xpub-normalized"
    assert wallet_json["AccountKeyPath"] == "84'/0'/0'"
    assert wallet_json["TaprootAccountKeyPath"] == "86'/0'/0'"
    assert wallet_json["PasswordVerified"] is True
    assert wallet_json["HdPubKeys"] == []
    assert wallet_json["MasterFingerprint"] is None
    assert wallet_json["MinGapLimit"] >= 100
    assert wallet_json["BlockchainState"]["Height"] == "0"
    assert "xprv" not in serialized
    assert "mnemonic" not in serialized
    assert "seed" not in serialized
    assert '"password":' not in json.dumps(wallet_json).lower()


def test_wasabi_public_only_skeleton_uses_only_public_material(monkeypatch, tmp_path):
    monkeypatch.setattr(wbr, "_normalize_to_x_or_t_pub", lambda value: "xpub-normalized")
    exp = WalletBaselineExperiment(
        id="wallet_wasabi_pct10",
        wallet="wasabi",
        adapter="import",
        amount_pct="10",
        network="mainnet",
        recipient_policy="same-wallet-fresh",
        fee_policy="wallet-native",
        psbt_file="wallet_imports/wasabi/wasabi_pct10.psbt",
        tags=["wallet-baseline", "wasabi"],
        xpub="zpub-public",
        wasabi_datadir=str(tmp_path),
        target_sats=10_000_000,
    )

    skeleton = build_wasabi_public_only_skeleton_json(exp)
    assert skeleton["EncryptedSecret"] is None
    assert skeleton["ExtPubKey"] == "xpub-normalized"
    assert skeleton["AccountKeyPath"] == "84'/0'/0'"
    assert skeleton["TaprootAccountKeyPath"] == "86'/0'/0'"
    assert skeleton["PasswordVerified"] is True
    assert skeleton["HdPubKeys"] == []
    assert skeleton["MasterFingerprint"] == "00000000"
    forbidden_values = {"xprv", "yprv", "zprv", "tprv", "seed", "mnemonic"}
    serialized = str(skeleton).lower()
    assert not any(marker in serialized for marker in forbidden_values)
    assert '"password":' not in json.dumps(skeleton).lower()


def test_wasabi_generate_public_only_skeleton_writes_status(monkeypatch, tmp_path):
    monkeypatch.setattr(wbr, "_normalize_to_x_or_t_pub", lambda value: "xpub-normalized")
    exp = WalletBaselineExperiment(
        id="wallet_wasabi_pct10",
        wallet="wasabi",
        adapter="import",
        amount_pct="10",
        network="mainnet",
        recipient_policy="same-wallet-fresh",
        fee_policy="wallet-native",
        psbt_file="wallet_imports/wasabi/wasabi_pct10.psbt",
        tags=["wallet-baseline", "wasabi"],
        xpub="zpub-public",
        wasabi_datadir=str(tmp_path),
        target_sats=10_000_000,
    )

    status = generate_wasabi_public_only_skeleton(exp, status_path=tmp_path / "status.json")
    skeleton_path = Path(status["skeleton_file"])
    skeleton = skeleton_path.read_text(encoding="utf-8")

    assert skeleton_path.exists()
    assert status["skeleton_source"] == "generated-public-only"
    assert status["master_fingerprint_source"] == "unknown-placeholder"
    assert status["psbt_generation_route"] == "wasabi-gui-psbt-workflow-export"
    assert "xpub-normalized" in skeleton


def test_wasabi_rpc_payload_does_not_allow_wallet_passwords():
    payload = wasabi_rpc_payload("loadwallet", ["wallet_baseline_wasabi_watchonly"])
    assert payload["method"] == "loadwallet"
    assert "password" not in str(payload).lower()

    with pytest.raises(BaselineError, match="sensitive"):
        wasabi_rpc_payload("loadwallet", {"walletName": "x", "password": "nope"})

    with pytest.raises(BaselineError, match="Refusing Wasabi RPC method"):
        wasabi_rpc_payload("build", {"payments": []})


def test_wasabi_prepare_rpc_preflight_with_mock(monkeypatch, tmp_path):
    monkeypatch.setattr(wbr, "_normalize_to_x_or_t_pub", lambda value: "xpub-normalized")
    calls = []

    def fake_rpc(exp, method, params=None, wallet=None, timeout_seconds=20):
        calls.append((method, params, wallet))
        if method == "listwallets":
            return []
        if method == "loadwallet":
            return {"walletName": params[0]}
        if method == "getwalletinfo":
            return {"walletName": wallet, "watchOnly": True}
        if method == "listunspentcoins":
            return [{"amount": 2_000_000} for _ in range(50)]
        if method == "listkeys":
            return [{"keyState": "clean"}]
        raise AssertionError(method)

    exp = WalletBaselineExperiment(
        id="wallet_wasabi_pct10",
        wallet="wasabi",
        adapter="import",
        amount_pct="10",
        network="mainnet",
        recipient_policy="same-wallet-fresh",
        fee_policy="wallet-native",
        psbt_file="wallet_imports/wasabi/wasabi_pct10.psbt",
        tags=["wallet-baseline", "wasabi"],
        xpub="zpub-public",
        wasabi_datadir=str(tmp_path),
        wasabi_wallet="wallet_baseline_wasabi_watchonly",
        target_sats=10_000_000,
    )

    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "skeleton_source": "generated-public-only",
                "skeleton_file": str(tmp_path / "coldcard_public_only_skeleton.json"),
                "psbt_generation_route": "wasabi-gui-psbt-workflow-export",
            }
        ),
        encoding="utf-8",
    )

    status = prepare_wasabi_watchonly(
        exp,
        rpc_check=True,
        status_path=status_path,
        rpc_call=fake_rpc,
    )

    assert status["watch_only_prepared"]
    assert status["rpc_ok"]
    assert status["prepared_via_rpc"]
    assert status["coin_count"] == 50
    assert status["coin_total_sats"] == 100_000_000
    assert status["wasabi_rpc_psbt_generation_supported"] is False
    assert status["skeleton_source"] == "generated-public-only"
    assert status["psbt_generation_route"] == "wasabi-gui-psbt-workflow-export"
    assert ("build", {"payments": []}, None) not in calls


def test_wasabi_rejects_rpc_build_hex_result():
    with pytest.raises(BaselineError, match="BIP-174"):
        validate_wasabi_rpc_psbt_result("02000000000100")


def test_sparrow_cli_is_manual_import_only_when_no_builder_command(tmp_path):
    fake = tmp_path / "fake_sparrow.py"
    fake.write_text(
        "print('Usage: sparrow [options]')\n"
        "print('--dir')\n"
        "print('--network')\n"
        "print('--terminal')\n",
        encoding="utf-8",
    )

    status = classify_sparrow_cli(f"{sys.executable} {fake}")
    assert status["status"] == "manual-import-only"
    assert status["manual_import_only"] is True
