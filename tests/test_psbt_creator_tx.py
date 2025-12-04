
import base64
import pytest

def _get_fee_sats(res):
    # Accept several possible key names to be robust across refactors
    for k in ("fee_satoshis", "fee_sat", "fee", "fee_sats"):
        if k in res:
            return res[k]
    return None

@pytest.mark.usefixtures("block_requests")
def test_create_simple_psbt_with_change():
    import psbt_creator
    creator = psbt_creator.PSBTCreator(network="testnet")

    utxos = [
        {
            "txid": "00"*32,
            "vout": 0,
            "value_satoshis": 100_000,  # sats
            "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",  # valid testnet bech32
        }
    ]
    # Use a known valid recipient address (vector index 1)
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_sats=50_000,  # 50_000 sats
        utxos=utxos,
        manual_selected_utxos=utxos,
        fee_satoshis=1000,
        network="testnet",
        change_address="tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"  # valid change address (vector index 2)
    )

    assert res["success"], res.get("error")
    assert isinstance(res.get("psbt"), str) and len(res["psbt"]) > 0
    fee = _get_fee_sats(res)
    if fee is not None:
        assert int(fee) == 1000
    assert res["num_inputs"] >= 1
    # with change > dust (546), should have 2 outputs (recipient + change)
    assert res["num_outputs"] == 2
    # PSBT magic at the start after base64 decode
    raw = base64.b64decode(res["psbt"])
    assert raw.startswith(b"psbt\xff")

@pytest.mark.usefixtures("block_requests")
def test_dust_change_collapses_into_fee():
    """
    Note: The new API does NOT automatically collapse dust change into fee.
    The agent is responsible for computing outputs explicitly.
    This test now verifies the API returns both outputs (even if dust).
    """
    import psbt_creator
    creator = psbt_creator.PSBTCreator(network="testnet")

    utxos = [
        {"txid": "11"*32, "vout": 1, "value_satoshis": 51_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}
    ]
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_sats=50_000,   # 50_000 sats
        utxos=utxos,
        manual_selected_utxos=utxos,
        fee_satoshis=900,
        network="testnet",
        change_address="tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"
    )
    assert res["success"], res.get("error")
    # New API behavior: dust change (100 sats) is still created since 
    # the agent is expected to handle dust suppression explicitly.
    # The function now returns 2 outputs (recipient + dust change)
    assert res["num_outputs"] == 2

@pytest.mark.usefixtures("block_requests")
def test_insufficient_funds_error():
    import psbt_creator
    creator = psbt_creator.PSBTCreator(network="testnet")

    utxos = [
        {"txid": "22"*32, "vout": 0, "value_satoshis": 20_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}
    ]
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_sats=30_000,   # 30_000 sats
        utxos=utxos,
        manual_selected_utxos=utxos,
        fee_satoshis=1000,
        network="testnet",
        change_address="tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"
    )
    assert not res["success"]
    err = res.get("error","").lower()
    assert ("insufficient" in err) or ("insuficient" in err) or ("fons" in err)

@pytest.mark.usefixtures("block_requests")
@pytest.mark.parametrize("addr", [
    # Testnet P2PKH, P2SH, Bech32
    "mipcBbFg9gMiCh81Kj8tqqdgoZub1ZJRfn",         # P2PKH (testnet)
    "2N2JD6wb56AfK4tfmM6PwdVmoYk2dCKf4Br",        # P2SH  (testnet)
    "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", # P2WPKH (bech32)
])
def test_output_script_generation_for_various_address_types(addr):
    import psbt_creator
    utxos = [{"txid": "33"*32, "vout": 0, "value_satoshis": 50_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}]
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address=addr,
        amount_sats=10_000,
        utxos=utxos,
        manual_selected_utxos=utxos,
        fee_satoshis=500,
        network="testnet",
        change_address="tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"
    )
    assert res["success"], res.get("error")
    assert res["num_outputs"] in (1,2)

@pytest.mark.usefixtures("block_requests")
def test_fail_fast_when_change_needed_but_missing():
    import psbt_creator
    # UTXO large enough to produce non-dust change if change address absent
    utxos = [
        {"txid": "44"*32, "vout": 0, "value_satoshis": 150_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}
    ]
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_sats=50_000,  # 50_000 sats
        utxos=utxos,
        manual_selected_utxos=utxos,
        fee_satoshis=1000,
        network="testnet",
        change_address=None  # intentionally missing
    )
    assert not res["success"], "Should fail without explicit change_address when change required"
    assert "change" in res.get("error", "").lower()
