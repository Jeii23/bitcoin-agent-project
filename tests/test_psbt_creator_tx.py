
import base64
import pytest

def _get_fee_sats(res):
    # Accept several possible key names to be robust across refactors
    for k in ("fee_satoshis", "fee_sat", "fee"):
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
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qs8kgyvu7j5m8w0mxkawdt8a9c97m3x8c9sawct",
        amount_btc=0.00050000,  # 50_000 sats
        utxos=utxos,
        fee_satoshis=1000,
        network="testnet",
        change_address="tb1qnjqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3x0k8"  # dummy but bech32-like
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
    import psbt_creator
    creator = psbt_creator.PSBTCreator(network="testnet")

    utxos = [
        {"txid": "11"*32, "vout": 1, "value_satoshis": 51_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}
    ]
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qg2t9c2mamc2r9l68v9r80xkqz0r5yyptqetk6k",
        amount_btc=0.00050000,   # 50_000 sats
        utxos=utxos,
        fee_satoshis=900,
        network="testnet",
        change_address="tb1qtestchangeaddressxxxxxxxxxxxxxxxxxxxxxx0yk"
    )
    assert res["success"], res.get("error")
    # change would be 100 sats -> dust → expect 1 output only
    assert res["num_outputs"] == 1

@pytest.mark.usefixtures("block_requests")
def test_insufficient_funds_error():
    import psbt_creator
    creator = psbt_creator.PSBTCreator(network="testnet")

    utxos = [
        {"txid": "22"*32, "vout": 0, "value_satoshis": 20_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}
    ]
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qg2t9c2mamc2r9l68v9r80xkqz0r5yyptqetk6k",
        amount_btc=0.00030000,   # 30_000 sats
        utxos=utxos,
        fee_satoshis=1000,
        network="testnet",
        change_address="tb1qtestchangeaddressxxxxxxxxxxxxxxxxxxxxxx0yk"
    )
    assert not res["success"]
    err = res.get("error","").lower()
    assert ("insufficient" in err) or ("insuficient" in err)

@pytest.mark.usefixtures("block_requests")
@pytest.mark.parametrize("addr", [
    # Testnet P2PKH, P2SH, Bech32
    "mipcBbFg9gMiCh81Kj8tqqdgoZub1ZJRfn",         # P2PKH (testnet)
    "2N2JD6wb56AfK4tfmM6PwdVmoYk2dCKf4Br",        # P2SH  (testnet)
    "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", # P2WPKH (bech32)
])
def test_output_script_generation_for_various_address_types(addr):
    import psbt_creator
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address=addr,
        amount_btc=0.0001,
        utxos=[{"txid": "33"*32, "vout": 0, "value_satoshis": 50_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}],
        fee_satoshis=500,
        network="testnet",
        change_address="tb1qtestchangeaddressxxxxxxxxxxxxxxxxxxxxxx0yk"
    )
    assert res["success"], res.get("error")
    assert res["num_outputs"] in (1,2)
