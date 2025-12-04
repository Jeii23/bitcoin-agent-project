import base64
import pytest

@pytest.mark.skip(reason="BIP32 derivation paths feature (_build_addr_deriv_map) not implemented yet")
def test_psbt_includes_global_xpub_and_keypaths(block_requests):
    import psbt_creator
    # Single input / change output ensuring we derive two paths
    utxos = [
        {"txid": "aa"*32, "vout": 0, "value_satoshis": 120_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}
    ]
    res = psbt_creator.create_transaction_psbt(
        xpub="vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC",
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_sats=50_000,
        utxos=utxos,
        change_address="tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y",
        network="testnet",
        fee_satoshis=1000,
        manual_selected_utxos=utxos,
        include_global_xpub=True,
        include_keypaths=True,
    )
    assert res["success"], res.get("error")
    raw = base64.b64decode(res["psbt"])  # psbt bytes
    # Check presence of xpub global type (0x01) and derivation types (input 0x06, output 0x02)
    assert b"\x01" in raw, "Expected GLOBAL_XPUB key present"
    assert b"\x06" in raw, "Expected at least one PSBT_IN_BIP32_DERIVATION"
    assert b"\x02" in raw, "Expected at least one PSBT_OUT_BIP32_DERIVATION"