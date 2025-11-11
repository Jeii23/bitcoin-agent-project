import pytest

@pytest.mark.usefixtures("block_requests")
@pytest.mark.parametrize("bad_addr", [
    # Bech32 with wrong checksum (alter last char)
    "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4x",  # last char changed
    # Bech32 mixed case invalid
    "Tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
    # Base58 invalid char (0 is not in alphabet)
    "mipcBbFg9gMiCh81Kj8tqqdgoZub1ZJRf0",
])
def test_invalid_recipient_address_rejected(bad_addr):
    import psbt_creator
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address=bad_addr,
        amount_sats=10_000,
        utxos=[{"txid": "44"*32, "vout": 0, "value_satoshis": 60_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}],
        fee_satoshis=1000,
        network="testnet",
        change_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"
    )
    assert not res["success"], f"Should fail for address {bad_addr}"
    assert "invalid" in res.get("error"," ").lower() or "fail" in res.get("error"," ").lower()

@pytest.mark.usefixtures("block_requests")
def test_invalid_change_address_rejected():
    import psbt_creator
    bad_change = "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4x"  # bad checksum
    res = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qg2t9c2mamc2r9l68v9r80xkqz0r5yyptqetk6k",
        amount_sats=10_000,
        utxos=[{"txid": "55"*32, "vout": 0, "value_satoshis": 70_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}],
        fee_satoshis=1000,
        network="testnet",
        change_address=bad_change
    )
    assert not res["success"], "Change address with bad checksum should fail"
    assert "invalid" in res.get("error"," ").lower() or "fail" in res.get("error"," ").lower()
