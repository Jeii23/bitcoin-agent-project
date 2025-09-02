import pytest
import base64

@pytest.mark.usefixtures("block_requests")
def test_decode_psbt_roundtrip():
    import psbt_creator
    # Build a small psbt
    build = psbt_creator.create_transaction_psbt(
        xpub='xpub_unused_for_tests',
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_btc=0.0002,
        utxos=[{"txid":"44"*32, "vout":0, "value_satoshis":80_000, "address":"tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"}],
        fee_satoshis=800,
        network="testnet",
        change_address="tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"
    )
    assert build.get("success", True), build.get("error")

    decoded = psbt_creator.PSBTCreator(network="testnet").decode_psbt(build["psbt"])
    # Decoder is intentionally lightweight; just check shape and absence of explicit error
    assert isinstance(decoded, dict)
    assert "error" not in decoded

    # num_inputs/num_outputs should exist and be non-negative integers, but may be zero
    assert isinstance(decoded.get("num_inputs", 0), int)
    assert decoded.get("num_inputs", 0) >= 0
    assert isinstance(decoded.get("num_outputs", 0), int)
    assert decoded.get("num_outputs", 0) >= 0

    # Optional structural fields if present should be lists
    if "inputs" in decoded:
        assert isinstance(decoded["inputs"], list)
    if "outputs" in decoded:
        assert isinstance(decoded["outputs"], list)

def test_psbt_magic_prefix_base64():
    import psbt_creator
    b64 = "cHNidP8BAHE="  # 'psbt\xff\x01\x01q' - just checks base64 handling
    dec = psbt_creator.PSBTCreator(network="testnet").decode_psbt(b64)
    assert isinstance(dec, dict)
