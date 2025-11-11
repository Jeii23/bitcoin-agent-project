import base64
import pytest

from psbt_creator import PSBTCreator

# Using BIP-350 test vector taproot addresses (witness v1, 32-byte program)
# We verify produced scriptPubKey matches the vector (OP_1 0x51 + push32 + program)
MAINNET_TAPROOT = (
    "bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",
    "512079be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
)
TESTNET_TAPROOT = (
    "tb1pqqqqp399et2xygdj5xreqhjjvcmzhxw4aywxecjdzew6hylgvsesf3hn0c",
    "5120000000c4a5cad46221b2a187905e5266362b99d5e91c6ce24d165dab93e86433",
)

@pytest.mark.parametrize("network, vector", [
    ("mainnet", MAINNET_TAPROOT),
    ("testnet", TESTNET_TAPROOT),
])
def test_taproot_output_script(network, vector):
    addr, expected_spk_hex = vector
    c = PSBTCreator(network=network)
    # Build a minimal PSBT with single output; dummy utxo input (value and address can be any valid bech32 v0 p2wpkh for building)
    dummy_input = {
        "txid": "aa"*32,
        "vout": 0,
        "value_satoshis": 100000,
        # Use a valid v0 bech32 address for witness fallback script building
        "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t" if network=="testnet" else "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
    }
    psbt_res = c.create_psbt(inputs=[dummy_input], outputs=[{"address": addr, "value": 50000}])
    assert psbt_res["success"]
    psbt_bytes = base64.b64decode(psbt_res["psbt"])
    # Parse unsigned tx outputs to extract scriptPubKey
    # After magic 'psbt\xff' global map etc; simpler: decode stored unsigned_tx_hex
    unsigned_hex = psbt_res["unsigned_tx_hex"]
    # Raw tx format: version(4) vin_count varint, vin(s)... vout_count varint, then outputs
    tx = bytes.fromhex(unsigned_hex)
    offset = 4  # version
    # vin count (assume <0xfd)
    vin_cnt = tx[offset]
    offset += 1
    # skip vins
    for _ in range(vin_cnt):
        offset += 32 + 4  # txid + vout
        script_len = tx[offset]
        offset += 1 + script_len  # scriptSig
        offset += 4  # sequence
    vout_cnt = tx[offset]
    offset += 1
    assert vout_cnt == 1
    value = int.from_bytes(tx[offset:offset+8], 'little')
    assert value == 50000
    offset += 8
    spk_len = tx[offset]
    offset += 1
    spk = tx[offset:offset+spk_len]
    # Expect script: OP_1 (0x51) PUSH32 0x20 + program
    assert spk[0] == 0x51 and spk[1] == 0x20
    assert spk_len == 34
    assert spk.hex() == expected_spk_hex
