import base64

# This test checks that when previous tx fetch fails, the witness_utxo scriptPubKey
# matches the input address type (P2PKH, P2SH, P2WPKH) instead of always P2WPKH.

def extract_input_maps(psbt_bytes: bytes):
    # Very lightweight parser: skip magic & global map until first 0x00 separator after global
    assert psbt_bytes.startswith(b"psbt\xff")
    off = 5
    # read global key-value pairs
    while True:
        key_len = psbt_bytes[off]
        off += 1
        if key_len == 0:
            break
        key = psbt_bytes[off:off+key_len]
        off += key_len
        # value
        # compact size read
        n = psbt_bytes[off]
        off += 1
        if n == 0xfd:
            n = int.from_bytes(psbt_bytes[off:off+2], 'little'); off += 2
        elif n == 0xfe:
            n = int.from_bytes(psbt_bytes[off:off+4], 'little'); off += 4
        elif n == 0xff:
            n = int.from_bytes(psbt_bytes[off:off+8], 'little'); off += 8
        val = psbt_bytes[off:off+n]
        off += n
    # Now at first input map
    input_maps = []
    while off < len(psbt_bytes):
        key_len = psbt_bytes[off]
        off += 1
        if key_len == 0:  # separator ends this input
            # end of one input map
            if psbt_bytes[off:off+1] == b"\x00":
                # next zero => probably output map start
                break
            continue
        key = psbt_bytes[off:off+key_len]
        off += key_len
        # read value
        n = psbt_bytes[off]; off +=1
        if n == 0xfd:
            n = int.from_bytes(psbt_bytes[off:off+2], 'little'); off += 2
        elif n == 0xfe:
            n = int.from_bytes(psbt_bytes[off:off+4], 'little'); off += 4
        elif n == 0xff:
            n = int.from_bytes(psbt_bytes[off:off+8], 'little'); off += 8
        val = psbt_bytes[off:off+n]; off += n
        input_maps.append((key, val))
    return input_maps


def test_witness_utxo_fallback_scripts(monkeypatch):
    import psbt_creator
    c = psbt_creator.PSBTCreator(network="testnet")

    # Force fetch failure
    monkeypatch.setattr(c, "_fetch_transaction", lambda txid: None, raising=True)

    test_vectors = [
        # (address, expected script prefix bytes)
        ("mipcBbFg9gMiCh81Kj8tqqdgoZub1ZJRfn", b"\x76\xa9\x14"),  # P2PKH testnet
        ("2NBFNJTktNa7GZusGbDbGKRZTxdK9VVez3n", b"\xa9\x14"),       # P2SH testnet
        ("tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", b"\x00\x14"), # P2WPKH
    ]
    for addr, prefix in test_vectors:
        psbt_res = c.create_psbt(
            inputs=[{"txid": "11"*32, "vout": 0, "value_satoshis": 12345, "address": addr}],
            outputs=[{"address": addr, "value": 10000}],
        )
        # create_psbt returns {"psbt": ...} on success, with no "success" key
        assert "psbt" in psbt_res, f"Missing psbt key: {psbt_res}"
        raw = base64.b64decode(psbt_res["psbt"])
        input_entries = extract_input_maps(raw)
        # find witness utxo entry key type 0x01 (PSBT_IN_WITNESS_UTXO)
        witness_vals = [v for k, v in input_entries if k.startswith(b"\x01")]
        assert witness_vals, "Missing witness utxo entry"
        w = witness_vals[0]
        # value(8) + compact script len + script
        assert len(w) >= 9
        script_len = w[8]
        script = w[9:9+script_len]
        assert script.startswith(prefix), f"Script {script.hex()} does not start with expected {prefix.hex()} for {addr}"
