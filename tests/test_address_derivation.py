import pytest
import sys
print("PYTHON:", sys.executable)
import hdwallet
print("HDWALLET:", hdwallet.__file__)


# Vectors coneguts (m/84'/1'/0'/0/i) que ens has passat
VECTORS = [
    (0, "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", "03144e30cd31663a3770469818f3125b9e7af76b196c867f54cf11a9a4a644d074"),
    (1, "tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx", "03611c88a2d3b9d956b4327c4873b0b76d3a2c811b54eea69fa00a67a729d94126"),
    (2, "tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y", "0376b4c718e43a1436902f71f8a18419b7fa9fdc3a38b945edc943482f952691a4"),
    (3, "tb1q8dfs3646l8d3yq7yvxq6e4vjn9d8p6jfnn57ph", "02e256422d1106b4006e6e08738240545e3a0368ed4c6042f85f346dd5921cb21f"),
    (4, "tb1q9cz3hem70j0m9s474ap3ppqz9yppkluvrfvljd", "038a3f2a93441bc4d1b215669fb2c7ce6e5184be55147623b542343b347430d223"),
    (5, "tb1qf20p2maezkz32s0khamya39kky26rk8x3ucyew", "03a00da0126fb1b5c523d17a30d27145cde91b7d1e3aed0a42d87de2eb0c136ee7"),
    (6, "tb1qy7jl6rvmhxq83rhnt66m9775026f928gglxg5v", "03042f6d1e52b9b9fb15e983817b8f2a0416aa6b8f04ce292af6421b0af735751d"),
    (7, "tb1q9emwn7vhtd5uhsr58dvk4wk7n994hygrvrvqnj", "03c77e9c6cfc9412ac6d029f438619ce7828191118b72996bcde1bf60ad178ffbc"),
    (8, "tb1qs090grvqtrp3ppc342refh5lauehrpdw5qa2rd", "02fbb0435caebe69b9a880e17b853e355c941012a1c250e0d1bf1e8f4af3fe402b"),
]

@pytest.mark.integration
def test_vectors_match_known_results_with_cli_xpub(xpub_cli, network_cli):
    # Fallback XPUB (testnet) si no es proporciona via CLI per evitar skip
    if not xpub_cli:
        xpub_cli = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"

    # Import tardà per evitar errors d'importació si falta alguna dep
    try:
        import address_derivation as ad
    except Exception as e:
        pytest.skip(f"address_derivation import failed: {e}")

    # Deriva recepció (change=False) a testnet (els vectors són de testnet)
    assert network_cli == "testnet", f"Aquests vectors són testnet; vas passar network={network_cli}"
    for index, expected_addr, expected_pub in VECTORS:
        res = ad.derive_bitcoin_address(xpub_or_zpub=xpub_cli, index=index, change=False, network=network_cli)
        got_addr = res.get("address") or res.get("addr") or ""
        assert got_addr == expected_addr, f"index {index}: got {got_addr}, expected {expected_addr}"

        pub = res.get("public_key") or res.get("public_key_hex") or res.get("pubkey") or ""
        pub = pub.lower().removeprefix("0x")
        assert pub == expected_pub.lower(), f"index {index}: pubkey mismatch"

        path_str = res.get("path") or res.get("derivation_path") or res.get("bip32_path") or ""
        assert path_str.startswith("m/")
        assert path_str.endswith(f"/0/{index}")
