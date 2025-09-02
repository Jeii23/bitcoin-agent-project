import pytest

# Test that derive_real_address_hdwallet no longer returns synthetic addresses
# when underlying derivation fails.

def test_derive_real_address_hdwallet_raises(monkeypatch):
    import address_derivation as ad

    def fake_derivation(xpub, index, change, network):
        return {"success": False, "error": "forced failure"}

    monkeypatch.setattr(ad, "derive_bitcoin_address", fake_derivation, raising=True)

    with pytest.raises(ValueError) as exc:
        ad.derive_real_address_hdwallet("vpubInvalidKey", "testnet", 0, change=False)

    assert "Derivation failed" in str(exc.value)
    assert "forced failure" in str(exc.value)
