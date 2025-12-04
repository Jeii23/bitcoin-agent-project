import pytest


def test_agent_derive_address_no_fake(monkeypatch):
    import bitcoin_ai_agent as agent

    # Clear RPC env vars to ensure hermetic test
    monkeypatch.delenv("BITCOIN_RPC_USER", raising=False)
    monkeypatch.delenv("BITCOIN_RPC_PASSWORD", raising=False)

    # Force both derivation layers to fail:
    if agent.CUSTOM_DERIVATION_AVAILABLE:
        def bad_custom(xpub, index, change, network):
            return {"success": False, "error": "custom-fail"}
        monkeypatch.setattr(agent, "derive_bitcoin_address", bad_custom, raising=True)
    if agent.HDWALLET_AVAILABLE:
        class FailHDW:
            def __init__(self, *a, **k):
                pass
            def from_xpublic_key(self, *a, **k):
                raise RuntimeError("hdwallet-fail")
        monkeypatch.setattr(agent, "HDWallet", FailHDW, raising=True)

    with pytest.raises(ValueError) as exc:
        agent.derive_address_and_path("invalidXpubLike", "mainnet", 0, change=False)

    s = str(exc.value)
    assert "Address derivation failed" in s
    # The error message should indicate derivation failed (no synthetic fallback)
    assert "derivation" in s.lower() or "failed" in s.lower()
