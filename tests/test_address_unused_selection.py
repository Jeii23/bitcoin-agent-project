import pytest
import os


VPUb = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"


@pytest.fixture
def force_rest_api(monkeypatch):
    """Force REST API usage by clearing RPC env vars and setting BLOCKCHAIN_API."""
    for k in ("BITCOIN_RPC_USER", "BITCOIN_RPC_PASSWORD", "BITCOIN_RPC_HOST", "BITCOIN_RPC_PORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BLOCKCHAIN_API", "blockstream")


def test_generate_address_first_unused_when_all_network_errors(monkeypatch, force_rest_api):
    """If every history lookup raises a network error, we should still return index 0.
    Regression guard: older logic marked errors as 'used' and skipped ahead.
    """
    import bitcoin_ai_agent as agent
    class Boom(Exception):
        pass
    def failing_get(*a, **k):
        raise Boom("net down")
    monkeypatch.setattr(agent.requests, "get", failing_get, raising=True)

    res = agent.generate_address.invoke({"xpub": VPUb, "network": "testnet", "require_unused": True})
    assert res["success"], res
    assert res["index"] == 0, f"Expected first index (0) when network unavailable, got {res['index']}"


def test_generate_address_skips_used_indices(monkeypatch, force_rest_api):
    """Simulate first two indices having history (funded_txo_count>0) then unused.
    Should return index 2 as first unused.
    """
    import bitcoin_ai_agent as agent
    # Derive first few addresses deterministically (no network calls here)
    addrs = [agent.derive_real_address(VPUb, "testnet", i, change=False) for i in range(3)]
    addr_index_map = {a: i for i, a in enumerate(addrs)}

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
        def json(self):
            return self._payload

    def fake_get(url, timeout=0):
        for a in addrs:
            if f"/address/{a}" in url and "/utxo" not in url:
                idx = addr_index_map[a]
                if idx < 2:  # mark as used
                    return FakeResp(200, {"chain_stats": {"funded_txo_count": 1}, "mempool_stats": {}})
                else:  # index 2 unused
                    return FakeResp(200, {"chain_stats": {"funded_txo_count": 0}, "mempool_stats": {}})
            if f"/address/{a}/utxo" in url:
                return FakeResp(200, [])
        return FakeResp(404, {})

    monkeypatch.setattr(agent.requests, "get", fake_get, raising=True)
    res = agent.generate_address.invoke({"xpub": VPUb, "network": "testnet", "require_unused": True})
    assert res["success"], res
    assert res["index"] == 2, f"Expected first unused index 2, got {res['index']}"


def test_list_utxos_all_network_failures_includes_note(monkeypatch, force_rest_api):
    """If every UTXO query fails, result should include diagnostic note and network_errors > 0."""
    import bitcoin_ai_agent as agent
    def failing_get(*a, **k):
        class R:
            status_code = 500
            def json(self):
                return {}
        return R()
    monkeypatch.setattr(agent.requests, "get", failing_get, raising=True)
    res = agent.list_utxos.invoke({"xpub": VPUb, "network": "testnet"})
    assert res["success"], res
    assert res.get("total_utxos") == 0
    assert "note" in res, "Expected diagnostic note when all network calls fail"
    assert res.get("network_errors", 0) > 0
