import pytest
import types

# We will monkeypatch requests.get used inside bitcoin_ai_agent.list_utxos

class FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        return self._payload

@pytest.fixture
def force_rest_api(monkeypatch):
    """Force REST API usage by clearing RPC env vars."""
    for k in ("BITCOIN_RPC_USER", "BITCOIN_RPC_PASSWORD", "BITCOIN_RPC_HOST", "BITCOIN_RPC_PORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BLOCKCHAIN_API", "blockstream")

@pytest.fixture
def fake_requests_get(monkeypatch, force_rest_api):
    import bitcoin_ai_agent as agent
    # Collect the first few receive and change addresses deterministically
    recv_addrs = [agent.derive_real_address("vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC", "testnet", i, change=False) for i in range(2)]
    chg_addrs = [agent.derive_real_address("vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC", "testnet", i, change=True) for i in range(1)]
    utxo_map = {
        recv_addrs[0]: [
            {"txid": "d"*64, "vout": 0, "value": 1111, "status": {"confirmations": 3}},
        ],
        recv_addrs[1]: [],
        chg_addrs[0]: [
            {"txid": "e"*64, "vout": 1, "value": 2222, "status": {"confirmations": 7}},
        ],
    }
    def _fake_get(url, timeout=5):
        for addr, payload in utxo_map.items():
            if f"/address/{addr}/utxo" in url:
                return FakeResp(200, payload)
        return FakeResp(200, [])
    monkeypatch.setattr(agent.requests, "get", _fake_get, raising=True)
    return utxo_map


def test_list_utxos_includes_change(fake_requests_get):
    import bitcoin_ai_agent as agent
    res = agent.list_utxos.invoke({"xpub": "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC", "network": "testnet"})
    assert res["success"]
    # Ensure at least one change utxo present
    change_utxos = [u for u in res["utxos"] if u.get("change") is True]
    assert change_utxos, "Expected at least one change UTXO"
    # Verify totals reflect both entries
    total = sum(u["value_satoshis"] for u in res["utxos"]) 
    assert total == 1111 + 2222
    # Ensure metadata fields exist
    assert res["receive_addresses_checked"] == 10
    assert res["change_addresses_checked"] == 5
