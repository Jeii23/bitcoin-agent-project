import pytest
import types
import threading
import time

# We will monkeypatch requests.get used inside bitcoin_ai_agent.list_utxos
VPUb = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"

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
    recv_addrs = [agent.derive_real_address(VPUb, "testnet", i, change=False) for i in range(2)]
    chg_addrs = [agent.derive_real_address(VPUb, "testnet", i, change=True) for i in range(1)]
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
    res = agent.list_utxos.invoke({"xpub": VPUb, "network": "testnet"})
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


def test_list_utxos_extends_when_receive_edge_is_active(monkeypatch, force_rest_api):
    import bitcoin_ai_agent as agent

    monkeypatch.delenv("BITCOIN_SCAN_RECEIVE", raising=False)
    monkeypatch.delenv("BITCOIN_SCAN_CHANGE", raising=False)

    recv_addrs = {
        i: agent.derive_real_address(VPUb, "testnet", i, change=False)
        for i in (0, 9, 11)
    }
    utxo_map = {
        recv_addrs[0]: [{"txid": "a" * 64, "vout": 0, "value": 1000, "status": {"confirmations": 1}}],
        recv_addrs[9]: [{"txid": "b" * 64, "vout": 1, "value": 2000, "status": {"confirmations": 2}}],
        recv_addrs[11]: [{"txid": "c" * 64, "vout": 0, "value": 3000, "status": {"confirmations": 3}}],
    }

    def fake_fetch(address, network, timeout=5.0, retries=2):
        return (utxo_map.get(address, []), True, "blockstream")

    monkeypatch.setattr(agent, "_fetch_address_utxos_status", fake_fetch, raising=True)

    res = agent.list_utxos.invoke({"xpub": VPUb, "network": "testnet"})
    assert res["success"], res
    assert res["total_utxos"] == 3
    assert res["receive_addresses_checked"] == 20
    assert max(u["index"] for u in res["utxos"] if not u["change"]) == 11


def test_get_balance_uses_same_adaptive_scan_as_list_utxos(monkeypatch, force_rest_api):
    import bitcoin_ai_agent as agent

    monkeypatch.delenv("BITCOIN_SCAN_RECEIVE", raising=False)
    monkeypatch.delenv("BITCOIN_SCAN_CHANGE", raising=False)

    recv_addrs = {
        i: agent.derive_real_address(VPUb, "testnet", i, change=False)
        for i in (9, 11)
    }
    utxo_map = {
        recv_addrs[9]: [{"txid": "d" * 64, "vout": 0, "value": 4000, "status": {"confirmations": 1}}],
        recv_addrs[11]: [{"txid": "e" * 64, "vout": 1, "value": 5000, "status": {"confirmations": 2}}],
    }

    def fake_fetch(address, network, timeout=5.0, retries=2):
        return (utxo_map.get(address, []), True, "blockstream")

    monkeypatch.setattr(agent, "_fetch_address_utxos_status", fake_fetch, raising=True)

    balance = agent.get_balance.invoke({"xpub": VPUb, "network": "testnet"})
    listed = agent.list_utxos.invoke({"xpub": VPUb, "network": "testnet"})

    assert balance["success"], balance
    assert listed["success"], listed
    assert balance["total_utxos"] == listed["total_utxos"] == 2
    assert balance["balance_satoshis"] == listed["total_value_satoshis"] == 9000
    assert balance["receive_addresses_checked"] == listed["receive_addresses_checked"] == 20


def test_list_utxos_core_preserves_confirmations(monkeypatch):
    import bitcoin_ai_agent as agent

    monkeypatch.setenv("BLOCKCHAIN_API", "core")

    address = agent.derive_real_address(VPUb, "testnet", 0, change=False)

    def fake_core_rpc_call(method, params):
        assert method == "scantxoutset"
        return {
            "height": 2127,
            "unspents": [
                {
                    "txid": "f" * 64,
                    "vout": 1,
                    "amount": 0.5,
                    "desc": f"addr({address})",
                    "height": 2117,
                }
            ],
        }

    monkeypatch.setattr(agent, "_core_rpc_call", fake_core_rpc_call, raising=True)

    res = agent.list_utxos.invoke({"xpub": VPUb, "network": "testnet"})
    assert res["success"], res
    assert res["total_utxos"] == 1
    assert res["utxos"][0]["confirmations"] == 11


def test_core_scantxoutset_is_serialized(monkeypatch):
    import bitcoin_ai_agent as agent

    active = 0
    overlap_detected = False
    active_lock = threading.Lock()

    def fake_core_rpc_call(method, params):
        nonlocal active, overlap_detected
        assert method == "scantxoutset"
        with active_lock:
            active += 1
            if active > 1:
                overlap_detected = True
        time.sleep(0.05)
        with active_lock:
            active -= 1
        addr = params[1][0][5:-1]
        return {
            "height": 2127,
            "unspents": [
                {
                    "txid": "a" * 64,
                    "vout": 0,
                    "amount": 0.1,
                    "desc": f"addr({addr})",
                    "height": 2127,
                }
            ],
        }

    monkeypatch.setattr(agent, "_core_rpc_call", fake_core_rpc_call, raising=True)

    threads = [
        threading.Thread(target=agent._core_scantxoutset, args=([f"bc1qtest{i}"],))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlap_detected is False
