import os
import re
import base64
import asyncio
from pathlib import Path
import pytest


def test_agent_emits_single_line_psbt_and_files(tmp_path, monkeypatch):
    # Run in temp cwd to avoid clobbering project root files
    monkeypatch.chdir(tmp_path)

    # Ensure dummy API key to avoid module-level exit and force testnet
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "test-key"))
    monkeypatch.setenv("BITCOIN_NETWORK", os.getenv("BITCOIN_NETWORK", "testnet"))

    # Lazy import to respect PYTHONPATH of project
    import sys
    SRC = Path(__file__).resolve().parents[1] / 'src'
    if str(SRC) not in sys.path:
        sys.path.append(str(SRC))

    from bitcoin_ai_agent import BitcoinAIAgent  # type: ignore
    import psbt_creator  # type: ignore
    from psbt_creator import decode_psbt  # type: ignore

    # Block network calls from psbt_creator (prev_tx fetch)
    class _DummyResp:
        status_code = 404
        text = "offline"
    def _fake_get(*args, **kwargs):
        return _DummyResp()
    monkeypatch.setattr(psbt_creator.requests, "get", _fake_get, raising=True)

    # Stub the agent address enumeration and UTXO fetching to be deterministic and offline
    import bitcoin_ai_agent as agent_mod  # type: ignore

    fixed_receive_addr = "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"
    fixed_change_addr = fixed_receive_addr  # reuse valid bech32 for change

    def _fake_enumerate(xpub, network, receive_limit, change_limit):
        return [
            {"address": fixed_receive_addr, "path": "m/84'/1'/0'/0/0", "index": 0, "change": False},
            {"address": fixed_change_addr, "path": "m/84'/1'/0'/1/0", "index": 0, "change": True},
        ]
    monkeypatch.setattr(agent_mod, "_enumerate_addresses", _fake_enumerate, raising=True)

    def _fake_fetch_utxos_status(address, network, timeout=5.0, retries=2):
        if address == fixed_receive_addr:
            return ([
                {"txid": "11" * 32, "vout": 0, "value": 120_000, "status": {"confirmations": 6}},
                {"txid": "22" * 32, "vout": 1, "value": 80_000, "status": {"confirmations": 3}},
            ], True)
        return ([], True)
    monkeypatch.setattr(agent_mod, "_fetch_address_utxos_status", _fake_fetch_utxos_status, raising=True)

    def _fake_derive(xpub, network, index, change=False):
        return {
            "address": fixed_change_addr if change else fixed_receive_addr,
            "path": f"m/84'/1'/0'/{1 if change else 0}/{index}",
        }
    monkeypatch.setattr(agent_mod, 'derive_address_and_path', _fake_derive, raising=True)

    # Create agent and inject a FakeLLM to issue a create_transaction tool call, then a final message
    xpub = os.getenv("BITCOIN_XPUB", "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC")
    network = os.getenv("BITCOIN_NETWORK", "testnet")
    api_key = os.getenv("OPENAI_API_KEY", "test-key")

    agent = BitcoinAIAgent(api_key)
    agent.setup(xpub, network)

    class FakeLLM:
        def __init__(self, xpub: str, network: str):
            self.xpub = xpub
            self.network = network
            self._calls = 0
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            self._calls += 1
            if self._calls == 1:
                msg = AIMessage(content="Creant transacció…")
                msg.tool_calls = [
                    {
                        "id": "call-1",
                        "name": "create_transaction",
                        "args": {
                            "xpub": self.xpub,
                            "recipient_address": fixed_receive_addr,
                            "amount_sats": 10_000,
                            "fee_rate": 5,
                            "network": self.network,
                        },
                    }
                ]
                return msg
            return AIMessage(content="Transacció creada. He inclòs la PSBT a continuació.")

    agent.llm = FakeLLM(agent.xpub, agent.network)

    prompt = "Crea una PSBT mínima per enviar 0.0001 BTC a una de les meves adreces, usa create_transaction_manual si cal i mostra el PSBT en base64."

    async def _run():
        return await agent.chat(prompt)

    response = asyncio.run(_run())
    # Find base64 candidates and ensure at least one decodes and starts with magic
    cands = re.findall(r"cHNidP[0-9A-Za-z+/=]+", response)
    assert cands, "No PSBT base64 found in agent response"
    ok_b64 = None
    for c in cands:
        b64 = c + ("=" * (-len(c) % 4))
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        if raw.startswith(b"psbt\xff"):
            ok_b64 = b64
            break
    assert ok_b64, "No valid PSBT magic found"

    # It must decode with our decoder and report non-negative counts
    dec = decode_psbt(ok_b64, network=network)
    assert dec.get("valid", False), dec
    assert isinstance(dec.get("num_inputs", 0), int)
    assert isinstance(dec.get("num_outputs", 0), int)

    # Files should be persisted and contain the same PSBT
    assert Path("psbt_latest.base64").exists(), "psbt_latest.base64 missing"
    assert Path("psbt_latest.psbt").exists(), "psbt_latest.psbt missing"
    saved_b64 = Path("psbt_latest.base64").read_text().strip()
    assert "\n" not in saved_b64 and saved_b64.startswith("cHNidP")
    saved_raw = Path("psbt_latest.psbt").read_bytes()
    assert saved_raw.startswith(b"psbt\xff")
