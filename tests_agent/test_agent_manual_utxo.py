#!/usr/bin/env python3
"""Test d'integració agent: selecció manual d'UTXOs via prompt natural.

Aquest test replica l'estil dels altres tests d'agent: es construeix un prompt
en català demanant explícitament que l'agent creï una PSBT utilitzant EXACTAMENT
unes UTXOs concretes. Per fiabilitat, obtenim primer les UTXOs via l'eina
`list_utxos` (fora del LLM) i després passem els identificadors dins del prompt.
"""

import os
import sys
import asyncio
from pathlib import Path
import base64
import re
import pytest

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from bitcoin_ai_agent import BitcoinAIAgent  # type: ignore

XPUB = os.getenv("BITCOIN_XPUB", "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC")
NETWORK = os.getenv("BITCOIN_NETWORK", "testnet")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY in ("your-key-here", "dummy-key"), reason="Require real OPENAI_API_KEY for full agent test")
def test_agent_manual_utxo_prompt_flow():
    print("\n================ MANUAL UTXO (PROMPT) AGENT TEST ================")
    agent = BitcoinAIAgent(OPENAI_API_KEY)
    agent.setup(XPUB, NETWORK)

    # Obtenir UTXOs directament (fora del LLM) per poder construir prompt fiable
    utxo_tool = [t for t in agent.tools if t.name == 'list_utxos'][0]
    utxos_res = utxo_tool.invoke({"xpub": XPUB, "network": NETWORK})
    assert utxos_res.get('success'), utxos_res
    utxos = utxos_res.get('utxos', [])
    if len(utxos) < 2:
        pytest.skip("Necessitem almenys 2 UTXOs per la prova manual")

    chosen = utxos[:2]
    chosen_ids = [f"{u.get('txid')}:{u.get('vout')}" for u in chosen]
    recipient = chosen[0]['address']  # reutilitzem la mateixa adreça com a destí (test simplificat)

    prompt = (
        "Crea'm una transacció PSBT utilitzant EXACTAMENT aquestes UTXOs (selecció manual) "
        f"amb la eina create_transaction_manual: {', '.join(chosen_ids)}. "
        f"Envia 0.0002 BTC a {recipient} amb fee rate 10 sat/vB. Mostra la PSBT base64 completa."
    )

    async def _run():
        response = await agent.chat(prompt)
        print("\nResposta de l'agent:\n", response)
        # Buscar totes les seqüències candidates (prefix cHNidP)
        candidates = re.findall(r"cHNidP[0-9A-Za-z+/=]+", response)
        assert candidates, "No s'ha trobat cap seqüència base64 candidata"
        chosen_psbt = None
        for cand in candidates:
            # Padding si cal (evita error per longitud 1 mod 4) sense alterar contingut si complet
            padded = cand + ("=" * (-len(cand) % 4))
            try:
                raw = base64.b64decode(padded)
            except Exception:
                continue
            if raw.startswith(b"psbt\xff") and len(raw) > 50:  # mida mínima raonable
                chosen_psbt = padded
                break
        assert chosen_psbt, "Cap PSBT vàlida detectada (totes truncades?)"
        print("PSBT detectada (inici):", chosen_psbt[:60], "...")
        # Valida amb el nostre decoder per assegurar compatibilitat Electrum
        import sys as _sys
        from pathlib import Path as _Path
        if str(_Path(__file__).resolve().parent.parent / 'src') not in _sys.path:
            _sys.path.append(str(_Path(__file__).resolve().parent.parent / 'src'))
        from psbt_creator import decode_psbt  # type: ignore
        dec = decode_psbt(chosen_psbt, network=NETWORK)
        assert dec.get("valid", False), dec
        assert dec.get("num_inputs", 0) >= 1
        assert dec.get("num_outputs", 0) >= 1
        # Fitxers han de existir i coincidir
        assert Path("psbt_latest.base64").exists(), "psbt_latest.base64 no existeix"
        assert Path("psbt_latest.psbt").exists(), "psbt_latest.psbt no existeix"
        saved_b64 = Path("psbt_latest.base64").read_text().strip()
        assert "\n" not in saved_b64 and saved_b64.startswith("cHNidP")
        saved_raw = Path("psbt_latest.psbt").read_bytes()
        assert saved_raw.startswith(b"psbt\xff")

    asyncio.run(_run())
    print("===============================================================\n")

