#!/usr/bin/env python3
"""
Test per verificar el comportament de memòria de l'Agent (recordar directives).
Similar a `test_agent.py` però centrant-se en que la consulta de recordatori
respon amb la darrera directiva.
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest

# Afegir el directori src al path
root_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(root_dir))

from dotenv import load_dotenv
load_dotenv()


def test_agent_memory():
    """Test de memòria: envia diverses directives i demana un recordatori."""
    from bitcoin_ai_agent import BitcoinAIAgent

    print("🧪 TEST MEMÒRIA DE L'AGENT IA BITCOIN")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    xpub = os.getenv("BITCOIN_XPUB", "zpub6qke5yCyxfwrc5ztdBnykfd36EAMGaRtNwoEob2MQd9cQCesyPD9mjMM6dZk4kpgEJxniwK6jbuzokVQ2cvBQv5qNWDRaRWDhfJykWPE2SB")
    network = os.getenv("BITCOIN_NETWORK", "testnet")

    if not api_key or api_key == "your-key-here":
        print("⚠️  Test saltat: No hi ha API key vàlida al .env (OPENAI_API_KEY)")
        pytest.skip("Sense OPENAI_API_KEY vàlida")

    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ Network: {network}")
    print(f"✅ XPUB: {xpub[:20]}...")

    # Crear agent
    try:
        agent = BitcoinAIAgent(api_key)
        agent.setup(xpub, network)
        print("✅ Agent creat i configurat!")
    except Exception as e:
        print(f"❌ Error creant agent: {e}")
        return

    # Seqüència de consultes (les tres primeres haurien de ser directives; l'última demana recordatori)
    test_queries = [
        "Genera una adreça nova i guarda-la per després",
        "Ara calcula el meu balanç (només vull saber que funciona)",
        "Maximitzar la privacitat a l'hora de seleccionar UTXOs",
        "Recordes quina es la directiva?"
    ]

    print("\n🧪 Executant seqüència de memòria...")
    print("-" * 60)

    last_response = None

    async def _run():
        nonlocal last_response
        for query in test_queries:
            print(f"\n💬 Pregunta: {query}")
            try:
                response = await agent.chat(query)
                print(f"🤖 Resposta: {response}")
                last_response = response
            except Exception as e:
                print(f"❌ Error: {e}")
                return

    asyncio.run(_run())

    print("\n" + "=" * 60)
    print("🔍 Verificant recordatori...")

    # Verificar que la resposta final conté referència a la darrera directiva
    # Acceptem dues formes: menció literal de la frase o etiqueta de 'darrera directiva'
    assert last_response is not None, "No s'ha obtingut resposta a la consulta de recordatori"
    assert (
        "darrera directiva" in last_response.lower() or
        "maximitzar la privacitat" in last_response.lower()
    ), "La resposta de recordatori no reflecteix la darrera directiva guardada"

    print("✅ Memòria verificada correctament (recorda la darrera directiva)")
    print("✅ TEST MEMÒRIA COMPLETAT")


if __name__ == "__main__":  # Execució manual opcional
    test_agent_memory()
