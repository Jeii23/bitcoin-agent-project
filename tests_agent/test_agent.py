#!/usr/bin/env python3
"""
Test simple per verificar que l'Agent IA funciona
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest

# Afegir el directori 'src' al PYTHONPATH
SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# Carregar .env
from dotenv import load_dotenv
load_dotenv()

def test_agent():
    """Test bàsic de l'agent"""
    
    # Importar després de configurar el path
    from bitcoin_ai_agent import BitcoinAIAgent
    
    print("🧪 TEST DE L'AGENT IA BITCOIN")
    print("=" * 50)
    
    # Obtenir configuració
    api_key = os.getenv("OPENAI_API_KEY")
    xpub = os.getenv("BITCOIN_XPUB", "zpub6qke5yCyxfwrc5ztdBnykfd36EAMGaRtNwoEob2MQd9cQCesyPD9mjMM6dZk4kpgEJxniwK6jbuzokVQ2cvBQv5qNWDRaRWDhfJykWPE2SB")
    network = os.getenv("BITCOIN_NETWORK", "testnet")
    
    if not api_key or api_key == "your-key-here":
        print("❌ No s'ha trobat API key vàlida al .env")
        return
    
    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ Network: {network}")
    print(f"✅ XPUB: {xpub[:20]}...")
    
    # Crear agent
    print("\n📦 Creant agent...")
    try:
        agent = BitcoinAIAgent(api_key)
        agent.setup(xpub, network)
        print("✅ Agent creat correctament!")
    except Exception as e:
        print(f"❌ Error creant agent: {e}")
        return
    
    # Tests
    test_queries = [
        "Hola, qui ets?",
        "Genera'm una adreça nova que no hagi fet servir",
        "Quin és el meu balanç? i diguem totes les adreçes que has comprobat",
        "Quines són les fees actuals?"
    ]
    
    print("\n🧪 Executant tests...")
    print("-" * 50)
    
    async def _run():
        for query in test_queries:
            print(f"\n💬 Pregunta: {query}")
            try:
                response = await agent.chat(query)
                print(f"🤖 Resposta: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")

    asyncio.run(_run())
    
    print("\n" + "=" * 50)
    print("✅ TEST COMPLETAT")

if __name__ == "__main__":
    asyncio.run(test_agent())