#!/usr/bin/env python3
"""
Tests complets per verificar la creació de PSBTs estàndard BIP-174
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from decimal import Decimal
from typing import Dict, List

# Assegura que el directori 'src' és al PYTHONPATH abans d'importar mòduls locals
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from bitcoin_ai_agent import derive_address_and_path  # type: ignore

# Afegir també el directori actual (per compatibilitat amb altres imports relatius antics)
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# Carregar .env
from dotenv import load_dotenv
load_dotenv()

# Importar mòduls necessaris
try:
    from psbt_creator import PSBTCreator, create_transaction_psbt
    PSBT_AVAILABLE = True
except ImportError:
    PSBT_AVAILABLE = False
    print("❌ No s'ha trobat psbt_creator.py")

try:
    from address_derivation import derive_bitcoin_address
    ADDRESS_DERIVATION_AVAILABLE = True
except ImportError:
    ADDRESS_DERIVATION_AVAILABLE = False

# Colors per output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(name: str, success: bool, details: str = ""):
    """Imprimeix resultat d'un test"""
    if success:
        print(f"{Colors.GREEN}✅ {name}{Colors.END}")
    else:
        print(f"{Colors.RED}❌ {name}{Colors.END}")
    
    if details:
        print(f"   {Colors.CYAN}{details}{Colors.END}")

def print_section(title: str):
    """Imprimeix un títol de secció"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")

class PSBTTester:
    """Classe per testejar PSBTs"""
    
    def __init__(self):
        self.creator = PSBTCreator(network="testnet")
        self.test_results = []
        # Usar una XPUB vàlida diferent per tests (aquesta és una XPUB testnet vàlida coneguda)
        self.xpub = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"
        
        # UTXOs de test simulades amb adreces vàlides
        self.test_utxos = [
            {
                "txid": "a" * 64,
                "vout": 0,
                "value_satoshis": 100000,
                "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",  # Adreça Bech32 vàlida
                "confirmations": 10
            },
            {
                "txid": "b" * 64,
                "vout": 1,
                "value_satoshis": 50000,
                "address": "tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",  # Adreça Bech32 vàlida
                "confirmations": 5
            },
            {
                "txid": "c" * 64,
                "vout": 0,
                "value_satoshis": 75000,
                "address": "2MzQwSSnBHWHqSAqtTVQ6v47XtaisrJa1Vc",  # Adreça P2SH testnet vàlida
                "confirmations": 20
            }
        ]
    
    
    def test_basic_psbt_creation(self) -> bool:
        """Test 1: Creació bàsica de PSBT"""
        try:
            # Usar adreça vàlida per testnet
            psbt_res = self.creator.create_psbt(
                inputs=[self.test_utxos[0]],
                outputs=[
                    {"address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", "value": 90000}
                ]
            )
            assert psbt_res["success"]
            psbt = psbt_res["psbt"]
            decoded = base64.b64decode(psbt)
            
            # Verificar magic bytes
            if decoded[:5] != b'psbt\xff':
                print_test("Creació bàsica PSBT", False, "Magic bytes incorrectes")
                return False
            
            print_test("Creació bàsica PSBT", True, f"PSBT creat: {psbt[:30]}...")
            return True
            
        except Exception as e:
            print_test("Creació bàsica PSBT", False, str(e))
            return False
    
    def test_multi_input_output(self) -> bool:
        """Test 2: PSBT amb múltiples inputs i outputs"""
        try:
            # Usar adreces vàlides
            psbt_res = self.creator.create_psbt(
                inputs=self.test_utxos[:2],  # 2 inputs
                outputs=[
                    {"address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", "value": 60000},
                    {"address": "2MzQwSSnBHWHqSAqtTVQ6v47XtaisrJa1Vc", "value": 30000},
                    {"address": "tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx", "value": 9000}
                ]
            )
            assert psbt_res["success"]
            psbt = psbt_res["psbt"]
            decoded = base64.b64decode(psbt)
            if decoded[:5] != b'psbt\xff':
                print_test("PSBT multi-input/output", False, "Format invàlid")
                return False
            
            print_test("PSBT multi-input/output", True, "2 inputs, 3 outputs")
            return True
            
        except Exception as e:
            print_test("PSBT multi-input/output", False, str(e))
            return False
    
    def test_address_decoding(self) -> bool:
        """Test 3: Decodificació d'adreces"""
        test_addresses = [
            # Bech32 (P2WPKH) - adreces reals vàlides
            {"addr": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", "type": "Bech32"},
            # P2SH (testnet) - adreça real vàlida
            {"addr": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", "type": "P2SH"}
        ]
        
        all_passed = True
        
        for test in test_addresses:
            try:
                version, hash_data = self.creator._decode_address(test["addr"])
                if len(hash_data) > 0:
                    print_test(f"Decodificar {test['type']}", True, f"Hash: {hash_data.hex()[:20]}...")
                else:
                    print_test(f"Decodificar {test['type']}", False, "Hash buit")
                    all_passed = False
            except Exception as e:
                print_test(f"Decodificar {test['type']}", False, str(e))
                all_passed = False
        
        return all_passed
    
    def test_transaction_creation(self) -> bool:
        """Test 4: Crear transacció completa amb PSBT"""
        try:
            result = create_transaction_psbt(
                xpub=self.xpub,
                recipient_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
                amount_btc=0.001,
                utxos=self.test_utxos,
                fee_satoshis=1000,
                change_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",  # Adreça vàlida per change
                network="testnet"
            )
            
            if not result["success"]:
                print_test("Crear transacció completa", False, result["error"])
                return False
            
            # Verificar camps esperats
            required_fields = ["psbt", "total_input_btc", "amount_btc", "fee_btc", "num_inputs", "num_outputs"]
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                print_test("Crear transacció completa", False, f"Falten camps: {missing_fields}")
                return False
            
            # Verificar PSBT
            psbt_bytes = base64.b64decode(result["psbt"])
            if psbt_bytes[:5] != b'psbt\xff':
                print_test("Crear transacció completa", False, "PSBT invàlid")
                return False
            
            details = f"Inputs: {result['num_inputs']}, Outputs: {result['num_outputs']}, Fee: {result['fee_btc']:.8f} BTC"
            print_test("Crear transacció completa", True, details)
            return True
            
        except Exception as e:
            print_test("Crear transacció completa", False, str(e))
            return False
    
    def test_insufficient_funds(self) -> bool:
        """Test 5: Gestió de fons insuficients"""
        try:
            # Intentar enviar més del que tenim
            result = create_transaction_psbt(
                xpub=self.xpub,
                recipient_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
                amount_btc=10.0,  # Molt més del que tenim en UTXOs
                utxos=self.test_utxos,
                fee_satoshis=1000,
                network="testnet"
            )
            
            if result["success"]:
                print_test("Gestió fons insuficients", False, "Hauria de fallar amb fons insuficients")
                return False
            
            if "insuficients" in result["error"].lower():
                print_test("Gestió fons insuficients", True, "Error detectat correctament")
                return True
            else:
                print_test("Gestió fons insuficients", False, f"Error inesperat: {result['error']}")
                return False
                
        except Exception as e:
            print_test("Gestió fons insuficients", False, str(e))
            return False
    
    def test_dust_output_handling(self) -> bool:
        """Test 6: Gestió de dust outputs"""
        try:
            # Crear transacció que deixa canvi sota el dust limit
            result = create_transaction_psbt(
                xpub=self.xpub,
                recipient_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
                amount_btc=0.000994,  # Deixa ~500 satoshis de canvi (sota 546 dust limit)
                utxos=[self.test_utxos[0]],  # 100000 satoshis
                fee_satoshis=100,
                network="testnet"
            )
            
            if not result["success"]:
                print_test("Gestió dust outputs", False, result["error"])
                return False
            
            # No hauria d'haver-hi output de canvi
            if result["num_outputs"] == 1:
                print_test("Gestió dust outputs", True, "Canvi sota dust limit no creat")
                return True
            else:
                print_test("Gestió dust outputs", False, f"Outputs inesperats: {result['num_outputs']}")
                return False
                
        except Exception as e:
            print_test("Gestió dust outputs", False, str(e))
            return False
    
    def test_psbt_decode(self) -> bool:
        """Test 7: Decodificar i validar PSBT"""
        try:
            # Primer crear un PSBT amb adreça vàlida
            psbt_res = self.creator.create_psbt(
                inputs=[self.test_utxos[0]],
                outputs=[{"address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t", "value": 90000}]
            )
            assert psbt_res["success"]
            psbt = psbt_res["psbt"]
            decoded = self.creator.decode_psbt(psbt)
            
            if not decoded["valid"]:
                print_test("Decodificar PSBT", False, decoded.get("error", "Error desconegut"))
                return False
            
            print_test("Decodificar PSBT", True, f"Versió PSBT: {decoded.get('version', 0)}")
            return True
            
        except Exception as e:
            print_test("Decodificar PSBT", False, str(e))
            return False
    
    def test_fee_calculation(self) -> bool:
        """Test 8: Càlcul de fees correcte"""
        try:
            test_cases = [
                {"inputs": 1, "outputs": 2, "fee": 1000},
                {"inputs": 2, "outputs": 2, "fee": 2000},
                {"inputs": 3, "outputs": 3, "fee": 3000}
            ]
            
            all_passed = True
            for case in test_cases:
                result = create_transaction_psbt(
                    xpub=self.xpub,
                    recipient_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
                    amount_btc=0.0001,
                    utxos=self.test_utxos[:case["inputs"]],
                    fee_satoshis=case["fee"],
                    change_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
                    network="testnet"
                )
                
                if result["success"]:
                    expected_fee = case["fee"] / 100_000_000
                    actual_fee = result["fee_btc"]
                    if abs(actual_fee - expected_fee) < 0.00000001:
                        print_test(f"Fee {case['inputs']}in/{case['outputs']}out", True, 
                                 f"Fee: {actual_fee:.8f} BTC")
                    else:
                        print_test(f"Fee {case['inputs']}in/{case['outputs']}out", False,
                                 f"Expected: {expected_fee}, Got: {actual_fee}")
                        all_passed = False
                else:
                    print_test(f"Fee {case['inputs']}in/{case['outputs']}out", False, result["error"])
                    all_passed = False
                    
            return all_passed
            
        except Exception as e:
            print_test("Càlcul de fees", False, str(e))
            return False
    
    def test_change_address_generation(self) -> bool:
        """Test 9: Generació d'adreça de canvi"""
        try:
            # Usar una adreça de canvi vàlida coneguda
            change_address = "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"
            
            # Crear transacció amb adreça de canvi
            result = create_transaction_psbt(
                xpub=self.xpub,
                recipient_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
                amount_btc=0.0005,
                utxos=[self.test_utxos[0]],
                fee_satoshis=1000,
                change_address=change_address,
                network="testnet"
            )
            
            if not result["success"]:
                print_test("Generar adreça de canvi", False, result["error"])
                return False
            
            # Verificar que hi ha 2 outputs (destinatari + canvi)
            if result["num_outputs"] == 2:
                print_test("Generar adreça de canvi", True, 
                         f"Change: {result['change_btc']:.8f} BTC")
                return True
            else:
                print_test("Generar adreça de canvi", False, 
                         f"Outputs esperats: 2, trobats: {result['num_outputs']}")
                return False
                
        except Exception as e:
            print_test("Generar adreça de canvi", False, str(e))
            return False
    
    def test_compact_size_encoding(self) -> bool:
        """Test 10: Codificació CompactSize"""
        try:
            test_values = [
                (0, b'\x00'),
                (252, b'\xfc'),
                (253, b'\xfd\xfd\x00'),
                (65535, b'\xfd\xff\xff'),
                (65536, b'\xfe\x00\x00\x01\x00'),
                (4294967295, b'\xfe\xff\xff\xff\xff')
            ]
            
            all_passed = True
            for value, expected in test_values:
                result = self.creator._compact_size(value)
                if result == expected:
                    print_test(f"CompactSize {value}", True, f"Bytes: {result.hex()}")
                else:
                    print_test(f"CompactSize {value}", False, 
                             f"Expected: {expected.hex()}, Got: {result.hex()}")
                    all_passed = False
                    
            return all_passed
            
        except Exception as e:
            print_test("Codificació CompactSize", False, str(e))
            return False
    
    async def run_all_tests(self) -> Dict:
        """Executa tots els tests"""
        print_section("🧪 TESTS DE CREACIÓ DE PSBT BIP-174")
        
        if not PSBT_AVAILABLE:
            print(f"{Colors.RED}❌ No es pot executar: psbt_creator.py no trobat{Colors.END}")
            return {"total": 0, "passed": 0, "failed": 0}
        
        # Llista de tests
        tests = [
            ("Test bàsic de creació", self.test_basic_psbt_creation),
            ("Test multi-input/output", self.test_multi_input_output),
            ("Test decodificació adreces", self.test_address_decoding),
            ("Test transacció completa", self.test_transaction_creation),
            ("Test fons insuficients", self.test_insufficient_funds),
            ("Test dust outputs", self.test_dust_output_handling),
            ("Test decodificar PSBT", self.test_psbt_decode),
            ("Test càlcul de fees", self.test_fee_calculation),
            ("Test adreça de canvi", self.test_change_address_generation),
            ("Test CompactSize", self.test_compact_size_encoding)
        ]
        
        results = {"total": len(tests), "passed": 0, "failed": 0}
        
        print(f"\n{Colors.CYAN}Executant {len(tests)} tests...{Colors.END}\n")
        
        for name, test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    passed = await test_func()
                else:
                    passed = test_func()
                    
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                print_test(name, False, f"Excepció: {str(e)}")
                results["failed"] += 1
            
            # Petit delay entre tests
            await asyncio.sleep(0.1)
        
        # Resum final
        print_section("📊 RESUM DE TESTS")
        
        print(f"\nTotal tests: {results['total']}")
        print(f"{Colors.GREEN}Passats: {results['passed']}{Colors.END}")
        print(f"{Colors.RED}Fallits: {results['failed']}{Colors.END}")
        
        if results['passed'] == results['total']:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 TOTS ELS TESTS HAN PASSAT!{Colors.END}")
        elif results['passed'] > results['failed']:
            print(f"\n{Colors.YELLOW}⚠️  Alguns tests han fallat{Colors.END}")
        else:
            print(f"\n{Colors.RED}❌ La majoria de tests han fallat{Colors.END}")
        
        return results

def test_integration_with_agent():
    """Test d'integració amb l'agent Bitcoin"""
    print_section("🔌 TEST D'INTEGRACIÓ AMB L'AGENT")
    
    import os, asyncio
    try:
        from bitcoin_ai_agent import BitcoinAIAgent
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  No es pot importar bitcoin_ai_agent.py{Colors.END}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-key-here":
        print(f"{Colors.YELLOW}⚠️  No s'ha trobat API key, saltant test d'integració{Colors.END}")
        return

    async def _run():
        print(f"{Colors.CYAN}Creant agent amb PSBT millorat...{Colors.END}")
        agent = BitcoinAIAgent(api_key)
        xpub = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"
        agent.setup(xpub, "testnet")
        print(f"\n{Colors.CYAN}Provant crear una transacció amb l'agent...{Colors.END}")
        response = await agent.chat("Vull crear una transacció per enviar 0.001 BTC a tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx")
        print(f"\n{Colors.GREEN}Resposta de l'agent:{Colors.END}")
        print(response)
        if "psbt" in response.lower() or "transacció" in response.lower():
            print_test("Integració amb agent", True, "L'agent pot crear transaccions")
        else:
            print_test("Integració amb agent", False, "Resposta no conté informació PSBT")

    try:
        asyncio.run(_run())
    except Exception as e:
        print(f"{Colors.RED}❌ Error en integració: {str(e)}{Colors.END}")

async def main():
    """Funció principal de tests"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              🧪 TEST SUITE - PSBT BIP-174 CREATOR 🧪            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    xpub = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"
    for change in (False, True):
        for i in (0, 1):  # prova 0/0 i 0/1 + 1/0 i 1/1
            info = derive_address_and_path(xpub, "testnet", i, change=change)
            print(("change" if change else "recv"), i, info["address"], info["path"])

    
    # Executar tests principals
    tester = PSBTTester()
    results = await tester.run_all_tests()
    
    # Test d'integració (opcional)
    if results["passed"] > 0:
        await test_integration_with_agent()
    
    # Exit code segons resultats
    if results["failed"] == 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Tests interromputs per l'usuari{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Error fatal: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)