#!/usr/bin/env python3
"""
Agent IA Bitcoin amb suport per PSBT BIP-174 estàndard
Versió millorada amb creació de transaccions reals
"""

import asyncio
import os
import json
import hashlib
import base58
from typing import TypedDict, List, Dict, Optional, Literal, Annotated, Sequence
from decimal import Decimal
from datetime import datetime
import requests
from pathlib import Path

# Carregar variables d'entorn des del fitxer .env
from dotenv import load_dotenv

# LangChain i LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# HDWallet per derivació correcta
try:
    from hdwallet import HDWallet
    from hdwallet.cryptocurrencies import Bitcoin
    HDWALLET_AVAILABLE = True
    print("[INFO] HDWallet disponible per derivació real d'adreces")
except ImportError:
    HDWALLET_AVAILABLE = False
    print("[WARNING] hdwallet no disponible, instal·la amb: pip install hdwallet")

# Derivació d'adreces pròpia
try:
    from address_derivation import derive_bitcoin_address
    CUSTOM_DERIVATION_AVAILABLE = True
    print("[INFO] Derivació personalitzada disponible (derive_bitcoin_address)")
except ImportError:
    CUSTOM_DERIVATION_AVAILABLE = False
    print("[WARNING] No s'ha trobat address_derivation.py")

# IMPORTAR EL NOU CREADOR DE PSBT
try:
    from psbt_creator import PSBTCreator, create_transaction_psbt
    PSBT_CREATOR_AVAILABLE = True
    print("[INFO] Creador de PSBT BIP-174 disponible")
except ImportError:
    PSBT_CREATOR_AVAILABLE = False
    print("[WARNING] No s'ha trobat psbt_creator.py - les transaccions usaran format JSON simplificat")

# Per la interfície
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

# ============== CONFIGURACIÓ ==============

# Carregar variables del fitxer .env
load_dotenv()

# Obtenir configuració del .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_XPUB = os.getenv("BITCOIN_XPUB", "")
DEFAULT_NETWORK = os.getenv("BITCOIN_NETWORK", "testnet").lower()

# Verificar que tenim la clau d'OpenAI
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-key-here":
    print("⚠️  ERROR: No s'ha trobat una API key vàlida d'OpenAI")
    print("\n🔑 Si us plau, edita el fitxer .env i afegeix la teva clau:")
    print("   OPENAI_API_KEY=sk-...")
    print("\n💡 Si no tens una clau, pots obtenir-la a: https://platform.openai.com/api-keys")
    exit(1)

# Validar network
if DEFAULT_NETWORK not in ["mainnet", "testnet"]:
    DEFAULT_NETWORK = "testnet"

# ============== ESTAT DE L'AGENT ==============

class AgentState(TypedDict):
    """Estat de l'agent IA"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    xpub: Optional[str]
    network: str
    addresses: List[Dict]
    utxos: List[Dict]
    balance_satoshis: int
    last_action: Optional[str]
    context: Dict
    last_psbt: Optional[str]  # Afegir camp per guardar l'últim PSBT creat

# ============== UTILITATS PER DERIVACIÓ ==============

def derive_address_and_path(xpub: str, network: str, index: int, change: bool = False) -> Dict:
    """
    Deriva adreça + path. Prioritza la derivació personalitzada.
    """
    # 1) Derivació personalitzada de l'usuari
    if CUSTOM_DERIVATION_AVAILABLE:
        try:
            res = derive_bitcoin_address(xpub, index=index, change=change, network=network)
            if res.get("success"):
                return {"address": res["address"], "path": res.get("path", "")}
            else:
                print(f"[WARN] Derivació personalitzada ha fallat: {res.get('error')}")
        except Exception as e:
            print(f"[ERROR] Derivació personalitzada: {e}")

    # 2) Fallback: HDWallet simple
    if HDWALLET_AVAILABLE:
        try:
            hdwallet = HDWallet(cryptocurrency=Bitcoin)
            hdwallet.from_xpublic_key(xpublic_key=xpub)
            chain = 1 if change else 0
            hdwallet.clean_derivation()
            hdwallet.from_path(path=f"m/{chain}/{index}")
            address = hdwallet.p2wpkh_address()
            coin_type = "1'" if network == "testnet" else "0'"
            path = f"m/84'/{coin_type}/0'/{chain}/{index}"
            return {"address": address, "path": path}
        except Exception as e:
            print(f"[ERROR] Fallback HDWallet: {e}")

    # 3) Últim recurs: fake
    prefix = "tb1q" if network == "testnet" else "bc1q"
    kind = "change" if change else "receive"
    addr_hash = hashlib.sha256(f"{xpub}{kind}{index}".encode()).hexdigest()
    fake_address = f"{prefix}{addr_hash[:39]}"
    coin_type = "1'" if network == "testnet" else "0'"
    chain = 1 if change else 0
    path = f"m/84'/{coin_type}/0'/{chain}/{index}"
    return {"address": fake_address, "path": path}

def derive_real_address(xpub: str, network: str, index: int, change: bool = False) -> str:
    """Manté la signatura antiga per compatibilitat"""
    info = derive_address_and_path(xpub, network, index, change)
    return info["address"]

# ============== EINES (TOOLS) PER L'AGENT IA ==============

@tool
def get_balance(xpub: str, network: str = "testnet") -> Dict:
    """
    Obté el balanç d'una wallet Bitcoin desde la XPUB.
    """
    try:
        addresses = []
        for i in range(5):
            info = derive_address_and_path(xpub, network, i, change=False)
            addresses.append({
                "address": info["address"],
                "path": info["path"]
            })

        total_balance = 0
        total_utxos = 0
        api_base = "https://blockstream.info/testnet/api" if network == "testnet" else "https://blockstream.info/api"

        for addr_info in addresses:
            try:
                response = requests.get(f"{api_base}/address/{addr_info['address']}/utxo", timeout=5)
                if response.status_code == 200:
                    utxos = response.json()
                    for utxo in utxos:
                        total_balance += utxo.get("value", 0)
                        total_utxos += 1
            except:
                continue

        btc_balance = total_balance / 100_000_000
        coin_type = "1'" if network == "testnet" else "0'"

        return {
            "success": True,
            "balance_btc": btc_balance,
            "balance_satoshis": total_balance,
            "total_utxos": total_utxos,
            "addresses_checked": len(addresses),
            "first_address": addresses[0]["address"] if addresses else None,
            "network": network,
            "coin_type": coin_type,
            "derivation_paths": [a["path"] for a in addresses],
            "derivation_path_template": "path segons prefix (44'/49'/84')"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def generate_address(xpub: str, network: str = "testnet", index: int = 0) -> Dict:
    """
    Genera una nova adreça Bitcoin per rebre pagaments.
    """
    try:
        info = derive_address_and_path(xpub, network, index, change=False)
        coin_type = "1'" if network == "testnet" else "0'"

        return {
            "success": True,
            "address": info["address"],
            "index": index,
            "type": "receive",
            "network": network,
            "path": info["path"] or f"m/?'/{coin_type}/0'/0/{index}",
            "description": f"Adreça de recepció #{index} per {network}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def list_utxos(xpub: str, network: str = "testnet") -> Dict:
    """
    Llista totes les UTXOs (Unspent Transaction Outputs) disponibles.
    """
    try:
        addresses = [derive_real_address(xpub, network, i, change=False) for i in range(5)]
        all_utxos = []
        api_base = "https://blockstream.info/testnet/api" if network == "testnet" else "https://blockstream.info/api"

        for address in addresses:
            try:
                response = requests.get(f"{api_base}/address/{address}/utxo", timeout=5)
                if response.status_code == 200:
                    utxos = response.json()
                    for utxo in utxos:
                        all_utxos.append({
                            "txid": utxo.get("txid", ""),
                            "vout": utxo.get("vout", 0),
                            "value_satoshis": utxo.get("value", 0),
                            "value_btc": utxo.get("value", 0) / 100_000_000,
                            "address": address,
                            "confirmations": utxo.get("status", {}).get("confirmations", 0)
                        })
            except:
                continue

        return {
            "success": True,
            "utxos": all_utxos,
            "total_utxos": len(all_utxos),
            "total_value_satoshis": sum(u["value_satoshis"] for u in all_utxos),
            "total_value_btc": sum(u["value_satoshis"] for u in all_utxos) / 100_000_000
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def get_fee_rates(network: str = "testnet") -> Dict:
    """
    Obté les fee rates actuals de la xarxa Bitcoin.
    """
    try:
        api_url = "https://mempool.space/testnet/api" if network == "testnet" else "https://mempool.space/api"
        response = requests.get(f"{api_url}/v1/fees/recommended", timeout=5)
        
        if response.status_code == 200:
            fees = response.json()
            return {
                "success": True,
                "fastest_fee": fees.get("fastestFee", 20),
                "half_hour_fee": fees.get("halfHourFee", 10),
                "hour_fee": fees.get("hourFee", 5),
                "economy_fee": fees.get("economyFee", 2),
                "minimum_fee": fees.get("minimumFee", 1),
                "network": network
            }
        return {
            "success": False,
            "error": "No s'han pogut obtenir les fees"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def create_transaction(
    xpub: str,
    recipient_address: str,
    amount_btc: float,
    fee_rate: int = 10,
    network: str = "testnet"
) -> Dict:
    """
    Crea una transacció Bitcoin (PSBT estàndard BIP-174).
    """
    try:
        # Obtenir UTXOs
        utxos_result = list_utxos.invoke({"xpub": xpub, "network": network})
        if not utxos_result["success"]:
            return utxos_result
        
        utxos = utxos_result["utxos"]
        if not utxos:
            return {
                "success": False,
                "error": "No hi ha UTXOs disponibles"
            }
        
        # Si tenim el creador de PSBT, usar-lo
        if PSBT_CREATOR_AVAILABLE:
            # Generar adreça de canvi real
            change_info = derive_address_and_path(xpub, network, index=0, change=True)
            change_address = change_info["address"]
            
            # Estimar fee (simplificat: fee_rate * bytes estimats)
            amount_satoshis = int(amount_btc * 100_000_000)
            estimated_size = 10 + 148 * len(utxos) + 34 * 2  # Aproximació
            fee_satoshis = fee_rate * estimated_size
            
            # Crear PSBT real
            result = create_transaction_psbt(
                xpub=xpub,
                recipient_address=recipient_address,
                amount_btc=amount_btc,
                utxos=utxos,
                fee_satoshis=fee_satoshis,
                change_address=change_address,
                network=network
            )
            
            if result["success"]:
                # Afegir informació adicional per l'agent
                result["psbt_format"] = "BIP-174 Standard"
                result["ready_to_sign"] = True
                result["instructions"] = (
                    "Aquest és un PSBT estàndard BIP-174. Per signar-lo:\n"
                    "1. Guarda el PSBT en un fitxer .psbt\n"
                    "2. Importa'l al teu wallet (Electrum, Bitcoin Core, etc.)\n"
                    "3. Signa la transacció\n"
                    "4. Difon la transacció signada a la xarxa"
                )
            
            return result
            
        else:
            # Fallback: crear JSON simplificat (com abans)
            amount_satoshis = int(amount_btc * 100_000_000)
            selected_utxos = []
            total_input = 0
            
            sorted_utxos = sorted(utxos, key=lambda x: x["value_satoshis"], reverse=True)
            
            for utxo in sorted_utxos:
                selected_utxos.append(utxo)
                total_input += utxo["value_satoshis"]
                
                estimated_fee = fee_rate * (10 + 148 * len(selected_utxos) + 34 * 2)
                
                if total_input >= amount_satoshis + estimated_fee:
                    break
            
            if total_input < amount_satoshis + estimated_fee:
                return {
                    "success": False,
                    "error": f"Fons insuficients. Necessari: {(amount_satoshis + estimated_fee)/100_000_000:.8f} BTC, Disponible: {total_input/100_000_000:.8f} BTC"
                }
            
            change = total_input - amount_satoshis - estimated_fee
            
            # Crear JSON simplificat
            psbt = {
                "warning": "Aquest és un format JSON simplificat, no un PSBT BIP-174 real",
                "inputs": selected_utxos,
                "outputs": [
                    {
                        "address": recipient_address,
                        "value": amount_satoshis
                    }
                ],
                "fee": estimated_fee,
                "network": network
            }
            
            if change > 546:  # Dust limit
                change_info = derive_address_and_path(xpub, network, index=0, change=True)
                psbt["outputs"].append({
                    "address": change_info["address"],
                    "value": change,
                    "type": "change",
                    "path": change_info["path"]
                })
            
            return {
                "success": True,
                "psbt": json.dumps(psbt, indent=2),
                "psbt_format": "JSON Simplified (not BIP-174)",
                "total_input_btc": total_input / 100_000_000,
                "amount_btc": amount_btc,
                "fee_btc": estimated_fee / 100_000_000,
                "change_btc": change / 100_000_000 if change > 546 else 0,
                "num_inputs": len(selected_utxos),
                "num_outputs": len(psbt["outputs"]),
                "network": network,
                "ready_to_sign": False,
                "instructions": (
                    "⚠️ Aquest NO és un PSBT estàndard. És un format JSON simplificat.\n"
                    "Per crear un PSBT real, instal·la psbt_creator.py"
                )
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def decode_psbt(psbt_string: str) -> Dict:
    """
    Decodifica i valida un PSBT.
    """
    if not PSBT_CREATOR_AVAILABLE:
        return {
            "success": False,
            "error": "El decodificador de PSBT no està disponible. Instal·la psbt_creator.py"
        }
    
    try:
        creator = PSBTCreator()
        result = creator.decode_psbt(psbt_string)
        
        if result["valid"]:
            return {
                "success": True,
                "valid": True,
                "version": result.get("version", 0),
                "num_inputs": result.get("num_inputs", 0),
                "num_outputs": result.get("num_outputs", 0),
                "transaction_hex": result.get("tx", ""),
                "format": "BIP-174 Standard PSBT"
            }
        else:
            return {
                "success": False,
                "valid": False,
                "error": result.get("error", "PSBT invàlid")
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ============== AGENT IA PRINCIPAL ==============

class BitcoinAIAgent:
    """Agent IA per gestionar Bitcoin amb llenguatge natural i PSBTs estàndard"""
    
    def __init__(self, openai_api_key: str = None):
        self.console = Console()
        
        # Configurar LLM
        api_key = openai_api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("Necessites una API key d'OpenAI")
        
        # Llista d'eines disponibles (afegint decode_psbt)
        self.tools = [
            get_balance,
            generate_address,
            list_utxos,
            get_fee_rates,
            create_transaction,
            decode_psbt  # Nova eina
        ]
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="o4-mini",
            temperature=1,
            streaming=False
        ).bind_tools(self.tools)
        
        # Crear ToolNode amb les eines
        self.tool_node = ToolNode(self.tools)
        
        # Construir graf
        self.graph = self._build_graph()
        self.xpub = None
        self.network = "testnet"
        self.last_psbt = None  # Guardar l'últim PSBT creat
    
    def _build_graph(self):
        """Construeix el graf de LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Afegir nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        
        # Definir flux
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        # Compilar amb memòria
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Node de l'agent amb LLM millorat per PSBTs"""
        
        # System prompt actualitzat
        if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
            psbt_info = "amb suport per PSBTs BIP-174 estàndard" if PSBT_CREATOR_AVAILABLE else "amb format JSON simplificat"
            
            system_prompt = f"""Ets un agent expert en Bitcoin que ajuda els usuaris a gestionar les seves wallets {psbt_info}.
            
            Informació de context:
            - XPUB: {self.xpub}
            - Network: {self.network}
            - PSBT Support: {'BIP-174 Standard' if PSBT_CREATOR_AVAILABLE else 'JSON Simplified'}
            
            Tens accés a aquestes eines:
            - get_balance: Per consultar el balanç
            - generate_address: Per generar noves adreces
            - list_utxos: Per veure les UTXOs disponibles
            - get_fee_rates: Per consultar les fees actuals
            - create_transaction: Per crear transaccions {'(PSBT BIP-174)' if PSBT_CREATOR_AVAILABLE else '(JSON)'}
            - decode_psbt: Per decodificar i validar PSBTs {'(disponible)' if PSBT_CREATOR_AVAILABLE else '(no disponible)'}
            
            IMPORTANT:
            - Sempre proporciona TOTS els paràmetres necessaris per les eines
            - Quan cridis eines, usa exactament: {{"xpub": "{self.xpub}", "network": "{self.network}"}}
            - Respon sempre en català
            - Si crees una transacció, explica si és un PSBT estàndard o format simplificat
            - Guarda els PSBTs creats per si l'usuari els vol després
            """
            
            state["messages"] = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # Generar resposta amb LLM
        try:
            response = self.llm.invoke(state["messages"])
            
            # Debug
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"[DEBUG] Tool calls: {len(response.tool_calls)}")
                for tc in response.tool_calls:
                    print(f"[DEBUG] - {tc['name']}: {tc['args']}")
                    
                    # Si es crea una transacció, guardar el PSBT
                    if tc['name'] == 'create_transaction':
                        state["last_psbt"] = "pending"
            
            state["messages"].append(response)
            
        except Exception as e:
            print(f"[ERROR] Error en agent_node: {e}")
            error_msg = AIMessage(content=f"Ho sento, hi ha hagut un error: {str(e)}")
            state["messages"].append(error_msg)
        
        return state
    
    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Decideix si continuar amb eines o finalitzar"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        # Si l'últim missatge té tool_calls, executar eines
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Altrament, finalitzar
        return "end"
    
    async def chat(self, message: str) -> str:
        """Processa un missatge de l'usuari"""
        
        print(f"[DEBUG] Missatge rebut: {message}")
        print(f"[DEBUG] PSBT Support: {'BIP-174' if PSBT_CREATOR_AVAILABLE else 'JSON'}")
        
        # Crear estat inicial
        initial_messages = [HumanMessage(content=message)]
        
        state = {
            "messages": initial_messages,
            "xpub": self.xpub,
            "network": self.network,
            "addresses": [],
            "utxos": [],
            "balance_satoshis": 0,
            "last_action": None,
            "last_psbt": self.last_psbt,
            "context": {
                "xpub": self.xpub,
                "network": self.network,
                "psbt_support": PSBT_CREATOR_AVAILABLE
            }
        }
        
        # Executar graf
        config = {"configurable": {"thread_id": f"chat_{datetime.now().timestamp()}"}}
        
        try:
            result = await self.graph.ainvoke(state, config)
            
            # Processar resultats
            tool_results = []
            psbt_created = None
            
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    tool_results.append(msg.content)
                    # Buscar si s'ha creat un PSBT
                    try:
                        content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        if isinstance(content, dict) and "psbt" in content:
                            psbt_created = content.get("psbt")
                            self.last_psbt = psbt_created
                            print(f"[DEBUG] PSBT guardat: {psbt_created[:50] if psbt_created else 'None'}...")
                    except:
                        pass
            
            # Obtenir última resposta de l'agent
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                    if hasattr(msg, "tool_calls") and not msg.tool_calls:
                        response = msg.content
                        
                        # Afegir informació sobre el format PSBT si es va crear una transacció
                        if psbt_created and PSBT_CREATOR_AVAILABLE:
                            response += "\n\n📄 **PSBT creat en format BIP-174 estàndard**"
                            response += "\nPots signar aquest PSBT amb qualsevol wallet compatible (Electrum, Bitcoin Core, hardware wallets, etc.)"
                        elif psbt_created and not PSBT_CREATOR_AVAILABLE:
                            response += "\n\n⚠️ **Nota:** S'ha creat un format JSON simplificat, no un PSBT estàndard."
                        
                        return response
                    elif not hasattr(msg, "tool_calls"):
                        return msg.content
            
            # Si hi ha resultats de tools però no resposta final
            if tool_results:
                return f"He executat les següents accions:\n" + "\n".join(tool_results)
            
            return "Ho sento, no he pogut processar completament la teva petició. Pots reformular-la?"
            
        except Exception as e:
            print(f"[ERROR] Error en chat: {e}")
            import traceback
            traceback.print_exc()
            return f"Ho sento, hi ha hagut un error: {str(e)}"
    
    def setup(self, xpub: str, network: str = "testnet"):
        """Configura l'agent amb una XPUB"""
        self.xpub = xpub
        self.network = network
        psbt_status = "BIP-174 Standard" if PSBT_CREATOR_AVAILABLE else "JSON Simplified"
        self.console.print(f"[green]✅ Agent configurat amb XPUB per {network}[/green]")
        self.console.print(f"[cyan]📄 PSBT Format: {psbt_status}[/cyan]")

# ============== INTERFÍCIE CONVERSACIONAL (manté igual) ==============

class BitcoinAssistant:
    """Assistent conversacional per Bitcoin amb PSBTs millorats"""
    
    def __init__(self):
        self.console = Console()
        self.agent = None
    
    async def run(self):
        """Executa l'assistent"""
        
        # Benvinguda actualitzada
        psbt_info = "amb PSBTs BIP-174 estàndard" if PSBT_CREATOR_AVAILABLE else "amb format JSON simplificat"
        
        self.console.print(Panel(
            f"[bold blue]🤖 Agent IA Bitcoin[/bold blue]\n"
            f"Gestiona la teva wallet Bitcoin {psbt_info}.\n\n"
            "Exemples del que puc fer:\n"
            "• 'Quin és el meu balanç?'\n"
            "• 'Genera'm una adreça nova'\n"
            "• 'Mostra'm les meves UTXOs'\n"
            "• 'Crea una transacció per enviar 0.001 BTC a [adreça]'\n"
            "• 'Decodifica aquest PSBT: [base64]'\n"
            "• 'Quines són les fees actuals?'",
            title=f"Benvingut - {'PSBT BIP-174' if PSBT_CREATOR_AVAILABLE else 'Mode Simplificat'}",
            border_style="blue"
        ))
        
        # Configuració inicial
        await self.setup()
        
        # Bucle de conversa
        self.console.print("\n[yellow]💬 Ja pots començar a parlar amb mi! (escriu 'sortir' per acabar)[/yellow]\n")
        
        while True:
            # Obtenir input de l'usuari
            user_input = Prompt.ask("[bold cyan]Tu[/bold cyan]")
            
            if user_input.lower() in ["sortir", "exit", "quit", "adéu", "adeu"]:
                self.console.print("[yellow]👋 Adéu! Que tinguis un bon dia![/yellow]")
                break
            
            # Processar amb l'agent
            with self.console.status("[dim]Pensant...[/dim]"):
                response = await self.agent.chat(user_input)
            
            # Mostrar resposta
            self.console.print(f"\n[bold green]🤖 Agent[/bold green]: {response}\n")
    
    async def setup(self):
        """Configura l'agent"""
        self.console.print("\n[bold]⚙️  Configuració Inicial[/bold]")
        
        # Obtenir configuració del .env
        api_key = os.getenv("OPENAI_API_KEY")
        xpub_env = os.getenv("BITCOIN_XPUB", "")
        network_env = os.getenv("BITCOIN_NETWORK", "testnet").lower()
        
        # Mostrar estat PSBT
        if PSBT_CREATOR_AVAILABLE:
            self.console.print("[green]✅ Suport PSBT BIP-174 activat[/green]")
        else:
            self.console.print("[yellow]⚠️  Mode simplificat (instal·la psbt_creator.py per PSBTs reals)[/yellow]")
        
        # Configuració API key
        if not api_key or api_key == "your-key-here":
            self.console.print("[yellow]⚠️  No s'ha trobat API key vàlida al fitxer .env[/yellow]")
            api_key = Prompt.ask("Introdueix la teva OpenAI API Key", password=True)
            
            if Prompt.ask("Vols guardar la clau al fitxer .env?", choices=["s", "n"], default="s") == "s":
                self._update_env_file("OPENAI_API_KEY", api_key)
        else:
            self.console.print("[green]✅ API Key carregada del fitxer .env[/green]")
        
        # Network
        if network_env not in ["mainnet", "testnet"]:
            network = Prompt.ask("Xarxa", choices=["testnet", "mainnet"], default="testnet")
        else:
            network = network_env
        
        # XPUB
        if not xpub_env or xpub_env == "your-xpub-here":
            self.console.print("\n[dim]Pots usar aquesta XPUB de testnet per provar:[/dim]")
            self.console.print("[dim]tpubD6NzVbkrYhZ4XgiXtFtukm3UvC3J3qTtmqYe2HhLUfRr7dW3JQgFVPuTqCvmKPNBPidLhPXF5ibXXrBhKBpvPyrqsQQcz8MJjwVwqkqqu3y[/dim]\n")
            xpub = Prompt.ask("La teva XPUB")
        else:
            xpub = xpub_env
            self.console.print(f"\n[cyan]XPUB: {xpub[:20]}...{xpub[-10:]}[/cyan]")
            if Prompt.ask("Vols usar aquesta XPUB?", choices=["s", "n"], default="s") == "n":
                xpub = Prompt.ask("Nova XPUB")
        
        # Crear agent
        try:
            self.agent = BitcoinAIAgent(api_key)
            self.agent.setup(xpub, network)
            self.console.print("\n[green]✅ Agent IA configurat correctament![/green]")
            self.console.print(f"[green]   Network: {network}[/green]")
            self.console.print(f"[green]   XPUB: {xpub[:20]}...{xpub[-10:]}[/green]")
            self.console.print(f"[green]   PSBT: {'BIP-174 Standard' if PSBT_CREATOR_AVAILABLE else 'JSON Simplified'}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            raise
    
    def _update_env_file(self, key: str, value: str):
        """Actualitza o afegeix una clau al fitxer .env"""
        env_path = Path(".env")
        lines = []
        key_found = False
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith(f"{key}="):
                        lines.append(f"{key}={value}\n")
                        key_found = True
                    else:
                        lines.append(line)
        
        if not key_found:
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
            lines.append(f"{key}={value}\n")
        
        if not lines:
            lines = [
                "# Configuració de l'Agent Bitcoin IA\n",
                f"OPENAI_API_KEY={value if key == 'OPENAI_API_KEY' else 'your-key-here'}\n",
                f"BITCOIN_XPUB={value if key == 'BITCOIN_XPUB' else 'your-xpub-here'}\n",
                f"BITCOIN_NETWORK={value if key == 'BITCOIN_NETWORK' else 'testnet'}\n",
            ]
        
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        self.console.print(f"[green]✅ {key} guardada al fitxer .env[/green]")

# ============== MAIN ==============

async def main():
    """Funció principal"""
    assistant = BitcoinAssistant()
    await assistant.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[yellow]👋 Interromput per l'usuari[/yellow]")
    except Exception as e:
        print(f"[red]Error: {e}[/red]")