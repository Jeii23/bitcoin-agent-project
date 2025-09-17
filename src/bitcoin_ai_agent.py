#!/usr/bin/env python3
"""
Agent IA Bitcoin amb suport per PSBT BIP-174 estàndard
Versió millorada amb creació de transaccions reals
"""

import asyncio
import os
import json
import hashlib
import base64
import re
import base58
from typing import TypedDict, List, Dict, Optional, Literal, Annotated, Sequence
from decimal import Decimal
from datetime import datetime
import requests
from pathlib import Path
from typing import Tuple

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
from psbt_creator import PSBTCreator, create_transaction_psbt

# Per la interfície
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from rich.text import Text

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

    # ============== HELPER ==============

# Guarda l'últim missatge de l'usuari per poder corregir adreces mal parsejades pel LLM
_LAST_USER_UTTERANCE: str = ""
_LAST_RECIPIENT_ADDRESS: Optional[str] = None

# Regex per candidates d'adreça (bech32 o base58 típiques)
_ADDR_RE = re.compile(r"(bc1[0-9a-z]{8,}|tb1[0-9a-z]{8,}|[13][a-km-zA-HJ-NP-Z1-9]{25,})", re.IGNORECASE)

def _find_first_valid_address_in_text(text: str, network: str) -> Optional[str]:
    """Troba la primera adreça vàlida dins d'un text lliure validant checksum/format.

    Retorna l'adreça o None si no en troba cap vàlida.
    """
    if not text:
        return None
    creator = PSBTCreator(network=network)
    for m in _ADDR_RE.finditer(text):
        cand = m.group(0)
        try:
            # Validació forta via decodificador d'adreces
            _ = creator._decode_address(cand)
            return cand
        except Exception:
            continue
    return None


def _normalize_address(addr: Optional[str]) -> Optional[str]:
    """Best-effort normalization for user/LLM-provided addresses.

    - Strips surrounding whitespace and common quotes/brackets.
    - Removes zero-width and non-breaking spaces.
    - Removes any embedded whitespace characters.
    - Returns None if input is falsy.
    """
    if not addr:
        return addr
    s = str(addr)
    # Strip surrounding whitespace and common punctuation wrappers
    s = s.strip().strip("’‘'\"“”<>[](){}.,:;`“”")
    # Remove zero-width and NBSP variants
    for ch in ("\u200b", "\ufeff", "\u2060", "\u00A0"):
        s = s.replace(ch.encode('utf-8').decode('unicode_escape'), "") if ch.startswith("\\u") else s.replace(ch, "")
    # Collapse/remove any whitespace inside
    s = re.sub(r"\s+", "", s)
    return s


def _sanitize_psbt_b64(b64_text: str) -> str:
    """Sanitize a PSBT base64 string by removing whitespace and validating magic bytes.

    Returns the cleaned single-line base64. Raises ValueError if decode fails or magic missing.
    """
    if not isinstance(b64_text, str):
        raise ValueError("PSBT must be a base64 string")
    # Remove all whitespace/newlines
    cleaned = re.sub(r"\s+", "", b64_text)
    # Validate by decoding and checking magic "psbt\x00"
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except Exception as e:
        # Some providers wrap without padding; try add padding
        pad = (-len(cleaned)) % 4
        try:
            raw = base64.b64decode(cleaned + ("=" * pad), validate=False)
        except Exception:
            raise ValueError(f"Invalid PSBT base64: {e}")
    if not (len(raw) >= 5 and raw[:5] == b"psbt\xff"):
        raise ValueError("Decoded PSBT missing magic 'psbt\\xff'")
    return cleaned


def _replace_psbt_in_text(text: str, clean_psbt: str) -> str:
    """Replace any PSBT-like base64 blobs in free-form text with the provided clean one-liner.

    Note: Historically, the LLM could emit a PSBT inline followed immediately by natural text (e.g., '==és').
    To avoid copy/paste issues, first strip any PSBT-like blobs entirely and later append a standardized
    section with a single clean PSBT on its own line.
    """
    if not text:
        return text
    pattern = re.compile(r"cHNidP[0-9A-Za-z+/=\s]+")
    # Remove all PSBT-like blobs to prevent duplicates or adjacent-text artifacts
    return pattern.sub("[PSBT ocultada; consulta la secció estandarditzada de PSBT més avall]", text)

def _count_psbt_blobs(text: str) -> int:
    """Counts occurrences of PSBT-like base64 blobs in text."""
    if not text:
        return 0
    return len(re.findall(r"cHNidP[0-9A-Za-z+/=\s]+", text))


def _address_has_history(address: str, network: str) -> bool:
    """Best-effort check of whether an address has ever received funds.

    Previous behaviour on network error returned True (conservative) which caused
    skipping fresh indexes when the API had transient issues. We now:
      1. Try /address for funded_txo_count.
      2. Fallback to /address/<addr>/utxo (detects only current UTXOs).
      3. On total failure, return False so address can still be offered; this
         restores prior UX where first unused appeared earlier instead of jumping ahead.
    """
    api_base = "https://blockstream.info/testnet/api" if network == "testnet" else "https://blockstream.info/api"
    try:
        r = requests.get(f"{api_base}/address/{address}", timeout=6)
        if r.status_code == 200:
            info = r.json() or {}
            chain = info.get("chain_stats", {}) or {}
            memp = info.get("mempool_stats", {}) or {}
            funded = chain.get("funded_txo_count", 0) + memp.get("funded_txo_count", 0)
            return funded > 0
        # fallback lightweight
        r2 = requests.get(f"{api_base}/address/{address}/utxo", timeout=4)
        if r2.status_code == 200:
            return len(r2.json()) > 0
        return False
    except Exception:
        # Network uncertainty: treat as unused (False) to avoid skipping deterministic indices.
        return False


# ================= SHARED UTXO SCAN HELPERS =================

def _api_base(network: str) -> str:
    return "https://blockstream.info/testnet/api" if network == "testnet" else "https://blockstream.info/api"


def _fetch_address_utxos(address: str, network: str, timeout: float = 5.0, retries: int = 2) -> List[Dict]:
    """Fetch UTXOs for a single address with small retry budget.

    Returns empty list if all attempts fail (never raises) to keep deterministic higher-level code.
    """
    base = _api_base(network)
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{base}/address/{address}/utxo", timeout=timeout)
            if r.status_code == 200:
                data = r.json() or []
                if isinstance(data, list):
                    return data
            # Non-200: keep trying
        except Exception as e:  # network / JSON
            last_error = e
        # tiny backoff (no asyncio so simple sleep) only if more attempts left
        if attempt < retries:
            try:
                import time
                time.sleep(0.15 * (attempt + 1))
            except Exception:
                pass
    # All failed
    return []


def _fetch_address_utxos_status(address: str, network: str, timeout: float = 5.0, retries: int = 2) -> Tuple[List[Dict], bool]:
    """Variant returning (utxos, ok_flag) where ok_flag=True only if an HTTP 200 was received.

    Needed so list_utxos can distinguish between 'truly empty' and network failure for diagnostics.
    """
    base = _api_base(network)
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{base}/address/{address}/utxo", timeout=timeout)
            if r.status_code == 200:
                data = r.json() or []
                return (data if isinstance(data, list) else [], True)
        except Exception:
            pass
        if attempt < retries:
            try:
                import time
                time.sleep(0.15 * (attempt + 1))
            except Exception:
                pass
    return ([], False)


def _enumerate_addresses(xpub: str, network: str, receive_limit: int, change_limit: int) -> List[Dict]:
    """Derive a deterministic set of candidate addresses (receive + change)."""
    addresses: List[Dict] = []
    for change_flag, limit in ((False, receive_limit), (True, change_limit)):
        for i in range(limit):
            info = derive_address_and_path(xpub, network, i, change=change_flag)
            addresses.append({
                "address": info["address"],
                "path": info.get("path", ""),
                "index": i,
                "change": change_flag,
            })
    return addresses





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
    last_error: Optional[str] = None
    # 1) Derivació personalitzada de l'usuari
    if CUSTOM_DERIVATION_AVAILABLE:
        try:
            res = derive_bitcoin_address(xpub, index=index, change=change, network=network)
            if res.get("success"):
                return {"address": res["address"], "path": res.get("path", "")}
            else:
                print(f"[WARN] Derivació personalitzada ha fallat: {res.get('error')}")
                last_error = res.get("error") or last_error
        except Exception as e:
            print(f"[ERROR] Derivació personalitzada: {e}")
            last_error = str(e)

    # 2) Fallback: HDWallet simple
    # 2) Fallback: HDWallet simple
    if HDWALLET_AVAILABLE:
        try:
            # ✨ NORMALITZA SLIP-132 (vpub/upub → tpub ; zpub/ypub → xpub)
            try:
                from address_derivation import _normalize_to_x_or_t_pub as _norm
                normalized = _norm(xpub)
            except Exception:
                normalized = xpub  # si no hi ha helper, prova igual

            hdwallet = HDWallet(cryptocurrency=Bitcoin)
            hdwallet.from_xpublic_key(xpublic_key=normalized, strict=False)
            chain = 1 if change else 0
            hdwallet.clean_derivation()
            hdwallet.from_path(path=f"m/{chain}/{index}")
            address = hdwallet.p2wpkh_address()
            coin_type = "1'" if network == "testnet" else "0'"
            path = f"m/84'/{coin_type}/0'/{chain}/{index}"
            return {"address": address, "path": path}
        except Exception as e:
            print(f"[ERROR] Fallback HDWallet: {e}")
            last_error = str(e)
    # 3) Ja no generem adreces sintètiques: llança error clar
    raise ValueError(f"Address derivation failed (no synthetic fallback). Last error: {last_error or 'unknown'}")

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
        # Reuse the exact same enumeration & fetching strategy as list_utxos to avoid discrepancies.
        scan_receive = 10
        scan_change = 5
        addresses = _enumerate_addresses(xpub, network, scan_receive, scan_change)
        utxos: List[Dict] = []
        for entry in addresses:
            for utxo in _fetch_address_utxos(entry["address"], network):
                utxos.append({
                    "txid": utxo.get("txid", ""),
                    "vout": utxo.get("vout", 0),
                    "value_satoshis": utxo.get("value", 0),
                    "value_btc": utxo.get("value", 0) / 100_000_000,
                    "address": entry["address"],
                    "path": entry["path"],
                    "index": entry["index"],
                    "change": entry["change"],
                    "confirmations": utxo.get("status", {}).get("confirmations", 0),
                })
        total_balance = sum(u["value_satoshis"] for u in utxos)
        btc_balance = total_balance / 100_000_000
        coin_type = "1'" if network == "testnet" else "0'"
        return {
            "success": True,
            "balance_btc": btc_balance,
            "balance_satoshis": total_balance,
            "total_utxos": len(utxos),
            "utxos": utxos,
            "addresses": addresses,
            "addresses_checked": len(addresses),
            "network": network,
            "coin_type": coin_type,
            "derivation_paths": [a["path"] for a in addresses],
            "derivation_path_template": "path segons prefix (44'/49'/84')",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def generate_address(xpub: str, network: str = "testnet", index: int = 0, require_unused: bool = True, max_scan: int = 200) -> Dict:
    """
    Genera una adreça de recepció. Per defecte, escaneja des de `index`
    fins trobar la primera adreça SENSE ús (cap funded_txo, gastat o no).
    - require_unused=True: obliga que l'adreça no tingui historial
    - max_scan: límit de seguretat per no escanejar infinitament
    """
    try:
        coin_type = "1'" if network == "testnet" else "0'"

        # Si no exigim "unused", manté el comportament antic
        if not require_unused:
            info = derive_address_and_path(xpub, network, index, change=False)
            return {
                "success": True,
                "address": info["address"],
                "index": index,
                "type": "receive",
                "network": network,
                "path": info["path"] or f"m/84'/{coin_type}/0'/0/{index}",
                "description": f"Adreça de recepció #{index} per {network}"
            }

        # 🔎 Cerca la primera adreça sense historial (m/.../0/i)
        tested = []
        for i in range(index, index + max_scan):
            info = derive_address_and_path(xpub, network, i, change=False)
            addr = info["address"]
            used = _address_has_history(addr, network)
            tested.append({"index": i, "address": addr, "used": used})
            if not used:
                return {
                    "success": True,
                    "address": addr,
                    "index": i,
                    "type": "receive",
                    "network": network,
                    "path": info["path"] or f"m/84'/{coin_type}/0'/0/{i}",
                    "description": f"Adreça de recepció (primera sense ús) #{i} per {network}",
                    "checked": tested  # opcional: útil per debugging
                }

        return {
            "success": False,
            "error": f"No s'ha trobat cap adreça sense ús en {max_scan} intents a partir de l'índex {index} (m/.../0/i).",
            "checked": tested
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def list_utxos(xpub: str, network: str = "testnet") -> Dict:
    """
    Llista totes les UTXOs (Unspent Transaction Outputs) disponibles.
    """
    try:
        scan_receive = 10
        scan_change = 5
        address_entries = _enumerate_addresses(xpub, network, scan_receive, scan_change)

        all_utxos: List[Dict] = []
        network_errors = 0
        for entry in address_entries:
            utxos_raw, ok = _fetch_address_utxos_status(entry["address"], network)
            if not ok:
                network_errors += 1
            for utxo in utxos_raw:
                all_utxos.append({
                    "txid": utxo.get("txid", ""),
                    "vout": utxo.get("vout", 0),
                    "value_satoshis": utxo.get("value", 0),
                    "value_btc": utxo.get("value", 0) / 100_000_000,
                    "address": entry["address"],
                    "path": entry["path"],
                    "index": entry["index"],
                    "change": entry["change"],
                    "confirmations": utxo.get("status", {}).get("confirmations", 0),
                })

        total_value_sat = sum(u["value_satoshis"] for u in all_utxos)
        result = {
            "success": True,
            "utxos": all_utxos,
            "total_utxos": len(all_utxos),
            "total_value_satoshis": total_value_sat,
            "total_value_btc": total_value_sat / 100_000_000,
            "receive_addresses_checked": scan_receive,
            "change_addresses_checked": scan_change,
            "network": network,
        }
        if len(all_utxos) == 0 and network_errors == len(address_entries):
            result["note"] = "Cap UTXO retornada i totes les consultes han fallat (possible problema de xarxa o rate-limit)."
            result["network_errors"] = network_errors
        return result
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
        # Safety clamp for pathological fee rates (LLM might pick fastestFee overly high on testnet)
        original_fee_rate = fee_rate
        fee_notes: List[str] = []
        # Hard upper cap
        if fee_rate > 150:
            fee_notes.append(f"Fee rate {fee_rate} sat/vB clamped to 150 sat/vB to avoid excessive fees.")
            fee_rate = 150
        # Additional proportional cap for very small sends (avoid >15% fee)
        amount_sats = int(round(amount_btc * 100_000_000))
        # Rough minimal vbytes (1in/2out segwit) ~140 => projected fee = fee_rate*140
        projected_fee = fee_rate * 140
        if projected_fee > amount_sats * 0.15:
            # Lower fee_rate so projected fee ~10% of amount (still generous); floor 2 sat/vB
            target_fee = max(int(amount_sats * 0.10), 200)  # at least 200 sats so not absurdly low
            new_rate = max(2, min(fee_rate, target_fee // 140))
            if new_rate < fee_rate:
                fee_notes.append(
                    f"Fee rate reduced from {fee_rate} to {new_rate} sat/vB to keep fee <=10% of amount (projected)."
                )
                fee_rate = new_rate

        # Si l'usuari respon amb confirmació sense repetir l'adreça, reutilitza la darrera coneguda
        if not recipient_address:
            global _LAST_RECIPIENT_ADDRESS
            if isinstance(_LAST_RECIPIENT_ADDRESS, str) and _LAST_RECIPIENT_ADDRESS:
                recipient_address = _LAST_RECIPIENT_ADDRESS
        # Correcció defensiva de l'adreça de destí si el LLM o l'input CLI l'ha corromput
        try:
            recipient_address = _normalize_address(recipient_address) or recipient_address
            PSBTCreator(network=network)._decode_address(recipient_address)
        except Exception:
            # Intenta extreure una adreça vàlida del darrer missatge de l'usuari
            cand = _find_first_valid_address_in_text(_LAST_USER_UTTERANCE, network)
            if cand:
                recipient_address = _normalize_address(cand) or cand
            else:
                return {
                    "success": False,
                    "error": f"Invalid recipient/change address: {recipient_address} (cap adreça vàlida detectada al missatge de l'usuari)",
                }

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
        # Generar adreça de canvi real
        change_info = derive_address_and_path(xpub, network, index=0, change=True)
        change_address = change_info["address"]

        # Crear PSBT real (estimació de comissió a psbt_creator amb els UTXOs seleccionats)
        result = create_transaction_psbt(
            xpub=xpub,
            recipient_address=recipient_address,
            amount_btc=amount_btc,
            utxos=utxos,
            change_address=change_address,
            network=network,
            fee_rate=fee_rate,
        )
        
        if result["success"]:
            # Afegir informació adicional per l'agent
            result["psbt_format"] = "BIP-174 Standard"
            result["ready_to_sign"] = True
            result["instructions"] = (
                "Aquest és un PSBT estàndard BIP-174. Per signar-lo:\n"
                "1. Guarda el PSBT en un fitxer .psbt\n"
                "2. Importa'l al teu wallet (Electrum, Bitcoin Core, etc.)\n"
                "3. Signa i broadcasteja la transacció"
            )
            # Annotate fee moderation if applied
            if fee_notes:
                result["fee_policy_notes"] = fee_notes
            if original_fee_rate != fee_rate:
                result["original_fee_rate"] = original_fee_rate
                result["effective_fee_rate"] = fee_rate
            else:
                result["effective_fee_rate"] = fee_rate
            return result
        else:
            return result
        
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

@tool
def create_transaction_manual(
    xpub: str,
    recipient_address: str,
    amount_btc: float,
    utxo_ids: List[str],
    fee_rate: int = 10,
    network: str = "testnet",
    change_index: int = 0,
) -> Dict:
    """Crea una PSBT triant manualment UTXOs (format utxo_ids: "<txid>:<vout>").

    1. Llista UTXOs via list_utxos.
    2. Filtra les que coincideixen amb utxo_ids.
    3. Passa-les a create_transaction_psbt com manual_selected_utxos.
    """
    try:
        # Si l'usuari confirma sense repetir l'adreça, reutilitza la darrera coneguda
        if not recipient_address:
            global _LAST_RECIPIENT_ADDRESS
            if isinstance(_LAST_RECIPIENT_ADDRESS, str) and _LAST_RECIPIENT_ADDRESS:
                recipient_address = _LAST_RECIPIENT_ADDRESS
        # Correcció defensiva de l'adreça de destí si el LLM o l'input CLI l'ha corromput
        try:
            recipient_address = _normalize_address(recipient_address) or recipient_address
            PSBTCreator(network=network)._decode_address(recipient_address)
        except Exception:
            cand = _find_first_valid_address_in_text(_LAST_USER_UTTERANCE, network)
            if cand:
                recipient_address = _normalize_address(cand) or cand
            else:
                return {
                    "success": False,
                    "error": f"Invalid recipient/change address: {recipient_address} (cap adreça vàlida detectada al missatge de l'usuari)",
                }

        utxos_result = list_utxos.invoke({"xpub": xpub, "network": network})
        if not utxos_result.get("success"):
            return utxos_result
        all_utxos = utxos_result.get("utxos", [])
        wanted = set(utxo_ids)
        selected = [u for u in all_utxos if f"{u.get('txid')}:{u.get('vout')}" in wanted]
        missing = wanted - {f"{u.get('txid')}:{u.get('vout')}" for u in selected}
        if not selected:
            # Fallback: si l'LLM ha proporcionat UTXOs inexistents, reverteix a selecció automàtica
            auto = create_transaction_psbt(
                xpub=xpub,
                recipient_address=recipient_address,
                amount_btc=amount_btc,
                utxos=all_utxos,
                change_address=derive_address_and_path(xpub, network, index=change_index, change=True)["address"],
                network=network,
                fee_rate=fee_rate,
            )
            if auto.get("success"):
                auto["selection_mode"] = "auto_fallback"
                auto["requested_utxos"] = utxo_ids
                auto["missing"] = list(missing)
            return auto

        # Derivar adreça de canvi (canvi index configurable)
        change_info = derive_address_and_path(xpub, network, index=change_index, change=True)
        change_address = change_info["address"]

        build = create_transaction_psbt(
            xpub=xpub,
            recipient_address=recipient_address,
            amount_btc=amount_btc,
            utxos=all_utxos,
            change_address=change_address,
            network=network,
            fee_rate=fee_rate,
            manual_selected_utxos=selected,
        )
        if build.get("success"):
            build["selection_mode"] = "manual"
            build["requested_utxos"] = utxo_ids
            build["used_utxos"] = [f"{u.get('txid')}:{u.get('vout')}" for u in selected]
            if missing:
                build["missing"] = list(missing)
        return build
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============== AGENT IA PRINCIPAL ==============

class BitcoinAIAgent:
    """Agent IA per gestionar Bitcoin amb llenguatge natural i PSBTs estàndard"""
    
    def __init__(self, openai_api_key: str = None):
        self.console = Console()
        # Historial de conversa (memòria simple en procés). Inclou SystemMessage + intercanvis.
        self._history: List[BaseMessage] = []
        # Límit màxim de missatges a retenir per evitar creixement indefinit
        self._history_limit = 40
        self._memory_enabled = True
        # Guarda la darrera directiva/instrucció explícita de l'usuari
        self._last_directive: Optional[str] = None

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
            create_transaction_manual,
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
            psbt_info = "amb suport per PSBTs BIP-174 estàndard"
            
            system_prompt = f"""Ets un agent expert en Bitcoin que ajuda els usuaris a gestionar les seves wallets {psbt_info}.
            
            Informació de context:
            - XPUB: {self.xpub}
            - Network: {self.network}
            - PSBT Support: {'BIP-174 Standard'}
            
            Tens accés a aquestes eines:
            - get_balance: Per consultar el balanç
            - generate_address: Per generar noves adreces
            - list_utxos: Per veure les UTXOs disponibles
            - get_fee_rates: Per consultar les fees actuals
            - create_transaction: Per crear transaccions {'(PSBT BIP-174)'}
            - create_transaction_manual: Per crear una transacció indicant EXACTAMENT quines UTXOs (format txid:vout) i una fee rate
            - decode_psbt: Per decodificar i validar PSBTs {'(disponible)'}
            
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
                    # Persistir la darrera adreça de destí proposada per reús en confirmacions
                    try:
                        if tc['name'] in ('create_transaction', 'create_transaction_manual'):
                            args = tc.get('args') or {}
                            ra = args.get('recipient_address')
                            if ra:
                                global _LAST_RECIPIENT_ADDRESS
                                _LAST_RECIPIENT_ADDRESS = _normalize_address(ra) or ra
                    except Exception:
                        pass
            
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
        global _LAST_USER_UTTERANCE
        _LAST_USER_UTTERANCE = message
        lower = message.lower().strip()
        # Heurística per detectar directiva (imperativa) i guardar-la
        directive_prefixes = [
            "crea ", "create ", "fes ", "envia ", "send ", "deriva ", "genera ",
            "construeix ", "build ", "calcula ", "calcular ", "necessito ", "vull "
        ]
        if any(lower.startswith(p) for p in directive_prefixes) or "maximitzar la privacitat" in lower:
            self._last_directive = message

        # Consultes de recordatori / memòria
        recall_triggers = [
            "que he dit abans", "què he dit abans", "que t'he dit abans", "recordes quina es la directiva",
            "quina es la directiva", "what did i say before", "remember what i asked"
        ]
        if any(t in lower for t in recall_triggers):
            if not self._memory_enabled:
                return "La memòria està desactivada actualment."
            # Preparar resposta
            if self._last_directive:
                answer = f"La darrera directiva registrada és: '{self._last_directive}'"
            else:
                # Recuperar últimes 3 aportacions humanes
                recent_humans = [m for m in self._history if isinstance(m, HumanMessage)][-3:]
                if recent_humans:
                    texts = [m.content for m in recent_humans]
                    answer = "Resum de les teves últimes indicacions: " + " | ".join(texts)
                else:
                    answer = "Encara no tinc cap directiva emmagatzemada."
            # Actualitzar memòria manualment (afegint aquest intercanvi) perquè bypassaríem el graf
            if self._memory_enabled:
                self._history.append(HumanMessage(content=message))
                self._history.append(AIMessage(content=answer))
                self._history = self._history[-self._history_limit:]
            return answer

        print(f"[DEBUG] Missatge rebut: {message}")
        print(f"[DEBUG] PSBT Support: {'BIP-174'}")
        
        # Construir missatges inicials amb memòria (si activada)
        if self._memory_enabled and self._history:
            initial_messages = list(self._history) + [HumanMessage(content=message)]
        else:
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
                "psbt_support": True
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
                    try:
                        print(f"[DEBUG] ToolMessage content type: {type(msg.content)}")
                        preview = str(msg.content)
                        if len(preview) > 200:
                            preview = preview[:200] + "..."
                        print(f"[DEBUG] ToolMessage content preview: {preview}")
                    except Exception:
                        pass
                    # Buscar si s'ha creat un PSBT
                    try:
                        # Normalitza contingut de ToolMessage (pot ser str JSON, dict o llista segmentada)
                        raw = msg.content
                        content = json.loads(raw) if isinstance(raw, str) else raw
                        def _extract_psbt(obj) -> Optional[str]:
                            if isinstance(obj, dict) and "psbt" in obj:
                                return obj.get("psbt")
                            # Alguns frameworks encapsulen en claus com 'content' o 'json'
                            if isinstance(obj, dict):
                                for k in ("content", "json", "data"):
                                    v = obj.get(k)
                                    if isinstance(v, str):
                                        try:
                                            jd = json.loads(v)
                                            if isinstance(jd, dict) and "psbt" in jd:
                                                return jd.get("psbt")
                                        except Exception:
                                            continue
                                    elif isinstance(v, dict) and "psbt" in v:
                                        return v.get("psbt")
                            if isinstance(obj, list):
                                for it in obj:
                                    r = _extract_psbt(it)
                                    if r:
                                        return r
                            return None
                        found = _extract_psbt(content)
                        if found:
                            psbt_created = found
                            self.last_psbt = psbt_created
                            # Evita imprimir el PSBT complet per pantalla per no forçar salts de línia
                            try:
                                _pv = psbt_created if isinstance(psbt_created, str) else str(psbt_created)
                                if len(_pv) > 120:
                                    _pv = _pv[:120] + "…"
                                print(f"[DEBUG] PSBT guardat (preview): {_pv}")
                            except Exception:
                                pass
                    except:
                        pass
            
            # Obtenir última resposta de l'agent
            final_response: Optional[str] = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                    if hasattr(msg, "tool_calls") and not msg.tool_calls:
                        final_response = msg.content
                        if psbt_created:
                            # Netejar qualsevol espai/blanc accidental i validar via decodificador intern
                            clean_psbt = _sanitize_psbt_b64(psbt_created)
                            try:
                                dec = PSBTCreator(network=self.network).decode_psbt(clean_psbt)
                            except Exception as _e_dec:
                                dec = {"valid": False, "error": str(_e_dec)}

                            if not dec.get("valid", False):
                                # No publiquem una PSBT potencialment corrupta; oferim diagnòstic
                                final_response = _replace_psbt_in_text(final_response, "")
                                final_response += "\n\n⚠️ No s'ha pogut validar la PSBT generada amb el decodificador intern."
                                final_response += f"\nDetalls: {dec.get('error', 'desconegut')}\n"
                            else:
                                # Assegura que qualsevol bloc anterior (multi-línia) al text es retirat
                                final_response = _replace_psbt_in_text(final_response, clean_psbt)
                                try:
                                    # Escriure arxius per ús fàcil a wallets
                                    Path("psbt_latest.psbt").write_bytes(base64.b64decode(clean_psbt))
                                    Path("psbt_latest.base64").write_text(clean_psbt)
                                except Exception as _e:
                                    print(f"[WARN] No s'ha pogut escriure fitxer PSBT: {_e}")

                                # Afegeix una secció estandarditzada, amb la PSBT aïllada en línia pròpia i línia en blanc després
                                final_response += "\n\n📄 **PSBT creat en format BIP-174 estàndard**"
                                final_response += "\nPots signar aquest PSBT amb qualsevol wallet compatible (Electrum, Bitcoin Core, hardware wallets, etc.)"
                                final_response += "\nElectrum: usa Tools > Load transaction > From text i enganxa el PSBT en Base64, o File > Open/Load transaction > From file amb psbt_latest.psbt (no el peguis com a HEX)."
                                final_response += "\nFitxers guardats: psbt_latest.psbt (binari) i psbt_latest.base64 (text)."
                                final_response += "\n\nPSBT (Base64 una sola línia, sense salts):\n" + clean_psbt + "\n\n"
                                # Debug defensiu: assegurar que només queda una PSBT al text final
                                try:
                                    cnt = _count_psbt_blobs(final_response)
                                    if cnt > 1:
                                        print(f"[WARN] S'han detectat {cnt} PSBTs al text final; això no hauria de passar")
                                except Exception:
                                    pass
                        break
                    elif not hasattr(msg, "tool_calls"):
                        final_response = msg.content
                        break

            # Actualitzar memòria (historial) amb els missatges produïts pel graf (abans de retornar)
            if self._memory_enabled:
                # El graf retorna tots els messages; mantenim només els últims fins límit
                # Evitem duplicar exactament la mateixa seqüència (si es repeteix per algun motiu)
                new_messages = result.get("messages", [])
                if new_messages:
                    # Substituir historial complet per l'últim estat (més senzill i robust)
                    self._history = list(new_messages)[-self._history_limit:]
            # Si tenim resposta final, retornar-la ara
            if final_response is not None:
                return final_response
            
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
        psbt_status = "BIP-174 Standard"
        self.console.print(f"[green]✅ Agent configurat amb XPUB per {network}[/green]")
        self.console.print(f"[cyan]📄 PSBT Format: {psbt_status}[/cyan]")
        # Netejar historial per nova sessió
        self._history.clear()

    def clear_memory(self):
        """Buida la memòria conversacional."""
        self._history.clear()

    def disable_memory(self):
        """Desactiva l'acumulació de memòria (les interaccions passen a ser stateless)."""
        self._memory_enabled = False
        self.clear_memory()

    def enable_memory(self):
        """Activa la memòria si estava desactivada."""
        self._memory_enabled = True

# ============== INTERFÍCIE CONVERSACIONAL (manté igual) ==============

class BitcoinAssistant:
    """Assistent conversacional per Bitcoin amb PSBTs millorats"""
    
    def __init__(self):
        self.console = Console()
        self.agent = None
    
    async def run(self):
        """Executa l'assistent"""
        
        # Benvinguda actualitzada
        psbt_info = "amb PSBTs BIP-174 estàndard"
        
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
            title=f"Benvingut - {'PSBT BIP-174'}",
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

            # Mostrar resposta: assegurem que la línia PSBT es mostra sense wrapping dur
            try:
                marker = "PSBT (Base64 una sola línia, sense salts):"
                if isinstance(response, str) and marker in response:
                    head, tail = response.split(marker, 1)
                    # capçalera + marcador
                    self.console.print(f"\n[bold green]🤖 Agent[/bold green]: {head}{marker}")
                    # primera línia no buida després del marcador és la PSBT
                    tail_lines = tail.splitlines()
                    psbt_line = ""
                    rem_start = 0
                    for i, ln in enumerate(tail_lines):
                        if ln.strip():
                            psbt_line = ln.strip()
                            rem_start = i + 1
                            break
                    if psbt_line:
                        # sanitzar i validar mínimament, però si falla imprimeix tal qual
                        try:
                            psbt_line = _sanitize_psbt_b64(psbt_line)
                        except Exception:
                            pass
                        # Imprimir en brut a stdout per evitar qualsevol format/espai injectat per la consola
                        try:
                            print(psbt_line, flush=True)
                            # línia en blanc per separar de la resta i evitar que Rich afegeixi padding a la mateixa línia
                            print("")
                        except Exception:
                            # fallback a rich si hi ha cap problema amb stdout
                            self.console.print(Text(psbt_line, no_wrap=True, overflow="ignore"))
                    remainder = "\n".join(tail_lines[rem_start:])
                    if remainder:
                        # imprimiu la resta en brut també
                        print(remainder + "\n")
                    else:
                        self.console.print("")
                else:
                    self.console.print(f"\n[bold green]🤖 Agent[/bold green]: {response}\n")
            except Exception:
                self.console.print(f"\n[bold green]🤖 Agent[/bold green]: {response}\n")
    
    async def setup(self):
        """Configura l'agent"""
        self.console.print("\n[bold]⚙️  Configuració Inicial[/bold]")
        
        # Obtenir configuració del .env
        api_key = os.getenv("OPENAI_API_KEY")
        xpub_env = os.getenv("BITCOIN_XPUB", "")
        network_env = os.getenv("BITCOIN_NETWORK", "testnet").lower()
        
        # Mostrar estat PSBT
        self.console.print("[green]✅ Suport PSBT BIP-174 activat[/green]")

        
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
            self.console.print(f"[green]   PSBT: {'BIP-174 Standard'}[/green]")
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