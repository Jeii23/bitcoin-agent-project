#!/usr/bin/env python3
"""
Bitcoin AI Agent with BIP-174 PSBT support.

Refactored for cleaner structure, proper logging, and reduced code duplication.
"""

import asyncio
import os
import json
import base64
import re
import logging
import time
from typing import TypedDict, List, Dict, Optional, Literal, Annotated, Sequence, Tuple
from decimal import Decimal
from datetime import datetime
import requests
from pathlib import Path


# Carregar variables d'entorn des del fitxer .env
from dotenv import load_dotenv
load_dotenv()

# ============== LOGGING CONFIGURATION ==============
# Configura el nivell de logging via variable d'entorn LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)
# Per defecte és INFO. Usa LOG_LEVEL=DEBUG per veure tot el que fa l'agent.
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Silencia els logs sorollosos de biblioteques externes (només mostrem els nostres)
for _noisy in ("httpcore", "httpx", "openai", "anthropic", "urllib3", "requests", "asyncio", "langchain", "langgraph", "google"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# LangChain i LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Optional LLM providers (imported lazily to avoid hard dependencies)
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ChatAnthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    ChatGoogleGenerativeAI = None
    GOOGLE_GENAI_AVAILABLE = False

# HDWallet for correct derivation
try:
    from hdwallet import HDWallet
    from hdwallet.cryptocurrencies import Bitcoin
    HDWALLET_AVAILABLE = True
except ImportError:
    HDWALLET_AVAILABLE = False

# Custom address derivation module
try:
    from address_derivation import derive_bitcoin_address, _normalize_to_x_or_t_pub
    CUSTOM_DERIVATION_AVAILABLE = True
except ImportError:
    CUSTOM_DERIVATION_AVAILABLE = False
    _normalize_to_x_or_t_pub = None

# PSBT Creator
from psbt_creator import PSBTCreator, create_transaction_psbt
from addressing import detect_address_type

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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_XPUB = os.getenv("BITCOIN_XPUB", "")
DEFAULT_NETWORK = os.getenv("BITCOIN_NETWORK", "testnet").lower()
# Note: Removed legacy privacy_preset handling; behavior now configured only via explicit flags.

# LLM Provider configuration
# Set LLM_PROVIDER to "openai", "anthropic", or "google" to switch models
# Set LLM_MODEL to override the default model for the provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")  # Empty = use provider default

# Validar network default
if DEFAULT_NETWORK not in ["mainnet", "testnet"]:
    DEFAULT_NETWORK = "testnet"

# Module-level logger
logger = logging.getLogger(__name__)


# ============== LLM FACTORY ==============

# Supported LLM providers with their default models and configurations
LLM_PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o3", "o1"],
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-20250514",
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "env_key": "ANTHROPIC_API_KEY",
    },
    "google": {
        "default_model": "gemini-1.5-pro",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-2.0-flash-exp"],
        "env_key": "GOOGLE_API_KEY",
    },
}


def create_llm(
    provider: str = None,
    model: str = None,
    api_key: str = None,
    temperature: float = 1.0,
    tools: List = None,
) -> "BaseChatModel":
    """
    Factory function to create an LLM instance with tool binding.

    Args:
        provider: LLM provider ("openai", "anthropic", "google"). Defaults to LLM_PROVIDER env.
        model: Specific model name. Defaults to LLM_MODEL env or provider's default.
        api_key: API key override. Falls back to environment variable.
        temperature: Temperature setting for generation.
        tools: List of tools to bind to the model.

    Returns:
        A LangChain chat model with tools bound.

    Raises:
        ValueError: If provider is unsupported or required dependencies are missing.

    Examples:
        # Use environment configuration
        llm = create_llm(tools=my_tools)

        # Explicitly select Claude
        llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022", tools=my_tools)

        # Use Gemini with specific API key
        llm = create_llm(provider="google", api_key="my-key", tools=my_tools)
    """
    # Resolve provider from env if not specified
    provider = (provider or LLM_PROVIDER).lower()

    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported: {list(LLM_PROVIDERS.keys())}"
        )

    config = LLM_PROVIDERS[provider]
    model = model or LLM_MODEL or config["default_model"]

    # Resolve API key
    if not api_key:
        api_key = os.getenv(config["env_key"])
        if not api_key:
            raise ValueError(
                f"Missing API key for {provider}. "
                f"Set {config['env_key']} environment variable or pass api_key parameter."
            )

    logger.info(f"Creating LLM: provider={provider}, model={model}")

    # Create the appropriate LLM instance
    if provider == "openai":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            streaming=False,
        )
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ValueError(
                "Anthropic (Claude) support requires langchain-anthropic. "
                "Install with: pip install langchain-anthropic"
            )
        llm = ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
    elif provider == "google":
        if not GOOGLE_GENAI_AVAILABLE:
            raise ValueError(
                "Google Gemini support requires langchain-google-genai. "
                "Install with: pip install langchain-google-genai"
            )
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
        )
    else:
        # This shouldn't happen given the earlier check, but be defensive
        raise ValueError(f"Provider '{provider}' not implemented")

    # Bind tools if provided
    if tools:
        llm = llm.bind_tools(tools)

    return llm


def get_available_providers() -> Dict[str, Dict]:
    """
    Returns information about available LLM providers and their configuration status.

    Returns:
        Dict with provider info including availability and configured status.
    """
    result = {}
    for name, config in LLM_PROVIDERS.items():
        has_key = bool(os.getenv(config["env_key"]))
        if name == "anthropic":
            available = ANTHROPIC_AVAILABLE
        elif name == "google":
            available = GOOGLE_GENAI_AVAILABLE
        else:
            available = True  # OpenAI always available via langchain-openai

        result[name] = {
            "available": available,
            "configured": has_key,
            "default_model": config["default_model"],
            "models": config["models"],
            "env_key": config["env_key"],
            "status": "ready" if (available and has_key) else (
                "missing_key" if available else "missing_package"
            ),
        }
    return result


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


def _extract_text_content(content) -> str:
    """Extract plain text from LLM response content.

    Handles different response formats:
    - str: returned as-is
    - list of dicts with 'text' key (Gemini format): concatenates all text parts
    - list of dicts with 'type'='text' (OpenAI/Anthropic format): extracts text
    - other: converts to string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                # Gemini/OpenAI format: {'type': 'text', 'text': '...'}
                if 'text' in item:
                    texts.append(str(item['text']))
                elif 'content' in item:
                    texts.append(str(item['content']))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else str(content)
    return str(content)


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


def _provider_sequence(network: str):
    """Return ordered list of (provider_name, base) based on BLOCKCHAIN_API env.
    Si hi ha credencials RPC configurades i no s’ha especificat res, prefereix Core.

    Supported values:
      - "core" | "bitcoin-core" | "rpc": Bitcoin Core JSON-RPC
      - "blockstream": Blockstream REST (fallback mempool.space)
      - "mempool": Mempool.space REST (fallback Blockstream)
      - other/"auto": default to Blockstream then Mempool
    """
    raw_pref = os.getenv("BLOCKCHAIN_API")
    if raw_pref is None:
        # Si l’usuari té RPC configurat, assumeix Core
        if all(os.getenv(k) for k in ("BITCOIN_RPC_USER","BITCOIN_RPC_PASSWORD","BITCOIN_RPC_HOST","BITCOIN_RPC_PORT")):
            pref = "core"
        else:
            pref = "auto"
    else:
        pref = raw_pref.lower()
    bs = ("blockstream", "https://blockstream.info/testnet/api" if network == "testnet" else "https://blockstream.info/api")
    mp = ("mempool", "https://mempool.space/testnet/api" if network == "testnet" else "https://mempool.space/api")
    if pref in ("core", "bitcoin-core", "rpc"):
        return [("core", "rpc")]
    if pref == "mempool":
        return [mp, bs]
    # default and "blockstream"
    return [bs, mp]


# ================= Bitcoin Core RPC helpers =================

def _core_rpc_endpoint() -> str:
    """Build Bitcoin Core RPC URL. Supports /wallet/<name> via BITCOIN_RPC_WALLET env."""
    from urllib.parse import quote
    user = os.getenv("BITCOIN_RPC_USER")
    pwd = os.getenv("BITCOIN_RPC_PASSWORD")
    host = os.getenv("BITCOIN_RPC_HOST")
    port = os.getenv("BITCOIN_RPC_PORT")
    # URL-encode user and password to handle special characters like @, !, etc.
    user_enc = quote(user, safe="") if user else ""
    pwd_enc = quote(pwd, safe="") if pwd else ""
    base = f"http://{user_enc}:{pwd_enc}@{host}:{port}"
    wallet = os.getenv("BITCOIN_RPC_WALLET") or os.getenv("BITCOIN_WALLET") or os.getenv("BITCOIN_CORE_WALLET")
    return f"{base}/wallet/{wallet}" if wallet else base


def _core_rpc_call(method: str, params: List) -> Dict:
    """Execute a JSON-RPC call to Bitcoin Core."""
    headers = {"content-type": "application/json"}
    payload = {"jsonrpc": "1.0", "id": "agent", "method": method, "params": params}
    url = _core_rpc_endpoint()
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"RPC {method} error: {j['error']}")
    return j.get("result")


def _core_scantxoutset(addresses: List[str]) -> List[Dict]:
    """Query Core via scantxoutset for UTXOs belonging to addresses (no wallet import required)."""
    scanobjects = [f"addr({a})" for a in addresses]
    res = _core_rpc_call("scantxoutset", ["start", scanobjects]) or {}
    utxos: List[Dict] = []
    for u in (res.get("unspents") or []):
        btc_amount = float(u.get("amount", 0.0))
        utxos.append({
            "txid": u.get("txid"),
            "vout": u.get("vout"),
            "value": int(round(btc_amount * 100_000_000)),
            "address": (u.get("desc", "").split("(")[-1].rstrip(")")) or None,
            "height": u.get("height"),
        })
    return utxos


def _core_getreceivedbyaddress(address: str, minconf: int = 0) -> Optional[Decimal]:
    """Retorna el total rebut (en BTC) segons la wallet de Core, o None si l’adreça no és a la wallet."""
    try:
        amt = _core_rpc_call("getreceivedbyaddress", [address, minconf])
        return Decimal(str(amt))
    except Exception:
        # Errors típics: adreça no present a la wallet (-5), wallet sense descriptors, etc.
        return None


def _address_has_history(address: str, network: str) -> bool:
    """Comprova si una adreça ha rebut mai fons.
    Amb Core: primer prova la wallet (getreceivedbyaddress). Si és desconeguda, cau a scantxoutset.
    Amb REST: usa funded_txo_count com abans."""
    providers = _provider_sequence(network)
    if providers and providers[0][0] == "core":
        try:
            amt = _core_getreceivedbyaddress(address, 0)
            if amt is not None:
                return amt > 0
            utxos = _core_scantxoutset([address])
            return len(utxos) > 0
        except Exception:
            return False
    for _name, base in providers:
        try:
            r = requests.get(f"{base}/address/{address}", timeout=6)
            if r.status_code == 200:
                info = r.json()
                chain_funded = (info.get("chain_stats", {}) or {}).get("funded_txo_count", 0)
                mempool_funded = (info.get("mempool_stats", {}) or {}).get("funded_txo_count", 0)
                if (chain_funded or 0) > 0 or (mempool_funded or 0) > 0:
                    return True
            r2 = requests.get(f"{base}/address/{address}/utxo", timeout=6)
            if r2.status_code == 200:
                utxos = r2.json() or []
                if isinstance(utxos, list) and len(utxos) > 0:
                    return True
        except Exception:
            continue
    return False


# ================= UTXO FETCH HELPERS =================

def _fetch_address_utxos(address: str, network: str, timeout: float = 5.0, retries: int = 2) -> List[Dict]:
    """Fetch UTXOs for a single address with small retry budget.

    Returns empty list if all attempts fail (never raises) to keep deterministic higher-level code.
    """
    providers = _provider_sequence(network)
    # Bitcoin Core path
    if providers and providers[0][0] == "core":
        try:
            core_utxos = _core_scantxoutset([address])
            # Normalize to REST-like schema for callers
            for u in core_utxos:
                if "value" not in u and "amount" in u:
                    u["value"] = int(round(float(u.get("amount", 0.0)) * 100_000_000))
            return core_utxos
        except Exception:
            return []
    # REST providers
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        for _name, base in providers:
            try:
                r = requests.get(f"{base}/address/{address}/utxo", timeout=timeout)
                if r.status_code == 200:
                    data = r.json() or []
                    if isinstance(data, list):
                        return data
                # Non-200: try next provider
            except Exception as e:  # network / JSON
                last_error = e
        # tiny backoff
        if attempt < retries:
            try:
                import time
                time.sleep(0.15 * (attempt + 1))
            except Exception:
                pass
    # All failed
    return []


def _fetch_address_utxos_status(address: str, network: str, timeout: float = 5.0, retries: int = 2) -> Tuple[List[Dict], bool, Optional[str]]:
    """Variant returning (utxos, ok_flag, provider_used).

    ok_flag=True only if an HTTP 200 was received by any provider.
    provider_used is the name of the provider that returned 200, else None.
    """
    providers = _provider_sequence(network)
    # Bitcoin Core path
    if providers and providers[0][0] == "core":
        try:
            core_utxos = _core_scantxoutset([address])
            for u in core_utxos:
                if "value" not in u and "amount" in u:
                    u["value"] = int(round(float(u.get("amount", 0.0)) * 100_000_000))
            return (core_utxos, True, "core")
        except Exception:
            return ([], False, "core")
    # REST providers
    for attempt in range(retries + 1):
        for name, base in providers:
            try:
                r = requests.get(f"{base}/address/{address}/utxo", timeout=timeout)
                if r.status_code == 200:
                    data = r.json() or []
                    return (data if isinstance(data, list) else [], True, name)
            except Exception:
                pass
        if attempt < retries:
            try:
                import time
                time.sleep(0.15 * (attempt + 1))
            except Exception:
                pass
    return ([], False, None)

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

# ============== ADDRESS DERIVATION ==============

def derive_address_and_path(xpub: str, network: str, index: int, change: bool = False) -> Dict:
    """Derive address + path. Prioritizes custom derivation, falls back to HDWallet."""
    last_error: Optional[str] = None
    
    # 1) Custom derivation
    if CUSTOM_DERIVATION_AVAILABLE:
        try:
            res = derive_bitcoin_address(xpub, index=index, change=change, network=network)
            if res.get("success"):
                return {"address": res["address"], "path": res.get("path", "")}
            last_error = res.get("error") or last_error
            logger.debug("Custom derivation returned error: %s", last_error)
        except Exception as e:
            logger.debug("Custom derivation exception: %s", e)
            last_error = str(e)

    # 2) Fallback: HDWallet
    if HDWALLET_AVAILABLE:
        try:
            # Normalize SLIP-132 extended keys (vpub/upub → tpub ; zpub/ypub → xpub)
            normalized = _normalize_to_x_or_t_pub(xpub) if _normalize_to_x_or_t_pub else xpub
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
            logger.debug("HDWallet fallback exception: %s", e)
            last_error = str(e)
    
    # No synthetic address generation
    raise ValueError(f"Address derivation failed. Last error: {last_error or 'unknown'}")


def derive_real_address(xpub: str, network: str, index: int, change: bool = False) -> str:
    """Compatibility wrapper for derive_address_and_path."""
    return derive_address_and_path(xpub, network, index, change)["address"]


# ============== AGENT TOOLS ==============

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
def verify_utxo(txid: str, network: str = "testnet") -> Dict:
    """Verifica si una transacció existeix i retorna els seus outputs i estat usant exploradors amb fallback.

    Retorna:
      - success: bool
      - provider: quin proveïdor ha respost
      - tx: informació bàsica (si disponible)
      - vout: llista d'outputs amb valor i adreça (si proporcionat pel proveïdor)
      - outspends: estat gastat per a cada vout si l'API ho permet
    """
    providers = _provider_sequence(network)
    last_error: Optional[str] = None
    for name, base in providers:
        try:
            # Nota: Blockstream: /tx/{txid}, /tx/{txid}/outspends; Mempool: /tx/{txid} i /tx/{txid}/outspends (similars)
            r = requests.get(f"{base}/tx/{txid}", timeout=7)
            if r.status_code != 200:
                continue
            tx = r.json() or {}
            # Outputs
            vout = tx.get("vout") or tx.get("outs") or []
            # Outspends (gastats?)
            outspends = None
            try:
                r2 = requests.get(f"{base}/tx/{txid}/outspends", timeout=7)
                if r2.status_code == 200:
                    outspends = r2.json()
            except Exception:
                pass
            return {
                "success": True,
                "provider": name,
                "txid": txid,
                "tx": tx,
                "vout": vout,
                "outspends": outspends,
                "network": network,
            }
        except Exception as e:
            last_error = str(e)
            continue
    return {"success": False, "error": last_error or "No provider returned tx info", "txid": txid, "network": network}

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
        # Allow environment overrides for scan limits to accommodate wallets with larger gaps
        try:
            scan_receive = int(os.getenv("BITCOIN_SCAN_RECEIVE", "10"))
        except Exception:
            scan_receive = 10
        try:
            scan_change = int(os.getenv("BITCOIN_SCAN_CHANGE", "5"))
        except Exception:
            scan_change = 5
        address_entries = _enumerate_addresses(xpub, network, scan_receive, scan_change)

        all_utxos: List[Dict] = []
        network_errors = 0
        providers_used: List[str] = []
        for entry in address_entries:
            _ret = _fetch_address_utxos_status(entry["address"], network)
            # Back-compat: allow tests to monkeypatch a 2-tuple (utxos, ok)
            if isinstance(_ret, tuple) and len(_ret) == 3:
                utxos_raw, ok, prov = _ret
            elif isinstance(_ret, tuple) and len(_ret) == 2:
                utxos_raw, ok = _ret
                prov = None
            else:
                utxos_raw, ok, prov = [], False, None
            if not ok:
                network_errors += 1
            if ok and prov:
                providers_used.append(prov)
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

        # If we found nothing and the network appeared healthy, attempt an extended scan once
        attempted_extended = False
        if not all_utxos and network_errors == 0:
            attempted_extended = True
            ext_receive = max(scan_receive, 50)
            ext_change = max(scan_change, 20)
            # Enumerate only the additional range beyond the original
            extra_addrs = _enumerate_addresses(xpub, network, ext_receive, ext_change)[len(address_entries):]
            for entry in extra_addrs:
                _ret = _fetch_address_utxos_status(entry["address"], network)
                if isinstance(_ret, tuple) and len(_ret) == 3:
                    utxos_raw, ok, prov = _ret
                elif isinstance(_ret, tuple) and len(_ret) == 2:
                    utxos_raw, ok = _ret
                    prov = None
                else:
                    utxos_raw, ok, prov = [], False, None
                if not ok:
                    network_errors += 1  # count failures even in extended scan
                if ok and prov:
                    providers_used.append(prov)
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
            "provider_primary": _provider_sequence(network)[0][0],
            "provider_any_used": list(sorted(set(providers_used))) if providers_used else [],
        }
        if len(all_utxos) == 0 and network_errors == len(address_entries):
            result["note"] = "Cap UTXO retornada i totes les consultes han fallat (possible problema de xarxa o rate-limit)."
            result["network_errors"] = network_errors
        elif len(all_utxos) == 0 and attempted_extended and network_errors == 0:
            # Extended scan attempted but still no UTXOs; inform user about possible gap limit issues
            result["note"] = (
                "Cap UTXO trobada després d'un escaneig ampliat (possible gap superior al llindar). "
                "Pots ajustar els límits amb BITCOIN_SCAN_RECEIVE i BITCOIN_SCAN_CHANGE."
            )
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
    amount_sats: int,
    utxo_ids: List[str],
    fee_rate: int = 10,                  # mantingut per compatibilitat; no s'usa
    network: str = "testnet",
    change_index: int = 0,
    tx_opts: Optional[Dict] = None,      # knobs explícits i sense política
    psbt_extras: Optional[Dict] = None,  # claus PSBT addicionals
    locktime_override: Optional[int] = None,  # NOVA: permet passar locktime directament
) -> Dict:
    """
    Create a fully customizable BIP-174 PSBT with manual UTXO selection and maximum freedom.

    This tool gives the agent COMPLETE CONTROL over every aspect of the PSBT:
    - Choose exactly which UTXOs to spend (order preserved)
    - Define any outputs (addresses OR raw scripts)
    - Set transaction-level parameters (locktime, version, RBF)
    - Inject arbitrary PSBT metadata (BIP32 paths, Taproot data, proprietary fields)
    - No automatic coin selection, fee estimation, or privacy heuristics

    ══════════════════════════════════════════════════════════════════════════════
    PARAMETERS
    ══════════════════════════════════════════════════════════════════════════════

    xpub: str
        Extended public key for the wallet (used for address derivation)

    recipient_address: str
        Destination address (only used in "simple mode" when tx_opts.outputs is not provided)

    amount_sats: int
        Amount to send in satoshis (only used in "simple mode")

    utxo_ids: List[str]
        List of UTXOs to spend, format: ["txid:vout", ...]
        ORDER IS PRESERVED in the final transaction inputs!

    network: str = "testnet"
        "testnet" or "mainnet"

    change_index: int = 0
        Index for deriving change address (only if change_address_override not set)

    locktime_override: Optional[int] = None
        Set nLockTime directly (overrides tx_opts.locktime_override)

    ══════════════════════════════════════════════════════════════════════════════
    tx_opts: Dict - TRANSACTION OPTIONS
    ══════════════════════════════════════════════════════════════════════════════

    EXPLICIT OUTPUTS MODE (recommended for full control):
      tx_opts.outputs: List[Dict]
          Define ALL outputs explicitly. Each output dict:
          - {"address": "tb1q...", "value": 50000}           # by address
          - {"script_hex": "76a914...88ac", "value": 50000}  # by raw scriptPubKey
          - {"script": <bytes>, "value": 50000}              # by bytes
          Fee is IMPLICIT = sum(inputs) - sum(outputs)

    SIMPLE MODE (single recipient + explicit fee):
      tx_opts.fee_satoshis: int (REQUIRED if not using outputs mode)
          Explicit fee in satoshis. Change is auto-calculated.
      tx_opts.change_address_override: str
          Force a specific change address instead of deriving one

    INPUT OVERRIDES:
      tx_opts.inputs_overrides: Dict[str, Dict]
          Per-input customization keyed by "txid:vout". Each override dict can contain:
          - "sequence": int          # Custom nSequence (e.g., 0xFFFFFFFD for RBF)
          - "scriptpubkey_hex": str  # Override scriptPubKey
          - "address": str           # Override address for derivation
          - "witness_utxo": {"value_satoshis": int, "scriptpubkey_hex": str}
          - "non_witness_utxo_hex": str  # Full previous transaction hex

    TRANSACTION FLAGS:
      tx_opts.rbf: bool
          Enable Replace-By-Fee (sets nSequence <= 0xFFFFFFFD)
      tx_opts.locktime_override: int
          Set nLockTime for the transaction
      tx_opts.include_global_xpub: bool
          Include GLOBAL_XPUB in PSBT for hardware wallet compatibility
      tx_opts.include_keypaths: bool
          Include BIP32 derivation paths for inputs/outputs
      tx_opts.prefer_legacy_witness_utxo: bool
          Prefer NON_WITNESS_UTXO over WITNESS_UTXO when possible

    ══════════════════════════════════════════════════════════════════════════════
    psbt_extras: Dict - ADVANCED PSBT METADATA INJECTION
    ══════════════════════════════════════════════════════════════════════════════

    This allows injecting BIP-174/BIP-371 fields into the PSBT.
    NOTE: PSBT version is ALWAYS 0 (BIP-174). Version 2 (BIP-370) is NOT supported
    by most wallets (Electrum, hardware wallets, etc.) and will be ignored.

    psbt_extras.global: Dict
        - "fallback_locktime": int          # BIP-370 fallback locktime (ignored in v0)
        - "tx_modifiable": int              # BIP-370 modifiable flags (ignored in v0)
        - "proprietary": List[Dict]         # Vendor-specific data
            [{"prefix_hex": "...", "subtype": int, "keydata_hex": "...", "value_hex": "..."}]
        - "raw_kv": List[Dict]              # Arbitrary key-value pairs
            [{"key_hex": "...", "value_hex": "..."}]

    psbt_extras.inputs: List[Dict]
        Per-input PSBT fields (list index corresponds to input index):
        - "sighash_type": int               # 1=ALL, 2=NONE, 3=SINGLE, +0x80=ANYONECANPAY
        - "redeem_script_hex": str          # P2SH redeem script
        - "witness_script_hex": str         # P2WSH witness script
        - "por_commitment_hex": str         # Proof-of-Reserves commitment
        - "ripemd160_preimage_hex": str     # RIPEMD160 preimage
        - "sha256_preimage_hex": str        # SHA256 preimage
        - "hash160_preimage_hex": str       # HASH160 preimage
        - "hash256_preimage_hex": str       # HASH256 preimage
        - "bip32_derivations": List[Dict]   # BIP32 paths
            [{"pubkey_hex": "...", "fingerprint_hex": "...", "path_indices": [2147483732, ...]}]

        TAPROOT FIELDS (BIP-371):
        - "tap_key_sig_hex": str            # Key-path signature (64 or 65 bytes)
        - "tap_internal_key_hex": str       # 32-byte x-only internal key
        - "tap_merkle_root_hex": str        # 32-byte Merkle root of script tree
        - "tap_leaf_scripts": List[Dict]    # Script-path leaves
            [{"control_block_hex": "...", "script_hex": "...", "leaf_version": 0xC0}]
        - "tap_script_sigs": List[Dict]     # Script-path signatures
            [{"xonly_pubkey_hex": "...", "leaf_hash_hex": "...", "signature_hex": "..."}]
        - "tap_bip32_derivations": List[Dict]  # Taproot BIP32 derivations
            [{"xonly_pubkey_hex": "...", "leaf_hashes_hex": ["..."], "fingerprint_hex": "...", "path_indices": [...]}]

        - "proprietary": List[Dict]         # Same format as global
        - "raw_kv": List[Dict]              # Same format as global

    psbt_extras.outputs: List[Dict]
        Per-output PSBT fields (list index corresponds to output index):
        - "bip32_derivations": List[Dict]   # Same format as inputs
        - "redeem_script_hex": str          # P2SH output script
        - "witness_script_hex": str         # P2WSH output script

        TAPROOT FIELDS (BIP-371):
        - "tap_internal_key_hex": str       # 32-byte x-only internal key
        - "tap_tree": List[Dict]            # Taproot script tree
            [{"depth": int, "leaf_version": 0xC0, "script_hex": "..."}]
        - "tap_bip32_derivations": List[Dict]  # Same format as inputs

        - "proprietary": List[Dict]         # Same format as global
        - "raw_kv": List[Dict]              # Same format as global

    ══════════════════════════════════════════════════════════════════════════════
    EXAMPLES
    ══════════════════════════════════════════════════════════════════════════════

    1. Simple send with explicit fee:
       create_transaction_manual(
           xpub="tpub...",
           recipient_address="tb1q...",
           amount_sats=50000,
           utxo_ids=["abc123...:0"],
           tx_opts={"fee_satoshis": 500}
       )

    2. Multi-output with RBF and locktime:
       create_transaction_manual(
           xpub="tpub...",
           recipient_address="",  # ignored in outputs mode
           amount_sats=0,          # ignored in outputs mode
           utxo_ids=["abc123...:0", "def456...:1"],
           tx_opts={
               "outputs": [
                   {"address": "tb1qrecipient1...", "value": 25000},
                   {"address": "tb1qrecipient2...", "value": 25000},
                   {"address": "tb1qchange...", "value": 49000}  # explicit change
               ],
               "rbf": True,
               "locktime_override": 850000
           }
       )

    3. Taproot script-path output:
       create_transaction_manual(
           ...,
           psbt_extras={
               "outputs": [{
                   "tap_internal_key_hex": "a]b2c3...32bytes...",
                   "tap_tree": [
                       {"depth": 0, "leaf_version": 192, "script_hex": "20ab...ac"}
                   ]
               }]
           }
       )

    4. OP_RETURN output (arbitrary data):
       create_transaction_manual(
           ...,
           tx_opts={
               "outputs": [
                   {"address": "tb1qrecipient...", "value": 50000},
                   {"script_hex": "6a0f68656c6c6f20626974636f696e", "value": 0}  # OP_RETURN "hello bitcoin"
               ],
               "fee_satoshis": 300  # only needed for simple mode, here fee is implicit
           }
       )

    ══════════════════════════════════════════════════════════════════════════════
    RETURNS
    ══════════════════════════════════════════════════════════════════════════════

    Success:
        {
            "success": True,
            "psbt": "cHNidP8B...",           # Base64 PSBT ready for signing
            "psbt_hex": "70736274...",       # Hex PSBT
            "num_inputs": int,
            "num_outputs": int,
            "fee_sats": int,                  # Computed as inputs - outputs
            "total_input_sats": int,
            "selection_mode": "manual",
            "used_utxos": ["txid:vout", ...],
            "outputs_detail": [...]
        }

    Failure:
        {"success": False, "error": "description..."}
    """
    try:
        # 0) Normalització d’adreça si s’usarà el mode simple
        use_outputs_mode = bool(tx_opts and isinstance(tx_opts.get("outputs"), list))
        if not use_outputs_mode:
            if not recipient_address:
                global _LAST_RECIPIENT_ADDRESS
                if isinstance(_LAST_RECIPIENT_ADDRESS, str) and _LAST_RECIPIENT_ADDRESS:
                    recipient_address = _LAST_RECIPIENT_ADDRESS
            recipient_address = _normalize_address(recipient_address) or recipient_address
            PSBTCreator(network=network)._decode_address(recipient_address)  # valida

        # 1) Llistar UTXOs i seleccionar EN L’ORDRE demanat
        utxos_result = list_utxos.invoke({"xpub": xpub, "network": network})
        if not utxos_result.get("success"):
            return utxos_result
        all_utxos = utxos_result.get("utxos", [])

        # Índex ràpid txid:vout -> utxo
        index = {f"{u.get('txid')}:{u.get('vout')}": u for u in all_utxos}

        # Mantén ordre i detecta absents
        selected = []
        missing = []
        for uid in utxo_ids:
            u = index.get(uid)
            if u is None:
                missing.append(uid)
            else:
                selected.append(dict(u))  # còpia mutable per aplicar overrides

        if not selected:
            return {"success": False, "error": "Cap UTXO coincideix amb els 'utxo_ids' demanats.", "missing": missing}

        # 2) Sobrescriptures per-input (sequence, scriptpubkey_hex, witness_utxo, etc.)
        inputs_overrides = (tx_opts or {}).get("inputs_overrides") if isinstance(tx_opts, dict) else None
        if isinstance(inputs_overrides, dict):
            for i, u in enumerate(selected):
                uid = f"{u.get('txid')}:{u.get('vout')}"
                ov = inputs_overrides.get(uid)
                if isinstance(ov, dict):
                    selected[i].update(ov)

        # 3) Decideix mode de sortides
        kwargs = {}
        # Propaga només knobs explícits i no-biaix
        if isinstance(tx_opts, dict):
            passthrough_keys = {
                "rbf",
                "locktime_override",
                "include_global_xpub",
                "include_keypaths",
                "prefer_legacy_witness_utxo",
            }
            for k in passthrough_keys:
                if k in tx_opts:
                    kwargs[k] = tx_opts[k]
        # Si es passa locktime_override top-level, sobreescriu el de tx_opts
        if locktime_override is not None:
            kwargs["locktime_override"] = int(locktime_override)

        # Injecció PSBT addicional (filtrant camps no suportats)
        if psbt_extras is not None:
            # Filter out unsupported PSBT version - only v0 (BIP-174) is supported
            # PSBT v2 (BIP-370) is rejected by most wallets including Electrum
            filtered_extras = dict(psbt_extras)
            if isinstance(filtered_extras.get("global"), dict):
                filtered_global = dict(filtered_extras["global"])
                if "version" in filtered_global:
                    logger.warning("PSBT version parameter ignored - only v0 (BIP-174) is supported")
                    del filtered_global["version"]
                filtered_extras["global"] = filtered_global
            kwargs["psbt_extras"] = filtered_extras

        creator = PSBTCreator(network=network)  # només per validar adreces si cal
        total_input = sum(int(u.get("value_satoshis", 0)) for u in selected)

        # 3.a) Mode explícit amb 'outputs'
        if use_outputs_mode:
            outs = tx_opts["outputs"]
            # Validació mínima i normalització d'outputs per adreça
            normalized_outs = []
            for o in outs:
                if "value" not in o:
                    return {"success": False, "error": "Cada output necessita 'value' en satoshis."}
                out = {"value": int(o["value"])}
                if "address" in o and o["address"] is not None:
                    addr = str(o["address"]).strip()
                    creator._decode_address(addr)  # valida
                    out["address"] = addr
                elif "script" in o and o["script"] is not None:
                    out["script"] = o["script"]
                elif "script_hex" in o and o["script_hex"] is not None:
                    out["script_hex"] = o["script_hex"]
                else:
                    return {"success": False, "error": "Cada output ha de tenir 'address' o 'script/script_hex'."}
                normalized_outs.append(out)

            # Cap fee automàtica: la fee queda implícita en inputs − outputs
            build = create_transaction_psbt(
                xpub=xpub,
                utxos=all_utxos,
                outputs=normalized_outs,
                manual_selected_utxos=selected,
                network=network,
                **kwargs,
            )

        # 3.b) Mode simple: un sol destinatari + fee explícita + canvi opcional
        else:
            fee_satoshis = None
            if isinstance(tx_opts, dict) and "fee_satoshis" in tx_opts:
                fee_satoshis = int(tx_opts["fee_satoshis"])

            if fee_satoshis is None:
                return {
                    "success": False,
                    "error": "En mode simple cal 'fee_satoshis' explícit a tx_opts. No s’estimen comissions.",
                }

            # Derivació o override d’adreça de canvi si calgués
            change_address = None
            if isinstance(tx_opts, dict) and tx_opts.get("change_address_override"):
                change_address = str(tx_opts["change_address_override"]).strip()
                creator._decode_address(change_address)
            else:
                change_info = derive_address_and_path(xpub, network, index=change_index, change=True)
                change_address = change_info["address"]

            build = create_transaction_psbt(
                xpub=xpub,
                utxos=all_utxos,
                recipient_address=recipient_address,
                amount_sats=int(amount_sats),
                change_address=change_address,
                network=network,
                manual_selected_utxos=selected,
                fee_satoshis=int(fee_satoshis),
                **kwargs,
            )

        # 4) Decoració del resultat
        if build.get("success"):
            build["selection_mode"] = "manual"
            build["requested_utxos"] = utxo_ids
            build["used_utxos"] = [f"{u.get('txid')}:{u.get('vout')}" for u in selected]
            if missing:
                build["missing"] = list(missing)
            if isinstance(tx_opts, dict):
                applied = {k: v for k, v in tx_opts.items() if k in {
                    "rbf", "locktime_override", "include_global_xpub", "include_keypaths",
                    "prefer_legacy_witness_utxo", "fee_satoshis", "outputs", "change_address_override", "inputs_overrides"
                }}
                if applied:
                    build["applied_tx_opts"] = applied
            if psbt_extras is not None:
                build["applied_psbt_extras"] = True

        return build

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============== AGENT IA PRINCIPAL ==============

class BitcoinAIAgent:
    """Agent IA per gestionar Bitcoin amb llenguatge natural i PSBTs estàndard.
    
    Supports multiple LLM providers: OpenAI (GPT), Anthropic (Claude), Google (Gemini).
    Configure via environment variables or constructor parameters.
    """
    
    def __init__(
        self,
        openai_api_key: str = None,
        *,
        llm_provider: str = None,
        llm_model: str = None,
        api_key: str = None,
        temperature: float = 1.0,
    ):
        """
        Initialize the Bitcoin AI Agent.

        Args:
            openai_api_key: Legacy parameter for backward compatibility (deprecated).
            llm_provider: LLM provider ("openai", "anthropic", "google"). 
                         Defaults to LLM_PROVIDER env var or "openai".
            llm_model: Specific model name. Defaults to LLM_MODEL env var or provider's default.
            api_key: API key for the selected provider. Falls back to env vars.
            temperature: LLM temperature setting (default 1.0).

        Environment variables:
            LLM_PROVIDER: Default provider ("openai", "anthropic", "google")
            LLM_MODEL: Default model name
            OPENAI_API_KEY: API key for OpenAI
            ANTHROPIC_API_KEY: API key for Anthropic (Claude)
            GOOGLE_API_KEY: API key for Google (Gemini)

        Examples:
            # Use default (reads from env)
            agent = BitcoinAIAgent()
            
            # Use Claude explicitly
            agent = BitcoinAIAgent(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")
            
            # Use Gemini with custom API key
            agent = BitcoinAIAgent(llm_provider="google", api_key="your-key")
        """
        self.console = Console()
        # Historial de conversa (memòria simple en procés). Inclou SystemMessage + intercanvis.
        self._history: List[BaseMessage] = []
        # Límit màxim de missatges a retenir per evitar creixement indefinit
        self._history_limit = 40
        self._memory_enabled = True
        # Guarda la darrera directiva/instrucció explícita de l'usuari
        self._last_directive: Optional[str] = None

        # Handle legacy openai_api_key parameter
        if openai_api_key and not api_key:
            api_key = openai_api_key
            if not llm_provider:
                llm_provider = "openai"
        
        # Resolve provider (with fallback to legacy OPENAI_API_KEY behavior)
        resolved_provider = llm_provider or LLM_PROVIDER
        if not api_key and resolved_provider == "openai" and not llm_provider:
            # Legacy mode: if no provider explicitly set and OPENAI_API_KEY exists, use it
            api_key = OPENAI_API_KEY
        
        # Llista d'eines disponibles (afegint decode_psbt)
        self.tools = [
            get_balance,
            generate_address,
            list_utxos,
            get_fee_rates,
            create_transaction_manual,
            decode_psbt,  # Nova eina
            verify_utxo,
        ]
        
        # Store provider info for display
        self.llm_provider = resolved_provider
        self.llm_model = llm_model or LLM_MODEL or LLM_PROVIDERS.get(resolved_provider, {}).get("default_model", "unknown")
        
        # Create LLM using the factory
        self.llm = create_llm(
            provider=resolved_provider,
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
            tools=self.tools,
        )
        
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
    
    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the agent."""
        return f"""Ets un agent expert en Bitcoin que ajuda els usuaris a gestionar les seves wallets amb suport per PSBTs BIP-174 estàndard.

        Informació de context:
        - XPUB: {self.xpub}
        - Network: {self.network}

        EINES DISPONIBLES:

        1. get_balance(xpub, network)
        Consulta el balanç total de la wallet escanejant les adreces derivades.

        2. generate_address(xpub, network, index, require_unused, max_scan)
        Genera una adreça de recepció. Per defecte busca la primera adreça sense ús.

        3. list_utxos(xpub, network)
        Llista totes les UTXOs disponibles amb detalls (txid, vout, valor, adreça).

        4. get_fee_rates(network)
        Obté les fee rates actuals de la xarxa (fastest, half_hour, hour, economy).

        5. verify_utxo(txid, network)
        Verifica si una transacció existeix i retorna els seus outputs i estat.

        6. create_transaction_manual(xpub, recipient_address, amount_sats, utxo_ids, network, ...)
        Crea un PSBT BIP-174 amb selecció manual d'UTXOs.
        - utxo_ids: llista de "txid:vout" a gastar
        - tx_opts: opcions avançades (fee_satoshis, outputs, rbf, locktime_override, etc.)
        - psbt_extras: metadades PSBT addicionals (Taproot, BIP32 paths, etc.)

        7. decode_psbt(psbt_string)
        Decodifica i valida un PSBT en format base64.

        INSTRUCCIONS:
        - Quan cridis eines, usa sempre: {{"xpub": "{self.xpub}", "network": "{self.network}"}}
        - Respon sempre en català
        - Quan crees transaccions, genera PSBTs BIP-174 estàndard compatibles amb Electrum, Bitcoin Core i hardware wallets
        - Guarda els PSBTs creats i ofereix-los en format base64 i fitxer
        - Per crear una transacció, primer llista les UTXOs disponibles (list_utxos) per escollir quines gastar
        """

    def _agent_node(self, state: AgentState) -> AgentState:
        """Node de l'agent amb LLM millorat per PSBTs.
        
        IMPORTANT: To avoid 'multiple non-consecutive system messages' errors with
        Anthropic/Claude, we do NOT modify state["messages"] with the SystemMessage.
        Instead, we build messages_for_llm separately for invocation.
        """
        # Build messages for LLM invocation without modifying state directly
        # Filter out any existing SystemMessages and prepend a single one
        non_system_msgs = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        messages_for_llm = [SystemMessage(content=self._get_system_prompt())] + non_system_msgs
        
        # Generate LLM response
        try:
            response = self.llm.invoke(messages_for_llm)
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                logger.debug("Tool calls: %d", len(response.tool_calls))
                for tc in response.tool_calls:
                    logger.debug("  - %s: %s", tc['name'], tc['args'])
                    
                    # Track PSBT creation
                    if tc['name'] == 'create_transaction':
                        state["last_psbt"] = "pending"
                    # Persist recipient address for confirmations
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
            logger.error("Error in agent_node: %s", e)
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

        logger.debug("Message received: %s", message[:100])
        
        # Build initial messages with memory (if enabled)
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
            
            # Process results
            tool_results = []
            psbt_created = None
            
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    tool_results.append(msg.content)
                    try:
                        logger.debug("ToolMessage content type: %s", type(msg.content))
                        preview = str(msg.content)
                        if len(preview) > 200:
                            preview = preview[:200] + "..."
                        logger.debug("ToolMessage content preview: %s", preview)
                    except Exception:
                        pass
                    # Look for PSBT in response
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
                            logger.debug("PSBT saved (preview): %s...", 
                                        str(psbt_created)[:120] if psbt_created else "")
                    except Exception:
                        pass
            
            # Get final agent response
            final_response: Optional[str] = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                    if hasattr(msg, "tool_calls") and not msg.tool_calls:
                        final_response = _extract_text_content(msg.content)
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
                                    # Write files for easy wallet import
                                    Path("psbt_latest.psbt").write_bytes(base64.b64decode(clean_psbt))
                                    Path("psbt_latest.base64").write_text(clean_psbt)
                                except Exception as _e:
                                    logger.warning("Could not write PSBT file: %s", _e)

                                # Add standardized section with PSBT on its own line
                                final_response += "\n\n📄 **PSBT creat en format BIP-174 estàndard**"
                                final_response += "\nPots signar aquest PSBT amb qualsevol wallet compatible (Electrum, Bitcoin Core, hardware wallets, etc.)"
                                final_response += "\nElectrum: usa Tools > Load transaction > From text i enganxa el PSBT en Base64, o File > Open/Load transaction > From file amb psbt_latest.psbt (no el peguis com a HEX)."
                                final_response += "\nFitxers guardats: psbt_latest.psbt (binari) i psbt_latest.base64 (text)."
                                final_response += "\n\nPSBT (Base64 una sola línia, sense salts):\n" + clean_psbt + "\n\n"
                                # Defensive check for multiple PSBTs
                                cnt = _count_psbt_blobs(final_response)
                                if cnt > 1:
                                    logger.warning("Detected %d PSBTs in final text; should be 1", cnt)
                        break
                    elif not hasattr(msg, "tool_calls"):
                        final_response = _extract_text_content(msg.content)
                        break

            # Update memory with messages from graph
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
            logger.error("Error in chat: %s", e, exc_info=True)
            return f"Ho sento, hi ha hagut un error: {str(e)}"
    
    def setup(self, xpub: str, network: str = "testnet"):
        """Configure the agent with an XPUB."""
        self.xpub = xpub
        self.network = network
        self.console.print(f"[green]✅ Agent configured for {network}[/green]")
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
        
        # Show available LLM providers
        providers = get_available_providers()
        current_provider = LLM_PROVIDER
        current_model = LLM_MODEL or LLM_PROVIDERS.get(current_provider, {}).get("default_model", "default")
        
        self.console.print(f"\n[cyan]LLM Provider: {current_provider.upper()} ({current_model})[/cyan]")
        self.console.print("[dim]Canvia amb LLM_PROVIDER i LLM_MODEL al .env[/dim]")
        
        # Show provider status
        for name, info in providers.items():
            status_icon = "✅" if info["status"] == "ready" else ("⚠️" if info["status"] == "missing_key" else "❌")
            status_text = {
                "ready": "configurat",
                "missing_key": f"falta {info['env_key']}",
                "missing_package": "falta paquet"
            }.get(info["status"], info["status"])
            self.console.print(f"  {status_icon} {name}: {status_text}")
        
        # Obtenir configuració del .env
        xpub_env = os.getenv("BITCOIN_XPUB", "")
        network_env = os.getenv("BITCOIN_NETWORK", "testnet").lower()
        
        # Mostrar estat PSBT
        self.console.print("\n[green]✅ Suport PSBT BIP-174 activat[/green]")

        # Determine which API key to check based on provider
        provider_config = LLM_PROVIDERS.get(current_provider, LLM_PROVIDERS["openai"])
        api_key_env = provider_config["env_key"]
        api_key = os.getenv(api_key_env)
        
        # Configuració API key
        if not api_key or api_key == "your-key-here":
            self.console.print(f"[yellow]⚠️  No s'ha trobat API key vàlida ({api_key_env}) al fitxer .env[/yellow]")
            api_key = Prompt.ask(f"Introdueix la teva {current_provider.upper()} API Key", password=True)
            
            if Prompt.ask("Vols guardar la clau al fitxer .env?", choices=["s", "n"], default="s") == "s":
                self._update_env_file(api_key_env, api_key)
        else:
            self.console.print(f"[green]✅ API Key ({api_key_env}) carregada del fitxer .env[/green]")
        
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
        
        # Crear agent with multi-provider support
        try:
            self.agent = BitcoinAIAgent(
                llm_provider=current_provider,
                llm_model=LLM_MODEL or None,  # Use env model or provider default
                api_key=api_key,
            )
            self.agent.setup(xpub, network)
            self.console.print("\n[green]✅ Agent IA configurat correctament![/green]")
            self.console.print(f"[green]   Network: {network}[/green]")
            self.console.print(f"[green]   XPUB: {xpub[:20]}...{xpub[-10:]}[/green]")
            self.console.print(f"[green]   LLM: {self.agent.llm_provider.upper()} / {self.agent.llm_model}[/green]")
            self.console.print(f"[green]   PSBT: BIP-174 Standard[/green]")
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
        