#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
import struct, hashlib, sys, traceback

# Shared codec utilities (deduplicated Base58Check, hash160, etc.)
import codec



def _hdwallet_bits():
    """
    Importa 'hdwallet' només quan el necessitem, per evitar ImportError
    durant la càrrega del mòdul (p.ex. en pytest collection).
    """
    try:
        from hdwallet import HDWallet
        from hdwallet.cryptocurrencies import Bitcoin
        from hdwallet.hds.bip32 import BIP32HD
        from hdwallet.derivations.custom import CustomDerivation
    except Exception as e:
        raise RuntimeError(
            "El paquet 'hdwallet' no és accessible des d'aquest entorn. "
            "Assegura't de tenir l'entorn virtual activat i haver instal·lat 'hdwallet' "
            "(p. ex. amb: python -m pip install hdwallet)."
        ) from e
    return HDWallet, Bitcoin, BIP32HD, CustomDerivation

# ---------- prefix → tipus adreça ----------
def _purpose_and_method(xpub_like: str):
    p = xpub_like[:4].lower()
    if p in ("zpub", "vpub"):  # BIP84
        return "84'", "p2wpkh_address"
    if p in ("ypub", "upub"):  # BIP49
        return "49'", "p2sh_p2wpkh_address"
    return "44'", "p2pkh_address"  # BIP44 per xpub/tpub

_b58decode_check = codec.b58decode_check  # Shim for existing internal usage
_b58encode_check = codec.b58encode_check


def _sanitize_extended_key(x: str) -> str:
    """Normalize an extended public key string to avoid Base58 decode errors.

    - Strips surrounding whitespace and common quotes/brackets.
    - Removes zero-width and NBSP characters often introduced by UI copy/paste.
    - Collapses internal whitespace.
    """
    if not isinstance(x, str):
        return x
    s = x.strip().strip("’‘'\"“”<>[](){}.,:;`“”")
    for ch in ("\u200b", "\ufeff", "\u2060", "\u00A0"):
        s = s.replace(ch, "")
    # Remove any remaining whitespace sequences
    try:
        import re
        s = re.sub(r"\s+", "", s)
    except Exception:
        pass
    return s

# ---------- mapes de versions SLIP-0132 ----------
_VERSION_MAP_TO_XPUB = {
    0x049D7CB2: 0x0488B21E,  # ypub → xpub
    0x04B24746: 0x0488B21E,  # zpub → xpub
}
_VERSION_MAP_TO_TPUB = {
    0x044A5262: 0x043587CF,  # upub → tpub
    0x045F1CF6: 0x043587CF,  # vpub → tpub
}

def _normalize_to_x_or_t_pub(xpub_like: str) -> str:
    xpub_like = _sanitize_extended_key(xpub_like)
    raw = _b58decode_check(xpub_like)
    if len(raw) != 78:
        raise ValueError(f"Extended key longitud incorrecta: {len(raw)} bytes (78 esperats)")
    ver = struct.unpack(">I", raw[:4])[0]
    body = raw[4:]
    if xpub_like.startswith(("xpub", "tpub")):
        return xpub_like
    if ver in _VERSION_MAP_TO_XPUB:
        return _b58encode_check(struct.pack(">I", _VERSION_MAP_TO_XPUB[ver]) + body)
    if ver in _VERSION_MAP_TO_TPUB:
        return _b58encode_check(struct.pack(">I", _VERSION_MAP_TO_TPUB[ver]) + body)
    return xpub_like

def _version_implies_testnet(xpub_like: str) -> bool:
    xpub_like = _sanitize_extended_key(xpub_like)
    raw = _b58decode_check(xpub_like)
    if len(raw) != 78:
        return False
    ver = struct.unpack(">I", raw[:4])[0]
    return ver in (0x043587CF, 0x044A5262, 0x045F1CF6)  # tpub/upub/vpub

# ---------- helpers d’adreça ----------
def _hash160(b: bytes) -> bytes:
    return codec.hash160(b)

_encode_segwit_address = codec.encode_segwit_address  # Reexport for existing usage

def _p2pkh_address(pubkey_bytes: bytes, network: str) -> str:
    vh = (b"\x6f" if network == "testnet" else b"\x00") + _hash160(pubkey_bytes)
    return _b58encode_check(vh)

def _p2sh_p2wpkh_address(pubkey_bytes: bytes, network: str) -> str:
    rs = b"\x00\x14" + _hash160(pubkey_bytes)
    vh = (b"\xc4" if network == "testnet" else b"\x05") + _hash160(rs)
    return _b58encode_check(vh)

def _p2wpkh_address(pubkey_bytes: bytes, network: str) -> str:
    hrp = "tb" if network == "testnet" else "bc"
    return _encode_segwit_address(hrp, 0, _hash160(pubkey_bytes))

# ---------- derivació principal amb DEBUG ----------
def derive_bitcoin_address(xpub_or_zpub: str, index: int = 0, change: bool = False, network: str = "mainnet") -> Dict:
    
    HDWallet, Bitcoin, BIP32HD, CustomDerivation = _hdwallet_bits()

    try:
        if index < 0 or index >= 2**31:
            raise ValueError("Index fora de rang no-hardened")

        # Sanitize key early to avoid base58 checksum spam from hidden characters
        xpub_or_zpub = _sanitize_extended_key(xpub_or_zpub)

        # Xarxa deduïda (version bytes)
        net = "testnet" if _version_implies_testnet(xpub_or_zpub) else network

        # Normalitza ypub/zpub → xpub | upub/vpub → tpub
        normalized = _normalize_to_x_or_t_pub(xpub_or_zpub)

        # Força HD = BIP32 i NO netegis derivació abans (encara no existeix)
        hdwallet = HDWallet(cryptocurrency=Bitcoin, hd=BIP32HD, network=net)

        # Inicialitza des de l'XPUB (no-root permès)
        hdwallet.from_xpublic_key(xpublic_key=normalized, strict=False)
       

        # Deriva NOMÉS m/<chain>/<index> (no-hardened)
        chain = 1 if change else 0

        # 1r intent: ruta completa en una sola passada
        try:
            hdwallet.from_path(f"m/{chain}/{index}")
        except Exception as e1:
            try:
                deriv = CustomDerivation().from_path(f"m/{chain}/{index}")
                hdwallet.update_derivation(derivation=deriv)
            except Exception as e2:
                # Últim recurs: dos salts no-hardened
                hdwallet.from_index(chain, hardened=False)
                hdwallet.from_index(index, hardened=False)

        # Public key comprimida
        pub_hex = hdwallet.public_key()
        if pub_hex.startswith("0x"):
            pub_hex = pub_hex[2:]
        pub_bytes = bytes.fromhex(pub_hex)
        if len(pub_bytes) != 33:
            raise ValueError("Public key no comprimida o mida inesperada")

        # Adreça segons prefix original
        purpose, _ = _purpose_and_method(xpub_or_zpub)
        if purpose == "84'":
            address = _p2wpkh_address(pub_bytes, net)
        elif purpose == "49'":
            address = _p2sh_p2wpkh_address(pub_bytes, net)
        else:
            address = _p2pkh_address(pub_bytes, net)

        coin_type = "1'" if net == "testnet" else "0'"
        full_path = f"m/{purpose}/{coin_type}/0'/{chain}/{index}"

        return {
            "success": True,
            "address": address,
            "public_key": pub_hex,
            "index": index,
            "change": change,
            "path": full_path,
            "derivation": f"{chain}/{index}",
            "network": net,
        }
    except Exception as e:
        # Avoid noisy stack traces by default; callers surface concise error messages.
        # Uncomment for debugging:
        # traceback.print_exc()
        return {"success": False, "error": str(e)}

# ---------- test ----------
def test_hdwallet():
    zpub = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"
    expected = "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"
    print("🧪 Test de derivació amb HDWallet")
    print("="*70)
    print(f"ZPUB: {zpub[:30]}...")
    print(f"Adreça esperada (index 0): {expected}")
    print("="*70)

    print("\n📊 Test principal (index 0):")
    result = derive_bitcoin_address(zpub, index=0, change=False, network="mainnet")
    if result["success"]:
        print("✅ Derivació exitosa!")
        print(f"   Adreça obtinguda: {result['address']}")
        print(f"   Public Key: {result['public_key'][:40]}...")
        print(f"   Path (informatiu): {result['path']}")
        print(f"   Derivació aplicada: {result['derivation']}")
        if result["address"] == expected:
            print("\n   🎉 PERFECTE! L'adreça coincideix!")
        else:
            print("\n   ⚠️ L'adreça no coincideix")
            print(f"   Esperada:  {expected}")
            print(f"   Obtinguda: {result['address']}")
    else:
        print(f"❌ Error: {result['error']}")

    print("\n📦 Altres adreces de recepció:")
    for i in range(1, 5):
        r = derive_bitcoin_address(zpub, index=i, change=False, network="testnet")
        print(f"   Index {i}: {r['address'] if r['success'] else 'Error - ' + r['error']}")

    print("\n💱 Adreces de canvi:")
    for i in range(0, 3):
        r = derive_bitcoin_address(zpub, index=i, change=True, network="testnet")
        print(f"   Change {i}: {r['address'] if r['success'] else 'Error - ' + r['error']}")

def derive_real_address_hdwallet(xpub: str, network: str, index: int, change: bool = False) -> str:
    res = derive_bitcoin_address(xpub, index, change, network)
    if res["success"]:
        return res["address"]
    # Eliminat el fallback de generació d'adreça sintètica perquè pot confondre l'usuari.
    # En lloc d'això aixequem un error clar indicant la causa real.
    error_msg = res.get("error") or "Unknown derivation failure"
    raise ValueError(f"Derivation failed (no fake address created): {error_msg}")

if __name__ == "__main__":
    test_hdwallet()
