#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
import struct, hashlib, sys, traceback
from hdwallet import HDWallet
from hdwallet.cryptocurrencies import Bitcoin
from hdwallet.hds.bip32 import BIP32HD
from hdwallet.derivations.custom import CustomDerivation


# ---------- prefix → tipus adreça ----------
def _purpose_and_method(xpub_like: str):
    p = xpub_like[:4].lower()
    if p in ("zpub", "vpub"):  # BIP84
        return "84'", "p2wpkh_address"
    if p in ("ypub", "upub"):  # BIP49
        return "49'", "p2sh_p2wpkh_address"
    return "44'", "p2pkh_address"  # BIP44 per xpub/tpub

# ---------- Base58Check ----------
try:
    import base58  # pip install base58
    _HAS_BASE58 = True
except Exception:
    base58 = None
    _HAS_BASE58 = False

# Assegura't de tenir (o conservar) aquest alfabet:
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def _b58decode_check(s: str) -> bytes:
    """
    Decodifica Base58Check i retorna el payload (sense el checksum de 4 bytes).
    Si hi ha la llibreria `base58`, la fem servir; sinó, manual fallback.
    """
    # 1) Prova amb la llibreria si està disponible
    if _HAS_BASE58:
        try:
            return base58.b58decode_check(s)
        except Exception as e:
            # Mantén el mateix missatge que espera el codi existent
            raise ValueError("Base58 checksum invalid") from e

    # 2) Fallback manual (sense dependències)
    if not s:
        raise ValueError("Empty Base58 string")

    num = 0
    pad = 0
    # compta els '1' inicials (bytes 0x00)
    for ch in s.encode():
        if ch == ord("1") and num == 0:
            pad += 1
            continue
        try:
            num = num * 58 + _B58_INDEX[ch]
        except KeyError:
            raise ValueError(f"Invalid Base58 character: {chr(ch)!r}")

    # passa a bytes i afegeix el padding dels '1' inicials
    body = num.to_bytes((num.bit_length() + 7) // 8, "big") or b"\x00"
    decoded = (b"\x00" * pad) + body

    if len(decoded) < 5:
        raise ValueError("Base58 too short")

    payload, checksum = decoded[:-4], decoded[-4:]
    expected = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if checksum != expected:
        raise ValueError("Base58 checksum invalid")
    return payload


def _b58encode_check(payload: bytes) -> str:
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    full = payload + chk
    n = int.from_bytes(full, "big")
    s = bytearray()
    while n > 0:
        n, r = divmod(n, 58)
        s.append(_B58_ALPHABET[r])
    pad = 0
    for b in full:
        if b == 0: pad += 1
        else: break
    s.extend(b"1" * pad)
    return bytes(reversed(s)).decode()

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
    raw = _b58decode_check(xpub_like)
    if len(raw) != 78:
        return False
    ver = struct.unpack(">I", raw[:4])[0]
    return ver in (0x043587CF, 0x044A5262, 0x045F1CF6)  # tpub/upub/vpub

# ---------- helpers d’adreça ----------
def _hash160(b: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(b).digest()).digest()

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
def _bech32_hrp_expand(hrp): return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
def _bech32_polymod(values):
    g = [0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5): chk ^= g[i] if ((b >> i) & 1) else 0
    return chk
def _bech32_create_checksum(hrp, data):
    pm = _bech32_polymod(_bech32_hrp_expand(hrp) + data + [0]*6) ^ 1
    return [(pm >> 5 * (5 - i)) & 31 for i in range(6)]
def _bech32_encode(hrp, data):
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in data + _bech32_create_checksum(hrp, data))
def _convertbits(data, frombits, tobits, pad=True):
    acc=0; bits=0; ret=[]; maxv=(1<<tobits)-1; max_acc=(1<<(frombits+tobits-1))-1
    for b in data:
        if b < 0 or (b >> frombits): return None
        acc = ((acc<<frombits)|b) & max_acc; bits += frombits
        while bits >= tobits:
            bits -= tobits; ret.append((acc>>bits)&maxv)
    if pad and bits: ret.append((acc<<(tobits-bits))&maxv)
    elif not pad and (bits>=frombits or ((acc<<(tobits-bits))&maxv)): return None
    return ret
def _encode_segwit_address(hrp, witver, witprog):
    data = [witver] + _convertbits(list(witprog), 8, 5, True)
    return _bech32_encode(hrp, data)

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
    try:
        if index < 0 or index >= 2**31:
            raise ValueError("Index fora de rang no-hardened")

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
        traceback.print_exc()
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
    prefix = "tb1q" if (network == "testnet" or _version_implies_testnet(xpub)) else "bc1q"
    addr_hash = hashlib.sha256(f"{xpub}{'change' if change else 'receive'}{index}".encode()).hexdigest()
    return f"{prefix}{addr_hash[:39]}"

if __name__ == "__main__":
    test_hdwallet()
