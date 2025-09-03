"""Lightweight encoding/decoding primitives shared across modules.

Currently exposes:
  - Base58Check encode/decode (with optional dependency on 'base58')
  - hash160 (RIPEMD160(SHA256(data)))
  - CompactSize (varint) encode/decode helpers

Design goals:
  - No behavioral change versus previous in-module implementations.
  - Error messages preserved where tests expect specific substrings
    (e.g. "Base58 checksum invalid").
  - Zero network / side effects.
"""
from __future__ import annotations

from typing import Tuple
import hashlib

try:  # Optional acceleration / battle‑tested implementation
    import base58  # type: ignore
    _HAS_BASE58 = True
except Exception:  # pragma: no cover - fallback path exercised in absence of lib
    base58 = None  # type: ignore
    _HAS_BASE58 = False

__all__ = [
    "b58encode_check",
    "b58decode_check",
    "hash160",
    "compact_size_encode",
    "compact_size_decode",
    # Bech32 / SegWit helpers
    "bech32_hrp_expand",
    "bech32_polymod",
    "bech32_create_checksum",
    "bech32_encode",
    "convertbits",
    "encode_segwit_address",
    "decode_segwit_address",
]


# ------------------------- Base58Check -------------------------
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def b58decode_check(s: str) -> bytes:
    """Return payload (version byte + data) after validating Base58Check checksum.

    Mirrors prior _b58decode_check behavior from address_derivation while allowing
    the external 'base58' library if present. Raises ValueError with identical
    messages so existing tests continue to pass.
    """
    if _HAS_BASE58:  # Fast path using library
        try:
            return base58.b58decode_check(s)  # type: ignore[attr-defined]
        except Exception as e:  # Library raises generic errors; normalize
            raise ValueError("Base58 checksum invalid") from e

    # Manual fallback
    if not s:
        raise ValueError("Empty Base58 string")

    num = 0
    pad = 0
    for ch in s.encode():
        if ch == ord("1") and num == 0:
            pad += 1
            continue
        try:
            num = num * 58 + _B58_INDEX[ch]
        except KeyError:
            raise ValueError(f"Invalid Base58 character: {chr(ch)!r}")

    body = num.to_bytes((num.bit_length() + 7) // 8, "big") or b"\x00"
    decoded = (b"\x00" * pad) + body

    if len(decoded) < 5:
        raise ValueError("Base58 too short")

    payload, checksum = decoded[:-4], decoded[-4:]
    expected = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if checksum != expected:
        raise ValueError("Base58 checksum invalid")
    return payload


def b58encode_check(payload: bytes) -> str:
    """Encode payload (version + data) with Base58Check checksum.

    Keeps exact algorithm used previously (double SHA256 then 4-byte checksum).
    """
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    full = payload + chk
    n = int.from_bytes(full, "big")
    s = bytearray()
    while n > 0:
        n, r = divmod(n, 58)
        s.append(_B58_ALPHABET[r])
    pad = 0
    for b in full:
        if b == 0:
            pad += 1
        else:
            break
    s.extend(b"1" * pad)
    return bytes(reversed(s)).decode()


# ------------------------- Hash helpers -------------------------
def hash160(data: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(data).digest()).digest()


# ------------------------- CompactSize (varint) -------------------------
def compact_size_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("Negative size")
    if n < 0xFD:
        return n.to_bytes(1, "little")
    if n <= 0xFFFF:
        return b"\xfd" + n.to_bytes(2, "little")
    if n <= 0xFFFFFFFF:
        return b"\xfe" + n.to_bytes(4, "little")
    return b"\xff" + n.to_bytes(8, "little")


def compact_size_decode(data: bytes) -> Tuple[int, int]:
    if not data:
        raise ValueError("Empty buffer")
    first = data[0]
    if first < 0xFD:
        return first, 1
    if first == 0xFD:
        if len(data) < 3:
            raise ValueError("Truncated CompactSize (0xFD)")
        return int.from_bytes(data[1:3], "little"), 3
    if first == 0xFE:
        if len(data) < 5:
            raise ValueError("Truncated CompactSize (0xFE)")
        return int.from_bytes(data[1:5], "little"), 5
    if len(data) < 9:
        raise ValueError("Truncated CompactSize (0xFF)")
    return int.from_bytes(data[1:9], "little"), 9


# ------------------------- Bech32 / SegWit -------------------------
_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def bech32_hrp_expand(hrp: str):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def bech32_polymod(values):
    generators = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= generators[i]
    return chk


def bech32_create_checksum(hrp: str, data):
    polymod = bech32_polymod(bech32_hrp_expand(hrp) + data + [0] * 6) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]


def bech32_encode(hrp: str, data) -> str:
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in data + bech32_create_checksum(hrp, data))


def convertbits(data, frombits: int, tobits: int, pad: bool = True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for b in data:
        if b < 0 or b >> frombits:
            raise ValueError("Valor fora de rang")
        acc = (acc << frombits | b) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        raise ValueError("Bits sobrants després de la conversió")
    return ret


def encode_segwit_address(hrp: str, witver: int, witprog: bytes) -> str:
    if not (0 <= witver <= 16):
        raise ValueError("Witness version fora de rang")
    data = [witver] + convertbits(list(witprog), 8, 5, True)
    return bech32_encode(hrp, data)


def decode_segwit_address(address: str):
    """Return (hrp, witness_version, witness_program) validating Bech32 / Bech32m.
    Error messages kept similar to existing code to avoid breaking tests.
    """
    BECH32_CONST = 1
    BECH32M_CONST = 0x2bc830a3

    if address != address.lower() and address != address.upper():
        raise ValueError("Bech32 mixed-case no permès")
    address = address.lower()
    pos = address.rfind("1")
    if pos < 1 or pos + 7 > len(address):
        raise ValueError("Format Bech32 invàlid")
    hrp = address[:pos]
    data_part = address[pos + 1 :]
    if any(ord(c) < 33 or ord(c) > 126 for c in address):
        raise ValueError("Caràcter fora de rang ASCII a Bech32")
    data = []
    for c in data_part:
        if c not in _BECH32_CHARSET:
            raise ValueError(f"Caràcter invàlid en Bech32: {c}")
        data.append(_BECH32_CHARSET.index(c))
    if len(data) < 7:
        raise ValueError("Massa curt per contenir dades + checksum")
    payload, checksum = data[:-6], data[-6:]
    polymod = bech32_polymod(bech32_hrp_expand(hrp) + payload + checksum)
    witness_version = payload[0]
    program5 = payload[1:]
    prog_bytes = bytes(convertbits(program5, 5, 8, False))
    const_expected = BECH32_CONST if witness_version == 0 else BECH32M_CONST
    if polymod != const_expected:
        raise ValueError("Checksum Bech32/Bech32m invàlid")
    if not (2 <= len(prog_bytes) <= 40):
        raise ValueError("Longitud de witness program fora de rang")
    return hrp, witness_version, prog_bytes
