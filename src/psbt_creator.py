#!/usr/bin/env python3
"""
Creació de PSBT (Partially Signed Bitcoin Transaction) estàndard BIP-174
amb càlcul de comissions per vbytes reals i selecció d'UTXOs coherent.
"""

import struct
import hashlib
import base64
import json
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import requests
import math

# Derivation helper
try:
    from address_derivation import derive_bitcoin_address as _derive_addr_internal
except Exception:  # pragma: no cover
    _derive_addr_internal = None

# Shared codec utilities (Base58Check + hash160 + varints)
import codec

# ==========================
# Constants per a PSBT
# ==========================
PSBT_MAGIC = b"psbt\xff"

# PSBT Global Types
PSBT_GLOBAL_UNSIGNED_TX = b"\x00"
PSBT_GLOBAL_XPUB = b"\x01"
PSBT_GLOBAL_TX_VERSION = b"\x02"
PSBT_GLOBAL_FALLBACK_LOCKTIME = b"\x03"
PSBT_GLOBAL_INPUT_COUNT = b"\x04"
PSBT_GLOBAL_OUTPUT_COUNT = b"\x05"
PSBT_GLOBAL_TX_MODIFIABLE = b"\x06"
PSBT_GLOBAL_VERSION = b"\xfb"

# PSBT Input Types
PSBT_IN_NON_WITNESS_UTXO = b"\x00"
PSBT_IN_WITNESS_UTXO = b"\x01"
PSBT_IN_PARTIAL_SIG = b"\x02"
PSBT_IN_SIGHASH_TYPE = b"\x03"
PSBT_IN_REDEEM_SCRIPT = b"\x04"
PSBT_IN_WITNESS_SCRIPT = b"\x05"
PSBT_IN_BIP32_DERIVATION = b"\x06"
PSBT_IN_FINAL_SCRIPTSIG = b"\x07"
PSBT_IN_FINAL_SCRIPTWITNESS = b"\x08"
PSBT_IN_POR_COMMITMENT = b"\x09"
PSBT_IN_RIPEMD160 = b"\x0a"
PSBT_IN_SHA256 = b"\x0b"
PSBT_IN_HASH160 = b"\x0c"
PSBT_IN_HASH256 = b"\x0d"

# PSBT Output Types
PSBT_OUT_REDEEM_SCRIPT = b"\x00"
PSBT_OUT_WITNESS_SCRIPT = b"\x01"
PSBT_OUT_BIP32_DERIVATION = b"\x02"

from addressing import detect_address_type  # shared classification

# ==========================
# Heurístiques vbytes i llindars
# ==========================
DUST_THRESHOLD = 546  # sats

# Valors típics (vbytes) acceptats àmpliament:
#   Inputs:  P2PKH ~148,  P2SH-P2WPKH ~91,  P2WPKH ~68,  P2TR ~57
#   Outputs: P2PKH ~34,   P2SH ~32,        P2WPKH ~31,   P2TR ~43
INPUT_VBYTES = {
    "p2pkh": 148,
    "p2sh-p2wpkh": 91,
    "p2wpkh": 68,
    "p2tr": 57,
}
OUTPUT_VBYTES = {
    "p2pkh": 34,
    "p2sh": 32,
    "p2wpkh": 31,
    "p2tr": 43,
}


def _detect_address_type(addr: str) -> str:  # shim for backward internal reference
    return detect_address_type(addr)


def _input_vbytes_for_address(addr: str) -> int:
    return INPUT_VBYTES.get(_detect_address_type(addr), 68)


def _output_vbytes_for_address(addr: str) -> int:
    t = _detect_address_type(addr)
    if t == "p2sh-p2wpkh":
        return OUTPUT_VBYTES["p2sh"]
    return OUTPUT_VBYTES.get(t, 31)


def _estimate_vbytes(
    selected_input_addrs: List[str],
    recipient_addr: str,
    include_change: bool,
    change_addr: Optional[str],
) -> int:
    # base ~10 vbytes (versió, locktime, marker/flag efectius)
    base = 10
    ins = sum(_input_vbytes_for_address(a) for a in selected_input_addrs)
    outs = _output_vbytes_for_address(recipient_addr)
    if include_change and change_addr:
        outs += _output_vbytes_for_address(change_addr)
    return base + ins + outs


def _select_utxos_vbytes(
    utxos: List[Dict],
    amount_sats: int,
    fee_rate_sat_vb: int,
    recipient_addr: str,
    change_addr: str,
) -> Dict:
    """
    Selecció greedy determinista per valor DESC,
    recalculant la comissió segons els vbytes dels inputs seleccionats i outputs previstos.
    """
    def _key(u):
        return (-int(u.get("value_satoshis", 0)), u.get("txid", ""), int(u.get("vout", 0)))

    pool = sorted(utxos, key=_key)
    selected: List[Dict] = []
    total_in = 0

    for u in pool:
        selected.append(u)
        total_in += int(u.get("value_satoshis", 0))
        addrs = [i.get("address", "") for i in selected]

        # Escenari sense canvi
        vb_no = _estimate_vbytes(addrs, recipient_addr, include_change=False, change_addr=None)
        fee_no = math.ceil(fee_rate_sat_vb * vb_no)
        change_no = total_in - amount_sats - fee_no
        if change_no < DUST_THRESHOLD and total_in >= amount_sats + fee_no:
            return {
                "success": True,
                "selected_utxos": selected,
                "fee_satoshis": fee_no,
                "change_satoshis": 0,
                "with_change": False,
            }

        # Escenari amb canvi
        vb_ch = _estimate_vbytes(addrs, recipient_addr, include_change=True, change_addr=change_addr)
        fee_ch = math.ceil(fee_rate_sat_vb * vb_ch)
        change_ch = total_in - amount_sats - fee_ch
        if change_ch >= DUST_THRESHOLD and total_in >= amount_sats + fee_ch:
            return {
                "success": True,
                "selected_utxos": selected,
                "fee_satoshis": fee_ch,
                "change_satoshis": change_ch,
                "with_change": True,
            }

    return {
        "success": False,
        "error": f"Fons insuficients. Necessari: {(amount_sats/1e8):.8f} BTC + fee, Disponible: {(total_in/1e8):.8f} BTC",
    }


# ==========================
# Creador de PSBT
# ==========================
class PSBTCreator:
    """Crea PSBTs estàndard BIP-174"""

    def __init__(self, network: str = "testnet"):
        self.network = network
        self.api_base = (
            "https://blockstream.info/testnet/api"
            if network == "testnet"
            else "https://blockstream.info/api"
        )

    # ---------- Utils binaris / hashing ----------
    def _compact_size(self, n: int) -> bytes:
        return codec.compact_size_encode(n)

    def _hash256(self, data: bytes) -> bytes:
        """Double SHA256"""
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    def _hash160(self, data: bytes) -> bytes:
        """RIPEMD160(SHA256(data))"""
        return hashlib.new("ripemd160", hashlib.sha256(data).digest()).digest()

    def _write_key_value(self, key_type: bytes, key_data: bytes, value: bytes) -> bytes:
        """Escriu un parell clau-valor en format PSBT"""
        key = key_type + key_data
        result = self._compact_size(len(key)) + key
        result += self._compact_size(len(value)) + value
        return result

    # ---------- Decodificació d’adreces ----------
    def _decode_address(self, address: str) -> Tuple[int, bytes]:
        """
        Decodifica una adreça Bitcoin i retorna (version/witness_version, hash_bytes).
        Suporta: P2PKH, P2SH, P2WPKH (Bech32 v0). Per Taproot (Bech32m) detectem, però no desempaquetem completament.
        """
        if address.startswith(("bc1", "tb1")):
            return self._decode_bech32(address)
        return self._decode_base58(address)

    def _decode_bech32(self, address: str) -> Tuple[int, bytes]:
        """Decodifica adreça Bech32/Bech32m amb validació de checksum (BIP-173/BIP-350)."""
        BECH32_CONST = 1
        BECH32M_CONST = 0x2bc830a3
        charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

        # Lower-case enforcement (mixed-case invalid)
        if address != address.lower() and address != address.upper():
            raise ValueError("Bech32 mixed-case no permès")
        address = address.lower()

        pos = address.rfind("1")
        if pos < 1 or pos + 7 > len(address):  # hrp + separator + 6 checksum min
            raise ValueError("Format Bech32 invàlid")

        hrp = address[:pos]
        data_part = address[pos + 1 :]
        if any(ord(c) < 33 or ord(c) > 126 for c in address):
            raise ValueError("Caràcter fora de rang ASCII a Bech32")

        data: List[int] = []
        for c in data_part:
            if c not in charset:
                raise ValueError(f"Caràcter invàlid en Bech32: {c}")
            data.append(charset.index(c))

        if len(data) < 7:
            raise ValueError("Massa curt per contenir dades + checksum")

        # Split payload + checksum
        payload = data[:-6]
        checksum = data[-6:]

        def _hrp_expand(h: str) -> List[int]:
            return [ord(x) >> 5 for x in h] + [0] + [ord(x) & 31 for x in h]

        def _polymod(vals: List[int]) -> int:
            GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
            chk = 1
            for v in vals:
                b = chk >> 25
                chk = (chk & 0x1ffffff) << 5 ^ v
                for i in range(5):
                    if (b >> i) & 1:
                        chk ^= GEN[i]
            return chk

        polymod = _polymod(_hrp_expand(hrp) + payload + checksum)

        witness_version = payload[0]
        program = self._convertbits(payload[1:], 5, 8, False)

        const_expected = BECH32_CONST if witness_version == 0 else BECH32M_CONST
        if polymod != const_expected:
            raise ValueError("Checksum Bech32/Bech32m invàlid")

        wp = bytes(program)
        if not (2 <= len(wp) <= 40):
            raise ValueError("Longitud de witness program fora de rang")
        # Per a P2WPKH i P2WSH longituds esperades 20 o 32; no forcem per flexibilitat
        return witness_version, wp

    def _convertbits(self, data: List[int], frombits: int, tobits: int, pad: bool) -> List[int]:
        """Converteix entre amplades de bits (per a Bech32)"""
        acc = 0
        bits = 0
        ret: List[int] = []
        maxv = (1 << tobits) - 1
        max_acc = (1 << (frombits + tobits - 1)) - 1

        for value in data:
            if value < 0 or (value >> frombits):
                raise ValueError("Valor fora de rang")
            acc = ((acc << frombits) | value) & max_acc
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

    def _decode_base58(self, address: str) -> Tuple[int, bytes]:
        """Decodifica adreça Base58Check (P2PKH/P2SH) utilitzant codec.b58decode_check."""
        try:
            payload = codec.b58decode_check(address)
            if len(payload) < 2:
                raise ValueError("Payload massa curt")
            return payload[0], payload[1:]
        except Exception as e:  # Normalitza el missatge consistent amb versió anterior
            raise ValueError(f"Error decodificant Base58: {e}")

    # ---------- Construcció de transacció ----------
    def _create_transaction_output(self, address: str, amount_satoshis: int) -> bytes:
        """Crea un output de transacció (amount + scriptPubKey)"""
        output = struct.pack("<Q", amount_satoshis)

        try:
            version, addr_hash = self._decode_address(address)
            if address.startswith(("bc1", "tb1")):
                # SegWit (version és witness version 0..16)
                wver = version
                # P2WPKH (v0, 20 bytes)
                if wver == 0 and len(addr_hash) == 20:
                    script = b"\x00\x14" + addr_hash
                # P2WSH (v0, 32 bytes)
                elif wver == 0 and len(addr_hash) == 32:
                    script = b"\x00\x20" + addr_hash
                # P2TR (Taproot v1, 32 bytes) OP_1 (0x51) + push32
                elif wver == 1 and len(addr_hash) == 32:
                    script = b"\x51\x20" + addr_hash
                else:
                    raise ValueError("Unsupported or invalid segwit witness program length/version")
            elif version in [0x00, 0x6F]:  # P2PKH
                script = b"\x76\xa9\x14" + addr_hash + b"\x88\xac"
            elif version in [0x05, 0xC4]:  # P2SH
                script = b"\xa9\x14" + addr_hash + b"\x87"
            else:
                raise ValueError("Unsupported address type")
        except Exception as e:
            raise ValueError(f"Invalid recipient/change address: {address} ({e})")

        output += self._compact_size(len(script)) + script
        return output

    def _fetch_transaction(self, txid: str) -> Optional[bytes]:
        """Obté una transacció (hex) des de l'API Blockstream; None si no disponible"""
        try:
            response = requests.get(f"{self.api_base}/tx/{txid}/hex", timeout=10)
            if response.status_code == 200:
                return bytes.fromhex(response.text.strip())
            return None
        except Exception as e:
            print(f"[ERROR] No s'ha pogut obtenir la tx {txid}: {e}")
            return None

    def create_psbt(
        self,
        inputs: List[Dict],
        outputs: List[Dict],
        locktime: int = 0,
        version: int = 2,
        xpub: Optional[str] = None,
        return_dict: bool = True,
        include_derivations: bool = True,
        proprietary: Optional[List[Tuple[bytes, int, bytes, bytes]]] = None,
    ) -> Dict:
        """
        Construeix una PSBT v0 amb la transacció unsigned i mapes d'inputs/outputs mínims.

        Nou comportament: retorna un diccionari estructurat amb camps clau.
        Compatibilitat: si es vol el string Base64 directament, accedir a ['psbt']
        o utilitzar el helper `create_psbt_base64` (veure més avall).
        """
        # 1) Transacció unsigned (sense witnesses)
        tx_bytes = b""
        tx_bytes += struct.pack("<I", version)


        tx_bytes += self._compact_size(len(inputs))

        for inp in inputs:
            txid_bytes = bytes.fromhex(inp["txid"])[::-1]  # little-endian
            tx_bytes += txid_bytes
            tx_bytes += struct.pack("<I", int(inp["vout"]))
            tx_bytes += b"\x00"  # scriptsig buit
            tx_bytes += b"\xff\xff\xff\xff"  # sequence

        tx_bytes += self._compact_size(len(outputs))
        for output in outputs:
            tx_bytes += self._create_transaction_output(output["address"], int(output["value"]))

        # Locktime
        tx_bytes += struct.pack("<I", locktime)

        # Guarda outputs per a decode_psbt
        self._last_outputs = outputs

        # 2) PSBT = magic + global map + input maps + output maps
        psbt_bytes = PSBT_MAGIC

        # Helper: extract serialized xpub & fingerprint if provided
        def _parse_xpub(xpub_s: str):
            """Return (raw_serialized, key_data, fingerprint, depth, child_num_val, version_int).

            If parsing fails returns all None.
            """
            try:
                raw = codec.b58decode_check(xpub_s)
                if len(raw) != 78:
                    return (None,)*6
                version_int = int.from_bytes(raw[0:4], "big")
                depth = raw[4]
                parent_fpr = raw[5:9]
                child_num_val = int.from_bytes(raw[9:13], "big")  # big-endian uint32
                chain_code = raw[13:45]
                key_data = raw[45:78]  # 33 bytes (0x02/0x03 compressed or 0x00 + priv)
                fp = codec.hash160(key_data)[:4]
                return raw, key_data, fp, depth, child_num_val, version_int
            except Exception:
                return (None,)*6

        def _guess_xpub_path(depth: int, child_num_val: int, version_int: int, network: str) -> Optional[List[int]]:
            """Heuristically reconstruct a standard account-level derivation path for common SLIP-0132 versions.

            Returns list of uint32 indices (with hardened bit where appropriate) whose length SHOULD equal depth.
            If we cannot confidently guess, returns None so we skip GLOBAL_XPUB to avoid wallet parser errors.
            """
            # Harden helper
            H = 0x80000000
            # Map version bytes to purpose'
            purpose_map = {
                # Mainnet
                0x0488B21E: 44,  # xpub (assume BIP44)
                0x049d7cb2: 49,  # ypub (BIP49 p2wpkh-p2sh)
                0x04b24746: 84,  # zpub (BIP84 p2wpkh)
                0x0295b43f: 48,  # Ypub (multisig p2wsh-p2sh) -> treat as 48'
                0x02aa7ed3: 48,  # Zpub (multisig p2wsh) -> 48'
                # Testnet
                0x043587CF: 44,  # tpub (assume BIP44)
                0x044a5262: 49,  # upub (BIP49 testnet)
                0x045f1cf6: 84,  # vpub (BIP84 testnet)
                0x024289ef: 48,  # Upub (multisig p2wsh-p2sh)
                0x02575483: 48,  # Vpub (multisig p2wsh)
            }
            coin_type = 1 if network == "testnet" else 0
            purpose = purpose_map.get(version_int)

            # Simple cases
            if depth == 0:
                return []
            if depth == 1:
                # Only child index known
                return [child_num_val]
            if depth == 2:
                if purpose is not None:
                    return [(purpose | 0x80000000), (coin_type | 0x80000000)]
                return None
            if depth == 3:
                if purpose is not None:
                    return [
                        (purpose | 0x80000000),
                        (coin_type | 0x80000000),
                        child_num_val,  # already includes hardened bit if set
                    ]
                return None
            # Deeper (account-level or change/address) xpubs not expected here; skip.
            return None

        serialized_xpub: Optional[bytes] = None
        root_pubkey: Optional[bytes] = None
        root_fp: Optional[bytes] = None
        xpub_depth: Optional[int] = None
        xpub_childnum: Optional[int] = None
        xpub_version_int: Optional[int] = None
        if xpub:
            (
                serialized_xpub,
                root_pubkey,
                root_fp,
                xpub_depth,
                xpub_childnum,
                xpub_version_int,
            ) = _parse_xpub(xpub)

        # Attempt derivation mapping address->(pubkey, path indices)
        addr_deriv_map: Dict[str, Tuple[bytes, List[int]]] = {}
        if include_derivations and xpub and _derive_addr_internal and root_fp and root_pubkey:
            # Collect all addresses we need to map (inputs + outputs)
            needed = {i.get("address", "") for i in inputs} | {o.get("address", "") for o in outputs}
            # Determine probable derivation purpose from first address (assume BIP84 for bc1/tb1 q)
            # We'll just scan both chains up to a reasonable bound (e.g., 150) or until resolved
            remaining = set(a for a in needed if a)
            MAX_SCAN = 150
            for change_flag in (False, True):
                for idx in range(MAX_SCAN):
                    if not remaining:
                        break
                    try:
                        res = _derive_addr_internal(xpub, index=idx, change=change_flag, network=self.network)
                        if res.get("success"):
                            addr = res.get("address")
                            if addr in remaining:
                                pub_hex = res.get("public_key") or ""
                                if pub_hex.startswith("0x"):
                                    pub_hex = pub_hex[2:]
                                pub_bytes = bytes.fromhex(pub_hex)
                                # Parse path into indices (skip leading 'm')
                                path = res.get("path", "")
                                indices: List[int] = []
                                if path.startswith("m/"):
                                    for comp in path[2:].split("/"):
                                        hardened = comp.endswith("'")
                                        num_str = comp.rstrip("'")
                                        try:
                                            val = int(num_str)
                                            if hardened:
                                                val |= 0x80000000
                                            indices.append(val)
                                        except Exception:
                                            indices = []
                                            break
                                addr_deriv_map[addr] = (pub_bytes, indices)
                                remaining.remove(addr)
                    except Exception:
                        continue

        # Global Map
        global_map = b""
        global_map += self._write_key_value(PSBT_GLOBAL_UNSIGNED_TX, b"", tx_bytes)
        global_map += self._write_key_value(PSBT_GLOBAL_VERSION, b"", struct.pack("<I", 0))  # PSBT v0
        if serialized_xpub and root_fp and xpub_depth is not None and xpub_childnum is not None and xpub_version_int is not None:
            path_indices = _guess_xpub_path(xpub_depth, xpub_childnum, xpub_version_int, self.network)
            if path_indices is not None and len(path_indices) == xpub_depth:
                value = root_fp + b"".join(struct.pack("<I", i) for i in path_indices)
                global_map += self._write_key_value(PSBT_GLOBAL_XPUB, serialized_xpub, value)
            # else: skip adding GLOBAL_XPUB to avoid depth/path mismatch
        # Proprietary global entries (0xFC) using provided tuples
        if proprietary:
            for prefix, subtype, keydata, value in proprietary:
                # key format: 0xFC | prefix | subtype | keydata  (prefix length embedded in CompactSize when writing key already handled by _write_key_value)
                pkey = b"\xFC" + prefix + bytes([subtype]) + keydata
                # BIP174 expects we write length(key) + key then length(value) + value; reuse _write_key_value logic by faking type
                # We can't directly call _write_key_value (it prepends type) so replicate minimal logic here:
                k = pkey
                global_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(value)) + value
        global_map += b"\x00"  # separator
        psbt_bytes += global_map

        # Input Maps
        for idx_inp, inp in enumerate(inputs):
            input_map = b""

            prev_tx = self._fetch_transaction(inp["txid"])
            if prev_tx:
                input_map += self._write_key_value(PSBT_IN_NON_WITNESS_UTXO, b"", prev_tx)
            else:
                # Witness UTXO mínim: amount (8 bytes) + scriptPubKey
                witness_utxo = struct.pack("<Q", int(inp.get("value_satoshis", 0)))
                try:
                    version_or_prefix, addr_hash = self._decode_address(inp.get("address", ""))
                    address_str = inp.get("address", "")
                    if address_str.startswith(("bc1", "tb1")):
                        wver = version_or_prefix
                        if wver == 0 and len(addr_hash) == 20:  # P2WPKH
                            script = b"\x00\x14" + addr_hash
                        elif wver == 0 and len(addr_hash) == 32:  # P2WSH
                            script = b"\x00\x20" + addr_hash
                        elif wver == 1 and len(addr_hash) == 32:  # P2TR
                            script = b"\x51\x20" + addr_hash
                        else:
                            raise ValueError("Unsupported segwit witness program for input")
                    elif version_or_prefix in (0x00, 0x6F) and len(addr_hash) >= 20:
                        script = b"\x76\xa9\x14" + addr_hash[:20] + b"\x88\xac"  # P2PKH
                    elif version_or_prefix in (0x05, 0xC4) and len(addr_hash) >= 20:
                        script = b"\xa9\x14" + addr_hash[:20] + b"\x87"  # P2SH
                    else:
                        raise ValueError("Unsupported legacy address type for input")
                    witness_utxo += self._compact_size(len(script)) + script
                except Exception as e:
                    raise ValueError(
                        f"Failed to build witness_utxo script for input address {inp.get('address')}: {e}"
                    )

                input_map += self._write_key_value(PSBT_IN_WITNESS_UTXO, b"", witness_utxo)

            # SIGHASH_ALL per defecte (no configuració avançada encara)
            input_map += self._write_key_value(PSBT_IN_SIGHASH_TYPE, b"", struct.pack("<I", 1))
            # BIP32 derivation (if known)
            if include_derivations and xpub and root_fp and inp.get("address") in addr_deriv_map:
                pub, indices = addr_deriv_map[inp.get("address")]  # type: ignore
                val = root_fp + b"".join(struct.pack("<I", i) for i in indices)
                input_map += self._write_key_value(PSBT_IN_BIP32_DERIVATION, pub, val)
            input_map += b"\x00"
            psbt_bytes += input_map

        # Output Maps (with optional derivations)
        for out in outputs:
            output_map = b""
            if include_derivations and xpub and root_fp and out.get("address") in addr_deriv_map:
                pub, indices = addr_deriv_map[out.get("address")]  # type: ignore
                val = root_fp + b"".join(struct.pack("<I", i) for i in indices)
                output_map += self._write_key_value(PSBT_OUT_BIP32_DERIVATION, pub, val)
            output_map += b"\x00"
            psbt_bytes += output_map

        psbt_base64 = base64.b64encode(psbt_bytes).decode("ascii")
        result: Dict = {
            "success": True,
            "psbt": psbt_base64,
            "num_inputs": len(inputs),
            "num_outputs": len(outputs),
            "unsigned_tx_hex": tx_bytes.hex(),
            "inputs": inputs,
            "outputs": outputs,
            "version": version,
        }
        return result

    # Backwards compatibility helper (antic comportament string)
    def create_psbt_base64(
        self,
        inputs: List[Dict],
        outputs: List[Dict],
        locktime: int = 0,
        version: int = 2,
        xpub: Optional[str] = None,
    ) -> str:
        return self.create_psbt(inputs, outputs, locktime, version, xpub)["psbt"]

    def decode_psbt(self, psbt_base64: str) -> Dict:
        """Decodificador simple per inspeccionar camps bàsics."""
        try:
            psbt_bytes = base64.b64decode(psbt_base64)
            if not psbt_bytes.startswith(PSBT_MAGIC):
                raise ValueError("No és un PSBT vàlid (magic bytes incorrectes)")

            offset = len(PSBT_MAGIC)
            result = {
                "version": 0,
                "tx": None,
                "inputs": [],
                "outputs": [],
                "valid": True,
            }

            # Llegim global map senzillament
            while offset < len(psbt_bytes):
                key_len = psbt_bytes[offset]
                offset += 1
                if key_len == 0:
                    break
                key = psbt_bytes[offset : offset + key_len]
                offset += key_len

                value_len, consumed = self._read_compact_size(psbt_bytes[offset:])
                offset += consumed
                value = psbt_bytes[offset : offset + value_len]
                offset += value_len

                if key[0:1] == PSBT_GLOBAL_UNSIGNED_TX:
                    result["tx"] = value.hex()
                elif key[0:1] == PSBT_GLOBAL_VERSION and len(value) >= 4:
                    result["version"] = struct.unpack("<I", value[:4])[0]

            # Comptes aproximats
            separator_count = psbt_bytes.count(b"\x00")
            estimated_inputs = max(0, (separator_count - 1) // 2)
            result["num_inputs"] = estimated_inputs
            result["num_outputs"] = len(getattr(self, "_last_outputs", []))

            return result
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _read_compact_size(self, data: bytes) -> Tuple[int, int]:
        try:
            return codec.compact_size_decode(data)
        except Exception:
            return 0, 0


# ==========================
# API d'alt nivell
# ==========================
def create_transaction_psbt(
    xpub: str,
    recipient_address: str,
    amount_btc: float,
    utxos: List[Dict],
    change_address: Optional[str] = None,
    network: str = "testnet",
    fee_rate: int = 10,
    fee_satoshis: Optional[int] = None,
    manual_selected_utxos: Optional[List[Dict]] = None,
) -> Dict:
    """
    Crea un PSBT per una transacció:
      - Si 'fee_satoshis' és None, calcula la comissió a partir de 'fee_rate' (sat/vB) segons vbytes reals.
      - Si 'fee_satoshis' ve informat, es respecta (mode compatibilitat).
    """
    creator = PSBTCreator(network)

    # amount a sats
    amount_satoshis = int(round(amount_btc * 100_000_000))

    # Manual UTXO selection path (explicit list bypasses greedy selection)
    if manual_selected_utxos:
        selected_utxos = manual_selected_utxos
        total_input = sum(int(u.get("value_satoshis", 0)) for u in selected_utxos)
        if fee_satoshis is None:
            # Estimate fee with chosen inputs (attempt both no-change and change scenarios)
            addrs = [u.get("address", "") for u in selected_utxos]
            vb_no = _estimate_vbytes(addrs, recipient_address, include_change=False, change_addr=None)
            est_fee_no = math.ceil(int(fee_rate) * vb_no)
            change_no = total_input - amount_satoshis - est_fee_no
            if change_no < DUST_THRESHOLD and total_input >= amount_satoshis + est_fee_no:
                fee_satoshis = est_fee_no
                change_satoshis = 0
            else:
                # Need change output
                vb_ch = _estimate_vbytes(addrs, recipient_address, include_change=True, change_addr=change_address)
                est_fee_ch = math.ceil(int(fee_rate) * vb_ch)
                change_ch = total_input - amount_satoshis - est_fee_ch
                if change_ch >= DUST_THRESHOLD and total_input >= amount_satoshis + est_fee_ch:
                    if not change_address:
                        return {"success": False, "error": "Change output necessari però no s'ha proporcionat change_address (manual_selected_utxos)."}
                    fee_satoshis = est_fee_ch
                    change_satoshis = change_ch
                else:
                    return {"success": False, "error": "Fons insuficients amb les UTXOs manuals seleccionades o canvi < dust."}
        else:
            # Fee fixed; validate funds
            if total_input < amount_satoshis + int(fee_satoshis):
                return {"success": False, "error": "Fons insuficients amb les UTXOs manuals seleccionades."}
            change_satoshis = total_input - amount_satoshis - int(fee_satoshis)
    else:
        # Selecció d’UTXOs i càlcul de fee (automàtic o compatibilitat)
        if fee_satoshis is None:
            sel = _select_utxos_vbytes(
                utxos=utxos,
                amount_sats=amount_satoshis,
                fee_rate_sat_vb=int(fee_rate),
                recipient_addr=recipient_address,
                change_addr=change_address,
            )
            if not sel["success"]:
                return {"success": False, "error": sel["error"]}
            selected_utxos = sel["selected_utxos"]
            fee_satoshis = sel["fee_satoshis"]
            change_satoshis = sel["change_satoshis"]
            total_input = sum(int(u.get("value_satoshis", 0)) for u in selected_utxos)
        else:
            # Compatibilitat: selecció simple segons fee fixa
            selected_utxos = []
            total_input = 0
            for utxo in sorted(utxos, key=lambda x: x.get("value_satoshis", 0), reverse=True):
                selected_utxos.append(utxo)
                total_input += int(utxo.get("value_satoshis", 0))
                if total_input >= amount_satoshis + int(fee_satoshis):
                    break
            if total_input < amount_satoshis + int(fee_satoshis):
                return {
                    "success": False,
                    "error": f"Fons insuficients. Necessari: {((amount_satoshis + int(fee_satoshis))/1e8):.8f} BTC, Disponible: {(total_input/1e8):.8f} BTC",
                }
            change_satoshis = total_input - amount_satoshis - int(fee_satoshis)

    # Outputs
    outputs = [{"address": recipient_address, "value": amount_satoshis}]
    if change_satoshis > DUST_THRESHOLD:
        if not change_address:
            return {
                "success": False,
                "error": "Change output required (> dust) but no change_address provided. Derive a change address from the XPUB and retry.",
            }
        outputs.append({"address": change_address, "value": int(change_satoshis)})

    # Crear PSBT
    try:
        psbt_build = creator.create_psbt(inputs=selected_utxos, outputs=outputs, xpub=xpub)

        psbt_base64 = psbt_build["psbt"]
        return {
            "success": True,
            "psbt": psbt_base64,
            "psbt_hex": base64.b64decode(psbt_base64).hex(),
            "total_input_btc": total_input / 1e8,
            "amount_btc": amount_btc,
            "fee_btc": int(fee_satoshis) / 1e8,
            "change_btc": (change_satoshis / 1e8) if change_satoshis > DUST_THRESHOLD else 0,
            "num_inputs": psbt_build.get("num_inputs", len(selected_utxos)),
            "num_outputs": psbt_build.get("num_outputs", len(outputs)),
            "selected_utxos": selected_utxos,
            "outputs": outputs,
            "network": network,
            "unsigned_tx_hex": psbt_build.get("unsigned_tx_hex"),
        }
    except Exception as e:
        return {"success": False, "error": f"Error creant PSBT: {str(e)}"}


if __name__ == "__main__":
    # Test bàsic manual (només per desenvolupament)
    print("🧪 Test del creador de PSBT")
    print("=" * 70)

    # UTXO de prova
    test_utxos = [
        {
            "txid": "1234567890abcdef" * 4,
            "vout": 0,
            "value_satoshis": 100_000,
            "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
        }
    ]

    result = create_transaction_psbt(
        xpub="tpubTEST",
        recipient_address="tb1qg2t9c2mamc2r9l68v9r80xkqz0r5yyptqetk6k",
        amount_btc=0.0005,
        utxos=test_utxos,
        network="testnet",
        fee_rate=10,  # sat/vB
    )

    if result["success"]:
        print("✅ PSBT creat correctament!")
        print(f"   PSBT (Base64): {result['psbt'][:60]}...")
        print(f"   Inputs: {result['num_inputs']}")
        print(f"   Outputs: {result['num_outputs']}")
        print(f"   Fee: {result['fee_btc']:.8f} BTC")
    else:
        print(f"❌ Error: {result['error']}")
