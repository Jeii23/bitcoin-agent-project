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
import random

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
PSBT_OUT_TAP_INTERNAL_KEY = b"\x05"
PSBT_OUT_TAP_TREE = b"\x06"
PSBT_OUT_TAP_BIP32_DERIVATION = b"\x07"

# PSBT Input Types for Taproot (BIP-371)
PSBT_IN_TAP_KEY_SIG = b"\x13"
PSBT_IN_TAP_SCRIPT_SIG = b"\x14"
PSBT_IN_TAP_LEAF_SCRIPT = b"\x15"
PSBT_IN_TAP_BIP32_DERIVATION = b"\x16"
PSBT_IN_TAP_INTERNAL_KEY = b"\x17"
PSBT_IN_TAP_MERKLE_ROOT = b"\x18"

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
        "error": f"Fons insuficients. Necessari: {amount_sats} sats + fee, Disponible: {total_in} sats",
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
            # Primer intent: Blockstream
            response = requests.get(f"{self.api_base}/tx/{txid}/hex", timeout=20)
            if response.status_code == 200:
                return bytes.fromhex(response.text.strip())

            # Fallback: mempool.space
            try:
                alt_base = (
                    "https://mempool.space/testnet/api"
                    if self.network == "testnet"
                    else "https://mempool.space/api"
                )
                r2 = requests.get(f"{alt_base}/tx/{txid}/hex", timeout=20)
                if r2.status_code == 200:
                    return bytes.fromhex(r2.text.strip())
            except Exception:
                # Ignore fallback errors and return None below
                pass

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
        include_derivations: bool = False,
        proprietary: Optional[List[Tuple[bytes, int, bytes, bytes]]] = None,
        include_global_xpub: bool = False,
        rbf: bool = False,
        prefer_legacy_witness_utxo: bool = False,
        extra_psbt: Optional[Dict] = None,  # globals/inputs/outputs: propietàries i K/V crus
    ) -> Dict:
        """
        Construeix una PSBT v0 sense polítiques per defecte.
        Accepta outputs per adreça o per script cru i admet injecció K/V.
        """

        # --- Helpers locals -----------------------------------------
        def _scriptpubkey_from_address(addr_str: str, ver_or_prefix: int, payload: bytes) -> bytes:
            # SegWit (bech32/bech32m)
            if addr_str.startswith(("bc1", "tb1", "bcrt1")):
                wver = ver_or_prefix
                prog = payload
                if wver == 0:
                    # OP_0 <pushlen> <programa>
                    return b"\x00" + bytes([len(prog)]) + prog
                else:
                    # OP_{wver} <pushlen> <programa>  (Taproot: wver=1)
                    return bytes([0x50 + wver]) + bytes([len(prog)]) + prog
            # P2SH
            if addr_str.startswith(("3", "2")):
                # OP_HASH160 0x14 <20-byte> OP_EQUAL
                return b"\xa9\x14" + payload + b"\x87"
            # P2PKH
            # '1' mainnet o 'm/n' testnet
            return b"\x76\xa9\x14" + payload + b"\x88\xac"

        def _pack_tx_output(value_sat: int, script_pubkey: bytes) -> bytes:
            return struct.pack("<Q", int(value_sat)) + self._compact_size(len(script_pubkey)) + script_pubkey

        # --- 1) TX unsigned (sense testimoni) -----------------------
        tx_bytes = b""
        tx_bytes += struct.pack("<I", version)

        # Inputs
        tx_bytes += self._compact_size(len(inputs))
        for inp in inputs:
            txid_bytes = bytes.fromhex(inp["txid"])[::-1]
            tx_bytes += txid_bytes
            tx_bytes += struct.pack("<I", int(inp["vout"]))
            tx_bytes += b"\x00"  # scriptSig buit
            # Per-input override si es proporciona, sinó rbf global, sinó seq final
            if "sequence" in inp:
                seq = int(inp["sequence"]) & 0xFFFFFFFF
            else:
                seq = 0xFFFFFFFD if rbf else 0xFFFFFFFF
            tx_bytes += struct.pack("<I", seq)

        # Outputs
        tx_bytes += self._compact_size(len(outputs))
        for out in outputs:
            value_sat = int(out["value"])
            if "script" in out and out["script"] is not None:
                # Accepta bytes o hex
                spk = out["script"] if isinstance(out["script"], (bytes, bytearray)) else bytes.fromhex(str(out["script"]))
                tx_bytes += _pack_tx_output(value_sat, spk)
            elif "script_hex" in out and out["script_hex"] is not None:
                spk = bytes.fromhex(str(out["script_hex"]))
                tx_bytes += _pack_tx_output(value_sat, spk)
            else:
                # Per adreça
                addr = str(out["address"]).strip()
                ver_or_prefix, payload = self._decode_address(addr)
                spk = _scriptpubkey_from_address(addr, ver_or_prefix, payload)
                tx_bytes += _pack_tx_output(value_sat, spk)

        tx_bytes += struct.pack("<I", locktime)
        self._last_outputs = outputs

        # --- 2) PSBT: magic + global + input maps + output maps -----
        psbt_bytes = PSBT_MAGIC

        # Helpers XPUB i derivacions
        serialized_xpub = None
        root_fp = None
        xpub_depth = None
        xpub_childnum = None
        xpub_version_int = None
        addr_deriv_map: Dict[str, Tuple[bytes, List[int]]] = {}

        if xpub:
            try:
                raw = codec.b58decode_check(xpub)
                if len(raw) == 78:
                    serialized_xpub = raw
                    xpub_version_int = int.from_bytes(raw[0:4], "big")
                    xpub_depth = raw[4]
                    root_fp = raw[5:9]
                    xpub_childnum = int.from_bytes(raw[9:13], "big")
                    addr_deriv_map = self._build_addr_deriv_map(xpub)
            except Exception:
                serialized_xpub = None
                root_fp = None

        # ---------- Global Map ----------
        g = (extra_psbt or {}).get("global", {}) if isinstance(extra_psbt, dict) else {}
        gv = g.get("version")

        global_map = b""
        global_map += self._write_key_value(PSBT_GLOBAL_UNSIGNED_TX, b"", tx_bytes)
        # PSBT v0 field
        global_map += self._write_key_value(PSBT_GLOBAL_VERSION, b"", struct.pack("<I", int(gv) if gv is not None else 0))

        # Fallback locktime i tx_modifiable
        if "fallback_locktime" in g:
            global_map += self._write_key_value(
                PSBT_GLOBAL_FALLBACK_LOCKTIME, b"", struct.pack("<I", int(g["fallback_locktime"]))
            )
        if "tx_modifiable" in g:
            tm = int(g["tx_modifiable"]) & 0xFF
            global_map += self._write_key_value(PSBT_GLOBAL_TX_MODIFIABLE, b"", bytes([tm]))

        # GLOBAL_XPUB opcional
        if (
            include_global_xpub and serialized_xpub and root_fp is not None
            and xpub_depth is not None and xpub_childnum is not None and xpub_version_int is not None
        ):
            path_indices = self._guess_xpub_path(xpub_depth, xpub_childnum, xpub_version_int, self.network)
            if path_indices is not None and len(path_indices) == xpub_depth:
                value = root_fp + b"".join(struct.pack("<I", i) for i in path_indices)
                global_map += self._write_key_value(PSBT_GLOBAL_XPUB, serialized_xpub, value)

        # Proprietary globals (API antiga)
        if proprietary:
            for prefix, subtype, keydata, value in proprietary:
                pkey = b"\xFC" + prefix + bytes([subtype]) + keydata
                global_map += codec.compact_size_encode(len(pkey)) + pkey + codec.compact_size_encode(len(value)) + value

        # Proprietary i K/V crus via extra_psbt
        for pref in (g.get("proprietary") or []):
            pfx = bytes.fromhex(pref["prefix_hex"])
            subtype = int(pref["subtype"]) & 0xFF
            kdata = bytes.fromhex(pref.get("keydata_hex", "")) if pref.get("keydata_hex") else b""
            val = bytes.fromhex(pref.get("value_hex", "")) if pref.get("value_hex") else b""
            k = b"\xFC" + pfx + bytes([subtype]) + kdata
            global_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(val)) + val

        for kv in (g.get("raw_kv") or []):
            k = bytes.fromhex(kv["key_hex"])
            v = bytes.fromhex(kv.get("value_hex", "")) if kv.get("value_hex") else b""
            global_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(v)) + v

        global_map += b"\x00"
        psbt_bytes += global_map

        # ---------- Input Maps ----------
        inputs_extras = (extra_psbt or {}).get("inputs", []) if isinstance(extra_psbt, dict) else []
        for idx_inp, inp in enumerate(inputs):
            input_map = b""

            # Selecció UTXO a incloure al PSBT
            # Prioritat:
            # 1) Non-witness UTXO si es passa explícit
            # 2) Witness UTXO si es passa explícit
            # 3) Si prefer_legacy_witness_utxo=True, intenta recollir prev_tx; si falla, intenta witness amb dades locals
            # 4) Si prefer_legacy_witness_utxo=False, intenta witness amb dades locals; si falla, recull prev_tx
            nwutxo_hex = inp.get("non_witness_utxo_hex")
            wutxo = inp.get("witness_utxo")  # dict amb value/scriptpubkey_hex
            spk_hex = inp.get("scriptpubkey_hex")
            address_str = inp.get("address")

            # 1) Non-witness explícit
            if nwutxo_hex:
                input_map += self._write_key_value(PSBT_IN_NON_WITNESS_UTXO, b"", bytes.fromhex(nwutxo_hex))

            # 2) Witness explícit
            elif wutxo:
                w_val = int(wutxo.get("value_satoshis") or wutxo.get("amount_sats") or wutxo.get("value") or inp.get("value_satoshis", 0))
                w_spk = bytes.fromhex(wutxo.get("scriptpubkey_hex")) if wutxo.get("scriptpubkey_hex") else None
                if w_spk is None and spk_hex:
                    w_spk = bytes.fromhex(spk_hex)
                if w_spk is None and address_str:
                    ver_or_prefix, payload = self._decode_address(str(address_str).strip())
                    w_spk = _scriptpubkey_from_address(str(address_str).strip(), ver_or_prefix, payload)
                if w_spk is None:
                    return {"success": False, "error": f"Input {idx_inp}: falta scriptPubKey per a witness_utxo."}
                witness_utxo = struct.pack("<Q", w_val) + self._compact_size(len(w_spk)) + w_spk
                input_map += self._write_key_value(PSBT_IN_WITNESS_UTXO, b"", witness_utxo)

            else:
                # 3) i 4) segons preferència
                def _try_build_witness_from_local() -> Optional[bytes]:
                    val = int(inp.get("value_satoshis", 0))
                    spk = bytes.fromhex(spk_hex) if spk_hex else None
                    if spk is None and address_str:
                        ver_or_prefix, payload = self._decode_address(str(address_str).strip())
                        spk = _scriptpubkey_from_address(str(address_str).strip(), ver_or_prefix, payload)
                    if spk is None or val <= 0:
                        return None
                    return struct.pack("<Q", val) + self._compact_size(len(spk)) + spk

                if prefer_legacy_witness_utxo:
                    prev_tx = self._fetch_transaction(inp["txid"])
                    if prev_tx:
                        input_map += self._write_key_value(PSBT_IN_NON_WITNESS_UTXO, b"", prev_tx)
                    else:
                        w = _try_build_witness_from_local()
                        if w is None:
                            return {"success": False, "error": f"Input {idx_inp}: no es pot obtenir UTXO (ni prev_tx ni witness dades)."}
                        input_map += self._write_key_value(PSBT_IN_WITNESS_UTXO, b"", w)
                else:
                    w = _try_build_witness_from_local()
                    if w is not None:
                        input_map += self._write_key_value(PSBT_IN_WITNESS_UTXO, b"", w)
                    else:
                        prev_tx = self._fetch_transaction(inp["txid"])
                        if not prev_tx:
                            return {"success": False, "error": f"Input {idx_inp}: no es pot obtenir UTXO (ni witness dades ni prev_tx)."}
                        input_map += self._write_key_value(PSBT_IN_NON_WITNESS_UTXO, b"", prev_tx)

            # SIGHASH per input (ALL=1 per defecte, override via extras)
            sighash_val = 1
            if idx_inp < len(inputs_extras) and isinstance(inputs_extras[idx_inp], dict) and "sighash_type" in inputs_extras[idx_inp]:
                sighash_val = int(inputs_extras[idx_inp]["sighash_type"])
            input_map += self._write_key_value(PSBT_IN_SIGHASH_TYPE, b"", struct.pack("<I", sighash_val))

            # Injecta claus estàndard
            if idx_inp < len(inputs_extras) and isinstance(inputs_extras[idx_inp], dict):
                ie = inputs_extras[idx_inp]
                if "redeem_script_hex" in ie:
                    input_map += self._write_key_value(PSBT_IN_REDEEM_SCRIPT, b"", bytes.fromhex(ie["redeem_script_hex"]))
                if "witness_script_hex" in ie:
                    input_map += self._write_key_value(PSBT_IN_WITNESS_SCRIPT, b"", bytes.fromhex(ie["witness_script_hex"]))
                if "por_commitment_hex" in ie:
                    input_map += self._write_key_value(PSBT_IN_POR_COMMITMENT, b"", bytes.fromhex(ie["por_commitment_hex"]))
                for kname, ptype in [
                    ("ripemd160_preimage_hex", PSBT_IN_RIPEMD160),
                    ("sha256_preimage_hex",   PSBT_IN_SHA256),
                    ("hash160_preimage_hex",  PSBT_IN_HASH160),
                    ("hash256_preimage_hex",  PSBT_IN_HASH256),
                ]:
                    if kname in ie:
                        input_map += self._write_key_value(ptype, b"", bytes.fromhex(ie[kname]))

                # BIP32 derivations específiques
                if "bip32_derivations" in ie:
                    for d in ie["bip32_derivations"]:
                        pub = bytes.fromhex(d["pubkey_hex"])
                        fp = bytes.fromhex(d["fingerprint_hex"])
                        inds = b"".join(struct.pack("<I", int(x)) for x in d["path_indices"])
                        input_map += self._write_key_value(PSBT_IN_BIP32_DERIVATION, pub, fp + inds)

                # Proprietary i K/V crus d'input
                for pref in (ie.get("proprietary") or []):
                    pfx = bytes.fromhex(pref["prefix_hex"])
                    subtype = int(pref["subtype"]) & 0xFF
                    kdata = bytes.fromhex(pref.get("keydata_hex", "")) if pref.get("keydata_hex") else b""
                    val = bytes.fromhex(pref.get("value_hex", "")) if pref.get("value_hex") else b""
                    k = b"\xFC" + pfx + bytes([subtype]) + kdata
                    input_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(val)) + val

                for kv in (ie.get("raw_kv") or []):
                    k = bytes.fromhex(kv["key_hex"])
                    v = bytes.fromhex(kv.get("value_hex", "")) if kv.get("value_hex") else b""
                    input_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(v)) + v

                # Taproot-specific input fields (BIP-371)
                if "tap_key_sig_hex" in ie:
                    input_map += self._write_key_value(PSBT_IN_TAP_KEY_SIG, b"", bytes.fromhex(ie["tap_key_sig_hex"]))
                if "tap_internal_key_hex" in ie:
                    input_map += self._write_key_value(PSBT_IN_TAP_INTERNAL_KEY, b"", bytes.fromhex(ie["tap_internal_key_hex"]))
                if "tap_merkle_root_hex" in ie:
                    input_map += self._write_key_value(PSBT_IN_TAP_MERKLE_ROOT, b"", bytes.fromhex(ie["tap_merkle_root_hex"]))
                # tap_leaf_scripts: list of {"control_block_hex": ..., "script_hex": ..., "leaf_version": int}
                for tls in (ie.get("tap_leaf_scripts") or []):
                    cb = bytes.fromhex(tls["control_block_hex"])
                    scr = bytes.fromhex(tls["script_hex"])
                    lv = int(tls.get("leaf_version", 0xC0)) & 0xFF
                    # Key = control_block, Value = script + leaf_version
                    input_map += self._write_key_value(PSBT_IN_TAP_LEAF_SCRIPT, cb, scr + bytes([lv]))
                # tap_script_sigs: list of {"xonly_pubkey_hex": ..., "leaf_hash_hex": ..., "signature_hex": ...}
                for tss in (ie.get("tap_script_sigs") or []):
                    xonly = bytes.fromhex(tss["xonly_pubkey_hex"])
                    lh = bytes.fromhex(tss["leaf_hash_hex"])
                    sig = bytes.fromhex(tss["signature_hex"])
                    input_map += self._write_key_value(PSBT_IN_TAP_SCRIPT_SIG, xonly + lh, sig)
                # tap_bip32_derivations: list of {"xonly_pubkey_hex", "leaf_hashes_hex" (list), "fingerprint_hex", "path_indices"}
                for tbd in (ie.get("tap_bip32_derivations") or []):
                    xonly = bytes.fromhex(tbd["xonly_pubkey_hex"])
                    lhashes = b"".join(bytes.fromhex(h) for h in (tbd.get("leaf_hashes_hex") or []))
                    fp = bytes.fromhex(tbd["fingerprint_hex"])
                    inds = b"".join(struct.pack("<I", int(x)) for x in tbd["path_indices"])
                    nleaves = len(tbd.get("leaf_hashes_hex") or [])
                    val = bytes([nleaves]) + lhashes + fp + inds
                    input_map += self._write_key_value(PSBT_IN_TAP_BIP32_DERIVATION, xonly, val)

            # Derivacions inferides només si les demanes
            if include_derivations and xpub and root_fp and address_str and address_str in addr_deriv_map:
                pub, indices = addr_deriv_map[address_str]  # type: ignore
                val = root_fp + b"".join(struct.pack("<I", i) for i in indices)
                input_map += self._write_key_value(PSBT_IN_BIP32_DERIVATION, pub, val)

            input_map += b"\x00"
            psbt_bytes += input_map

        # ---------- Output Maps ----------
        outputs_extras = (extra_psbt or {}).get("outputs", []) if isinstance(extra_psbt, dict) else []
        for j, out in enumerate(outputs):
            output_map = b""

            if j < len(outputs_extras) and isinstance(outputs_extras[j], dict):
                oe = outputs_extras[j]
                if "bip32_derivations" in oe:
                    for d in oe["bip32_derivations"]:
                        pub = bytes.fromhex(d["pubkey_hex"])
                        fp = bytes.fromhex(d["fingerprint_hex"])
                        inds = b"".join(struct.pack("<I", int(x)) for x in d["path_indices"])
                        output_map += self._write_key_value(PSBT_OUT_BIP32_DERIVATION, pub, fp + inds)
                if "redeem_script_hex" in oe:
                    output_map += self._write_key_value(PSBT_OUT_REDEEM_SCRIPT, b"", bytes.fromhex(oe["redeem_script_hex"]))
                if "witness_script_hex" in oe:
                    output_map += self._write_key_value(PSBT_OUT_WITNESS_SCRIPT, b"", bytes.fromhex(oe["witness_script_hex"]))

                # Taproot-specific output fields (BIP-371)
                if "tap_internal_key_hex" in oe:
                    output_map += self._write_key_value(PSBT_OUT_TAP_INTERNAL_KEY, b"", bytes.fromhex(oe["tap_internal_key_hex"]))
                # tap_tree: list of {"depth": int, "leaf_version": int, "script_hex": str}
                if "tap_tree" in oe and oe["tap_tree"]:
                    tree_bytes = b""
                    for leaf in oe["tap_tree"]:
                        d = int(leaf["depth"]) & 0xFF
                        lv = int(leaf.get("leaf_version", 0xC0)) & 0xFF
                        scr = bytes.fromhex(leaf["script_hex"])
                        tree_bytes += bytes([d, lv]) + codec.compact_size_encode(len(scr)) + scr
                    output_map += self._write_key_value(PSBT_OUT_TAP_TREE, b"", tree_bytes)
                # tap_bip32_derivations for outputs
                for tbd in (oe.get("tap_bip32_derivations") or []):
                    xonly = bytes.fromhex(tbd["xonly_pubkey_hex"])
                    lhashes = b"".join(bytes.fromhex(h) for h in (tbd.get("leaf_hashes_hex") or []))
                    fp = bytes.fromhex(tbd["fingerprint_hex"])
                    inds = b"".join(struct.pack("<I", int(x)) for x in tbd["path_indices"])
                    nleaves = len(tbd.get("leaf_hashes_hex") or [])
                    val = bytes([nleaves]) + lhashes + fp + inds
                    output_map += self._write_key_value(PSBT_OUT_TAP_BIP32_DERIVATION, xonly, val)

                # Proprietary i K/V crus d'output
                for pref in (oe.get("proprietary") or []):
                    pfx = bytes.fromhex(pref["prefix_hex"])
                    subtype = int(pref["subtype"]) & 0xFF
                    kdata = bytes.fromhex(pref.get("keydata_hex", "")) if pref.get("keydata_hex") else b""
                    val = bytes.fromhex(pref.get("value_hex", "")) if pref.get("value_hex") else b""
                    k = b"\xFC" + pfx + bytes([subtype]) + kdata
                    output_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(val)) + val

                for kv in (oe.get("raw_kv") or []):
                    k = bytes.fromhex(kv["key_hex"])
                    v = bytes.fromhex(kv.get("value_hex", "")) if kv.get("value_hex") else b""
                    output_map += codec.compact_size_encode(len(k)) + k + codec.compact_size_encode(len(v)) + v

            # Derivacions inferides opcionalment
            addr_out = out.get("address")
            if include_derivations and xpub and root_fp and addr_out and addr_out in addr_deriv_map:
                pub, indices = addr_deriv_map[addr_out]  # type: ignore
                val = root_fp + b"".join(struct.pack("<I", i) for i in indices)
                output_map += self._write_key_value(PSBT_OUT_BIP32_DERIVATION, pub, val)

            output_map += b"\x00"
            psbt_bytes += output_map

        # --- 3) Serialitza i retorna --------------------------------
        psbt_base64 = base64.b64encode(psbt_bytes).decode("ascii")
        if not return_dict:
            return {"psbt": psbt_base64}

        return {
            "psbt": psbt_base64,
            "inputs": inputs,
            "outputs": outputs,
            "locktime": locktime,
            "version": version,
            "rbf": bool(rbf),
            "include_global_xpub": bool(include_global_xpub),
            "include_derivations": bool(include_derivations),
        }


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

    def _read_compact_size(self, data: bytes) -> Tuple[int, int]:
        """Wrapper segur per CompactSize (varint) que no llença excepcions."""
        try:
            return codec.compact_size_decode(data)
        except Exception:
            return 0, 0

    def decode_psbt(self, psbt_base64: str) -> Dict:
        """Decodificador per comprovar validesa i extreure comptadors reals.

        - Valida base64 i magic bytes.
        - Llegeix el Global Map i extreu la transacció unsigned si hi és.
        - Parsea la transacció unsigned per obtenir num_inputs/num_outputs reals.
        - Opcionalment detecta si cada input té "witness_utxo" present.
        """
        try:
            # Neteja espais i intenta padding si cal
            if isinstance(psbt_base64, str):
                b64_clean = "".join(psbt_base64.split())
            else:
                return {"valid": False, "error": "PSBT ha de ser str base64"}
            try:
                psbt_bytes = base64.b64decode(b64_clean, validate=True)
            except Exception:
                pad = (-len(b64_clean)) % 4
                psbt_bytes = base64.b64decode(b64_clean + ("=" * pad), validate=False)

            if not psbt_bytes.startswith(PSBT_MAGIC):
                raise ValueError("No és un PSBT vàlid (magic bytes incorrectes)")

            offset = len(PSBT_MAGIC)
            result: Dict = {
                "valid": True,
                "version": 0,  # PSBT v0
                "tx": None,
            }

            # Helpers locals
            def _read_cs(buf: bytes, ofs: int) -> Tuple[int, int]:
                val, used = self._read_compact_size(buf[ofs:])
                if used == 0 and val == 0 and buf[ofs:ofs+1] != b"\x00":
                    # Defensa per casos estranys
                    raise ValueError("CompactSize invàlid")
                return val, used

            # 1) Global map: clau-valor fins separador 0x00
            unsigned_tx_bytes: Optional[bytes] = None
            while offset < len(psbt_bytes):
                # key length (CompactSize)
                if psbt_bytes[offset:offset+1] == b"\x00":
                    offset += 1
                    break  # end of global map
                key_len, used = _read_cs(psbt_bytes, offset)
                offset += used
                key = psbt_bytes[offset: offset + key_len]
                offset += key_len
                val_len, used = _read_cs(psbt_bytes, offset)
                offset += used
                value = psbt_bytes[offset: offset + val_len]
                offset += val_len

                if key[:1] == PSBT_GLOBAL_UNSIGNED_TX:
                    unsigned_tx_bytes = value
                elif key[:1] == PSBT_GLOBAL_VERSION and len(value) >= 4:
                    result["version"] = struct.unpack("<I", value[:4])[0]

            # 2) Comptes reals a partir de l'unsigned tx
            n_in = 0
            n_out = 0
            tx_version = None
            locktime = None
            if unsigned_tx_bytes:
                try:
                    n_in, n_out, tx_version, locktime = self._parse_unsigned_tx_counts(unsigned_tx_bytes)
                except Exception:
                    pass

            result["num_inputs"] = int(n_in)
            result["num_outputs"] = int(n_out)
            if unsigned_tx_bytes is not None:
                result["tx"] = unsigned_tx_bytes.hex()
            if tx_version is not None:
                result["tx_version"] = tx_version
            if locktime is not None:
                result["locktime"] = locktime

            # 3) (Opcional) detectar PSBT_IN_WITNESS_UTXO per input
            if n_in:
                has_wit = []
                for _i in range(n_in):
                    saw_wit = False
                    while True:
                        if psbt_bytes[offset:offset+1] == b"\x00":
                            offset += 1
                            break  # end of this input map
                        klen, used = _read_cs(psbt_bytes, offset)
                        offset += used
                        key = psbt_bytes[offset: offset + klen]
                        offset += klen
                        vlen, used = _read_cs(psbt_bytes, offset)
                        offset += used
                        # We don't need the value bytes themselves; just skip
                        offset += vlen
                        if key[:1] == PSBT_IN_WITNESS_UTXO:
                            saw_wit = True
                    has_wit.append(saw_wit)
                result["has_witness_utxo"] = has_wit

            # 4) Saltar output maps si cal (no inspectem contingut)
            for _o in range(n_out):
                while True:
                    if psbt_bytes[offset:offset+1] == b"\x00":
                        offset += 1
                        break
                    klen, used = _read_cs(psbt_bytes, offset)
                    offset += used
                    offset += klen
                    vlen, used = _read_cs(psbt_bytes, offset)
                    offset += used
                    offset += vlen

            return result
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _parse_unsigned_tx_counts(self, tx_bytes: bytes) -> Tuple[int, int, int, int]:
        """Parsea una transacció unsigned per obtenir (n_inputs, n_outputs, tx_version, locktime)."""
        ofs = 0
        if len(tx_bytes) < 10:
            raise ValueError("Unsigned tx massa curta")
        tx_version = struct.unpack("<I", tx_bytes[ofs:ofs+4])[0]
        ofs += 4
        # inputs
        n_in, used = codec.compact_size_decode(tx_bytes[ofs:])
        ofs += used
        for _ in range(n_in):
            ofs += 32  # txid
            ofs += 4   # vout
            slen, used = codec.compact_size_decode(tx_bytes[ofs:])
            ofs += used
            ofs += slen  # scriptSig
            ofs += 4   # sequence
        # outputs
        n_out, used = codec.compact_size_decode(tx_bytes[ofs:])
        ofs += used
        for _ in range(n_out):
            ofs += 8  # amount
            slen, used = codec.compact_size_decode(tx_bytes[ofs:])
            ofs += used
            ofs += slen  # scriptPubKey
        locktime = struct.unpack("<I", tx_bytes[ofs:ofs+4])[0]
        return n_in, n_out, tx_version, locktime


# Senzill helper d'alt nivell per a testing: decode directament
def decode_psbt(psbt_base64: str, network: str = "testnet") -> Dict:
    """Decodifica un PSBT (base64) retornant un dict estable amb 'valid' i comptadors útils.

    Exemple mínim:
      dec = decode_psbt(b64)
      assert dec["valid"]
      assert dec["num_inputs"] >= 0
    """
    return PSBTCreator(network=network).decode_psbt(psbt_base64)


# ==========================
# API d'alt nivell
# ==========================
def create_transaction_psbt(
    xpub: str,
    utxos: List[Dict],                          # llistat d’UTXOs disponibles (no s’usa per seleccionar)
    # Mode explícit recomanat: defineix TOTS els outputs i l’ordre tal com vols que apareguin
    outputs: Optional[List[Dict]] = None,       # [{'address': str OR 'script': bytes/hex, 'value': int}, ...]
    # Mode simple opcional (un sol recipient) — només si vols una via ràpida i explícita
    recipient_address: Optional[str] = None,
    amount_sats: Optional[int] = None,
    change_address: Optional[str] = None,

    # Paràmetres d’entorn i serialització
    network: str = "testnet",
    include_global_xpub: bool = False,
    include_keypaths: bool = False,
    rbf: Optional[bool] = None,                 # None = no tocar nSequence; True/False = fixar
    prefer_legacy_witness_utxo: Optional[bool] = None,
    locktime_override: Optional[int] = None,

    # Inputs i fee decidits per l’agent
    manual_selected_utxos: Optional[List[Dict]] = None,  # ORDRE = ORDRE A LA TX
    fee_satoshis: Optional[int] = None,                  # només per al mode simple

    # Camps addicionals PSBT (globals/inputs/outputs i propietaris)
    psbt_extras: Optional[Dict] = None,

    # Paràmetres mantinguts per compatibilitat però IGNORATS per evitar biaix
    shuffle_inputs: bool = False,
    shuffle_outputs: bool = False,
    avoid_change: bool = False,
    min_change_sats: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    privacy_preset: Optional[str] = None,
) -> Dict:
    """
    Crea una PSBT sense decisions de privacitat per defecte.
    — No fa coin selection, no estima fees, no reordena inputs ni outputs, no aplica llindars de dust, no crea change automàtic.
    — L’agent ha d’aportar: inputs exactes (i ordre), outputs exactes (i ordre), i opcionalment rbf/locktime/derivacions/xpub global.
    — 'psbt_extras' permet injectar claus PSBT globals/inputs/outputs i propietàries.
    """
    creator = PSBTCreator(network=network)

    # 1) Inputs: obligatoris i en ordre donat per l’agent
    if not manual_selected_utxos or len(manual_selected_utxos) == 0:
        return {"success": False, "error": "Cal 'manual_selected_utxos' amb l’ordre final dels inputs."}
    selected_utxos: List[Dict] = list(manual_selected_utxos)
    total_input = sum(int(u.get("value_satoshis", 0)) for u in selected_utxos)

    # 2) Outputs: via 'outputs' explícits (recomanat) o bé mode simple estrictament controlat
    final_outputs: List[Dict] = []

    if outputs is not None:
        # Mode explícit: l’agent defineix exactament adreces/scripts, valors i ordre
        for o in outputs:
            if "value" not in o:
                return {"success": False, "error": "Cada output ha d’incloure 'value' en satoshis."}
            out = {"value": int(o["value"])}
            if "address" in o and o["address"] is not None:
                addr = str(o["address"]).strip()
                creator._decode_address(addr)  # valida
                out["address"] = addr
            elif "script" in o and o["script"] is not None:
                # Assumeix que PSBTCreator admet 'script' cru o hex per outputs
                out["script"] = o["script"]
            else:
                return {"success": False, "error": "Cada output necessita 'address' o 'script'."}
            final_outputs.append(out)

        implied_fee = total_input - sum(o["value"] for o in final_outputs)
        if implied_fee < 0:
            return {"success": False, "error": "Outputs > inputs. Ajusta valors o inputs."}

    else:
        # Mode simple: un sol recipient i, si cal, change explícit i fee explícita
        if recipient_address is None or amount_sats is None:
            return {"success": False, "error": "En mode simple cal 'recipient_address' i 'amount_sats'."}

        recipient_address = recipient_address.strip()
        creator._decode_address(recipient_address)
        amount_satoshis = int(amount_sats)
        if amount_satoshis <= 0:
            return {"success": False, "error": "amount_sats ha de ser > 0"}

        if fee_satoshis is None:
            return {"success": False, "error": "En mode simple cal 'fee_satoshis' explícit. No s’estima fee."}

        change_satoshis = total_input - amount_satoshis - int(fee_satoshis)
        if change_satoshis < 0:
            return {"success": False, "error": "Fons insuficients amb la fee indicada."}

        final_outputs.append({"address": recipient_address, "value": amount_satoshis})
        if change_satoshis > 0:
            if not change_address:
                return {"success": False, "error": "Cal 'change_address' si hi ha change > 0."}
            change_address = change_address.strip()
            creator._decode_address(change_address)
            final_outputs.append({"address": change_address, "value": int(change_satoshis)})

        implied_fee = total_input - sum(o["value"] for o in final_outputs)

    # 3) Construcció de la PSBT: només s’estableix el que demanes
    create_kwargs = {
        "inputs": selected_utxos,
        "outputs": final_outputs,
        "xpub": xpub,
        "extra_psbt": psbt_extras,
    }
    if include_keypaths:
        create_kwargs["include_derivations"] = True
    if include_global_xpub:
        create_kwargs["include_global_xpub"] = True
    if rbf is not None:
        create_kwargs["rbf"] = bool(rbf)
    if prefer_legacy_witness_utxo is not None:
        create_kwargs["prefer_legacy_witness_utxo"] = bool(prefer_legacy_witness_utxo)
    if locktime_override is not None:
        create_kwargs["locktime"] = int(locktime_override)

    try:
        psbt_build = creator.create_psbt(**create_kwargs)
        psbt_base64 = psbt_build["psbt"]
        return {
            "success": True,
            "psbt": psbt_base64,
            "psbt_hex": base64.b64decode(psbt_base64).hex(),
            "total_input_sats": int(total_input),
            "num_inputs": len(selected_utxos),
            "num_outputs": len(final_outputs),
            "fee_sats": int(implied_fee),                  # derivada de inputs − outputs
            "outputs_detail": final_outputs,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



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
        amount_sats=50_000,
        utxos=test_utxos,
        network="testnet",
        fee_rate=10,  # sat/vB
    )

    if result["success"]:
        print("✅ PSBT creat correctament!")
        print(f"   PSBT (Base64): {result['psbt'][:60]}...")
        print(f"   Inputs: {result['num_inputs']}")
        print(f"   Outputs: {result['num_outputs']}")
        print(f"   Fee: {result['fee_sats']} sats")
    else:
        print(f"❌ Error: {result['error']}")
