"""Address classification utilities.

Provides a single public helper detect_address_type used for fee/vbyte
heuristics. Separated to avoid psbt_creator owning unrelated concerns.
"""
from __future__ import annotations

__all__ = ["detect_address_type"]


def detect_address_type(addr: str) -> str:
    a = (addr or "").lower()
    if a.startswith(("bc1p", "tb1p")):
        return "p2tr"
    if a.startswith(("bc1q", "tb1q")):
        return "p2wpkh"
    if a.startswith(("1", "m", "n")):
        return "p2pkh"
    if a.startswith(("3", "2")):
        return "p2sh-p2wpkh"  # conservative higher cost
    return "p2wpkh"
