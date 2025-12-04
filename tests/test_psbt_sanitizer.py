import base64
import pytest

from pathlib import Path


def test_sanitize_psbt_b64_roundtrip(monkeypatch):
    # Import helper from agent module
    from src.bitcoin_ai_agent import _sanitize_psbt_b64

    # A tiny fake PSBT header + minimal body, base64-encoded (BIP-174 magic 'psbt\xff')
    raw = b"psbt\xff" + b"\x01\x02\x03\x04" * 5
    b64 = base64.b64encode(raw).decode()
    multi = "\n".join([b64[:20], b64[20:40], b64[40:]])

    cleaned = _sanitize_psbt_b64(multi)
    assert "\n" not in cleaned and " " not in cleaned
    decoded = base64.b64decode(cleaned)
    assert decoded.startswith(b"psbt\xff")
