
import sys, pathlib
import pytest

# --- Ensure src/ is importable for a src-layout project ---
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

class DummyResponse:
    def __init__(self, status_code=404, content=b"", text=""):
        self.status_code = status_code
        self.content = content
        self.text = text
    def json(self):
        return {}

@pytest.fixture
def block_requests(monkeypatch):
    """Block outgoing HTTP requests from psbt_creator by default so tests run offline.
    psbt_creator.create_psbt() will then fall back to the WITNESS_UTXO path.
    """
    try:
        import psbt_creator
    except Exception:
        pytest.skip("psbt_creator module not importable")
    def fake_get(*args, **kwargs):
        return DummyResponse(status_code=404, text="offline")
    monkeypatch.setattr(psbt_creator.requests, "get", fake_get, raising=True)
    return True


# --- Opcions CLI per a integració: --xpub i --network ---
def pytest_addoption(parser):
    parser.addoption("--xpub", action="store", default=None, help="XPUB/YPUB/ZPUB/VPUB/UPUB per als tests d'integració")
    parser.addoption("--network", action="store", default="testnet", help="Xarxa: 'testnet' o 'mainnet'")

import pytest

@pytest.fixture(scope="session")
def xpub_cli(request):
    val = request.config.getoption("--xpub")
    if val:
        return val.strip().strip('"').strip("'")
    return None

@pytest.fixture(scope="session")
def network_cli(request):
    return (request.config.getoption("--network") or "testnet").strip().lower()
