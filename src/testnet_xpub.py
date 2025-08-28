#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hdwallet import HDWallet
from hdwallet.cryptocurrencies import Bitcoin
from hdwallet.hds.bip32 import BIP32HD
from hdwallet.derivations.custom import CustomDerivation
from hdwallet.mnemonics import BIP39Mnemonic
from hdwallet.addresses.p2wpkh import P2WPKHAddress
import struct

# Helpers Base58Check del teu projecte
from address_derivation import _b58decode_check, _b58encode_check

# ───────────────────────────────────────────────────────────────────
# CONFIGURA el teu mnemonic/passphrase de proves (no el publiquis)
MNEMONIC   = "green spin grace fat social disease dizzy private foam mansion enter color amused purse nation"
PASSPHRASE = ""
# ───────────────────────────────────────────────────────────────────

def remap_version(xpub_like: str, new_version: int) -> str:
    raw = _b58decode_check(xpub_like)
    return _b58encode_check(struct.pack(">I", new_version) + raw[4:])

def first_addr_from_account_xpub(xpub_like: str, network) -> str:
    """Dona una xpub/tpub (nivell compte) i retorna la primera adreça (m/0/0)."""
    w = HDWallet(cryptocurrency=Bitcoin, hd=BIP32HD, network=network)
    w.from_xpublic_key(xpublic_key=xpub_like)
    # External chain 0, index 0
    w.from_derivation(CustomDerivation(path="m/0/0"))
    hrp = "bc" if network == Bitcoin.NETWORKS.MAINNET else "tb"
    return w.address(address=P2WPKHAddress, hrp=hrp)

def derive_account(network, coin_type: int):
    """Retorna (xpub_like, first_addr) per BIP84 m/84'/coin_type'/0'."""
    mn = BIP39Mnemonic(MNEMONIC)
    w = HDWallet(cryptocurrency=Bitcoin, hd=BIP32HD,
                 network=network, passphrase=PASSPHRASE).from_mnemonic(mnemonic=mn)

    account_path = f"m/84'/{coin_type}'/0'"
    w.from_derivation(CustomDerivation(path=account_path))
    xpub_like = w.xpublic_key()  # xpub (mainnet) o tpub (testnet)

    # primera adreça: /0/0
    w.from_derivation(CustomDerivation(path=account_path + "/0/0"))
    hrp = "bc" if network == Bitcoin.NETWORKS.MAINNET else "tb"
    first_addr = w.address(address=P2WPKHAddress, hrp=hrp)
    return xpub_like, first_addr

def main():
    # ── MAINNET (BIP84: m/84'/0'/0')
    xpub_main, addr_main = derive_account(Bitcoin.NETWORKS.MAINNET, coin_type=0)
    # Remapeig SLIP-0132 → zpub (0x04B24746)
    zpub_main = remap_version(xpub_main, 0x04B24746)

    print("MAINNET — BIP84 m/84'/0'/0'")
    print("  xpub:", xpub_main)
    print("  zpub:", zpub_main)
    print("  first addr (m/84'/0'/0'/0/0):", addr_main)

    # ── TESTNET (BIP84: m/84'/1'/0')
    tpub_test, addr_test = derive_account(Bitcoin.NETWORKS.TESTNET, coin_type=1)
    # Remapeig SLIP-0132 → vpub (0x045F1CF6)
    vpub_test = remap_version(tpub_test, 0x045F1CF6)

    print("\nTESTNET — BIP84 m/84'/1'/0'")
    print("  tpub:", tpub_test)
    print("  vpub:", vpub_test)
    print("  first addr (m/84'/1'/0'/0/0):", addr_test)

    # (Opcional) Exemple de “watch-only” des d’una xpub/tpub d’account:
    # print("first from provided xpub:", first_addr_from_account_xpub("xpub/tpub aquí", Bitcoin.NETWORKS.MAINNET))

if __name__ == "__main__":
    main()
