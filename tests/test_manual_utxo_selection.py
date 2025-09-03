#!/usr/bin/env python3
"""Tests per selecció manual d'UTXOs en create_transaction_psbt"""

import base64
from typing import List, Dict
from psbt_creator import create_transaction_psbt

XPUB = "vpub5Zs16Jexbgj86exZZdj2LT3ukA2gPXdGgdLZokbng1MgbP5jrm8eRkqAffKEJN2BnMzjkJ3G64Sk2XoB6FyAEnXAfmu7nthCGFXy1snAQHC"

# UTXOs simulats
UTXOS: List[Dict] = [
    {"txid": "a" * 64, "vout": 0, "value_satoshis": 60_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"},
    {"txid": "b" * 64, "vout": 1, "value_satoshis": 55_000, "address": "tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx"},
    {"txid": "c" * 64, "vout": 2, "value_satoshis": 50_000, "address": "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"},
]

def test_manual_selection_no_change():
    # Seleccionem exactament dues UTXOs per cobrir amount + fee sense canvi
    manual = UTXOS[:2]  # 60k + 55k = 115k
    amount_btc = 0.001   # 100k sats
    fee_rate = 10
    res = create_transaction_psbt(
        xpub=XPUB,
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_btc=amount_btc,
        utxos=UTXOS,
        change_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
        fee_rate=fee_rate,
        manual_selected_utxos=manual,
    )
    assert res["success"], res
    assert res["num_inputs"] == len(manual)
    # Hauria d'haver-hi 1 o 2 outputs (segons si queda canvi); si change_satoshis < dust => no-change
    assert res["num_outputs"] in (1, 2)
    psbt_bytes = base64.b64decode(res["psbt"])
    assert psbt_bytes[:5] == b"psbt\xff"


def test_manual_selection_with_change():
    # Seleccionem una sola UTXO gran i forcem canvi
    manual = [UTXOS[0]]  # 60k
    amount_btc = 0.0003   # 30k sats
    fee_rate = 5
    res = create_transaction_psbt(
        xpub=XPUB,
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_btc=amount_btc,
        utxos=UTXOS,
        change_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
        fee_rate=fee_rate,
        manual_selected_utxos=manual,
    )
    assert res["success"], res
    assert res["num_inputs"] == 1
    # Amb una sola UTXO gran i amount petit + fee → hi hauria de sortir canvi > dust
    assert res["num_outputs"] == 2


def test_manual_selection_insufficient():
    # Triar només una UTXO insuficient per amount + fee
    manual = [UTXOS[2]]  # 50k
    amount_btc = 0.00049  # 49k sats
    fee_rate = 50  # força fee alta
    res = create_transaction_psbt(
        xpub=XPUB,
        recipient_address="tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx",
        amount_btc=amount_btc,
        utxos=UTXOS,
        change_address="tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t",
        fee_rate=fee_rate,
        manual_selected_utxos=manual,
    )
    assert not res["success"], res
    assert "insuficients" in res["error"].lower() or "insuficient" in res["error"].lower()
