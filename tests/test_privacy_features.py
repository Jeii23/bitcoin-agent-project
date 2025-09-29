import sys
import base64
import json
from typing import List, Dict

import pytest

# Ensure module import
sys.path.append('/home/jaume/bitcoin-agent-project/src')
from psbt_creator import create_transaction_psbt, decode_psbt

# Common testnet vectors used across repo
RECIP = "tb1qfqzk956wtxlvvghewk5hqu6vwqjtjm5qmua7wx"
CHANGE = "tb1q07cj0eftvl2v2505hnfuzjxlyn00cthh7pfc3y"
INPUT_ADDR = "tb1q0wwa08elht6gq8uzjsl66mdhjl7rcsetakcf4t"
LEGACY_P2PKH = "mk2QpYatsKicvFVuTAQLBryyccRXMUaGHP"


def test_privacy_preset_max_builds_and_decodes():
    utxos = [{
        'txid': 'aa'*32,
        'vout': 0,
        'value_satoshis': 120_000,
        'address': INPUT_ADDR,
    }]
    res = create_transaction_psbt(
        xpub="tpubDUMMY",
        recipient_address=RECIP,
        amount_btc=0.0005,  # 50_000 sats
        utxos=utxos,
        change_address=CHANGE,
        network='testnet',
        fee_rate=10,
        privacy_preset='max',  # includes rbf, no xpub/keypaths, avoid_change, etc.
        shuffle_seed=7,
    )
    assert res['success'], res
    info = decode_psbt(res['psbt'], network='testnet')
    assert info['valid']
    assert info['num_inputs'] == res['num_inputs']
    assert info['num_outputs'] == res['num_outputs']


def test_deterministic_shuffle_outputs_with_seed():
    # Choose amounts so that change is > dust to guarantee 2 outputs
    utxos = [{
        'txid': 'bb'*32,
        'vout': 0,
        'value_satoshis': 150_000,
        'address': INPUT_ADDR,
    }]
    kwargs = dict(
        xpub="tpubDUMMY",
        recipient_address=RECIP,
        amount_btc=0.0006,  # 60_000 sats
        utxos=utxos,
        change_address=CHANGE,
        network='testnet',
        fee_rate=3,
        shuffle_inputs=True,
        shuffle_outputs=True,
        prefer_legacy_witness_utxo=True,  # avoid any network fetch
    )
    res_a = create_transaction_psbt(shuffle_seed=123, **kwargs)
    res_b = create_transaction_psbt(shuffle_seed=123, **kwargs)
    assert res_a['success'] and res_b['success']
    assert res_a['num_outputs'] == 2
    assert res_a['outputs'] == res_b['outputs']  # deterministic with same seed


def test_rbf_sequences_present_with_rbf_true():
    utxos = [{
        'txid': 'cc'*32,
        'vout': 1,
        'value_satoshis': 90_000,
        'address': INPUT_ADDR,
    }]
    res = create_transaction_psbt(
        xpub="tpubDUMMY",
        recipient_address=RECIP,
        amount_btc=0.0004,  # 40_000 sats
        utxos=utxos,
        change_address=CHANGE,
        network='testnet',
        fee_rate=5,
        rbf=True,
        prefer_legacy_witness_utxo=True,
    )
    assert res['success'], res
    hex_tx = res['unsigned_tx_hex']
    # sequence 0xFFFFFFFD (little-endian in hex) -> fdffffff per input
    assert hex_tx.count('fdffffff') == res['num_inputs'], (hex_tx, res['num_inputs'])


def test_prefer_legacy_witness_utxo_sets_witness_for_legacy_input():
    utxos = [{
        'txid': 'dd'*32,
        'vout': 0,
        'value_satoshis': 80_000,
        'address': LEGACY_P2PKH,
    }]
    res = create_transaction_psbt(
        xpub="tpubDUMMY",
        recipient_address=RECIP,
        amount_btc=0.0003,  # 30_000 sats
        utxos=utxos,
        change_address=CHANGE,
        network='testnet',
        fee_rate=5,
        prefer_legacy_witness_utxo=True,  # force WITNESS_UTXO instead of fetch
        rbf=True,
    )
    assert res['success'], res
    info = decode_psbt(res['psbt'], network='testnet')
    assert info['valid']
    assert info.get('has_witness_utxo', [False])[0] is True, info


def test_avoid_change_suppresses_small_change_into_fee():
    utxos = [{
        'txid': 'ee'*32,
        'vout': 0,
        'value_satoshis': 60_000,
        'address': INPUT_ADDR,
    }]
    res = create_transaction_psbt(
        xpub="tpubDUMMY",
        recipient_address=RECIP,
        amount_btc=0.000551,  # 55_100 sats
        utxos=utxos,
        change_address=CHANGE,
        network='testnet',
        fee_rate=1,
        avoid_change=True,
        min_change_sats=5000,  # fold small change into fee
        prefer_legacy_witness_utxo=True,
    )
    assert res['success'], res
    # With avoid_change set and small leftover, we expect only recipient output
    assert res['num_outputs'] == 1, res
