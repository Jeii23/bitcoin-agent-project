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
    shuffle_seed: Optional[int] = None,
    avoid_change: bool = False,
    min_change_sats: Optional[int] = None,
    rbf: bool = False,
    prefer_legacy_witness_utxo: bool = False,
) -> Dict:
    """
    Create a PSBT for a transaction with the specified parameters.
    """
    # Amount in satoshis
    amount_satoshis = int(round(amount_btc * 100_000_000))

    # Shuffle inputs and outputs if a seed is provided
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(utxos)

    # Manual UTXO selection path (explicit list bypasses greedy selection)
    if manual_selected_utxos:
        selected_utxos = manual_selected_utxos
    else:
        selected_utxos = _select_utxos_vbytes(utxos, amount_satoshis, fee_rate, recipient_address, change_address)

    # Calculate the effective threshold for change suppression
    eff_threshold = DUST_THRESHOLD if min_change_sats is None else int(min_change_sats)

    # Create the PSBT
    psbt = PSBTCreator(network=network).create_psbt(
        inputs=selected_utxos,
        outputs=[{"address": recipient_address, "amount": amount_satoshis}],
        rbf=rbf,
        include_global_xpub=False,  # Omit GLOBAL_XPUB
        include_keypaths=False,      # Omit BIP32 keypaths
    )

    # Handle change if avoid_change is set
    if avoid_change:
        change_amount = sum(u['amount'] for u in selected_utxos) - amount_satoshis
        if change_amount > 0 and change_amount <= eff_threshold:
            # Fold small change into the fee
            fee_satoshis = (fee_satoshis or 0) + change_amount

    return psbt