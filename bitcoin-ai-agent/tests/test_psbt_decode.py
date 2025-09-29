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
    shuffle_inputs: bool = False,
    shuffle_outputs: bool = False,
    avoid_change: bool = False,
    min_change_sats: Optional[int] = None,
    include_global_xpub: bool = False,  # Omit GLOBAL_XPUB
    include_keypaths: bool = False,      # Omit BIP32 keypaths
    rbf: bool = True,                     # Enable RBF
    prefer_legacy_witness_utxo: bool = True,
    locktime_override: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> Dict:
    """
    Create a PSBT for a transaction with the specified parameters.
    """
    # Apply privacy preset if requested (opt-in; conservative defaults otherwise)
    if privacy_preset and privacy_preset.lower() in ("max", "maximum", "paranoid"):
        # Apply privacy settings here if needed

    creator = PSBTCreator(network)

    # Convert amount to satoshis
    amount_satoshis = int(round(amount_btc * 100_000_000))

    # Manual UTXO selection path (explicit list bypasses greedy selection)
    if manual_selected_utxos:
        selected_utxos = manual_selected_utxos
    else:
        selected_utxos = _select_utxos_vbytes(
            utxos,
            amount_satoshis,
            fee_rate,
            recipient_address,
            change_address,
        )

    # Shuffle inputs and outputs if requested
    if shuffle_inputs or shuffle_outputs:
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        if shuffle_inputs:
            random.shuffle(selected_utxos)
        if shuffle_outputs:
            # Assuming outputs are defined somewhere
            random.shuffle(outputs)

    # Calculate the effective fee and handle small change
    effective_fee = fee_satoshis if fee_satoshis is not None else calculate_fee(selected_utxos, fee_rate)
    total_input_value = sum(u['value'] for u in selected_utxos)
    total_output_value = amount_satoshis + effective_fee

    if avoid_change and (total_input_value - total_output_value) <= (min_change_sats or DUST_THRESHOLD):
        effective_fee += total_input_value - total_output_value
        total_output_value = amount_satoshis  # No change output

    # Create the PSBT
    psbt = creator.create_psbt(
        inputs=selected_utxos,
        outputs=[{"address": recipient_address, "amount": amount_satoshis}],
        locktime=locktime_override or 0,
        rbf=rbf,
    )

    return psbt