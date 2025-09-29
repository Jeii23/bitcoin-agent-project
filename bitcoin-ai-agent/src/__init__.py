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
) -> Dict:
    """
    Crea un PSBT per una transacció amb les opcions especificades.
    """
    # Amount to sats
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

    # Calculate total input value
    total_input_value = sum(u['value'] for u in selected_utxos)

    # Calculate fee if not provided
    if fee_satoshis is None:
        estimated_vbytes = _estimate_vbytes(selected_utxos, recipient_address, change_address is not None, change_address)
        fee_satoshis = fee_rate * estimated_vbytes

    # Check if we have enough funds
    if total_input_value < amount_satoshis + fee_satoshis:
        return {
            "success": False,
            "error": "Insufficient funds."
        }

    # Create transaction outputs
    outputs = [{ "address": recipient_address, "amount": amount_satoshis }]
    
    # Handle change if required
    if change_address and (total_input_value - amount_satoshis - fee_satoshis > 0):
        change_amount = total_input_value - amount_satoshis - fee_satoshis
        if avoid_change and change_amount <= DUST_THRESHOLD:
            fee_satoshis += change_amount  # Fold change into fee
        else:
            outputs.append({ "address": change_address, "amount": change_amount })

    # Create PSBT without GLOBAL_XPUB and BIP32 keypaths
    psbt = PSBTCreator(network).create_psbt(
        inputs=selected_utxos,
        outputs=outputs,
        rbf=rbf,
        include_global_xpub=False,
        include_keypaths=False,
    )

    return {
        "success": True,
        "psbt": psbt,
    }