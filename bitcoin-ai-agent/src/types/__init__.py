# ============== EINES (TOOLS) PER L'AGENT IA ==============

@tool
async def create_psbt_with_shuffling(
    xpub: str,
    recipient_address: str,
    amount_btc: float,
    utxos: List[Dict],
    change_address: Optional[str] = None,
    network: str = DEFAULT_NETWORK,
    fee_rate: int = 10,
    shuffle_seed: Optional[int] = None,
    avoid_small_change: bool = True,
) -> Dict:
    """
    Crea un PSBT amb barrejament determinista d'inputs i outputs.
    """
    # Convert amount to satoshis
    amount_satoshis = int(round(amount_btc * 100_000_000))

    # Shuffle inputs and outputs if a seed is provided
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(utxos)

    # Prepare inputs and outputs for PSBT creation
    inputs = []
    outputs = [{"address": recipient_address, "amount": amount_satoshis}]

    # Select UTXOs and calculate total input amount
    selected_utxos = _select_utxos_vbytes(utxos, amount_satoshis, fee_rate, recipient_address, change_address)
    
    if not selected_utxos["success"]:
        return selected_utxos  # Return error if UTXO selection fails

    total_input_amount = sum(u['amount'] for u in selected_utxos['utxos'])
    
    # Handle small change
    if avoid_small_change and (total_input_amount - amount_satoshis < DUST_THRESHOLD):
        # Adjust the output to include the change in the fee
        outputs[0]["amount"] += (total_input_amount - amount_satoshis)

    # Create the PSBT
    psbt_creator = PSBTCreator(network=network)
    psbt = psbt_creator.create_psbt(
        inputs=selected_utxos['utxos'],
        outputs=outputs,
        rbf=True,  # Enable RBF
        include_global_xpub=False,  # Omit GLOBAL_XPUB
        include_keypaths=False,  # Omit BIP32 keypaths
        prefer_legacy_witness_utxo=True  # Prefer WITNESS_UTXO for legacy inputs
    )

    return psbt

# ============== AGENT IA PRINCIPAL ==============

class BitcoinAIAgent:
    def __init__(self):
        # Initialize any necessary state or configurations
        pass

    async def handle_create_psbt(self, xpub: str, recipient_address: str, amount_btc: float, utxos: List[Dict], change_address: Optional[str] = None, network: str = DEFAULT_NETWORK, fee_rate: int = 10, shuffle_seed: Optional[int] = None, avoid_small_change: bool = True):
        return await create_psbt_with_shuffling(xpub, recipient_address, amount_btc, utxos, change_address, network, fee_rate, shuffle_seed, avoid_small_change)

# ============== MAIN ==============

async def main():
    agent = BitcoinAIAgent()
    # Example usage
    result = await agent.handle_create_psbt(
        xpub=DEFAULT_XPUB,
        recipient_address="tb1q...",
        amount_btc=0.01,
        utxos=[{"txid": "abc123...", "vout": 0, "amount": 100000}],
        change_address="tb1q...",
        network=DEFAULT_NETWORK,
        fee_rate=10,
        shuffle_seed=42,
        avoid_small_change=True
    )
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())