class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str, privacy_preset: str):
        self.xpub = xpub
        self.network = network
        self.privacy_preset = privacy_preset

    async def create_psbt(self, inputs: List[Dict], outputs: List[Dict], shuffle_seed: Optional[int] = None) -> Dict:
        # Shuffle inputs and outputs if a seed is provided
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)

        # Prepare the PSBT creation parameters
        psbt_params = {
            "inputs": inputs,
            "outputs": outputs,
            "include_global_xpub": False,  # Omit GLOBAL_XPUB
            "include_keypaths": False,      # Omit BIP32 keypaths
            "rbf": True,                    # Enable RBF
            "prefer_legacy_witness_utxo": True,  # Prefer WITNESS_UTXO for legacy inputs
        }

        # Calculate the total amount to be sent and check for small change
        total_output_amount = sum(output['amount'] for output in outputs)
        total_input_amount = sum(input['amount'] for input in inputs)

        # Check if change is small and fold it into the fee
        if total_input_amount - total_output_amount < DUST_THRESHOLD:
            # Adjust the outputs to avoid small change
            total_fee = total_input_amount - total_output_amount
            outputs[-1]['amount'] += total_fee  # Add the fee to the last output

        # Create the PSBT
        psbt = create_transaction_psbt(**psbt_params)
        return psbt

    # Other methods for the agent...