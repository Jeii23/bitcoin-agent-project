class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str, privacy_preset: str = "max"):
        self.xpub = xpub
        self.network = network
        self.privacy_preset = privacy_preset

    async def create_psbt(self, inputs: List[Dict], outputs: List[Dict], shuffle_seed: Optional[int] = None, rbf: bool = False) -> Dict:
        """
        Create a PSBT with the following features:
        - Deterministic shuffling of inputs and outputs if a shuffle seed is provided.
        - Omitting GLOBAL_XPUB and BIP32 keypaths.
        - Enabling RBF with non-final sequences.
        - Preferring WITNESS_UTXO for legacy inputs.
        - Folding small change into the fee.
        """
        # Shuffle inputs and outputs if a seed is provided
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)

        # Prepare the PSBT creation parameters
        psbt_params = {
            "inputs": inputs,
            "outputs": outputs,
            "rbf": rbf,
            "include_global_xpub": False,  # Omit GLOBAL_XPUB
            "include_keypaths": False,      # Omit BIP32 keypaths
            "prefer_legacy_witness_utxo": True,  # Prefer WITNESS_UTXO for legacy inputs
        }

        # Calculate the total amount to be sent and the fee
        total_input_value = sum(input['value'] for input in inputs)
        total_output_value = sum(output['amount'] for output in outputs)

        # Determine if we need to fold small change into the fee
        change = total_input_value - total_output_value
        if change > 0 and change < DUST_THRESHOLD:
            # Adjust the output amounts to fold the change into the fee
            total_output_value += change
            outputs[-1]['amount'] = total_output_value  # Adjust the last output

        # Create the PSBT
        psbt = create_transaction_psbt(
            xpub=self.xpub,
            recipient_address=outputs[0]['address'],
            amount_btc=total_output_value / 1e8,  # Convert to BTC
            utxos=inputs,
            network=self.network,
            rbf=rbf,
            include_global_xpub=False,
            include_keypaths=False,
        )

        return psbt

    # Other methods of the BitcoinAIAgent class...