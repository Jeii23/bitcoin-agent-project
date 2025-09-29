class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str, privacy_preset: str = "max"):
        self.xpub = xpub
        self.network = network
        self.privacy_preset = privacy_preset

    async def create_psbt(self, inputs: List[Dict], outputs: List[Dict], shuffle_seed: Optional[int] = None, rbf: bool = False) -> Dict:
        """
        Create a PSBT with the specified parameters.
        - shuffle_seed: If provided, shuffles inputs and outputs deterministically.
        - rbf: Enables Replace-By-Fee with non-final sequences.
        """
        # Shuffle inputs and outputs if a seed is provided
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)

        # Calculate total input value and prepare for change handling
        total_input_value = sum(u['value'] for u in inputs)
        total_output_value = sum(o['amount'] for o in outputs)

        # Determine if we need to fold small change into the fee
        change_amount = total_input_value - total_output_value
        if change_amount > 0 and change_amount < DUST_THRESHOLD:
            # Adjust the output amounts to fold the change into the fee
            fee_adjustment = change_amount
            # Optionally, you can adjust the last output amount or the fee
            if outputs:
                outputs[-1]['amount'] += fee_adjustment
            else:
                # If no outputs, just return an error or handle accordingly
                return {"success": False, "error": "No outputs to adjust for change."}

        # Create the PSBT
        psbt = create_transaction_psbt(
            xpub=self.xpub,
            inputs=inputs,
            outputs=outputs,
            rbf=rbf,
            include_global_xpub=False,  # Omit GLOBAL_XPUB
            include_keypaths=False,      # Omit BIP32 keypaths
            prefer_legacy_witness_utxo=True  # Prefer WITNESS_UTXO for legacy inputs
        )

        return psbt

    # Additional methods for the agent can be added here