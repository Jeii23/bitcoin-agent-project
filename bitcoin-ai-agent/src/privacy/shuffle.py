class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str, privacy_preset: str):
        self.xpub = xpub
        self.network = network
        self.privacy_preset = privacy_preset

    def create_psbt(self, inputs: List[Dict], outputs: List[Dict], shuffle_seed: Optional[int] = None, rbf: bool = False) -> Dict:
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
            "rbf": rbf,                     # Enable RBF
            "prefer_legacy_witness_utxo": True,  # Prefer WITNESS_UTXO for legacy inputs
        }

        # Calculate total input value and total output value
        total_input_value = sum(input['value'] for input in inputs)
        total_output_value = sum(output['value'] for output in outputs)

        # Check for small change and fold it into the fee if necessary
        change_threshold = DUST_THRESHOLD  # Define your dust threshold
        if total_input_value - total_output_value < change_threshold:
            # Adjust the last output to include the small change in the fee
            if outputs:
                last_output = outputs[-1]
                last_output['value'] += total_input_value - total_output_value
                outputs[-1] = last_output
            else:
                # If there are no outputs, create a dummy output to absorb the change
                outputs.append({"address": self.xpub, "value": total_input_value - total_output_value})

        # Create the PSBT using the PSBTCreator
        psbt_creator = PSBTCreator(network=self.network)
        psbt = psbt_creator.create_psbt(**psbt_params)

        return psbt

    # Other methods of the BitcoinAIAgent class...