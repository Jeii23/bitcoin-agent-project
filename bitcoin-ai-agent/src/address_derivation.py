import random
import hashlib

class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str):
        self.xpub = xpub
        self.network = network

    def _shuffle_inputs_outputs(self, inputs: List[Dict], outputs: List[Dict], shuffle_seed: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """Shuffle inputs and outputs deterministically based on the provided seed."""
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)
        return inputs, outputs

    def create_psbt(self, inputs: List[Dict], outputs: List[Dict], shuffle_seed: Optional[int] = None, rbf: bool = False) -> Dict:
        """Create a PSBT with the specified inputs and outputs."""
        # Shuffle inputs and outputs if a seed is provided
        inputs, outputs = self._shuffle_inputs_outputs(inputs, outputs, shuffle_seed)

        # Calculate total input value and total output value
        total_input_value = sum(u['value'] for u in inputs)
        total_output_value = sum(o['amount'] for o in outputs)

        # Handle small change by folding it into the fee
        change_threshold = DUST_THRESHOLD  # Define your dust threshold
        if total_input_value > total_output_value:
            change = total_input_value - total_output_value
            if change <= change_threshold:
                # Adjust outputs to include the change in the fee
                # You can modify the last output or create a fee output
                outputs[-1]['amount'] += change  # Add change to the last output

        # Create the PSBT using the PSBTCreator
        psbt_creator = PSBTCreator(network=self.network)
        psbt = psbt_creator.create_psbt(
            inputs=inputs,
            outputs=outputs,
            rbf=rbf,
            include_global_xpub=False,  # Omit GLOBAL_XPUB
            include_keypaths=False,      # Omit BIP32 keypaths
            prefer_legacy_witness_utxo=True  # Prefer WITNESS_UTXO for legacy inputs
        )

        return psbt

# Example usage
agent = BitcoinAIAgent(xpub=DEFAULT_XPUB, network=DEFAULT_NETWORK)
inputs = [{'txid': '...', 'vout': 0, 'value': 100000}]  # Example inputs
outputs = [{'address': '...', 'amount': 50000}]  # Example outputs
shuffle_seed = 12345  # Example shuffle seed
psbt = agent.create_psbt(inputs, outputs, shuffle_seed=shuffle_seed, rbf=True)