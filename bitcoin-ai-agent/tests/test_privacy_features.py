import os
import random
from typing import List, Dict, Optional
from psbt_creator import PSBTCreator, create_transaction_psbt

class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str, shuffle_seed: Optional[int] = None):
        self.xpub = xpub
        self.network = network
        self.shuffle_seed = shuffle_seed

    def _shuffle_inputs_outputs(self, inputs: List[Dict], outputs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Shuffle inputs and outputs deterministically based on the provided seed."""
        if self.shuffle_seed is not None:
            random.seed(self.shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)
        return inputs, outputs

    def create_psbt(self, inputs: List[Dict], outputs: List[Dict], rbf: bool = True) -> Dict:
        """Create a PSBT with the specified inputs and outputs."""
        # Shuffle inputs and outputs if a seed is provided
        inputs, outputs = self._shuffle_inputs_outputs(inputs, outputs)

        # Prepare the PSBT creator
        creator = PSBTCreator(network=self.network)

        # Set RBF by modifying the sequence number of inputs
        for input in inputs:
            input['sequence'] = 0xFFFFFFFE  # Non-final sequence for RBF

        # Avoid small change by folding it into the fee
        total_output_value = sum(output['amount'] for output in outputs)
        total_input_value = sum(input['amount'] for input in inputs)

        if total_input_value - total_output_value < DUST_THRESHOLD:
            # Adjust the fee to absorb the small change
            fee_adjustment = total_input_value - total_output_value
            outputs[-1]['amount'] += fee_adjustment  # Add to the last output

        # Create the PSBT without GLOBAL_XPUB and BIP32 keypaths
        psbt = creator.create_psbt(
            inputs=inputs,
            outputs=outputs,
            include_global_xpub=False,
            include_keypaths=False,
            rbf=rbf
        )
        return psbt

# Example usage
if __name__ == "__main__":
    agent = BitcoinAIAgent(xpub="your_xpub_here", network="testnet", shuffle_seed=12345)
    inputs = [{"txid": "input_txid", "vout": 0, "amount": 100000}]  # Example input
    outputs = [{"address": "recipient_address", "amount": 50000}]  # Example output
    psbt = agent.create_psbt(inputs, outputs)
    print(psbt)