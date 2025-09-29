import os
import random
from typing import List, Dict, Optional
from psbt_creator import create_transaction_psbt

class BitcoinAIAgent:
    def __init__(self, xpub: str, network: str, privacy_preset: str = "max"):
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
            "locktime": 0,
            "version": 2,
            "include_global_xpub": False,  # Omit GLOBAL_XPUB
            "include_keypaths": False,      # Omit BIP32 keypaths
            "rbf": rbf,                     # Enable RBF
            "prefer_legacy_witness_utxo": True,  # Prefer WITNESS_UTXO for legacy inputs
        }

        # Avoid small change by folding it into the fee
        self._avoid_small_change(psbt_params)

        # Create the PSBT
        psbt = create_transaction_psbt(
            xpub=self.xpub,
            recipient_address=outputs[0]['address'],  # Assuming the first output is the recipient
            amount_btc=outputs[0]['amount'] / 1e8,  # Convert to BTC
            utxos=inputs,
            network=self.network,
            rbf=rbf
        )

        return psbt

    def _avoid_small_change(self, psbt_params: Dict):
        """Adjust the outputs to avoid small change."""
        total_output_amount = sum(output['amount'] for output in psbt_params['outputs'])
        total_input_amount = sum(input['amount'] for input in psbt_params['inputs'])

        # Calculate the fee based on the difference
        fee = total_input_amount - total_output_amount

        # If the change is below a certain threshold, fold it into the fee
        DUST_THRESHOLD = 546  # in satoshis
        for output in psbt_params['outputs']:
            if output['amount'] < DUST_THRESHOLD:
                fee += output['amount']
                output['amount'] = 0  # Remove the small output

        # Update the first output to reflect the new fee
        if psbt_params['outputs']:
            psbt_params['outputs'][0]['amount'] += fee

# Example usage
if __name__ == "__main__":
    agent = BitcoinAIAgent(xpub="your_xpub_here", network="testnet")
    inputs = [{"txid": "example_txid", "vout": 0, "amount": 100000}]  # Example input
    outputs = [{"address": "recipient_address_here", "amount": 50000}]  # Example output
    psbt = agent.create_psbt(inputs, outputs, shuffle_seed=42, rbf=True)
    print(psbt)