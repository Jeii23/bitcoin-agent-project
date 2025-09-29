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
        """
        Create a PSBT with the specified inputs and outputs.
        If shuffle_seed is provided, shuffle inputs and outputs deterministically.
        """
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)

        # Adjust outputs to avoid small change
        adjusted_outputs = self._adjust_outputs(outputs)

        # Create PSBT without GLOBAL_XPUB and BIP32 keypaths
        psbt = create_transaction_psbt(
            xpub=self.xpub,
            recipient_address=adjusted_outputs[0]['address'],  # Assuming first output is the main recipient
            amount_btc=adjusted_outputs[0]['amount'],
            utxos=inputs,
            rbf=rbf,
            include_global_xpub=False,
            include_keypaths=False,
            prefer_legacy_witness_utxo=True
        )

        return psbt

    def _adjust_outputs(self, outputs: List[Dict]) -> List[Dict]:
        """
        Adjust outputs to avoid small change by folding it into the fee.
        """
        DUST_THRESHOLD = 546  # in satoshis
        total_amount = sum(output['amount'] for output in outputs)

        # Filter out small outputs
        adjusted_outputs = []
        for output in outputs:
            if output['amount'] > DUST_THRESHOLD:
                adjusted_outputs.append(output)

        # If there's a small change, fold it into the fee
        if len(adjusted_outputs) > 1:
            change_amount = total_amount - sum(output['amount'] for output in adjusted_outputs)
            if change_amount > 0 and change_amount <= DUST_THRESHOLD:
                # Fold change into the fee by not including it in outputs
                adjusted_outputs = adjusted_outputs[:-1]  # Remove last output (assumed to be change)

        return adjusted_outputs

# Example usage
if __name__ == "__main__":
    agent = BitcoinAIAgent(xpub="your_xpub_here", network="testnet")
    inputs = [{"txid": "example_txid", "vout": 0, "amount": 100000}]  # Example UTXOs
    outputs = [{"address": "recipient_address_here", "amount": 50000}, {"address": "change_address_here", "amount": 1000}]
    
    psbt = agent.create_psbt(inputs, outputs, shuffle_seed=42, rbf=True)
    print(psbt)