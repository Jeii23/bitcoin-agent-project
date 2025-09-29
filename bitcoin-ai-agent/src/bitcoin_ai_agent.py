import random
import hashlib

# ... other imports ...

class BitcoinAIAgent:
    # ... existing methods ...

    async def create_psbt_with_options(
        self,
        inputs: List[Dict],
        outputs: List[Dict],
        shuffle_seed: Optional[int] = None,
        rbf: bool = False,
        avoid_small_change: bool = False,
        network: str = DEFAULT_NETWORK,
    ) -> Dict:
        """
        Create a PSBT with options for shuffling, RBF, and small change handling.

        Parameters:
        - inputs: List of input UTXOs.
        - outputs: List of output addresses and amounts.
        - shuffle_seed: Optional seed for deterministic shuffling.
        - rbf: Enable Replace-By-Fee.
        - avoid_small_change: If True, small change will be folded into the fee.

        Returns:
        - A dictionary containing the PSBT or an error message.
        """
        # Shuffle inputs and outputs if a seed is provided
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)

        # Calculate total input value
        total_input_value = sum(u['value'] for u in inputs)

        # Calculate total output value
        total_output_value = sum(o['amount'] for o in outputs)

        # Determine if we need to fold small change into the fee
        change_amount = total_input_value - total_output_value
        if avoid_small_change and change_amount > 0 and change_amount < DUST_THRESHOLD:
            # Adjust the last output to include the change in the fee
            if outputs:
                outputs[-1]['amount'] += change_amount
                change_amount = 0

        # Create the PSBT
        psbt = create_transaction_psbt(
            inputs=inputs,
            outputs=outputs,
            rbf=rbf,
            include_global_xpub=False,  # Omit GLOBAL_XPUB
            include_keypaths=False,      # Omit BIP32 keypaths
            prefer_legacy_witness_utxo=True,  # Prefer WITNESS_UTXO for legacy inputs
        )

        return psbt

# ... other methods ...

# Example usage
async def main():
    agent = BitcoinAIAgent()
    inputs = [...]  # Define your inputs
    outputs = [...]  # Define your outputs
    shuffle_seed = 12345  # Example seed for shuffling
    rbf = True  # Enable RBF
    avoid_small_change = True  # Avoid small change

    psbt = await agent.create_psbt_with_options(inputs, outputs, shuffle_seed, rbf, avoid_small_change)
    print(psbt)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())