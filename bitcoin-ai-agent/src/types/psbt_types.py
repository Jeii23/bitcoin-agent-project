class PSBTCreator:
    """Crea PSBTs estàndard BIP-174"""

    def create_psbt(
        self,
        inputs: List[Dict],
        outputs: List[Dict],
        locktime: int = 0,
        version: int = 2,
        rbf: bool = False,
        shuffle_seed: Optional[int] = None,
        avoid_small_change: bool = True,
        min_change_sats: int = DUST_THRESHOLD,
        prefer_witness_utxo: bool = True,
    ) -> Dict:
        # Shuffle inputs and outputs if a seed is provided
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(inputs)
            random.shuffle(outputs)

        # Implement logic to avoid small change
        if avoid_small_change:
            for output in outputs:
                if output['amount'] <= min_change_sats:
                    output['amount'] += min_change_sats  # Fold into fee

        # Create PSBT logic here...
        # Set RBF by adjusting sequence numbers
        if rbf:
            for input in inputs:
                input['sequence'] = 0xFFFFFFFE  # Non-final sequence

        # Prefer WITNESS_UTXO for legacy inputs
        if prefer_witness_utxo:
            for input in inputs:
                if 'legacy' in input and input['legacy']:
                    input['witness_utxo'] = input.get('utxo')

        # Continue with PSBT creation...