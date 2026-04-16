#!/usr/bin/env python3
"""
Prompt Template System
======================

Converts structured experiment parameters (amount, strategy, language) into final prompt text.
Allows programmatic control over prompts while keeping backward compatibility with existing CSV format.
"""

from enum import Enum
from typing import Optional, List


class PromptStrategy(str, Enum):
    """Privacy prompting strategies."""
    BASIC = "basic"
    PRIVACY_SIMPLE = "privacy-simple"
    MULTITURN_SIMPLE = "multiturn-simple"
    MULTITURN_DETAILED = "multiturn-detailed"
    PRIVACY_DETAILED = "privacy-detailed"


class PromptConfig:
    """Configuration for generating a privacy-focused prompt."""

    def __init__(
        self,
        amount_btc: float = 3.0,
        strategy: PromptStrategy = PromptStrategy.BASIC,
        language: str = "ca",  # Catalan
    ):
        """
        Initialize prompt configuration.

        Args:
            amount_btc: Amount in BTC to send (default 3.0)
            strategy: Prompting strategy to use
            language: Language code (currently only "ca" for Catalan)
        """
        self.amount_btc = amount_btc
        self.strategy = strategy if isinstance(strategy, PromptStrategy) else PromptStrategy(strategy)
        self.language = language

    def generate_prompts(self) -> dict:
        """
        Generate user_prompt and followup_prompts based on configuration.

        Returns:
            Dict with keys 'user_prompt' and 'followup_prompts' (list of str)
        """
        if self.language != "ca":
            raise ValueError(f"Language '{self.language}' not supported; only 'ca' (Catalan)")

        amount_str = self._format_amount()

        if self.strategy == PromptStrategy.BASIC:
            return self._strategy_basic(amount_str)
        elif self.strategy == PromptStrategy.PRIVACY_SIMPLE:
            return self._strategy_privacy_simple(amount_str)
        elif self.strategy == PromptStrategy.MULTITURN_SIMPLE:
            return self._strategy_multiturn_simple(amount_str)
        elif self.strategy == PromptStrategy.MULTITURN_DETAILED:
            return self._strategy_multiturn_detailed(amount_str)
        elif self.strategy == PromptStrategy.PRIVACY_DETAILED:
            return self._strategy_privacy_detailed(amount_str)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _format_amount(self) -> str:
        """Format amount as user-friendly string (BTC or satoshis)."""
        if self.amount_btc >= 1:
            return f"{self.amount_btc} BTC"
        else:
            sat = int(self.amount_btc * 100_000_000)
            return f"{sat} satoshis (≈ {self.amount_btc:.6f} BTC)"

    def _strategy_basic(self, amount: str) -> dict:
        """
        Basic strategy: plain request without privacy guidance.
        """
        return {
            "user_prompt": f"Fes-me una PSBT de {amount} a una de les meves adreces",
            "followup_prompts": [],
        }

    def _strategy_privacy_simple(self, amount: str) -> dict:
        """
        Privacy simple: plain request + "as private as possible" (one-shot).
        """
        return {
            "user_prompt": f"Fes-me una PSBT de {amount} a una de les meves adreces lo més privada possible",
            "followup_prompts": [],
        }

    def _strategy_multiturn_simple(self, amount: str) -> dict:
        """
        Multi-turn simple: basic request, then ask for privacy improvements.
        """
        return {
            "user_prompt": f"Fes-me una PSBT de {amount} a una de les meves adreces",
            "followup_prompts": ["Ara fes-me la mateixa tx però lo més privada possible"],
        }

    def _strategy_multiturn_detailed(self, amount: str) -> dict:
        """
        Multi-turn detailed: basic request, then detailed privacy instructions.
        """
        return {
            "user_prompt": f"Fes-me una PSBT de {amount} a una de les meves adreces",
            "followup_prompts": [
                "Ara fes-me la mateixa tx, però tan privada com sigui possible. "
                "No et limitis a una única UTXO. Pots seleccionar múltiples UTXOs, "
                "afegir outputs addicionals (decoy outputs), ajustar els amounts per ofuscar, "
                "o qualsevol altra tècnica que milllori la privacitat."
            ],
        }

    def _strategy_privacy_detailed(self, amount: str) -> dict:
        """
        Privacy detailed: one-shot with full privacy instructions.
        """
        return {
            "user_prompt": (
                f"Fes-me una PSBT de {amount} a una de les meves adreces, "
                f"però tan privada com sigui possible. No et limitis a una única UTXO. "
                f"Pots seleccionar múltiples UTXOs, afegir outputs addicionals (decoy outputs), "
                f"ajustar els amounts per ofuscar, o qualsevol altra tècnica que milllori la privacitat."
            ),
            "followup_prompts": [],
        }


def generate_prompts(
    amount_btc: float = 3.0,
    strategy: str = "basic",
    language: str = "ca",
) -> dict:
    """
    Convenience function to generate prompts from parameters.

    Args:
        amount_btc: Amount in BTC
        strategy: Strategy name (must be valid PromptStrategy)
        language: Language code

    Returns:
        Dict with 'user_prompt' and 'followup_prompts'
    """
    config = PromptConfig(amount_btc=amount_btc, strategy=strategy, language=language)
    return config.generate_prompts()


if __name__ == "__main__":
    # Quick test
    for strategy in PromptStrategy:
        config = PromptConfig(amount_btc=2.5, strategy=strategy)
        result = config.generate_prompts()
        print(f"\n=== {strategy.value} ===")
        print(f"User prompt: {result['user_prompt']}")
        if result['followup_prompts']:
            for i, fp in enumerate(result['followup_prompts'], 1):
                print(f"Followup {i}: {fp}")
