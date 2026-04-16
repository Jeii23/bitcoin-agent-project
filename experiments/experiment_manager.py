#!/usr/bin/env python3
"""
Experiment Manager
==================

Handles CSV operations for experiments, including:
- Reading existing experiments
- Writing new/modified experiments
- Backward-compatible schema handling and legacy inference
- Converting between internal format and CSV format
"""

import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from prompt_templates import PromptStrategy, generate_prompts


@dataclass
class ExperimentMeta:
    """Metadata for an experiment (UI-level representation)."""
    id: str
    name: str
    description: str = ""
    amount_btc: float = 3.0
    strategy: str = "basic"  # PromptStrategy value
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 1.0
    repetitions: int = 3
    timeout_seconds: int = 300
    network: str = "mainnet"
    tags: List[str] = None
    enabled: bool = True
    priority: int = 1
    prompt_mode: str = "template"  # template or custom
    system_prompt: str = ""
    user_prompt: str = ""
    followup_prompts: List[str] = None
    xpub: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.followup_prompts is None:
            self.followup_prompts = []


class ExperimentManager:
    """Manages experiment CSV with backward compatibility."""

    CSV_COLUMNS = [
        "id", "name", "description", "amount_btc", "strategy", "prompt_mode",
        "provider", "model", "temperature", "system_prompt", "user_prompt",
        "followup_prompts", "repetitions", "timeout_seconds", "network",
        "tags", "enabled", "priority", "xpub"
    ]

    def __init__(self, csv_path: Path):
        """Initialize with path to experiments CSV."""
        self.csv_path = Path(csv_path)

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Normalize prompt text for legacy inference."""
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()

    @staticmethod
    def _split_pipe(value: str) -> List[str]:
        return [item.strip() for item in (value or "").split("|") if item.strip()]

    @staticmethod
    def _join_pipe(value) -> str:
        if isinstance(value, list):
            return "|".join(str(item).strip() for item in value if str(item).strip())
        return str(value) if value is not None else ""

    @staticmethod
    def _truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "y", "s", "si")

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            if value is None or str(value).strip() == "":
                return default
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            if value is None or str(value).strip() == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def infer_amount_btc(cls, row: Dict) -> float:
        """Infer amount from explicit CSV field or legacy Catalan prompt text."""
        explicit = row.get("amount_btc")
        if explicit:
            return cls._safe_float(explicit, 3.0)

        prompt_text = " ".join([
            row.get("user_prompt", ""),
            row.get("followup_prompts", ""),
        ])
        normalized = prompt_text.replace("_", "").replace(",", ".")

        sat_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*satoshis?\b", normalized, re.IGNORECASE)
        if sat_match:
            return float(sat_match.group(1)) / 100_000_000

        btc_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*BTC\b", normalized, re.IGNORECASE)
        if btc_match:
            return float(btc_match.group(1))

        return 3.0

    @classmethod
    def infer_strategy(cls, row: Dict) -> str:
        """Infer prompt strategy for legacy rows without changing their prompt text."""
        explicit = (row.get("strategy") or "").strip()
        valid = {strategy.value for strategy in PromptStrategy}
        if explicit in valid:
            return explicit

        tags = {tag.lower() for tag in cls._split_pipe(row.get("tags", ""))}
        followups = cls._split_pipe(row.get("followup_prompts", ""))
        text = cls._strip_accents(" ".join([row.get("user_prompt", "")] + followups))

        if "privacy-simple" in tags:
            return PromptStrategy.PRIVACY_SIMPLE.value
        if "multi-turn" in tags and "privacy-detailed" in tags:
            return PromptStrategy.MULTITURN_DETAILED.value
        if "multi-turn" in tags:
            return PromptStrategy.MULTITURN_SIMPLE.value
        if ("one-shot" in tags and "privacy-detailed" in tags) or (
            "privacy-detailed" in tags and "max-privacy" in tags
        ):
            return PromptStrategy.PRIVACY_DETAILED.value

        if followups:
            if any(marker in text for marker in ("multiples utxos", "decoy", "ofuscar", "tecnica")):
                return PromptStrategy.MULTITURN_DETAILED.value
            return PromptStrategy.MULTITURN_SIMPLE.value

        if any(marker in text for marker in ("no et limitis", "multiples utxos", "decoy", "ofuscar", "tecnica")):
            return PromptStrategy.PRIVACY_DETAILED.value
        if "privada possible" in text or "mes privada" in text:
            return PromptStrategy.PRIVACY_SIMPLE.value
        return PromptStrategy.BASIC.value

    @classmethod
    def infer_prompt_mode(cls, row: Dict) -> str:
        """Infer whether this row should be edited as template-backed or custom text."""
        explicit = (row.get("prompt_mode") or "").strip().lower()
        if explicit in ("template", "custom"):
            return explicit
        # Legacy rows had neither structured amount nor strategy; keep their prompts custom
        # unless the row already carries one of the structured fields.
        if row.get("amount_btc") or row.get("strategy"):
            return "template"
        return "custom"

    def read_experiments(self) -> List[Dict]:
        """
        Read all experiments from CSV.

        Returns:
            List of experiment dicts (raw from CSV)
        """
        if not self.csv_path.exists():
            return []

        experiments = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []

            for row in reader:
                if row.get('id', '').strip():
                    experiments.append(row)

        return experiments

    def read_columns(self) -> List[str]:
        """Return the CSV header columns, if the file exists."""
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or [])

    def _write_columns(self, experiments: List[Dict]) -> List[str]:
        """Preserve existing columns and append supported/new columns."""
        columns: List[str] = []
        for col in self.read_columns() + self.CSV_COLUMNS:
            if col and col not in columns:
                columns.append(col)
        for exp in experiments:
            for col in exp.keys():
                if col and col not in columns:
                    columns.append(col)
        return columns

    def read_experiment_by_id(self, exp_id: str) -> Optional[Dict]:
        """Read a single experiment by ID."""
        experiments = self.read_experiments()
        for exp in experiments:
            if exp.get('id', '').strip() == exp_id:
                return exp
        return None

    def write_experiments(self, experiments: List[Dict]) -> None:
        """Write experiments to CSV (overwrites existing file)."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns = self._write_columns(experiments)

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            for exp in experiments:
                # Ensure all columns present
                row = {col: exp.get(col, '') for col in columns}
                writer.writerow(row)

    def add_experiment(self, exp_meta: ExperimentMeta, xpub: Optional[str] = None) -> None:
        """Add a new experiment to the CSV."""
        experiments = self.read_experiments()

        # Check ID collision
        if any(e.get('id') == exp_meta.id for e in experiments):
            raise ValueError(f"Experiment ID '{exp_meta.id}' already exists")

        prompt_mode = exp_meta.prompt_mode if exp_meta.prompt_mode in ("template", "custom") else "template"
        if prompt_mode == "template":
            prompt_config = generate_prompts(
                amount_btc=exp_meta.amount_btc,
                strategy=exp_meta.strategy,
                language="ca"
            )
            user_prompt = prompt_config["user_prompt"]
            followup_prompts = self._join_pipe(prompt_config["followup_prompts"])
        else:
            user_prompt = exp_meta.user_prompt
            followup_prompts = self._join_pipe(exp_meta.followup_prompts)

        # Create CSV row
        row = {
            'id': exp_meta.id,
            'name': exp_meta.name,
            'description': exp_meta.description,
            'amount_btc': str(exp_meta.amount_btc),
            'strategy': exp_meta.strategy,
            'prompt_mode': prompt_mode,
            'provider': exp_meta.provider,
            'model': exp_meta.model,
            'temperature': str(exp_meta.temperature),
            'system_prompt': exp_meta.system_prompt,
            'user_prompt': user_prompt,
            'followup_prompts': followup_prompts,
            'repetitions': str(exp_meta.repetitions),
            'timeout_seconds': str(exp_meta.timeout_seconds),
            'network': exp_meta.network,
            'tags': '|'.join(exp_meta.tags) if exp_meta.tags else '',
            'enabled': 'true' if exp_meta.enabled else 'false',
            'priority': str(exp_meta.priority),
            'xpub': xpub if xpub is not None else exp_meta.xpub,
        }

        experiments.append(row)
        self.write_experiments(experiments)

    def update_experiment(self, exp_id: str, updates: Dict) -> None:
        """Update an existing experiment."""
        experiments = self.read_experiments()

        found = False
        for i, exp in enumerate(experiments):
            if exp.get('id') == exp_id:
                updates = dict(updates or {})
                regenerate_requested = self._truthy(updates.pop('_regenerate_prompts', False))
                current_prompt_mode = self.infer_prompt_mode(exp)
                next_prompt_mode = updates.get('prompt_mode', current_prompt_mode)
                next_prompt_mode = next_prompt_mode if next_prompt_mode in ('template', 'custom') else current_prompt_mode
                current_amount = self.infer_amount_btc(exp)
                next_amount = self._safe_float(updates.get('amount_btc'), current_amount)
                current_strategy = self.infer_strategy(exp)
                next_strategy = updates.get('strategy') or current_strategy
                prompt_fields_changed = (
                    ('amount_btc' in updates and next_amount != current_amount)
                    or ('strategy' in updates and next_strategy != current_strategy)
                )

                if next_prompt_mode == 'template' and (
                    regenerate_requested or prompt_fields_changed or current_prompt_mode != 'template'
                ):
                    prompt_config = generate_prompts(
                        amount_btc=next_amount,
                        strategy=next_strategy,
                        language="ca"
                    )
                    updates['user_prompt'] = prompt_config['user_prompt']
                    updates['followup_prompts'] = self._join_pipe(prompt_config['followup_prompts'])
                    updates['amount_btc'] = str(next_amount)
                    updates['strategy'] = next_strategy
                updates['prompt_mode'] = next_prompt_mode

                # Merge updates
                for key, value in updates.items():
                    if key in ('tags',) and isinstance(value, list):
                        exp[key] = self._join_pipe(value)
                    elif key in ('followup_prompts',) and isinstance(value, list):
                        exp[key] = self._join_pipe(value)
                    elif key in ('enabled',) and isinstance(value, bool):
                        exp[key] = 'true' if value else 'false'
                    else:
                        exp[key] = str(value)

                experiments[i] = exp
                found = True
                break

        if not found:
            raise ValueError(f"Experiment '{exp_id}' not found")

        self.write_experiments(experiments)

    def delete_experiment(self, exp_id: str) -> None:
        """Delete an experiment by ID."""
        experiments = self.read_experiments()
        experiments = [e for e in experiments if e.get('id') != exp_id]
        self.write_experiments(experiments)

    def clone_experiment(self, src_id: str, new_id: str, updates: Dict = None) -> None:
        """Clone an experiment with a new ID and optional updates."""
        src_exp = self.read_experiment_by_id(src_id)
        if not src_exp:
            raise ValueError(f"Source experiment '{src_id}' not found")

        if self.read_experiment_by_id(new_id):
            raise ValueError(f"Target experiment ID '{new_id}' already exists")

        # Copy and apply updates
        new_exp = src_exp.copy()
        new_exp['id'] = new_id

        if updates:
            updates = dict(updates)
            current_prompt_mode = self.infer_prompt_mode(new_exp)
            next_prompt_mode = updates.get('prompt_mode', current_prompt_mode)
            next_prompt_mode = next_prompt_mode if next_prompt_mode in ('template', 'custom') else current_prompt_mode
            should_regenerate = next_prompt_mode == 'template' and (
                'amount_btc' in updates or 'strategy' in updates or current_prompt_mode != 'template'
            )
            for key, value in updates.items():
                if key in ('tags',) and isinstance(value, list):
                    new_exp[key] = self._join_pipe(value)
                elif key in ('followup_prompts',) and isinstance(value, list):
                    new_exp[key] = self._join_pipe(value)
                elif key in ('enabled',) and isinstance(value, bool):
                    new_exp[key] = 'true' if value else 'false'
                else:
                    new_exp[key] = str(value)
            new_exp['prompt_mode'] = next_prompt_mode
            if should_regenerate:
                amount_btc = self._safe_float(new_exp.get('amount_btc'), self.infer_amount_btc(new_exp))
                strategy = new_exp.get('strategy') or self.infer_strategy(new_exp)
                prompt_config = generate_prompts(amount_btc=amount_btc, strategy=strategy, language="ca")
                new_exp['amount_btc'] = str(amount_btc)
                new_exp['strategy'] = strategy
                new_exp['user_prompt'] = prompt_config['user_prompt']
                new_exp['followup_prompts'] = self._join_pipe(prompt_config['followup_prompts'])

        experiments = self.read_experiments()
        experiments.append(new_exp)
        self.write_experiments(experiments)

    def parse_csv_row_to_meta(self, row: Dict) -> ExperimentMeta:
        """Convert a CSV row to ExperimentMeta (for UI editing)."""
        tags_str = row.get('tags', '')
        tags = self._split_pipe(tags_str) if tags_str else []
        followups = self._split_pipe(row.get('followup_prompts', ''))

        return ExperimentMeta(
            id=row.get('id', ''),
            name=row.get('name', row.get('id', '')),
            description=row.get('description', ''),
            amount_btc=self.infer_amount_btc(row),
            strategy=self.infer_strategy(row),
            provider=row.get('provider', 'openai'),
            model=row.get('model', 'gpt-4o'),
            temperature=self._safe_float(row.get('temperature'), 1.0),
            repetitions=self._safe_int(row.get('repetitions'), 3),
            timeout_seconds=self._safe_int(row.get('timeout_seconds'), 300),
            network=row.get('network', 'mainnet'),
            tags=tags,
            enabled=self._truthy(row.get('enabled', 'true')),
            priority=self._safe_int(row.get('priority'), 1),
            prompt_mode=self.infer_prompt_mode(row),
            system_prompt=row.get('system_prompt', ''),
            user_prompt=row.get('user_prompt', ''),
            followup_prompts=followups,
            xpub=row.get('xpub', ''),
        )


if __name__ == "__main__":
    # Quick test
    csv_path = Path("experiments_test.csv")
    manager = ExperimentManager(csv_path)

    # Add test experiment
    exp = ExperimentMeta(
        id="test_basic",
        name="Test Basic",
        amount_btc=0.5,
        strategy="basic",
        provider="anthropic",
        model="claude-opus-4",
        tags=["test", "basic"],
    )

    try:
        manager.add_experiment(exp)
        print("✓ Added experiment")

        experiments = manager.read_experiments()
        print(f"✓ Read {len(experiments)} experiments")

        if experiments:
            exp_row = experiments[0]
            print(f"  First experiment: {exp_row.get('id')} - Prompt: {exp_row.get('user_prompt')[:50]}...")
    finally:
        # Cleanup
        if csv_path.exists():
            csv_path.unlink()
