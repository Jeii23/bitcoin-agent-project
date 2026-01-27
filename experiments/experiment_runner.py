#!/usr/bin/env python3
"""
Privacy Experiment Runner
=========================

Automated execution of Bitcoin Agent privacy experiments defined in CSV format.
Each experiment generates a PSBT which is evaluated by the Privacy Scorer.

Usage:
    python experiment_runner.py experiments.csv
    python experiment_runner.py experiments.csv --filter tag:privacy-high
    python experiment_runner.py experiments.csv --filter id:exp_openai_basic
    python experiment_runner.py experiments.csv --dry-run

CSV Format:
    id,name,provider,model,temperature,user_prompt,followup_prompts,repetitions,timeout_seconds,network,tags,enabled
    
    - followup_prompts: pipe-separated (|) for multiple followups
    - tags: pipe-separated (|)
    - enabled: true/false

Output:
    - results/experiments_YYYYMMDD_HHMMSS.csv (tabular results)
    - results/experiments_YYYYMMDD_HHMMSS.json (detailed results)
    - results/psbts/<experiment_id>_<rep>.psbt (generated PSBTs)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

# Add source paths
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
SCORING_DIR = SCRIPT_DIR.parent.parent / "scoring"
PROJECT_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCORING_DIR))

# Load .env file before importing agent (which requires dotenv)
try:
    from dotenv import load_dotenv
    # Try to load .env from project directory
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv not available, try to load manually
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Remove inline comments
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    os.environ.setdefault(key, value)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy external loggers
for _noisy in ("httpcore", "httpx", "openai", "anthropic", "urllib3", "langchain", "langgraph"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class LLMConfig:
    """LLM provider and model configuration."""
    provider: str  # openai, anthropic, google
    model: str
    temperature: float = 1.0
    api_key_env: Optional[str] = None


@dataclass
class PromptConfig:
    """Prompt configuration for an experiment."""
    system_prompt: Optional[str] = None
    user_prompt: str = ""
    followup_prompts: List[str] = field(default_factory=list)


@dataclass
class WalletConfig:
    """Bitcoin wallet configuration."""
    xpub: Optional[str] = None
    network: str = "testnet"


@dataclass
class ExperimentSettings:
    """Execution settings for an experiment."""
    timeout_seconds: int = 120
    repetitions: int = 1
    save_psbt: bool = True
    output_dir: Optional[str] = None


@dataclass
class Experiment:
    """Single experiment definition."""
    id: str
    name: str
    description: str
    llm: LLMConfig
    prompts: PromptConfig
    wallet: WalletConfig
    settings: ExperimentSettings
    tags: List[str]
    enabled: bool = True
    priority: int = 1


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str
    experiment_name: str
    repetition: int
    timestamp: str
    
    # LLM Info
    llm_provider: str
    llm_model: str
    llm_temperature: float
    
    # Prompts
    system_prompt: Optional[str]
    user_prompt: str
    
    # Execution
    success: bool
    error_message: Optional[str]
    execution_time_seconds: float
    
    # PSBT
    psbt_generated: bool
    psbt_base64: Optional[str]
    psbt_file: Optional[str]
    
    # Privacy Score
    privacy_score: Optional[int]
    privacy_grade: Optional[str]
    privacy_breakdown: Optional[Dict[str, Any]]
    
    # Agent response (for debugging)
    agent_response: Optional[str] = None
    
    # Tags for analysis
    tags: List[str] = field(default_factory=list)


# ==============================================================================
# CSV Parser
# ==============================================================================

class ExperimentCSVParser:
    """Parser for experiment CSV files.
    
    CSV Format:
        id,name,provider,model,temperature,user_prompt,followup_prompts,repetitions,timeout_seconds,network,tags,enabled
        
    Notes:
        - followup_prompts: pipe-separated (|) for multiple followups
        - tags: pipe-separated (|)
        - enabled: true/false
    """
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
    
    def parse(self) -> List[Experiment]:
        """Parse CSV file and return list of experiments."""
        experiments = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    exp = self._parse_row(row)
                    if exp:
                        experiments.append(exp)
                except Exception as e:
                    logger.warning(f"Error parsing row {row.get('id', '?')}: {e}")
        
        return experiments
    
    def _parse_row(self, row: Dict[str, str]) -> Optional[Experiment]:
        """Parse a single CSV row into an Experiment."""
        exp_id = row.get('id', '').strip()
        if not exp_id:
            return None
        
        # Parse followup prompts (pipe-separated)
        followups_raw = row.get('followup_prompts', '').strip()
        followups = [f.strip() for f in followups_raw.split('|') if f.strip()] if followups_raw else []
        
        # Parse tags (pipe-separated)
        tags_raw = row.get('tags', '').strip()
        tags = [t.strip() for t in tags_raw.split('|') if t.strip()] if tags_raw else []
        
        # Parse enabled
        enabled_str = row.get('enabled', 'true').strip().lower()
        enabled = enabled_str in ('true', '1', 'yes', 'y')
        
        # Get XPUB from CSV, or from env (BITCOIN_XPUB or XPUB)
        xpub = row.get('xpub', '').strip() or os.getenv('BITCOIN_XPUB') or os.getenv('XPUB')
        
        # Build experiment
        return Experiment(
            id=exp_id,
            name=row.get('name', exp_id).strip(),
            description=row.get('description', '').strip(),
            llm=LLMConfig(
                provider=row.get('provider', 'openai').strip(),
                model=row.get('model', 'gpt-4o').strip(),
                temperature=float(row.get('temperature', '1.0').strip() or '1.0'),
                api_key_env=row.get('api_key_env', '').strip() or None,
            ),
            prompts=PromptConfig(
                system_prompt=row.get('system_prompt', '').strip() or None,
                user_prompt=row.get('user_prompt', '').strip(),
                followup_prompts=followups,
            ),
            wallet=WalletConfig(
                xpub=xpub,
                network=row.get('network', 'testnet').strip(),
            ),
            settings=ExperimentSettings(
                timeout_seconds=int(row.get('timeout_seconds', '120').strip() or '120'),
                repetitions=int(row.get('repetitions', '1').strip() or '1'),
            ),
            tags=tags,
            enabled=enabled,
            priority=int(row.get('priority', '1').strip() or '1'),
        )


# ==============================================================================
# Experiment Runner
# ==============================================================================

class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
        self._agent_class = None
        self._privacy_scorer_class = None
    
    def _lazy_import_agent(self):
        """Lazy import of BitcoinAIAgent to avoid import at module level."""
        if self._agent_class is None:
            from bitcoin_ai_agent import BitcoinAIAgent
            self._agent_class = BitcoinAIAgent
        return self._agent_class
    
    def _lazy_import_scorer(self):
        """Lazy import of PrivacyScorer."""
        if self._privacy_scorer_class is None:
            try:
                from privacy_scorer import PrivacyScorer
                self._privacy_scorer_class = PrivacyScorer
            except ImportError:
                logger.warning("Could not import PrivacyScorer, scoring will be disabled")
                self._privacy_scorer_class = None
        return self._privacy_scorer_class
    
    async def run_experiment(self, exp: Experiment, repetition: int = 1, max_retries: int = 3) -> ExperimentResult:
        """Run a single experiment with retry logic for transient errors."""
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Running experiment: {exp.id} (rep {repetition}/{exp.settings.repetitions})")
        logger.info(f"  LLM: {exp.llm.provider}/{exp.llm.model} @ temp={exp.llm.temperature}")
        logger.info(f"  Prompt: {exp.prompts.user_prompt[:80]}...")
        
        result = ExperimentResult(
            experiment_id=exp.id,
            experiment_name=exp.name,
            repetition=repetition,
            timestamp=timestamp,
            llm_provider=exp.llm.provider,
            llm_model=exp.llm.model,
            llm_temperature=exp.llm.temperature,
            system_prompt=exp.prompts.system_prompt,
            user_prompt=exp.prompts.user_prompt,
            success=False,
            error_message=None,
            execution_time_seconds=0.0,
            psbt_generated=False,
            psbt_base64=None,
            psbt_file=None,
            privacy_score=None,
            privacy_grade=None,
            privacy_breakdown=None,
            tags=exp.tags,
        )
        
        try:
            # Resolve API key from environment if specified
            api_key = None
            if exp.llm.api_key_env:
                api_key = os.getenv(exp.llm.api_key_env)
            
            # Create agent with specific LLM config
            BitcoinAIAgent = self._lazy_import_agent()
            agent = BitcoinAIAgent(
                llm_provider=exp.llm.provider,
                llm_model=exp.llm.model,
                api_key=api_key,
                temperature=exp.llm.temperature,
            )
            
            # Configure wallet
            if exp.wallet.xpub:
                agent.setup(exp.wallet.xpub, exp.wallet.network)
            else:
                agent.network = exp.wallet.network
            
            # Inject custom system prompt if provided
            if exp.prompts.system_prompt:
                # Override the system prompt method
                original_get_system_prompt = agent._get_system_prompt
                agent._get_system_prompt = lambda: exp.prompts.system_prompt
            
            # Execute the main prompt with retry logic
            response = None
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    response = await asyncio.wait_for(
                        agent.chat(exp.prompts.user_prompt),
                        timeout=exp.settings.timeout_seconds
                    )
                    logger.debug(f"  Response preview: {str(response)[:200]}...")
                    break  # Success, exit retry loop
                except asyncio.TimeoutError:
                    result.error_message = f"Timeout after {exp.settings.timeout_seconds}s"
                    return result
                except Exception as e:
                    error_str = str(e)
                    # Check for retryable errors (rate limit, overloaded)
                    if any(code in error_str for code in ['429', '529', '503', 'overloaded', 'rate_limit', 'UNAVAILABLE']):
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 2 ** retry_count * 5  # 10s, 20s, 40s
                            logger.warning(f"  Retryable error, waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                            continue
                    last_error = e
                    break
            
            if response is None and last_error:
                raise last_error
            
            # Execute follow-up prompts if any (with retry logic)
            for followup in exp.prompts.followup_prompts:
                followup_retry = 0
                while followup_retry <= max_retries:
                    try:
                        response = await asyncio.wait_for(
                            agent.chat(followup),
                            timeout=exp.settings.timeout_seconds
                        )
                        logger.debug(f"  Follow-up response preview: {str(response)[:200]}...")
                        break
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout on followup prompt: {followup[:50]}...")
                        break
                    except Exception as e:
                        error_str = str(e)
                        if any(code in error_str for code in ['429', '529', '503', 'overloaded', 'rate_limit', 'UNAVAILABLE']):
                            followup_retry += 1
                            if followup_retry <= max_retries:
                                wait_time = 2 ** followup_retry * 5
                                logger.warning(f"  Retryable error on followup, waiting {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                        logger.warning(f"Error on followup prompt: {e}")
                        break
            
            # Extract PSBT from agent (primary method)
            psbt_base64 = agent.last_psbt
            
            # Fallback: try to extract PSBT from response text if agent.last_psbt is empty
            if not psbt_base64 and response:
                import re
                # Look for PSBT base64 pattern in response
                psbt_pattern = re.compile(r'(cHNidP[A-Za-z0-9+/=]{50,})')
                match = psbt_pattern.search(str(response))
                if match:
                    psbt_base64 = match.group(1).strip()
                    # Clean whitespace
                    psbt_base64 = re.sub(r'\s+', '', psbt_base64)
                    logger.info(f"  PSBT extracted from response text (fallback)")
            
            if psbt_base64:
                result.psbt_generated = True
                result.psbt_base64 = psbt_base64
                
                # Save PSBT to file
                if exp.settings.save_psbt:
                    psbt_dir = self.output_dir / "psbts"
                    psbt_dir.mkdir(parents=True, exist_ok=True)
                    psbt_filename = f"{exp.id}_rep{repetition}.psbt"
                    psbt_path = psbt_dir / psbt_filename
                    
                    import base64
                    psbt_path.write_bytes(base64.b64decode(psbt_base64))
                    result.psbt_file = str(psbt_path)
                    
                    # Also save base64
                    (psbt_dir / f"{exp.id}_rep{repetition}.base64").write_text(psbt_base64)
                
                # Calculate privacy score
                PrivacyScorer = self._lazy_import_scorer()
                if PrivacyScorer:
                    try:
                        scorer = PrivacyScorer(network=exp.wallet.network)
                        score, breakdown = scorer.score_psbt(psbt_base64)
                        result.privacy_score = score
                        result.privacy_grade = self._get_grade(score)
                        result.privacy_breakdown = breakdown
                        logger.info(f"  Privacy Score: {score}/100 ({result.privacy_grade})")
                    except Exception as e:
                        logger.warning(f"  Privacy scoring failed: {e}")
                        result.privacy_breakdown = {"error": str(e)}
            else:
                logger.warning(f"  No PSBT generated")
                # Save agent response for debugging
                if response:
                    result.agent_response = str(response)[:2000]  # Truncate for storage
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Experiment {exp.id} failed: {e}")
            result.error_message = str(e)
        
        finally:
            end_time = datetime.now()
            result.execution_time_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        elif score >= 30:
            return "E"
        else:
            return "F"
    
    async def run_all(self, experiments: List[Experiment], filter_fn=None, interleave: bool = False, delay: float = 0) -> List[ExperimentResult]:
        """Run all experiments.
        
        Args:
            experiments: List of experiments to run
            filter_fn: Optional filter function
            interleave: If True, interleave by provider (round-robin)
            delay: Delay in seconds between experiments
        """
        # Filter enabled experiments
        active_experiments = [e for e in experiments if e.enabled]
        
        # Apply custom filter if provided
        if filter_fn:
            active_experiments = [e for e in active_experiments if filter_fn(e)]
        
        # Sort by priority
        active_experiments.sort(key=lambda e: e.priority)
        
        # Interleave by provider if requested (round-robin)
        if interleave:
            active_experiments = self._interleave_by_provider(active_experiments)
            logger.info("Interleaving experiments by provider (round-robin)")
        
        logger.info(f"Running {len(active_experiments)} experiments...")
        
        for i, exp in enumerate(active_experiments):
            for rep in range(1, exp.settings.repetitions + 1):
                result = await self.run_experiment(exp, rep)
                self.results.append(result)
                
                # Add delay between experiments (helps with rate limits)
                if delay > 0 and (i < len(active_experiments) - 1 or rep < exp.settings.repetitions):
                    await asyncio.sleep(delay)
        
        return self.results
    
    def _interleave_by_provider(self, experiments: List[Experiment]) -> List[Experiment]:
        """Interleave experiments by provider for better rate limit distribution.
        
        Groups experiments by provider, then returns them in round-robin order.
        Example: [openai1, anthropic1, google1, openai2, anthropic2, google2, ...]
        """
        from collections import defaultdict
        
        # Group by provider
        by_provider: Dict[str, List[Experiment]] = defaultdict(list)
        for exp in experiments:
            by_provider[exp.llm.provider].append(exp)
        
        # Round-robin interleave
        result = []
        providers = list(by_provider.keys())
        max_len = max(len(v) for v in by_provider.values()) if by_provider else 0
        
        for i in range(max_len):
            for provider in providers:
                if i < len(by_provider[provider]):
                    result.append(by_provider[provider][i])
        
        return result
    
    def save_results(self):
        """Save results to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (detailed)
        json_path = self.output_dir / f"experiments_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        logger.info(f"Detailed results saved to: {json_path}")
        
        # Save CSV (summary)
        csv_path = self.output_dir / f"experiments_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "experiment_id",
                "experiment_name",
                "repetition",
                "timestamp",
                "llm_provider",
                "llm_model",
                "llm_temperature",
                "user_prompt",
                "success",
                "error_message",
                "execution_time_seconds",
                "psbt_generated",
                "privacy_score",
                "privacy_grade",
                "tags",
            ])
            # Data rows
            for r in self.results:
                writer.writerow([
                    r.experiment_id,
                    r.experiment_name,
                    r.repetition,
                    r.timestamp,
                    r.llm_provider,
                    r.llm_model,
                    r.llm_temperature,
                    r.user_prompt[:100] + "..." if len(r.user_prompt) > 100 else r.user_prompt,
                    r.success,
                    r.error_message or "",
                    f"{r.execution_time_seconds:.2f}",
                    r.psbt_generated,
                    r.privacy_score or "",
                    r.privacy_grade or "",
                    ";".join(r.tags),
                ])
        logger.info(f"Summary CSV saved to: {csv_path}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print experiment summary to console."""
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 70)
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        with_psbt = sum(1 for r in self.results if r.psbt_generated)
        
        print(f"\nTotal experiments: {total}")
        print(f"Successful: {successful} ({100*successful/total:.1f}%)" if total else "N/A")
        print(f"PSBTs generated: {with_psbt}")
        
        # Privacy score statistics
        scores = [r.privacy_score for r in self.results if r.privacy_score is not None]
        if scores:
            print(f"\nPrivacy Scores:")
            print(f"  Min: {min(scores)}")
            print(f"  Max: {max(scores)}")
            print(f"  Avg: {sum(scores)/len(scores):.1f}")
        
        # By provider
        print(f"\nBy LLM Provider:")
        providers = set(r.llm_provider for r in self.results)
        for provider in sorted(providers):
            prov_results = [r for r in self.results if r.llm_provider == provider]
            prov_scores = [r.privacy_score for r in prov_results if r.privacy_score is not None]
            avg = sum(prov_scores) / len(prov_scores) if prov_scores else 0
            print(f"  {provider}: {len(prov_results)} experiments, avg score: {avg:.1f}")
        
        # By model
        print(f"\nBy Model:")
        models = set((r.llm_provider, r.llm_model) for r in self.results)
        for provider, model in sorted(models):
            mod_results = [r for r in self.results if r.llm_model == model]
            mod_scores = [r.privacy_score for r in mod_results if r.privacy_score is not None]
            avg = sum(mod_scores) / len(mod_scores) if mod_scores else 0
            print(f"  {provider}/{model}: {len(mod_results)} exp, avg: {avg:.1f}")
        
        print("\n" + "=" * 70)


# ==============================================================================
# Filter Functions
# ==============================================================================

def create_filter(filter_str: str):
    """Create a filter function from a filter string.
    
    Formats:
        tag:privacy-high  - Filter by tag
        provider:openai   - Filter by provider
        model:gpt-4o      - Filter by model
    """
    if not filter_str:
        return None
    
    parts = filter_str.split(":", 1)
    if len(parts) != 2:
        logger.warning(f"Invalid filter format: {filter_str}")
        return None
    
    filter_type, filter_value = parts[0].lower(), parts[1].lower()
    
    if filter_type == "tag":
        return lambda e: any(t.lower() == filter_value for t in e.tags)
    elif filter_type == "provider":
        return lambda e: e.llm.provider.lower() == filter_value
    elif filter_type == "model":
        return lambda e: filter_value in e.llm.model.lower()
    elif filter_type == "id":
        return lambda e: e.id.lower() == filter_value
    elif filter_type == "name":
        return lambda e: filter_value in e.name.lower()
    else:
        logger.warning(f"Unknown filter type: {filter_type}")
        return None


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Bitcoin Agent privacy experiments from XML or CSV definition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments from CSV file (recommended)
    python experiment_runner.py experiments.csv
    
    # Run all experiments from XML file
    python experiment_runner.py experiments.xml
    
    # Run only experiments tagged 'privacy-high'
    python experiment_runner.py experiments.csv --filter tag:privacy-high
    
    # Run experiments using only OpenAI
    python experiment_runner.py experiments.csv --filter provider:openai
    
    # Run a specific experiment by ID
    python experiment_runner.py experiments.csv --filter id:exp_openai_basic
    
    # Dry run (parse and validate without executing)
    python experiment_runner.py experiments.csv --dry-run
    
    # Custom output directory
    python experiment_runner.py experiments.csv --output ./my_results
        """
    )
    parser.add_argument("input_file", type=Path, help="Path to experiments CSV file")
    parser.add_argument("--filter", type=str, help="Filter experiments (e.g., tag:privacy-high, provider:openai, id:exp_001)")
    parser.add_argument("--experiment", type=str, help="Run only a specific experiment by ID (shortcut for --filter id:...)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without executing")
    parser.add_argument("--interleave", action="store_true", help="Interleave experiments by provider (round-robin: openai, anthropic, google)")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between experiments (helps avoid rate limits)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check input file exists
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Parse CSV
    logger.info(f"Parsing experiments from: {args.input_file}")
    experiments = ExperimentCSVParser(args.input_file).parse()
    
    if not experiments:
        logger.error("No experiments found in CSV file")
        sys.exit(1)
    
    logger.info(f"Found {len(experiments)} experiments")
    
    # Apply filters
    filter_fn = None
    if args.experiment:
        filter_fn = lambda e: e.id == args.experiment
    elif args.filter:
        filter_fn = create_filter(args.filter)
    
    # Dry run: just print parsed experiments
    if args.dry_run:
        print("\n=== DRY RUN: Parsed Experiments ===\n")
        for exp in experiments:
            if filter_fn and not filter_fn(exp):
                continue
            print(f"ID: {exp.id}")
            print(f"  Name: {exp.name}")
            print(f"  LLM: {exp.llm.provider}/{exp.llm.model} @ {exp.llm.temperature}")
            print(f"  Prompt: {exp.prompts.user_prompt[:80]}...")
            print(f"  Network: {exp.wallet.network}")
            print(f"  Repetitions: {exp.settings.repetitions}")
            print(f"  Tags: {', '.join(exp.tags)}")
            print(f"  Enabled: {exp.enabled}")
            print()
        sys.exit(0)
    
    # Run experiments
    runner = ExperimentRunner(args.output)
    asyncio.run(runner.run_all(experiments, filter_fn, interleave=args.interleave, delay=args.delay))
    
    # Save results
    runner.save_results()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
