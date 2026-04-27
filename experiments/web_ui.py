#!/usr/bin/env python3
"""
Bitcoin Agent Experiments Web UI
=================================

Streamlit-based web interface for managing and running Bitcoin privacy experiments.

Usage:
    streamlit run web_ui.py
"""

import streamlit as st
import pandas as pd
import subprocess
import logging
import sys
import re
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

# Make local helper imports robust whether Streamlit is launched from
# experiments/ or from the project root/testing harness.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_manager import ExperimentManager, ExperimentMeta, PromptStrategy
from paper_charts import (
    ANALYSIS_CHARTS_DIR,
    PAPER_CHART_OPTIONS,
    aggregate_current_results,
    build_paper_chart,
    current_results_v2_scores,
    load_paper_chart_sources,
    prepare_aggregated_dataframe,
    prepare_v2_scores_dataframe,
)
from prompt_templates import generate_prompts
from result_utils import arrow_safe_dataframe, display_columns, load_many_results_dataframes, load_results_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bitcoin Privacy Experiments",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = SCRIPT_DIR / "experiments.csv"
RESULTS_DIR = SCRIPT_DIR / "results"


PROVIDER_OPTIONS = ["openai", "anthropic", "google", "openrouter"]
NETWORK_OPTIONS = ["mainnet", "testnet"]
PROMPT_MODE_OPTIONS = ["template", "custom"]


def get_manager() -> ExperimentManager:
    """Get experiment manager instance."""
    return ExperimentManager(CSV_PATH)


def show_dataframe(df: pd.DataFrame, **kwargs):
    """Display a dataframe after making object columns Arrow-compatible."""
    st.dataframe(arrow_safe_dataframe(df), **kwargs)


def show_pdf_preview(pdf_path: Path, *, height: int = 720):
    """Embed a generated PDF chart in Streamlit."""
    st.iframe(pdf_path, width="stretch", height=height)


_RESULT_FILE_RE = re.compile(r"experiments_(\d{8}_\d{6})\.csv$")


@dataclass
class RunnerCommandInfo:
    pid: Optional[int]
    elapsed_seconds: int
    command: str
    input_file: Optional[Path]
    output_dir: Path
    filter_expr: Optional[str] = None
    experiment_id: Optional[str] = None
    dry_run: bool = False
    include_disabled: bool = False


@dataclass
class LiveExecutionStatus:
    command_info: RunnerCommandInfo
    expected_runs: Optional[int]
    result_file: Optional[Path]
    completed_runs: int
    successful_runs: int
    psbt_runs: int
    failed_runs: int
    last_timestamp: Optional[str]
    last_experiment_id: Optional[str]
    last_model: Optional[str]


def load_css():
    """Load custom CSS for better styling."""
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; }
    .error-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)


def option_index(options: List[str], value: str, default: int = 0) -> int:
    """Return a safe Streamlit selectbox index."""
    return options.index(value) if value in options else default


def truthy_series(series: pd.Series) -> pd.Series:
    """Normalize CSV bool values that may arrive as bools or strings."""
    values = series.astype("object").where(series.notna(), "")
    return values.astype(str).str.lower().isin({"true", "1", "1.0", "yes", "y"})


def format_duration(seconds: Optional[int]) -> str:
    """Format a duration compactly for live status displays."""
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def split_filter_values(value: str) -> List[str]:
    """Split pasted ID/tag text on commas, pipes, whitespace, or newlines."""
    if not value:
        return []
    normalized = value.replace("|", ",").replace("\n", ",").replace("\t", ",")
    values = []
    for chunk in normalized.split(","):
        for item in chunk.split():
            item = item.strip()
            if item:
                values.append(item)
    return values


def safe_prompt_mode(manager: ExperimentManager, exp: Dict, meta: ExperimentMeta) -> str:
    """Return a prompt mode even if an older ExperimentMeta lacks the field."""
    explicit = getattr(meta, "prompt_mode", None)
    if explicit in PROMPT_MODE_OPTIONS:
        return explicit
    if hasattr(manager, "infer_prompt_mode"):
        inferred = manager.infer_prompt_mode(exp)
        if inferred in PROMPT_MODE_OPTIONS:
            return inferred
    raw = str(exp.get("prompt_mode", "")).strip().lower()
    if raw in PROMPT_MODE_OPTIONS:
        return raw
    return "template" if exp.get("amount_btc") or exp.get("strategy") else "custom"


def sort_results_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Show the newest completed runs first, even when multiple files are loaded."""
    if df is None or df.empty:
        return df

    sorted_df = df.copy()
    sort_columns = []
    ascending = []

    if "timestamp" in sorted_df.columns:
        sorted_df["_sort_timestamp"] = pd.to_datetime(sorted_df["timestamp"], errors="coerce")
        sort_columns.append("_sort_timestamp")
        ascending.append(False)
    if "source_file" in sorted_df.columns:
        sort_columns.append("source_file")
        ascending.append(False)
    if "repetition" in sorted_df.columns:
        sort_columns.append("repetition")
        ascending.append(False)

    if not sort_columns:
        return sorted_df

    sorted_df = sorted_df.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="stable",
        na_position="last",
    )
    return sorted_df.drop(columns=["_sort_timestamp"], errors="ignore")


def all_tags_from_dataframe(exp_df: pd.DataFrame) -> List[str]:
    """Return sorted unique pipe-separated tags from an experiments dataframe."""
    tags = set()
    if "tags" not in exp_df.columns:
        return []
    for value in exp_df["tags"].fillna(""):
        tags.update(tag.strip() for tag in str(value).split("|") if tag.strip())
    return sorted(tags)


def filter_run_experiments(
    exp_df: pd.DataFrame,
    *,
    enabled_scope: str = "Enabled only",
    providers: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    networks: Optional[List[str]] = None,
    prompt_modes: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    tag_match: str = "Any selected tag",
    search: str = "",
    amount_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """Apply the Run Experiments selection filters."""
    if exp_df.empty:
        return exp_df.copy()

    filtered = exp_df.copy()

    if enabled_scope == "Enabled only" and "enabled" in filtered.columns:
        filtered = filtered[filtered["enabled"] == True]  # noqa: E712
    elif enabled_scope == "Disabled only" and "enabled" in filtered.columns:
        filtered = filtered[filtered["enabled"] == False]  # noqa: E712

    for column, selected in (
        ("provider", providers or []),
        ("model", models or []),
        ("strategy", strategies or []),
        ("network", networks or []),
        ("prompt_mode", prompt_modes or []),
    ):
        if selected and column in filtered.columns:
            filtered = filtered[filtered[column].isin(selected)]

    if tags and "tags" in filtered.columns:
        wanted = set(tags)

        def row_matches_tags(value: str) -> bool:
            row_tags = {tag.strip() for tag in str(value or "").split("|") if tag.strip()}
            if tag_match == "All selected tags":
                return wanted.issubset(row_tags)
            return bool(row_tags & wanted)

        filtered = filtered[filtered["tags"].apply(row_matches_tags)]

    if search:
        needle = search.strip().lower()
        if needle:
            haystack = (
                filtered.get("id", "").astype(str) + " "
                + filtered.get("name", "").astype(str) + " "
                + filtered.get("model", "").astype(str) + " "
                + filtered.get("provider", "").astype(str) + " "
                + filtered.get("tags", "").astype(str)
            ).str.lower()
            filtered = filtered[haystack.str.contains(needle, regex=False, na=False)]

    if amount_range and "amount_btc" in filtered.columns:
        min_amount, max_amount = amount_range
        amounts = pd.to_numeric(filtered["amount_btc"], errors="coerce")
        filtered = filtered[amounts.between(min_amount, max_amount, inclusive="both")]

    return filtered


def experiments_dataframe(manager: ExperimentManager, experiments: List[Dict]) -> pd.DataFrame:
    """Build a display dataframe with inferred legacy fields."""
    rows = []
    for exp in experiments:
        meta = manager.parse_csv_row_to_meta(exp)
        prompt_mode = safe_prompt_mode(manager, exp, meta)
        amount_btc = None if manager.infer_amount_percent(exp) is not None else meta.amount_btc
        rows.append({
            "id": meta.id,
            "name": meta.name,
            "provider": meta.provider,
            "model": meta.model,
            "strategy": meta.strategy,
            "amount_display": manager.infer_amount_display(exp),
            "amount_btc": amount_btc,
            "prompt_mode": prompt_mode,
            "temperature": meta.temperature,
            "repetitions": meta.repetitions,
            "timeout_seconds": meta.timeout_seconds,
            "network": meta.network,
            "enabled": meta.enabled,
            "tags": "|".join(meta.tags),
        })
    return pd.DataFrame(rows)


def build_runner_command(
    selected_ids: List[str],
    *,
    interleave: bool = False,
    delay: float = 0,
    dry_run: bool = False,
    include_disabled: bool = False,
    parallel_profile: str = "sequential",
    max_concurrency: Optional[int] = None,
    provider_limits: str = "",
    model_limit: Optional[int] = None,
) -> List[str]:
    """Build the CLI command the UI will execute."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "experiment_runner.py"),
        str(CSV_PATH),
        "--output",
        str(RESULTS_DIR),
    ]
    if selected_ids:
        cmd.extend(["--filter", f"ids:{','.join(selected_ids)}"])
    if interleave:
        cmd.append("--interleave")
    if delay > 0:
        cmd.extend(["--delay", str(delay)])
    if parallel_profile and parallel_profile != "sequential":
        cmd.extend(["--parallel-profile", parallel_profile])
    if max_concurrency and (parallel_profile != "sequential" or int(max_concurrency) != 1):
        cmd.extend(["--max-concurrency", str(max_concurrency)])
    if provider_limits.strip():
        cmd.extend(["--provider-limits", provider_limits.strip()])
    if model_limit and parallel_profile == "model":
        cmd.extend(["--model-limit", str(model_limit)])
    if dry_run:
        cmd.append("--dry-run")
    if include_disabled:
        cmd.append("--include-disabled")
    return cmd


def estimate_runner_timeout_seconds(
    selected_meta: List[ExperimentMeta],
    *,
    parallel_profile: str = "sequential",
    max_concurrency: int = 1,
    model_limit: int = 1,
    delay: float = 0,
) -> int:
    """Estimate subprocess timeout with a conservative concurrency-aware upper bound."""
    task_costs = [meta.timeout_seconds for meta in selected_meta for _ in range(meta.repetitions)]
    if not task_costs:
        return 120

    sequential_seconds = sum(task_costs)
    if parallel_profile == "sequential":
        estimate = sequential_seconds
    elif parallel_profile == "provider":
        by_provider: Dict[str, int] = {}
        for meta in selected_meta:
            by_provider[meta.provider] = by_provider.get(meta.provider, 0) + meta.timeout_seconds * meta.repetitions
        estimate = max(by_provider.values()) if by_provider else sequential_seconds
    else:
        by_model: Dict[Tuple[str, str], int] = {}
        for meta in selected_meta:
            key = (meta.provider, meta.model)
            by_model[key] = by_model.get(key, 0) + meta.timeout_seconds * meta.repetitions
        lane_bound = max(by_model.values()) if by_model else sequential_seconds
        throughput_bound = sequential_seconds / max(1, max_concurrency)
        estimate = max(lane_bound / max(1, model_limit), throughput_bound)

    estimate += delay * max(0, len(task_costs) - 1)
    return int(estimate + max(120, 0.15 * estimate))


def run_runner_command(cmd: List[str], timeout_seconds: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run the experiment runner from the experiments directory."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
        timeout=timeout_seconds,
    )


def _resolve_process_path(value: Optional[str], cwd: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (cwd / path).resolve()


def _process_cwd(pid: Optional[int]) -> Path:
    if not pid:
        return SCRIPT_DIR
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return SCRIPT_DIR


def parse_runner_command_info(command: str, *, pid: Optional[int] = None, elapsed_seconds: int = 0) -> Optional[RunnerCommandInfo]:
    """Extract runner CSV/output/filter details from a process command line."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    runner_idx = None
    for idx, token in enumerate(tokens):
        if Path(token).name == "experiment_runner.py":
            runner_idx = idx
            break
    if runner_idx is None:
        return None

    cwd = _process_cwd(pid)
    input_token = tokens[runner_idx + 1] if runner_idx + 1 < len(tokens) else None
    if input_token and input_token.startswith("-"):
        input_token = None

    output_token = "results"
    filter_expr = None
    experiment_id = None
    dry_run = False
    include_disabled = False
    idx = runner_idx + 2
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--output" and idx + 1 < len(tokens):
            output_token = tokens[idx + 1]
            idx += 2
            continue
        if token.startswith("--output="):
            output_token = token.split("=", 1)[1]
        elif token == "--filter" and idx + 1 < len(tokens):
            filter_expr = tokens[idx + 1]
            idx += 2
            continue
        elif token.startswith("--filter="):
            filter_expr = token.split("=", 1)[1]
        elif token == "--experiment" and idx + 1 < len(tokens):
            experiment_id = tokens[idx + 1]
            idx += 2
            continue
        elif token.startswith("--experiment="):
            experiment_id = token.split("=", 1)[1]
        elif token == "--dry-run":
            dry_run = True
        elif token == "--include-disabled":
            include_disabled = True
        idx += 1

    output_dir = _resolve_process_path(output_token, cwd) or (SCRIPT_DIR / "results")
    return RunnerCommandInfo(
        pid=pid,
        elapsed_seconds=elapsed_seconds,
        command=command,
        input_file=_resolve_process_path(input_token, cwd),
        output_dir=output_dir,
        filter_expr=filter_expr,
        experiment_id=experiment_id,
        dry_run=dry_run,
        include_disabled=include_disabled,
    )


def find_running_runner_commands() -> List[RunnerCommandInfo]:
    """Find active experiment_runner.py processes started from the local machine."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []

    commands: List[RunnerCommandInfo] = []
    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid_text, elapsed_text, command = parts
        if "experiment_runner.py" not in command:
            continue
        try:
            pid = int(pid_text)
            elapsed = int(elapsed_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        info = parse_runner_command_info(command, pid=pid, elapsed_seconds=elapsed)
        if info is not None:
            commands.append(info)
    return commands


def expected_runs_for_command(info: RunnerCommandInfo) -> Optional[int]:
    """Reconstruct planned repetitions for a live runner command when possible."""
    if info.input_file is None or not info.input_file.exists() or info.dry_run:
        return None
    try:
        from experiment_runner import ExperimentCSVParser, create_filter

        experiments = ExperimentCSVParser(info.input_file).parse()
        active = list(experiments) if info.include_disabled else [exp for exp in experiments if exp.enabled]
        if info.experiment_id:
            active = [exp for exp in active if exp.id == info.experiment_id]
        elif info.filter_expr:
            filter_fn = create_filter(info.filter_expr)
            if filter_fn:
                active = [exp for exp in active if filter_fn(exp)]
        return sum(exp.settings.repetitions for exp in active)
    except Exception as exc:
        logger.debug("Could not reconstruct runner task count: %s", exc)
        return None


def latest_result_file_in(output_dir: Path) -> Optional[Path]:
    """Return the newest runner CSV directly under an output directory."""
    if not output_dir.exists():
        return None
    files = [path for path in output_dir.glob("experiments_*.csv") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def summarize_result_file(csv_path: Optional[Path]) -> Dict[str, Any]:
    """Summarize a runner CSV that may still be growing."""
    empty = {
        "completed_runs": 0,
        "successful_runs": 0,
        "psbt_runs": 0,
        "failed_runs": 0,
        "last_timestamp": None,
        "last_experiment_id": None,
        "last_model": None,
    }
    if csv_path is None or not csv_path.exists():
        return empty
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError):
        return empty
    if df.empty:
        return empty

    success = int(truthy_series(df["success"]).sum()) if "success" in df.columns else 0
    psbts = int(truthy_series(df["psbt_generated"]).sum()) if "psbt_generated" in df.columns else 0
    failed = int((~truthy_series(df["success"])).sum()) if "success" in df.columns else 0

    recent = df.iloc[-1]
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().any():
            recent = df.loc[parsed.idxmax()]

    return {
        "completed_runs": len(df),
        "successful_runs": success,
        "psbt_runs": psbts,
        "failed_runs": failed,
        "last_timestamp": str(recent.get("timestamp", "")) or None,
        "last_experiment_id": str(recent.get("experiment_id", "")) or None,
        "last_model": str(recent.get("llm_model", "")) or None,
    }


def discover_live_execution_statuses() -> List[LiveExecutionStatus]:
    """Build live status rows for every active runner process."""
    statuses: List[LiveExecutionStatus] = []
    for info in find_running_runner_commands():
        result_file = latest_result_file_in(info.output_dir)
        summary = summarize_result_file(result_file)
        statuses.append(LiveExecutionStatus(
            command_info=info,
            expected_runs=expected_runs_for_command(info),
            result_file=result_file,
            completed_runs=summary["completed_runs"],
            successful_runs=summary["successful_runs"],
            psbt_runs=summary["psbt_runs"],
            failed_runs=summary["failed_runs"],
            last_timestamp=summary["last_timestamp"],
            last_experiment_id=summary["last_experiment_id"],
            last_model=summary["last_model"],
        ))
    return statuses


def load_result_files() -> List[Path]:
    """Get result CSV files from results/ and its named subdirectories."""
    if not RESULTS_DIR.exists():
        return []

    def sort_key(path: Path):
        match = _RESULT_FILE_RE.search(path.name)
        timestamp = match.group(1) if match else ""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        relative = path.relative_to(RESULTS_DIR)
        return (timestamp, mtime, str(relative))

    csv_files = [path for path in RESULTS_DIR.rglob("experiments_*.csv") if path.is_file()]
    return sorted(csv_files, key=sort_key)


def format_result_path(path: Path) -> str:
    """Return a readable path label relative to experiments/results."""
    try:
        return str(path.relative_to(RESULTS_DIR))
    except ValueError:
        return path.name


def load_latest_results(manager: ExperimentManager) -> Optional[pd.DataFrame]:
    """Load the most recent results CSV plus sibling JSON fields."""
    files = load_result_files()
    if not files:
        return None
    return load_results_dataframe(files[-1], manager=manager)


def choose_results_dataframe(
    manager: ExperimentManager,
    *,
    key_prefix: str,
    default_scope: str = "All result files",
) -> tuple[Optional[pd.DataFrame], List[Path]]:
    """Let the user choose one, many, or all result files."""
    result_files = load_result_files()
    if not result_files:
        return None, []

    scope_options = ["All result files", "Latest result file", "Choose result files"]
    scope = st.radio(
        "Result scope",
        scope_options,
        index=option_index(scope_options, default_scope),
        horizontal=True,
        key=f"{key_prefix}_result_scope",
    )

    if scope == "All result files":
        selected_files = result_files
    elif scope == "Latest result file":
        selected_files = [result_files[-1]]
    else:
        selected_files = st.multiselect(
            "Result files",
            result_files,
            default=[result_files[-1]],
            format_func=format_result_path,
            key=f"{key_prefix}_result_files",
        )

    if not selected_files:
        st.warning("No result files selected.")
        return None, []

    results_df = load_many_results_dataframes(selected_files, manager=manager)
    return results_df, selected_files


def build_comparison_summary(results_df: pd.DataFrame, by_col: str) -> pd.DataFrame:
    """Aggregate comparison rows without dropping failed/no-score attempts."""
    chart_df = results_df.copy()
    chart_df['privacy_score'] = pd.to_numeric(chart_df['privacy_score'], errors='coerce')
    chart_df['execution_time_seconds'] = pd.to_numeric(chart_df['execution_time_seconds'], errors='coerce')
    chart_df['success_rate'] = truthy_series(chart_df['success']).astype(float)

    score_grouped = chart_df.dropna(subset=['privacy_score']).groupby(by_col).agg({
        'privacy_score': ['mean', 'std', 'min', 'max', 'count'],
        'execution_time_seconds': 'mean',
    }).round(2)
    score_grouped.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in score_grouped.columns.to_flat_index()
    ]
    success_grouped = (chart_df.groupby(by_col)['success_rate'].mean() * 100).round(2).rename("success_rate_pct")
    attempts_grouped = chart_df.groupby(by_col).size().rename("attempts")
    return score_grouped.join(success_grouped, how="outer").join(attempts_grouped, how="outer")


@st.fragment(run_every="10s")
def show_live_execution_monitor():
    """Display live runner progress, including terminal-launched batches."""
    st.subheader("Live Execution")
    statuses = discover_live_execution_statuses()

    if statuses:
        st.caption("Auto-refreshes every 10 seconds while this tab is open.")
        for status in statuses:
            info = status.command_info
            label = f"PID {info.pid} running for {format_duration(info.elapsed_seconds)}"
            with st.container(border=True):
                st.write(f"**{label}**")
                if info.input_file:
                    st.caption(f"Input: `{info.input_file}`")
                st.caption(f"Output: `{info.output_dir}`")

                if status.expected_runs:
                    ratio = min(status.completed_runs / status.expected_runs, 1.0)
                    st.progress(
                        ratio,
                        text=f"{status.completed_runs}/{status.expected_runs} runs completed ({ratio * 100:.1f}%)",
                    )
                else:
                    st.progress(0.0, text=f"{status.completed_runs} runs completed")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Completed", status.completed_runs)
                col2.metric("Successful", status.successful_runs)
                col3.metric("PSBTs", status.psbt_runs)
                col4.metric("Failures", status.failed_runs)

                if status.last_experiment_id:
                    st.caption(
                        f"Last completed: `{status.last_experiment_id}`"
                        f"{f' / `{status.last_model}`' if status.last_model else ''}"
                        f"{f' at {status.last_timestamp}' if status.last_timestamp else ''}"
                    )
                if status.result_file:
                    st.caption(f"Live CSV: `{format_result_path(status.result_file)}`")
                else:
                    st.caption("Waiting for the runner to create its first result CSV.")
        return

    latest = load_result_files()[-1] if load_result_files() else None
    if latest:
        summary = summarize_result_file(latest)
        try:
            age_seconds = int(time.time() - latest.stat().st_mtime)
        except OSError:
            age_seconds = None
        st.info(
            "No active experiment_runner process detected. "
            f"Latest result file appears idle: `{format_result_path(latest)}` "
            f"with {summary['completed_runs']} rows"
            f"{f', last updated {format_duration(age_seconds)} ago' if age_seconds is not None else ''}."
        )
    else:
        st.info("No active experiment_runner process detected and no result files found yet.")


def show_paper_chart_gallery(
    results_df: Optional[pd.DataFrame] = None,
    *,
    key_prefix: str = "paper_charts",
):
    """Show paper-style charts backed by analysis/charts or the selected result file."""
    st.subheader("Paper-Style Charts")

    sources = load_paper_chart_sources()
    source_options: List[str] = []
    if results_df is not None and not results_df.empty:
        source_options.append("Selected result data")
    if not sources["aggregated"].empty:
        source_options.append("analysis/charts aggregate")

    if not source_options:
        st.info("No chart data found in the selected results or analysis/charts.")
        missing = sources.get("missing", [])
        if missing:
            with st.expander("Missing chart source files"):
                for path in missing:
                    st.code(path)
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        data_source = st.radio(
            "Chart data",
            source_options,
            horizontal=True,
            key=f"{key_prefix}_source",
        )
    with col2:
        chart_name = st.selectbox(
            "Paper chart",
            PAPER_CHART_OPTIONS,
            key=f"{key_prefix}_chart",
        )

    if data_source == "Selected result data":
        agg = aggregate_current_results(results_df)
        v2_df = current_results_v2_scores(results_df)
        st.caption("Built from the result rows currently loaded in the web UI.")
    else:
        agg = prepare_aggregated_dataframe(sources["aggregated"])
        v2_df = prepare_v2_scores_dataframe(sources["v2_scores"])
        st.caption(f"Loaded from {ANALYSIS_CHARTS_DIR}")

    chart = build_paper_chart(chart_name, agg, v2_df)
    if chart is None:
        st.info("This chart needs score, fee, timing, or sub-score data that is not available for the selected source.")
    else:
        st.altair_chart(chart, width="stretch")

    pdfs = sources.get("pdfs", [])
    if pdfs:
        st.subheader("Generated PDF Chart Preview")
        selected_pdf = st.selectbox(
            "Generated chart PDF",
            pdfs,
            format_func=lambda p: p.name,
            key=f"{key_prefix}_pdf_preview",
        )
        show_pdf_preview(selected_pdf)
    else:
        st.info(f"No generated PDF charts found under {ANALYSIS_CHARTS_DIR / 'charts'}")

    with st.expander("Generated PDF charts from analysis/charts"):
        if not pdfs:
            st.info(f"No PDF charts found under {ANALYSIS_CHARTS_DIR / 'charts'}")
        else:
            st.caption("Exact PDFs generated by the analysis/charts scripts.")
            pdf_cols = st.columns(2)
            for idx, pdf_path in enumerate(pdfs):
                with pdf_cols[idx % 2]:
                    st.download_button(
                        f"Download {pdf_path.name}",
                        data=pdf_path.read_bytes(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=f"{key_prefix}_pdf_{idx}_{pdf_path.stem}",
                    )

    with st.expander("Aggregated chart data"):
        if agg.empty:
            st.info("No aggregated rows available for this source.")
        else:
            show_dataframe(agg, width="stretch")


def main():
    """Main Streamlit app."""
    load_css()

    # Sidebar navigation
    st.sidebar.title("🔐 Bitcoin Privacy Experiments")
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🧪 Experiments Browser",
            "➕ Create Experiment",
            "✏️ Edit Experiment",
            "📋 Clone Experiment",
            "▶️ Run Experiments",
            "📈 Results",
            "🔍 Compare Results",
        ],
    )

    manager = get_manager()

    if page == "📊 Dashboard":
        show_dashboard(manager)
    elif page == "🧪 Experiments Browser":
        show_experiments_browser(manager)
    elif page == "➕ Create Experiment":
        show_create_experiment(manager)
    elif page == "✏️ Edit Experiment":
        show_edit_experiment(manager)
    elif page == "📋 Clone Experiment":
        show_clone_experiment(manager)
    elif page == "▶️ Run Experiments":
        show_run_experiments(manager)
    elif page == "📈 Results":
        show_results(manager)
    elif page == "🔍 Compare Results":
        show_compare_results(manager)


def show_dashboard(manager: ExperimentManager):
    """Show main dashboard."""
    st.title("🔐 Bitcoin Privacy Experiments Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    # Count experiments
    experiments = manager.read_experiments()
    exp_df = experiments_dataframe(manager, experiments)
    enabled_count = int(exp_df["enabled"].sum()) if not exp_df.empty else 0

    with col1:
        st.metric("Total Experiments", len(experiments))
    with col2:
        st.metric("Enabled", enabled_count)
    with col3:
        st.metric("Disabled", len(experiments) - enabled_count)

    # Result files
    result_files = load_result_files()
    with col4:
        st.metric("Result Files", len(result_files))

    st.divider()

    # Recent experiments table
    st.subheader("📋 Recent Experiments (First 10)")
    if not exp_df.empty:
        df = exp_df.head(10)
        display_cols = ['id', 'name', 'provider', 'model', 'strategy', 'amount_display', 'enabled']
        df_display = df[[c for c in display_cols if c in df.columns]]
        show_dataframe(df_display, width='stretch')

    # Recent results
    st.subheader("📊 Latest Results")
    results_df = load_latest_results(manager)
    if results_df is not None:
        st.write(f"**Loaded from**: {load_result_files()[-1].name if load_result_files() else 'N/A'}")
        st.caption(f"Path: {format_result_path(load_result_files()[-1])}" if load_result_files() else "")

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            success_rate = (truthy_series(results_df['success']).mean() * 100) if len(results_df) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col2:
            avg_score = pd.to_numeric(results_df['privacy_score'], errors='coerce').mean()
            st.metric("Avg Privacy Score", f"{avg_score:.1f}" if pd.notna(avg_score) else "N/A")
        with col3:
            psbt_gen_rate = (truthy_series(results_df['psbt_generated']).mean() * 100) if len(results_df) > 0 else 0
            st.metric("PSBT Generation Rate", f"{psbt_gen_rate:.1f}%")

        # Recent results table
        recent = results_df.tail(10)
        show_dataframe(recent[display_columns(recent.columns)], width='stretch')
    else:
        st.info("No results yet. Run some experiments to see results here.")

    st.divider()
    st.caption("💡 Tip: Use 'Run Experiments' to execute tests, then view results here.")


def show_experiments_browser(manager: ExperimentManager):
    """Browse and filter experiments."""
    st.title("🧪 Experiments Browser")

    experiments = manager.read_experiments()

    if not experiments:
        st.warning("No experiments found. Create one to get started!")
        return
    exp_df = experiments_dataframe(manager, experiments)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        providers = sorted(exp_df['provider'].dropna().unique())
        selected_provider = st.multiselect("Provider", providers, default=[])

    with col2:
        models = sorted(exp_df['model'].dropna().unique())
        selected_model = st.multiselect("Model", models, default=[])

    with col3:
        strategies = sorted(exp_df['strategy'].dropna().unique())
        selected_strategy = st.multiselect("Strategy", strategies, default=[])

    # Filter experiments
    filtered_df = exp_df.copy()
    if selected_provider:
        filtered_df = filtered_df[filtered_df['provider'].isin(selected_provider)]
    if selected_model:
        filtered_df = filtered_df[filtered_df['model'].isin(selected_model)]
    if selected_strategy:
        filtered_df = filtered_df[filtered_df['strategy'].isin(selected_strategy)]

    st.subheader(f"Found {len(filtered_df)} experiments")

    if not filtered_df.empty:
        df = filtered_df
        display_cols = [
            'id', 'name', 'provider', 'model', 'strategy', 'amount_display',
            'prompt_mode', 'temperature', 'repetitions', 'enabled'
        ]
        df_display = df[[c for c in display_cols if c in df.columns]]
        show_dataframe(df_display, width='stretch')

        # Show details of selected experiment
        selected_id = st.selectbox("Select experiment for details", filtered_df['id'].tolist())
        if selected_id:
            exp = manager.read_experiment_by_id(selected_id)
            if exp:
                meta = manager.parse_csv_row_to_meta(exp)
                prompt_mode = safe_prompt_mode(manager, exp, meta)
                st.subheader(f"Details: {selected_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name**: {meta.name}")
                    st.write(f"**Provider**: {meta.provider}")
                    st.write(f"**Model**: {meta.model}")
                    st.write(f"**Strategy**: {meta.strategy}")
                    st.write(f"**Amount target**: {manager.infer_amount_display(exp)}")

                with col2:
                    st.write(f"**Prompt mode**: {prompt_mode}")
                    st.write(f"**Temperature**: {meta.temperature}")
                    st.write(f"**Repetitions**: {meta.repetitions}")
                    st.write(f"**Timeout (s)**: {meta.timeout_seconds}")
                    st.write(f"**Network**: {meta.network}")
                    st.write(f"**Enabled**: {meta.enabled}")

                st.subheader("Prompts")
                st.text(f"User Prompt:\n{meta.user_prompt or 'N/A'}")

                if meta.followup_prompts:
                    for i, fp in enumerate(meta.followup_prompts, 1):
                        st.text(f"Followup {i}:\n{fp}")

                st.subheader("Tags")
                if meta.tags:
                    for tag in meta.tags:
                        st.write(f"  • {tag}")


def show_create_experiment(manager: ExperimentManager):
    """Create a new experiment with interactive form."""
    st.title("➕ Create New Experiment")

    with st.form("create_exp_form"):
        col1, col2 = st.columns(2)

        with col1:
            exp_id = st.text_input("Experiment ID (unique)", help="e.g., exp_claude_privacy_v2")
            name = st.text_input("Name", help="e.g., Claude 3.5 Privacy Test")
            description = st.text_area("Description", height=80)

        with col2:
            st.subheader("Parameters")
            prompt_mode = st.radio("Prompt mode", PROMPT_MODE_OPTIONS, horizontal=True)
            amount_btc = st.number_input("Amount (BTC)", min_value=0.0001, value=3.0, step=0.1)
            strategy = st.selectbox("Strategy", [s.value for s in PromptStrategy])
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            provider = st.selectbox("Provider", ["openai", "anthropic", "google", "openrouter"])
            model = st.text_input("Model", value="gpt-4o", help="Full model name (e.g., gpt-4o, claude-opus-4)")

        with col2:
            repetitions = st.number_input("Repetitions", min_value=1, max_value=10, value=3)
            timeout_seconds = st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=300)

        with col3:
            network = st.selectbox("Network", ["mainnet", "testnet"])
            priority = st.number_input("Priority", min_value=1, max_value=10, value=1)

        tags_input = st.text_input("Tags (comma-separated)", help="e.g., privacy, openai, baseline")
        tags = [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else []

        enabled = st.checkbox("Enabled", value=True)

        if prompt_mode == "template":
            prompts = generate_prompts(amount_btc, strategy, "ca")
            st.subheader("Prompt Preview")
            st.text(f"User Prompt:\n{prompts['user_prompt']}")
            for i, fp in enumerate(prompts['followup_prompts'], 1):
                st.text(f"Followup {i}:\n{fp}")
            custom_user_prompt = ""
            custom_followups_input = ""
        else:
            st.subheader("Custom Prompt")
            custom_user_prompt = st.text_area("User Prompt", height=120)
            custom_followups_input = st.text_area(
                "Followups (one per line)",
                help="The UI will store these as pipe-separated followup_prompts in the CSV.",
                height=100,
            )

        submitted = st.form_submit_button("✓ Create Experiment")

    if submitted:
        if not exp_id:
            st.error("Experiment ID is required")
            return
        if prompt_mode == "custom" and not custom_user_prompt.strip():
            st.error("Custom prompt mode requires a user prompt")
            return

        try:
            exp_meta = ExperimentMeta(
                id=exp_id,
                name=name or exp_id,
                description=description,
                amount_btc=amount_btc,
                strategy=strategy,
                provider=provider,
                model=model,
                temperature=temperature,
                repetitions=repetitions,
                timeout_seconds=timeout_seconds,
                network=network,
                tags=tags,
                enabled=enabled,
                priority=priority,
                prompt_mode=prompt_mode,
                user_prompt=custom_user_prompt,
                followup_prompts=[line.strip() for line in custom_followups_input.splitlines() if line.strip()],
            )

            manager.add_experiment(exp_meta)
            st.success(f"✓ Experiment '{exp_id}' created successfully!")

            st.caption(f"Prompt mode: {prompt_mode}")

        except Exception as e:
            st.error(f"Error: {e}")


def show_edit_experiment(manager: ExperimentManager):
    """Edit an existing experiment."""
    st.title("✏️ Edit Experiment")

    experiments = manager.read_experiments()
    if not experiments:
        st.warning("No experiments to edit")
        return

    exp_id = st.selectbox("Select experiment to edit", [e['id'] for e in experiments])

    exp = manager.read_experiment_by_id(exp_id)
    if not exp:
        st.error("Experiment not found")
        return

    # Parse into editable form
    meta = manager.parse_csv_row_to_meta(exp)
    current_prompt_mode = safe_prompt_mode(manager, exp, meta)

    with st.form("edit_exp_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Name", value=meta.name)
            description = st.text_area("Description", value=meta.description, height=80)

        with col2:
            st.subheader("Parameters")
            prompt_mode = st.radio(
                "Prompt mode",
                PROMPT_MODE_OPTIONS,
                index=option_index(PROMPT_MODE_OPTIONS, current_prompt_mode),
                horizontal=True,
            )
            amount_btc = st.number_input("Amount (BTC)", min_value=0.0001, value=meta.amount_btc, step=0.1)
            strategy = st.selectbox("Strategy", [s.value for s in PromptStrategy], index=[s.value for s in PromptStrategy].index(meta.strategy))
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=meta.temperature, step=0.1)

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            provider = st.selectbox("Provider", PROVIDER_OPTIONS, index=option_index(PROVIDER_OPTIONS, meta.provider))
            model = st.text_input("Model", value=meta.model)

        with col2:
            repetitions = st.number_input("Repetitions", min_value=1, max_value=10, value=meta.repetitions)
            timeout_seconds = st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=meta.timeout_seconds)

        with col3:
            network = st.selectbox("Network", NETWORK_OPTIONS, index=option_index(NETWORK_OPTIONS, meta.network))
            priority = st.number_input("Priority", min_value=1, max_value=10, value=meta.priority)

        tags_input = st.text_input("Tags (comma-separated)", value=", ".join(meta.tags) if meta.tags else "")
        tags = [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else []

        enabled = st.checkbox("Enabled", value=meta.enabled)

        if prompt_mode == "template":
            prompts = generate_prompts(amount_btc, strategy, "ca")
            st.subheader("Prompt Preview")
            st.text(f"User Prompt:\n{prompts['user_prompt']}")
            for i, fp in enumerate(prompts['followup_prompts'], 1):
                st.text(f"Followup {i}:\n{fp}")
            template_fields_changed = (
                prompt_mode != current_prompt_mode
                or amount_btc != meta.amount_btc
                or strategy != meta.strategy
            )
            regenerate_prompts = st.checkbox(
                "Regenerate prompts from template on save",
                value=template_fields_changed,
                help="Amount/strategy changes in Template mode regenerate automatically; this also lets you refresh unchanged template prompts.",
            )
            custom_user_prompt = meta.user_prompt
            custom_followups_input = "\n".join(meta.followup_prompts)
        else:
            st.subheader("Custom Prompt")
            custom_user_prompt = st.text_area("User Prompt", value=meta.user_prompt, height=120)
            custom_followups_input = st.text_area(
                "Followups (one per line)",
                value="\n".join(meta.followup_prompts),
                height=100,
            )

        submitted = st.form_submit_button("✓ Update Experiment")

    if submitted:
        try:
            updates = {
                'name': name,
                'description': description,
                'provider': provider,
                'model': model,
                'temperature': str(temperature),
                'repetitions': str(repetitions),
                'timeout_seconds': str(timeout_seconds),
                'network': network,
                'priority': str(priority),
                'tags': tags,
                'enabled': enabled,
                'prompt_mode': prompt_mode,
                '_regenerate_prompts': regenerate_prompts if prompt_mode == 'template' else False,
            }
            if prompt_mode == 'template':
                updates['amount_btc'] = str(amount_btc)
                updates['strategy'] = strategy
            else:
                updates['amount_btc'] = ''
                updates['strategy'] = ''
            if prompt_mode == 'custom':
                if not custom_user_prompt.strip():
                    st.error("Custom prompt mode requires a user prompt")
                    return
                updates['user_prompt'] = custom_user_prompt
                updates['followup_prompts'] = [line.strip() for line in custom_followups_input.splitlines() if line.strip()]

            manager.update_experiment(exp_id, updates)
            st.success(f"✓ Experiment '{exp_id}' updated successfully!")

        except Exception as e:
            st.error(f"Error: {e}")


def show_clone_experiment(manager: ExperimentManager):
    """Clone an existing experiment."""
    st.title("📋 Clone Experiment")

    experiments = manager.read_experiments()
    if not experiments:
        st.warning("No experiments to clone")
        return

    src_id = st.selectbox("Select source experiment", [e['id'] for e in experiments])

    src_exp = manager.read_experiment_by_id(src_id)
    if not src_exp:
        st.error("Source experiment not found")
        return
    src_meta = manager.parse_csv_row_to_meta(src_exp)

    st.subheader("Clone Settings")

    col1, col2 = st.columns(2)

    with col1:
        new_id = st.text_input("New Experiment ID", value=f"{src_id}_v2", help="Must be unique")
        modify_model = st.checkbox("Modify Model")

    with col2:
        modify_amount = st.checkbox("Modify Amount")
        modify_strategy = st.checkbox("Modify Strategy")

    new_model = None
    new_amount = None
    new_strategy = None
    clone_prompt_mode = st.radio(
        "Prompt handling",
        ["keep current prompt text", "use template if amount/strategy changes"],
        index=0,
    )

    if modify_model:
        new_model = st.text_input("New Model", value=src_meta.model)

    if modify_amount:
        new_amount = st.number_input("New Amount (BTC)", min_value=0.0001, value=src_meta.amount_btc, step=0.1)

    if modify_strategy:
        new_strategy = st.selectbox("New Strategy", [s.value for s in PromptStrategy], index=[s.value for s in PromptStrategy].index(src_meta.strategy))
    if (modify_amount or modify_strategy) and clone_prompt_mode.startswith("keep current"):
        st.caption("Amount/strategy changes only affect the prompt when template regeneration is selected.")

    if st.button("✓ Clone Experiment"):
        try:
            updates = {}
            if modify_model and new_model:
                updates['model'] = new_model
            if (modify_amount or modify_strategy) and clone_prompt_mode.startswith("use template"):
                updates['prompt_mode'] = 'template'
                if modify_amount and new_amount:
                    updates['amount_btc'] = str(new_amount)
                if modify_strategy and new_strategy:
                    updates['strategy'] = new_strategy

            manager.clone_experiment(src_id, new_id, updates)
            st.success(f"✓ Experiment cloned to '{new_id}'!")

        except Exception as e:
            st.error(f"Error: {e}")


def show_run_experiments(manager: ExperimentManager):
    """Run experiments with various options."""
    st.title("▶️ Run Experiments")

    experiments = manager.read_experiments()
    if not experiments:
        st.warning("No experiments to run")
        return

    exp_df = experiments_dataframe(manager, experiments)
    if exp_df.empty:
        st.warning("No experiments to run")
        return

    st.subheader("Build Run Selection")

    col1, col2, col3 = st.columns(3)
    with col1:
        enabled_scope = st.radio(
            "Enabled filter",
            ["Enabled only", "All experiments", "Disabled only"],
            horizontal=True,
        )
        selected_providers = st.multiselect("Provider", sorted(exp_df["provider"].dropna().unique()), default=[])
        selected_strategies = st.multiselect("Strategy", sorted(exp_df["strategy"].dropna().unique()), default=[])
    with col2:
        selected_models = st.multiselect("Model", sorted(exp_df["model"].dropna().unique()), default=[])
        selected_networks = st.multiselect("Network", sorted(exp_df["network"].dropna().unique()), default=[])
        selected_prompt_modes = st.multiselect(
            "Prompt mode",
            sorted(exp_df["prompt_mode"].dropna().unique()),
            default=[],
        )
    with col3:
        search = st.text_input("Search ID / name / model / tags")
        selected_tags = st.multiselect("Tags", all_tags_from_dataframe(exp_df), default=[])
        tag_match = st.radio(
            "Tag matching",
            ["Any selected tag", "All selected tags"],
            horizontal=True,
            disabled=not selected_tags,
        )

    amount_range = None
    amount_values = pd.to_numeric(exp_df["amount_btc"], errors="coerce").dropna()
    amount_labels = exp_df.get("amount_display", pd.Series(dtype="object")).dropna().astype(str)
    relative_targets = amount_labels.str.contains("%", regex=False).any()
    if not amount_values.empty and amount_values.min() < amount_values.max() and not relative_targets:
        amount_range = st.slider(
            "Amount range (BTC)",
            min_value=float(amount_values.min()),
            max_value=float(amount_values.max()),
            value=(float(amount_values.min()), float(amount_values.max())),
        )
    elif relative_targets:
        unique_targets = ", ".join(dict.fromkeys(amount_labels.tolist()))
        st.caption(f"Amount filter not shown: current rows use prompt-defined percentage targets ({unique_targets}).")
    elif not amount_values.empty:
        st.caption(f"Amount filter not shown: all visible experiments infer {amount_values.iloc[0]:g} BTC.")

    filtered_df = filter_run_experiments(
        exp_df,
        enabled_scope=enabled_scope,
        providers=selected_providers,
        models=selected_models,
        strategies=selected_strategies,
        networks=selected_networks,
        prompt_modes=selected_prompt_modes,
        tags=selected_tags,
        tag_match=tag_match,
        search=search,
        amount_range=amount_range,
    )

    st.subheader(f"Matching Experiments: {len(filtered_df)}")
    preview_cols = [
        "id", "name", "provider", "model", "strategy", "amount_display",
        "prompt_mode", "network", "enabled", "tags",
    ]
    show_dataframe(filtered_df[[c for c in preview_cols if c in filtered_df.columns]], width='stretch')

    st.subheader("Choose Final IDs")
    selection_mode = st.radio(
        "Selection mode",
        ["Pick from matching rows", "Run all matching rows", "Paste IDs"],
        horizontal=True,
    )

    selected_ids: List[str] = []
    filtered_ids = filtered_df["id"].tolist() if "id" in filtered_df.columns else []
    id_labels = {
        row["id"]: f"{row['id']} — {row['provider']} / {row['model']} / {row['strategy']}"
        for _, row in filtered_df.iterrows()
    }

    if selection_mode == "Run all matching rows":
        selected_ids = filtered_ids
    elif selection_mode == "Paste IDs":
        pasted_ids = split_filter_values(st.text_area("IDs to run", help="Comma, whitespace, newline or pipe separated."))
        existing_ids = set(exp_df["id"].tolist())
        unknown_ids = [exp_id for exp_id in pasted_ids if exp_id not in existing_ids]
        selected_ids = [exp_id for exp_id in pasted_ids if exp_id in existing_ids]
        if unknown_ids:
            st.warning(f"Unknown IDs ignored: {', '.join(unknown_ids)}")
    else:
        selected_ids = st.multiselect(
            "Specific matching experiments",
            filtered_ids,
            default=[],
            format_func=lambda exp_id: id_labels.get(exp_id, exp_id),
        )

    limit_enabled = st.checkbox("Limit run count", value=False)
    if limit_enabled and selected_ids:
        limit_count = st.number_input("Max experiments to run", min_value=1, max_value=len(selected_ids), value=min(5, len(selected_ids)))
        selected_ids = selected_ids[:int(limit_count)]

    st.info(f"Selected {len(selected_ids)} experiments for this run.")
    if selected_ids:
        selected_preview = exp_df[exp_df["id"].isin(selected_ids)]
        show_dataframe(selected_preview[[c for c in preview_cols if c in selected_preview.columns]], width='stretch')

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        parallel_label = st.selectbox(
            "Execution profile",
            ["Sequential", "Provider lanes", "Model lanes"],
            help="Provider/model lanes run experiments concurrently with CLI safety limits.",
        )
        parallel_profile = {
            "Sequential": "sequential",
            "Provider lanes": "provider",
            "Model lanes": "model",
        }[parallel_label]

    with col2:
        max_concurrency = st.number_input(
            "Max concurrency",
            min_value=1,
            value=1 if parallel_profile == "sequential" else 5,
            step=1,
        )

    with col3:
        provider_limits = st.text_input(
            "Provider limits",
            value="openai=1,anthropic=1,openrouter=3" if parallel_profile == "model" else "",
            help="Comma-separated provider=N limits. OpenRouter may still be throttled by upstream providers.",
        )

    with col4:
        model_limit = st.number_input("Model lane limit", min_value=1, value=1, step=1)

    col5, col6, col7 = st.columns(3)
    with col5:
        interleave = st.checkbox("Interleave by provider", help="Round-robin ordering; useful for sequential runs and readable logs")

    with col6:
        delay = st.number_input("Delay between starts (sec)", min_value=0, value=0, step=1)

    with col7:
        dry_run = st.checkbox("Dry run", help="Parse without executing")

    if parallel_profile != "sequential":
        st.caption("Parallel runs are local batch execution only: the agent remains xpub-only and does not sign or broadcast.")

    if st.button("▶️ Start Running", type="primary"):
        if not selected_ids:
            st.error("No experiments selected")
            return

        st.subheader("Execution Progress")

        selected_experiments = [e for e in experiments if e.get('id') in selected_ids]
        selected_meta = [manager.parse_csv_row_to_meta(e) for e in selected_experiments]
        include_disabled = any(not manager.parse_csv_row_to_meta(e).enabled for e in selected_experiments)
        if include_disabled:
            st.warning("This selection includes disabled rows; the runner will use --include-disabled for this batch.")
        estimated_timeout = estimate_runner_timeout_seconds(
            selected_meta,
            parallel_profile=parallel_profile,
            max_concurrency=int(max_concurrency),
            model_limit=int(model_limit),
            delay=delay,
        )
        if dry_run:
            estimated_timeout = 120

        cmd = build_runner_command(
            selected_ids,
            interleave=interleave,
            delay=delay,
            dry_run=dry_run,
            include_disabled=include_disabled,
            parallel_profile=parallel_profile,
            max_concurrency=int(max_concurrency),
            provider_limits=provider_limits,
            model_limit=int(model_limit),
        )
        st.code(" ".join(cmd), language="bash")

        with st.spinner("Running dry-run..." if dry_run else "Running selected experiments..."):
            try:
                result = run_runner_command(cmd, timeout_seconds=estimated_timeout)
            except subprocess.TimeoutExpired as exc:
                st.error(f"Runner timed out after {estimated_timeout}s")
                with st.expander("Partial stdout/stderr", expanded=True):
                    st.text(exc.stdout or "")
                    st.text(exc.stderr or "")
                return

        if result.returncode == 0:
            st.success("✓ Runner completed successfully" if dry_run else "✓ Selected experiments completed")
        else:
            st.error(f"✗ Runner failed with exit code {result.returncode}")

        with st.expander("Runner stdout", expanded=not dry_run):
            st.code(result.stdout or "(empty)")
        if result.stderr:
            with st.expander("Runner stderr", expanded=True):
                st.code(result.stderr)


def show_results(manager: ExperimentManager):
    """Visualize results."""
    st.title("📈 Experiment Results")

    show_live_execution_monitor()
    st.divider()

    result_files = load_result_files()
    if not result_files:
        st.info("No results yet. Run experiments to generate results.")
        return

    results_df, selected_files = choose_results_dataframe(
        manager,
        key_prefix="results",
        default_scope="Latest result file",
    )
    if results_df is None or results_df.empty:
        st.info("No rows found in the selected result files.")
        return

    results_df = sort_results_for_display(results_df)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Loaded files**: {len(selected_files)}")
        st.caption(f"{len(results_df)} result rows loaded.")
        with st.expander("Loaded result files"):
            for path in selected_files:
                st.code(format_result_path(path))
    with col2:
        if st.button("🔄 Reload"):
            st.rerun()

    st.caption("The latest result file is updated after each completed run. Use Reload to pick up new rows from an in-progress batch.")

    # Summary stats
    st.subheader("Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)

    total = len(results_df)
    with col1:
        st.metric("Total Runs", total)

    successful = int(truthy_series(results_df['success']).sum()) if 'success' in results_df.columns else 0
    with col2:
        st.metric("Successful", f"{successful} ({100*successful/total:.1f}%)" if total > 0 else "N/A")

    with col3:
        psbt_gen = int(truthy_series(results_df['psbt_generated']).sum()) if 'psbt_generated' in results_df.columns else 0
        st.metric("PSBTs Generated", f"{psbt_gen} ({100*psbt_gen/total:.1f}%)" if total > 0 else "N/A")

    avg_score = pd.to_numeric(results_df['privacy_score'], errors='coerce').mean() if 'privacy_score' in results_df.columns else None
    with col4:
        st.metric("Avg Privacy Score", f"{avg_score:.1f}" if pd.notna(avg_score) else "N/A")

    avg_time = pd.to_numeric(results_df['execution_time_seconds'], errors='coerce').mean() if 'execution_time_seconds' in results_df.columns else None
    with col5:
        st.metric("Avg Execution Time", f"{avg_time:.1f}s" if pd.notna(avg_time) else "N/A")

    st.divider()

    # Detailed results table
    st.subheader("Detailed Results")

    # Apply filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        providers = sorted(results_df['llm_provider'].dropna().astype(str).unique()) if 'llm_provider' in results_df.columns else []
        filter_provider = st.multiselect("Provider", providers, default=[])

    with col2:
        models = sorted(results_df['llm_model'].dropna().astype(str).unique()) if 'llm_model' in results_df.columns else []
        filter_model = st.multiselect("Model", models, default=[])

    with col3:
        success_filter = st.selectbox("Status", ["All", "Success Only", "Failures Only"])

    with col4:
        sanity_values = sorted(v for v in results_df.get('sanity_status', pd.Series(dtype=str)).dropna().unique() if str(v))
        sanity_filter = st.multiselect("Sanity", sanity_values, default=[])

    # Filter
    filtered_df = results_df
    if filter_provider:
        filtered_df = filtered_df[filtered_df['llm_provider'].isin(filter_provider)]
    if filter_model:
        filtered_df = filtered_df[filtered_df['llm_model'].isin(filter_model)]

    if success_filter == "Success Only":
        filtered_df = filtered_df[truthy_series(filtered_df['success'])]
    elif success_filter == "Failures Only":
        filtered_df = filtered_df[~truthy_series(filtered_df['success'])]
    if sanity_filter and 'sanity_status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sanity_status'].isin(sanity_filter)]

    filtered_df = sort_results_for_display(filtered_df)
    show_dataframe(filtered_df[display_columns(filtered_df.columns)], width='stretch')

    # Charts
    st.subheader("Charts")

    col1, col2 = st.columns(2)

    with col1:
        if 'privacy_score' in results_df.columns and 'llm_model' in results_df.columns:
            st.subheader("Average Privacy Score by Model")
            chart_df = results_df.copy()
            chart_df['privacy_score'] = pd.to_numeric(chart_df['privacy_score'], errors='coerce')
            score_by_model = chart_df.dropna(subset=['privacy_score']).groupby('llm_model')['privacy_score'].mean().sort_values(ascending=False)
            st.bar_chart(score_by_model)

    with col2:
        if 'privacy_grade' in results_df.columns:
            st.subheader("Privacy Grade Distribution")
            grade_counts = results_df['privacy_grade'].value_counts()
            st.bar_chart(grade_counts)

    st.divider()
    show_paper_chart_gallery(results_df, key_prefix="results_paper")


def show_compare_results(manager: ExperimentManager):
    """Compare results across experiments."""
    st.title("🔍 Compare Results")

    result_files = load_result_files()
    if not result_files:
        st.info("No results yet.")
        return

    results_df, selected_files = choose_results_dataframe(
        manager,
        key_prefix="compare",
        default_scope="All result files",
    )
    if results_df is None or results_df.empty:
        st.info("No rows found in the selected result files.")
        return

    st.caption(f"Comparing {len(results_df)} rows from {len(selected_files)} result file(s).")

    comparison_type = st.selectbox(
        "Compare by",
        ["Model", "Strategy", "Provider", "Amount Target", "Amount (BTC)"]
    )

    if comparison_type == "Model" and 'llm_model' in results_df.columns:
        by_col = 'llm_model'
    elif comparison_type == "Strategy" and 'strategy' in results_df.columns:
        by_col = 'strategy'
    elif comparison_type == "Provider" and 'llm_provider' in results_df.columns:
        by_col = 'llm_provider'
    elif comparison_type == "Amount Target" and 'amount_display' in results_df.columns:
        by_col = 'amount_display'
    elif comparison_type == "Amount (BTC)" and 'amount_btc' in results_df.columns:
        by_col = 'amount_btc'
    else:
        st.warning("Comparison not available for this dimension")
        return
    if results_df[by_col].dropna().empty:
        st.warning("Comparison not available because this dimension has no concrete values in the selected results.")
        return

    # Group and aggregate
    chart_df = results_df.copy()
    chart_df['privacy_score'] = pd.to_numeric(chart_df['privacy_score'], errors='coerce')
    chart_df['execution_time_seconds'] = pd.to_numeric(chart_df['execution_time_seconds'], errors='coerce')
    chart_df['success_rate'] = truthy_series(chart_df['success']).astype(float)
    grouped = build_comparison_summary(results_df, by_col)

    st.subheader(f"Comparison by {comparison_type}")
    show_dataframe(grouped, width='stretch')

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Average Privacy Score by {comparison_type}")
        by_stats = chart_df.dropna(subset=['privacy_score']).groupby(by_col)['privacy_score'].mean().sort_values(ascending=False)
        st.bar_chart(by_stats)

    with col2:
        st.subheader(f"Success Rate (%) by {comparison_type}")
        by_success = chart_df.groupby(by_col)['success_rate'].mean() * 100
        st.bar_chart(by_success)

    st.divider()
    show_paper_chart_gallery(results_df, key_prefix="compare_paper")


if __name__ == "__main__":
    main()
