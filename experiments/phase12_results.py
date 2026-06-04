#!/usr/bin/env python3
"""Shared 2026 prompt-corpus result normalization and cost helpers.

The current 2026 corpus is stored in two execution lots tagged as phase-1 and
phase-2. Those tags are preserved for traceability, but the substantive analysis
is prompt-first: basic, privacy-simple, and multi-turn-detailed are prompt
strategies run through the same agent, wallet, runner, and scorer. Rerun result
files are treated as an overlay over the primary execution lots so downstream
paper tables and figures read one canonical corpus.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent

DEFAULT_PHASE12_PRIMARY_RESULT_FILES = [
    SCRIPT_DIR / "results" / "phase1" / "experiments_20260422_114240.csv",
    SCRIPT_DIR / "results" / "phase2_all_phase1_models_20260427_rerun" / "experiments_20260427_093606.csv",
]

DEFAULT_PHASE12_RERUN_RESULT_FILES = [
    SCRIPT_DIR / "results" / "timeout_rerun_20260512" / "experiments_20260512_093603.csv",
    SCRIPT_DIR / "results" / "timeout_rerun_20260518_600" / "experiments_20260518_083522.csv",
    SCRIPT_DIR / "results" / "timeout_rerun_20260518_1200" / "experiments_20260518_091832.csv",
    SCRIPT_DIR / "results" / "timeout_rerun_20260518_missing_gemma_1200" / "experiments_20260518_120308.csv",
    SCRIPT_DIR / "results" / "timeout_rerun_20260518_final_gemma_1200" / "experiments_20260518_125005.csv",
]

# Backwards-compatible name for callers that explicitly need the original lots.
DEFAULT_PHASE12_RESULT_FILES = DEFAULT_PHASE12_PRIMARY_RESULT_FILES

DEFAULT_COST_TABLE = SCRIPT_DIR / "model_costs_phase12.csv"
PRICE_VERSION = "phase12-local-estimates-2026-04"

PROMPT_LABELS = {
    "basic": "Basic",
    "privacy_simple": "Privacy Simple",
    "multiturn_detailed": "Multi-turn Detailed",
}

PHASE_LABELS = {
    "phase-1": "Phase 1",
    "phase-2": "Phase 2",
}

EXECUTION_LOT_LABELS = {
    "phase-1": "Lot: basic/privacy-simple",
    "phase-2": "Lot: multi-turn-detailed",
}

AMOUNT_ORDER = [10, 30, 50, 80, 95]
TEMP_ORDER = [0.3, 1.0]

# Transparent fallback profiles for result files that predate token capture.
# They are not intended as billing truth; they make model-price tradeoffs visible
# until provider dashboard exports or response metadata are available.
ESTIMATED_TOKEN_PROFILES = {
    "basic": {"input_tokens": 10_500, "output_tokens": 1_300},
    "privacy_simple": {"input_tokens": 11_500, "output_tokens": 1_500},
    "multiturn_detailed": {"input_tokens": 24_000, "output_tokens": 3_000},
    "unknown": {"input_tokens": 12_000, "output_tokens": 1_600},
}

TRUE_VALUES = {"true", "1", "1.0", "yes", "y"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n"}


def split_tags(value: Any) -> List[str]:
    """Return normalized tag tokens from list or separator-delimited strings."""
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").replace("|", ";").replace(",", ";")
        raw_items = text.split(";")
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _tag_set(value: Any) -> set[str]:
    return {tag.lower() for tag in split_tags(value)}


def _first_tag(tags: Iterable[str], prefix: str) -> Optional[str]:
    for tag in tags:
        if tag.startswith(prefix):
            return tag
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        result = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _to_int(value: Any) -> Optional[int]:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def infer_phase(tags: Any, experiment_id: Any = "", experiment_name: Any = "") -> str:
    tag_values = _tag_set(tags)
    if "phase-2" in tag_values:
        return "phase-2"
    if "phase-1" in tag_values:
        return "phase-1"
    text = f"{experiment_id} {experiment_name}".lower()
    if "phase 2" in text or "phase-2" in text:
        return "phase-2"
    if "phase 1" in text or "phase-1" in text:
        return "phase-1"
    return "unknown"


def infer_amount_pct(tags: Any, prompt_text: Any = "") -> Optional[float]:
    for tag in _tag_set(tags):
        match = re.fullmatch(r"amt-pct-([0-9]+(?:\.[0-9]+)?)", tag)
        if match:
            return float(match.group(1))
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*%", str(prompt_text or ""))
    if match:
        return float(match.group(1))
    return None


def infer_prompt_type(tags: Any, prompt_text: Any = "") -> str:
    tag_values = _tag_set(tags)
    prompt_tag = _first_tag(tag_values, "prompt-")
    if prompt_tag:
        return prompt_tag.removeprefix("prompt-").replace("-", "_")

    text = str(prompt_text or "").lower()
    if "no et limitis" in text or "decoy" in text or "ofuscar" in text:
        return "multiturn_detailed"
    if "privada possible" in text or "mes privada" in text:
        return "privacy_simple"
    return "basic"


def infer_temperature(tags: Any, value: Any = None) -> Optional[float]:
    numeric = _to_float(value)
    if numeric is not None:
        return numeric
    for tag in _tag_set(tags):
        match = re.fullmatch(r"temp-([0-9]+(?:\.[0-9]+)?)", tag)
        if match:
            return float(match.group(1))
    return None


def infer_category(provider: Any, model: Any, tags: Any) -> str:
    tag_values = {tag.replace("-", "_") for tag in _tag_set(tags)}
    model_text = str(model or "").lower()
    provider_text = str(provider or "").lower()

    if tag_values & {"open_weight", "open_source", "opensource"}:
        return "open-source"
    if any(marker in model_text for marker in ("gemma", "glm", "llama", "mistral", "qwen", "deepseek")):
        return "open-source"
    if tag_values & {"frontier", "closed_source", "closed"}:
        return "closed-source"
    if provider_text == "openrouter":
        return "open-source"
    if provider_text:
        return "closed-source"
    return "unknown"


def _json_result_index(json_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return {}
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("experiment_id", "")), str(item.get("repetition", "")))
        result[key] = item
    return result


def flatten_privacy_breakdown(breakdown: Any) -> Dict[str, Any]:
    """Flatten scorer JSON into columns used by the 2026 prompt analysis."""
    if not isinstance(breakdown, dict):
        return {
            "score_overall": None,
            "score_clustering": None,
            "score_change_detection": None,
            "score_fingerprinting": None,
            "score_metadata_leakage": None,
            "confidence": "",
            "confidence_numeric": None,
            "fee_sanity_ok": None,
            "sanity_status": "",
            "fee_rate_sat_vb": None,
            "fee_sats": None,
            "num_inputs": None,
            "num_outputs": None,
            "total_input_sats": None,
            "total_output_sats": None,
            "equal_output_pattern": "",
            "max_equal_outputs": None,
            "change_output_index": None,
            "change_probability": None,
        }

    scores = breakdown.get("scores") or {}
    fee_analysis = breakdown.get("fee_analysis") or {}
    metadata = breakdown.get("metadata") or {}
    equal_outputs = breakdown.get("equal_output_classification") or {}
    change_guess = breakdown.get("change_guess") or {}

    return {
        "score_overall": scores.get("overall"),
        "score_clustering": scores.get("clustering"),
        "score_change_detection": scores.get("change_detection"),
        "score_fingerprinting": scores.get("fingerprinting"),
        "score_metadata_leakage": scores.get("metadata_leakage"),
        "confidence": breakdown.get("confidence", ""),
        "confidence_numeric": breakdown.get("confidence_numeric"),
        "fee_sanity_ok": breakdown.get("fee_sanity_ok"),
        "sanity_status": breakdown.get("sanity_status", ""),
        "fee_rate_sat_vb": fee_analysis.get("fee_rate_sat_vb"),
        "fee_sats": fee_analysis.get("fee_sats"),
        "num_inputs": metadata.get("num_inputs"),
        "num_outputs": metadata.get("num_outputs"),
        "total_input_sats": metadata.get("total_input_sats"),
        "total_output_sats": metadata.get("total_output_sats"),
        "equal_output_pattern": equal_outputs.get("pattern", ""),
        "max_equal_outputs": equal_outputs.get("max_equal_outputs"),
        "change_output_index": change_guess.get("output_index"),
        "change_probability": change_guess.get("probability"),
    }


def load_cost_table(cost_table_path: Path = DEFAULT_COST_TABLE) -> pd.DataFrame:
    """Load the versioned local model-cost table."""
    path = Path(cost_table_path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "price_version",
                "provider",
                "model",
                "category",
                "input_price_per_1m_tokens",
                "output_price_per_1m_tokens",
                "pricing_note",
            ]
        )
    df = pd.read_csv(path)
    for column in ("provider", "model"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.lower()
    for column in ("input_price_per_1m_tokens", "output_price_per_1m_tokens"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _price_row_for(
    model: Any,
    provider: Any = "",
    cost_table: Optional[pd.DataFrame] = None,
) -> Optional[pd.Series]:
    table = cost_table if cost_table is not None else load_cost_table()
    if table.empty:
        return None
    model_text = str(model or "").strip().lower()
    provider_text = str(provider or "").strip().lower()
    exact = table[table["model"] == model_text]
    if provider_text and not exact.empty:
        provider_exact = exact[exact["provider"] == provider_text]
        if not provider_exact.empty:
            return provider_exact.iloc[0]
    if not exact.empty:
        return exact.iloc[0]
    return None


def estimated_tokens_for(prompt_type: str) -> Dict[str, int]:
    return dict(ESTIMATED_TOKEN_PROFILES.get(prompt_type, ESTIMATED_TOKEN_PROFILES["unknown"]))


def estimate_result_cost(
    *,
    model: Any,
    provider: Any = "",
    tags: Any = "",
    prompt_type: Optional[str] = None,
    input_tokens: Any = None,
    output_tokens: Any = None,
    cost_table: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Return cost fields using actual tokens when present, estimated otherwise."""
    resolved_prompt = prompt_type or infer_prompt_type(tags)
    actual_input = _to_int(input_tokens)
    actual_output = _to_int(output_tokens)
    has_actual_tokens = actual_input is not None and actual_output is not None

    if has_actual_tokens:
        used_input = actual_input or 0
        used_output = actual_output or 0
        source = "actual"
    else:
        estimates = estimated_tokens_for(resolved_prompt)
        used_input = estimates["input_tokens"]
        used_output = estimates["output_tokens"]
        source = "estimated"

    price_row = _price_row_for(model, provider, cost_table)
    if price_row is None:
        return {
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "total_tokens": (actual_input or 0) + (actual_output or 0) if has_actual_tokens else None,
            "estimated_input_tokens": used_input if not has_actual_tokens else None,
            "estimated_output_tokens": used_output if not has_actual_tokens else None,
            "estimated_cost_usd": None,
            "cost_source": "missing-price",
            "price_version": "",
            "cost_note": f"No local price row for model={model}",
        }

    input_price = _to_float(price_row.get("input_price_per_1m_tokens")) or 0.0
    output_price = _to_float(price_row.get("output_price_per_1m_tokens")) or 0.0
    cost = (used_input / 1_000_000) * input_price + (used_output / 1_000_000) * output_price
    note = "Actual token usage from provider metadata" if source == "actual" else "Estimated tokens; old result files did not store usage"

    return {
        "input_tokens": actual_input,
        "output_tokens": actual_output,
        "total_tokens": (actual_input or 0) + (actual_output or 0) if has_actual_tokens else None,
        "estimated_input_tokens": used_input if not has_actual_tokens else None,
        "estimated_output_tokens": used_output if not has_actual_tokens else None,
        "estimated_cost_usd": round(cost, 6),
        "cost_source": source,
        "price_version": price_row.get("price_version", PRICE_VERSION),
        "cost_note": note,
    }


def _read_result_csv(csv_path: Path, *, source_kind: str = "primary", source_rank: int = 0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["source_file"] = str(csv_path)
    df["result_source_file"] = str(csv_path)
    df["source_label"] = csv_path.parent.name
    df["result_source_kind"] = source_kind
    df["result_source_rank"] = source_rank
    df["rerun_overlay_bool"] = source_kind == "rerun"
    return df


def _default_phase12_sources() -> List[Tuple[Path, str]]:
    return [(path, "primary") for path in DEFAULT_PHASE12_PRIMARY_RESULT_FILES] + [
        (path, "rerun") for path in DEFAULT_PHASE12_RERUN_RESULT_FILES
    ]


def _explicit_phase12_sources(
    csv_paths: Iterable[Path],
    rerun_csv_paths: Optional[Iterable[Path]] = None,
) -> List[Tuple[Path, str]]:
    sources = [(Path(path), "primary") for path in csv_paths]
    sources.extend((Path(path), "rerun") for path in (rerun_csv_paths or []))
    return sources


def _apply_rerun_overlay(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate execution rows using rerun results as a transparent overlay."""
    if df.empty or "experiment_id" not in df.columns or "repetition" not in df.columns:
        return df

    ranked = df.copy()
    ranked["_overlay_experiment_id"] = ranked["experiment_id"].astype(str)
    ranked["_overlay_repetition"] = ranked["repetition"].astype(str)
    ranked["_overlay_timestamp"] = pd.to_datetime(ranked.get("timestamp"), errors="coerce")
    ranked["_overlay_has_psbt"] = ranked.get("psbt_generated_bool", False).fillna(False).astype(bool)
    ranked["_overlay_source_rank"] = pd.to_numeric(ranked.get("result_source_rank"), errors="coerce").fillna(0)
    ranked["_overlay_row_rank"] = range(len(ranked))

    ranked = ranked.sort_values(
        [
            "_overlay_experiment_id",
            "_overlay_repetition",
            "_overlay_has_psbt",
            "_overlay_timestamp",
            "_overlay_source_rank",
            "_overlay_row_rank",
        ],
        ascending=[True, True, True, True, True, True],
        na_position="first",
    )
    deduped = ranked.groupby(["_overlay_experiment_id", "_overlay_repetition"], dropna=False).tail(1)
    return deduped.drop(
        columns=[
            "_overlay_experiment_id",
            "_overlay_repetition",
            "_overlay_timestamp",
            "_overlay_has_psbt",
            "_overlay_source_rank",
            "_overlay_row_rank",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def normalize_phase12_results(
    csv_paths: Optional[Iterable[Path]] = None,
    *,
    cost_table_path: Path = DEFAULT_COST_TABLE,
    rerun_csv_paths: Optional[Iterable[Path]] = None,
    apply_rerun_overlay: bool = True,
) -> pd.DataFrame:
    """Load the 2026 prompt-corpus CSV+JSON result pairs as one normalized table."""
    sources = _default_phase12_sources() if csv_paths is None else _explicit_phase12_sources(csv_paths, rerun_csv_paths)
    cost_table = load_cost_table(cost_table_path)
    frames: List[pd.DataFrame] = []

    for source_rank, (csv_path, source_kind) in enumerate(sources):
        if not csv_path.exists():
            continue
        df = _read_result_csv(csv_path, source_kind=source_kind, source_rank=source_rank)
        json_index = _json_result_index(csv_path.with_suffix(".json"))
        flattened_rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            key = (str(row.get("experiment_id", "")), str(row.get("repetition", "")))
            json_row = json_index.get(key, {})
            breakdown = json_row.get("privacy_breakdown") if json_row else None
            flat = flatten_privacy_breakdown(breakdown)

            csv_tags = row.get("tags", "")
            json_tags = json_row.get("tags") if json_row else None
            tags = split_tags(json_tags) or split_tags(csv_tags)
            prompt_text = " ".join(
                str(value or "")
                for value in (
                    row.get("user_prompt"),
                    json_row.get("user_prompt") if json_row else "",
                )
            )
            prompt_type = infer_prompt_type(tags, prompt_text)
            phase = infer_phase(tags, row.get("experiment_id"), row.get("experiment_name"))
            amount_pct = infer_amount_pct(tags, prompt_text)
            temperature = infer_temperature(tags, row.get("llm_temperature"))
            provider = row.get("llm_provider") or json_row.get("llm_provider", "")
            model = row.get("llm_model") or json_row.get("llm_model", "")

            privacy_score = _to_float(row.get("privacy_score"))
            if privacy_score is None:
                privacy_score = _to_float(json_row.get("privacy_score") if json_row else None)
            if flat["score_overall"] is None:
                flat["score_overall"] = privacy_score

            psbt_generated = _to_bool(row.get("psbt_generated"))
            if psbt_generated is None:
                psbt_generated = _to_bool(json_row.get("psbt_generated") if json_row else None)
            psbt_generated = bool(psbt_generated)

            fee_ok_raw = row.get("fee_sanity_ok")
            if str(fee_ok_raw or "").strip() == "":
                fee_ok_raw = flat.get("fee_sanity_ok")
            fee_ok_bool = _to_bool(fee_ok_raw)
            fee_ok_int = 1 if fee_ok_bool is True else 0 if fee_ok_bool is False else None
            fee_bad = bool(psbt_generated and fee_ok_bool is False)

            input_tokens = row.get("input_tokens", json_row.get("input_tokens") if json_row else None)
            output_tokens = row.get("output_tokens", json_row.get("output_tokens") if json_row else None)
            cost = estimate_result_cost(
                model=model,
                provider=provider,
                tags=tags,
                prompt_type=prompt_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_table=cost_table,
            )

            flattened_rows.append(
                {
                    **flat,
                    **cost,
                    "phase": phase,
                    "phase_label": PHASE_LABELS.get(phase, phase),
                    "execution_lot_label": EXECUTION_LOT_LABELS.get(phase, phase),
                    "amount_pct": amount_pct,
                    "amount_label": f"{amount_pct:g}%" if amount_pct is not None else "",
                    "prompt_type": prompt_type,
                    "prompt_label": PROMPT_LABELS.get(prompt_type, prompt_type.replace("_", " ").title()),
                    "prompt_comparison_label": PROMPT_LABELS.get(prompt_type, prompt_type.replace("_", " ").title()),
                    "temperature": temperature,
                    "temperature_label": f"T={temperature:g}" if temperature is not None else "",
                    "category": infer_category(provider, model, tags),
                    "tags_normalized": "|".join(tags),
                    "psbt_generated_bool": psbt_generated,
                    "fee_sanity_ok_int": fee_ok_int,
                    "fee_ok_bool": bool(fee_ok_bool is True),
                    "fee_bad_bool": fee_bad,
                    "failed_bool": not psbt_generated,
                    "usable_score": privacy_score if fee_ok_bool is True else None,
                }
            )

        extras = pd.DataFrame(flattened_rows)
        merged = df.reset_index(drop=True).copy()
        for column in extras.columns:
            values = extras[column].reset_index(drop=True)
            if column in merged.columns:
                current = merged[column]
                missing = current.isna() | current.astype("object").astype(str).str.strip().isin({"", "nan", "None"})
                merged[column] = current.where(~missing, values)
            else:
                merged[column] = values
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    numeric_columns = [
        "privacy_score", "score_overall", "score_clustering", "score_change_detection",
        "score_fingerprinting", "score_metadata_leakage", "confidence_numeric",
        "fee_rate_sat_vb", "fee_sats", "num_inputs", "num_outputs",
        "total_input_sats", "total_output_sats", "change_probability",
        "amount_pct", "temperature", "execution_time_seconds", "estimated_cost_usd",
        "input_tokens", "output_tokens", "total_tokens", "estimated_input_tokens",
        "estimated_output_tokens",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if apply_rerun_overlay:
        result = _apply_rerun_overlay(result)
    return result


def build_phase12_amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    fee_ok = df[df["fee_ok_bool"]].copy()
    if fee_ok.empty:
        return pd.DataFrame()
    grouped = fee_ok.groupby(["phase", "phase_label", "prompt_type", "prompt_label", "amount_pct", "amount_label"], dropna=False)
    return grouped.agg(
        n_fee_ok=("usable_score", "count"),
        mean_score=("usable_score", "mean"),
        std_score=("usable_score", "std"),
        mean_inputs=("num_inputs", "mean"),
        mean_outputs=("num_outputs", "mean"),
        mean_clustering=("score_clustering", "mean"),
        mean_change_detection=("score_change_detection", "mean"),
        mean_fingerprinting=("score_fingerprinting", "mean"),
        median_fee_sats=("fee_sats", "median"),
        median_fee_rate_sat_vb=("fee_rate_sat_vb", "median"),
    ).reset_index()


def build_phase12_reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["phase", "phase_label", "execution_lot_label", "prompt_type", "prompt_label", "llm_model", "category"],
        dropna=False,
    )
    summary = grouped.agg(
        attempts=("experiment_id", "count"),
        fee_ok=("fee_ok_bool", "sum"),
        fee_bad=("fee_bad_bool", "sum"),
        failed=("failed_bool", "sum"),
        mean_fee_ok_score=("usable_score", "mean"),
    ).reset_index()
    summary["fee_ok_rate"] = summary["fee_ok"] / summary["attempts"]
    summary["fee_bad_rate"] = summary["fee_bad"] / summary["attempts"]
    summary["failed_rate"] = summary["failed"] / summary["attempts"]
    return summary


def build_phase12_uplift_summary(df: pd.DataFrame) -> pd.DataFrame:
    fee_ok = df[df["fee_ok_bool"]].copy()
    if fee_ok.empty:
        return pd.DataFrame()
    privacy_simple = (
        fee_ok[fee_ok["prompt_type"] == "privacy_simple"]
        .groupby(["llm_model", "amount_pct", "temperature"], dropna=False)["usable_score"]
        .mean()
        .reset_index()
        .rename(columns={"usable_score": "privacy_simple_score"})
    )
    multiturn = (
        fee_ok[fee_ok["prompt_type"] == "multiturn_detailed"]
        .groupby(["llm_model", "amount_pct", "temperature"], dropna=False)["usable_score"]
        .mean()
        .reset_index()
        .rename(columns={"usable_score": "multiturn_detailed_score"})
    )
    merged = multiturn.merge(privacy_simple, on=["llm_model", "amount_pct", "temperature"], how="inner")
    merged["prompt_delta_score"] = merged["multiturn_detailed_score"] - merged["privacy_simple_score"]
    merged["uplift_score"] = merged["prompt_delta_score"]
    merged["amount_label"] = merged["amount_pct"].map(lambda value: f"{value:g}%" if pd.notna(value) else "")
    merged["temperature_label"] = merged["temperature"].map(lambda value: f"T={value:g}" if pd.notna(value) else "")
    return merged


def build_phase12_cost_efficiency_summary(df: pd.DataFrame) -> pd.DataFrame:
    fee_ok = df[(df["fee_ok_bool"]) & (df["estimated_cost_usd"].notna())].copy()
    if fee_ok.empty:
        return pd.DataFrame()
    grouped = fee_ok.groupby(["llm_model", "category", "cost_source"], dropna=False).agg(
        n_fee_ok=("usable_score", "count"),
        mean_score=("usable_score", "mean"),
        best_score=("usable_score", "max"),
        mean_cost_usd=("estimated_cost_usd", "mean"),
        total_cost_usd=("estimated_cost_usd", "sum"),
    ).reset_index()
    grouped["score_per_estimated_dollar"] = grouped["mean_score"] / grouped["mean_cost_usd"]
    return grouped
