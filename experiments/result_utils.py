#!/usr/bin/env python3
"""Result loading helpers for the local experiments web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


TRUE_VALUES = {"true", "1", "1.0", "yes", "y"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n"}
BOOL_COLUMNS = ["success", "psbt_generated", "psbt_available", "fee_sanity_ok"]
NUMERIC_COLUMNS = [
    "repetition",
    "llm_temperature",
    "execution_time_seconds",
    "privacy_score",
    "amount_btc",
    "fee_rate_sat_vb",
    "fee_sats",
    "confidence_numeric",
    "num_inputs",
    "num_outputs",
]


def _key(experiment_id: Any, repetition: Any) -> Tuple[str, str]:
    return (str(experiment_id), str(repetition))


def _to_nullable_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool/int/string result columns to pandas BooleanDtype."""
    values = series.astype("object").where(series.notna(), "")
    text = values.astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.isin(TRUE_VALUES)] = True
    result[text.isin(FALSE_VALUES)] = False
    return result


def normalize_result_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed CSV/JSON result types before Streamlit/Arrow sees them."""
    if df is None or df.empty:
        return df

    normalized = df.copy()
    for col in BOOL_COLUMNS:
        if col in normalized.columns:
            normalized[col] = _to_nullable_bool(normalized[col])
    for col in NUMERIC_COLUMNS:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only dataframe with object columns safe for PyArrow."""
    if df is None or df.empty:
        return df

    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_object_dtype(safe[col]):
            safe[col] = safe[col].map(_display_value)
    return safe


def _json_result_index(json_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    return {
        _key(item.get("experiment_id"), item.get("repetition")): item
        for item in data
        if isinstance(item, dict)
    }


def flatten_privacy_breakdown(breakdown: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten the scorer fields the UI and summary CSV care about."""
    if not isinstance(breakdown, dict):
        return {
            "fee_sanity_ok": "",
            "sanity_status": "",
            "fee_rate_sat_vb": "",
            "fee_sats": "",
            "confidence": "",
            "confidence_numeric": "",
            "num_inputs": "",
            "num_outputs": "",
        }

    fee_analysis = breakdown.get("fee_analysis") or {}
    metadata = breakdown.get("metadata") or {}
    return {
        "fee_sanity_ok": breakdown.get("fee_sanity_ok", ""),
        "sanity_status": breakdown.get("sanity_status", ""),
        "fee_rate_sat_vb": fee_analysis.get("fee_rate_sat_vb", ""),
        "fee_sats": fee_analysis.get("fee_sats", ""),
        "confidence": breakdown.get("confidence", ""),
        "confidence_numeric": breakdown.get("confidence_numeric", ""),
        "num_inputs": metadata.get("num_inputs", ""),
        "num_outputs": metadata.get("num_outputs", ""),
    }


def load_results_dataframe(csv_path: Path, manager=None) -> pd.DataFrame:
    """Load a result CSV and enrich it with sibling JSON and experiment metadata."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    result_index = _json_result_index(csv_path.with_suffix(".json"))
    extra_rows = []
    for _, row in df.iterrows():
        extra: Dict[str, Any] = {}
        result_json = result_index.get(_key(row.get("experiment_id"), row.get("repetition")), {})
        if result_json:
            extra["psbt_file"] = row.get("psbt_file") or result_json.get("psbt_file") or ""
            extra.update(flatten_privacy_breakdown(result_json.get("privacy_breakdown")))
        else:
            extra["psbt_file"] = row.get("psbt_file", "")
            extra.update(flatten_privacy_breakdown(None))
        extra_rows.append(extra)

    extras = pd.DataFrame(extra_rows)
    for col in extras.columns:
        if col not in df.columns:
            df[col] = extras[col]
        else:
            df[col] = df[col].where(df[col].notna() & (df[col].astype(str) != ""), extras[col])

    psbt_generated = (
        df["psbt_generated"]
        if "psbt_generated" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )
    psbt_file = (
        df["psbt_file"]
        if "psbt_file" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    df["psbt_available"] = psbt_generated.astype(str).str.lower().isin(
        {"true", "1", "yes"}
    ) | psbt_file.fillna("").astype(str).ne("")

    if manager is not None:
        metadata_by_id = {}
        for exp in manager.read_experiments():
            meta = manager.parse_csv_row_to_meta(exp)
            prompt_mode = getattr(meta, "prompt_mode", None)
            if not prompt_mode and hasattr(manager, "infer_prompt_mode"):
                prompt_mode = manager.infer_prompt_mode(exp)
            if prompt_mode not in ("template", "custom"):
                prompt_mode = "template" if exp.get("amount_btc") or exp.get("strategy") else "custom"
            amount_display = None
            if hasattr(manager, "infer_amount_display"):
                amount_display = manager.infer_amount_display(exp)
            metadata_by_id[meta.id] = {
                "strategy": meta.strategy,
                "amount_display": amount_display,
                "amount_btc": meta.amount_btc,
                "experiment_tags": "|".join(meta.tags),
                "prompt_mode": prompt_mode,
            }

        for col in ("strategy", "amount_display", "amount_btc", "experiment_tags", "prompt_mode"):
            if col not in df.columns:
                df[col] = pd.Series([None] * len(df), dtype="object")
            else:
                df[col] = df[col].astype("object")
        for idx, row in df.iterrows():
            meta = metadata_by_id.get(row.get("experiment_id"))
            if not meta:
                continue
            for col, value in meta.items():
                if str(df.at[idx, col]) in ("", "nan", "None"):
                    df.at[idx, col] = value

    return normalize_result_types(df)


def load_many_results_dataframes(csv_paths: Iterable[Path], manager=None) -> pd.DataFrame:
    """Load and concatenate several result CSV/JSON pairs with source file tracking."""
    frames = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        df = load_results_dataframe(path, manager=manager)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return normalize_result_types(pd.concat(frames, ignore_index=True, sort=False))


def display_columns(columns: Iterable[str]) -> list[str]:
    """Return a compact, stable column order for Streamlit tables."""
    preferred = [
        "source_file",
        "experiment_id",
        "experiment_name",
        "repetition",
        "llm_provider",
        "llm_model",
        "strategy",
        "amount_display",
        "amount_btc",
        "success",
        "psbt_generated",
        "psbt_available",
        "privacy_score",
        "privacy_grade",
        "fee_sanity_ok",
        "sanity_status",
        "fee_rate_sat_vb",
        "fee_sats",
        "execution_time_seconds",
        "psbt_file",
        "error_message",
    ]
    available = list(columns)
    return [col for col in preferred if col in available]
