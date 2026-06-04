#!/usr/bin/env python3
"""Paper-style chart helpers for the local experiments web UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import altair as alt
import pandas as pd

from phase12_results import (
    AMOUNT_ORDER,
    build_phase12_amount_summary,
    build_phase12_cost_efficiency_summary,
    build_phase12_reliability_summary,
    build_phase12_uplift_summary,
    normalize_phase12_results,
)
from wallet_baseline_results import (
    WALLET_LABELS,
    WALLET_ORDER,
    build_wallet_agent_amount_comparison,
    build_wallet_reliability_summary,
    normalize_wallet_baseline_results,
    select_latest_successful_by_wallet_amount,
)


ANALYSIS_CHARTS_DIR = Path(
    os.getenv(
        "ANALYSIS_CHARTS_DIR",
        Path(__file__).resolve().parents[2] / "analysis" / "charts",
    )
)
alt.data_transformers.disable_max_rows()

MODEL_ORDER = [
    "Claude Opus 4.5", "Claude Opus 4.6", "Claude Sonnet 4.5",
    "GPT-5.2", "GPT-5.2 Pro", "Gemini 3 Pro",
    "DeepSeek V3.2", "Llama 3.3 70B", "Mistral Large", "Qwen 2.5 72B",
]

PROMPT_ORDER = [
    "basic", "privacy_simple", "multiturn_simple",
    "multiturn_detailed", "privacy_detailed",
]

PROMPT_LABELS = {
    "basic": "Basic",
    "privacy_simple": "Privacy Simple",
    "multiturn_simple": "Multi-turn Simple",
    "multiturn_detailed": "Multi-turn Detailed",
    "privacy_detailed": "Privacy Detailed",
}

PROMPT_LABEL_ORDER = [PROMPT_LABELS[prompt] for prompt in PROMPT_ORDER]

CATEGORY_COLORS = {
    "closed-source": "#E07B39",
    "open-source": "#3B7DD8",
    "unknown": "#808080",
}

PAPER_CHART_OPTIONS = [
    "Score heatmap",
    "Fee insanity heatmap",
    "Best model scores",
    "Top 3 model scores",
    "Prompt strategy effect",
    "Success and fee sanity",
    "Open-source vs closed-source",
    "Score distribution by prompt",
    "Fee rate distribution",
    "Execution time vs score",
    "Top model sub-scores",
]

PHASE12_CHART_OPTIONS = [
    "Wallet % vs score",
    "Model amount heatmaps",
    "Prompt delta",
    "Reliability by prompt",
    "Structure difficulty",
    "Sub-score tradeoff",
    "Fee tradeoff",
    "Cost vs score",
    "Cost efficiency",
    "Temperature effect",
]

WALLET_BASELINE_CHART_OPTIONS = [
    "Wallet score by amount",
    "Wallet amount heatmap",
    "Wallet vs agents",
    "Wallet delta vs agents",
    "Wallet reliability",
    "Wallet structure difficulty",
    "Wallet fee tradeoff",
    "Wallet sub-score tradeoff",
]

PHASE12_AXIS_LABEL_LIMIT = 220
PHASE12_LEGEND_LABEL_LIMIT = 180
PHASE12_CHART_WIDTH = 680
PHASE12_DETAIL_CHART_WIDTH = 820


def _truthy_series(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(series.notna(), "")
    return values.astype(str).str.lower().isin({"true", "1", "1.0", "yes", "y"})


def _fee_ok_series(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(series.notna(), "")
    return values.astype(str).str.lower().isin({"true", "1", "1.0", "yes", "y"})


def _series(df: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    """Return a column as a Series, or a same-length default Series."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _has_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(column in df.columns for column in columns)


def prompt_type_from_strategy(value: object) -> str:
    """Map UI strategy strings to the paper aggregation prompt_type naming."""
    prompt = _normalize_prompt_token(value)
    aliases = {
        "privacy_simple": "privacy_simple",
        "privacy_detailed": "privacy_detailed",
        "multiturn_simple": "multiturn_simple",
        "multi_turn_simple": "multiturn_simple",
        "multiturn_detailed": "multiturn_detailed",
        "multi_turn_detailed": "multiturn_detailed",
        "basic": "basic",
    }
    return aliases.get(prompt, prompt or "unknown")


def _normalize_prompt_token(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip().lower().replace("-", "_")


def _known_prompt_type(value: object) -> Optional[str]:
    prompt_type = prompt_type_from_strategy(value)
    return prompt_type if prompt_type in PROMPT_LABELS else None


def _tokens_from_row(row: pd.Series) -> List[str]:
    values = []
    for column in ("strategy", "prompt_type", "tags", "experiment_tags", "experiment_id", "experiment_name"):
        if column in row.index:
            values.append(row.get(column))
    tokens: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        for separator in (";", "|", ",", " "):
            text = text.replace(separator, " ")
        tokens.extend(part for part in text.split() if part)
    return tokens


def prompt_type_from_row(row: pd.Series) -> str:
    """Infer prompt type from structured metadata, then tags used by temporary matrices."""
    for column in ("strategy", "prompt_type"):
        if column in row.index:
            prompt_type = prompt_type_from_strategy(row.get(column))
            if prompt_type != "unknown":
                return prompt_type

    for token in _tokens_from_row(row):
        normalized = _normalize_prompt_token(token)
        if normalized.startswith("prompt_"):
            normalized = normalized.removeprefix("prompt_")
        prompt_type = _known_prompt_type(normalized)
        if prompt_type:
            return prompt_type
    return "unknown"


def category_from_provider(provider: object) -> str:
    """Match the paper's open-source/closed-source split for current UI rows."""
    provider_text = str(provider or "").strip().lower()
    if provider_text == "openrouter":
        return "open-source"
    if provider_text:
        return "closed-source"
    return "unknown"


def category_from_row(row: pd.Series) -> str:
    """Infer open/closed category from tags when provider alone is ambiguous."""
    tokens = {_normalize_prompt_token(token) for token in _tokens_from_row(row)}
    model = str(row.get("llm_model", row.get("model_short", "")) or "").lower()
    provider = row.get("llm_provider", row.get("provider", ""))

    if tokens & {"open_weight", "open_source", "opensource"}:
        return "open-source"
    if any(marker in model for marker in ("glm", "gemma", "llama", "mistral", "qwen", "deepseek")):
        return "open-source"
    if tokens & {"frontier", "closed_source", "closed"}:
        return "closed-source"
    return category_from_provider(provider)


def model_sort_order(models: Iterable[str]) -> List[str]:
    """Paper model order first, then any current-run model names."""
    seen = set()
    ordered: List[str] = []
    for model in MODEL_ORDER + sorted(str(model) for model in models):
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def load_paper_chart_sources(base_dir: Path = ANALYSIS_CHARTS_DIR) -> Dict[str, object]:
    """Load chart data generated by analysis/charts/run_aggregation.py."""
    base_dir = Path(base_dir)
    sources: Dict[str, object] = {
        "base_dir": base_dir,
        "missing": [],
        "pdfs": sorted((base_dir / "charts").glob("*.pdf")) if (base_dir / "charts").exists() else [],
    }

    for key, filename in (
        ("aggregated", "aggregated_results.csv"),
        ("model_summary", "model_summary.csv"),
        ("model_costs", "model_costs.csv"),
    ):
        path = base_dir / filename
        if path.exists():
            sources[key] = pd.read_csv(path)
        else:
            sources["missing"].append(str(path))
            sources[key] = pd.DataFrame()

    scores_path = base_dir / "v2_scores.json"
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8") as f:
            sources["v2_scores"] = pd.DataFrame(json.load(f))
    else:
        sources["missing"].append(str(scores_path))
        sources["v2_scores"] = pd.DataFrame()

    return sources


def prepare_aggregated_dataframe(agg: pd.DataFrame) -> pd.DataFrame:
    """Normalize aggregated chart data for Altair."""
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = agg.copy()
    numeric_cols = [
        "n_psbts", "n_total_attempts", "success_rate", "avg_v2_score",
        "std_v2_score", "min_v2_score", "max_v2_score",
        "avg_v2_score_fee_filtered", "n_fee_sane", "n_fee_insane",
        "fee_insanity_rate", "avg_fee_rate_sat_vb",
        "avg_execution_time_seconds", "avg_clustering",
        "avg_change_detection", "avg_fingerprinting",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "prompt_type" in df.columns:
        df["prompt_label"] = df["prompt_type"].map(PROMPT_LABELS).fillna(df["prompt_type"].astype(str))
    if "category" not in df.columns and "provider" in df.columns:
        df["category"] = df["provider"].map(category_from_provider)
    return df


def prepare_v2_scores_dataframe(v2_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize v2 score rows for paper-style distribution charts."""
    if v2_df is None or v2_df.empty:
        return pd.DataFrame()
    df = v2_df.copy()
    for col in ("v2_overall_score", "fee_sanity_ok", "fee_rate_sat_vb"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "prompt_type" in df.columns:
        df["prompt_label"] = df["prompt_type"].map(PROMPT_LABELS).fillna(df["prompt_type"].astype(str))
    return df


def aggregate_current_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build a paper-like aggregate table from the currently selected result CSV."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["model_full"] = _series(df, "llm_model").astype(str)
    df["provider"] = _series(df, "llm_provider").astype(str)
    df["category"] = df.apply(category_from_row, axis=1)
    df["prompt_type"] = df.apply(prompt_type_from_row, axis=1)
    df["privacy_score_numeric"] = pd.to_numeric(_series(df, "privacy_score"), errors="coerce")
    df["execution_time_numeric"] = pd.to_numeric(_series(df, "execution_time_seconds"), errors="coerce")
    df["fee_rate_numeric"] = pd.to_numeric(_series(df, "fee_rate_sat_vb"), errors="coerce")

    if "psbt_available" in df.columns:
        df["psbt_ok"] = _truthy_series(df["psbt_available"])
    elif "psbt_generated" in df.columns:
        df["psbt_ok"] = _truthy_series(df["psbt_generated"])
    else:
        df["psbt_ok"] = False

    if "fee_sanity_ok" in df.columns:
        df["fee_ok"] = _fee_ok_series(df["fee_sanity_ok"])
        df["fee_bad"] = df["psbt_ok"] & df["fee_sanity_ok"].notna() & ~df["fee_ok"]
    else:
        df["fee_ok"] = False
        df["fee_bad"] = False

    rows = []
    group_cols = ["model_full", "provider", "category", "prompt_type"]
    for keys, group in df.groupby(group_cols, dropna=False):
        model_full, provider, category, prompt_type = keys
        n_total = len(group)
        psbt_group = group[group["psbt_ok"]]
        fee_ok_group = group[group["fee_ok"]]
        fee_bad_group = group[group["fee_bad"]]
        score_values = psbt_group["privacy_score_numeric"].dropna()
        fee_score_values = fee_ok_group["privacy_score_numeric"].dropna()
        rows.append({
            "model_short": model_full,
            "model_full": model_full,
            "provider": provider,
            "category": category,
            "prompt_type": prompt_type,
            "n_psbts": len(psbt_group),
            "n_total_attempts": n_total,
            "success_rate": len(psbt_group) / n_total if n_total else 0,
            "avg_v2_score": score_values.mean() if not score_values.empty else pd.NA,
            "std_v2_score": score_values.std() if len(score_values) > 1 else 0,
            "min_v2_score": score_values.min() if not score_values.empty else pd.NA,
            "max_v2_score": score_values.max() if not score_values.empty else pd.NA,
            "avg_v2_score_fee_filtered": fee_score_values.mean() if not fee_score_values.empty else pd.NA,
            "n_fee_sane": len(fee_ok_group),
            "n_fee_insane": len(fee_bad_group),
            "fee_insanity_rate": len(fee_bad_group) / len(psbt_group) if len(psbt_group) else pd.NA,
            "avg_fee_rate_sat_vb": fee_ok_group["fee_rate_numeric"].mean(),
            "avg_execution_time_seconds": group["execution_time_numeric"].mean(),
            "avg_clustering": pd.NA,
            "avg_change_detection": pd.NA,
            "avg_fingerprinting": pd.NA,
        })

    return prepare_aggregated_dataframe(pd.DataFrame(rows))


def current_results_v2_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build v2-score-like rows from the currently selected result CSV."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    if "fee_sanity_ok" in df.columns:
        fee_sanity = _fee_ok_series(df["fee_sanity_ok"]).astype(int)
    else:
        fee_sanity = pd.Series([pd.NA] * len(df), index=df.index)
    return prepare_v2_scores_dataframe(pd.DataFrame({
        "model_short": _series(df, "llm_model"),
        "category": df.apply(category_from_row, axis=1),
        "prompt_type": df.apply(prompt_type_from_row, axis=1),
        "v2_overall_score": pd.to_numeric(_series(df, "privacy_score"), errors="coerce"),
        "fee_sanity_ok": fee_sanity,
        "fee_rate_sat_vb": pd.to_numeric(_series(df, "fee_rate_sat_vb"), errors="coerce"),
    }))


def chart_score_heatmap(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "prompt_label", "avg_v2_score_fee_filtered"]):
        return None
    df = df.dropna(subset=["avg_v2_score_fee_filtered"])
    if df.empty:
        return None
    model_order = model_sort_order(df["model_full"].unique())
    base = alt.Chart(df).encode(
        x=alt.X("prompt_label:N", sort=PROMPT_LABEL_ORDER, title="Prompt strategy"),
        y=alt.Y("model_full:N", sort=model_order, title=None),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "avg_v2_score_fee_filtered:Q",
            title="Privacy score",
            scale=alt.Scale(scheme="redyellowgreen", domain=[40, 100]),
        ),
        tooltip=["model_full", "prompt_label", alt.Tooltip("avg_v2_score_fee_filtered:Q", format=".1f")],
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("avg_v2_score_fee_filtered:Q", format=".1f"))
    return (heatmap + text).properties(title="Average Privacy Score by Model and Prompt Strategy")


def chart_fee_insanity_heatmap(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "prompt_label", "fee_insanity_rate"]):
        return None
    df = df.dropna(subset=["fee_insanity_rate"]).copy()
    if df.empty:
        return None
    df["fee_insanity_pct"] = df["fee_insanity_rate"] * 100
    model_order = model_sort_order(df["model_full"].unique())
    base = alt.Chart(df).encode(
        x=alt.X("prompt_label:N", sort=PROMPT_LABEL_ORDER, title="Prompt strategy"),
        y=alt.Y("model_full:N", sort=model_order, title=None),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "fee_insanity_pct:Q",
            title="Fee insanity (%)",
            scale=alt.Scale(scheme="redyellowgreen", reverse=True, domain=[0, 100]),
        ),
        tooltip=["model_full", "prompt_label", alt.Tooltip("fee_insanity_pct:Q", format=".0f")],
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("fee_insanity_pct:Q", format=".0f"))
    return (heatmap + text).properties(title="Fee Insanity Rate by Model and Prompt Strategy")


def chart_best_model_scores(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "category", "avg_v2_score", "avg_v2_score_fee_filtered"]):
        return None
    best = df.groupby(["model_full", "category"], dropna=False).agg(
        best_raw=("avg_v2_score", "max"),
        best_fee_filtered=("avg_v2_score_fee_filtered", "max"),
    ).reset_index()
    long = best.melt(
        id_vars=["model_full", "category"],
        value_vars=["best_raw", "best_fee_filtered"],
        var_name="score_type",
        value_name="score",
    ).dropna(subset=["score"])
    if long.empty:
        return None
    long["score_type"] = long["score_type"].map({
        "best_raw": "Best raw score",
        "best_fee_filtered": "Best fee-filtered score",
    })
    model_order = best.sort_values("best_fee_filtered", ascending=False)["model_full"].tolist()
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("model_full:N", sort=model_order, title=None, axis=alt.Axis(labelAngle=-30)),
        xOffset="score_type:N",
        y=alt.Y("score:Q", title="Privacy score", scale=alt.Scale(domain=[0, 105])),
        color=alt.Color("score_type:N", title="Score type"),
        tooltip=["model_full", "score_type", alt.Tooltip("score:Q", format=".1f"), "category"],
    ).properties(title="Best Privacy Score per Model", width=760, height=420)


def chart_top3_model_scores(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "category", "avg_v2_score_fee_filtered"]):
        return None
    top = (
        df.groupby(["model_full", "category"], dropna=False)["avg_v2_score_fee_filtered"]
        .max()
        .reset_index()
        .dropna(subset=["avg_v2_score_fee_filtered"])
        .sort_values("avg_v2_score_fee_filtered", ascending=False)
        .head(3)
    )
    if top.empty:
        return None
    return alt.Chart(top).mark_bar().encode(
        x=alt.X("model_full:N", sort=top["model_full"].tolist(), title=None, axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("avg_v2_score_fee_filtered:Q", title="Best fee-filtered privacy score", scale=alt.Scale(domain=[0, 105])),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
            title="Category",
        ),
        tooltip=["model_full", "category", alt.Tooltip("avg_v2_score_fee_filtered:Q", format=".1f")],
    ).properties(title="Top 3 Models by Fee-Filtered Privacy Score", width=620, height=420)


def chart_prompt_strategy_effect(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "category", "prompt_type", "prompt_label", "avg_v2_score_fee_filtered"]):
        return None
    df = df.dropna(subset=["avg_v2_score_fee_filtered"])
    if df.empty:
        return None
    model_lines = alt.Chart(df).mark_line(point=True, opacity=0.35).encode(
        x=alt.X("prompt_label:N", sort=PROMPT_LABEL_ORDER, title="Prompt strategy"),
        y=alt.Y("avg_v2_score_fee_filtered:Q", title="Privacy score", scale=alt.Scale(domain=[40, 100])),
        detail="model_full:N",
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
            title="Category",
        ),
        tooltip=["model_full", "prompt_label", alt.Tooltip("avg_v2_score_fee_filtered:Q", format=".1f")],
    )
    avg_df = df.groupby(["prompt_type", "prompt_label"], as_index=False)["avg_v2_score_fee_filtered"].mean()
    avg_line = alt.Chart(avg_df).mark_line(point=True, color="black", strokeWidth=4).encode(
        x=alt.X("prompt_label:N", sort=PROMPT_LABEL_ORDER),
        y="avg_v2_score_fee_filtered:Q",
        tooltip=["prompt_label", alt.Tooltip("avg_v2_score_fee_filtered:Q", format=".1f")],
    )
    return (model_lines + avg_line).properties(title="Privacy Score by Prompt Strategy", width=760, height=420)


def chart_success_fee_sanity(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "category", "n_fee_sane", "n_fee_insane", "n_total_attempts"]):
        return None
    by_model = df.groupby(["model_full", "category"], dropna=False).agg(
        fee_ok=("n_fee_sane", "sum"),
        fee_bad=("n_fee_insane", "sum"),
        total=("n_total_attempts", "sum"),
    ).reset_index()
    by_model["failed"] = (by_model["total"] - by_model["fee_ok"] - by_model["fee_bad"]).clip(lower=0)
    long = by_model.melt(
        id_vars=["model_full", "category", "total"],
        value_vars=["fee_ok", "fee_bad", "failed"],
        var_name="status",
        value_name="count",
    )
    long["status"] = long["status"].map({
        "fee_ok": "Valid PSBT + Fee OK",
        "fee_bad": "Valid PSBT + Fee Bad",
        "failed": "Failed to generate",
    })
    model_order = model_sort_order(by_model["model_full"].unique())
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("model_full:N", sort=model_order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("count:Q", title="Number of experiments"),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=["Valid PSBT + Fee OK", "Valid PSBT + Fee Bad", "Failed to generate"],
                range=["#2ecc71", "#e74c3c", "#bdc3c7"],
            ),
            title="Status",
        ),
        tooltip=["model_full", "status", "count", "total"],
    ).properties(title="PSBT Generation Success and Fee Sanity by Model", width=760, height=420)


def chart_open_closed_box(v2_df: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_v2_scores_dataframe(v2_df)
    if df.empty or "model_short" not in df.columns:
        return None
    category_map = {
        "anthropic_opus46": "closed-source", "anthropic_opus": "closed-source",
        "anthropic_sonnet": "closed-source", "google": "closed-source",
        "openai": "closed-source", "openai_pro": "closed-source",
        "deepseek": "open-source", "llama": "open-source",
        "mistral": "open-source", "qwen": "open-source",
    }
    if "category" not in df.columns:
        df["category"] = df["model_short"].map(category_map).fillna("unknown")
    else:
        df["category"] = df["category"].fillna(df["model_short"].map(category_map)).fillna("unknown")
    df = df[(df["fee_sanity_ok"] == 1) & df["v2_overall_score"].notna()]
    if df.empty:
        return None
    return alt.Chart(df).mark_boxplot(size=60).encode(
        x=alt.X("category:N", title=None, sort=["closed-source", "open-source", "unknown"]),
        y=alt.Y("v2_overall_score:Q", title="Privacy score"),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
            legend=None,
        ),
        tooltip=["category", alt.Tooltip("v2_overall_score:Q", format=".1f")],
    ).properties(title="Open-Source vs Closed-Source Privacy Performance", width=520, height=420)


def chart_score_distribution_by_prompt(v2_df: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_v2_scores_dataframe(v2_df)
    if df.empty or not _has_columns(df, ["prompt_label", "fee_sanity_ok", "v2_overall_score"]):
        return None
    df = df[(df["fee_sanity_ok"] == 1) & df["v2_overall_score"].notna()].copy()
    if df.empty:
        return None
    base = alt.Chart(df).encode(
        x=alt.X("prompt_label:N", sort=PROMPT_LABEL_ORDER, title="Prompt strategy"),
        y=alt.Y("v2_overall_score:Q", title="Privacy score"),
    )
    boxes = base.mark_boxplot(size=55, extent="min-max", color="#3B7DD8")
    points = base.transform_calculate(
        jitter="random() * 2 - 1"
    ).mark_circle(size=28, opacity=0.35, color="black").encode(
        xOffset=alt.XOffset("jitter:Q", scale=alt.Scale(domain=[-1, 1], range=[-24, 24])),
        tooltip=["prompt_label", alt.Tooltip("v2_overall_score:Q", format=".1f")],
    )
    return (boxes + points).properties(title="Privacy Score Distribution by Prompt Strategy", width=760, height=420)


def chart_fee_rate_distribution(v2_df: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_v2_scores_dataframe(v2_df)
    if df.empty or not _has_columns(df, ["fee_rate_sat_vb", "fee_sanity_ok"]):
        return None
    df = df.dropna(subset=["fee_rate_sat_vb"])
    if df.empty:
        return None
    df = df[df["fee_rate_sat_vb"] > 0].copy()
    df["fee_status"] = df["fee_sanity_ok"].map({1: "Fee-Sane", 0: "Fee-Insane"}).fillna("Unknown")
    return alt.Chart(df).mark_bar(opacity=0.75).encode(
        x=alt.X(
            "fee_rate_sat_vb:Q",
            bin=alt.Bin(maxbins=35),
            scale=alt.Scale(type="log"),
            title="Fee rate (sat/vB, log scale)",
        ),
        y=alt.Y("count():Q", title="Count"),
        color=alt.Color(
            "fee_status:N",
            scale=alt.Scale(domain=["Fee-Sane", "Fee-Insane", "Unknown"], range=["#2ecc71", "#e74c3c", "#808080"]),
        ),
        tooltip=["fee_status", "count()"],
    ).properties(title="Fee Rate Distribution: Normal vs Astronomical", width=760, height=420)


def chart_execution_time_vs_score(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    if df.empty or not _has_columns(df, ["model_full", "prompt_label", "avg_v2_score_fee_filtered", "avg_execution_time_seconds"]):
        return None
    df = df.dropna(subset=["avg_v2_score_fee_filtered", "avg_execution_time_seconds"])
    if df.empty:
        return None
    return alt.Chart(df).mark_circle(size=110, opacity=0.8).encode(
        x=alt.X("avg_execution_time_seconds:Q", title="Average execution time (seconds)"),
        y=alt.Y("avg_v2_score_fee_filtered:Q", title="Privacy score", scale=alt.Scale(domain=[40, 100])),
        color=alt.Color(
            "model_full:N",
            sort=model_sort_order(df["model_full"].unique()),
            title="Model",
        ),
        shape=alt.Shape("prompt_label:N", sort=PROMPT_LABEL_ORDER, title="Prompt"),
        tooltip=[
            "model_full", "prompt_label",
            alt.Tooltip("avg_execution_time_seconds:Q", format=".1f"),
            alt.Tooltip("avg_v2_score_fee_filtered:Q", format=".1f"),
        ],
    ).properties(title="Execution Time vs Privacy Score", width=760, height=460)


def chart_top_model_subscores(agg: pd.DataFrame) -> Optional[alt.Chart]:
    df = prepare_aggregated_dataframe(agg)
    needed = ["avg_clustering", "avg_change_detection", "avg_fingerprinting"]
    if df.empty or not _has_columns(df, ["model_full", "avg_v2_score_fee_filtered", *needed]):
        return None
    df = df.dropna(subset=["avg_v2_score_fee_filtered"])
    if df.empty:
        return None
    best_idx = df.groupby("model_full")["avg_v2_score_fee_filtered"].idxmax()
    top = df.loc[best_idx].sort_values("avg_v2_score_fee_filtered", ascending=False).head(3)
    long = top.melt(
        id_vars=["model_full", "avg_v2_score_fee_filtered"],
        value_vars=["avg_clustering", "avg_change_detection", "avg_fingerprinting", "avg_v2_score_fee_filtered"],
        var_name="subscore",
        value_name="score",
    )
    long["subscore"] = long["subscore"].map({
        "avg_clustering": "Clustering",
        "avg_change_detection": "Change detection",
        "avg_fingerprinting": "Fingerprinting",
        "avg_v2_score_fee_filtered": "Overall",
    })
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("subscore:N", title=None),
        y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 105])),
        color=alt.Color("model_full:N", title="Model"),
        xOffset="model_full:N",
        tooltip=["model_full", "subscore", alt.Tooltip("score:Q", format=".1f")],
    ).properties(title="Privacy Sub-Score Profile of Top Models", width=760, height=420)


def build_paper_chart(chart_name: str, agg: pd.DataFrame, v2_df: pd.DataFrame) -> Optional[alt.Chart]:
    """Return the requested paper-style Altair chart."""
    builders = {
        "Score heatmap": lambda: chart_score_heatmap(agg),
        "Fee insanity heatmap": lambda: chart_fee_insanity_heatmap(agg),
        "Best model scores": lambda: chart_best_model_scores(agg),
        "Top 3 model scores": lambda: chart_top3_model_scores(agg),
        "Prompt strategy effect": lambda: chart_prompt_strategy_effect(agg),
        "Success and fee sanity": lambda: chart_success_fee_sanity(agg),
        "Open-source vs closed-source": lambda: chart_open_closed_box(v2_df),
        "Score distribution by prompt": lambda: chart_score_distribution_by_prompt(v2_df),
        "Fee rate distribution": lambda: chart_fee_rate_distribution(v2_df),
        "Execution time vs score": lambda: chart_execution_time_vs_score(agg),
        "Top model sub-scores": lambda: chart_top_model_subscores(agg),
    }
    builder = builders.get(chart_name)
    return builder() if builder else None


def load_phase12_chart_dataframe(csv_paths: Optional[Iterable] = None) -> pd.DataFrame:
    """Load the canonical 2026 prompt corpus for interactive charts."""
    return normalize_phase12_results(csv_paths)


def _phase12_fee_ok(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "fee_ok_bool" not in df.columns:
        return pd.DataFrame()
    return df[df["fee_ok_bool"] == True].copy()  # noqa: E712


def _amount_sort() -> List[float]:
    return [float(value) for value in AMOUNT_ORDER]


def _model_sort(df: pd.DataFrame) -> List[str]:
    preferred = [
        "gpt-5.4",
        "claude-sonnet-4-6",
        "google/gemini-3.1-pro-preview",
        "google/gemma-4-31b-it",
        "z-ai/glm-5.1",
    ]
    seen = set()
    result: List[str] = []
    for model in preferred + sorted(str(model) for model in df.get("llm_model", pd.Series(dtype=str)).dropna().unique()):
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _phase12_style(chart):
    """Apply web-friendly defaults so 2026 prompt charts fit the Streamlit page."""
    return (
        chart.configure_axis(
            labelLimit=PHASE12_AXIS_LABEL_LIMIT,
            titleLimit=PHASE12_AXIS_LABEL_LIMIT,
        )
        .configure_legend(
            labelLimit=PHASE12_LEGEND_LABEL_LIMIT,
            titleLimit=PHASE12_LEGEND_LABEL_LIMIT,
            orient="bottom",
            columns=3,
        )
        .configure_title(anchor="start")
        .configure_view(strokeWidth=0)
    )


def chart_phase12_wallet_pct_vs_score(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _phase12_fee_ok(df)
    if data.empty:
        return None
    grouped = data.groupby(["prompt_label", "amount_pct"], dropna=False).agg(
        mean_score=("usable_score", "mean"),
        n=("usable_score", "count"),
    ).reset_index()
    chart = alt.Chart(grouped).mark_line(point=True).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("mean_score:Q", title="Mean fee-sane privacy score", scale=alt.Scale(domain=[65, 100])),
        color=alt.Color("prompt_label:N", title="Prompt strategy"),
        tooltip=[
            "prompt_label", "amount_pct",
            alt.Tooltip("mean_score:Q", format=".2f"), "n",
        ],
    ).properties(title="Wallet Percentage vs Privacy Score by Prompt", width=PHASE12_CHART_WIDTH, height=310)
    return _phase12_style(chart)


def chart_phase12_model_amount_heatmaps(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _phase12_fee_ok(df)
    if data.empty:
        return None
    data = data[data["prompt_type"].isin(["basic", "privacy_simple", "multiturn_detailed"])].copy()
    data = data[
        ((data["phase"] == "phase-1") & data["prompt_type"].isin(["basic", "privacy_simple"]))
        | ((data["phase"] == "phase-2") & (data["prompt_type"] == "multiturn_detailed"))
    ]
    grouped = data.groupby(["prompt_label", "llm_model", "amount_pct"], dropna=False)["usable_score"].mean().reset_index()
    if grouped.empty:
        return None
    panel_order = [
        "Basic",
        "Privacy Simple",
        "Multi-turn Detailed",
    ]
    base = alt.Chart(grouped).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y(
            "llm_model:N",
            sort=_model_sort(grouped),
            title=None,
            axis=alt.Axis(labelLimit=PHASE12_AXIS_LABEL_LIMIT),
        ),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("usable_score:Q", title="Privacy score", scale=alt.Scale(scheme="yellowgreenblue", domain=[65, 100])),
        tooltip=["prompt_label", "llm_model", "amount_pct", alt.Tooltip("usable_score:Q", format=".1f")],
    )
    text = base.mark_text(fontSize=11).encode(text=alt.Text("usable_score:Q", format=".1f"))
    chart = (heat + text).properties(width=PHASE12_CHART_WIDTH, height=120).facet(
        row=alt.Row(
            "prompt_label:N",
            sort=panel_order,
            title=None,
            header=alt.Header(labelOrient="top", labelFontSize=13, labelFontWeight="bold"),
        )
    ).properties(title="Model x Wallet Percentage Heatmaps by Prompt")
    return _phase12_style(chart)


def chart_phase12_uplift(df: pd.DataFrame) -> Optional[alt.Chart]:
    uplift = build_phase12_uplift_summary(df)
    if uplift.empty:
        return None
    grouped = uplift.groupby(["llm_model", "amount_pct"], dropna=False)["prompt_delta_score"].mean().reset_index()
    base = alt.Chart(grouped).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("llm_model:N", sort=_model_sort(grouped), title=None),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("prompt_delta_score:Q", title="Prompt delta", scale=alt.Scale(scheme="redblue", domainMid=0)),
        tooltip=["llm_model", "amount_pct", alt.Tooltip("prompt_delta_score:Q", format="+.2f")],
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("prompt_delta_score:Q", format="+.1f"))
    chart_height = max(240, min(340, grouped["llm_model"].nunique() * 52))
    chart = (heat + text).properties(
        title="Prompt Delta: Multi-turn Detailed vs Privacy Simple",
        width=PHASE12_DETAIL_CHART_WIDTH,
        height=chart_height,
    )
    return _phase12_style(chart)


def chart_phase12_reliability(df: pd.DataFrame) -> Optional[alt.Chart]:
    summary = build_phase12_reliability_summary(df)
    if summary.empty:
        return None
    long = summary.melt(
        id_vars=["prompt_label", "execution_lot_label", "llm_model", "category", "attempts"],
        value_vars=["fee_ok_rate", "fee_bad_rate", "failed_rate"],
        var_name="status",
        value_name="rate",
    )
    long["status"] = long["status"].map({
        "fee_ok_rate": "Fee-ok PSBT",
        "fee_bad_rate": "Fee-bad PSBT",
        "failed_rate": "No PSBT",
    })
    long["rate_pct"] = long["rate"] * 100
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("rate_pct:Q", title="Share of runs (%)", stack="zero", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y(
            "llm_model:N",
            sort=_model_sort(summary),
            title=None,
            axis=alt.Axis(labelLimit=PHASE12_AXIS_LABEL_LIMIT),
        ),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["Fee-ok PSBT", "Fee-bad PSBT", "No PSBT"], range=["#2ca25f", "#de2d26", "#bdbdbd"]),
            title="Status",
        ),
        row=alt.Row(
            "prompt_label:N",
            sort=["Basic", "Privacy Simple", "Multi-turn Detailed"],
            title=None,
            header=alt.Header(labelOrient="top", labelFontSize=13, labelFontWeight="bold"),
        ),
        tooltip=[
            "prompt_label", "execution_lot_label", "llm_model", "status",
            alt.Tooltip("rate_pct:Q", format=".1f"),
            "attempts",
        ],
    ).properties(title="Reliability by Model and Prompt", width=PHASE12_CHART_WIDTH, height=155)
    return _phase12_style(chart)


def chart_phase12_structure_difficulty(df: pd.DataFrame) -> Optional[alt.Chart]:
    summary = build_phase12_amount_summary(df)
    if summary.empty:
        return None
    collapsed = summary.groupby(["prompt_label", "amount_pct"], dropna=False).agg(
        mean_inputs=("mean_inputs", "mean"),
        mean_outputs=("mean_outputs", "mean"),
        mean_score=("mean_score", "mean"),
    ).reset_index()
    long = collapsed.melt(
        id_vars=["prompt_label", "amount_pct", "mean_score"],
        value_vars=["mean_inputs", "mean_outputs"],
        var_name="metric",
        value_name="count",
    )
    charts = []
    for prompt_label in [label for label in ["Basic", "Privacy Simple", "Multi-turn Detailed"] if label in set(collapsed["prompt_label"])]:
        prompt_long = long[long["prompt_label"] == prompt_label]
        prompt_scores = collapsed[collapsed["prompt_label"] == prompt_label]
        bars = alt.Chart(prompt_long).mark_bar(opacity=0.75).encode(
            x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
            y=alt.Y("count:Q", title="Mean inputs / outputs"),
            color=alt.Color("metric:N", title="Structure"),
            xOffset="metric:N",
            tooltip=["prompt_label", "amount_pct", "metric", alt.Tooltip("count:Q", format=".2f")],
        )
        score = alt.Chart(prompt_scores).mark_line(point=True, color="#cb181d", strokeWidth=2).encode(
            x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
            y=alt.Y("mean_score:Q", title="Mean fee-sane score"),
            tooltip=["prompt_label", "amount_pct", alt.Tooltip("mean_score:Q", format=".2f")],
        )
        charts.append(
            (bars + score)
            .resolve_scale(y="independent")
            .properties(title=prompt_label, width=PHASE12_CHART_WIDTH, height=220)
        )
    if not charts:
        return None
    return _phase12_style(
        alt.vconcat(*charts, spacing=24).properties(title="Structural Difficulty by Wallet Percentage and Prompt")
    )


def chart_phase12_subscore_tradeoff(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _phase12_fee_ok(df)
    if data.empty:
        return None
    summary = data.groupby(["prompt_label", "amount_pct"], dropna=False).agg(
        clustering=("score_clustering", "mean"),
        change_detection=("score_change_detection", "mean"),
        fingerprinting=("score_fingerprinting", "mean"),
    ).reset_index()
    long = summary.melt(
        id_vars=["prompt_label", "amount_pct"],
        value_vars=["clustering", "change_detection", "fingerprinting"],
        var_name="subscore",
        value_name="score",
    )
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("score:Q", title="Sub-score", scale=alt.Scale(domain=[80, 100])),
        color=alt.Color("subscore:N", title="Sub-score"),
        strokeDash=alt.StrokeDash("prompt_label:N", title="Prompt"),
        tooltip=["prompt_label", "amount_pct", "subscore", alt.Tooltip("score:Q", format=".2f")],
    ).properties(title="Sub-score Tradeoff by Wallet Percentage and Prompt", width=PHASE12_CHART_WIDTH, height=310)
    return _phase12_style(chart)


def chart_phase12_fee_tradeoff(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = df[(df["psbt_generated_bool"] == True) & df["fee_rate_sat_vb"].notna() & df["privacy_score"].notna()].copy()  # noqa: E712
    if data.empty:
        return None
    data["fee_status"] = data["fee_ok_bool"].map({True: "Fee-ok", False: "Fee-bad"})
    chart = alt.Chart(data).mark_circle(size=105, opacity=0.78).encode(
        x=alt.X("fee_rate_sat_vb:Q", scale=alt.Scale(type="log"), title="Fee rate (sat/vB, log)"),
        y=alt.Y("privacy_score:Q", title="Privacy score", scale=alt.Scale(domain=[45, 105])),
        color=alt.Color("prompt_label:N", title="Prompt"),
        shape=alt.Shape("fee_status:N", title="Fee status"),
        size=alt.Size("amount_pct:Q", title="Wallet target (%)"),
        tooltip=["prompt_label", "execution_lot_label", "llm_model", "amount_pct", "fee_status", alt.Tooltip("fee_rate_sat_vb:Q", format=".2f"), alt.Tooltip("privacy_score:Q", format=".1f")],
    ).properties(title="Fee Tradeoff by Prompt: Fee Rate vs Privacy Score", width=PHASE12_DETAIL_CHART_WIDTH, height=420)
    return _phase12_style(chart)


def _phase12_pareto(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values("estimated_cost_usd")
    rows = []
    best = -1.0
    for _, row in ordered.iterrows():
        score = row.get("usable_score")
        if pd.notna(score) and score > best:
            rows.append(row)
            best = score
    return pd.DataFrame(rows)


def chart_phase12_cost_vs_score(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = df[(df["fee_ok_bool"] == True) & df["estimated_cost_usd"].notna()].copy()  # noqa: E712
    if data.empty:
        return None
    points = alt.Chart(data).mark_circle(size=105, opacity=0.68).encode(
        x=alt.X("estimated_cost_usd:Q", title="Estimated cost per run (USD)"),
        y=alt.Y("usable_score:Q", title="Fee-sane privacy score", scale=alt.Scale(domain=[65, 105])),
        color=alt.Color("llm_model:N", sort=_model_sort(data), title="Model"),
        shape=alt.Shape("cost_source:N", title="Cost source"),
        tooltip=["llm_model", "phase_label", "prompt_label", "amount_pct", "cost_source", alt.Tooltip("estimated_cost_usd:Q", format=".5f"), alt.Tooltip("usable_score:Q", format=".1f")],
    )
    frontier = _phase12_pareto(data)
    line = alt.Chart(frontier).mark_line(point=True, color="black", strokeWidth=2).encode(
        x="estimated_cost_usd:Q",
        y="usable_score:Q",
        tooltip=["llm_model", alt.Tooltip("estimated_cost_usd:Q", format=".5f"), alt.Tooltip("usable_score:Q", format=".1f")],
    )
    chart = (points + line).properties(title="Estimated Cost vs Usable Privacy Score", width=PHASE12_DETAIL_CHART_WIDTH, height=420)
    return _phase12_style(chart)


def chart_phase12_cost_efficiency(df: pd.DataFrame) -> Optional[alt.Chart]:
    summary = build_phase12_cost_efficiency_summary(df)
    if summary.empty:
        return None
    summary = summary.sort_values("score_per_estimated_dollar", ascending=False)
    chart_height = max(260, min(360, len(summary) * 48))
    chart = alt.Chart(summary).mark_bar(size=26).encode(
        x=alt.X("score_per_estimated_dollar:Q", title="Mean score per estimated USD", scale=alt.Scale(type="log")),
        y=alt.Y(
            "llm_model:N",
            sort=summary["llm_model"].tolist(),
            title=None,
            axis=alt.Axis(labelLimit=PHASE12_AXIS_LABEL_LIMIT),
        ),
        color=alt.Color("category:N", title="Category"),
        tooltip=[
            "llm_model", "category", "cost_source",
            alt.Tooltip("mean_score:Q", format=".2f"),
            alt.Tooltip("mean_cost_usd:Q", format=".5f"),
            alt.Tooltip("score_per_estimated_dollar:Q", format=".1f"),
        ],
    ).properties(title="Estimated Cost-Efficiency by Model", width=PHASE12_DETAIL_CHART_WIDTH, height=chart_height)
    return _phase12_style(chart)


def chart_phase12_temperature_effect(df: pd.DataFrame) -> Optional[alt.Chart]:
    fee_ok = _phase12_fee_ok(df)
    if fee_ok.empty:
        return None
    score = fee_ok.groupby(["prompt_label", "temperature"], dropna=False)["usable_score"].mean().reset_index()
    reliability = df.groupby(["prompt_label", "temperature"], dropna=False).agg(
        attempts=("experiment_id", "count"),
        fee_ok=("fee_ok_bool", "sum"),
    ).reset_index()
    reliability["fee_ok_rate"] = reliability["fee_ok"] / reliability["attempts"]

    prompt_color = alt.Color(
        "prompt_label:N",
        sort=["Basic", "Privacy Simple", "Multi-turn Detailed"],
        title="Prompt",
    )
    prompt_offset = alt.XOffset(
        "prompt_label:N",
        sort=["Basic", "Privacy Simple", "Multi-turn Detailed"],
    )

    score_chart = alt.Chart(score).mark_bar(size=18).encode(
        x=alt.X("temperature:O", title="Temperature"),
        y=alt.Y(
            "usable_score:Q",
            title="Mean fee-sane privacy score",
            scale=alt.Scale(domain=[65, 90]),
        ),
        color=prompt_color,
        xOffset=prompt_offset,
        tooltip=[
            "prompt_label",
            "temperature",
            alt.Tooltip("usable_score:Q", title="Mean score", format=".2f"),
        ],
    ).properties(title="Score by Temperature", width=PHASE12_CHART_WIDTH, height=230)

    reliability_chart = alt.Chart(reliability).mark_bar(size=18).encode(
        x=alt.X("temperature:O", title="Temperature"),
        y=alt.Y(
            "fee_ok_rate:Q",
            title="Fee-ok run rate",
            axis=alt.Axis(format=".0%"),
            scale=alt.Scale(domain=[0, 1.05]),
        ),
        color=alt.Color(
            "prompt_label:N",
            sort=["Basic", "Privacy Simple", "Multi-turn Detailed"],
            title="Prompt",
            legend=None,
        ),
        xOffset=prompt_offset,
        tooltip=[
            "prompt_label",
            "temperature",
            "attempts",
            alt.Tooltip("fee_ok_rate:Q", title="Fee-ok rate", format=".1%"),
        ],
    ).properties(title="Reliability by Temperature", width=PHASE12_CHART_WIDTH, height=230)

    chart = alt.vconcat(score_chart, reliability_chart, spacing=24).properties(
        title="Temperature Effect on Score and Reliability by Prompt"
    )
    return _phase12_style(chart)


def build_phase12_chart(chart_name: str, df: pd.DataFrame) -> Optional[alt.Chart]:
    builders = {
        "Wallet % vs score": lambda: chart_phase12_wallet_pct_vs_score(df),
        "Model amount heatmaps": lambda: chart_phase12_model_amount_heatmaps(df),
        "Prompt delta": lambda: chart_phase12_uplift(df),
        "Phase 2 uplift": lambda: chart_phase12_uplift(df),
        "Reliability by prompt": lambda: chart_phase12_reliability(df),
        "Reliability by model": lambda: chart_phase12_reliability(df),
        "Structure difficulty": lambda: chart_phase12_structure_difficulty(df),
        "Sub-score tradeoff": lambda: chart_phase12_subscore_tradeoff(df),
        "Fee tradeoff": lambda: chart_phase12_fee_tradeoff(df),
        "Cost vs score": lambda: chart_phase12_cost_vs_score(df),
        "Cost efficiency": lambda: chart_phase12_cost_efficiency(df),
        "Temperature effect": lambda: chart_phase12_temperature_effect(df),
    }
    builder = builders.get(chart_name)
    return builder() if builder else None


def load_wallet_baseline_chart_dataframe(csv_paths: Optional[Iterable] = None) -> pd.DataFrame:
    """Load all wallet baseline lots as one normalized table."""
    return normalize_wallet_baseline_results(csv_paths)


def _wallet_label_sort() -> List[str]:
    labels = []
    for wallet in WALLET_ORDER:
        label = WALLET_LABELS.get(wallet, wallet.replace("-", " ").title())
        labels.append(label)
    return labels


def _wallet_fee_ok_latest(df: pd.DataFrame) -> pd.DataFrame:
    latest = select_latest_successful_by_wallet_amount(df)
    if latest.empty or "fee_ok_bool" not in latest.columns:
        return latest.iloc[0:0].copy()
    return latest[latest["fee_ok_bool"] == True].copy()  # noqa: E712


def _wallet_style(chart):
    return chart.configure_axis(labelFontSize=11, titleFontSize=12).configure_title(
        fontSize=15,
        anchor="start",
    ).configure_legend(labelLimit=PHASE12_LEGEND_LABEL_LIMIT)


def chart_wallet_score_by_amount(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _wallet_fee_ok_latest(df)
    if data.empty:
        return None
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("privacy_score:Q", title="Fee-sane privacy score", scale=alt.Scale(domain=[50, 100])),
        color=alt.Color("wallet_label:N", sort=_wallet_label_sort(), title="Wallet"),
        tooltip=[
            "wallet_label", "amount_pct",
            alt.Tooltip("privacy_score:Q", format=".1f"),
            alt.Tooltip("fee_sats:Q", format=",.0f"),
            alt.Tooltip("num_inputs:Q", format=".0f"),
            alt.Tooltip("num_outputs:Q", format=".0f"),
        ],
    ).properties(title="Wallet Score by Amount", width=PHASE12_CHART_WIDTH, height=320)
    return _wallet_style(chart)


def chart_wallet_amount_heatmap(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _wallet_fee_ok_latest(df)
    if data.empty:
        return None
    base = alt.Chart(data).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("wallet_label:N", sort=_wallet_label_sort(), title=None),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("privacy_score:Q", title="Privacy score", scale=alt.Scale(scheme="yellowgreenblue", domain=[50, 100])),
        tooltip=[
            "wallet_label", "amount_pct",
            alt.Tooltip("privacy_score:Q", format=".1f"),
            alt.Tooltip("fee_rate_sat_vb:Q", format=".2f"),
        ],
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("privacy_score:Q", format=".0f"))
    chart = (heat + text).properties(title="Wallet x Amount Score Heatmap", width=PHASE12_CHART_WIDTH, height=190)
    return _wallet_style(chart)


def chart_wallet_vs_agents(df: pd.DataFrame) -> Optional[alt.Chart]:
    comparison = build_wallet_agent_amount_comparison(df)
    if comparison.empty:
        return None
    wallet_rows = comparison.rename(columns={"privacy_score": "score"}).copy()
    wallet_rows["series"] = wallet_rows["wallet_label"]
    wallet_rows["kind"] = "Wallet"
    agent_rows = comparison.drop_duplicates("amount_pct").copy()
    agent_rows["score"] = agent_rows["agent_mean_privacy_score"]
    agent_rows["series"] = "Agent corpus mean"
    agent_rows["kind"] = "Agent"
    long = pd.concat(
        [
            wallet_rows[["amount_pct", "score", "series", "kind"]],
            agent_rows[["amount_pct", "score", "series", "kind"]],
        ],
        ignore_index=True,
    ).dropna(subset=["score"])
    if long.empty:
        return None
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("score:Q", title="Privacy score", scale=alt.Scale(domain=[50, 100])),
        color=alt.Color("series:N", title="Series"),
        strokeDash=alt.StrokeDash("kind:N", title="Type"),
        tooltip=["series", "kind", "amount_pct", alt.Tooltip("score:Q", format=".2f")],
    ).properties(title="Wallet Baselines vs Agent Corpus Mean", width=PHASE12_DETAIL_CHART_WIDTH, height=360)
    return _wallet_style(chart)


def chart_wallet_delta_vs_agents(df: pd.DataFrame) -> Optional[alt.Chart]:
    comparison = build_wallet_agent_amount_comparison(df)
    if comparison.empty or "wallet_minus_agent_mean" not in comparison.columns:
        return None
    data = comparison.dropna(subset=["wallet_minus_agent_mean"]).copy()
    if data.empty:
        return None
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("wallet_minus_agent_mean:Q", title="Wallet score - agent mean"),
        color=alt.Color(
            "wallet_minus_agent_mean:Q",
            scale=alt.Scale(scheme="redblue", domainMid=0),
            title="Delta",
        ),
        column=alt.Column("wallet_label:N", sort=_wallet_label_sort(), title=None),
        tooltip=[
            "wallet_label", "amount_pct",
            alt.Tooltip("privacy_score:Q", title="Wallet score", format=".1f"),
            alt.Tooltip("agent_mean_privacy_score:Q", title="Agent mean", format=".2f"),
            alt.Tooltip("wallet_minus_agent_mean:Q", title="Delta", format="+.2f"),
        ],
    ).properties(title="Wallet Delta vs Agent Corpus Mean", width=150, height=260)
    return _wallet_style(chart)


def chart_wallet_reliability(df: pd.DataFrame) -> Optional[alt.Chart]:
    summary = build_wallet_reliability_summary(df)
    if summary.empty:
        return None
    long = summary.melt(
        id_vars=["wallet", "wallet_label", "attempts"],
        value_vars=["fee_ok", "fee_bad", "failed"],
        var_name="status",
        value_name="count",
    )
    long["status"] = long["status"].map({
        "fee_ok": "Fee-ok PSBT",
        "fee_bad": "Fee-bad PSBT",
        "failed": "No PSBT",
    })
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("count:Q", title="Runs", stack="zero"),
        y=alt.Y("wallet_label:N", sort=_wallet_label_sort(), title=None),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["Fee-ok PSBT", "Fee-bad PSBT", "No PSBT"], range=["#2ca25f", "#de2d26", "#bdbdbd"]),
            title="Status",
        ),
        tooltip=["wallet_label", "status", "count", "attempts"],
    ).properties(title="Wallet Baseline Reliability", width=PHASE12_CHART_WIDTH, height=220)
    return _wallet_style(chart)


def chart_wallet_structure_difficulty(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _wallet_fee_ok_latest(df)
    if data.empty:
        return None
    long = data.melt(
        id_vars=["wallet_label", "amount_pct", "privacy_score"],
        value_vars=["num_inputs", "num_outputs"],
        var_name="metric",
        value_name="count",
    ).dropna(subset=["count"])
    if long.empty:
        return None
    charts = []
    for label in [label for label in _wallet_label_sort() if label in set(data["wallet_label"])]:
        wallet_long = long[long["wallet_label"] == label]
        wallet_scores = data[data["wallet_label"] == label]
        bars = alt.Chart(wallet_long).mark_bar(opacity=0.72).encode(
            x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
            y=alt.Y("count:Q", title="Inputs / outputs"),
            color=alt.Color("metric:N", title="Structure"),
            xOffset="metric:N",
            tooltip=["wallet_label", "amount_pct", "metric", alt.Tooltip("count:Q", format=".0f")],
        )
        scores = alt.Chart(wallet_scores).mark_line(point=True, color="#cb181d", strokeWidth=2).encode(
            x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
            y=alt.Y("privacy_score:Q", title="Privacy score"),
            tooltip=["wallet_label", "amount_pct", alt.Tooltip("privacy_score:Q", format=".1f")],
        )
        charts.append(
            (bars + scores)
            .resolve_scale(y="independent")
            .properties(title=label, width=PHASE12_CHART_WIDTH, height=170)
        )
    if not charts:
        return None
    return _wallet_style(
        alt.vconcat(*charts, spacing=22).properties(title="Wallet Structure Difficulty")
    )


def chart_wallet_fee_tradeoff(df: pd.DataFrame) -> Optional[alt.Chart]:
    latest = select_latest_successful_by_wallet_amount(df)
    data = latest[(latest["psbt_generated_bool"] == True) & latest["fee_rate_sat_vb"].notna() & latest["privacy_score"].notna()].copy()  # noqa: E712
    if data.empty:
        return None
    data["fee_status"] = data["fee_ok_bool"].map({True: "Fee-ok", False: "Fee-bad"})
    chart = alt.Chart(data).mark_circle(size=115, opacity=0.78).encode(
        x=alt.X("fee_rate_sat_vb:Q", scale=alt.Scale(type="log"), title="Fee rate (sat/vB, log)"),
        y=alt.Y("privacy_score:Q", title="Privacy score", scale=alt.Scale(domain=[45, 105])),
        color=alt.Color("wallet_label:N", sort=_wallet_label_sort(), title="Wallet"),
        shape=alt.Shape("fee_status:N", title="Fee status"),
        size=alt.Size("amount_pct:Q", title="Wallet target (%)"),
        tooltip=[
            "wallet_label", "amount_pct", "fee_status",
            alt.Tooltip("fee_rate_sat_vb:Q", format=".2f"),
            alt.Tooltip("fee_sats:Q", format=",.0f"),
            alt.Tooltip("privacy_score:Q", format=".1f"),
        ],
    ).properties(title="Wallet Fee Tradeoff", width=PHASE12_DETAIL_CHART_WIDTH, height=420)
    return _wallet_style(chart)


def chart_wallet_subscore_tradeoff(df: pd.DataFrame) -> Optional[alt.Chart]:
    data = _wallet_fee_ok_latest(df)
    score_columns = ["score_clustering", "score_change_detection", "score_fingerprinting"]
    if data.empty or not _has_columns(data, score_columns):
        return None
    long = data.melt(
        id_vars=["wallet_label", "amount_pct"],
        value_vars=score_columns,
        var_name="subscore",
        value_name="score",
    ).dropna(subset=["score"])
    if long.empty:
        return None
    long["subscore"] = long["subscore"].map({
        "score_clustering": "Clustering",
        "score_change_detection": "Change detection",
        "score_fingerprinting": "Fingerprinting",
    })
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("amount_pct:O", sort=_amount_sort(), title="Wallet target (%)"),
        y=alt.Y("score:Q", title="Sub-score", scale=alt.Scale(domain=[40, 100])),
        color=alt.Color("subscore:N", title="Sub-score"),
        strokeDash=alt.StrokeDash("wallet_label:N", sort=_wallet_label_sort(), title="Wallet"),
        tooltip=["wallet_label", "amount_pct", "subscore", alt.Tooltip("score:Q", format=".1f")],
    ).properties(title="Wallet Sub-score Tradeoff", width=PHASE12_DETAIL_CHART_WIDTH, height=360)
    return _wallet_style(chart)


def build_wallet_baseline_chart(chart_name: str, df: pd.DataFrame) -> Optional[alt.Chart]:
    builders = {
        "Wallet score by amount": lambda: chart_wallet_score_by_amount(df),
        "Wallet amount heatmap": lambda: chart_wallet_amount_heatmap(df),
        "Wallet vs agents": lambda: chart_wallet_vs_agents(df),
        "Wallet delta vs agents": lambda: chart_wallet_delta_vs_agents(df),
        "Wallet reliability": lambda: chart_wallet_reliability(df),
        "Wallet structure difficulty": lambda: chart_wallet_structure_difficulty(df),
        "Wallet fee tradeoff": lambda: chart_wallet_fee_tradeoff(df),
        "Wallet sub-score tradeoff": lambda: chart_wallet_subscore_tradeoff(df),
    }
    builder = builders.get(chart_name)
    return builder() if builder else None
