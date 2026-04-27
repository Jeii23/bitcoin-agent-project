import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_manager import ExperimentManager
from experiment_runner import ExperimentResult, ExperimentRunner, create_filter
from paper_charts import (
    PAPER_CHART_OPTIONS,
    aggregate_current_results,
    build_paper_chart,
    current_results_v2_scores,
    load_paper_chart_sources,
    prepare_aggregated_dataframe,
    prepare_v2_scores_dataframe,
)
from result_utils import (
    arrow_safe_dataframe,
    display_columns,
    flatten_privacy_breakdown,
    load_many_results_dataframes,
    load_results_dataframe,
)
from streamlit.testing.v1 import AppTest
from web_ui import (
    all_tags_from_dataframe,
    experiments_dataframe,
    filter_run_experiments,
    format_result_path,
    load_result_files,
    safe_prompt_mode,
    sort_results_for_display,
    split_filter_values,
)


def test_legacy_csv_strategy_and_amount_inference():
    manager = ExperimentManager(EXPERIMENTS_DIR / "experiments_paper_legacy.csv")
    rows = {row["id"]: row for row in manager.read_experiments()}

    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_basic"]).strategy == "basic"
    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_privacy_simple"]).strategy == "privacy-simple"
    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_multiturn_simple"]).strategy == "multiturn-simple"
    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_multiturn_detailed"]).strategy == "multiturn-detailed"
    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_privacy_detailed"]).strategy == "privacy-detailed"
    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_privacy_detailed"]).amount_btc == 3.0
    assert manager.parse_csv_row_to_meta(rows["exp_anthropic_opus_privacy_detailed"]).prompt_mode == "custom"


def test_amount_inference_from_satoshis_prompt():
    row = {"user_prompt": "Fes-me una PSBT de 500000 satoshis (≈ 0.005000 BTC)"}
    assert ExperimentManager.infer_amount_btc(row) == 0.005


def test_amount_display_inference_from_percentage_tags():
    row = {
        "tags": "phase-1|prompt-basic|amt-pct-30|temp-0.3|openai|gpt54|frontier",
        "user_prompt": "Fes-me una PSBT que envii aproximadament el 30% del saldo actual del wallet a una de les meves adreces",
    }
    assert ExperimentManager.infer_amount_display(row) == "30% of wallet balance"


def test_current_csv_2026_matrix_shape_and_model_selection():
    manager = ExperimentManager(EXPERIMENTS_DIR / "experiments.csv")
    rows = manager.read_experiments()

    assert len(rows) == 180
    assert {row["provider"] for row in rows} == {"openai", "anthropic", "openrouter"}
    assert all(row["provider"] != "google" for row in rows)
    assert sum(row["enabled"].lower() == "true" for row in rows) == 140
    assert sum("phase-1" in row["tags"].split("|") for row in rows) == 140
    assert sum("phase-2" in row["tags"].split("|") for row in rows) == 40
    assert sum("prompt-basic" in row["tags"].split("|") for row in rows) == 70
    assert sum("prompt-privacy-simple" in row["tags"].split("|") for row in rows) == 70
    assert sum("prompt-multiturn-detailed" in row["tags"].split("|") for row in rows) == 40
    assert {row["model"] for row in rows} == {
        "gpt-5.4",
        "gpt-5.4-pro",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "google/gemini-3.1-pro-preview",
        "z-ai/glm-5.1",
        "google/gemma-4-31b-it",
    }
    assert {row["timeout_seconds"] for row in rows if row["model"] == "gpt-5.4-pro"} == {"600"}


def test_update_legacy_custom_prompt_preserves_prompt_when_structured_fields_present(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    original_prompt = "Fes-me una PSBT de 3 BTC a una de les meves adreces lo mes privada possible"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "name",
                "provider",
                "model",
                "temperature",
                "user_prompt",
                "followup_prompts",
                "repetitions",
                "timeout_seconds",
                "network",
                "tags",
                "enabled",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "id": "legacy",
            "name": "Legacy",
            "provider": "openai",
            "model": "gpt-5.2-chat-latest",
            "temperature": "1.0",
            "user_prompt": original_prompt,
            "followup_prompts": "",
            "repetitions": "3",
            "timeout_seconds": "300",
            "network": "mainnet",
            "tags": "openai|privacy-simple",
            "enabled": "true",
        })

    manager = ExperimentManager(csv_path)
    manager.update_experiment("legacy", {
        "model": "gpt-5.2-pro",
        "amount_btc": "1.0",
        "strategy": "basic",
        "prompt_mode": "custom",
    })

    updated = manager.read_experiment_by_id("legacy")
    assert updated["user_prompt"] == original_prompt
    assert updated["prompt_mode"] == "custom"
    assert updated["model"] == "gpt-5.2-pro"


def test_template_mode_regenerates_prompt_on_amount_change(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=ExperimentManager.CSV_COLUMNS,
        )
        writer.writeheader()
        writer.writerow({
            "id": "template",
            "name": "Template",
            "amount_btc": "1.0",
            "strategy": "basic",
            "prompt_mode": "template",
            "provider": "openai",
            "model": "gpt-5.2-chat-latest",
            "temperature": "1.0",
            "user_prompt": "old prompt",
            "followup_prompts": "",
            "repetitions": "1",
            "timeout_seconds": "300",
            "network": "mainnet",
            "tags": "test",
            "enabled": "true",
        })

    manager = ExperimentManager(csv_path)
    manager.update_experiment("template", {
        "amount_btc": "0.5",
        "strategy": "privacy-simple",
        "prompt_mode": "template",
    })

    updated = manager.read_experiment_by_id("template")
    assert "50000000 satoshis" in updated["user_prompt"]
    assert "privada possible" in updated["user_prompt"]
    assert updated["followup_prompts"] == ""


def test_template_mode_preserves_prompt_when_unrelated_field_changes(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    prompt = "customized template prompt that should stay"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ExperimentManager.CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "id": "template",
            "name": "Template",
            "amount_btc": "1.0",
            "strategy": "basic",
            "prompt_mode": "template",
            "provider": "openai",
            "model": "gpt-5.2-chat-latest",
            "temperature": "1.0",
            "user_prompt": prompt,
            "followup_prompts": "",
            "repetitions": "1",
            "timeout_seconds": "300",
            "network": "mainnet",
            "tags": "test",
            "enabled": "true",
        })

    manager = ExperimentManager(csv_path)
    manager.update_experiment("template", {
        "model": "gpt-5.2-pro",
        "amount_btc": "1.0",
        "strategy": "basic",
        "prompt_mode": "template",
    })

    updated = manager.read_experiment_by_id("template")
    assert updated["user_prompt"] == prompt
    assert updated["model"] == "gpt-5.2-pro"


def test_ids_filter_and_scorer_import():
    from experiment_runner import ExperimentCSVParser

    parsed = ExperimentCSVParser(EXPERIMENTS_DIR / "experiments.csv").parse()
    fn = create_filter(
        "ids:exp_openai_gpt54_basic_pct10_t03,exp_openrouter_gemini31pro_basic_pct10_t03"
    )
    assert [exp.id for exp in parsed if fn(exp)] == [
        "exp_openai_gpt54_basic_pct10_t03",
        "exp_openrouter_gemini31pro_basic_pct10_t03",
    ]
    scorer = ExperimentRunner(Path("results"))._lazy_import_scorer()
    scorer_file = PROJECT_ROOT.parent / "analysis" / "scoring" / "privacy_scorer_v2.py"
    if scorer_file.exists():
        assert callable(scorer)
    else:
        assert scorer is None


def test_experiment_runner_updates_a_single_live_result_file(tmp_path):
    runner = ExperimentRunner(tmp_path)
    runner.results.append(ExperimentResult(
        experiment_id="exp1",
        experiment_name="Experiment 1",
        repetition=1,
        timestamp="2026-04-22 10:00:00",
        llm_provider="openai",
        llm_model="gpt-test",
        llm_temperature=1.0,
        system_prompt=None,
        user_prompt="prompt 1",
        success=True,
        error_message=None,
        execution_time_seconds=1.23,
        psbt_generated=True,
        psbt_base64="cHNidP8BAHECAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/////AQEAAAAAAAAAFgAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==",
        psbt_file="results/psbts/exp1_rep1.psbt",
        privacy_score=80,
        privacy_grade="A",
        privacy_breakdown={
            "fee_sanity_ok": 1,
            "sanity_status": "ok",
            "fee_analysis": {"fee_rate_sat_vb": 2.1, "fee_sats": 321},
        },
        tags=["phase-1", "openai"],
    ))
    runner._persist_results_snapshot()

    csv_files = sorted(tmp_path.glob("experiments_*.csv"))
    json_files = sorted(tmp_path.glob("experiments_*.json"))
    assert len(csv_files) == 1
    assert len(json_files) == 1
    csv_path = csv_files[0]
    json_path = json_files[0]

    first_df = load_results_dataframe(csv_path)
    assert first_df["experiment_id"].tolist() == ["exp1"]
    assert first_df.loc[0, "fee_rate_sat_vb"] == 2.1

    runner.results.append(ExperimentResult(
        experiment_id="exp2",
        experiment_name="Experiment 2",
        repetition=1,
        timestamp="2026-04-22 10:05:00",
        llm_provider="anthropic",
        llm_model="claude-test",
        llm_temperature=0.3,
        system_prompt=None,
        user_prompt="prompt 2",
        success=False,
        error_message="timeout",
        execution_time_seconds=300.0,
        psbt_generated=False,
        psbt_base64=None,
        psbt_file=None,
        privacy_score=None,
        privacy_grade=None,
        privacy_breakdown=None,
        tags=["phase-1", "anthropic"],
    ))
    runner._persist_results_snapshot()

    assert sorted(tmp_path.glob("experiments_*.csv")) == [csv_path]
    assert sorted(tmp_path.glob("experiments_*.json")) == [json_path]

    updated_df = load_results_dataframe(csv_path)
    assert updated_df["experiment_id"].tolist() == ["exp1", "exp2"]
    assert bool(updated_df.loc[0, "psbt_available"]) is True
    assert bool(updated_df.loc[1, "psbt_available"]) is False


def test_run_selection_filters_can_be_combined():
    manager = ExperimentManager(EXPERIMENTS_DIR / "experiments.csv")
    df = experiments_dataframe(manager, manager.read_experiments())

    filtered = filter_run_experiments(
        df,
        enabled_scope="Enabled only",
        providers=["openai"],
        strategies=["basic"],
        tags=["phase-1", "frontier"],
        tag_match="All selected tags",
        search="gpt-5.4-pro",
    )

    assert len(filtered) == 10
    assert set(filtered["model"]) == {"gpt-5.4-pro"}
    assert set(filtered["strategy"]) == {"basic"}


def test_sort_results_for_display_shows_newest_rows_first():
    df = pd.DataFrame([
        {"experiment_id": "older", "timestamp": "2026-04-22 10:00:00", "source_file": "experiments_20260422_100000.csv", "repetition": 1},
        {"experiment_id": "newer", "timestamp": "2026-04-22 10:05:00", "source_file": "experiments_20260422_100000.csv", "repetition": 1},
        {"experiment_id": "fallback", "timestamp": "", "source_file": "experiments_20260421_090000.csv", "repetition": 1},
    ])

    sorted_df = sort_results_for_display(df)
    assert sorted_df["experiment_id"].tolist()[:2] == ["newer", "older"]


def test_load_result_files_recurses_into_named_subdirectories(tmp_path, monkeypatch):
    root_csv = tmp_path / "experiments_20260209_104914.csv"
    nested_dir = tmp_path / "smoke_tests_20260422"
    nested_dir.mkdir()
    nested_csv = nested_dir / "experiments_20260422_092616.csv"
    newer_nested_dir = tmp_path / "investigate_tooltrace_lockfix_20260422"
    newer_nested_dir.mkdir()
    newer_nested_csv = newer_nested_dir / "experiments_20260422_105516.csv"

    for path in (root_csv, nested_csv, newer_nested_csv):
        path.write_text("experiment_id\nexp\n", encoding="utf-8")

    monkeypatch.setattr("web_ui.RESULTS_DIR", tmp_path)

    files = load_result_files()
    assert files == [root_csv, nested_csv, newer_nested_csv]
    assert format_result_path(newer_nested_csv) == "investigate_tooltrace_lockfix_20260422/experiments_20260422_105516.csv"


def test_run_selection_tags_and_pasted_ids_helpers():
    manager = ExperimentManager(EXPERIMENTS_DIR / "experiments.csv")
    df = experiments_dataframe(manager, manager.read_experiments())

    tags = all_tags_from_dataframe(df)
    assert "prompt-multiturn-detailed" in tags
    assert split_filter_values("exp_a, exp_b\nexp_c|exp_d") == ["exp_a", "exp_b", "exp_c", "exp_d"]

    any_tag = filter_run_experiments(
        df,
        enabled_scope="Enabled only",
        tags=["prompt-privacy-simple", "gpt54pro"],
        tag_match="Any selected tag",
    )
    all_tags = filter_run_experiments(
        df,
        enabled_scope="Enabled only",
        tags=["prompt-privacy-simple", "gpt54pro"],
        tag_match="All selected tags",
    )

    assert len(any_tag) > len(all_tags)
    assert len(all_tags) == 10
    assert set(all_tags["model"]) == {"gpt-5.4-pro"}
    assert set(all_tags["strategy"]) == {"privacy-simple"}


def test_result_loading_flattens_json_breakdown(tmp_path):
    csv_path = tmp_path / "experiments_20260414_000000.csv"
    csv_path.write_text(
        "experiment_id,experiment_name,repetition,timestamp,llm_provider,llm_model,llm_temperature,user_prompt,success,error_message,execution_time_seconds,psbt_generated,privacy_score,privacy_grade,tags\n"
        "exp1,Experiment,1,now,openai,gpt,1.0,prompt,True,,1.23,True,80,A,openai;basic\n",
        encoding="utf-8",
    )
    csv_path.with_suffix(".json").write_text(json.dumps([{
        "experiment_id": "exp1",
        "repetition": 1,
        "psbt_file": "results/psbts/exp1_rep1.psbt",
        "privacy_breakdown": {
            "fee_sanity_ok": 0,
            "sanity_status": "suspicious",
            "confidence": "high",
            "confidence_numeric": 1.0,
            "fee_analysis": {"fee_rate_sat_vb": 1234.5, "fee_sats": 999},
            "metadata": {"num_inputs": 1, "num_outputs": 2},
        },
    }]), encoding="utf-8")

    df = load_results_dataframe(csv_path)
    assert df.loc[0, "psbt_file"] == "results/psbts/exp1_rep1.psbt"
    assert df.loc[0, "fee_sanity_ok"] == 0
    assert df.loc[0, "sanity_status"] == "suspicious"
    assert df.loc[0, "fee_rate_sat_vb"] == 1234.5
    assert df.loc[0, "fee_sats"] == 999
    assert bool(df.loc[0, "psbt_available"]) is True


def test_result_loading_can_combine_multiple_result_files(tmp_path):
    paths = []
    for idx, model in enumerate(["gpt-test", "qwen-test"], start=1):
        csv_path = tmp_path / f"experiments_20260414_00000{idx}.csv"
        csv_path.write_text(
            "experiment_id,experiment_name,repetition,success,psbt_generated,privacy_score,privacy_grade,llm_model\n"
            f"exp{idx},Experiment {idx},1,True,True,{70 + idx},B,{model}\n",
            encoding="utf-8",
        )
        csv_path.with_suffix(".json").write_text("[]", encoding="utf-8")
        paths.append(csv_path)

    df = load_many_results_dataframes(paths)
    assert df["source_file"].tolist() == [paths[0].name, paths[1].name]
    assert df["experiment_id"].tolist() == ["exp1", "exp2"]
    assert set(df["llm_model"]) == {"gpt-test", "qwen-test"}


def test_results_display_dataframe_is_arrow_compatible_with_mixed_fee_sanity(tmp_path):
    csv_path = tmp_path / "experiments_20260414_000010.csv"
    csv_path.write_text(
        "experiment_id,experiment_name,repetition,success,psbt_generated,privacy_score,privacy_grade,llm_model\n"
        "exp1,Experiment 1,1,True,True,70,B,gpt-test\n"
        "exp2,Experiment 2,1,False,False,,F,qwen-test\n",
        encoding="utf-8",
    )
    csv_path.with_suffix(".json").write_text(json.dumps([
        {
            "experiment_id": "exp1",
            "repetition": 1,
            "privacy_breakdown": {"fee_sanity_ok": 1, "fee_analysis": {"fee_rate_sat_vb": 2.1}},
        },
        {
            "experiment_id": "exp2",
            "repetition": 1,
            "privacy_breakdown": {"fee_sanity_ok": "", "fee_analysis": {}},
        },
    ]), encoding="utf-8")

    df = load_results_dataframe(csv_path)
    assert str(df["fee_sanity_ok"].dtype) == "boolean"
    table = pa.Table.from_pandas(arrow_safe_dataframe(df[display_columns(df.columns)]))
    assert table.num_rows == 2


def test_result_files_are_arrow_compatible_with_experiment_metadata(tmp_path):
    experiments_csv = tmp_path / "experiments.csv"
    with experiments_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ExperimentManager.CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "id": "exp1",
            "name": "Experiment 1",
            "amount_btc": "0.5",
            "strategy": "privacy-simple",
            "prompt_mode": "template",
            "provider": "openai",
            "model": "gpt-test",
            "temperature": "1.0",
            "user_prompt": "prompt",
            "followup_prompts": "",
            "repetitions": "1",
            "timeout_seconds": "300",
            "network": "mainnet",
            "tags": "openai|privacy-simple",
            "enabled": "true",
            "priority": "1",
            "xpub": "",
        })
    result_csv = tmp_path / "experiments_20260414_000020.csv"
    result_csv.write_text(
        "experiment_id,experiment_name,repetition,success,psbt_generated,privacy_score,privacy_grade,llm_model\n"
        "exp1,Experiment 1,1,True,True,80,A,gpt-test\n",
        encoding="utf-8",
    )
    result_csv.with_suffix(".json").write_text(json.dumps([{
        "experiment_id": "exp1",
        "repetition": 1,
        "privacy_breakdown": {
            "fee_sanity_ok": 1,
            "sanity_status": "ok",
            "fee_analysis": {"fee_rate_sat_vb": 2.1, "fee_sats": 320},
        },
    }]), encoding="utf-8")

    manager = ExperimentManager(experiments_csv)
    df = load_many_results_dataframes([result_csv], manager=manager)
    assert df.loc[0, "strategy"] == "privacy-simple"
    assert df.loc[0, "amount_btc"] == 0.5
    table = pa.Table.from_pandas(arrow_safe_dataframe(df[display_columns(df.columns)]))
    assert table.num_rows == len(df)


def test_web_ui_does_not_use_deprecated_streamlit_components_html():
    source = (EXPERIMENTS_DIR / "web_ui.py").read_text(encoding="utf-8")
    assert "streamlit.components.v1" not in source
    assert "components.html" not in source
    assert "st.components.v1.html" not in source


def test_result_loading_handles_meta_without_prompt_mode(tmp_path):
    csv_path = tmp_path / "experiments_20260414_000001.csv"
    csv_path.write_text(
        "experiment_id,experiment_name,repetition,success,psbt_generated,privacy_score,privacy_grade\n"
        "legacy,Legacy,1,True,False,50,C\n",
        encoding="utf-8",
    )

    class LegacyManager:
        def read_experiments(self):
            return [{"id": "legacy", "tags": "legacy"}]

        def parse_csv_row_to_meta(self, row):
            return SimpleNamespace(
                id="legacy",
                strategy="basic",
                amount_btc=3.0,
                tags=["legacy"],
            )

        def infer_prompt_mode(self, row):
            return "custom"

    df = load_results_dataframe(csv_path, manager=LegacyManager())
    assert df.loc[0, "prompt_mode"] == "custom"


def test_paper_chart_sources_missing_data_is_tolerated_and_builders_work(tmp_path):
    sources = load_paper_chart_sources(tmp_path)
    assert sources["aggregated"].empty
    assert sources["v2_scores"].empty
    assert sources["missing"]

    agg = prepare_aggregated_dataframe(pd.DataFrame([
        {
            "model_short": "GPT Test",
            "model_full": "GPT Test",
            "provider": "openai",
            "category": "closed-source",
            "prompt_type": "basic",
            "n_psbts": 3,
            "n_total_attempts": 3,
            "success_rate": 1.0,
            "avg_v2_score": 80.0,
            "std_v2_score": 0.0,
            "min_v2_score": 80.0,
            "max_v2_score": 80.0,
            "avg_v2_score_fee_filtered": 80.0,
            "n_fee_sane": 3,
            "n_fee_insane": 0,
            "fee_insanity_rate": 0.0,
            "avg_fee_rate_sat_vb": 2.1,
            "avg_execution_time_seconds": 4.5,
            "avg_clustering": 100.0,
            "avg_change_detection": 85.0,
            "avg_fingerprinting": 95.0,
        },
        {
            "model_short": "Claude Test",
            "model_full": "Claude Test",
            "provider": "anthropic",
            "category": "closed-source",
            "prompt_type": "privacy_simple",
            "n_psbts": 3,
            "n_total_attempts": 3,
            "success_rate": 1.0,
            "avg_v2_score": 84.0,
            "std_v2_score": 2.0,
            "min_v2_score": 82.0,
            "max_v2_score": 86.0,
            "avg_v2_score_fee_filtered": 84.0,
            "n_fee_sane": 3,
            "n_fee_insane": 0,
            "fee_insanity_rate": 0.0,
            "avg_fee_rate_sat_vb": 3.0,
            "avg_execution_time_seconds": 5.0,
            "avg_clustering": 98.0,
            "avg_change_detection": 88.0,
            "avg_fingerprinting": 95.0,
        },
        {
            "model_short": "Open Test",
            "model_full": "Open Test",
            "provider": "openrouter",
            "category": "open-source",
            "prompt_type": "multiturn_detailed",
            "n_psbts": 3,
            "n_total_attempts": 3,
            "success_rate": 1.0,
            "avg_v2_score": 70.0,
            "std_v2_score": 5.0,
            "min_v2_score": 65.0,
            "max_v2_score": 75.0,
            "avg_v2_score_fee_filtered": 70.0,
            "n_fee_sane": 2,
            "n_fee_insane": 1,
            "fee_insanity_rate": 0.333,
            "avg_fee_rate_sat_vb": 8.0,
            "avg_execution_time_seconds": 8.0,
            "avg_clustering": 92.0,
            "avg_change_detection": 75.0,
            "avg_fingerprinting": 90.0,
        },
    ]))
    v2_df = prepare_v2_scores_dataframe(pd.DataFrame([
        {
            "model_short": "GPT Test",
            "category": "closed-source",
            "prompt_type": "basic",
            "v2_overall_score": 80,
            "fee_sanity_ok": 1,
            "fee_rate_sat_vb": 2.1,
        },
        {
            "model_short": "Claude Test",
            "category": "closed-source",
            "prompt_type": "privacy_simple",
            "v2_overall_score": 84,
            "fee_sanity_ok": 1,
            "fee_rate_sat_vb": 3.0,
        },
        {
            "model_short": "Open Test",
            "category": "open-source",
            "prompt_type": "multiturn_detailed",
            "v2_overall_score": 70,
            "fee_sanity_ok": 0,
            "fee_rate_sat_vb": 9000,
        },
    ]))

    for chart_name in PAPER_CHART_OPTIONS:
        chart = build_paper_chart(chart_name, agg, v2_df)
        assert chart is not None, f"{chart_name} should be available from synthetic chart data"
        assert chart.to_dict()


def test_paper_charts_aggregate_current_results_and_handle_missing_fee_columns():
    current = pd.DataFrame([
        {
            "llm_model": "gpt-test",
            "llm_provider": "openai",
            "strategy": "privacy-simple",
            "privacy_score": "81",
            "execution_time_seconds": "4.5",
            "psbt_generated": "True",
            "fee_sanity_ok": "True",
            "fee_rate_sat_vb": "2.1",
        },
        {
            "llm_model": "open-model",
            "llm_provider": "openrouter",
            "strategy": "multiturn-detailed",
            "privacy_score": "62",
            "execution_time_seconds": "8.0",
            "psbt_generated": "True",
            "fee_sanity_ok": "False",
            "fee_rate_sat_vb": "9999",
        },
    ])

    agg = aggregate_current_results(current)
    v2_df = current_results_v2_scores(current.drop(columns=["fee_sanity_ok"]))

    assert set(agg["prompt_type"]) == {"privacy_simple", "multiturn_detailed"}
    assert agg["n_psbts"].sum() == 2
    assert "fee_sanity_ok" in v2_df.columns
    assert build_paper_chart("Score heatmap", agg, v2_df).to_dict()
    assert build_paper_chart("Success and fee sanity", agg, v2_df).to_dict()


def test_safe_prompt_mode_handles_older_meta_without_prompt_mode():
    class LegacyManager:
        def infer_prompt_mode(self, row):
            return "custom"

    mode = safe_prompt_mode(
        LegacyManager(),
        {"id": "legacy", "tags": "legacy"},
        SimpleNamespace(id="legacy"),
    )
    assert mode == "custom"


def test_flatten_privacy_breakdown_empty_defaults():
    flat = flatten_privacy_breakdown(None)
    assert flat["fee_sanity_ok"] == ""
    assert flat["sanity_status"] == ""


def test_streamlit_all_pages_render_without_exceptions():
    pages = [
        "📊 Dashboard",
        "🧪 Experiments Browser",
        "➕ Create Experiment",
        "✏️ Edit Experiment",
        "📋 Clone Experiment",
        "▶️ Run Experiments",
        "📈 Results",
        "🔍 Compare Results",
    ]
    app = AppTest.from_file(str(EXPERIMENTS_DIR / "web_ui.py"), default_timeout=20)
    app.run()

    for page in pages:
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception, f"{page} raised: {app.exception}"
